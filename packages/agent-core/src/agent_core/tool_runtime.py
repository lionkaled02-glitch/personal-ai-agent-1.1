"""The tool runtime: the agent's only execution path for planned tool calls.

Pipeline for one invocation (in order):

1. **Permission backstop** — the caller must pass an explicit
   :data:`~agent_core.permissions.PermissionDecision.ALLOWED`; anything else
   raises :class:`~agent_core.errors.PermissionDeniedError` before the tool
   is even resolved. This makes it architecturally impossible to execute a
   tool through the runtime without having passed the permission system
   (the executor obtains the decision from the
   :class:`~agent_core.permissions.PermissionManager`).
2. **TOOL_STARTED** event (tool name + input),
3. **input validation + execution + output validation** via the
   :class:`~agent_core.tools.ToolRegistry` controlled interface
   (Phase 0 behavior, unchanged — no duplicated pipeline),
4. **structured observation** — the outcome is a
   :class:`~agent_core.tools.ToolResult` with a machine-readable
   ``error_code`` and execution ``metadata`` (tool version, determinism,
   duration), and the matching lifecycle events:

   - success            -> ``TOOL_COMPLETED``
   - unknown tool       -> ``TOOL_FAILED`` (``tool_not_found``)
   - invalid input      -> ``TOOL_INPUT_INVALID`` + ``TOOL_FAILED``
   - output not matching-> ``TOOL_OUTPUT_INVALID`` + ``TOOL_FAILED``
   - tool fault         -> ``TOOL_FAILED`` (``execution_error`` or the
     tool's own domain code)

Tool exceptions are contained: the runtime never lets a tool fault crash the
agent process.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from .errors import (
    PermissionDeniedError,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
)
from .events import Clock, EventBus, EventType, bounded_text, bounded_value, utc_now
from .permissions import PermissionDecision
from .tools import ToolRegistry, ToolResult

__all__ = ["ToolInvocation", "ToolRuntime"]


class ToolInvocation(BaseModel):
    """One planned tool call, with the task/step context for events."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    step_id: str
    tool_name: str
    input: dict[str, Any]


class ToolRuntime:
    """Executes tool invocations through the registry with permission gating.

    The runtime is provider-independent: it only knows about the registry,
    the event bus, and permission decisions — never about models.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        events: EventBus,
        clock: Clock | None = None,
    ) -> None:
        self._registry = registry
        self._events = events
        self._clock: Clock = clock or utc_now

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def execute(
        self,
        invocation: ToolInvocation,
        *,
        decision: PermissionDecision,
        redact_sensitive_io: bool = False,
    ) -> ToolResult:
        """Run one invocation. Requires ALLOWED; optional caller privacy redaction."""
        if decision is not PermissionDecision.ALLOWED:
            raise PermissionDeniedError(
                f"tool {invocation.tool_name!r} cannot execute: "
                f"permission decision is {decision.value}, expected ALLOWED"
            )

        started = self._clock()
        registered_tool = self._registry.get(invocation.tool_name)
        sensitive_input = redact_sensitive_io or (
            registered_tool.spec.sensitive_input if registered_tool is not None else False
        )
        self._events.emit(
            EventType.TOOL_STARTED,
            task_id=invocation.task_id,
            step_id=invocation.step_id,
            data={
                "tool_name": invocation.tool_name,
                "input": {"redacted": True} if sensitive_input else bounded_value(invocation.input),
            },
        )

        try:
            result = self._registry.execute(invocation.tool_name, invocation.input)
        except ToolNotFoundError as exc:
            result = ToolResult(ok=False, error=str(exc), error_code="tool_not_found")
        except ToolInputError as exc:
            result = ToolResult(ok=False, error=str(exc), error_code="input_invalid")
            self._emit(
                EventType.TOOL_INPUT_INVALID,
                invocation,
                error="voice tool input failed validation" if redact_sensitive_io else str(exc),
                error_code="input_invalid",
            )
        except ToolExecutionError as exc:
            result = ToolResult(ok=False, error=str(exc), error_code="execution_error")
        else:
            if not result.ok and result.error_code == "output_invalid":
                self._emit(
                    EventType.TOOL_OUTPUT_INVALID,
                    invocation,
                    error=(
                        "voice tool output failed validation"
                        if redact_sensitive_io
                        else result.error or ""
                    ),
                    error_code="output_invalid",
                )

        result.metadata = self._metadata(invocation.tool_name, started)

        if result.ok:
            current_tool = self._registry.get(invocation.tool_name)
            sensitive_output = redact_sensitive_io or (
                current_tool.spec.sensitive_output if current_tool is not None else False
            )
            self._events.emit(
                EventType.TOOL_COMPLETED,
                task_id=invocation.task_id,
                step_id=invocation.step_id,
                data={
                    "tool_name": invocation.tool_name,
                    "output": {"redacted": True}
                    if sensitive_output
                    else bounded_value(result.output),
                },
            )
        else:
            self._events.emit(
                EventType.TOOL_FAILED,
                task_id=invocation.task_id,
                step_id=invocation.step_id,
                data={
                    "tool_name": invocation.tool_name,
                    "error": "voice tool failed"
                    if redact_sensitive_io
                    else bounded_text(result.error or "tool failed"),
                    "error_code": "voice_tool_failed" if redact_sensitive_io else result.error_code,
                },
            )
        return result

    def _emit(
        self,
        event_type: EventType,
        invocation: ToolInvocation,
        *,
        error: str,
        error_code: str,
    ) -> None:
        self._events.emit(
            event_type,
            task_id=invocation.task_id,
            step_id=invocation.step_id,
            data={
                "tool_name": invocation.tool_name,
                "error": bounded_text(error),
                "error_code": error_code,
            },
        )

    def _metadata(self, tool_name: str, started: datetime) -> dict[str, Any]:
        """Execution metadata: identity, version, determinism, duration."""
        spec = self._registry.get(tool_name)
        duration_ms = (self._clock() - started).total_seconds() * 1000.0
        return {
            "tool_name": tool_name,
            "version": spec.spec.version if spec is not None else None,
            "deterministic": spec.spec.deterministic if spec is not None else None,
            "duration_ms": duration_ms,
        }
