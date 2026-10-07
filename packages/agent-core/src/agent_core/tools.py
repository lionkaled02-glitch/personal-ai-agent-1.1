"""Tool abstraction, the tool registry, and the controlled execution path.

A tool is anything with a declarative :class:`ToolSpec` (name, description,
JSON-Schema input/output contracts, permission level) and a synchronous
``run`` method.

The registry is the controlled-execution boundary: it validates input and
output against the declared schemas and contains tool exceptions. Permission
enforcement is deliberately NOT done here — agent executions go through the
:class:`~agent_core.tool_runtime.ToolRuntime`, which requires an explicit
ALLOWED permission decision (the executor obtains it from the
:class:`~agent_core.permissions.PermissionManager`) and owns the tool
lifecycle events. The registry stays permission-free so it remains reusable
outside the agent loop (e.g. a future tool-development CLI).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from .errors import (
    ToolAlreadyRegisteredError,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
)
from .permissions import PermissionLevel
from .schema import validate_against_schema


class ToolSpec(BaseModel):
    """Declarative description of a tool.

    Schemas are plain JSON-Schema documents (validated against the subset in
    :mod:`agent_core.schema`). JSON Schema keeps tool definitions portable
    across model providers (function-calling APIs expect JSON Schema) and
    testable without importing any provider SDK.

    Phase 2 additions (backwards-compatible defaults): ``version`` (compat
    metadata for future tool evolution) and ``deterministic`` (tools that
    read the clock or external state must declare ``False``).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permission_level: PermissionLevel
    version: str = "1.0.0"
    deterministic: bool = True
    # Sensitive computer observations and typed text are returned to the
    # caller but redacted from the in-memory operational event stream.
    sensitive_input: bool = False
    sensitive_output: bool = False


class ToolResult(BaseModel):
    """Outcome of one tool execution. Tools report failure via ``ok=False``.

    Phase 2 additions (backwards-compatible defaults):

    - ``error_code``: a short machine-readable failure class. Runtime codes:
      ``tool_not_found``, ``input_invalid``, ``output_invalid``,
      ``execution_error``. Tools may add domain codes (e.g.
      ``division_by_zero``, ``invalid_json``) for structured observation.
    - ``metadata``: execution metadata attached by the tool runtime
      (tool version, determinism flag, duration).
    """

    ok: bool
    output: Any | None = None
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] | None = None


class Tool(Protocol):
    """Anything the agent can execute.

    Concrete tools implement a ``spec`` attribute and a synchronous ``run``
    method. Tools must be deterministic (or say so in their description),
    must not read secrets from the environment, and must keep side effects
    inside the areas their permission level covers.
    """

    spec: ToolSpec

    def run(self, input: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """Registration and controlled execution of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            # Typed error (Phase 2); still a ValueError for Phase 0 callers.
            raise ToolAlreadyRegisteredError(f"tool already registered: {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"tool not registered: {name!r}")
        return tool

    def list_tools(self) -> list[ToolSpec]:
        """All registered specs, sorted by name (deterministic order)."""
        return [self._tools[name].spec for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, name: str, input: dict[str, Any]) -> ToolResult:
        """Validate, run, and validate the output of one registered tool.

        Raises :class:`ToolNotFoundError` / :class:`ToolInputError` for
        registry-level problems; tool faults are contained in the returned
        :class:`ToolResult`.
        """
        tool = self.require(name)
        spec = tool.spec

        input_errors = validate_against_schema(input, spec.input_schema)
        if input_errors:
            raise ToolInputError(f"invalid input for tool {name!r}: " + "; ".join(input_errors))

        try:
            result = tool.run(input)
        except ToolExecutionError:
            raise
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                error_code="execution_error",
            )

        if result.ok:
            output_errors = validate_against_schema(result.output, spec.output_schema)
            if output_errors:
                return ToolResult(
                    ok=False,
                    error=(
                        f"tool {name!r} returned output not matching its output schema: "
                        + "; ".join(output_errors)
                    ),
                    error_code="output_invalid",
                )
        return result
