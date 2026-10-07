"""Tool runtime tests: permission backstop, structured results, events.

All offline and deterministic: scripted tools + injected fixed clock.
"""

from __future__ import annotations

import pytest
from agent_core import (
    EventBus,
    EventType,
    PermissionDecision,
    ToolAlreadyRegisteredError,
    ToolInputError,
    ToolInvocation,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)
from agent_core.errors import PermissionDeniedError
from agent_core.permissions import PermissionLevel
from conftest import FIXED_NOW

RuntimeFixture = tuple[ToolRuntime, ToolRegistry, EventBus]


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="test tool echo",
        input_schema={
            "type": "object",
            "required": ["v"],
            "properties": {"v": {"type": "string"}},
        },
        output_schema={"type": "object", "required": ["ok"]},
        permission_level=PermissionLevel.LOW,
        version="2.1.0",
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"ok": True, "v": input["v"]})


class LyingTool:
    """Declares required output keys it never returns."""

    spec = ToolSpec(
        name="lying",
        description="test tool lying",
        input_schema={"type": "object"},
        output_schema={"type": "object", "required": ["missing_key"]},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"other": 1})


class ExplodingTool:
    spec = ToolSpec(
        name="exploding",
        description="test tool exploding",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        raise RuntimeError("kaboom")


class DomainTool:
    """Fails with a tool-specific domain error code."""

    spec = ToolSpec(
        name="domain",
        description="test tool domain",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=False, error="division by zero", error_code="division_by_zero")


@pytest.fixture
def runtime() -> tuple[ToolRuntime, ToolRegistry, EventBus]:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(LyingTool())
    registry.register(ExplodingTool())
    registry.register(DomainTool())
    events = EventBus(clock=lambda: FIXED_NOW)
    return ToolRuntime(registry, events, clock=lambda: FIXED_NOW), registry, events


def types(events: EventBus) -> list[EventType]:
    return [event.type for event in events.history]


def _inv(tool_name: str, input: dict[str, object]) -> ToolInvocation:
    return ToolInvocation(task_id="t-1", step_id="s-1", tool_name=tool_name, input=input)


class TestPermissionBackstop:
    def test_denied_decision_raises_without_executing(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        with pytest.raises(PermissionDeniedError, match="DENIED"):
            rt.execute(_inv("echo", {"v": "x"}), decision=PermissionDecision.DENIED)
        assert len(events) == 0  # nothing observable: no execution, no events

    def test_requires_approval_decision_also_raises(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        with pytest.raises(PermissionDeniedError, match="REQUIRES_APPROVAL"):
            rt.execute(_inv("echo", {}), decision=PermissionDecision.REQUIRES_APPROVAL)
        assert len(events) == 0

    def test_allowed_decision_executes(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("echo", {"v": "x"}), decision=PermissionDecision.ALLOWED)
        assert result.ok
        assert types(events) == [EventType.TOOL_STARTED, EventType.TOOL_COMPLETED]


class TestStructuredOutcomes:
    def test_success_result_has_metadata(self, runtime: RuntimeFixture) -> None:
        rt, _, _ = runtime
        result = rt.execute(_inv("echo", {"v": "x"}), decision=PermissionDecision.ALLOWED)
        assert result.ok
        assert result.metadata is not None
        assert result.metadata["tool_name"] == "echo"
        assert result.metadata["version"] == "2.1.0"
        assert result.metadata["deterministic"] is True
        assert result.metadata["duration_ms"] == 0.0  # fixed clock

    def test_input_validation_failure_is_structured(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("echo", {}), decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "input_invalid"
        assert result.error is not None and "missing required property" in result.error
        assert result.metadata is not None
        assert types(events) == [
            EventType.TOOL_STARTED,
            EventType.TOOL_INPUT_INVALID,
            EventType.TOOL_FAILED,
        ]
        invalid = events.events_of_type(EventType.TOOL_INPUT_INVALID)[0]
        assert invalid.data["tool_name"] == "echo"
        assert invalid.data["error_code"] == "input_invalid"
        assert invalid.task_id == "t-1" and invalid.step_id == "s-1"

    def test_output_validation_failure_is_structured(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("lying", {}), decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "output_invalid"
        assert "output schema" in (result.error or "")
        assert types(events) == [
            EventType.TOOL_STARTED,
            EventType.TOOL_OUTPUT_INVALID,
            EventType.TOOL_FAILED,
        ]

    def test_tool_exception_is_contained_and_structured(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("exploding", {}), decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "execution_error"
        assert "kaboom" in (result.error or "")
        assert types(events) == [EventType.TOOL_STARTED, EventType.TOOL_FAILED]

    def test_tool_domain_error_code_preserved(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("domain", {}), decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "division_by_zero"
        failed = events.events_of_type(EventType.TOOL_FAILED)[0]
        assert failed.data["error_code"] == "division_by_zero"

    def test_unknown_tool_is_structured_not_fatal(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        result = rt.execute(_inv("ghost", {}), decision=PermissionDecision.ALLOWED)
        assert not result.ok
        assert result.error_code == "tool_not_found"
        assert types(events) == [EventType.TOOL_STARTED, EventType.TOOL_FAILED]

    def test_failed_events_carry_bounded_errors(self, runtime: RuntimeFixture) -> None:
        rt, _, events = runtime
        rt.execute(_inv("echo", {}), decision=PermissionDecision.ALLOWED)
        failed = events.events_of_type(EventType.TOOL_FAILED)[0]
        assert isinstance(failed.data["error"], str) and len(failed.data["error"]) <= 201


class TestRegistryLevel:
    def test_duplicate_registration_is_typed_and_value_error(self, runtime: RuntimeFixture) -> None:
        _, registry, _ = runtime
        with pytest.raises(ToolAlreadyRegisteredError, match="already registered"):
            registry.register(EchoTool())
        # Backwards compatibility: still catchable as ValueError.
        try:
            registry.register(EchoTool())
        except ValueError:
            pass
        else:  # pragma: no cover
            pytest.fail("expected ValueError")

    def test_registry_still_raises_for_direct_callers(self, runtime: RuntimeFixture) -> None:
        # Phase 0 contract for non-agent callers of registry.execute.
        _, registry, _ = runtime
        with pytest.raises(ToolInputError):
            registry.execute("echo", {})
        with pytest.raises(ToolNotFoundError):
            registry.execute("ghost", {})
