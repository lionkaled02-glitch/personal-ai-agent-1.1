"""End-to-end agent flow tests.

Proves the architecture works: user request -> Agent -> Planner ->
Tool Registry -> Tool -> Result -> Task completion, with structured events
along the way. Everything runs on the mock provider: no external APIs,
deterministic, injectable clock.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest
from agent_core import (
    Agent,
    ApprovalCallback,
    DemoTool,
    EventBus,
    EventType,
    MockModelProvider,
    ModelPlanner,
    PermissionDecision,
    PermissionLevel,
    PermissionManager,
    PermissionPolicy,
    Settings,
    StepStatus,
    TaskState,
    Tool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from conftest import DEMO_PLAN_JSON, FIXED_NOW


def medium_plan_json() -> str:
    return (
        '{"steps": [{"tool_name": "medium_tool", '
        '"description": "Run the medium tool", "input": {"x": "y"}}]}'
    )


def high_plan_json() -> str:
    return (
        '{"steps": [{"tool_name": "high_tool", "description": "Run the high tool", "input": {}}]}'
    )


def failing_plan_json() -> str:
    return (
        '{"steps": [{"tool_name": "failing_tool", '
        '"description": "Run the failing tool", "input": {}}]}'
    )


def _plan_json(tool_name: str, tool_input: dict[str, object] | None = None) -> str:
    import json

    return json.dumps(
        {
            "steps": [
                {
                    "tool_name": tool_name,
                    "description": f"Run {tool_name}",
                    "input": tool_input or {},
                }
            ]
        }
    )


class MediumTool:
    spec = ToolSpec(
        name="medium_tool",
        description="A MEDIUM-permission test tool.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        permission_level=PermissionLevel.MEDIUM,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"ok": True})


class HighTool:
    spec = ToolSpec(
        name="high_tool",
        description="A HIGH-permission test tool.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.HIGH,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"ok": True})


class FailingTool:
    spec = ToolSpec(
        name="failing_tool",
        description="Always fails.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=False, error="the tool deliberately failed")


Clock = Callable[[], datetime]


def build_agent(
    plan_json: str,
    tools: list[Tool],
    *,
    approval: ApprovalCallback | None = None,
    policy: PermissionPolicy | None = None,
    fixed_clock: Clock | None = None,
) -> Agent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    clock: Clock = fixed_clock if fixed_clock is not None else (lambda: FIXED_NOW)
    return Agent(
        planner=ModelPlanner(MockModelProvider(responses=[plan_json])),
        registry=registry,
        permissions=PermissionManager(policy=policy, approval=approval),
        events=EventBus(clock=clock),
        clock=clock,
    )


def event_types(agent: Agent) -> list[EventType]:
    return [event.type for event in agent.events.history]


class TestHappyPath:
    def test_run_demo_tool_completes(self, demo_agent: Agent) -> None:
        task = demo_agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED
        assert task.error is None
        assert task.result == {"tool": "demo_tool", "message": "hi from test"}
        assert len(task.steps) == 1
        assert task.steps[0].status is StepStatus.COMPLETED
        assert task.steps[0].output == task.result

    def test_event_sequence_is_exactly_the_flow(self, demo_agent: Agent) -> None:
        demo_agent.run("Run the demo tool.")
        # Phase 2: TOOL_REQUESTED marks the step pickup before the permission
        # check; the rest of the sequence is unchanged from Phase 0.
        assert event_types(demo_agent) == [
            EventType.TASK_CREATED,
            EventType.PLAN_CREATED,
            EventType.TOOL_REQUESTED,
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TASK_COMPLETED,
        ]

    def test_event_payloads_are_operational(self, demo_agent: Agent) -> None:
        demo_agent.run("Run the demo tool.")
        events = demo_agent.events.history
        created = events[0]
        assert created.task_id is not None
        assert created.data["request"] == "Run the demo tool."
        plan_event = events[1]
        assert plan_event.data["steps"][0]["tool_name"] == "demo_tool"
        requested = events[2]
        assert requested.data["tool_name"] == "demo_tool"
        assert requested.data["permission_level"] == "LOW"
        started = events[3]
        assert started.step_id is not None
        assert started.data["tool_name"] == "demo_tool"
        assert started.data["input"] == {"message": "hi from test"}
        completed = events[4]
        assert completed.data["output"] == {"tool": "demo_tool", "message": "hi from test"}
        finished = events[5]
        assert finished.data["steps_completed"] == 1

    def test_timestamps_use_injected_clock(self, demo_agent: Agent) -> None:
        task = demo_agent.run("Run the demo tool.")
        assert task.created_at == FIXED_NOW
        assert task.updated_at == FIXED_NOW
        for event in demo_agent.events.history:
            assert event.timestamp == FIXED_NOW

    def test_all_events_reference_the_task(self, demo_agent: Agent) -> None:
        task = demo_agent.run("Run the demo tool.")
        assert all(event.task_id == task.id for event in demo_agent.events.history)


class TestPlanningFailures:
    def test_invalid_model_output_fails_task(self, fixed_clock: Clock) -> None:
        agent = build_agent("not json at all", [DemoTool()], fixed_clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.FAILED
        assert task.error is not None and "planning failed" in task.error
        assert event_types(agent) == [EventType.TASK_CREATED, EventType.TASK_FAILED]
        assert task.steps == []

    def test_unknown_tool_in_plan_fails_task(self, fixed_clock: Clock) -> None:
        bad_plan = _plan_json("ghost")
        agent = build_agent(bad_plan, [DemoTool()], fixed_clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.FAILED
        assert task.error is not None and "unknown tools" in task.error


class TestToolFailures:
    def test_tool_failure_fails_task_with_events(self, fixed_clock: Clock) -> None:
        agent = build_agent(failing_plan_json(), [FailingTool()], fixed_clock=fixed_clock)
        task = agent.run("make it fail")
        assert task.state is TaskState.FAILED
        assert task.error == "the tool deliberately failed"
        assert task.steps[0].status is StepStatus.FAILED
        assert task.result is None
        assert event_types(agent) == [
            EventType.TASK_CREATED,
            EventType.PLAN_CREATED,
            EventType.TOOL_REQUESTED,
            EventType.TOOL_STARTED,
            EventType.TOOL_FAILED,
            EventType.TASK_FAILED,
        ]
        failed_event = agent.events.events_of_type(EventType.TOOL_FAILED)[0]
        assert failed_event.data["error"] == "the tool deliberately failed"

    def test_plan_referencing_unregistered_tool_fails(self, fixed_clock: Clock) -> None:
        agent = build_agent(DEMO_PLAN_JSON, [], fixed_clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        # No tools registered -> planner refuses before execution.
        assert task.state is TaskState.FAILED
        assert task.error is not None and "no tools available" in task.error


class TestApprovalFlows:
    def test_medium_tool_with_approval_completes(self, fixed_clock: Clock) -> None:
        agent = build_agent(
            medium_plan_json(), [MediumTool()], approval=lambda _r: True, fixed_clock=fixed_clock
        )
        task = agent.run("run the medium tool")
        assert task.state is TaskState.COMPLETED
        assert EventType.APPROVAL_REQUIRED in event_types(agent)
        approval_event = agent.events.events_of_type(EventType.APPROVAL_REQUIRED)[0]
        assert approval_event.data["tool_name"] == "medium_tool"
        assert approval_event.data["permission_level"] == "MEDIUM"

    def test_medium_tool_without_approval_denied_cancels(self, fixed_clock: Clock) -> None:
        agent = build_agent(
            medium_plan_json(), [MediumTool()], approval=lambda _r: False, fixed_clock=fixed_clock
        )
        task = agent.run("run the medium tool")
        assert task.state is TaskState.CANCELLED
        assert task.error is not None and "approval denied" in task.error
        types = event_types(agent)
        assert types.index(EventType.APPROVAL_REQUIRED) < types.index(EventType.TASK_CANCELLED)
        assert EventType.TOOL_STARTED not in types  # never executed

    def test_no_approval_channel_is_fail_safe(self, fixed_clock: Clock) -> None:
        agent = build_agent(medium_plan_json(), [MediumTool()], fixed_clock=fixed_clock)
        task = agent.run("run the medium tool")
        assert task.state is TaskState.CANCELLED
        assert task.error is not None and "approval denied" in task.error

    def test_high_level_denied_by_policy_cancels(self, fixed_clock: Clock) -> None:
        policy = PermissionPolicy(high=PermissionDecision.DENIED)
        agent = build_agent(
            high_plan_json(),
            [HighTool()],
            policy=policy,
            approval=lambda _r: True,
            fixed_clock=fixed_clock,
        )
        task = agent.run("run the high tool")
        assert task.state is TaskState.CANCELLED
        assert task.error is not None and "denied by permission policy" in task.error

    def test_deny_list_cancels_even_with_approval(self, fixed_clock: Clock) -> None:
        policy = PermissionPolicy(denied_tools=frozenset({"medium_tool"}))
        agent = build_agent(
            medium_plan_json(),
            [MediumTool()],
            policy=policy,
            approval=lambda _r: True,
            fixed_clock=fixed_clock,
        )
        task = agent.run("run the medium tool")
        assert task.state is TaskState.CANCELLED
        assert task.error is not None and "denied by permission policy" in task.error


class TestDemoFactory:
    def test_create_demo_runs_happy_path_offline(self) -> None:
        agent = Agent.create_demo()
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED
        assert task.result == {
            "tool": "demo_tool",
            "message": "hello from the agent core",
        }


class TestConfiguredFactory:
    """Agent.create_configured (Phase 1): settings-driven provider wiring."""

    def test_defaults_are_fully_offline(self, fixed_clock: Clock) -> None:
        agent = Agent.create_configured(settings=Settings(), clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED
        assert task.result is not None
        assert task.created_at == FIXED_NOW

    def test_mock_selection_via_settings(self, fixed_clock: Clock) -> None:
        settings = Settings(model_provider="mock", model_name="test-model")
        agent = Agent.create_configured(settings=settings, clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED

    def test_unknown_provider_is_a_controlled_configuration_error(self) -> None:
        from agent_core import ProviderConfigurationError

        settings = Settings(model_provider="bogus")
        with pytest.raises(ProviderConfigurationError, match="bogus"):
            Agent.create_configured(settings=settings)


class TestToolRuntimeIntegration:
    """Executor + ToolRuntime + built-in tools, end to end (offline)."""

    def test_calculator_tool_completes_via_agent(self, fixed_clock: Clock) -> None:
        from agent_core import CalculatorTool

        agent = build_agent(
            _plan_json("calculator", {"expression": "6 * 7"}),
            [CalculatorTool()],
            fixed_clock=fixed_clock,
        )
        task = agent.run("calculate 6 * 7")
        assert task.state is TaskState.COMPLETED
        assert task.result == {"expression": "6 * 7", "result": 42, "type": "integer"}

    def test_default_agent_can_plan_and_run_calculator(self, fixed_clock: Clock) -> None:
        # create_demo registers the built-ins; a plan for calculator works.
        from agent_core import CalculatorTool
        from agent_core.builtin_tools import register_default_tools

        registry = register_default_tools(ToolRegistry())
        clock: Clock = fixed_clock
        agent = Agent(
            planner=ModelPlanner(
                MockModelProvider(responses=[_plan_json("calculator", {"expression": "1 + 1"})])
            ),
            registry=registry,
            permissions=PermissionManager(),
            events=EventBus(clock=clock),
            clock=clock,
        )
        assert CalculatorTool.spec.name == "calculator"
        task = agent.run("calculate one plus one")
        assert task.state is TaskState.COMPLETED
        assert task.result is not None and task.result["result"] == 2

    def test_invalid_tool_input_emits_validation_event_and_fails(self, fixed_clock: Clock) -> None:
        # Plan schema allows any input object; the registry's input-schema
        # check catches the bad type at runtime (not at planning time).
        plan = _plan_json("demo_tool", {"message": 42})
        agent = build_agent(plan, [DemoTool()], fixed_clock=fixed_clock)
        task = agent.run("run the demo tool with a number")
        assert task.state is TaskState.FAILED
        assert task.error is not None and "invalid input" in task.error
        types = event_types(agent)
        assert EventType.TOOL_INPUT_INVALID in types
        assert types.index(EventType.TOOL_INPUT_INVALID) < types.index(EventType.TOOL_FAILED)
        invalid = agent.events.events_of_type(EventType.TOOL_INPUT_INVALID)[0]
        assert invalid.data["error_code"] == "input_invalid"

    def test_permission_denial_emits_tool_denied_event(self, fixed_clock: Clock) -> None:
        policy = PermissionPolicy(denied_tools=frozenset({"demo_tool"}))
        agent = build_agent(DEMO_PLAN_JSON, [DemoTool()], policy=policy, fixed_clock=fixed_clock)
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.CANCELLED
        denied = agent.events.events_of_type(EventType.TOOL_DENIED)
        assert len(denied) == 1
        assert denied[0].data["tool_name"] == "demo_tool"
        assert denied[0].data["reason"] == "denied_by_policy"
        assert EventType.TOOL_STARTED not in event_types(agent)  # never executed

    def test_approval_denial_emits_tool_denied_event(self, fixed_clock: Clock) -> None:
        agent = build_agent(
            medium_plan_json(),
            [MediumTool()],
            approval=lambda _r: False,
            fixed_clock=fixed_clock,
        )
        task = agent.run("run the medium tool")
        assert task.state is TaskState.CANCELLED
        denied = agent.events.events_of_type(EventType.TOOL_DENIED)
        assert len(denied) == 1
        assert denied[0].data["reason"] == "approval_denied"

    def test_execution_metadata_captured_on_completed_step(self, fixed_clock: Clock) -> None:
        agent = build_agent(
            _plan_json("demo_tool", {"message": "meta"}),
            [DemoTool()],
            fixed_clock=fixed_clock,
        )
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED
        started = agent.events.events_of_type(EventType.TOOL_STARTED)[0]
        assert started.data["tool_name"] == "demo_tool"
        # TOOL_REQUESTED precedes the permission-gated TOOL_STARTED.
        types = event_types(agent)
        assert types.index(EventType.TOOL_REQUESTED) < types.index(EventType.TOOL_STARTED)


@pytest.mark.parametrize(
    "text",
    ["Run the demo tool.", "run the DEMO TOOL please", "please execute demo_tool now"],
)
def test_mock_keyword_matching_is_case_insensitive(text: str) -> None:
    task = Agent.create_demo().run(text)
    assert task.state is TaskState.COMPLETED
