"""ModelPlanner tests: strict-JSON parsing and tool-name validation."""

from __future__ import annotations

import json

import pytest
from agent_core import (
    MockModelProvider,
    ModelPlanner,
    ModelRequest,
    ModelResponse,
    PlanningError,
    ProviderError,
    ToolRegistry,
)
from conftest import DEMO_PLAN_JSON


def two_step_plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {"tool_name": "demo_tool", "description": "first", "input": {"message": "a"}},
                {"tool_name": "demo_tool", "description": "second", "input": {"message": "b"}},
            ]
        }
    )


def make_planner(content: str) -> tuple[ModelPlanner, MockModelProvider]:
    provider = MockModelProvider(responses=[content])
    return ModelPlanner(provider), provider


class TestModelPlanner:
    def test_parses_scripted_plan(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner(DEMO_PLAN_JSON)
        plan = planner.plan("Run the demo tool.", demo_registry.list_tools())
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "demo_tool"
        assert plan.steps[0].input == {"message": "hi from test"}

    def test_preserves_step_order(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner(two_step_plan_json())
        plan = planner.plan("two steps", demo_registry.list_tools())
        assert [s.input["message"] for s in plan.steps] == ["a", "b"]

    def test_system_prompt_contains_available_tools(self, demo_registry: ToolRegistry) -> None:
        planner, provider = make_planner(DEMO_PLAN_JSON)
        planner.plan("Run the demo tool.", demo_registry.list_tools())
        assert len(provider.requests) == 1
        system_prompt = provider.requests[0].messages[0].content
        assert provider.requests[0].messages[0].role == "system"
        assert "demo_tool" in system_prompt
        assert provider.requests[0].messages[1].role == "user"
        assert provider.requests[0].messages[1].content == "Run the demo tool."

    def test_invalid_json_raises_planning_error(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner("definitely not json")
        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan("x", demo_registry.list_tools())

    def test_non_object_json_raises(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner("[1, 2, 3]")
        with pytest.raises(PlanningError, match="JSON object"):
            planner.plan("x", demo_registry.list_tools())

    def test_unknown_tool_name_raises(self, demo_registry: ToolRegistry) -> None:
        bad = json.dumps({"steps": [{"tool_name": "root_shell", "description": "no"}]})
        planner, _ = make_planner(bad)
        with pytest.raises(PlanningError, match="unknown tools"):
            planner.plan("x", demo_registry.list_tools())

    def test_empty_steps_raise(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner('{"steps": []}')
        with pytest.raises(PlanningError, match="no steps"):
            planner.plan("x", demo_registry.list_tools())

    def test_malformed_step_shape_raises(self, demo_registry: ToolRegistry) -> None:
        planner, _ = make_planner('{"steps": [{"tool_name": "demo_tool"}]}')
        with pytest.raises(PlanningError, match="plan schema"):
            planner.plan("x", demo_registry.list_tools())

    def test_no_available_tools_raises(self) -> None:
        planner, _ = make_planner(DEMO_PLAN_JSON)
        with pytest.raises(PlanningError, match="no tools available"):
            planner.plan("x", ToolRegistry().list_tools())

    def test_request_carries_explicit_structured_contract(
        self, demo_registry: ToolRegistry
    ) -> None:
        from agent_core.planner import plan_json_schema

        planner, provider = make_planner(DEMO_PLAN_JSON)
        planner.plan("Run the demo tool.", demo_registry.list_tools())
        request = provider.requests[0]
        assert request.response_format is not None
        assert request.response_format["type"] == "json_object"
        assert request.response_format["schema"] == plan_json_schema()

    def test_markdown_fenced_json_is_accepted(self, demo_registry: ToolRegistry) -> None:
        fenced = "```json\n" + DEMO_PLAN_JSON + "\n```"
        planner, _ = make_planner(fenced)
        plan = planner.plan("Run the demo tool.", demo_registry.list_tools())
        assert plan.steps[0].tool_name == "demo_tool"

    def test_markdown_fence_with_surrounding_text_is_rejected(
        self, demo_registry: ToolRegistry
    ) -> None:
        messy = "Sure! Here is the plan:\n```json\n" + DEMO_PLAN_JSON + "\n```\nHope that helps!"
        planner, _ = make_planner(messy)
        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan("Run the demo tool.", demo_registry.list_tools())

    def test_provider_fault_wrapped_in_planning_error(self, demo_registry: ToolRegistry) -> None:
        class ExplodingProvider(MockModelProvider):
            def complete(self, request: ModelRequest) -> ModelResponse:
                raise ProviderError("boom")

        planner = ModelPlanner(ExplodingProvider())
        with pytest.raises(PlanningError, match="model provider failed"):
            planner.plan("x", demo_registry.list_tools())


def test_tool_spec_serializes_for_prompt(demo_registry: ToolRegistry) -> None:
    specs = demo_registry.list_tools()
    assert [s.name for s in specs] == ["demo_tool"]
    dumped = specs[0].model_dump(mode="json")
    assert dumped["permission_level"] == 1  # LOW as int in JSON
    json.dumps(dumped)  # must be JSON-serializable
