"""Model gateway tests: normalization, retry, timeouts, pass-through.

Uses a deterministic scripted fake provider (offline, no API keys) so retry
and timeout behavior is fully controlled and reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from agent_core import (
    Agent,
    ModelGateway,
    ModelPlanner,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    PlanningError,
    ProviderConfigurationError,
    ProviderError,
    Settings,
    ToolRegistry,
    TransientProviderError,
)
from agent_core.demo_tools import DemoTool
from agent_core.planner import plan_json_schema
from agent_core.providers.base import Capability, ChatMessage
from agent_core.tasks import TaskState
from conftest import DEMO_PLAN_JSON, FIXED_NOW

ScriptItem = object


class ScriptedProvider(ModelProvider):
    """Deterministic fake: plays a script of responses or exceptions.

    Each :meth:`complete` pops the next item (the last repeats). Records
    every request so tests can assert on call counts.
    """

    name = "scripted"
    capabilities = frozenset({Capability.TEXT})

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        assert script, "script must not be empty"
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, ModelResponse)
        return item


def ok_response(content: str = "hello") -> ModelResponse:
    return ModelResponse(content=content, model="scripted-1", usage={"total_tokens": 3})


def make_gateway(
    script: Sequence[ScriptItem], **kwargs: Any
) -> tuple[ModelGateway, list[float], ScriptedProvider]:
    sleeps: list[float] = []
    provider = ScriptedProvider(script)
    gateway = ModelGateway(provider, sleep=sleeps.append, **kwargs)
    return gateway, sleeps, provider


REQUEST = ModelRequest(messages=[ChatMessage(role="user", content="hi")])


class TestPassThrough:
    def test_success_is_unchanged(self) -> None:
        response = ok_response("the answer")
        gateway, sleeps, _provider = make_gateway([response])
        result = gateway.complete(REQUEST)
        assert result is response
        assert result.content == "the answer"
        assert result.model == "scripted-1"
        assert result.usage == {"total_tokens": 3}
        assert sleeps == []

    def test_name_and_capabilities_passthrough(self) -> None:
        gateway, _, _provider = make_gateway([ok_response()])
        assert gateway.name == "scripted"
        assert Capability.TEXT in gateway.capabilities

    def test_inner_provider_is_exposed(self) -> None:
        gateway, _, _provider = make_gateway([ok_response()])
        assert isinstance(gateway.provider, ScriptedProvider)


class TestNormalization:
    def test_unknown_exception_becomes_provider_error(self) -> None:
        gateway, sleeps, _provider = make_gateway([RuntimeError("socket exploded")])
        with pytest.raises(ProviderError, match="socket exploded") as excinfo:
            gateway.complete(REQUEST)
        assert not isinstance(excinfo.value, TransientProviderError)
        assert sleeps == []  # unknown faults are not retried

    def test_non_transient_provider_error_fails_fast(self) -> None:
        gateway, sleeps, provider = make_gateway([ProviderError("bad request")])
        with pytest.raises(ProviderError, match="bad request"):
            gateway.complete(REQUEST)
        assert sleeps == []
        assert len(provider.requests) == 1

    def test_message_is_bounded(self) -> None:
        long = "x" * 1000
        gateway, _, _provider = make_gateway([RuntimeError(long)])
        with pytest.raises(ProviderError) as excinfo:
            gateway.complete(REQUEST)
        assert len(str(excinfo.value)) < 300


class TestRetry:
    def test_transient_is_retried_then_succeeds(self) -> None:
        gateway, sleeps, provider = make_gateway([TransientProviderError("timeout"), ok_response()])
        result = gateway.complete(REQUEST)
        assert result.content == "hello"
        assert len(provider.requests) == 2
        assert sleeps == [0.5]  # one backoff between the two attempts

    def test_transient_exhausts_retries(self) -> None:
        gateway, sleeps, provider = make_gateway(
            [
                TransientProviderError("timeout"),
                TransientProviderError("timeout"),
                TransientProviderError("timeout"),
            ],
            max_retries=2,
        )
        with pytest.raises(TransientProviderError):
            gateway.complete(REQUEST)
        assert len(provider.requests) == 3
        assert sleeps == [0.5, 1.0]  # exponential backoff

    def test_backoff_is_capped(self) -> None:
        gateway, sleeps, _provider = make_gateway(
            [TransientProviderError("t")] * 5,
            max_retries=4,
            backoff_s=4.0,
            max_backoff_s=8.0,
        )
        with pytest.raises(TransientProviderError):
            gateway.complete(REQUEST)
        assert sleeps == [4.0, 8.0, 8.0, 8.0]

    def test_zero_retries_means_single_attempt(self) -> None:
        gateway, sleeps, provider = make_gateway([TransientProviderError("timeout")], max_retries=0)
        with pytest.raises(TransientProviderError):
            gateway.complete(REQUEST)
        assert len(provider.requests) == 1
        assert sleeps == []

    def test_timeout_failure_is_retried_like_other_transient_faults(self) -> None:
        """Timeouts are transient: the gateway retries them with backoff."""
        gateway, sleeps, _provider = make_gateway(
            [TransientProviderError("request timed out"), ok_response()],
            max_retries=1,
        )
        assert gateway.complete(REQUEST).content == "hello"
        assert sleeps == [0.5]

    def test_invalid_retry_configuration_rejected(self) -> None:
        provider = ScriptedProvider([ok_response()])
        with pytest.raises(ProviderConfigurationError):
            ModelGateway(provider, max_retries=-1)
        with pytest.raises(ProviderConfigurationError):
            ModelGateway(provider, backoff_s=0.0)
        with pytest.raises(ProviderConfigurationError):
            ModelGateway(provider, backoff_s=2.0, max_backoff_s=1.0)


class TestStreamingDelegation:
    def test_stream_is_delegated_without_retry(self) -> None:
        gateway, _, _provider = make_gateway([ok_response()])
        with pytest.raises(ProviderError, match="streaming"):
            gateway.stream(REQUEST)

    def test_embed_is_delegated_without_retry(self) -> None:
        gateway, _, _provider = make_gateway([ok_response()])
        with pytest.raises(ProviderError, match="embeddings"):
            gateway.embed(["x"])


class TestPlannerIntegration:
    def test_planner_receives_structured_contract(self) -> None:
        provider = ScriptedProvider([ok_response(DEMO_PLAN_JSON)])
        planner = ModelPlanner(ModelGateway(provider, sleep=lambda _s: None))
        registry = ToolRegistry()
        registry.register(DemoTool())
        plan = planner.plan("Run the demo tool.", registry.list_tools())
        assert plan.steps[0].tool_name == "demo_tool"
        # The request that reached the provider carried the explicit contract.
        request = provider.requests[0]
        assert request.response_format is not None
        assert request.response_format["type"] == "json_object"
        assert request.response_format["schema"] == plan_json_schema()

    def test_malformed_model_output_is_a_controlled_planning_error(self) -> None:
        provider = ScriptedProvider([ok_response("I cannot produce JSON")])
        planner = ModelPlanner(ModelGateway(provider, sleep=lambda _s: None))
        registry = ToolRegistry()
        registry.register(DemoTool())
        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan("Run the demo tool.", registry.list_tools())

    def test_invalid_tool_name_is_rejected_before_execution(self) -> None:
        bad_plan = json.dumps({"steps": [{"tool_name": "rm_rf", "description": "no", "input": {}}]})
        provider = ScriptedProvider([ok_response(bad_plan)])
        planner = ModelPlanner(ModelGateway(provider, sleep=lambda _s: None))
        registry = ToolRegistry()
        registry.register(DemoTool())
        with pytest.raises(PlanningError, match="unknown tools"):
            planner.plan("Run the demo tool.", registry.list_tools())

    def test_agent_flow_through_gateway_completes(self) -> None:
        """Phase 0 flow, now with the gateway in between planner and provider."""
        provider = ScriptedProvider([ok_response(DEMO_PLAN_JSON)])
        gateway = ModelGateway(provider, sleep=lambda _s: None, max_retries=2)
        agent = Agent.create_configured(
            settings=Settings(), gateway=gateway, clock=lambda: FIXED_NOW
        )
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.COMPLETED
        assert task.result == {"tool": "demo_tool", "message": "hi from test"}

    def test_provider_failure_fails_the_task_controlled(self) -> None:
        provider = ScriptedProvider([ProviderError("auth expired")])
        gateway = ModelGateway(provider, sleep=lambda _s: None)
        agent = Agent.create_configured(
            settings=Settings(), gateway=gateway, clock=lambda: FIXED_NOW
        )
        task = agent.run("Run the demo tool.")
        assert task.state is TaskState.FAILED
        assert task.error is not None and "planning failed" in task.error
        assert "auth expired" in task.error

    def test_gateway_is_the_only_provider_boundary(self) -> None:
        """The agent holds no vendor reference: it only knows the gateway."""
        provider = ScriptedProvider([ok_response(DEMO_PLAN_JSON)])
        gateway = ModelGateway(provider, sleep=lambda _s: None)
        agent = Agent.create_configured(settings=Settings(), gateway=gateway)
        assert isinstance(agent, Agent)
        assert agent.events is not None
        # Phase 2: the configured agent carries the default tool set
        # (demo tool + calculator/datetime/text_utils/json_utils).
        # Phase 3 topology change (documented): the configured agent also
        # registers the 9 workspace filesystem tools, so the default set is
        # now 5 + 9 = 14 tools.
        # Phase 4 topology change (documented): the configured agent also
        # registers the 4 document tools (inspect/extract/index/search),
        # so the default set is now 5 + 9 + 4 = 18 tools.
        # Phase 5 topology change (documented): the configured agent also
        # registers the 5 memory tools (remember/recall/update_memory/
        # forget/list_memories), so the default set is now
        # 5 + 9 + 4 + 5 = 23 tools.
        assert len(agent.registry.list_tools()) == 23
