"""Mock model provider and base interface tests (offline, deterministic)."""

from __future__ import annotations

import json

import pytest
from agent_core import (
    Capability,
    ChatMessage,
    ModelProvider,
    ModelRequest,
    ProviderError,
)
from agent_core.providers.mock import MockModelProvider


def make_request() -> ModelRequest:
    return ModelRequest(
        messages=[
            ChatMessage(role="system", content="plan in JSON"),
            ChatMessage(role="user", content="Run the demo tool."),
        ],
    )


class TestScriptedMode:
    def test_pops_responses_in_order(self) -> None:
        provider = MockModelProvider(responses=["first", "second"])
        assert provider.complete(make_request()).content == "first"
        assert provider.complete(make_request()).content == "second"

    def test_last_response_repeats(self) -> None:
        provider = MockModelProvider(responses=["only"])
        assert provider.complete(make_request()).content == "only"
        assert provider.complete(make_request()).content == "only"

    def test_records_requests(self) -> None:
        provider = MockModelProvider(responses=["x"])
        request = make_request()
        provider.complete(request)
        assert provider.requests == [request]

    def test_response_metadata(self) -> None:
        provider = MockModelProvider(responses=["x"], model="mock-9")
        response = provider.complete(make_request())
        assert response.model == "mock-9"
        assert response.content == "x"


class TestKeywordMode:
    def test_demo_tool_keyword_produces_plan(self) -> None:
        provider = MockModelProvider()
        response = provider.complete(make_request())
        plan = json.loads(response.content)
        assert plan["steps"][0]["tool_name"] == "demo_tool"
        assert plan["steps"][0]["input"] == {"message": "hello from the agent core"}

    def test_unmatched_request_raises_provider_error(self) -> None:
        provider = MockModelProvider()
        request = ModelRequest(messages=[ChatMessage(role="user", content="fly to mars")])
        with pytest.raises(ProviderError, match="no scripted response"):
            provider.complete(request)


class TestInterface:
    def test_capabilities(self) -> None:
        provider = MockModelProvider()
        assert provider.name == "mock"
        assert Capability.TEXT in provider.capabilities
        assert Capability.STRUCTURED_OUTPUT in provider.capabilities
        assert Capability.STREAMING not in provider.capabilities

    def test_unimplemented_stream_raises(self) -> None:
        provider = MockModelProvider()
        with pytest.raises(ProviderError, match="streaming"):
            provider.stream(make_request())

    def test_unimplemented_embed_raises(self) -> None:
        provider = MockModelProvider()
        with pytest.raises(ProviderError, match="embeddings"):
            provider.embed(["hello"])

    def test_request_supports_structured_output_contract(self) -> None:
        request = ModelRequest(
            messages=[ChatMessage(role="user", content="hi")],
            response_format={"type": "json_object"},
        )
        assert request.response_format == {"type": "json_object"}
        dumped = request.model_dump()
        assert dumped["response_format"] == {"type": "json_object"}

    def test_message_role_is_constrained(self) -> None:
        with pytest.raises(ValueError):
            ChatMessage(role="robot", content="hi")  # type: ignore[arg-type]

    def test_mock_is_a_model_provider(self) -> None:
        provider: ModelProvider = MockModelProvider()
        assert isinstance(provider, ModelProvider)


class TestOpenAIStreaming:
    def test_stream_maps_text_deltas_and_capability(self) -> None:
        from agent_core.providers.openai_provider import OpenAIProvider

        class Delta:
            def __init__(self, content: str | None) -> None:
                self.content = content

        class Choice:
            def __init__(self, content: str | None) -> None:
                self.delta = Delta(content)

        class Chunk:
            def __init__(self, content: str | None) -> None:
                self.choices = [Choice(content)]

        class Completions:
            def create(self, **kwargs):
                assert kwargs["stream"] is True
                return iter([Chunk("Hel"), Chunk(None), Chunk("lo")])

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        provider = OpenAIProvider(api_key="test-key", client=Client())
        assert Capability.STREAMING in provider.capabilities
        assert list(provider.stream(make_request())) == ["Hel", "lo"]


def test_mock_provider_handles_basic_greeting():
    provider = MockModelProvider()
    response = provider.complete(
        ModelRequest(messages=[{"role": "user", "content": "مرحبا كيف انت"}])
    )
    assert "demo_tool" in response.content
    assert "مساعد" in json.loads(response.content)["steps"][0]["input"]["message"]
