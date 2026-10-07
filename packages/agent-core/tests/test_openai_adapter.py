"""OpenAI adapter tests: request mapping, error normalization, no key leaks.

Runs fully offline: a stub client stands in for the SDK transport, and real
SDK *exception types* (from the installed optional extra) verify the
normalization mapping. No network calls, no real credentials.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("openai")  # optional extra: skip offline installs
pytest.importorskip("httpx")  # ships with the openai SDK

import httpx
import openai
from agent_core import (
    ModelGateway,
    ModelRequest,
    OpenAIProvider,
    ProviderConfigurationError,
    ProviderError,
    TransientProviderError,
)
from agent_core.providers.base import ChatMessage

FAKE_KEY = "sk-test-not-a-real-key"


# ---------------------------------------------------------------------------
# Stubs for the SDK client and its response objects
# ---------------------------------------------------------------------------


class _StubCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: dict[str, Any] | None = None
        self.raw: Any = None
        self.error: Exception | None = None

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.raw


class _StubClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _StubCompletions()})()


def _raw_response(content: str, model: str = "gpt-4o-mini") -> Any:
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    usage = type("Usage", (), {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})()
    return type("Raw", (), {"choices": [choice], "model": model, "usage": usage})()


def _provider_with_stub() -> tuple[OpenAIProvider, _StubCompletions]:
    stub_client = _StubClient()
    provider = OpenAIProvider(
        api_key=FAKE_KEY, model="gpt-4o-mini", timeout_s=45.0, client=stub_client
    )
    return provider, stub_client.chat.completions


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return openai.APIStatusError(f"status {status}", response=response, body=None)


def _request() -> ModelRequest:
    return ModelRequest(
        messages=[
            ChatMessage(role="system", content="plan in JSON"),
            ChatMessage(role="user", content="Run the demo tool."),
        ],
        response_format={"type": "json_object", "schema": {"type": "object"}},
    )


class TestRequestMapping:
    def test_happy_path_maps_response(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = _raw_response('{"steps": []}', model="gpt-4o-mini-2024")
        response = provider.complete(_request())
        assert response.content == '{"steps": []}'
        assert response.model == "gpt-4o-mini-2024"
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

    def test_request_fields_are_mapped(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = _raw_response("ok")
        request = ModelRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=128,
        )
        provider.complete(request)
        kwargs = completions.kwargs
        assert kwargs is not None
        assert kwargs["model"] == "gpt-4o"  # request-level override
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 128
        assert kwargs["timeout"] == 45.0
        assert "response_format" not in kwargs  # no contract on this request

    def test_provider_default_model_used_when_request_has_none(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = _raw_response("ok")
        provider.complete(ModelRequest(messages=[ChatMessage(role="user", content="hi")]))
        assert completions.kwargs is not None
        assert completions.kwargs["model"] == "gpt-4o-mini"

    def test_structured_contract_maps_to_json_object_mode(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = _raw_response("ok")
        provider.complete(_request())
        assert completions.kwargs is not None
        assert completions.kwargs["response_format"] == {"type": "json_object"}

    def test_optional_params_omitted_when_none(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = _raw_response("ok")
        provider.complete(ModelRequest(messages=[ChatMessage(role="user", content="hi")]))
        kwargs = completions.kwargs
        assert kwargs is not None
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs


class TestCredentials:
    def test_missing_key_rejected(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
            OpenAIProvider(api_key="")

    def test_error_and_provider_never_leak_the_key(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = _status_error(401)
        with pytest.raises(ProviderError) as excinfo:
            provider.complete(_request())
        assert FAKE_KEY not in str(excinfo.value)
        assert FAKE_KEY not in repr(provider)

    def test_real_client_is_built_with_no_sdk_retries(self) -> None:
        # Constructing the client performs no network I/O.
        provider = OpenAIProvider(api_key=FAKE_KEY, timeout_s=30.0)
        client = provider._client  # SDK client, public enough for a test
        assert client.max_retries == 0  # retry policy owned by the ModelGateway
        assert FAKE_KEY not in repr(client.timeout)


class TestErrorNormalization:
    def test_timeout_is_transient(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = openai.APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )
        with pytest.raises(TransientProviderError, match="timed out"):
            provider.complete(_request())

    def test_connection_error_is_transient(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )
        with pytest.raises(TransientProviderError):
            provider.complete(_request())

    def test_rate_limit_is_transient(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = _status_error(429)
        with pytest.raises(TransientProviderError, match="429"):
            provider.complete(_request())

    def test_server_error_is_transient(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = _status_error(503)
        with pytest.raises(TransientProviderError, match="503"):
            provider.complete(_request())

    def test_auth_failure_is_configuration_error(self) -> None:
        for status in (401, 403):
            provider, completions = _provider_with_stub()
            completions.error = _status_error(status)
            with pytest.raises(ProviderConfigurationError, match="authentication"):
                provider.complete(_request())

    def test_client_error_is_non_transient_provider_error(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = _status_error(400)
        with pytest.raises(ProviderError) as excinfo:
            provider.complete(_request())
        assert not isinstance(excinfo.value, TransientProviderError)
        assert "400" in str(excinfo.value)

    def test_unexpected_exception_is_normalized(self) -> None:
        provider, completions = _provider_with_stub()
        completions.error = ValueError("unexpected transport bug")
        with pytest.raises(ProviderError, match="ValueError"):
            provider.complete(_request())

    def test_error_detail_is_bounded(self) -> None:
        provider, completions = _provider_with_stub()
        long_body = "bad " * 500
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(400, request=request)
        completions.error = openai.APIStatusError(long_body, response=response, body=None)
        with pytest.raises(ProviderError) as excinfo:
            provider.complete(_request())
        assert len(str(excinfo.value)) < 400


class TestResponseValidation:
    def test_missing_choices_rejected(self) -> None:
        provider, completions = _provider_with_stub()
        completions.raw = type("Raw", (), {"choices": [], "model": "m", "usage": None})()
        with pytest.raises(ProviderError, match="no choices"):
            provider.complete(_request())

    def test_missing_content_rejected(self) -> None:
        provider, completions = _provider_with_stub()
        message = type("Message", (), {"content": None})()
        choice = type("Choice", (), {"message": message})()
        completions.raw = type("Raw", (), {"choices": [choice], "model": "m", "usage": None})()
        with pytest.raises(ProviderError, match="no message content"):
            provider.complete(_request())


class TestGatewayIntegration:
    def test_transient_failure_retried_then_success(self) -> None:
        sleeps: list[float] = []

        class FirstCallFails(_StubCompletions):
            def create(self, **kwargs: Any) -> Any:
                self.calls += 1
                self.kwargs = kwargs
                if self.calls == 1:
                    raise openai.APIConnectionError(
                        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                    )
                return self.raw

        client = _StubClient()
        completions = FirstCallFails()
        completions.raw = _raw_response("ok")
        client.chat.completions = completions
        adapter = OpenAIProvider(api_key=FAKE_KEY, client=client)
        gateway = ModelGateway(adapter, max_retries=2, sleep=sleeps.append)
        response = gateway.complete(_request())
        assert response.content == "ok"
        assert sleeps == [0.5]  # one retry after the connection failure
