"""OpenAI Chat Completions adapter (real provider, Phase 1).

Design notes:

- The ``openai`` SDK is an **optional extra** and is imported **lazily**
  (only when a client is actually built or a request runs), so
  ``import agent_core`` works in minimal, offline installs.
- Credentials come from the environment (``OPENAI_API_KEY``) via the
  factory. The key is passed straight to the SDK client and is **never
  stored on the provider, in Settings, in logs, or in error messages**.
- Timeouts are transport-level: the SDK client is created with the
  configured timeout and every request repeats it. Timeouts map to
  :class:`~agent_core.errors.TransientProviderError` so the gateway may
  retry them.
- The SDK's own retry mechanism is disabled (``max_retries=0``) so retry
  policy lives in exactly one place: the :class:`~agent_core.providers.gateway.ModelGateway`.
- Error messages are sanitized: status codes and bounded detail only.
  Model output is untrusted data; validation happens in the planner.
"""

from __future__ import annotations

from typing import Any

from ..errors import ProviderConfigurationError, ProviderError, TransientProviderError
from .base import Capability, ModelProvider, ModelRequest, ModelResponse

DEFAULT_MODEL = "gpt-4o-mini"

# Bounded detail from vendor errors — keeps messages useful but concise.
_DETAIL_LIMIT = 200


def _wants_json(response_format: dict[str, Any] | None) -> bool:
    """True when the neutral structured-output contract asks for JSON.

    The adapter maps the neutral contract to the vendor's ``json_object``
    mode. (Strict ``json_schema`` mode is intentionally not used: plan
    ``input`` objects vary per tool, and strict mode requires a closed
    schema. Reliability comes from prompt schema + post-validation.)
    """
    return response_format is not None and response_format.get("type") == "json_object"


class OpenAIProvider(ModelProvider):
    """Chat Completions adapter for OpenAI (and OpenAI-compatible) APIs."""

    name = "openai"
    capabilities = frozenset({Capability.TEXT, Capability.STREAMING, Capability.STRUCTURED_OUTPUT})

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        timeout_s: float = 60.0,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is not set; refusing to create the OpenAI provider"
            )
        self._model = model or DEFAULT_MODEL
        self._timeout_s = timeout_s
        # ``client`` exists for tests (stub injection); production passes None.
        self._client: Any = (
            client if client is not None else self._build_client(api_key, timeout_s, base_url)
        )

    @staticmethod
    def _build_client(api_key: str, timeout_s: float, base_url: str | None) -> Any:
        import openai  # lazy: keep the package importable without the SDK

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_s,
            "max_retries": 0,  # retry policy is owned by the ModelGateway
        }
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "timeout": self._timeout_s,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if _wants_json(request.response_format):
            kwargs["response_format"] = {"type": "json_object"}

        try:
            raw = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._translate_error(exc) from exc

        return self._to_response(raw)

    def stream(self, request: ModelRequest):
        """Yield bounded text deltas from an OpenAI streaming response.

        Streaming is deliberately not retried: once a caller has consumed
        part of a response, replaying it could duplicate user-visible output.
        """
        kwargs: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "timeout": self._timeout_s,
            "stream": True,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if _wants_json(request.response_format):
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    yield content
        except Exception as exc:
            raise self._translate_error(exc) from exc

    def _translate_error(self, exc: Exception) -> ProviderError:
        import openai  # SDK exception classes (module already imported at runtime)

        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return TransientProviderError("OpenAI request timed out or the connection failed")
        if isinstance(exc, openai.APIStatusError):
            status = exc.status_code
            if status == 429 or status >= 500:
                return TransientProviderError(f"OpenAI API transient failure (status {status})")
            if status in (401, 403):
                return ProviderConfigurationError(
                    f"OpenAI authentication failed (status {status}); check OPENAI_API_KEY"
                )
            detail = str(getattr(exc, "message", "") or "")[:_DETAIL_LIMIT]
            return ProviderError(f"OpenAI API rejected the request (status {status}): {detail}")
        if isinstance(exc, openai.OpenAIError):
            return ProviderError(f"OpenAI provider error: {type(exc).__name__}")
        return ProviderError(
            f"unexpected provider error: {type(exc).__name__}: {str(exc)[:_DETAIL_LIMIT]}"
        )

    @staticmethod
    def _to_response(raw: Any) -> ModelResponse:
        choices = getattr(raw, "choices", None)
        if not choices:
            raise ProviderError("OpenAI response contained no choices")
        content = getattr(choices[0].message, "content", None)
        if content is None:
            raise ProviderError("OpenAI response contained no message content")
        usage = None
        raw_usage = getattr(raw, "usage", None)
        if raw_usage is not None:
            try:
                usage = {
                    "prompt_tokens": raw_usage.prompt_tokens,
                    "completion_tokens": raw_usage.completion_tokens,
                    "total_tokens": raw_usage.total_tokens,
                }
            except AttributeError:
                usage = None
        return ModelResponse(content=content, model=getattr(raw, "model", None), usage=usage)
