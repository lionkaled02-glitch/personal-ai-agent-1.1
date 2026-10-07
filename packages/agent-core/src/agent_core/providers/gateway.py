"""Model gateway: the single, stable provider-facing boundary for the core.

The gateway is a :class:`ModelProvider` *decorator*. Everything upstream
(planner, agent) depends only on :class:`ModelProvider`; the gateway is how
a configured provider is plugged in without the core ever learning about
vendors.

Responsibilities (centralized, so individual adapters stay thin):

- **Error normalization** — any exception a provider raises that is not
  already a project error becomes a :class:`ProviderError` with a bounded,
  secret-free message.
- **Safe retry** — :class:`TransientProviderError` (timeout, 5xx, rate
  limit, connection) is retried up to ``max_retries`` times with
  exponential backoff. Completion requests are idempotent from the
  provider's point of view, so retrying is safe. Non-transient errors fail
  fast — never retried.
- **Pass-through** — successful :class:`ModelResponse` objects (structured
  content, model, usage) are returned unchanged.

The gateway knows nothing about specific vendors: it speaks only the
``ModelProvider``/``ProviderError`` contract. ``stream``/``embed`` are
delegated without retry (streaming cannot be safely replayed).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence

from ..errors import (
    ProviderConfigurationError,
    ProviderError,
    TransientProviderError,
)
from .base import Capability, ModelProvider, ModelRequest, ModelResponse

Sleep = Callable[[float], None]


class ModelGateway(ModelProvider):
    """Wraps a concrete provider with normalization and retry policy."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        max_backoff_s: float = 8.0,
        sleep: Sleep = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ProviderConfigurationError("max_retries must be >= 0")
        if backoff_s <= 0 or max_backoff_s < backoff_s:
            raise ProviderConfigurationError("require backoff_s > 0 and max_backoff_s >= backoff_s")
        self._provider = provider
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._max_backoff_s = max_backoff_s
        self._sleep = sleep
        # Expose the inner provider's identity transparently.
        self.name = provider.name
        self.capabilities: frozenset[Capability] = provider.capabilities

    @property
    def provider(self) -> ModelProvider:
        """The wrapped provider (introspection/tests only)."""
        return self._provider

    def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: TransientProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._provider.complete(request)
            except TransientProviderError as exc:
                last_error = exc
            except ProviderError:
                raise  # known non-transient project error: fail fast
            except Exception as exc:
                raise ProviderError(
                    f"provider {self._provider.name!r} failed: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                ) from exc
            if attempt < self._max_retries:
                self._sleep(self._delay_for(attempt))
        # Unreachable unless every attempt was transient.
        assert last_error is not None
        raise last_error

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Delegated without retry: a half-consumed stream cannot be replayed."""
        return self._provider.stream(request)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def _delay_for(self, attempt: int) -> float:
        """Exponential backoff: base * 2^attempt, capped at the maximum."""
        delay = self._backoff_s
        for _ in range(attempt):
            delay = min(self._max_backoff_s, delay * 2.0)
        return delay
