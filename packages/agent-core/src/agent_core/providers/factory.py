"""Provider selection and gateway construction from configuration.

The factory is the single place that knows which provider *names* exist and
how each is constructed. Agent core code never names a vendor: it calls
:func:`build_gateway` with :class:`~agent_core.config.Settings` and receives
a :class:`~agent_core.providers.gateway.ModelGateway`.

Credentials (``OPENAI_API_KEY``) are read from the environment mapping —
never from :class:`Settings` (which stays secret-free by design) and never
written to logs. To keep tests deterministic, an explicit ``env`` mapping
may be supplied (default: ``os.environ``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from ..config import Settings
from ..errors import ProviderConfigurationError
from .base import ModelProvider
from .gateway import ModelGateway
from .mock import MockModelProvider

#: Provider names accepted in ``MODEL_PROVIDER``.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("mock", "openai")


def create_provider(settings: Settings, env: Mapping[str, str] | None = None) -> ModelProvider:
    """Instantiate the configured provider (unwrapped)."""
    source: Mapping[str, str] = os.environ if env is None else env
    name = settings.model_provider.strip().lower()

    if name == "mock":
        return MockModelProvider(model=settings.model_name or "mock-1")
    if name == "openai":
        from .openai_provider import OpenAIProvider  # lazy: SDK is optional

        return OpenAIProvider(
            api_key=source.get("OPENAI_API_KEY", "").strip(),
            model=settings.model_name or None,
            timeout_s=settings.model_timeout_s,
            base_url=source.get("OPENAI_BASE_URL", "").strip() or None,
        )
    raise ProviderConfigurationError(
        f"unknown MODEL_PROVIDER {name!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def build_gateway(settings: Settings, env: Mapping[str, str] | None = None) -> ModelGateway:
    """Build the configured provider wrapped in a ModelGateway.

    This is the entry point the agent uses; the returned object satisfies
    the :class:`~agent_core.providers.base.ModelProvider` interface, so the
    planner and agent remain provider-agnostic.
    """
    provider = create_provider(settings, env=env)
    return ModelGateway(provider, max_retries=settings.model_max_retries)
