"""Model provider layer: gateway, factory, and adapters.

Phase 1: vendor-neutral ``ModelProvider`` interface, ``ModelGateway``
(normalization + safe retry), a configuration-driven factory, the
deterministic ``MockModelProvider``, and the real ``OpenAIProvider``
(optional ``openai`` extra, lazy SDK import).

Further adapters must implement ``ModelProvider`` and be registered in
:mod:`agent_core.providers.factory` — never hard-coded into the core.
"""

from .base import (
    Capability,
    ChatMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from .factory import SUPPORTED_PROVIDERS, build_gateway, create_provider
from .gateway import ModelGateway
from .mock import MockModelProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "SUPPORTED_PROVIDERS",
    "Capability",
    "ChatMessage",
    "MockModelProvider",
    "ModelGateway",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "OpenAIProvider",
    "build_gateway",
    "create_provider",
]
