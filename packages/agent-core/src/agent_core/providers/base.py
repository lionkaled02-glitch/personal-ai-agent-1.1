"""Vendor-neutral model provider interface.

The agent core depends ONLY on :class:`ModelProvider` — never on a vendor
SDK. Concrete adapters (OpenAI, Anthropic, local models, ...) are added in
later phases and must implement this interface.

Capabilities beyond plain text completion (streaming, embeddings, vision,
native structured output) are declared via :attr:`ModelProvider.capabilities`
and raise :class:`~agent_core.errors.ProviderError` on the base class until
an adapter implements them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from ..errors import ProviderError


class Capability(StrEnum):
    TEXT = "text"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    EMBEDDINGS = "embeddings"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(BaseModel):
    """Provider-agnostic completion request.

    ``response_format`` carries an optional structured-output contract in a
    neutral shape; adapters translate it to their native mechanism.
    """

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


class ModelResponse(BaseModel):
    content: str
    model: str | None = None
    usage: dict[str, int] | None = None


class ModelProvider(ABC):
    """The single interface the core uses to talk to a language model."""

    name: str = "abstract"
    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """One blocking text completion."""

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Incremental completion (PLANNED for real adapters, Phase 1)."""
        raise ProviderError(f"provider {self.name!r} does not implement streaming yet")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Text embeddings (PLANNED, needed by the RAG phase)."""
        raise ProviderError(f"provider {self.name!r} does not implement embeddings yet")
