"""Provider-neutral memory retrieval (Phase 5).

``MemoryRetriever`` is the stable interface consumers (the ``recall`` tool,
the RAG context builder) depend on. The initial implementation,
:class:`LexicalMemoryRetriever`, delegates to the store's deterministic
lexical recall — consistent with Phase 4 (token-based, no embeddings, no
external model, stable tie-breaking).

A future semantic/vector retriever implements the same protocol (possibly
over a different store) without changing the recall tool or the context
builder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import Memory, MemoryType
from .store import MemoryStore

__all__ = [
    "LexicalMemoryRetriever",
    "MemoryRetriever",
]


@runtime_checkable
class MemoryRetriever(Protocol):
    """Deterministic memory recall over a memory store."""

    def recall(
        self,
        query: str,
        *,
        memory_type: MemoryType | str | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[Memory]: ...


class LexicalMemoryRetriever:
    """Default retriever: deterministic lexical recall via the store."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def store(self) -> MemoryStore:
        return self._store

    def recall(
        self,
        query: str,
        *,
        memory_type: MemoryType | str | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[Memory]:
        return self._store.recall(query, memory_type=memory_type, limit=limit, now=now)
