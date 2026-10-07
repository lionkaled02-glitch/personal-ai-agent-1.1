"""Provider-neutral RAG context assembly (Phase 5).

The :class:`ContextBuilder` combines **retrieved memories** (Phase 5
memory layer) and **retrieved document chunks** (Phase 4 knowledge layer)
into a structured, bounded :class:`Context`.

Responsibilities (and only these):

- retrieval (via provider-neutral interfaces — ``MemoryRetriever`` and
  ``RetrievalIndex`` — so a future embedding/vector provider is swappable
  without touching this logic),
- bounded context assembly (max chars + max items, deterministic order),
- provenance preservation (every item carries its source id, source ref,
  provenance category, and — for documents — page/slide/sheet location),
- an explicit distinction between ``memory`` and ``document`` items.

It **never generates answers** and never interprets retrieved content:
document chunks and memory content are untrusted DATA in the context,
labeled as such, so downstream consumers (a model prompt, a UI) can treat
them accordingly. Omission under the size budget is always *reported*
(``truncated`` + ``omitted_items``), never silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..documents.models import SearchResult
from ..documents.retrieval import RetrievalIndex
from ..events import Clock, utc_now
from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.limits import MemoryLimits
from ..memory.models import Memory, MemoryType
from ..memory.retrieval import MemoryRetriever

ContextKind = Literal["memory", "document"]


class ContextItem(BaseModel):
    """One provenance-labeled piece of assembled context (data only)."""

    kind: ContextKind
    #: Stable id of the source item (``memory_id`` or ``document_id``).
    source_id: str
    #: Provenance reference: memory ``source_ref`` or document
    #: workspace-relative ``source_path``.
    source_ref: str | None = None
    #: Provenance category: the memory's source category, or ``"document"``.
    provenance: str
    #: Source location (page/slide/sheet for document chunks; None otherwise).
    location: dict[str, Any] | None = None
    title: str | None = None
    score: float | None = None
    text: str


class Context(BaseModel):
    """Structured, bounded, deterministically ordered retrieval context."""

    items: list[ContextItem] = Field(default_factory=list)
    total_chars: int = 0
    memory_items: int = 0
    document_items: int = 0
    truncated: bool = False
    #: Items omitted because the context size/item budget was exhausted.
    omitted_items: int = 0


@dataclass(frozen=True)
class ContextRequest:
    """Explicit, bounded retrieval request for one context assembly."""

    memory_query: str | None = None
    document_query: str | None = None
    memory_type: MemoryType | str | None = None
    limit: int | None = None  # total item cap (default: max_context_items)


class ContextBuilder:
    """Retrieval + bounded assembly over memory and knowledge layers."""

    def __init__(
        self,
        *,
        memory_retriever: MemoryRetriever,
        knowledge_store: RetrievalIndex,
        limits: MemoryLimits,
        clock: Clock | None = None,
    ) -> None:
        self._retriever = memory_retriever
        self._knowledge = knowledge_store
        self._limits = limits
        self._clock: Clock = clock or utc_now

    @property
    def limits(self) -> MemoryLimits:
        return self._limits

    def build(self, request: ContextRequest, *, now: datetime | None = None) -> Context:
        """Assemble a bounded context for the request. Deterministic.

        Order: memories (recall rank order) first, then document chunks
        (search rank order). Items are dropped whole (never mid-item) once
        the char budget is exhausted, and every drop is reported.
        """
        now = now if now is not None else self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        self._check_query(request.memory_query)
        self._check_query(request.document_query)

        limit = request.limit if request.limit is not None else self._limits.max_context_items
        if not isinstance(limit, int) or limit < 1:
            raise MemoryStoreError(MEMORY_INVALID_INPUT, "limit must be a positive integer")
        limit = min(limit, self._limits.max_context_items)

        candidates: list[ContextItem] = []

        # Per-source retrieval is bounded by the natural retrieval cap
        # (each store clamps to its own configured maximum as well). The
        # final context budget is then applied to the combined candidate
        # list, so every drop is visible in ``omitted_items``.
        if request.memory_query and request.memory_query.strip():
            memories = self._retriever.recall(
                request.memory_query,
                memory_type=request.memory_type,
                limit=self._limits.max_recall_results,
                now=now,
            )
            candidates.extend(self._memory_items(memories))

        if request.document_query and request.document_query.strip():
            hits = self._knowledge.search(
                request.document_query,
                limit=self._limits.max_recall_results,
            )
            candidates.extend(self._document_items(hits))

        items: list[ContextItem] = []
        total_chars = 0
        omitted = 0
        for item in candidates:
            if len(items) >= limit:
                omitted += 1
                continue
            if total_chars + len(item.text) > self._limits.max_context_chars:
                omitted += 1
                continue
            items.append(item)
            total_chars += len(item.text)

        return Context(
            items=items,
            total_chars=total_chars,
            memory_items=sum(1 for i in items if i.kind == "memory"),
            document_items=sum(1 for i in items if i.kind == "document"),
            truncated=omitted > 0,
            omitted_items=omitted,
        )

    # -- internal -----------------------------------------------------------

    def _check_query(self, query: str | None) -> None:
        if query is None:
            return
        if not isinstance(query, str):
            raise MemoryStoreError(MEMORY_INVALID_INPUT, "queries must be strings")
        if len(query) > self._limits.max_content_chars:
            raise MemoryStoreError(
                MEMORY_INVALID_INPUT,
                f"query is too long (max {self._limits.max_content_chars} characters)",
            )

    @staticmethod
    def _memory_items(memories: list[Memory]) -> list[ContextItem]:
        return [
            ContextItem(
                kind="memory",
                source_id=memory.memory_id,
                source_ref=memory.source_ref,
                provenance=str(memory.source),
                location=None,
                title=None,
                score=None,
                text=memory.content,
            )
            for memory in memories
        ]

    @staticmethod
    def _document_items(hits: list[SearchResult]) -> list[ContextItem]:
        items: list[ContextItem] = []
        for hit in hits:
            chunk = hit.chunk
            items.append(
                ContextItem(
                    kind="document",
                    source_id=chunk.document_id,
                    source_ref=chunk.metadata.get("source_path"),
                    provenance="document",
                    location=chunk.location,
                    title=hit.document_title,
                    score=hit.score,
                    text=chunk.text,
                )
            )
        return items
