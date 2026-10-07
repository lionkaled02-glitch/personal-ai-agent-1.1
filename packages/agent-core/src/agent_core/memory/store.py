"""Provider-neutral memory storage (Phase 5).

``MemoryStore`` is the stable interface the rest of the agent depends on;
``InMemoryMemoryStore`` is the required initial implementation (no external
database, no network). A future durable store (SQLite, vector DB, ...)
implements the same protocol without changing tools or the agent.

Guarantees:

- Deterministic ids (see :func:`make_memory_id`) and deterministic ordering
  (``created_at``, then ``memory_id``).
- Limits/policy are enforced at the store layer (defense in depth): type
  policy, content length, metadata size, item cap, secret heuristic.
- Expiration: ``short_term``/``working`` get an implicit TTL at creation
  (from limits); ``long_term``/``knowledge`` never expire implicitly and
  are never touched by :meth:`purge_expired`.
- ``forget`` is a **soft deactivation** by default (explicit, testable);
  ``hard=True`` deletes. A forget never touches other memories.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from ..documents.retrieval import tokenize  # reuse the Phase 4 lexical tokenizer
from ..events import Clock, utc_now
from .errors import (
    MEMORY_INVALID_INPUT,
    MEMORY_LIMIT_EXCEEDED,
    MEMORY_NOT_FOUND,
    MEMORY_TYPE_NOT_ALLOWED,
    SECRET_LIKE_CONTENT,
    MemoryStoreError,
)
from .guards import contains_secret_like_content
from .limits import MemoryLimits
from .models import (
    Memory,
    MemoryType,
    SourceCategory,
    make_memory_id,
    metadata_size_bytes,
)

__all__ = [
    "InMemoryMemoryStore",
    "MemoryStore",
]

# Alias so the ``list`` method signatures below can reference the built-in
# ``list`` — inside a class body the method name shadows the builtin for
# signature annotations.
MemoryList = list[Memory]


@runtime_checkable
class MemoryStore(Protocol):
    """Provider-neutral memory storage + recall interface."""

    @property
    def limits(self) -> MemoryLimits: ...

    def remember(
        self,
        *,
        memory_type: MemoryType | str,
        content: str,
        source: SourceCategory | str,
        source_ref: str | None = None,
        metadata: dict[str, object] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> Memory: ...

    def get(self, memory_id: str) -> Memory | None: ...

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        active: bool | None = None,
        now: datetime | None = None,
    ) -> Memory: ...

    def forget(
        self, memory_id: str, *, hard: bool = False, now: datetime | None = None
    ) -> Memory: ...

    def list(
        self,
        *,
        memory_type: MemoryType | str | None = None,
        active_only: bool = True,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> MemoryList: ...

    def recall(
        self,
        query: str,
        *,
        memory_type: MemoryType | str | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> MemoryList: ...

    def purge_expired(self, now: datetime | None = None) -> int: ...


def _ensure_aware(dt: datetime) -> datetime:
    """Naive datetimes are assumed UTC (documented contract)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _as_type(value: MemoryType | str, field_name: str) -> MemoryType:
    try:
        return MemoryType(value)
    except ValueError as exc:
        raise MemoryStoreError(
            MEMORY_INVALID_INPUT,
            f"{field_name} must be one of: {', '.join(t.value for t in MemoryType)}",
        ) from exc


def _as_source(value: SourceCategory | str) -> SourceCategory:
    try:
        return SourceCategory(value)
    except ValueError as exc:
        raise MemoryStoreError(
            MEMORY_INVALID_INPUT,
            f"source must be one of: {', '.join(s.value for s in SourceCategory)}",
        ) from exc


class InMemoryMemoryStore:
    """Deterministic in-memory :class:`MemoryStore` (the required baseline)."""

    def __init__(
        self,
        limits: MemoryLimits | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._limits = limits or MemoryLimits()
        self._clock: Clock = clock or utc_now
        self._items: dict[str, Memory] = {}

    @property
    def limits(self) -> MemoryLimits:
        return self._limits

    # -- internal validation ------------------------------------------------

    def _validate_content(self, content: str) -> None:
        if not isinstance(content, str) or not content.strip():
            raise MemoryStoreError(MEMORY_INVALID_INPUT, "content must be a non-empty string")
        if len(content) > self._limits.max_content_chars:
            raise MemoryStoreError(
                MEMORY_LIMIT_EXCEEDED,
                f"content is {len(content)} chars; limit is {self._limits.max_content_chars} chars",
            )
        if contains_secret_like_content(content):
            raise MemoryStoreError(
                SECRET_LIKE_CONTENT,
                "content looks like credential material and is not stored (conservative heuristic)",
            )

    def _validate_metadata(self, metadata: dict[str, object]) -> None:
        size = metadata_size_bytes(metadata)
        if size > self._limits.max_metadata_bytes:
            raise MemoryStoreError(
                MEMORY_LIMIT_EXCEEDED,
                f"metadata is {size} bytes; limit is {self._limits.max_metadata_bytes} bytes",
            )

    def _default_expiration(
        self, memory_type: MemoryType, expires_at: datetime | None, now: datetime
    ) -> datetime | None:
        if expires_at is not None:
            return _ensure_aware(expires_at)
        if memory_type is MemoryType.SHORT_TERM:
            return now + timedelta(seconds=self._limits.short_term_ttl_s)
        if memory_type is MemoryType.WORKING:
            return now + timedelta(seconds=self._limits.working_ttl_s)
        # long_term / knowledge: never expire implicitly.
        return None

    # -- MemoryStore protocol ------------------------------------------------

    def remember(
        self,
        *,
        memory_type: MemoryType | str,
        content: str,
        source: SourceCategory | str,
        source_ref: str | None = None,
        metadata: dict[str, object] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> Memory:
        """Create (or idempotently replace) a memory. Enforces all policy."""
        now = _ensure_aware(now if now is not None else self._clock())
        mtype = _as_type(memory_type, "memory_type")
        src = _as_source(source)
        if mtype not in self._limits.allowed_types:
            raise MemoryStoreError(
                MEMORY_TYPE_NOT_ALLOWED,
                f"memory type {mtype.value!r} is not allowed by policy",
            )
        self._validate_content(content)
        meta = dict(metadata or {})
        self._validate_metadata(meta)
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise MemoryStoreError(MEMORY_INVALID_INPUT, "confidence must be in [0, 1]")
        memory_id = make_memory_id(mtype, src, content, now)
        if memory_id not in self._items and len(self._items) >= self._limits.max_items:
            raise MemoryStoreError(
                MEMORY_LIMIT_EXCEEDED,
                f"memory store is full ({len(self._items)} memories; "
                f"limit is {self._limits.max_items})",
            )
        memory = Memory(
            memory_id=memory_id,
            memory_type=mtype,
            content=content,
            source=src,
            source_ref=source_ref,
            metadata=meta,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            expires_at=self._default_expiration(mtype, expires_at, now),
            active=True,
        )
        self._items[memory_id] = memory
        return memory

    def get(self, memory_id: str) -> Memory | None:
        return self._items.get(memory_id)

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        active: bool | None = None,
        now: datetime | None = None,
    ) -> Memory:
        """Update mutable fields of an existing memory.

        Identity fields (``memory_id``, ``memory_type``, ``source``,
        ``source_ref``, ``created_at``) are immutable by construction —
        this API cannot change them. Passing ``None`` for an optional
        field leaves the stored value untouched (updates are
        field-wise, not full replacements).
        """
        now = _ensure_aware(now if now is not None else self._clock())
        memory = self._items.get(memory_id)
        if memory is None:
            raise MemoryStoreError(MEMORY_NOT_FOUND, f"memory not found: {memory_id}")
        if content is not None:
            self._validate_content(content)
        if metadata is not None:
            self._validate_metadata(metadata)
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise MemoryStoreError(MEMORY_INVALID_INPUT, "confidence must be in [0, 1]")
        updated = memory.model_copy(
            update={
                "content": content if content is not None else memory.content,
                "metadata": dict(metadata) if metadata is not None else memory.metadata,
                "confidence": confidence if confidence is not None else memory.confidence,
                "expires_at": (
                    _ensure_aware(expires_at) if expires_at is not None else memory.expires_at
                ),
                "active": active if active is not None else memory.active,
                "updated_at": now,
            }
        )
        self._items[memory_id] = updated
        return updated

    def forget(self, memory_id: str, *, hard: bool = False, now: datetime | None = None) -> Memory:
        """Soft-deactivate (default) or hard-delete ONE memory.

        Never recursive: only the given id is affected. Returns the memory
        in its final state (deactivated, or as it was before deletion).
        """
        now = _ensure_aware(now if now is not None else self._clock())
        memory = self._items.get(memory_id)
        if memory is None:
            raise MemoryStoreError(MEMORY_NOT_FOUND, f"memory not found: {memory_id}")
        if hard:
            del self._items[memory_id]
            return memory
        deactivated = memory.model_copy(update={"active": False, "updated_at": now})
        self._items[memory_id] = deactivated
        return deactivated

    def list(
        self,
        *,
        memory_type: MemoryType | str | None = None,
        active_only: bool = True,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> MemoryList:
        """List memories in deterministic (created_at, memory_id) order.

        ``active_only=True`` (default) returns memories that are active AND
        not expired (expired short-term/working memories are never
        'active'). ``active_only=False`` returns everything (including
        soft-forgotten), still in deterministic order. The result is
        always bounded: ``limit=None`` defaults to the configured recall
        cap, and any explicit limit is clamped to that cap.
        """
        now = _ensure_aware(now if now is not None else self._clock())
        mtype = _as_type(memory_type, "memory_type") if memory_type is not None else None
        ordered = sorted(self._items.values(), key=lambda m: (m.created_at, m.memory_id))
        out: list[Memory] = []
        for memory in ordered:
            if mtype is not None and memory.memory_type is not mtype:
                continue
            if active_only and not memory.is_effectively_active(now):
                continue
            out.append(memory)
        if limit is None:
            limit = self._limits.max_recall_results
        limit = max(1, min(int(limit), self._limits.max_recall_results))
        return out[:limit]

    def recall(
        self,
        query: str,
        *,
        memory_type: MemoryType | str | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> MemoryList:
        """Deterministic lexical recall over active, non-expired memories.

        Same ranking as Phase 4 document retrieval: sum of idf*tf per query
        term over in-scope memories, ties broken by ``memory_id``. Only
        ``active`` memories that are not expired are in scope.
        """
        now = _ensure_aware(now if now is not None else self._clock())
        if not isinstance(query, str) or not query.strip():
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_set = set(query_terms)

        mtype = _as_type(memory_type, "memory_type") if memory_type is not None else None
        scope: list[Memory] = []
        for memory in self._items.values():
            if not memory.is_effectively_active(now):
                continue
            if mtype is not None and memory.memory_type is not mtype:
                continue
            scope.append(memory)
        if not scope:
            return []

        df: dict[str, int] = {}
        term_freqs: list[dict[str, int]] = []
        for memory in scope:
            freq: dict[str, int] = {}
            for term in tokenize(memory.content):
                freq[term] = freq.get(term, 0) + 1
            term_freqs.append(freq)
            for term in freq:
                if term in query_set:
                    df[term] = df.get(term, 0) + 1
        n = len(scope)

        scored: list[tuple[float, str, Memory]] = []
        for memory, freq in zip(scope, term_freqs, strict=True):
            score = 0.0
            for term in query_terms:
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
                score += idf * tf
            if score > 0.0:
                scored.append((score, memory.memory_id, memory))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if limit is None:
            limit = self._limits.max_recall_results
        limit = max(1, min(int(limit), self._limits.max_recall_results))
        return [memory for _score, _id, memory in scored[:limit]]

    def purge_expired(self, now: datetime | None = None) -> int:
        """Hard-remove expired ``short_term``/``working`` memories.

        ``long_term``/``knowledge`` are never purged (protected). Returns
        how many memories were removed.
        """
        now = _ensure_aware(now if now is not None else self._clock())
        expired = [
            memory
            for memory in self._items.values()
            if memory.memory_type in (MemoryType.SHORT_TERM, MemoryType.WORKING)
            and memory.is_expired(now)
        ]
        for memory in expired:
            del self._items[memory.memory_id]
        return len(expired)
