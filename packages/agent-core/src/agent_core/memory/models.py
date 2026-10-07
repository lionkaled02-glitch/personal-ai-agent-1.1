"""Strongly typed, provider-neutral memory models (Phase 5).

Memories are information **about the user, task state, project state, or
prior decisions** — distinct from *knowledge* (content retrieved from
documents, which lives in the Phase 4 knowledge store). A memory of type
``knowledge`` is a deliberately stored knowledge *fact* (with provenance);
it is still a memory, and it is never loaded from a document implicitly.

Determinism: memory ids are stable functions of their inputs (type,
provenance, content, creation time), so the same remember() call made at
the same instant yields the same identity. Ordering is deterministic
elsewhere (created_at, then id).

The model stores no secrets by design: creation is explicit (permission-
gated tool or a clearly defined internal pathway), never an automatic
capture of conversation text, and a conservative secret heuristic rejects
obvious credential patterns at the store layer (a heuristic, not a
guarantee — see SECURITY.md).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Lifecycle class of a memory.

    - ``short_term``: transient (auto-TTL from ``MEMORY_SHORT_TERM_TTL_S``)
    - ``working``: task/project-scoped (auto-TTL from ``MEMORY_WORKING_TTL_S``)
    - ``long_term``: durable by explicit intent; never auto-expired,
      never created implicitly
    - ``knowledge``: a stored knowledge fact (provenance-tracked)
    """

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    KNOWLEDGE = "knowledge"


class SourceCategory(StrEnum):
    """Provenance of a memory: where it came from."""

    USER_EXPLICIT = "user_explicit"
    TASK = "task"
    AGENT = "agent"
    DOCUMENT = "document"
    SYSTEM = "system"


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def make_memory_id(
    memory_type: MemoryType | str,
    source: SourceCategory | str,
    content: str,
    created_at: datetime,
) -> str:
    """Stable memory id derived from its creation inputs.

    Idempotent: the same type/source/content/created_at always yields the
    same id. The id is immutable for the memory's lifetime (updates never
    change it).
    """
    return f"mem-{_digest(str(memory_type), str(source), content, created_at.isoformat())}"


def metadata_size_bytes(metadata: dict[str, Any]) -> int:
    """Deterministic serialized size of a metadata mapping (JSON, UTF-8)."""
    return len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))


class Memory(BaseModel):
    """One stored memory. Data only — never instructions."""

    memory_id: str
    memory_type: MemoryType
    content: str
    #: Provenance category (where it came from).
    source: SourceCategory
    #: Optional provenance reference (e.g. a workspace-relative document
    #: path for ``document`` sources, a task id for ``task`` sources).
    source_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Optional confidence in [0, 1] where applicable.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    #: Explicit expiration; ``short_term``/``working`` get one from their
    #: TTL policy at creation when omitted. ``long_term``/``knowledge``
    #: never expire implicitly.
    expires_at: datetime | None = None
    #: Soft-delete flag: ``forget`` deactivates by default. Inactive
    #: memories are hidden from recall and active listings.
    active: bool = True

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at < now

    def is_effectively_active(self, now: datetime) -> bool:
        """Active and not expired — what recall/context see as 'active'."""
        return self.active and not self.is_expired(now)
