"""Shared helpers for memory tools (Phase 5).

Memory tools operate on the memory store/retriever only — no filesystem,
no network. Memory content in tool output is bounded by
``MEMORY_MAX_CONTENT_CHARS`` (the store enforces the same cap at write
time), and event payloads stay bounded by the Tool Runtime's
``bounded_value``. Tool error messages carry ids and stable codes —
never host paths, never full content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError


def parse_expires_at(raw: Any) -> datetime | None:
    """Parse an optional ISO-8601 expiration into an aware datetime.

    Naive datetimes are assumed UTC (documented contract). ``None`` passes
    through (no explicit expiration).
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise MemoryStoreError(MEMORY_INVALID_INPUT, "expires_at must be an ISO 8601 string")
    try:
        dt = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise MemoryStoreError(
            MEMORY_INVALID_INPUT, f"expires_at is not valid ISO 8601: {raw!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def iso(dt: datetime | None) -> str | None:
    """ISO-8601 serialization for tool outputs (None passes through)."""
    return dt.isoformat() if dt is not None else None


def memory_view(memory: Any) -> dict[str, Any]:
    """Stable, model-field-only view of a memory for tool outputs.

    Exposes only the public ``Memory`` fields — no storage internals.
    """
    out: dict[str, Any] = {
        "memory_id": memory.memory_id,
        "memory_type": str(memory.memory_type),
        "content": memory.content,
        "source": str(memory.source),
        "created_at": iso(memory.created_at),
        "updated_at": iso(memory.updated_at),
        "active": memory.active,
    }
    if memory.source_ref is not None:
        out["source_ref"] = memory.source_ref
    if memory.confidence is not None:
        out["confidence"] = memory.confidence
    if memory.expires_at is not None:
        out["expires_at"] = iso(memory.expires_at)
    return out


def memory_meta_view(memory: Any) -> dict[str, Any]:
    """Metadata-only view (no content) — for confirmation outputs."""
    out = memory_view(memory)
    out.pop("content")
    return out
