"""Configurable safety limits/policies for the memory layer (Phase 5).

Limits fail safely: violations raise :class:`MemoryStoreError` with a
stable code. Policies are explicit data — including *which memory types
are allowed at all* — so deployments can tighten them without code changes.

Long-term memory protection (by design):

- ``long_term`` and ``knowledge`` memories get **no implicit TTL** — they
  never expire just because time passed, and ``purge_expired`` never
  touches them.
- Conversely, they are never *created implicitly*: memory creation always
  requires an explicit permission-gated tool call (or a clearly defined
  trusted internal pathway). Nothing auto-saves conversation text.
- Deletion of any memory goes through the HIGH-permission ``forget`` tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from .models import MemoryType


@dataclass(frozen=True)
class MemoryLimits:
    """Safety limits and policy for memory storage, recall, and context."""

    max_items: int = 1_000
    max_content_chars: int = 4_000
    max_metadata_bytes: int = 4_096
    max_recall_results: int = 10
    max_context_chars: int = 8_000
    max_context_items: int = 20
    short_term_ttl_s: int = 3_600
    working_ttl_s: int = 86_400
    #: Policy: which memory types may be created at all.
    allowed_types: frozenset[MemoryType] = field(default_factory=lambda: frozenset(MemoryType))

    def __post_init__(self) -> None:
        for name in (
            "max_items",
            "max_content_chars",
            "max_metadata_bytes",
            "max_recall_results",
            "max_context_chars",
            "max_context_items",
            "short_term_ttl_s",
            "working_ttl_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not self.allowed_types:
            raise ValueError("allowed_types must not be empty")
        if not self.allowed_types <= set(MemoryType):
            raise ValueError(f"allowed_types contains unknown types: {self.allowed_types}")

    @classmethod
    def from_settings(cls, settings: Settings) -> MemoryLimits:
        return cls(
            max_items=settings.memory_max_items,
            max_content_chars=settings.memory_max_content_chars,
            max_metadata_bytes=settings.memory_max_metadata_bytes,
            max_recall_results=settings.memory_max_recall_results,
            max_context_chars=settings.memory_max_context_chars,
            max_context_items=settings.memory_max_context_items,
            short_term_ttl_s=settings.memory_short_term_ttl_s,
            working_ttl_s=settings.memory_working_ttl_s,
        )
