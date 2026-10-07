"""Memory layer (Phase 5).

Explicit, permission-gated, provider-neutral memory:

- **Model** — ``Memory`` (Pydantic, deterministic ids): type
  (short_term/working/long_term/knowledge), content, provenance
  (user_explicit/task/agent/document/system + optional source ref),
  confidence, timestamps, optional expiration, active flag.
- **Storage** — ``MemoryStore`` protocol + ``InMemoryMemoryStore``
  (limits/policy enforced at the store layer; deterministic ordering;
  soft-delete ``forget``; expiration handling with long-term protection).
- **Retrieval** — ``MemoryRetriever`` protocol + ``LexicalMemoryRetriever``
  (deterministic lexical recall; no embeddings, no external model).
- **Policies** — ``MemoryLimits`` (allowed types, content/metadata/item
  caps, recall/context caps, short-term/working TTLs).
- **Secrets** — conservative heuristic guard + the structural guarantee
  that nothing auto-persists conversation text.

Memories are *about* the user/tasks/decisions; document *knowledge* stays
in the Phase 4 knowledge store. Retrieved content is untrusted data.
"""

from __future__ import annotations

from .errors import (
    MEMORY_FIELD_IMMUTABLE,
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
from .retrieval import LexicalMemoryRetriever, MemoryRetriever
from .sqlite_store import SQLiteMemoryStore
from .store import InMemoryMemoryStore, MemoryStore

__all__ = [
    "MEMORY_FIELD_IMMUTABLE",
    "MEMORY_INVALID_INPUT",
    "MEMORY_LIMIT_EXCEEDED",
    "MEMORY_NOT_FOUND",
    "MEMORY_TYPE_NOT_ALLOWED",
    "SECRET_LIKE_CONTENT",
    "InMemoryMemoryStore",
    "LexicalMemoryRetriever",
    "Memory",
    "MemoryLimits",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryType",
    "SQLiteMemoryStore",
    "SourceCategory",
    "contains_secret_like_content",
    "make_memory_id",
    "metadata_size_bytes",
]
