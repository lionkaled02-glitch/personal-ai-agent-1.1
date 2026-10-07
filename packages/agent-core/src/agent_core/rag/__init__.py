"""RAG context assembly (Phase 5).

Provider-neutral combination of retrieved memories (Phase 5) and retrieved
document chunks (Phase 4) into a structured, bounded, provenance-labeled
context. Retrieval-only: it never generates answers and never interprets
retrieved content (which is untrusted data). A future embedding/vector
provider plugs in through the existing ``MemoryRetriever`` /
``RetrievalIndex`` interfaces without changing this layer.
"""

from __future__ import annotations

from .context import Context, ContextBuilder, ContextItem, ContextRequest

__all__ = [
    "Context",
    "ContextBuilder",
    "ContextItem",
    "ContextRequest",
]
