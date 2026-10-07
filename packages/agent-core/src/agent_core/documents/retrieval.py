"""Deterministic lexical retrieval over indexed document chunks (Phase 4).

The :class:`RetrievalIndex` protocol is the stable seam: a future
vector/embedding-based store can implement the same interface without
changing the document model or the tools. :class:`KnowledgeStore` is the
in-memory lexical implementation — token-based ranking (TF weighted by
document frequency / inverse document frequency), fully deterministic, no
external model required.
"""

from __future__ import annotations

import math
import re
from typing import Any, Protocol, runtime_checkable

from .errors import DOCUMENT_NOT_INDEXED, DocumentError
from .limits import DocumentLimits
from .models import Document, DocumentChunk, SearchResult

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters/digits/unicode word chars)."""
    return _TOKEN_RE.findall(text.casefold())


@runtime_checkable
class RetrievalIndex(Protocol):
    """Provider-neutral retrieval interface (lexical today, vectors later)."""

    def add_document(self, document: Document, chunks: list[DocumentChunk]) -> None: ...

    def add_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None: ...

    def remove_document(self, document_id: str) -> bool: ...

    def get_document(self, document_id: str) -> Document | None: ...

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None: ...

    def list_documents(self) -> list[Document]: ...

    def search(
        self, query: str, *, document_id: str | None = None, limit: int | None = None
    ) -> list[SearchResult]: ...


class KnowledgeStore:
    """In-memory lexical index of normalized documents and their chunks."""

    def __init__(self, limits: DocumentLimits | None = None) -> None:
        self._limits = limits or DocumentLimits()
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._chunk_by_id: dict[str, tuple[str, DocumentChunk]] = {}

    @property
    def limits(self) -> DocumentLimits:
        return self._limits

    # -- indexing ---------------------------------------------------------

    def add_document(self, document: Document, chunks: list[DocumentChunk]) -> None:
        """Index a document with its (replacing) chunk set."""
        self._documents[document.document_id] = document
        self._replace_chunks(document.document_id, chunks)

    def add_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None:
        """Replace the chunk set of an already-indexed document."""
        if document_id not in self._documents:
            raise DocumentError(DOCUMENT_NOT_INDEXED, f"document is not indexed: {document_id}")
        self._replace_chunks(document_id, chunks)

    def _replace_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None:
        for old in self._chunks.get(document_id, ()):
            self._chunk_by_id.pop(old.chunk_id, None)
        self._chunks[document_id] = list(chunks)
        for chunk in chunks:
            self._chunk_by_id[chunk.chunk_id] = (document_id, chunk)

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all its chunks. Returns True if it existed."""
        if document_id not in self._documents:
            return False
        del self._documents[document_id]
        for old in self._chunks.pop(document_id, ()):
            self._chunk_by_id.pop(old.chunk_id, None)
        return True

    # -- lookup -----------------------------------------------------------

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        entry = self._chunk_by_id.get(chunk_id)
        return entry[1] if entry is not None else None

    def list_documents(self) -> list[Document]:
        """All indexed documents, in deterministic (source_path) order."""
        return sorted(self._documents.values(), key=lambda d: d.source_path)

    # -- search -----------------------------------------------------------

    def search(
        self, query: str, *, document_id: str | None = None, limit: int | None = None
    ) -> list[SearchResult]:
        """Deterministic lexical search over chunk text.

        Ranking: sum over query terms of ``idf * tf`` where
        ``idf = ln((N + 1) / (df + 1)) + 1`` (N = chunks in scope, df = chunks
        containing the term). Ties break on chunk_id, so results are stable.
        ``limit`` is clamped to ``[1, max_search_results]``.
        """
        if limit is None:
            limit = self._limits.max_search_results
        limit = max(1, min(int(limit), self._limits.max_search_results))
        seen: set[str] = set()
        query_terms: list[str] = []
        for term in tokenize(query):
            if term not in seen:
                seen.add(term)
                query_terms.append(term)
        if not query_terms:
            return []
        query_set = set(query_terms)

        scope: list[DocumentChunk] = []
        if document_id is not None:
            scope = list(self._chunks.get(document_id, ()))
        else:
            for doc_id in sorted(self._chunks):
                scope.extend(self._chunks[doc_id])
        if not scope:
            return []

        # Document frequency per query term within scope.
        df: dict[str, int] = {}
        term_freqs: list[dict[str, int]] = []
        for chunk in scope:
            freq: dict[str, int] = {}
            for term in tokenize(chunk.text):
                freq[term] = freq.get(term, 0) + 1
            term_freqs.append(freq)
            for term in freq:
                if term in query_set:
                    df[term] = df.get(term, 0) + 1
        n = len(scope)

        results: list[SearchResult] = []
        for chunk, freq in zip(scope, term_freqs, strict=True):
            matched: list[str] = []
            score = 0.0
            for term in query_terms:
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                matched.append(term)
                idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
                score += idf * tf
            if score > 0.0:
                doc = self._documents.get(chunk.document_id)
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(score, 6),
                        matched_terms=matched,
                        document_title=doc.title if doc is not None else None,
                    )
                )

        results.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return results[:limit]


def chunk_search_metadata(result: SearchResult) -> dict[str, Any]:
    """Bounded, metadata-only view of a search hit for tool outputs."""
    return {
        "chunk_id": result.chunk.chunk_id,
        "document_id": result.chunk.document_id,
        "section_id": result.chunk.section_id,
        "index": result.chunk.index,
        "score": result.score,
        "matched_terms": result.matched_terms,
        "title": result.document_title,
        "location": result.chunk.location,
    }
