"""Provider-neutral normalized document models (Phase 4).

These are the *only* document shapes the rest of the system sees. Parsers
produce them; chunking splits them; retrieval indexes them. No parser
library type leaks past this boundary.

The models are deterministic and serializable (Pydantic). IDs are stable
functions of their inputs so re-indexing the same document yields the same
identifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Deterministic id helpers
# ---------------------------------------------------------------------------


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def make_document_id(source_path: str) -> str:
    """Stable document id derived from the workspace-relative path."""
    return f"doc-{_digest(source_path)}"


def make_section_id(document_id: str, index: int) -> str:
    return f"{document_id}-s{index:04d}"


def make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-c{index:04d}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DocumentSection(BaseModel):
    """One normalized block of a document, in document order.

    ``location`` carries optional page/slide/sheet metadata, e.g.
    ``{"page": 3}``, ``{"slide": 2}``, or ``{"sheet": "Data"}``.
    """

    section_id: str
    document_id: str
    section_type: str  # "text" | "heading" | "page" | "slide" | "sheet" | "table"
    heading: str | None = None
    text: str
    index: int
    location: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A fully parsed, normalized document (data only — never instructions)."""

    document_id: str
    source_path: str  # workspace-relative only
    filename: str
    media_type: str
    document_type: str  # "text" | "markdown" | "pdf" | "docx" | "pptx" | "xlsx"
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[DocumentSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False


class DocumentChunk(BaseModel):
    """A bounded slice of a document, in document order."""

    chunk_id: str
    document_id: str
    section_id: str | None = None
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    location: dict[str, Any] | None = None


class SearchResult(BaseModel):
    """One grounded retrieval hit: a chunk plus deterministic rank metadata."""

    chunk: DocumentChunk
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    document_title: str | None = None

    @property
    def source_path(self) -> str:
        value = self.chunk.metadata.get("source_path", "")
        return value if isinstance(value, str) else ""
