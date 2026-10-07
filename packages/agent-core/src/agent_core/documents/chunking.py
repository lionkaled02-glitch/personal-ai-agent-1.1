"""Deterministic chunking of normalized documents (Phase 4).

Chunking is pure text processing over the normalized :class:`Document`:
no embeddings, no external provider, fully deterministic. It prefers
paragraph boundaries, falls back to sentence-free hard splits only when a
single paragraph exceeds the chunk size, carries a configurable overlap
between consecutive chunks, and enforces a hard chunk-count cap with an
explicit truncation report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .limits import DocumentLimits
from .models import Document, DocumentChunk, make_chunk_id

_BLOCK_RE = re.compile(r"\n{2,}")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into pieces of at most ``size`` characters.

    - Splits on blank-line paragraph boundaries first (logical structure).
    - A single paragraph longer than ``size`` is hard-split with ``overlap``
      characters of carry-over between its slices.
    - Consecutive chunks share up to ``overlap`` characters (a tail of the
      previous chunk prepended to the next) so context is not lost at
      boundaries.
    - Deterministic: same input, same output, always.
    """
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    overlap = max(0, min(overlap, size - 1))

    blocks = [b.strip() for b in _BLOCK_RE.split(text)]
    blocks = [b for b in blocks if b]
    if not blocks:
        return []

    # Units: paragraphs, hard-split when a single paragraph exceeds size.
    units: list[str] = []
    for block in blocks:
        if len(block) <= size:
            units.append(block)
            continue
        step = size - overlap
        pos = 0
        while pos < len(block):
            units.append(block[pos : pos + size])
            if pos + size >= len(block):
                break
            pos += step

    # Pack units into chunks of at most size characters.
    chunks: list[str] = []
    pieces: list[str] = []
    length = 0
    for unit in units:
        cost = len(unit) + (1 if pieces else 0)
        if pieces and length + cost > size:
            chunks.append("\n".join(pieces))
            # Carry the overlap tail of the previous chunk into the next one.
            tail = chunks[-1][-overlap:] if overlap else ""
            room = size - len(unit) - (1 if tail else 0)
            tail = tail[: max(room, 0)]
            pieces = [tail, unit] if tail else [unit]
            length = len("\n".join(pieces))
        else:
            pieces.append(unit)
            length += cost
    if pieces:
        chunks.append("\n".join(pieces))
    return [c for c in chunks if c.strip()]


@dataclass(frozen=True)
class ChunkingResult:
    """Chunks plus an explicit report of any limit-driven truncation."""

    chunks: list[DocumentChunk] = field(default_factory=list)
    truncated: bool = False
    skipped_chunks: int = 0


def chunk_document(document: Document, limits: DocumentLimits) -> ChunkingResult:
    """Deterministically chunk a normalized document.

    Preserves document/section ids and source location metadata on every
    chunk. The global chunk order follows document order. Enforces
    ``limits.max_chunks`` (reporting truncation + how many were skipped).
    """
    size = limits.chunk_size
    overlap = limits.chunk_overlap
    max_chunks = limits.max_chunks

    chunks: list[DocumentChunk] = []
    skipped = 0
    truncated = False

    for section in document.sections:
        pieces = split_text(section.text, size, overlap)
        for piece in pieces:
            if len(chunks) >= max_chunks:
                truncated = True
                skipped += 1
                continue
            metadata: dict[str, object] = {
                "source_path": document.source_path,
                "document_type": document.document_type,
            }
            if section.location:
                metadata["location"] = section.location
            if section.heading:
                metadata["heading"] = section.heading
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(document.document_id, len(chunks)),
                    document_id=document.document_id,
                    section_id=section.section_id,
                    text=piece,
                    index=len(chunks),
                    metadata=metadata,
                    location=section.location,
                )
            )
    return ChunkingResult(chunks=chunks, truncated=truncated, skipped_chunks=skipped)
