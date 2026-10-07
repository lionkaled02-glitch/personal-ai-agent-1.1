"""Configurable safety limits for document processing (Phase 4).

Limits fail safely: exceeding an input-size limit is a structured error
(``document_too_large``); exceeding extraction/capacity limits during
parsing produces an explicitly **reported** truncation (``Document.truncated``
+ a warning), never a silent cut; exceeding a hard per-unit limit is
``parser_limit_exceeded``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


@dataclass(frozen=True)
class DocumentLimits:
    """Safety limits for parsing, chunking, and retrieval."""

    max_input_bytes: int = 10_485_760  # 10 MiB raw file
    max_extracted_chars: int = 500_000  # total extracted text budget
    max_pages: int = 200  # PDF
    max_slides: int = 100  # PPTX
    max_sheets: int = 20  # XLSX
    max_sections: int = 500
    max_chunks: int = 500
    chunk_size: int = 800  # max characters per chunk
    chunk_overlap: int = 100  # characters of overlap between consecutive chunks
    max_search_results: int = 10
    max_query_chars: int = 500

    def __post_init__(self) -> None:
        for name in (
            "max_input_bytes",
            "max_extracted_chars",
            "max_pages",
            "max_slides",
            "max_sheets",
            "max_sections",
            "max_chunks",
            "chunk_size",
            "max_search_results",
            "max_query_chars",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be a non-negative integer, got {self.chunk_overlap!r}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @classmethod
    def from_settings(cls, settings: Settings) -> DocumentLimits:
        return cls(
            max_input_bytes=settings.document_max_input_bytes,
            max_extracted_chars=settings.document_max_extracted_chars,
            max_pages=settings.document_max_pages,
            max_slides=settings.document_max_slides,
            max_sheets=settings.document_max_sheets,
            max_sections=settings.document_max_sections,
            max_chunks=settings.document_max_chunks,
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap,
            max_search_results=settings.document_max_search_results,
            max_query_chars=settings.document_max_query_chars,
        )
