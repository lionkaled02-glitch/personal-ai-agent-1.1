"""Document processing & knowledge foundation (Phase 4).

Safe, modular, provider-agnostic:

- **Models** — normalized ``Document`` / ``DocumentSection`` /
  ``DocumentChunk`` (Pydantic, deterministic, serializable).
- **Parsers** — ``DocumentParser`` interface + registry; built-ins for TXT,
  Markdown, PDF, DOCX, PPTX, XLSX (binary parsers use optional libraries
  imported lazily). Document content is untrusted **data** only.
- **Chunking** — deterministic, bounded, overlap-capable.
- **Retrieval** — :class:`KnowledgeStore` (deterministic lexical search)
  behind the :class:`RetrievalIndex` protocol (a future vector store can
  implement the same interface).
"""

from __future__ import annotations

from .chunking import ChunkingResult, chunk_document, split_text
from .errors import (
    CHUNK_NOT_FOUND,
    DECODE_FAILED,
    DOCUMENT_CORRUPT,
    DOCUMENT_NOT_FOUND,
    DOCUMENT_NOT_INDEXED,
    DOCUMENT_TOO_LARGE,
    EXTRACTION_FAILED,
    INVALID_DOCUMENT,
    INVALID_INPUT,
    INVALID_QUERY,
    PARSER_LIMIT_EXCEEDED,
    PARSER_UNAVAILABLE,
    SECURITY_VIOLATION,
    UNSUPPORTED_DOCUMENT_TYPE,
    DocumentError,
)
from .limits import DocumentLimits
from .models import (
    Document,
    DocumentChunk,
    DocumentSection,
    SearchResult,
    make_chunk_id,
    make_document_id,
    make_section_id,
)
from .parsers import (
    DocumentParser,
    DocxParser,
    ExtractedContent,
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    PptxParser,
    RawSection,
    TextParser,
    XlsxParser,
    default_registry,
)
from .retrieval import KnowledgeStore, RetrievalIndex, tokenize

__all__ = [
    "CHUNK_NOT_FOUND",
    "DECODE_FAILED",
    "DOCUMENT_CORRUPT",
    "DOCUMENT_NOT_FOUND",
    "DOCUMENT_NOT_INDEXED",
    "DOCUMENT_TOO_LARGE",
    "EXTRACTION_FAILED",
    "INVALID_DOCUMENT",
    "INVALID_INPUT",
    "INVALID_QUERY",
    "PARSER_LIMIT_EXCEEDED",
    "PARSER_UNAVAILABLE",
    "SECURITY_VIOLATION",
    "UNSUPPORTED_DOCUMENT_TYPE",
    "ChunkingResult",
    "Document",
    "DocumentChunk",
    "DocumentError",
    "DocumentLimits",
    "DocumentParser",
    "DocumentSection",
    "DocxParser",
    "ExtractedContent",
    "KnowledgeStore",
    "MarkdownParser",
    "ParserRegistry",
    "PdfParser",
    "PptxParser",
    "RawSection",
    "RetrievalIndex",
    "SQLiteKnowledgeStore",
    "SearchResult",
    "TextParser",
    "XlsxParser",
    "chunk_document",
    "default_registry",
    "make_chunk_id",
    "make_document_id",
    "make_section_id",
    "split_text",
    "tokenize",
]

from .sqlite_retrieval import SQLiteKnowledgeStore
