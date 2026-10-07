"""Document parser registry (Phase 4).

``default_registry`` assembles the built-in parsers. Binary parsers declare
their optional libraries up front but import them lazily at parse time; if
a library is missing the parse fails with a structured ``parser_unavailable``
error instead of an import crash.
"""

from __future__ import annotations

from .base import (
    DocumentParser,
    ExtractedContent,
    ParserRegistry,
    RawSection,
    normalize_newlines,
    split_markdown,
)
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .pptx_parser import PptxParser
from .text_parser import MarkdownParser, TextParser, decode_text
from .xlsx_parser import XlsxParser

__all__ = [
    "DocumentParser",
    "DocxParser",
    "ExtractedContent",
    "MarkdownParser",
    "ParserRegistry",
    "PdfParser",
    "PptxParser",
    "RawSection",
    "TextParser",
    "XlsxParser",
    "decode_text",
    "default_registry",
    "normalize_newlines",
    "split_markdown",
]


def default_registry() -> ParserRegistry:
    """A fresh registry with all built-in parsers registered."""
    registry = ParserRegistry()
    for parser in (
        TextParser(),
        MarkdownParser(),
        PdfParser(),
        DocxParser(),
        PptxParser(),
        XlsxParser(),
    ):
        registry.register(parser)
    return registry
