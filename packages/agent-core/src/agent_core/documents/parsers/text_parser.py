"""Plain-text and Markdown parsers (Phase 4). No external dependencies.

Text files are decoded as UTF-8 (BOM tolerated); if that fails the content
is checked for binary markers (NUL bytes) and rejected with ``decode_failed``
otherwise decoded as Latin-1 **with a recorded warning** (Latin-1 never
fails, so a text file is never lost — the fallback is always reported).

Content is data only: markdown "headings" and any instruction-like text are
preserved verbatim and never interpreted.
"""

from __future__ import annotations

from ..errors import DECODE_FAILED, DocumentError
from ..limits import DocumentLimits
from .base import (
    SECTION_TEXT,
    DocumentParser,
    ExtractedContent,
    RawSection,
    normalize_newlines,
    split_markdown,
)

_MEDIA_TYPES = {
    "text": "text/plain",
    "markdown": "text/markdown",
}


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """Decode bytes to text: UTF-8 (BOM-aware) first, then a *reported*
    Latin-1 fallback; binary-looking content (NUL bytes) is rejected."""
    if b"\x00" in data:
        raise DocumentError(DECODE_FAILED, "content is binary and cannot be decoded as text")
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return data.decode("latin-1"), ["content is not valid UTF-8; decoded as Latin-1 (fallback)"]


class _TextLikeParser(DocumentParser):
    document_type: str

    def _extract(self, data: bytes, limits: DocumentLimits) -> ExtractedContent:
        text, warnings = decode_text(data)
        text = normalize_newlines(text)
        if not text.strip():
            return ExtractedContent(
                document_type=self.document_type,
                media_type=_MEDIA_TYPES[self.document_type],
                title=None,
                sections=[],
                warnings=[*warnings, "document contains no text"],
                stats={"chars": 0},
            )
        return ExtractedContent(
            document_type=self.document_type,
            media_type=_MEDIA_TYPES[self.document_type],
            title=None,
            sections=[RawSection(section_type=SECTION_TEXT, text=text)]
            if self.document_type == "text"
            else split_markdown(text),
            warnings=warnings,
            stats={"chars": len(text)},
        )


class TextParser(_TextLikeParser):
    name = "text"
    document_type = "text"
    supported_extensions = ("txt",)
    supported_media_types = ("text/plain",)


class MarkdownParser(_TextLikeParser):
    name = "markdown"
    document_type = "markdown"
    supported_extensions = ("md", "markdown")
    supported_media_types = ("text/markdown",)
