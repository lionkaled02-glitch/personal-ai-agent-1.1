"""Document parser abstraction (Phase 4).

A parser turns raw file bytes into a normalized :class:`Document`. The core
document model is coupled to this interface only — never to a specific
parser library. Binary parsers import their (optional) library lazily inside
``_extract``; if the library is missing they raise a structured
``parser_unavailable`` error.

**Document content is untrusted data.** Parsers extract text and structure
only. They never execute anything found in a document (no macros, scripts,
shell, or code of any kind) and never interpret text as instructions.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..errors import PARSER_LIMIT_EXCEEDED, DocumentError
from ..limits import DocumentLimits
from ..models import (
    Document,
    DocumentSection,
    make_section_id,
)

#: Section kinds used by the built-in parsers.
SECTION_TEXT = "text"
SECTION_HEADING = "heading"
SECTION_PAGE = "page"
SECTION_SLIDE = "slide"
SECTION_SHEET = "sheet"
SECTION_TABLE = "table"


@dataclass
class RawSection:
    """One unbounded section produced by a parser's ``_extract``."""

    section_type: str
    text: str
    heading: str | None = None
    location: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedContent:
    """Intermediate parser output before normalization/limiting."""

    document_type: str
    media_type: str
    title: str | None
    sections: list[RawSection]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False  # parser-level truncation (e.g. page/slide/sheet cap)


def normalize_newlines(text: str) -> str:
    """Universal newlines (\\r\\n / \\r -> \\n) without touching other text."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def require_unit_limit(value: int, limit: int, unit: str) -> None:
    """Hard per-unit limit (one page/slide/sheet) — fail closed."""
    if value > limit:
        raise DocumentError(
            PARSER_LIMIT_EXCEEDED,
            f"a single {unit} exceeds the extraction limit ({value} > {limit} chars)",
        )


def split_at_budget(text: str, budget: int) -> tuple[str, bool]:
    """Split text at a character budget, preferring a line boundary.

    Returns ``(kept, was_cut)``. The cut is made on the last newline within
    the budget so a logical line is not split in the middle when possible.
    """
    if len(text) <= budget:
        return text, False
    window = text[:budget]
    cut = window.rfind("\n")
    if cut > budget // 2:
        return window[:cut], True
    return window, True


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def split_markdown(text: str) -> list[RawSection]:
    """Split markdown into sections at heading lines (level 1-6).

    The preamble before the first heading (if any) is one ``text`` section.
    Each heading starts a section whose ``heading`` is the heading text and
    whose ``text`` is the heading line plus the body up to the next heading
    (verbatim, so sub-headings are preserved inside the section text).
    """
    lines = normalize_newlines(text).split("\n")
    sections: list[RawSection] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    current_start = 0

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_heading
        body = "\n".join(current_lines).strip("\n")
        if body.strip() or current_heading is not None:
            if current_heading is None:
                if body.strip():
                    sections.append(
                        RawSection(
                            section_type=SECTION_TEXT,
                            text=body,
                            location={"line": current_start + 1},
                        )
                    )
            else:
                # Verbatim block: heading line + body up to the next heading,
                # so sub-headings are preserved inside the section text.
                sections.append(
                    RawSection(
                        section_type=SECTION_HEADING,
                        text=body,
                        heading=current_heading,
                        location={"line": current_start + 1},
                    )
                )
        current_lines = []
        current_heading = None

    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and (current_heading is not None or current_lines):
            flush(i)
            current_start = i
        if match:
            current_heading = match.group(2)
        current_lines.append(line)
    flush(len(lines))
    return sections


class DocumentParser(ABC):
    """Provider-neutral parser interface.

    Subclasses declare the extensions/media types they handle and implement
    ``_extract`` (library-specific). ``parse`` and ``normalize`` are shared:
    extract -> normalize (limits, truncation reporting, stats, ids).
    """

    name: ClassVar[str]
    supported_extensions: ClassVar[tuple[str, ...]] = ()
    supported_media_types: ClassVar[tuple[str, ...]] = ()

    def parse(
        self,
        data: bytes,
        *,
        source_path: str,
        filename: str,
        document_id: str,
        limits: DocumentLimits,
    ) -> Document:
        """Parse raw bytes into a normalized :class:`Document`."""
        extracted = self._extract(data, limits)
        return self.normalize(
            extracted,
            source_path=source_path,
            filename=filename,
            document_id=document_id,
            limits=limits,
        )

    def normalize(
        self,
        extracted: ExtractedContent,
        *,
        source_path: str,
        filename: str,
        document_id: str,
        limits: DocumentLimits,
    ) -> Document:
        """Apply limits + deterministic section building to extracted content."""
        sections: list[DocumentSection] = []
        warnings: list[str] = list(extracted.warnings)
        truncated = extracted.truncated
        remaining = limits.max_extracted_chars
        skipped = 0

        for raw in extracted.sections:
            text = normalize_newlines(raw.text)
            if remaining <= 0:
                skipped += 1
                continue
            if len(text) > remaining:
                text, cut = split_at_budget(text, remaining)
                if cut:
                    truncated = True
                    warnings.append(
                        "extraction truncated at the max extracted character "
                        f"limit ({limits.max_extracted_chars})"
                    )
            remaining -= len(text)
            if len(sections) >= limits.max_sections:
                skipped += 1
                continue
            sections.append(
                DocumentSection(
                    section_id=make_section_id(document_id, len(sections)),
                    document_id=document_id,
                    section_type=raw.section_type,
                    heading=raw.heading,
                    text=text,
                    index=len(sections),
                    location=raw.location,
                    metadata=dict(raw.metadata),
                )
            )
        if skipped:
            truncated = True
            warnings.append(
                f"{skipped} section(s) skipped due to extraction limits "
                "(max sections / max extracted chars)"
            )

        total_chars = sum(len(s.text) for s in sections)
        # Parser-specific stats first; the canonical normalized keys always
        # win so limits-driven truncation is reflected in the final stats.
        stats: dict[str, Any] = dict(extracted.stats)
        stats["chars"] = total_chars
        stats["sections"] = len(sections)
        stats["warnings"] = len(warnings)

        return Document(
            document_id=document_id,
            source_path=source_path,
            filename=filename,
            media_type=extracted.media_type,
            document_type=extracted.document_type,
            title=extracted.title,
            sections=sections,
            warnings=warnings,
            stats=stats,
            truncated=truncated,
        )

    @abstractmethod
    def _extract(self, data: bytes, limits: DocumentLimits) -> ExtractedContent:
        """Read raw bytes into :class:`ExtractedContent` (library-specific)."""


class ParserRegistry:
    """Selects the parser for a file by extension or media type."""

    def __init__(self) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        self._by_media_type: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser) -> None:
        for ext in parser.supported_extensions:
            self._by_extension[ext.lstrip(".").lower()] = parser
        for media in parser.supported_media_types:
            self._by_media_type[media.lower()] = parser

    def for_extension(self, extension: str) -> DocumentParser | None:
        return self._by_extension.get(extension.lstrip(".").lower())

    def for_media_type(self, media_type: str) -> DocumentParser | None:
        return self._by_media_type.get(media_type.lower())

    def for_filename(self, filename: str) -> DocumentParser | None:
        name = filename.lower()
        for ext, parser in self._by_extension.items():
            if name.endswith("." + ext):
                return parser
        return None

    def list_parsers(self) -> Sequence[DocumentParser]:
        seen: dict[str, DocumentParser] = {}
        for parser in self._by_extension.values():
            seen.setdefault(parser.name, parser)
        return tuple(seen.values())
