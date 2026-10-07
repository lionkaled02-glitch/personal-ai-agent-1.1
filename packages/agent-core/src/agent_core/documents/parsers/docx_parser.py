"""DOCX parser (Phase 4) — python-docx behind the parser interface.

python-docx is an *optional* dependency (``agent-core[docs]``) imported
lazily. It reads OOXML document structure only: paragraphs (with heading
styles) and tables in body order. python-docx never executes VBA/macros
(.docx is the macro-free part of the Office Open XML family); nothing in
this parser runs code found in the document.
"""

from __future__ import annotations

import io
import re
from typing import Any

from ..errors import (
    DOCUMENT_CORRUPT,
    EXTRACTION_FAILED,
    PARSER_UNAVAILABLE,
    DocumentError,
)
from ..limits import DocumentLimits
from .base import (
    SECTION_HEADING,
    SECTION_TABLE,
    SECTION_TEXT,
    DocumentParser,
    ExtractedContent,
    RawSection,
)

_HEADING_LEVEL_RE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)


def table_to_text(rows: Any) -> str:
    """Deterministic textual rendering of table rows (list of cell values).

    Cells are joined with `` | ``; rows with ``\\n``; cell newlines collapse
    to a single space so the table stays one line per row.
    """
    lines = []
    for row in rows:
        cells = []
        for value in row:
            cell = "" if value is None else str(value)
            cell = " ".join(cell.split())
            cells.append(cell)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


class DocxParser(DocumentParser):
    name = "docx"
    document_type = "docx"
    supported_extensions = ("docx",)
    supported_media_types = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def _extract(self, data: bytes, limits: DocumentLimits) -> ExtractedContent:
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise DocumentError(
                PARSER_UNAVAILABLE,
                "DOCX parsing requires the 'python-docx' package (install agent-core[docs])",
            ) from exc

        try:
            doc = Document(io.BytesIO(data))
        except Exception as exc:
            raise DocumentError(
                DOCUMENT_CORRUPT, f"DOCX could not be parsed: {type(exc).__name__}"
            ) from exc

        try:
            sections: list[RawSection] = []
            title: str | None = None
            body = doc.element.body
            for child in body.iterchildren():
                tag = child.tag.split("}")[-1]
                if tag == "p":
                    paragraph = Paragraph(child, doc)
                    text = paragraph.text.strip()
                    if not text:
                        continue
                    style_name = paragraph.style.name if paragraph.style is not None else ""
                    if title is None and style_name.lower() == "title":
                        title = text
                        continue  # the title is document metadata, not a section
                    level_match = _HEADING_LEVEL_RE.match(style_name)
                    if level_match:
                        sections.append(
                            RawSection(
                                section_type=SECTION_HEADING,
                                text=text,
                                heading=text,
                                metadata={"level": int(level_match.group(1))},
                            )
                        )
                    else:
                        sections.append(RawSection(section_type=SECTION_TEXT, text=text))
                elif tag == "tbl":
                    table = Table(child, doc)
                    rows = [[cell.text for cell in row.cells] for row in table.rows]
                    text = table_to_text(rows)
                    sections.append(
                        RawSection(
                            section_type=SECTION_TABLE,
                            text=text,
                            metadata={
                                "rows": len(rows),
                                "columns": max((len(r) for r in rows), default=0),
                            },
                        )
                    )
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(
                EXTRACTION_FAILED, f"DOCX text extraction failed: {type(exc).__name__}"
            ) from exc

        return ExtractedContent(
            document_type=self.document_type,
            media_type=self.supported_media_types[0],
            title=title,
            sections=sections,
            warnings=[],
            stats={"sections_raw": len(sections)},
        )
