"""PDF parser (Phase 4) — pypdf behind the parser interface.

pypdf is an *optional* dependency (``agent-core[docs]``) imported lazily so
the core stays importable without it. Extraction is text-only: pypdf does
not execute JavaScript or open external URIs found in a PDF; we additionally
never follow links. One section per page, preserving page numbers.
"""

from __future__ import annotations

import io
from typing import Any

from ..errors import (
    DOCUMENT_CORRUPT,
    EXTRACTION_FAILED,
    PARSER_UNAVAILABLE,
    DocumentError,
)
from ..limits import DocumentLimits
from .base import SECTION_PAGE, DocumentParser, ExtractedContent, RawSection


class PdfParser(DocumentParser):
    name = "pdf"
    document_type = "pdf"
    supported_extensions = ("pdf",)
    supported_media_types = ("application/pdf",)

    def _extract(self, data: bytes, limits: DocumentLimits) -> ExtractedContent:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise DocumentError(
                PARSER_UNAVAILABLE,
                "PDF parsing requires the 'pypdf' package (install agent-core[docs])",
            ) from exc

        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            pages = reader.pages
        except Exception as exc:
            raise DocumentError(
                DOCUMENT_CORRUPT, f"PDF could not be parsed: {type(exc).__name__}"
            ) from exc

        try:
            title: str | None = None
            try:
                info = reader.metadata
                if info is not None:
                    raw_title = info.get("/Title")
                    if raw_title:
                        title = str(raw_title)
            except Exception:  # metadata is best-effort only
                pass

            sections: list[RawSection] = []
            warnings: list[str] = []
            truncated = False
            max_pages = limits.max_pages
            if len(pages) > max_pages:
                truncated = True
                warnings.append(
                    f"only the first {max_pages} pages were extracted "
                    f"({len(pages) - max_pages} page(s) skipped)"
                )
            for i in range(min(len(pages), max_pages)):
                try:
                    text = pages[i].extract_text() or ""
                except Exception as exc:
                    warnings.append(f"page {i + 1}: text extraction failed ({type(exc).__name__})")
                    text = ""
                sections.append(
                    RawSection(
                        section_type=SECTION_PAGE,
                        text=text,
                        location={"page": i + 1},
                    )
                )
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(
                EXTRACTION_FAILED, f"PDF text extraction failed: {type(exc).__name__}"
            ) from exc

        stats: dict[str, Any] = {
            "pages": len(pages),
            "pages_extracted": min(len(pages), limits.max_pages),
        }
        return ExtractedContent(
            document_type=self.document_type,
            media_type=self.supported_media_types[0],
            title=title,
            sections=sections,
            warnings=warnings,
            stats=stats,
            truncated=truncated,
        )
