"""Document processing: parsing, normalization, limits, and chunking (Phase 4).

Deterministic temp fixtures only (see ``document_fixtures``); no real user
files, no network. Covers: all six formats, malformed/corrupt inputs,
extraction limits with explicit truncation reporting, section/page/slide/
sheet caps, table normalization, deterministic chunking with overlap and
count caps, and metadata/location preservation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_core.documents import (
    Document,
    DocumentLimits,
    chunk_document,
    default_registry,
    make_document_id,
    split_text,
)
from agent_core.documents.errors import DocumentError
from document_fixtures import make_docx, make_pdf, make_pptx, make_xlsx

LIMITS = DocumentLimits()


def parse_file(tmp_path: Path, name: str, data: bytes, limits: DocumentLimits = LIMITS) -> Document:
    """Parse bytes as if they lived at ``name`` in the workspace."""
    parser = default_registry().for_filename(name)
    assert parser is not None, f"no parser for {name}"
    return parser.parse(
        data,
        source_path=name,
        filename=name,
        document_id=make_document_id(name),
        limits=limits,
    )


# ---------------------------------------------------------------------------
# Parsing: all six formats
# ---------------------------------------------------------------------------


class TestParsingFormats:
    def test_text(self, tmp_path: Path) -> None:
        # write_bytes so the fixture contains literal CRLF on every OS —
        # write_text performs newline translation on Windows (\r -> \r\r\n).
        (tmp_path / "a.txt").write_bytes(b"hello world\r\nsecond line")
        doc = parse_file(tmp_path, "a.txt", (tmp_path / "a.txt").read_bytes())
        assert doc.document_type == "text"
        assert doc.media_type == "text/plain"
        assert doc.sections[0].text == "hello world\nsecond line"  # CRLF normalized
        assert doc.truncated is False
        assert doc.stats["sections"] == 1

    def test_markdown_headings(self, tmp_path: Path) -> None:
        text = "Preamble line.\n\n# One\n\nbody one\n\n## Two\n\nbody two\n"
        doc = parse_file(tmp_path, "a.md", text.encode("utf-8"))
        assert doc.document_type == "markdown"
        headings = [s.heading for s in doc.sections if s.heading]
        assert headings == ["One", "Two"]
        # Preamble is its own text section; heading sections carry their body.
        assert doc.sections[0].section_type == "text"
        assert doc.sections[0].text == "Preamble line."
        one = next(s for s in doc.sections if s.heading == "One")
        assert "body one" in one.text
        assert one.section_type == "heading"

    def test_markdown_preamble_only(self, tmp_path: Path) -> None:
        doc = parse_file(tmp_path, "a.md", b"just text, no headings\n")
        assert len(doc.sections) == 1
        assert doc.sections[0].section_type == "text"

    def test_pdf_pages_and_locations(self, tmp_path: Path) -> None:
        make_pdf(tmp_path / "r.pdf", ["First page of the report.", "Second page mentions budget."])
        doc = parse_file(tmp_path, "r.pdf", (tmp_path / "r.pdf").read_bytes())
        assert doc.document_type == "pdf"
        assert len(doc.sections) == 2
        assert doc.sections[0].location == {"page": 1}
        assert doc.sections[1].location == {"page": 2}
        assert "First page of the report." in doc.sections[0].text
        assert doc.stats["pages"] == 2

    def test_docx_headings_and_title(self, tmp_path: Path) -> None:
        make_docx(tmp_path / "r.docx")
        doc = parse_file(tmp_path, "r.docx", (tmp_path / "r.docx").read_bytes())
        assert doc.document_type == "docx"
        assert doc.title == "Quarterly Report"
        headings = [s for s in doc.sections if s.section_type == "heading"]
        assert [h.heading for h in headings] == ["Revenue", "Expenses"]
        levels = [h.metadata.get("level") for h in headings]
        assert levels == [1, 2]

    def test_docx_table_normalization(self, tmp_path: Path) -> None:
        make_docx(tmp_path / "r.docx")
        doc = parse_file(tmp_path, "r.docx", (tmp_path / "r.docx").read_bytes())
        tables = [s for s in doc.sections if s.section_type == "table"]
        assert len(tables) == 1
        assert tables[0].text == "Metric | Value\nRevenue | 12%"
        assert tables[0].metadata == {"rows": 2, "columns": 2}

    def test_docx_body_order(self, tmp_path: Path) -> None:
        """Headings, paragraphs, and tables appear in document order."""
        make_docx(tmp_path / "r.docx")
        doc = parse_file(tmp_path, "r.docx", (tmp_path / "r.docx").read_bytes())
        sequence = [s.section_type for s in doc.sections]
        assert sequence == [
            "heading",  # Revenue
            "text",
            "heading",  # Expenses
            "text",
            "table",
        ]

    def test_pptx_slides_and_locations(self, tmp_path: Path) -> None:
        make_pptx(tmp_path / "d.pptx")
        doc = parse_file(tmp_path, "d.pptx", (tmp_path / "d.pptx").read_bytes())
        assert doc.document_type == "pptx"
        assert len(doc.sections) == 2
        assert doc.sections[0].location == {"slide": 1}
        assert doc.sections[1].location == {"slide": 2}
        assert doc.sections[0].heading == "Slide One"
        assert "Quarterly summary goes here." in doc.sections[0].text
        assert doc.title == "Slide One"  # first slide's title

    def test_xlsx_sheets_and_grid(self, tmp_path: Path) -> None:
        make_xlsx(tmp_path / "d.xlsx")
        doc = parse_file(tmp_path, "d.xlsx", (tmp_path / "d.xlsx").read_bytes())
        assert doc.document_type == "xlsx"
        assert len(doc.sections) == 2
        data = doc.sections[0]
        assert data.location == {"sheet": "Data"}
        assert data.text == "Product | Units\nWidget | 30"
        assert data.metadata == {"rows": 2, "columns": 2}
        notes = doc.sections[1]
        assert notes.location == {"sheet": "Notes"}
        assert notes.text == "Shipped in Q3"


# ---------------------------------------------------------------------------
# Decoding behavior
# ---------------------------------------------------------------------------


class TestDecoding:
    def test_utf8_bom_tolerated(self, tmp_path: Path) -> None:
        doc = parse_file(tmp_path, "a.txt", b"\xef\xbb\xbfhello")
        assert doc.sections[0].text == "hello"
        assert doc.warnings == []

    def test_latin1_fallback_is_reported(self, tmp_path: Path) -> None:
        data = "café".encode("latin-1")  # 0xE9 is invalid UTF-8
        doc = parse_file(tmp_path, "a.txt", data)
        assert doc.sections[0].text == "café"
        assert any("Latin-1" in w for w in doc.warnings)

    def test_binary_content_is_decode_failed(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentError) as excinfo:
            parse_file(tmp_path, "a.txt", b"head\x00tail")
        assert excinfo.value.code == "decode_failed"

    def test_empty_text_reports_no_text(self, tmp_path: Path) -> None:
        doc = parse_file(tmp_path, "a.txt", b"   \n  ")
        assert doc.sections == []
        assert doc.warnings == ["document contains no text"]


# ---------------------------------------------------------------------------
# Malformed / corrupt inputs (structured errors, never crashes)
# ---------------------------------------------------------------------------


class TestMalformed:
    @pytest.mark.parametrize(
        "name, data",
        [
            ("a.pdf", b"%PDF-1.4 not really a pdf"),
            ("a.pdf", b""),
            ("a.docx", b"PK\x03\x04 not a zip"),
            ("a.docx", b""),
            ("a.pptx", b"not a presentation at all"),
            ("a.pptx", b""),
            ("a.xlsx", b"\x00\x01\x02 not a workbook"),
            ("a.xlsx", b""),
        ],
    )
    def test_corrupt_raises_document_error(self, tmp_path: Path, name: str, data: bytes) -> None:
        with pytest.raises(DocumentError) as excinfo:
            parse_file(tmp_path, name, data)
        assert excinfo.value.code in (
            "document_corrupt",
            "extraction_failed",
            "invalid_document",
        )

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        assert default_registry().for_filename("archive.zip") is None
        assert default_registry().for_filename("noext") is None

    def test_registry_lists_all_six_parsers(self) -> None:
        names = {p.name for p in default_registry().list_parsers()}
        assert names == {"text", "markdown", "pdf", "docx", "pptx", "xlsx"}


# ---------------------------------------------------------------------------
# Limits: extraction caps always report truncation (never silent)
# ---------------------------------------------------------------------------


class TestExtractionLimits:
    def test_max_extracted_chars_truncates_with_warning(self, tmp_path: Path) -> None:
        limits = DocumentLimits(max_extracted_chars=50)
        doc = parse_file(tmp_path, "a.txt", b"x" * 200, limits=limits)
        assert doc.truncated is True
        assert doc.stats["chars"] <= 50
        assert any("truncated" in w for w in doc.warnings)

    def test_max_sections_skips_with_report(self, tmp_path: Path) -> None:
        limits = DocumentLimits(max_sections=3)
        text = "".join(f"## H{i}\n\nbody {i}\n" for i in range(10))
        doc = parse_file(tmp_path, "a.md", text.encode("utf-8"), limits=limits)
        assert len(doc.sections) == 3
        assert doc.truncated is True
        assert any("skipped" in w for w in doc.warnings)

    def test_pdf_page_cap(self, tmp_path: Path) -> None:
        limits = DocumentLimits(max_pages=2)
        make_pdf(tmp_path / "r.pdf", [f"page {i}" for i in range(5)])
        doc = parse_file(tmp_path, "r.pdf", (tmp_path / "r.pdf").read_bytes(), limits=limits)
        assert len(doc.sections) == 2
        assert doc.truncated is True
        assert doc.stats["pages"] == 5
        assert doc.stats["pages_extracted"] == 2
        assert any("page" in w for w in doc.warnings)

    def test_pptx_slide_cap(self, tmp_path: Path) -> None:
        limits = DocumentLimits(max_slides=2)
        from pptx import Presentation

        prs = Presentation()
        for i in range(4):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = f"S{i}"
        prs.save(str(tmp_path / "d.pptx"))
        doc = parse_file(tmp_path, "d.pptx", (tmp_path / "d.pptx").read_bytes(), limits=limits)
        assert len(doc.sections) == 2
        assert doc.truncated is True
        assert doc.stats["slides"] == 4
        assert doc.stats["slides_extracted"] == 2

    def test_xlsx_sheet_cap(self, tmp_path: Path) -> None:
        limits = DocumentLimits(max_sheets=2)
        from openpyxl import Workbook

        wb = Workbook()
        for i in range(4):
            sheet = wb.active if i == 0 else wb.create_sheet()
            sheet.title = f"S{i}"
            sheet["A1"] = f"value {i}"
        wb.save(str(tmp_path / "d.xlsx"))
        doc = parse_file(tmp_path, "d.xlsx", (tmp_path / "d.xlsx").read_bytes(), limits=limits)
        assert len(doc.sections) == 2
        assert doc.truncated is True
        assert doc.stats["sheets"] == 4
        assert doc.stats["sheets_extracted"] == 2
        assert [s.location for s in doc.sections] == [{"sheet": "S0"}, {"sheet": "S1"}]


class TestLimitsValidation:
    def test_overlap_must_be_below_size(self) -> None:
        with pytest.raises(ValueError):
            DocumentLimits(chunk_size=100, chunk_overlap=100)

    def test_positive_limits_required(self) -> None:
        with pytest.raises(ValueError):
            DocumentLimits(max_chunks=0)
        with pytest.raises(ValueError):
            DocumentLimits(max_input_bytes=0)


# ---------------------------------------------------------------------------
# Chunking: deterministic, bounded, metadata-preserving
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_small_text_is_one_chunk(self) -> None:
        # Blocks are joined with a single newline inside a chunk.
        assert split_text("hello\n\nworld", 800, 100) == ["hello\nworld"]

    def test_empty_text(self) -> None:
        assert split_text("", 800, 100) == []
        assert split_text("   \n  ", 800, 100) == []

    def test_paragraph_boundaries_preferred(self) -> None:
        a = ("alpha " * 50).strip()  # 299 chars
        b = ("beta " * 50).strip()  # 249 chars
        chunks = split_text(f"{a}\n\n{b}", 500, 50)
        assert all(len(c) <= 500 for c in chunks)
        # Each paragraph fits within a chunk: no paragraph is split.
        assert any(a in c for c in chunks)
        assert any(b in c for c in chunks)

    def test_long_paragraph_hard_split_with_overlap(self) -> None:
        block = "".join(f"word{i} " for i in range(200))  # ~1300 chars
        chunks = split_text(block, 400, 100)
        assert all(len(c) <= 400 for c in chunks)
        # Overlap: consecutive hard slices share up to 100 chars.
        if len(chunks) > 1:
            prev_tail = chunks[0][-100:]
            assert any(prev_tail in c for c in chunks[1:])

    def test_never_exceeds_size(self) -> None:
        text = "par " * 500 + "\n\n" + "x" * 5000
        chunks = split_text(text, 300, 30)
        assert all(len(c) <= 300 for c in chunks)

    def test_deterministic(self) -> None:
        text = "one two three\n\nfour five six\n\n" + "g" * 2500
        a = split_text(text, 500, 75)
        b = split_text(text, 500, 75)
        assert a == b


class TestChunkDocument:
    def _doc(self, tmp_path: Path, text: str, name: str = "a.txt") -> Document:
        return parse_file(tmp_path, name, text.encode("utf-8"))

    def test_chunk_ids_and_order(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "para one\n\npara two\n\npara three\n")
        result = chunk_document(doc, LIMITS)
        ids = [c.chunk_id for c in result.chunks]
        assert ids == [f"{doc.document_id}-c{i:04d}" for i in range(len(ids))]
        assert [c.index for c in result.chunks] == list(range(len(result.chunks)))
        assert all(c.document_id == doc.document_id for c in result.chunks)

    def test_chunk_section_and_location_metadata(self, tmp_path: Path) -> None:
        make_pdf(tmp_path / "r.pdf", ["First page of the report.", "Second page mentions budget."])
        doc = parse_file(tmp_path, "r.pdf", (tmp_path / "r.pdf").read_bytes())
        result = chunk_document(doc, LIMITS)
        assert result.chunks[0].location == {"page": 1}
        assert result.chunks[1].location == {"page": 2}
        assert all(
            c.metadata["source_path"] == "r.pdf" and c.metadata["document_type"] == "pdf"
            for c in result.chunks
        )

    def test_chunk_size_bound(self, tmp_path: Path) -> None:
        limits = DocumentLimits(chunk_size=200, chunk_overlap=20)
        doc = self._doc(tmp_path, ("some text content here. " * 400))
        result = chunk_document(doc, limits)
        assert result.chunks
        assert all(len(c.text) <= 200 for c in result.chunks)

    def test_max_chunks_cap_reports_truncation(self, tmp_path: Path) -> None:
        limits = DocumentLimits(chunk_size=50, chunk_overlap=5, max_chunks=3)
        doc = self._doc(tmp_path, " ".join(f"w{i}" for i in range(2000)))
        result = chunk_document(doc, limits)
        assert len(result.chunks) == 3
        assert result.truncated is True
        assert result.skipped_chunks > 0

    def test_no_truncation_when_fitting(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "small doc\n")
        result = chunk_document(doc, LIMITS)
        assert result.truncated is False
        assert result.skipped_chunks == 0

    def test_deterministic(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "alpha beta gamma\n\ndelta epsilon\n\n" + "z" * 2000)
        a = chunk_document(doc, LIMITS)
        b = chunk_document(doc, LIMITS)
        assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
        assert [c.text for c in a.chunks] == [c.text for c in b.chunks]

    def test_whitespace_only_section_produces_no_chunk(self, tmp_path: Path) -> None:
        doc = self._doc(tmp_path, "\n\n   \n")
        result = chunk_document(doc, LIMITS)
        assert result.chunks == []


class TestDocumentModel:
    def test_document_id_deterministic(self) -> None:
        assert make_document_id("docs/a.txt") == make_document_id("docs/a.txt")
        assert make_document_id("docs/a.txt") != make_document_id("docs/b.txt")

    def test_document_is_serializable(self, tmp_path: Path) -> None:
        doc = parse_file(tmp_path, "a.txt", b"round trip")
        payload = doc.model_dump_json()
        again = Document.model_validate_json(payload)
        assert again.document_id == doc.document_id
        assert again.sections[0].text == "round trip"
