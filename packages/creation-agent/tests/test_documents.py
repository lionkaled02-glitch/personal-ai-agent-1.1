from pathlib import Path

from creation_agent import generate_docx, generate_pptx, generate_xlsx


def test_document_generators(tmp_path: Path):
    generate_docx("Title", ["Hello"], tmp_path / "a.docx")
    generate_xlsx("Data", [["a", "b"], [1, 2]], tmp_path / "b.xlsx")
    generate_pptx("Deck", [("Slide", "Body")], tmp_path / "c.pptx")
    assert (tmp_path / "a.docx").stat().st_size > 0
    assert (tmp_path / "b.xlsx").stat().st_size > 0
    assert (tmp_path / "c.pptx").stat().st_size > 0
