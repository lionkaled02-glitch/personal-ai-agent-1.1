from pathlib import Path

from agent_core.documents.chunking import chunk_document
from agent_core.documents.limits import DocumentLimits
from agent_core.documents.models import Document, DocumentSection
from agent_core.documents.sqlite_retrieval import SQLiteKnowledgeStore


def _doc() -> Document:
    return Document(
        document_id="doc-test",
        source_path="notes.txt",
        filename="notes.txt",
        media_type="text/plain",
        document_type="text",
        title="Notes",
        sections=[
            DocumentSection(
                section_id="doc-test-s0000",
                document_id="doc-test",
                section_type="text",
                text="alpha beta gamma",
                index=0,
            )
        ],
    )


def test_sqlite_knowledge_store_survives_restart(tmp_path: Path) -> None:
    limits = DocumentLimits(chunk_size=100, chunk_overlap=0)
    path = tmp_path / "knowledge.sqlite3"
    first = SQLiteKnowledgeStore(path, limits)
    doc = _doc()
    first.add_document(doc, chunk_document(doc, limits).chunks)
    first.close()

    second = SQLiteKnowledgeStore(path, limits)
    assert second.get_document("doc-test") is not None
    hits = second.search("alpha")
    assert len(hits) == 1
    assert hits[0].chunk.text == "alpha beta gamma"
    second.close()
