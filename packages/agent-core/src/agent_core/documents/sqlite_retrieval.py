"""Durable SQLite-backed document retrieval index.

The index stores normalized documents/chunks as JSON and rebuilds the bounded
lexical index on startup. It keeps the existing RetrievalIndex contract, so a
future vector index can replace it without changing document tools.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .limits import DocumentLimits
from .models import Document, DocumentChunk
from .retrieval import KnowledgeStore


class SQLiteKnowledgeStore(KnowledgeStore):
    """Persistent bounded lexical knowledge store."""

    def __init__(self, path: Path, limits: DocumentLimits | None = None) -> None:
        super().__init__(limits)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._db.commit()
        self._load()

    def _load(self) -> None:
        documents: dict[str, Document] = {}
        chunks_by_doc: dict[str, list[DocumentChunk]] = {}
        for document_id, payload in self._db.execute("SELECT document_id, payload FROM documents"):
            try:
                documents[document_id] = Document.model_validate(json.loads(payload))
            except Exception:
                continue
        for _chunk_id, document_id, payload in self._db.execute(
            "SELECT chunk_id, document_id, payload FROM chunks"
        ):
            try:
                chunk = DocumentChunk.model_validate(json.loads(payload))
            except Exception:
                continue
            chunks_by_doc.setdefault(document_id, []).append(chunk)
        for document_id in sorted(documents):
            self.add_document(
                documents[document_id],
                sorted(chunks_by_doc.get(document_id, []), key=lambda c: c.index),
                persist=False,
            )

    def add_document(
        self, document: Document, chunks: list[DocumentChunk], *, persist: bool = True
    ) -> None:
        super().add_document(document, chunks)
        if not persist:
            return
        with self._db:
            self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
            self._db.execute(
                "INSERT OR REPLACE INTO documents(document_id,payload) VALUES (?,?)",
                (
                    document.document_id,
                    json.dumps(document.model_dump(mode="json"), ensure_ascii=False),
                ),
            )
            self._db.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id,document_id,payload) VALUES (?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        json.dumps(c.model_dump(mode="json"), ensure_ascii=False),
                    )
                    for c in chunks
                ],
            )

    def add_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None:
        document = self.get_document(document_id)
        if document is None:
            return
        self.add_document(document, chunks)

    def remove_document(self, document_id: str) -> bool:
        existed = super().remove_document(document_id)
        if existed:
            with self._db:
                self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self._db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return existed

    def close(self) -> None:
        self._db.close()
