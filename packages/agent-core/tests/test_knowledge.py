"""Knowledge store: deterministic lexical retrieval (Phase 4).

Covers indexing, replacement, removal, lookup, and the search semantics:
token-based TF/IDF ranking, deterministic ordering, document filtering,
bounded results, and query handling. Fully in-memory and offline — no
embeddings, no external model.
"""

from __future__ import annotations

import pytest
from agent_core.documents import (
    Document,
    DocumentChunk,
    DocumentLimits,
    KnowledgeStore,
    chunk_document,
    default_registry,
    make_document_id,
)
from agent_core.documents.errors import DocumentError
from agent_core.documents.retrieval import tokenize

LIMITS = DocumentLimits()


def doc_with_chunks(source_path: str, text: str) -> tuple[Document, list[DocumentChunk]]:
    parser = default_registry().for_filename(source_path)
    assert parser is not None
    document = parser.parse(
        text.encode("utf-8"),
        source_path=source_path,
        filename=source_path.rsplit("/", 1)[-1],
        document_id=make_document_id(source_path),
        limits=LIMITS,
    )
    return document, chunk_document(document, LIMITS).chunks


class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("The Quick-Brown fox!") == ["the", "quick", "brown", "fox"]

    def test_unicode_words(self) -> None:
        assert tokenize("café naïve") == ["café", "naïve"]

    def test_numbers(self) -> None:
        assert tokenize("order 42 and 4.5") == ["order", "42", "and", "4", "5"]


class TestIndexing:
    def test_add_and_get_document(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "hello world")
        store.add_document(document, chunks)
        assert store.get_document(document.document_id) is document
        assert len(store.list_documents()) == 1

    def test_add_document_replaces_chunks(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "old content")
        store.add_document(document, chunks)
        first_ids = {hit.chunk.chunk_id for hit in store.search("old", limit=10)}
        assert first_ids  # sanity: the term is found before replacement
        new_chunks = chunk_document(document, LIMITS).chunks
        store.add_document(document, new_chunks)
        assert {hit.chunk.chunk_id for hit in store.search("old", limit=10)} == first_ids
        assert len(store.list_documents()) == 1

    def test_add_chunks_requires_indexed_document(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "hello")
        with pytest.raises(DocumentError) as excinfo:
            store.add_chunks(document.document_id, chunks)
        assert excinfo.value.code == "document_not_indexed"

    def test_remove_document(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "hello")
        store.add_document(document, chunks)
        chunk_id = chunks[0].chunk_id
        assert store.remove_document(document.document_id) is True
        assert store.get_document(document.document_id) is None
        assert store.get_chunk(chunk_id) is None
        assert store.remove_document(document.document_id) is False
        assert store.list_documents() == []

    def test_list_documents_sorted(self) -> None:
        store = KnowledgeStore(LIMITS)
        b, b_chunks = doc_with_chunks("zeta.txt", "zeta text")
        a, a_chunks = doc_with_chunks("alpha.txt", "alpha text")
        store.add_document(b, b_chunks)
        store.add_document(a, a_chunks)
        assert [d.source_path for d in store.list_documents()] == ["alpha.txt", "zeta.txt"]

    def test_get_chunk(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "hello world")
        store.add_document(document, chunks)
        assert store.get_chunk(chunks[0].chunk_id) is chunks[0]
        assert store.get_chunk("doc-0000000000000000-c0000") is None


class TestSearch:
    def test_finds_matching_chunk(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "the quarterly budget review")
        store.add_document(document, chunks)
        hits = store.search("budget")
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == chunks[0].chunk_id
        assert hits[0].matched_terms == ["budget"]
        assert hits[0].document_title is None  # txt has no title

    def test_no_match_is_empty(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "nothing relevant here")
        store.add_document(document, chunks)
        assert store.search("budget") == []
        assert store.search("   ") == []  # no tokens

    def test_ranking_prefers_more_occurrences(self) -> None:
        store = KnowledgeStore(LIMITS)
        dense, dense_chunks = doc_with_chunks("dense.txt", "budget budget budget notes")
        sparse, sparse_chunks = doc_with_chunks("sparse.txt", "the budget line")
        store.add_document(dense, dense_chunks)
        store.add_document(sparse, sparse_chunks)
        hits = store.search("budget", limit=10)
        assert [h.chunk.document_id for h in hits] == [
            dense.document_id,
            sparse.document_id,
        ]
        assert hits[0].score > hits[1].score

    def test_scores_are_descending(self) -> None:
        store = KnowledgeStore(LIMITS)
        documents = []
        for name, text in (
            ("a.txt", "alpha beta gamma"),
            ("b.txt", "alpha delta"),
            ("c.txt", "alpha epsilon zeta"),
        ):
            document, chunks = doc_with_chunks(name, text)
            store.add_document(document, chunks)
            documents.append(document)
        hits = store.search("alpha", limit=10)
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))

    def test_deterministic_order(self) -> None:
        store = KnowledgeStore(LIMITS)
        for name in ("one.txt", "two.txt", "three.txt"):
            document, chunks = doc_with_chunks(name, f"shared term in {name}")
            store.add_document(document, chunks)
        first = store.search("shared", limit=10)
        second = store.search("shared", limit=10)
        assert [h.chunk.chunk_id for h in first] == [h.chunk.chunk_id for h in second]

    def test_limit_clamped_to_max(self) -> None:
        store = KnowledgeStore(DocumentLimits(max_search_results=2))
        for i in range(5):
            document, chunks = doc_with_chunks(f"f{i}.txt", "common word")
            store.add_document(document, chunks)
        hits = store.search("common", limit=999)
        assert len(hits) <= 2

    def test_limit_minimum_one(self) -> None:
        store = KnowledgeStore(LIMITS)
        document, chunks = doc_with_chunks("a.txt", "match me")
        store.add_document(document, chunks)
        assert len(store.search("match", limit=1)) == 1

    def test_document_filter(self) -> None:
        store = KnowledgeStore(LIMITS)
        a, a_chunks = doc_with_chunks("a.txt", "unique word")
        b, b_chunks = doc_with_chunks("b.txt", "unique word")
        store.add_document(a, a_chunks)
        store.add_document(b, b_chunks)
        hits = store.search("unique", document_id=a.document_id, limit=10)
        assert hits and all(h.chunk.document_id == a.document_id for h in hits)
        # Unknown document filter yields nothing, no error.
        assert store.search("unique", document_id="doc-unknown", limit=10) == []

    def test_empty_store(self) -> None:
        store = KnowledgeStore(LIMITS)
        assert store.search("anything") == []
