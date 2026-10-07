"""Memory model, store, retrieval, and policy tests (Phase 5).

Covers: deterministic identity and ordering, type/source validation,
limits enforcement (content/metadata/items), provenance, confidence
bounds, soft vs hard forget, expiration semantics (TTL only for
short_term/working; long-term protection), lexical recall ranking and
tie-breaking, purge semantics, and the secret heuristic.

Fully deterministic via an injected fixed clock; no network, no APIs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from agent_core import (
    InMemoryMemoryStore,
    LexicalMemoryRetriever,
    Memory,
    MemoryLimits,
    MemoryStoreError,
    MemoryType,
    SourceCategory,
    make_memory_id,
    metadata_size_bytes,
)
from agent_core.memory.errors import (
    MEMORY_FIELD_IMMUTABLE,  # noqa: F401  (exported for API surface)
    MEMORY_INVALID_INPUT,
    MEMORY_LIMIT_EXCEEDED,
    MEMORY_NOT_FOUND,
    MEMORY_TYPE_NOT_ALLOWED,
    SECRET_LIKE_CONTENT,
)
from agent_core.memory.guards import contains_secret_like_content
from conftest import FIXED_NOW

FIXED: datetime = FIXED_NOW


@pytest.fixture
def limits() -> MemoryLimits:
    return MemoryLimits()


@pytest.fixture
def store(limits: MemoryLimits) -> InMemoryMemoryStore:
    return InMemoryMemoryStore(limits, clock=lambda: FIXED)


def remember(
    store: InMemoryMemoryStore,
    content: str,
    *,
    mtype: MemoryType | str = "long_term",
    source: SourceCategory | str = "user_explicit",
    **kwargs: Any,
) -> Memory:
    return store.remember(memory_type=mtype, content=content, source=source, **kwargs)


# ---------------------------------------------------------------------------
# Identity, model, and determinism
# ---------------------------------------------------------------------------


class TestIdentityAndModel:
    def test_id_is_stable_for_same_inputs(self) -> None:
        a = make_memory_id("long_term", "user_explicit", "c", FIXED)
        b = make_memory_id("long_term", "user_explicit", "c", FIXED)
        assert a == b
        assert a.startswith("mem-")
        assert len(a) == len("mem-") + 16

    def test_id_changes_with_any_input(self) -> None:
        base = make_memory_id("long_term", "user_explicit", "c", FIXED)
        assert make_memory_id("working", "user_explicit", "c", FIXED) != base
        assert make_memory_id("long_term", "agent", "c", FIXED) != base
        assert make_memory_id("long_term", "user_explicit", "d", FIXED) != base
        later = FIXED + timedelta(seconds=1)
        assert make_memory_id("long_term", "user_explicit", "c", later) != base

    def test_remember_is_idempotent_for_same_instant(self, store: InMemoryMemoryStore) -> None:
        first = remember(store, "alpha")
        second = remember(store, "alpha")
        assert first.memory_id == second.memory_id
        assert store.list() == [first]

    def test_same_content_different_time_gives_different_id(
        self, store: InMemoryMemoryStore
    ) -> None:
        first = store.remember(
            memory_type="long_term", content="alpha", source="user_explicit", now=FIXED
        )
        second = store.remember(
            memory_type="long_term",
            content="alpha",
            source="user_explicit",
            now=FIXED + timedelta(seconds=5),
        )
        assert first.memory_id != second.memory_id
        assert len(store.list()) == 2

    def test_metadata_size_bytes_is_deterministic(self) -> None:
        assert metadata_size_bytes({"a": 1, "b": [1, 2]}) == metadata_size_bytes(
            {"b": [1, 2], "a": 1}
        )
        assert metadata_size_bytes({}) == len(b"{}")
        assert metadata_size_bytes({"x": "y" * 100}) > metadata_size_bytes({"x": "y"})

    def test_is_expired_and_effectively_active(self) -> None:
        memory = Memory(
            memory_id="mem-x",
            memory_type=MemoryType.SHORT_TERM,
            content="c",
            source=SourceCategory.USER_EXPLICIT,
            created_at=FIXED,
            updated_at=FIXED,
            expires_at=FIXED + timedelta(seconds=10),
        )
        assert not memory.is_expired(FIXED)
        assert memory.is_effectively_active(FIXED)
        later = FIXED + timedelta(seconds=11)
        assert memory.is_expired(later)
        assert not memory.is_effectively_active(later)
        memory.active = False
        assert not memory.is_effectively_active(FIXED)


# ---------------------------------------------------------------------------
# Validation: types, sources, content, confidence, metadata
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_type_rejected(self, store: InMemoryMemoryStore) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "c", mtype="nope")
        assert exc.value.code == MEMORY_INVALID_INPUT

    def test_disallowed_type_by_policy_rejected(self) -> None:
        limits = MemoryLimits(allowed_types=frozenset({MemoryType.LONG_TERM}))
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "c", mtype="short_term")
        assert exc.value.code == MEMORY_TYPE_NOT_ALLOWED

    def test_unknown_source_rejected(self, store: InMemoryMemoryStore) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "c", source="nope")
        assert exc.value.code == MEMORY_INVALID_INPUT

    @pytest.mark.parametrize("content", ["", "   ", None, 42])
    def test_bad_content_rejected(self, store: InMemoryMemoryStore, content: Any) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            store.remember(memory_type="long_term", content=content, source="user_explicit")
        assert exc.value.code == MEMORY_INVALID_INPUT

    def test_oversized_content_rejected(self) -> None:
        limits = MemoryLimits(max_content_chars=10)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "x" * 11)
        assert exc.value.code == MEMORY_LIMIT_EXCEEDED

    def test_content_at_limit_accepted(self) -> None:
        limits = MemoryLimits(max_content_chars=10)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        memory = remember(store, "x" * 10)
        assert len(memory.content) == 10

    def test_oversized_metadata_rejected(self) -> None:
        limits = MemoryLimits(max_metadata_bytes=32)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "c", metadata={"big": "y" * 50})
        assert exc.value.code == MEMORY_LIMIT_EXCEEDED

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
    def test_confidence_out_of_range_rejected(
        self, store: InMemoryMemoryStore, confidence: float
    ) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "c", confidence=confidence)
        assert exc.value.code == MEMORY_INVALID_INPUT

    def test_confidence_bounds_accepted(self, store: InMemoryMemoryStore) -> None:
        assert remember(store, "a", confidence=0.0).confidence == 0.0
        assert remember(store, "b", confidence=1.0).confidence == 1.0

    def test_get_missing_returns_none(self, store: InMemoryMemoryStore) -> None:
        assert store.get("mem-does-not-exist") is None

    def test_update_missing_raises_not_found(self, store: InMemoryMemoryStore) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            store.update("mem-does-not-exist", content="x")
        assert exc.value.code == MEMORY_NOT_FOUND

    def test_forget_missing_raises_not_found(self, store: InMemoryMemoryStore) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            store.forget("mem-does-not-exist")
        assert exc.value.code == MEMORY_NOT_FOUND

    def test_limit_validation_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError):
            MemoryLimits(max_items=0)
        with pytest.raises(ValueError):
            MemoryLimits(max_items=True)  # bool is an int subclass; rejected at runtime
        with pytest.raises(ValueError):
            MemoryLimits(allowed_types=frozenset())


# ---------------------------------------------------------------------------
# Provenance and immutability
# ---------------------------------------------------------------------------


class TestProvenanceAndImmutability:
    def test_provenance_fields_preserved(self, store: InMemoryMemoryStore) -> None:
        memory = remember(
            store,
            "fact about the docs",
            mtype="knowledge",
            source="document",
            source_ref="notes.md",
            metadata={"section": "intro"},
        )
        assert memory.source is SourceCategory.DOCUMENT
        assert memory.source_ref == "notes.md"
        assert memory.metadata == {"section": "intro"}
        assert store.get(memory.memory_id) is not None

    def test_all_source_categories_accepted(self, store: InMemoryMemoryStore) -> None:
        for source in SourceCategory:
            memory = remember(store, f"content for {source.value}", source=source.value)
            assert memory.source is source

    def test_update_cannot_change_identity_fields(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "original", mtype="long_term", source="user_explicit")
        updated = store.update(memory.memory_id, content="changed content")
        assert updated.memory_id == memory.memory_id
        assert updated.memory_type is memory.memory_type
        assert updated.source is memory.source
        assert updated.created_at == memory.created_at
        assert updated.content == "changed content"
        assert updated.updated_at >= memory.updated_at

    def test_update_metadata_replaces_not_merges(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c", metadata={"a": 1})
        updated = store.update(memory.memory_id, metadata={"b": 2})
        assert updated.metadata == {"b": 2}

    def test_update_with_none_leaves_fields(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c", confidence=0.9, expires_at=FIXED + timedelta(hours=1))
        updated = store.update(memory.memory_id, content="new")
        assert updated.confidence == 0.9
        assert updated.expires_at == memory.expires_at


# ---------------------------------------------------------------------------
# Forget: soft default, hard opt-in, single-item, no recursion
# ---------------------------------------------------------------------------


class TestForget:
    def test_soft_forget_deactivates_and_hides(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "secret plan")
        result = store.forget(memory.memory_id)
        assert result.active is False
        assert store.get(memory.memory_id) is not None  # still auditable
        assert store.list() == []  # hidden from active listings
        assert store.recall("secret plan") == []

    def test_soft_forget_reversible_via_update(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c")
        store.forget(memory.memory_id)
        reactivated = store.update(memory.memory_id, active=True)
        assert reactivated.active is True
        assert len(store.list()) == 1

    def test_hard_forget_deletes(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c")
        result = store.forget(memory.memory_id, hard=True)
        assert result.memory_id == memory.memory_id
        assert store.get(memory.memory_id) is None
        assert store.list(active_only=False) == []

    def test_forget_affects_only_one_memory(self, store: InMemoryMemoryStore) -> None:
        first = remember(store, "one")
        second = remember(store, "two")
        store.forget(first.memory_id, hard=True)
        assert store.get(first.memory_id) is None
        assert store.get(second.memory_id) is not None

    def test_forget_expired_memory_still_forgettable(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c", mtype="short_term", expires_at=FIXED)
        assert memory.is_expired(FIXED + timedelta(seconds=1))
        result = store.forget(memory.memory_id, now=FIXED + timedelta(seconds=1))
        assert result.active is False


# ---------------------------------------------------------------------------
# Expiration and long-term protection
# ---------------------------------------------------------------------------


class TestExpiration:
    def test_short_term_gets_default_ttl(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c", mtype="short_term")
        assert memory.expires_at == FIXED + timedelta(seconds=3600)

    def test_working_gets_default_ttl(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "c", mtype="working")
        assert memory.expires_at == FIXED + timedelta(seconds=86400)

    def test_long_term_and_knowledge_never_implicit_expire(
        self, store: InMemoryMemoryStore
    ) -> None:
        for mtype in ("long_term", "knowledge"):
            memory = remember(store, f"c {mtype}", mtype=mtype)
            assert memory.expires_at is None

    def test_explicit_expiration_respected(self, store: InMemoryMemoryStore) -> None:
        explicit = FIXED + timedelta(minutes=5)
        memory = remember(store, "c", mtype="short_term", expires_at=explicit)
        assert memory.expires_at == explicit

    def test_expired_not_returned_as_active(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "ephemeral", mtype="short_term", expires_at=FIXED)
        later = FIXED + timedelta(seconds=1)
        assert store.list(now=later) == []
        assert store.recall("ephemeral", now=later) == []
        # but visible when explicitly asking for everything
        assert len(store.list(active_only=False, now=later)) == 1
        assert memory.memory_id == store.list(active_only=False)[0].memory_id

    def test_purge_removes_only_expired_short_term_working(
        self, store: InMemoryMemoryStore
    ) -> None:
        expired_short = remember(store, "s", mtype="short_term", expires_at=FIXED)
        expired_working = remember(store, "w", mtype="working", expires_at=FIXED)
        live_long = remember(store, "l", mtype="long_term", expires_at=FIXED)
        live_knowledge = remember(store, "k", mtype="knowledge", expires_at=FIXED)
        live_short = remember(store, "s2", mtype="short_term")
        now = FIXED + timedelta(seconds=1)
        removed = store.purge_expired(now=now)
        assert removed == 2
        assert store.get(expired_short.memory_id) is None
        assert store.get(expired_working.memory_id) is None
        # long_term/knowledge are PROTECTED: explicit expiration alone does
        # not make them purgeable (long-term protection).
        assert store.get(live_long.memory_id) is not None
        assert store.get(live_knowledge.memory_id) is not None
        assert store.get(live_short.memory_id) is not None

    def test_expired_excluded_from_recall_ranking(self, store: InMemoryMemoryStore) -> None:
        expired = remember(
            store,
            "apple apple apple",
            mtype="short_term",
            expires_at=FIXED,
        )
        live = remember(store, "apple", mtype="long_term")
        now = FIXED + timedelta(seconds=1)
        results = store.recall("apple", now=now)
        assert [m.memory_id for m in results] == [live.memory_id]
        assert expired.memory_id not in [m.memory_id for m in results]


# ---------------------------------------------------------------------------
# Item cap
# ---------------------------------------------------------------------------


class TestItemCap:
    def test_max_items_enforced(self) -> None:
        limits = MemoryLimits(max_items=3)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        for i in range(3):
            remember(store, f"content {i}")
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "one too many")
        assert exc.value.code == MEMORY_LIMIT_EXCEEDED

    def test_replacing_existing_id_does_not_count_again(self) -> None:
        limits = MemoryLimits(max_items=1)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        first = remember(store, "only")
        second = remember(store, "only")  # same instant => same id => replace
        assert first.memory_id == second.memory_id
        assert len(store.list()) == 1


# ---------------------------------------------------------------------------
# Recall: lexical ranking, filters, bounds, determinism
# ---------------------------------------------------------------------------


class TestRecall:
    def test_empty_query_returns_empty(self, store: InMemoryMemoryStore) -> None:
        remember(store, "anything")
        assert store.recall("") == []
        assert store.recall("   ") == []

    def test_no_match_returns_empty(self, store: InMemoryMemoryStore) -> None:
        remember(store, "the quick brown fox")
        assert store.recall("zebra quantum") == []

    def test_ranking_prefers_term_frequency(self, store: InMemoryMemoryStore) -> None:
        many = remember(store, "coffee coffee coffee")
        one = remember(store, "coffee and tea")
        results = store.recall("coffee")
        assert [m.memory_id for m in results] == [many.memory_id, one.memory_id]

    def test_rarer_term_wins_and_non_matches_excluded(self, store: InMemoryMemoryStore) -> None:
        common_a = remember(store, "meeting notes")
        common_b = remember(store, "meeting minutes")
        rare = remember(store, "zebra meeting")
        # Only memories containing the query term are returned, and the
        # rarer (higher-idf) match ranks first.
        results = store.recall("zebra")
        assert [m.memory_id for m in results] == [rare.memory_id]
        # A common term matches all three; ranking is then by tf/idf with
        # deterministic id tie-breaks.
        all_results = store.recall("meeting")
        assert {m.memory_id for m in all_results} == {
            common_a.memory_id,
            common_b.memory_id,
            rare.memory_id,
        }
        assert len(all_results) == 3

    def test_ties_broken_by_memory_id(self) -> None:
        store = InMemoryMemoryStore(MemoryLimits(), clock=lambda: FIXED)
        a = store.remember(
            memory_type="long_term",
            content="alpha beta",
            source="user_explicit",
            now=FIXED,
        )
        b = store.remember(
            memory_type="long_term",
            content="beta alpha",
            source="agent",
            now=FIXED,
        )
        results = store.recall("alpha beta")
        assert {m.memory_id for m in results} == {a.memory_id, b.memory_id}
        assert [m.memory_id for m in results] == sorted(
            [a.memory_id, b.memory_id]
        )  # equal scores => id order

    def test_type_filter(self, store: InMemoryMemoryStore) -> None:
        remember(store, "report findings", mtype="working")
        remember(store, "report deadline", mtype="long_term")
        results = store.recall("report", memory_type="working")
        assert len(results) == 1
        assert results[0].memory_type is MemoryType.WORKING

    def test_limit_bounded_by_configured_max(self) -> None:
        limits = MemoryLimits(max_recall_results=3)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        for i in range(10):
            remember(store, f"shared token {i}")
        results = store.recall("shared token", limit=100)  # clamped to 3
        assert len(results) == 3

    def test_inactive_excluded_from_recall(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "hidden thought")
        store.forget(memory.memory_id)
        assert store.recall("hidden thought") == []

    def test_recall_is_deterministic_across_calls(self, store: InMemoryMemoryStore) -> None:
        for i in range(5):
            remember(store, f"topic {i} shared")
        first = [m.memory_id for m in store.recall("topic shared")]
        second = [m.memory_id for m in store.recall("topic shared")]
        assert first == second


# ---------------------------------------------------------------------------
# List: ordering, filters, bounds
# ---------------------------------------------------------------------------


class TestList:
    def test_ordering_is_created_at_then_id(self, store: InMemoryMemoryStore) -> None:
        second = remember(store, "second", mtype="working")
        first = store.remember(
            memory_type="long_term",
            content="first",
            source="user_explicit",
            now=FIXED,
        )
        # 'second' was created at FIXED via the store clock; ensure ordering
        assert [m.memory_id for m in store.list()] == [
            min(first.memory_id, second.memory_id),
            max(first.memory_id, second.memory_id),
        ]

    def test_different_timestamps_order_chronologically(self, store: InMemoryMemoryStore) -> None:
        early = store.remember(
            memory_type="long_term",
            content="early",
            source="user_explicit",
            now=FIXED,
        )
        late = store.remember(
            memory_type="long_term",
            content="late",
            source="user_explicit",
            now=FIXED + timedelta(seconds=1),
        )
        assert [m.memory_id for m in store.list()] == [early.memory_id, late.memory_id]

    def test_type_filter(self, store: InMemoryMemoryStore) -> None:
        remember(store, "a", mtype="short_term")
        remember(store, "b", mtype="long_term")
        results = store.list(memory_type="short_term")
        assert len(results) == 1
        assert results[0].memory_type is MemoryType.SHORT_TERM

    def test_active_only_default_true(self, store: InMemoryMemoryStore) -> None:
        memory = remember(store, "a")
        store.forget(memory.memory_id)
        assert store.list() == []
        assert len(store.list(active_only=False)) == 1

    def test_limit_clamped_to_recall_cap(self) -> None:
        limits = MemoryLimits(max_recall_results=2)
        store = InMemoryMemoryStore(limits, clock=lambda: FIXED)
        for i in range(5):
            remember(store, f"m{i}")
        assert len(store.list(limit=50)) == 2
        assert len(store.list(limit=None)) == 2


# ---------------------------------------------------------------------------
# Retrieval adapter
# ---------------------------------------------------------------------------


class TestRetrieverAdapter:
    def test_adapter_delegates_to_store(self) -> None:
        store = InMemoryMemoryStore(MemoryLimits(), clock=lambda: FIXED)
        remember(store, "adapter target")
        retriever = LexicalMemoryRetriever(store)
        assert isinstance(retriever, object)
        results = retriever.recall("adapter")
        assert len(results) == 1
        assert retriever.store is store

    def test_store_satisfies_protocol(self) -> None:
        from agent_core.memory import MemoryStore

        store = InMemoryMemoryStore()
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# Secret heuristic
# ---------------------------------------------------------------------------


class TestSecretGuards:
    @pytest.mark.parametrize(
        "content",
        [
            "OPENAI_API_KEY=sk-abcdef1234567890",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
            "token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            "AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEp...\n-----END RSA PRIVATE KEY-----",
            "https://user:password@host.example/db",
        ],
    )
    def test_secret_like_content_detected(self, content: str) -> None:
        assert contains_secret_like_content(content)

    @pytest.mark.parametrize(
        "content",
        [
            "The user prefers dark mode in the editor.",
            "Meeting notes: ship the release on Friday.",
            "API documentation is in docs/api.md",
            "My favorite number is 42 and my password policy is '12 characters'",
        ],
    )
    def test_normal_content_not_flagged(self, content: str) -> None:
        assert not contains_secret_like_content(content)

    def test_secret_rejected_at_store(self, store: InMemoryMemoryStore) -> None:
        with pytest.raises(MemoryStoreError) as exc:
            remember(store, "password: hunter2secretvalue123!")
        assert exc.value.code == SECRET_LIKE_CONTENT

    def test_heuristic_does_not_block_normal_words(self, store: InMemoryMemoryStore) -> None:
        # 'secret' alone is fine; the heuristic is conservative, not a blocklist.
        memory = remember(store, "this is not a secret, just notes")
        assert memory.content == "this is not a secret, just notes"
