"""Memory tools: behavior, validation, and permission tests (Phase 5).

Covers all five tools (remember, recall, update_memory, forget,
list_memories): permission levels, input validation, structured error
codes, output contracts (metadata-only confirmations; stable public fields
for reads), immutable identity fields, soft vs hard forget, and the full
agent-level permission flow (approval required for MEDIUM/HIGH, fail-safe
denial without an approval channel, no mutation after denial, policy deny
wins over approval).

Deterministic fixed clock; no network, no external APIs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from agent_core import (
    Agent,
    ApprovalCallback,
    EventBus,
    EventType,
    InMemoryMemoryStore,
    MemoryLimits,
    MockModelProvider,
    ModelPlanner,
    PermissionLevel,
    PermissionManager,
    PermissionPolicy,
    StepStatus,
    TaskState,
    Tool,
    ToolRegistry,
    ToolResult,
    register_memory_tools,
)
from conftest import FIXED_NOW

FIXED: datetime = FIXED_NOW
LIMITS = MemoryLimits()


@pytest.fixture
def store() -> InMemoryMemoryStore:
    return InMemoryMemoryStore(LIMITS, clock=lambda: FIXED)


@pytest.fixture
def registry(store: InMemoryMemoryStore) -> ToolRegistry:
    reg = ToolRegistry()
    register_memory_tools(reg, store, LIMITS, clock=lambda: FIXED)
    return reg


@pytest.fixture
def tools(store: InMemoryMemoryStore, registry: ToolRegistry) -> dict[str, Tool]:
    by_name: dict[str, Tool] = {}
    for spec in registry.list_tools():
        tool = registry.get(spec.name)
        assert tool is not None
        by_name[spec.name] = tool
    return by_name


def call(tools: dict[str, Tool], name: str, payload: dict[str, Any]) -> ToolResult:
    return tools[name].run(payload)


def out(result: ToolResult) -> dict[str, Any]:
    assert result.ok, result.error
    assert isinstance(result.output, dict)
    return result.output


def remember_ok(
    tools: dict[str, Tool],
    content: str,
    mtype: str = "long_term",
    source: str = "user_explicit",
) -> dict[str, Any]:
    return out(
        call(tools, "remember", {"memory_type": mtype, "content": content, "source": source})
    )


# ---------------------------------------------------------------------------
# Specs and permission levels
# ---------------------------------------------------------------------------


class TestToolSpecs:
    def test_all_five_registered(self, registry: ToolRegistry) -> None:
        assert set(registry.names()) == {
            "remember",
            "recall",
            "update_memory",
            "forget",
            "list_memories",
        }

    def test_permission_levels_contract(self, registry: ToolRegistry) -> None:
        by_name = {s.name: s for s in registry.list_tools()}
        assert by_name["recall"].permission_level is PermissionLevel.LOW
        assert by_name["list_memories"].permission_level is PermissionLevel.LOW
        assert by_name["remember"].permission_level is PermissionLevel.MEDIUM
        assert by_name["update_memory"].permission_level is PermissionLevel.MEDIUM
        assert by_name["forget"].permission_level is PermissionLevel.HIGH

    def test_no_high_tool_without_explicit_opt_in(self, registry: ToolRegistry) -> None:
        by_name = {s.name: s for s in registry.list_tools()}
        # forget is HIGH by design and MUST require approval; it is the only
        # HIGH-permission memory tool.
        assert by_name["forget"].permission_level is PermissionLevel.HIGH
        for name in ("remember", "update_memory", "recall", "list_memories"):
            assert by_name[name].permission_level is not PermissionLevel.HIGH


# ---------------------------------------------------------------------------
# remember (MEDIUM)
# ---------------------------------------------------------------------------


class TestRemember:
    def test_creates_memory_and_returns_metadata_only(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        result = remember_ok(tools, "the user prefers tabs over spaces")
        assert "content" not in result  # confirmation is metadata-only
        assert result["memory_type"] == "long_term"
        assert result["source"] == "user_explicit"
        assert result["active"] is True
        assert result["created_at"] == FIXED.isoformat()
        stored = store.get(result["memory_id"])
        assert stored is not None
        assert stored.content == "the user prefers tabs over spaces"

    def test_stored_content_matches_input(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        result = remember_ok(tools, "project deadline is Friday", mtype="working")
        stored = store.get(result["memory_id"])
        assert stored is not None
        assert stored.content == "project deadline is Friday"
        assert stored.memory_type.value == "working"

    def test_source_ref_and_metadata_and_confidence_stored(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        result = out(
            call(
                tools,
                "remember",
                {
                    "memory_type": "knowledge",
                    "content": "API rate limit is 100 rpm",
                    "source": "document",
                    "source_ref": "docs/api.md",
                    "metadata": {"section": "limits"},
                    "confidence": 0.9,
                },
            )
        )
        stored = store.get(result["memory_id"])
        assert stored is not None
        assert stored.source_ref == "docs/api.md"
        assert stored.metadata == {"section": "limits"}
        assert stored.confidence == 0.9

    def test_explicit_expires_at_parsed(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        expires = (FIXED + timedelta(minutes=5)).isoformat()
        result = out(
            call(
                tools,
                "remember",
                {
                    "memory_type": "short_term",
                    "content": "transient",
                    "source": "task",
                    "expires_at": expires,
                },
            )
        )
        stored = store.get(result["memory_id"])
        assert stored is not None
        assert stored.expires_at == FIXED + timedelta(minutes=5)

    def test_naive_expires_at_assumed_utc(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        naive = (FIXED + timedelta(minutes=5)).replace(tzinfo=None)
        result = out(
            call(
                tools,
                "remember",
                {
                    "memory_type": "short_term",
                    "content": "transient",
                    "source": "task",
                    "expires_at": naive.isoformat(),
                },
            )
        )
        stored = store.get(result["memory_id"])
        assert stored is not None
        assert stored.expires_at == FIXED + timedelta(minutes=5)
        assert stored.expires_at.tzinfo is not None

    def test_invalid_expires_at_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(
            tools,
            "remember",
            {
                "memory_type": "short_term",
                "content": "transient",
                "source": "task",
                "expires_at": "not-a-date",
            },
        )
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_idempotent_same_instant(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        first = remember_ok(tools, "same call")
        second = remember_ok(tools, "same call")
        assert first["memory_id"] == second["memory_id"]
        assert len(store.list()) == 1

    @pytest.mark.parametrize(
        "payload",
        [
            {"memory_type": "nope", "content": "c", "source": "user_explicit"},
            {"memory_type": "long_term", "content": "c", "source": "nope"},
            {"memory_type": "long_term", "content": "", "source": "user_explicit"},
            {"memory_type": "long_term", "content": "   ", "source": "user_explicit"},
            {"memory_type": "long_term", "source": "user_explicit"},
            {"content": "c", "source": "user_explicit"},
        ],
    )
    def test_invalid_input_structured_errors(
        self, tools: dict[str, Tool], payload: dict[str, Any]
    ) -> None:
        result = call(tools, "remember", payload)
        assert not result.ok
        assert result.error_code in ("memory_invalid_input",)

    def test_oversized_content_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(
            tools,
            "remember",
            {
                "memory_type": "long_term",
                "content": "x" * (LIMITS.max_content_chars + 1),
                "source": "user_explicit",
            },
        )
        assert not result.ok
        assert result.error_code == "memory_limit_exceeded"

    def test_secret_like_content_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(
            tools,
            "remember",
            {
                "memory_type": "long_term",
                "content": "OPENAI_API_KEY=sk-abcdef1234567890abcdef",
                "source": "user_explicit",
            },
        )
        assert not result.ok
        assert result.error_code == "secret_like_content"

    @pytest.mark.parametrize("confidence", [True, -0.5, 1.5, "high", ["0.5"]])
    def test_bad_confidence_rejected(self, tools: dict[str, Tool], confidence: Any) -> None:
        result = call(
            tools,
            "remember",
            {
                "memory_type": "long_term",
                "content": "c",
                "source": "user_explicit",
                "confidence": confidence,
            },
        )
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_bad_metadata_type_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(
            tools,
            "remember",
            {
                "memory_type": "long_term",
                "content": "c",
                "source": "user_explicit",
                "metadata": "not-an-object",
            },
        )
        assert not result.ok
        assert result.error_code == "memory_invalid_input"


# ---------------------------------------------------------------------------
# recall (LOW)
# ---------------------------------------------------------------------------


class TestRecall:
    def test_returns_results_with_provenance_and_ids(self, tools: dict[str, Tool]) -> None:
        remember_ok(tools, "the deploy pipeline uses GitHub Actions")
        result = out(call(tools, "recall", {"query": "deploy pipeline"}))
        assert result["count"] == 1
        entry = result["results"][0]
        assert entry["memory_id"].startswith("mem-")
        assert entry["content"] == "the deploy pipeline uses GitHub Actions"
        assert entry["source"] == "user_explicit"
        assert entry["memory_type"] == "long_term"
        assert entry["active"] is True

    def test_type_filter(self, tools: dict[str, Tool]) -> None:
        remember_ok(tools, "task note: remember the flag", mtype="working")
        remember_ok(tools, "user note: remember the flag", mtype="long_term")
        result = out(
            call(tools, "recall", {"query": "remember the flag", "memory_type": "working"})
        )
        assert result["count"] == 1
        assert result["results"][0]["memory_type"] == "working"

    def test_no_match_returns_zero_count(self, tools: dict[str, Tool]) -> None:
        remember_ok(tools, "something entirely different")
        result = out(call(tools, "recall", {"query": "quantum zebra"}))
        assert result["count"] == 0
        assert result["results"] == []

    def test_empty_query_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "recall", {"query": "   "})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_oversized_query_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "recall", {"query": "x" * (LIMITS.max_content_chars + 1)})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_limit_clamped_to_configured_max(self, tools: dict[str, Tool]) -> None:
        for i in range(15):
            remember_ok(tools, f"shared topic {i}")
        result = out(call(tools, "recall", {"query": "shared topic", "limit": 100}))
        assert result["count"] <= LIMITS.max_recall_results

    @pytest.mark.parametrize("limit", [0, -1, "five", 1.5])
    def test_bad_limit_rejected(self, tools: dict[str, Tool], limit: Any) -> None:
        result = call(tools, "recall", {"query": "x", "limit": limit})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_expired_never_returned(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        remember_ok(
            tools,
            "ephemeral thought about cookies",
            mtype="short_term",
        )
        # Force expiration: the store clock is FIXED; expire far in the past
        # by updating expires_at directly via the store (trusted pathway).
        memory = store.list()[0]
        store.update(memory.memory_id, expires_at=FIXED - timedelta(seconds=1))
        result = out(call(tools, "recall", {"query": "ephemeral cookies"}))
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# update_memory (MEDIUM)
# ---------------------------------------------------------------------------


class TestUpdateMemory:
    def test_updates_content_and_returns_metadata(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        first = remember_ok(tools, "original statement")
        result = out(
            call(
                tools,
                "update_memory",
                {"memory_id": first["memory_id"], "content": "revised statement"},
            )
        )
        assert "content" not in result
        assert result["active"] is True
        stored = store.get(first["memory_id"])
        assert stored is not None
        assert stored.content == "revised statement"

    def test_updates_metadata_confidence_and_expiration(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        first = remember_ok(tools, "c")
        new_expires = (FIXED + timedelta(days=1)).isoformat()
        result = out(
            call(
                tools,
                "update_memory",
                {
                    "memory_id": first["memory_id"],
                    "metadata": {"k": "v"},
                    "confidence": 0.4,
                    "expires_at": new_expires,
                },
            )
        )
        assert result["expires_at"] == new_expires
        stored = store.get(first["memory_id"])
        assert stored is not None
        assert stored.metadata == {"k": "v"}
        assert stored.confidence == 0.4
        assert stored.expires_at == FIXED + timedelta(days=1)

    @pytest.mark.parametrize("field", ["memory_type", "source", "source_ref", "created_at"])
    def test_identity_fields_rejected(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore, field: str
    ) -> None:
        first = remember_ok(tools, "immutability test")
        value: Any = {
            "memory_type": "working",
            "source": "agent",
            "source_ref": "x",
            "created_at": FIXED.isoformat(),
        }[field]
        result = call(
            tools,
            "update_memory",
            {"memory_id": first["memory_id"], field: value},
        )
        assert not result.ok
        assert result.error_code == "memory_field_immutable"
        stored = store.get(first["memory_id"])
        assert stored is not None
        assert stored.content == "immutability test"  # unchanged

    def test_missing_id_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "update_memory", {"content": "x"})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_unknown_id_not_found(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "update_memory", {"memory_id": "mem-0000000000000000", "content": "x"})
        assert not result.ok
        assert result.error_code == "memory_not_found"

    def test_secret_content_rejected_on_update(self, tools: dict[str, Tool]) -> None:
        first = remember_ok(tools, "safe content")
        result = call(
            tools,
            "update_memory",
            {
                "memory_id": first["memory_id"],
                "content": "password: supersecretpass123",
            },
        )
        assert not result.ok
        assert result.error_code == "secret_like_content"

    def test_bad_confidence_rejected(self, tools: dict[str, Tool]) -> None:
        first = remember_ok(tools, "c")
        result = call(
            tools,
            "update_memory",
            {"memory_id": first["memory_id"], "confidence": 2.0},
        )
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_reactivate_after_soft_forget(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        first = remember_ok(tools, "forgotten thing")
        call(tools, "forget", {"memory_id": first["memory_id"]})
        result = out(
            call(
                tools,
                "update_memory",
                {"memory_id": first["memory_id"], "active": True},
            )
        )
        assert result["active"] is True
        assert store.get(first["memory_id"]) is not None


# ---------------------------------------------------------------------------
# forget (HIGH)
# ---------------------------------------------------------------------------


class TestForget:
    def test_soft_forget_default(self, tools: dict[str, Tool], store: InMemoryMemoryStore) -> None:
        first = remember_ok(tools, "deactivate me")
        result = out(call(tools, "forget", {"memory_id": first["memory_id"]}))
        assert result["hard"] is False
        assert result["deleted"] is False
        assert result["active"] is False
        stored = store.get(first["memory_id"])
        assert stored is not None  # auditable
        assert stored.active is False

    def test_hard_forget_deletes(self, tools: dict[str, Tool], store: InMemoryMemoryStore) -> None:
        first = remember_ok(tools, "delete me")
        result = out(call(tools, "forget", {"memory_id": first["memory_id"], "hard": True}))
        assert result["hard"] is True
        assert result["deleted"] is True
        assert store.get(first["memory_id"]) is None

    def test_forget_does_not_touch_others(
        self, tools: dict[str, Tool], store: InMemoryMemoryStore
    ) -> None:
        first = remember_ok(tools, "one")
        second = remember_ok(tools, "two")
        call(tools, "forget", {"memory_id": first["memory_id"], "hard": True})
        assert store.get(first["memory_id"]) is None
        assert store.get(second["memory_id"]) is not None

    def test_unknown_id_not_found(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "forget", {"memory_id": "mem-1111111111111111"})
        assert not result.ok
        assert result.error_code == "memory_not_found"

    @pytest.mark.parametrize("hard", ["yes", 1, None, ["true"]])
    def test_bad_hard_rejected(self, tools: dict[str, Tool], hard: Any) -> None:
        first = remember_ok(tools, "c")
        result = call(tools, "forget", {"memory_id": first["memory_id"], "hard": hard})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_missing_id_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "forget", {"hard": True})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"


# ---------------------------------------------------------------------------
# list_memories (LOW)
# ---------------------------------------------------------------------------


class TestListMemories:
    def test_lists_public_fields_only(self, tools: dict[str, Tool]) -> None:
        remember_ok(tools, "first memory")
        result = out(call(tools, "list_memories", {}))
        assert result["count"] == 1
        entry = result["results"][0]
        assert set(entry.keys()) <= {
            "memory_id",
            "memory_type",
            "content",
            "source",
            "source_ref",
            "confidence",
            "created_at",
            "updated_at",
            "expires_at",
            "active",
        }

    def test_type_filter(self, tools: dict[str, Tool]) -> None:
        remember_ok(tools, "a", mtype="short_term")
        remember_ok(tools, "b", mtype="long_term")
        result = out(call(tools, "list_memories", {"memory_type": "short_term"}))
        assert result["count"] == 1
        assert result["results"][0]["memory_type"] == "short_term"

    def test_active_only_default_excludes_forgotten(self, tools: dict[str, Tool]) -> None:
        first = remember_ok(tools, "gone")
        second = remember_ok(tools, "stays")
        call(tools, "forget", {"memory_id": first["memory_id"]})
        result = out(call(tools, "list_memories", {}))
        assert result["count"] == 1
        assert result["results"][0]["memory_id"] == second["memory_id"]

    def test_active_only_false_includes_forgotten(self, tools: dict[str, Tool]) -> None:
        first = remember_ok(tools, "gone")
        call(tools, "forget", {"memory_id": first["memory_id"]})
        result = out(call(tools, "list_memories", {"active_only": False}))
        assert result["count"] == 1
        assert result["results"][0]["active"] is False

    def test_limit_bounded(self, tools: dict[str, Tool]) -> None:
        for i in range(15):
            remember_ok(tools, f"item {i}")
        result = out(call(tools, "list_memories", {"limit": 100}))
        assert result["count"] <= LIMITS.max_recall_results

    @pytest.mark.parametrize("limit", [0, -3, "many"])
    def test_bad_limit_rejected(self, tools: dict[str, Tool], limit: Any) -> None:
        result = call(tools, "list_memories", {"limit": limit})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"

    def test_bad_type_rejected(self, tools: dict[str, Tool]) -> None:
        result = call(tools, "list_memories", {"memory_type": 42})
        assert not result.ok
        assert result.error_code == "memory_invalid_input"


# ---------------------------------------------------------------------------
# Agent-level permission flow (MEDIUM/HIGH require approval)
# ---------------------------------------------------------------------------


def _plan_json(tool_name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "tool_name": tool_name,
                    "description": f"run {tool_name}",
                    "input": tool_input,
                }
            ]
        }
    )


Clock = Callable[[], datetime]


def build_memory_agent(
    store: InMemoryMemoryStore,
    plan_json: str,
    *,
    approval: ApprovalCallback | None = None,
    policy: PermissionPolicy | None = None,
) -> Agent:
    registry: ToolRegistry = ToolRegistry()
    register_memory_tools(registry, store, LIMITS, clock=lambda: FIXED)
    return Agent(
        planner=ModelPlanner(MockModelProvider(responses=[plan_json])),
        registry=registry,
        permissions=PermissionManager(policy=policy, approval=approval),
        events=EventBus(clock=lambda: FIXED),
        clock=lambda: FIXED,
        memory_store=store,
    )


class TestPermissionBehavior:
    def test_low_recall_runs_without_approval(self, store: InMemoryMemoryStore) -> None:
        store.remember(
            memory_type="long_term",
            content="the answer is portable",
            source="user_explicit",
            now=FIXED,
        )
        agent = build_memory_agent(store, _plan_json("recall", {"query": "portable"}))
        task = agent.run("recall it")
        assert task.state is TaskState.COMPLETED
        step = task.steps[0]
        assert step.status is StepStatus.COMPLETED
        assert agent.events.events_of_type(EventType.APPROVAL_REQUIRED) == []

    def test_medium_remember_with_approval_completes(self, store: InMemoryMemoryStore) -> None:
        agent = build_memory_agent(
            store,
            _plan_json(
                "remember",
                {"memory_type": "long_term", "content": "approved fact", "source": "user_explicit"},
            ),
            approval=lambda _r: True,
        )
        task = agent.run("remember it")
        assert task.state is TaskState.COMPLETED
        assert len(store.list()) == 1
        approval = agent.events.events_of_type(EventType.APPROVAL_REQUIRED)[0]
        assert approval.data["tool_name"] == "remember"
        assert approval.data["permission_level"] == "MEDIUM"

    def test_medium_remember_denied_performs_no_mutation(self, store: InMemoryMemoryStore) -> None:
        agent = build_memory_agent(
            store,
            _plan_json(
                "remember",
                {"memory_type": "long_term", "content": "denied fact", "source": "user_explicit"},
            ),
            approval=lambda _r: False,
        )
        task = agent.run("remember it")
        assert task.state is TaskState.CANCELLED
        assert store.list(active_only=False) == []  # NO mutation after denial
        denied = agent.events.events_of_type(EventType.TOOL_DENIED)[0]
        assert denied.data["reason"] == "approval_denied"

    def test_medium_remember_without_channel_fail_safe(self, store: InMemoryMemoryStore) -> None:
        agent = build_memory_agent(
            store,
            _plan_json(
                "remember",
                {"memory_type": "long_term", "content": "silent fact", "source": "user_explicit"},
            ),
        )
        task = agent.run("remember it")
        assert task.state is TaskState.CANCELLED
        assert store.list(active_only=False) == []

    def test_high_forget_requires_approval(self, store: InMemoryMemoryStore) -> None:
        store.remember(
            memory_type="long_term",
            content="keep until approved",
            source="user_explicit",
            now=FIXED,
        )
        memory = store.list()[0]
        agent = build_memory_agent(
            store,
            _plan_json("forget", {"memory_id": memory.memory_id}),
            approval=lambda _r: True,
        )
        task = agent.run("forget it")
        assert task.state is TaskState.COMPLETED
        approval = agent.events.events_of_type(EventType.APPROVAL_REQUIRED)[0]
        assert approval.data["tool_name"] == "forget"
        assert approval.data["permission_level"] == "HIGH"
        deactivated = store.get(memory.memory_id)
        assert deactivated is not None
        assert deactivated.active is False

    def test_high_forget_denied_keeps_memory(self, store: InMemoryMemoryStore) -> None:
        store.remember(
            memory_type="long_term",
            content="protected",
            source="user_explicit",
            now=FIXED,
        )
        memory = store.list()[0]
        agent = build_memory_agent(
            store,
            _plan_json("forget", {"memory_id": memory.memory_id}),
            approval=lambda _r: False,
        )
        task = agent.run("forget it")
        assert task.state is TaskState.CANCELLED
        kept = store.get(memory.memory_id)
        assert kept is not None
        assert kept.active is True  # unchanged

    def test_policy_deny_wins_over_approval(self, store: InMemoryMemoryStore) -> None:
        policy = PermissionPolicy(denied_tools=frozenset({"remember"}))
        agent = build_memory_agent(
            store,
            _plan_json(
                "remember",
                {"memory_type": "long_term", "content": "blocked", "source": "user_explicit"},
            ),
            policy=policy,
            approval=lambda _r: True,
        )
        task = agent.run("remember it")
        assert task.state is TaskState.CANCELLED
        assert store.list(active_only=False) == []


# ---------------------------------------------------------------------------
# Event privacy: no full memory content in event payloads
# ---------------------------------------------------------------------------


class TestEventPrivacy:
    def test_completed_event_for_remember_has_no_content(self, store: InMemoryMemoryStore) -> None:
        long_content = "A very long fact about the project " * 50  # > 200 chars
        agent = build_memory_agent(
            store,
            _plan_json(
                "remember",
                {
                    "memory_type": "long_term",
                    "content": long_content,
                    "source": "user_explicit",
                },
            ),
            approval=lambda _r: True,
        )
        agent.run("remember it")
        completed = agent.events.events_of_type(EventType.TOOL_COMPLETED)
        assert len(completed) == 1
        payload = json.dumps(completed[0].data)
        assert long_content not in payload  # full content never logged
        assert "content" not in completed[0].data.get("output", {})

    def test_event_strings_are_bounded(self, store: InMemoryMemoryStore) -> None:
        long_content = "B" * 4000  # at the max content limit
        agent = build_memory_agent(
            store,
            _plan_json(
                "recall",
                {"query": "B"},
            ),
            approval=lambda _r: True,
        )
        # Seed the memory directly (trusted internal pathway, no approval).
        store.remember(
            memory_type="long_term",
            content=long_content,
            source="user_explicit",
            now=FIXED,
        )
        agent.run("recall it")
        for event in agent.events.history:
            for value in _iter_strings(event.data):
                assert len(value) <= 200

    def test_full_run_does_not_auto_persist_conversation(self, store: InMemoryMemoryStore) -> None:
        """Running the agent must never create memories by itself."""
        plan = json.dumps(
            {"steps": [{"tool_name": "recall", "description": "r", "input": {"query": "nothing"}}]}
        )
        agent = build_memory_agent(store, plan)
        task = agent.run("please recall nothing in particular, hello world")
        assert task.state is TaskState.COMPLETED
        assert store.list(active_only=False) == []


def _iter_strings(data: Any) -> list[str]:
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return out
