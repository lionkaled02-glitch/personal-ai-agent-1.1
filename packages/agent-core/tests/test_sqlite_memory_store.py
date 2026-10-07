from __future__ import annotations

from datetime import UTC, datetime

from agent_core.memory.limits import MemoryLimits
from agent_core.memory.models import MemoryType, SourceCategory
from agent_core.memory.sqlite_store import SQLiteMemoryStore


def test_sqlite_memory_round_trip(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = SQLiteMemoryStore(path, MemoryLimits(), clock=lambda: now)
    item = first.remember(
        memory_type=MemoryType.LONG_TERM,
        content="The user prefers concise answers.",
        source=SourceCategory.USER_EXPLICIT,
        source_ref="test",
        metadata={"kind": "preference"},
        confidence=1.0,
        now=now,
    )
    second = SQLiteMemoryStore(path, MemoryLimits(), clock=lambda: now)
    restored = second.get(item.memory_id)
    assert restored is not None
    assert restored.content == item.content
    assert restored.metadata == item.metadata
    assert second.recall("concise answers")
