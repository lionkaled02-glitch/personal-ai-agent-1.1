from datetime import UTC, datetime

from agent_core import MemoryType, SourceCategory, SQLiteMemoryStore


def test_sqlite_memory_survives_restart(tmp_path):
    path = tmp_path / "memory.sqlite3"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = SQLiteMemoryStore(path)
    item = first.remember(
        memory_type=MemoryType.LONG_TERM,
        content="prefers concise answers",
        source=SourceCategory.USER_EXPLICIT,
        now=now,
    )
    second = SQLiteMemoryStore(path)
    loaded = second.get(item.memory_id)
    assert loaded is not None
    assert loaded.content == item.content
    assert loaded.source == SourceCategory.USER_EXPLICIT
