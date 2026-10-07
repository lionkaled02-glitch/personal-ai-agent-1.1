"""Durable SQLite-backed MemoryStore using the same policy engine as the in-memory store."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from ..events import Clock
from .limits import MemoryLimits
from .models import Memory
from .store import InMemoryMemoryStore


class SQLiteMemoryStore:
    """Durable memory implementation with atomic SQLite persistence.

    Policy validation remains delegated to ``InMemoryMemoryStore``; SQLite is
    storage only. Memory content is data, never executable instructions.
    """

    def __init__(
        self, path: Path, limits: MemoryLimits | None = None, clock: Clock | None = None
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._memory = InMemoryMemoryStore(limits=limits, clock=clock)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=5000")
            db.execute(
                "CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
        self._load()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    @property
    def limits(self) -> MemoryLimits:
        return self._memory.limits

    def _load(self) -> None:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT payload FROM memories ORDER BY memory_id").fetchall()
        for (payload,) in rows:
            item = Memory.model_validate(json.loads(payload))
            restored = self._memory.remember(
                memory_type=item.memory_type,
                content=item.content,
                source=item.source,
                source_ref=item.source_ref,
                metadata=item.metadata,
                confidence=item.confidence,
                expires_at=item.expires_at,
                now=item.created_at,
            )
            restored.updated_at = item.updated_at
            restored.active = item.active

    def _persist(self, item: Memory) -> None:
        payload = json.dumps(
            item.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO memories(memory_id,payload) VALUES(?,?) ON CONFLICT(memory_id) DO UPDATE SET payload=excluded.payload",
                (item.memory_id, payload),
            )

    def remember(self, **kwargs) -> Memory:
        item = self._memory.remember(**kwargs)
        self._persist(item)
        return item

    def get(self, memory_id: str) -> Memory | None:
        return self._memory.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> Memory:
        item = self._memory.update(memory_id, **kwargs)
        self._persist(item)
        return item

    def forget(self, memory_id: str, *, hard: bool = False, now: datetime | None = None) -> Memory:
        item = self._memory.forget(memory_id, hard=hard, now=now)
        with self._lock, self._connect() as db:
            if hard:
                db.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
            else:
                self._persist(item)
        return item

    def list(self, **kwargs) -> list[Memory]:
        return self._memory.list(**kwargs)

    def recall(self, query: str, **kwargs) -> list[Memory]:
        return self._memory.recall(query, **kwargs)

    def purge_expired(self, now: datetime | None = None) -> int:
        count = self._memory.purge_expired(now)
        if count:
            active_ids = {m.memory_id for m in self._memory.list(active_only=False, limit=None)}
            with self._lock, self._connect() as db:
                rows = db.execute("SELECT memory_id FROM memories").fetchall()
                for (memory_id,) in rows:
                    if memory_id not in active_ids:
                        db.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        return count
