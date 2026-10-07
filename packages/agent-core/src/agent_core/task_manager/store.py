"""Durable SQLite task/event persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..tasks import Task


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                "payload TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, "
                "task_id TEXT, payload TEXT NOT NULL, timestamp TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS idempotency ("
                "key TEXT PRIMARY KEY, task_id TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    def save(self, task: Task) -> None:
        payload = json.dumps(
            task.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO tasks(id,state,payload,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "state=excluded.state,payload=excluded.payload,"
                "updated_at=excluded.updated_at",
                (task.id, task.state.value, payload, task.updated_at.isoformat()),
            )

    def bind_idempotency(self, key: str, task_id: str, created_at: str) -> bool:
        """Atomically bind an idempotency key; return False if already bound."""
        with self._lock, self._connect() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO idempotency(key,task_id,created_at) VALUES(?,?,?)",
                (key, task_id, created_at),
            )
            return cur.rowcount == 1

    def task_for_idempotency(self, key: str) -> str | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT task_id FROM idempotency WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def get(self, task_id: str) -> Task | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
        return Task.model_validate(json.loads(row[0])) if row else None

    def list(self, limit: int = 100) -> list[Task]:
        limit = max(1, min(limit, 1000))
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Task.model_validate(json.loads(row[0])) for row in rows]

    def recover_incomplete(self) -> int:
        """Safely pause tasks left non-terminal by a process restart.

        A RUNNING step is never auto-resumed because its side effect may have
        happened immediately before the crash. Such tasks remain paused and
        require explicit inspection rather than risking duplicate effects.
        """
        changed = 0
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM tasks WHERE state IN (?, ?, ?)",
                ("RUNNING", "PLANNING", "WAITING_FOR_USER"),
            ).fetchall()
            for (payload,) in rows:
                task = Task.model_validate(json.loads(payload))
                if task.is_terminal():
                    continue
                task.state = task.state.__class__.PAUSED
                task.updated_at = task.updated_at
                task.error = "task paused after service restart; explicit resume required"
                db.execute(
                    "UPDATE tasks SET state=?,payload=?,updated_at=? WHERE id=?",
                    (
                        task.state.value,
                        json.dumps(
                            task.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
                        ),
                        task.updated_at.isoformat(),
                        task.id,
                    ),
                )
                changed += 1
        return changed

    def append_event(self, event: dict[str, Any]) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO events(id,task_id,payload,timestamp) VALUES(?,?,?,?)",
                (
                    event["event_id"],
                    event.get("task_id"),
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                    event["timestamp"],
                ),
            )

    def events(
        self, task_id: str, limit: int = 200, after_event_id: str | None = None
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._lock, self._connect() as db:
            if after_event_id:
                row = db.execute(
                    "SELECT timestamp, id FROM events WHERE id=? AND task_id=?",
                    (after_event_id, task_id),
                ).fetchone()
                if row:
                    rows = db.execute(
                        "SELECT payload FROM events WHERE task_id=? AND "
                        "(timestamp > ? OR (timestamp = ? AND id > ?)) "
                        "ORDER BY timestamp ASC, id ASC LIMIT ?",
                        (task_id, row[0], row[0], row[1], limit),
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT payload FROM events WHERE task_id=? "
                        "ORDER BY timestamp ASC, id ASC LIMIT ?",
                        (task_id, limit),
                    ).fetchall()
            else:
                rows = db.execute(
                    "SELECT payload FROM events WHERE task_id=? "
                    "ORDER BY timestamp ASC, id ASC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]
