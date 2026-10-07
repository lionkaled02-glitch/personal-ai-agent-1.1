from pathlib import Path

import pytest
from agent_core import Agent
from agent_core.task_manager import TaskManager, TaskStore


def test_idempotency_reuses_existing_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = TaskManager(
        lambda: Agent.create_demo(workspace_root=tmp_path / "workspace"), store, max_workers=1
    )
    try:
        first = manager.submit("Run the demo tool.", idempotency_key="abc")
        second = manager.submit("Run the demo tool.", idempotency_key="abc")
        assert first == second
    finally:
        manager.shutdown()


def test_idempotency_rejects_different_request(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = TaskManager(
        lambda: Agent.create_demo(workspace_root=tmp_path / "workspace"), store, max_workers=1
    )
    try:
        manager.submit("Run the demo tool.", idempotency_key="abc")
        with pytest.raises(ValueError, match="different request"):
            manager.submit("Calculate 2+2.", idempotency_key="abc")
    finally:
        manager.shutdown()
