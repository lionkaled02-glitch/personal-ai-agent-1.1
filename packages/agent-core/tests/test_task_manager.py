from __future__ import annotations

import time
from pathlib import Path

from agent_core import Agent, TaskManager, TaskStore


def test_task_store_round_trip(tmp_path: Path) -> None:
    agent = Agent.create_demo(workspace_root=tmp_path / "workspace")
    task = agent.run("Run the demo tool.")
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.save(task)
    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.id == task.id
    assert loaded.state == task.state


def test_task_manager_runs_and_persists(tmp_path: Path) -> None:
    def factory() -> Agent:
        return Agent.create_demo(workspace_root=tmp_path / "workspace")

    manager = TaskManager(factory, TaskStore(tmp_path / "tasks.sqlite3"), max_workers=1)
    task_id = manager.submit("Run the demo tool.")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        task = manager.get(task_id)
        if task is not None and task.is_terminal():
            break
        time.sleep(0.01)
    task = manager.get(task_id)
    manager.shutdown()
    assert task is not None
    assert task.id == task_id
    assert task.state.value == "COMPLETED"


def test_task_manager_cancels_queued_task(tmp_path: Path) -> None:
    import threading

    gate = threading.Event()

    def factory() -> Agent:
        gate.wait(2)
        return Agent.create_demo(workspace_root=tmp_path / "workspace")

    manager = TaskManager(factory, TaskStore(tmp_path / "tasks.sqlite3"), max_workers=1)
    first = manager.submit("Run the demo tool.")
    second = manager.submit("Run the demo tool.")
    assert manager.cancel(second) is True
    gate.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        task = manager.get(first)
        if task is not None and task.is_terminal():
            break
        time.sleep(0.01)
    cancelled = manager.get(second)
    manager.shutdown()
    assert cancelled is not None
    assert cancelled.state.value == "CANCELLED"


def test_event_store_supports_after_event_cursor(tmp_path) -> None:
    from agent_core.task_manager.store import TaskStore

    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.append_event(
        {
            "event_id": "e1",
            "task_id": "t",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "TASK_CREATED",
            "step_id": None,
            "data": {},
        }
    )
    store.append_event(
        {
            "event_id": "e2",
            "task_id": "t",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "type": "TASK_COMPLETED",
            "step_id": None,
            "data": {},
        }
    )
    assert [e["event_id"] for e in store.events("t", after_event_id="e1")] == ["e2"]
