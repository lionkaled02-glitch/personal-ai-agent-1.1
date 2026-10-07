from __future__ import annotations

import time

import pytest
from agent_core import Agent
from agent_core.task_manager import TaskManager, TaskStore, WorkerRuntime, WorkerRuntimeError


def make_runtime(tmp_path, max_workers=2):
    def factory():
        return Agent.create_demo(workspace_root=tmp_path / "workspace")

    manager = TaskManager(factory, TaskStore(tmp_path / "tasks.sqlite3"), max_workers=max_workers)
    runtime = WorkerRuntime(manager, max_workers=max_workers)
    runtime.start()
    return runtime


def wait_terminal(runtime, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        task = runtime.manager.get(task_id)
        if task is not None and task.is_terminal():
            return task
        time.sleep(0.01)
    return runtime.manager.get(task_id)


def test_worker_lifecycle_and_snapshot(tmp_path):
    runtime = make_runtime(tmp_path)
    task_id = runtime.submit("Run the demo tool.")
    task = wait_terminal(runtime, task_id)
    assert task is not None and task.state.value == "COMPLETED"
    snap = runtime.snapshot()
    assert snap.state == "RUNNING"
    assert snap.max_workers == 2
    runtime.shutdown()
    assert runtime.snapshot().state == "STOPPED"
    with pytest.raises(WorkerRuntimeError):
        runtime.submit("Run the demo tool.")


def test_resource_aware_submission_is_safe_and_bounded(tmp_path):
    runtime = make_runtime(tmp_path, max_workers=2)
    first = runtime.submit_with_resources(
        "Run the demo tool.", resource_keys=("workspace:a", "browser:default")
    )
    second = runtime.submit_with_resources(
        "Run the demo tool.", resource_keys=("browser:default", "workspace:a")
    )
    assert wait_terminal(runtime, first).state.value == "COMPLETED"
    assert wait_terminal(runtime, second).state.value == "COMPLETED"
    runtime.shutdown()


def test_too_many_resource_keys_rejected(tmp_path):
    runtime = make_runtime(tmp_path)
    with pytest.raises(WorkerRuntimeError):
        runtime.submit_with_resources(
            "Run the demo tool.", resource_keys=[str(i) for i in range(17)]
        )
    runtime.shutdown()
