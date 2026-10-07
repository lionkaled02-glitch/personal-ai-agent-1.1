from __future__ import annotations

from pathlib import Path

from agent_core.events import utc_now
from agent_core.task_manager.store import TaskStore
from agent_core.tasks import StepStatus, Task, TaskState, TaskStep


def _task(state: TaskState, step_status: StepStatus = StepStatus.PENDING) -> Task:
    now = utc_now()
    return Task(
        id="recovery-task",
        request="recover",
        state=state,
        created_at=now,
        updated_at=now,
        steps=[TaskStep(id="step-1", tool_name="demo", description="demo", status=step_status)],
    )


def test_recover_pauses_nonterminal_tasks(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.save(_task(TaskState.RUNNING))
    assert store.recover_incomplete() == 1
    recovered = store.get("recovery-task")
    assert recovered is not None
    assert recovered.state is TaskState.PAUSED
    assert recovered.steps[0].status is StepStatus.PENDING


def test_recovery_does_not_make_interrupted_step_resumable(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.save(_task(TaskState.RUNNING, StepStatus.RUNNING))
    store.recover_incomplete()
    recovered = store.get("recovery-task")
    assert recovered is not None
    assert recovered.state is TaskState.PAUSED
    assert recovered.steps[0].status is StepStatus.RUNNING
