"""Task and step state machine tests (no external APIs, fully deterministic)."""

from __future__ import annotations

import pytest
from agent_core import StepStatus, Task, TaskState, TaskStateError, TaskStep
from conftest import FIXED_NOW


def make_task(request: str = "do something") -> Task:
    return Task.create(request, now=FIXED_NOW)


def make_step(tool_name: str = "demo_tool") -> TaskStep:
    return TaskStep(id="step-1", tool_name=tool_name, description="d", input={"message": "x"})


def status_of(step: TaskStep) -> StepStatus:
    """Read the status across a function boundary (mypy type-narrowing helper)."""
    return step.status


class TestTaskLifecycle:
    def test_create_starts_in_created(self) -> None:
        task = make_task()
        assert task.state is TaskState.CREATED
        assert task.request == "do something"
        assert task.steps == []
        assert task.created_at == FIXED_NOW
        assert task.updated_at == FIXED_NOW
        assert not task.is_terminal()

    def test_happy_path_chain(self) -> None:
        task = make_task()
        for state in (
            TaskState.PLANNING,
            TaskState.RUNNING,
            TaskState.VERIFYING,
            TaskState.COMPLETED,
        ):
            task.transition(state, now=FIXED_NOW)
        assert task.state is TaskState.COMPLETED
        assert task.is_terminal()

    def test_illegal_transition_raises(self) -> None:
        task = make_task()
        with pytest.raises(TaskStateError, match="CREATED -> RUNNING"):
            task.transition(TaskState.RUNNING)
        assert task.state is TaskState.CREATED

    def test_terminal_states_refuse_further_transitions(self) -> None:
        task = make_task()
        task.transition(TaskState.PLANNING)
        task.transition(TaskState.RUNNING)
        task.transition(TaskState.FAILED)
        with pytest.raises(TaskStateError):
            task.transition(TaskState.RUNNING)

    def test_waiting_for_user_roundtrip(self) -> None:
        task = make_task()
        task.transition(TaskState.PLANNING)
        task.transition(TaskState.RUNNING)
        task.transition(TaskState.WAITING_FOR_USER)
        assert task.can_transition(TaskState.RUNNING)
        task.transition(TaskState.RUNNING)
        assert task.state is TaskState.RUNNING

    def test_pause_resume_roundtrip(self) -> None:
        task = make_task()
        task.transition(TaskState.PLANNING)
        task.transition(TaskState.RUNNING)
        task.transition(TaskState.PAUSED)
        assert not task.can_transition(TaskState.COMPLETED)
        task.transition(TaskState.RUNNING)
        task.transition(TaskState.VERIFYING)
        task.transition(TaskState.COMPLETED)
        assert task.state is TaskState.COMPLETED

    def test_cancellation_from_non_terminal_states(self) -> None:
        for start in (
            TaskState.CREATED,
            TaskState.PLANNING,
            TaskState.RUNNING,
            TaskState.WAITING_FOR_USER,
            TaskState.PAUSED,
            TaskState.VERIFYING,
        ):
            task = make_task()
            for state in (TaskState.PLANNING, TaskState.RUNNING):
                if task.state is not start:
                    task.transition(state)
            if start is TaskState.WAITING_FOR_USER:
                task.transition(TaskState.WAITING_FOR_USER)
            elif start is TaskState.PAUSED:
                task.transition(TaskState.PAUSED)
            elif start is TaskState.VERIFYING:
                task.transition(TaskState.VERIFYING)
            task.transition(TaskState.CANCELLED)
            assert task.is_terminal()

    def test_transition_updates_updated_at(self) -> None:
        later = FIXED_NOW.replace(second=5)
        task = make_task()
        task.transition(TaskState.PLANNING, now=later)
        assert task.updated_at == later
        assert task.created_at == FIXED_NOW


class TestStepStatus:
    def test_step_happy_path(self) -> None:
        step = make_step()
        assert status_of(step) == StepStatus.PENDING
        step.transition(StepStatus.RUNNING)
        step.transition(StepStatus.COMPLETED)
        assert status_of(step) == StepStatus.COMPLETED

    def test_step_illegal_transition_raises(self) -> None:
        step = make_step()
        with pytest.raises(TaskStateError):
            step.transition(StepStatus.COMPLETED)  # PENDING -> COMPLETED illegal
        step.transition(StepStatus.RUNNING)
        with pytest.raises(TaskStateError):
            step.transition(StepStatus.RUNNING)

    def test_step_terminal_states_are_final(self) -> None:
        step = make_step()
        step.transition(StepStatus.RUNNING)
        step.transition(StepStatus.FAILED)
        with pytest.raises(TaskStateError):
            step.transition(StepStatus.RUNNING)
