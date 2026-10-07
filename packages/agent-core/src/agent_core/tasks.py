"""Task and step state models.

A :class:`Task` is one user request plus the ordered list of
:class:`TaskStep` the agent planned to fulfil it. Transitions are governed
by an explicit map, which keeps future multi-step workflows (pause/resume,
waiting for the user, verification) predictable and testable.

States: CREATED, PLANNING, RUNNING, WAITING_FOR_USER, PAUSED, VERIFYING,
COMPLETED, FAILED, CANCELLED. The last three are terminal.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import TaskStateError


class TaskState(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    PAUSED = "PAUSED"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)

_ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.CREATED: frozenset({TaskState.PLANNING, TaskState.CANCELLED}),
    TaskState.PLANNING: frozenset({TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.WAITING_FOR_USER,
            TaskState.PAUSED,
            TaskState.VERIFYING,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.WAITING_FOR_USER: frozenset(
        {TaskState.RUNNING, TaskState.PAUSED, TaskState.CANCELLED}
    ),
    TaskState.PAUSED: frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
    TaskState.VERIFYING: frozenset(
        {TaskState.COMPLETED, TaskState.FAILED, TaskState.RUNNING, TaskState.CANCELLED}
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


class StepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.RUNNING, StepStatus.CANCELLED}),
    StepStatus.RUNNING: frozenset({StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED}),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
}


class TaskStep(BaseModel):
    """One planned unit of work. Today a step is always a tool call; future
    steps may target sub-agents — the ``tool_name`` field will be replaced by
    a step-type field at that time (documented in ARCHITECTURE.md)."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    tool_name: str
    description: str
    input: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    output: Any | None = None
    error: str | None = None

    def transition(self, to: StepStatus) -> None:
        if to not in _STEP_TRANSITIONS[self.status]:
            raise TaskStateError(f"illegal step transition {self.status.value} -> {to.value}")
        self.status = to


class Task(BaseModel):
    """One user request and everything the agent does about it."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    request: str
    state: TaskState = TaskState.CREATED
    steps: list[TaskStep] = Field(default_factory=list)
    result: Any | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, request: str, now: datetime) -> Task:
        return cls(id=str(uuid.uuid4()), request=request, created_at=now, updated_at=now)

    def can_transition(self, to: TaskState) -> bool:
        return to in _ALLOWED_TRANSITIONS[self.state]

    def transition(self, to: TaskState, now: datetime | None = None) -> None:
        if not self.can_transition(to):
            raise TaskStateError(f"illegal task transition {self.state.value} -> {to.value}")
        self.state = to
        if now is not None:
            self.updated_at = now

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
