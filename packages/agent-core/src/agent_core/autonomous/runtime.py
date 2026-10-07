"""Conservative retry/recovery around the canonical Agent flow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..agent import Agent
from ..permissions import PermissionLevel
from ..tasks import Task


class RecoveryDecision(StrEnum):
    COMPLETED = "COMPLETED"
    RETRY_LOW_RISK = "RETRY_LOW_RISK"
    STOP_REQUIRES_USER = "STOP_REQUIRES_USER"
    STOP_RETRY_LIMIT = "STOP_RETRY_LIMIT"


@dataclass(frozen=True)
class AutonomousLimits:
    """Hard bounds for autonomous recovery."""

    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")


class AutonomousRunner:
    """Run requests with bounded, side-effect-aware recovery.

    A failed task is retried only when every completed step was LOW permission.
    Any completed MEDIUM/HIGH step stops automatic retry to avoid duplicating
    side effects such as writes, purchases, sends, or destructive actions.
    """

    def __init__(self, agent: Agent, limits: AutonomousLimits | None = None) -> None:
        self._agent = agent
        self._limits = limits or AutonomousLimits()

    def run(self, request: str, *, input_channel: str = "text") -> tuple[Task, RecoveryDecision]:
        last: Task | None = None
        for attempt in range(self._limits.max_attempts):
            task = self._agent.run(request, input_channel=input_channel)
            last = task
            if task.is_terminal() and task.state.value == "COMPLETED":
                return task, RecoveryDecision.COMPLETED
            if not self._can_retry(task):
                return task, RecoveryDecision.STOP_REQUIRES_USER
            if attempt + 1 >= self._limits.max_attempts:
                return task, RecoveryDecision.STOP_RETRY_LIMIT
        assert last is not None
        return last, RecoveryDecision.STOP_RETRY_LIMIT

    def _can_retry(self, task: Task) -> bool:
        """Return true only when no completed step had side effects."""
        registry = self._agent.registry
        for step in task.steps:
            if step.status.value != "COMPLETED":
                continue
            tool = registry.require(step.tool_name)
            if tool.spec.permission_level is not PermissionLevel.LOW:
                return False
        return True
