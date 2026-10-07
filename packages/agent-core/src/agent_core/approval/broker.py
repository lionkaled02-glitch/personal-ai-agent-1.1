"""Bounded human-approval broker for consequential agent actions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
from time import monotonic

from ..permissions import ApprovalRequest


@dataclass(frozen=True)
class PendingApproval:
    request: ApprovalRequest
    created_at: float


class ApprovalBroker:
    """Synchronous approval callback backed by a bounded in-process queue.

    The agent worker blocks while an approval is pending. No operation is
    executed before the callback returns True. Expiry fails closed.
    """

    def __init__(self, timeout_s: float = 300.0, max_pending: int = 32) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not 1 <= max_pending <= 256:
            raise ValueError("max_pending must be between 1 and 256")
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._pending: dict[tuple[str, str], PendingApproval] = {}
        self._decisions: dict[tuple[str, str], bool] = {}

    def request(self, request: ApprovalRequest) -> bool:
        key = (request.task_id, request.step_id)
        with self._condition:
            if len(self._pending) >= self.max_pending:
                return False
            self._pending[key] = PendingApproval(request=request, created_at=monotonic())
            self._condition.notify_all()
            deadline = monotonic() + self.timeout_s
            while key not in self._decisions:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._pending.pop(key, None)
                    return False
                self._condition.wait(timeout=remaining)
            approved = self._decisions.pop(key)
            self._pending.pop(key, None)
            return approved

    def decide(self, task_id: str, step_id: str, approved: bool) -> bool:
        key = (task_id, step_id)
        with self._condition:
            if key not in self._pending:
                return False
            self._decisions[key] = bool(approved)
            self._condition.notify_all()
            return True

    def cancel_task(self, task_id: str) -> int:
        """Deny all currently pending approvals for one task."""
        with self._condition:
            keys = [key for key in self._pending if key[0] == task_id]
            for key in keys:
                self._decisions[key] = False
            if keys:
                self._condition.notify_all()
            return len(keys)

    def list_pending(self) -> list[PendingApproval]:
        with self._lock:
            return list(self._pending.values())

    def has_pending(self, task_id: str, step_id: str) -> bool:
        with self._lock:
            return (task_id, step_id) in self._pending
