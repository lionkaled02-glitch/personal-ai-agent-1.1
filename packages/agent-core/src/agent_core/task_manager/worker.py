"""Bounded worker runtime for durable agent tasks.

The runtime owns lifecycle state around :class:`TaskManager` without exposing
arbitrary threads or shell execution.  Resource keys are optional logical
locks (for example ``browser:profile:default`` or ``workspace:project-a``)
used to prevent conflicting tasks from running concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, RLock

from .manager import TaskManager


class WorkerRuntimeError(RuntimeError):
    """Raised when the worker runtime cannot accept an operation."""


@dataclass(frozen=True)
class WorkerSnapshot:
    state: str
    max_workers: int
    active_tasks: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int


class WorkerRuntime:
    """Lifecycle and conflict-control layer around ``TaskManager``.

    The underlying manager remains responsible for durable task state. This
    layer adds explicit startup/shutdown state, bounded metrics, and optional
    per-resource serialization. It never force-kills a running task.
    """

    def __init__(self, manager: TaskManager, max_workers: int = 4) -> None:
        self.manager = manager
        self.max_workers = max(1, min(max_workers, 16))
        self._state = "CREATED"
        self._lock = RLock()
        self._resource_locks: dict[str, Lock] = {}
        self._metrics = {"completed": 0, "failed": 0, "cancelled": 0}

    def start(self) -> None:
        with self._lock:
            if self._state == "RUNNING":
                return
            if self._state == "STOPPED":
                raise WorkerRuntimeError("worker runtime cannot be restarted")
            self._state = "RUNNING"

    def submit(self, request: str, *, idempotency_key: str | None = None) -> str:
        with self._lock:
            if self._state not in {"RUNNING", "DRAINING"}:
                raise WorkerRuntimeError("worker runtime is not running")
            if self._state == "DRAINING":
                raise WorkerRuntimeError("worker runtime is draining")
        return self.manager.submit(request, idempotency_key=idempotency_key)

    def submit_with_resources(
        self,
        request: str,
        *,
        resource_keys: list[str] | tuple[str, ...] = (),
        idempotency_key: str | None = None,
    ) -> str:
        """Submit a task whose execution is serialized by logical resources.

        Locks are acquired inside the worker thread, so submitting a task does
        not block the API caller. Keys are normalized and sorted to avoid lock
        ordering deadlocks.
        """
        keys = tuple(sorted({k.strip() for k in resource_keys if k and k.strip()}))
        if len(keys) > 16:
            raise WorkerRuntimeError("too many resource keys")
        # TaskManager does not currently expose a per-job callback. Store the
        # resource reservation as a wrapper request is not safe, so this method
        # uses a dedicated manager factory hook when available.
        submit_with_resources = getattr(self.manager, "submit_with_resources", None)
        if submit_with_resources is None:
            # No unsafe fallback: resource serialization must be enforced by
            # the manager rather than merely recorded.
            raise WorkerRuntimeError("resource-aware submission is unavailable")
        return submit_with_resources(request, resource_keys=keys, idempotency_key=idempotency_key)

    def resume(self, task_id: str) -> bool:
        with self._lock:
            if self._state not in {"RUNNING", "DRAINING"}:
                return False
        return self.manager.resume(task_id)

    def cancel(self, task_id: str) -> bool:
        return self.manager.cancel(task_id)

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            state = self._state
        tasks = self.manager.list(1000)
        active = sum(1 for task in tasks if self.manager.active(task.id))
        queued = sum(
            1
            for task in tasks
            if task.state.value in {"CREATED", "PLANNING"} and not self.manager.active(task.id)
        )
        completed = sum(1 for task in tasks if task.state.value == "COMPLETED")
        failed = sum(1 for task in tasks if task.state.value == "FAILED")
        cancelled = sum(1 for task in tasks if task.state.value == "CANCELLED")
        return WorkerSnapshot(
            state=state,
            max_workers=self.max_workers,
            active_tasks=active,
            queued_tasks=queued,
            completed_tasks=completed,
            failed_tasks=failed,
            cancelled_tasks=cancelled,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._state == "STOPPED":
                return
            self._state = "DRAINING"
        self.manager.shutdown() if wait else self.manager.shutdown(wait=False)
        with self._lock:
            self._state = "STOPPED"
