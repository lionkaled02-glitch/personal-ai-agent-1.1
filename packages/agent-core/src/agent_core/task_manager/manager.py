"""Bounded asynchronous task manager around the synchronous Agent core."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from threading import Lock, RLock

from ..agent import Agent
from ..events import AgentEvent
from ..tasks import Task
from .store import TaskStore


class TaskManager:
    def __init__(
        self, agent_factory, store: TaskStore, max_workers: int = 4, approval_broker=None
    ) -> None:
        self._agent_factory = agent_factory
        self.store = store
        self._approval_broker = approval_broker
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, 16)), thread_name_prefix="agent-task"
        )
        self._lock = RLock()
        self._futures: dict[str, Future[Task]] = {}
        self._resource_locks: dict[str, Lock] = {}
        self._closed = False

    def submit(self, request: str, *, idempotency_key: str | None = None) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("task manager is shut down")
        if idempotency_key is not None:
            existing = self.store.task_for_idempotency(idempotency_key)
            if existing is not None:
                task = self.store.get(existing)
                if task is not None and task.request != request:
                    raise ValueError("idempotency key is already bound to a different request")
                return existing
        agent = self._agent_factory()
        placeholder = Task.create(request, now=agent.clock())
        if idempotency_key is not None and not self.store.bind_idempotency(
            idempotency_key, placeholder.id, placeholder.created_at.isoformat()
        ):
            existing = self.store.task_for_idempotency(idempotency_key)
            if existing is None:
                raise RuntimeError("idempotency binding failed")
            task = self.store.get(existing)
            if task is not None and task.request != request:
                raise ValueError("idempotency key is already bound to a different request")
            return existing
        self.store.save(placeholder)
        agent.events.subscribe(self._persist_event)
        future = self._executor.submit(self._run, agent, request, placeholder.id)
        with self._lock:
            self._futures[placeholder.id] = future
        return placeholder.id

    def _run(self, agent: Agent, request: str, task_id: str) -> Task:
        task = agent.run(request, task_id=task_id)
        self.store.save(task)
        return task

    def _persist_event(self, event: AgentEvent) -> None:
        self.store.append_event(event.to_dict())

    def submit_with_resources(
        self,
        request: str,
        *,
        resource_keys: tuple[str, ...] = (),
        idempotency_key: str | None = None,
    ) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("task manager is shut down")
        if len(resource_keys) > 16:
            raise ValueError("too many resource keys")
        keys = tuple(sorted({key.strip() for key in resource_keys if key and key.strip()}))
        if not keys:
            return self.submit(request, idempotency_key=idempotency_key)
        if idempotency_key is not None:
            existing = self.store.task_for_idempotency(idempotency_key)
            if existing is not None:
                task = self.store.get(existing)
                if task is not None and task.request != request:
                    raise ValueError("idempotency key is already bound to a different request")
                return existing
        agent = self._agent_factory()
        placeholder = Task.create(request, now=agent.clock())
        if idempotency_key is not None and not self.store.bind_idempotency(
            idempotency_key, placeholder.id, placeholder.created_at.isoformat()
        ):
            existing = self.store.task_for_idempotency(idempotency_key)
            if existing is None:
                raise RuntimeError("idempotency binding failed")
            return existing
        self.store.save(placeholder)
        agent.events.subscribe(self._persist_event)
        future = self._executor.submit(
            self._run_with_resources, agent, request, placeholder.id, keys
        )
        with self._lock:
            self._futures[placeholder.id] = future
        return placeholder.id

    def _resource_lock(self, key: str) -> Lock:
        with self._lock:
            return self._resource_locks.setdefault(key, Lock())

    def _run_with_resources(
        self, agent: Agent, request: str, task_id: str, keys: tuple[str, ...]
    ) -> Task:
        with ExitStack() as stack:
            for key in keys:
                lock = self._resource_lock(key)
                stack.enter_context(lock)
            return self._run(agent, request, task_id)

    def resume(self, task_id: str) -> bool:
        """Resume a persisted PAUSED task in a fresh worker."""
        with self._lock:
            if task_id in self._futures and not self._futures[task_id].done():
                return False
        task = self.store.get(task_id)
        if task is None or task.state.value != "PAUSED":
            return False
        agent = self._agent_factory()
        agent.events.subscribe(self._persist_event)
        future = self._executor.submit(self._resume, agent, task)
        with self._lock:
            self._futures[task_id] = future
        return True

    def _resume(self, agent: Agent, task: Task) -> Task:
        result = agent.resume(task)
        self.store.save(result)
        return result

    def cancel(self, task_id: str) -> bool:
        """Cancel a task that has not started executing yet.

        Running agent tasks are deliberately not force-killed; callers must
        use the cooperative cancellation support of a future runtime.
        """
        with self._lock:
            future = self._futures.get(task_id)
            if future is None:
                return False
            cancelled = future.cancel()
        if not cancelled and self._approval_broker is not None:
            released = self._approval_broker.cancel_task(task_id)
            if released:
                return True
        if cancelled:
            task = self.store.get(task_id)
            if task is not None and not task.is_terminal():
                task.transition(task.state.CANCELLED, now=task.updated_at)
                task.error = "task cancelled before execution"
                self.store.save(task)
        return cancelled

    def active(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
            return bool(future is not None and not future.done())

    def get(self, task_id: str) -> Task | None:
        return self.store.get(task_id)

    def list(self, limit: int = 100) -> list[Task]:
        return self.store.list(limit)

    def events(
        self, task_id: str, limit: int = 200, after_event_id: str | None = None
    ) -> list[dict]:
        return self.store.events(task_id, limit, after_event_id)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
