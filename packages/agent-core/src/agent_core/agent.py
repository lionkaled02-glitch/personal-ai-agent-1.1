"""The agent core facade.

:func:`Agent.run` wires together the complete end-to-end flow:

    user request -> Task (CREATED) -> Planner (PLANNING) ->
    Executor (RUNNING: per-step permission check + tool call) ->
    Verifier (VERIFYING) -> Task (COMPLETED / FAILED / CANCELLED)

Every observable transition emits a structured event on the bus. The agent
is synchronous by design for Phase 0 (see ARCHITECTURE.md, decision D7).
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .browser import BrowserLimits, BrowserProvider, BrowserRuntime, register_browser_tools
from .builtin_tools import register_default_tools
from .coding import CodingLimits, register_coding_tools
from .computer import ComputerLimits, ComputerProvider, ComputerRuntime
from .computer_tools import register_computer_tools
from .config import Settings
from .document_tools import register_document_tools
from .documents.limits import DocumentLimits
from .documents.retrieval import KnowledgeStore
from .errors import PlanningError
from .events import Clock, EventBus, EventType, bounded_text, utc_now
from .executor import BasicVerifier, Executor, Verifier
from .memory.limits import MemoryLimits
from .memory.retrieval import LexicalMemoryRetriever
from .memory.sqlite_store import SQLiteMemoryStore
from .memory.store import InMemoryMemoryStore, MemoryStore
from .memory_tools import register_memory_tools
from .permissions import ApprovalCallback, PermissionManager
from .planner import ModelPlanner, Planner
from .providers.factory import build_gateway
from .providers.gateway import ModelGateway
from .providers.mock import MockModelProvider
from .rag.context import Context, ContextBuilder, ContextRequest
from .tasks import Task, TaskState, TaskStep
from .tools import ToolRegistry
from .vision.limits import VisionLimits
from .vision.runtime import VisionRuntime
from .vision.tools import register_vision_tools
from .workspace import Workspace
from .workspace_tools import register_workspace_tools


def _format_context_for_planner(context: Context) -> str:
    """Serialize retrieved data into a bounded, injection-resistant prompt block."""
    lines: list[str] = []
    for index, item in enumerate(context.items, start=1):
        source = item.source_ref or item.source_id
        location = f" location={item.location!r}" if item.location else ""
        lines.append(
            f"[{index}] kind={item.kind} provenance={item.provenance!r} "
            f"source={source!r}{location}\n{item.text}"
        )
    return "\n\n".join(lines)


class Agent:
    """Minimal agent core: request in, task out, events along the way."""

    def __init__(
        self,
        *,
        planner: Planner,
        registry: ToolRegistry,
        permissions: PermissionManager,
        events: EventBus,
        verifier: Verifier | None = None,
        clock: Clock | None = None,
        knowledge_store: KnowledgeStore | None = None,
        memory_store: MemoryStore | None = None,
        context_builder: ContextBuilder | None = None,
        checkpoint: Callable[[Task], None] | None = None,
    ) -> None:
        self._planner = planner
        self._registry = registry
        self._permissions = permissions
        self._events = events
        self._knowledge_store = knowledge_store if knowledge_store is not None else KnowledgeStore()
        self._memory_store = (
            memory_store
            if memory_store is not None
            else InMemoryMemoryStore(MemoryLimits(), clock=clock if clock is not None else utc_now)
        )
        self._context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder(
                memory_retriever=LexicalMemoryRetriever(self._memory_store),
                knowledge_store=self._knowledge_store,
                limits=self._memory_store.limits,
                clock=clock if clock is not None else utc_now,
            )
        )
        self._executor = Executor(
            registry=registry,
            permissions=permissions,
            events=events,
            verifier=verifier,
            clock=clock,
            checkpoint=checkpoint,
        )
        self._clock: Clock = clock or utc_now

    @classmethod
    def create_demo(
        cls,
        approval: ApprovalCallback | None = None,
        clock: Clock | None = None,
        workspace_root: Path | None = None,
        browser_provider: BrowserProvider | None = None,
        checkpoint: Callable[[Task], None] | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> Agent:
        """A fully wired agent using only in-process fakes.

        No API keys, no network — used by the demo entry point
        (apps/backend/src/main.py) and by integration tests. Registers the
        default tool set (demo tool + Phase 2 safe built-ins) plus the Phase 3
        workspace tools bound to the workspace root (default
        ``data/workspace``), the Phase 4 document tools bound to the same
        boundary and an in-memory knowledge store, and the Phase 5 memory
        tools bound to an in-memory memory store. Filesystem and document
        tools only act inside the workspace boundary; memory is explicit
        (permission-gated creation, never automatic). Phase 9 browser tools are
        registered only when an explicit provider is supplied and share the
        same permissions, events, and local vision boundary.
        """
        registry = ToolRegistry()
        register_default_tools(registry)
        root = workspace_root if workspace_root is not None else Settings().workspace_root
        workspace = Workspace(root)
        register_workspace_tools(registry, workspace)
        register_coding_tools(registry, workspace, CodingLimits())
        store = KnowledgeStore()
        register_document_tools(registry, workspace, store)
        memory = InMemoryMemoryStore(MemoryLimits(), clock=clock)
        register_memory_tools(registry, memory, clock=clock)
        provider = MockModelProvider()
        permissions = PermissionManager(approval=approval)
        events = EventBus(clock=clock)
        if browser_provider is not None:
            browser_vision = VisionRuntime(events=events, clock=clock)
            browser_runtime = BrowserRuntime(
                provider=browser_provider,
                permissions=permissions,
                events=events,
                limits=BrowserLimits(),
                vision=browser_vision,
                clock=clock,
            )
            register_browser_tools(registry, browser_runtime)
        return cls(
            planner=ModelPlanner(provider),
            registry=registry,
            permissions=permissions,
            events=events,
            verifier=BasicVerifier(),
            clock=clock,
            knowledge_store=store,
            memory_store=memory,
            checkpoint=checkpoint,
        )

    @classmethod
    def create_configured(
        cls,
        settings: Settings | None = None,
        approval: ApprovalCallback | None = None,
        clock: Clock | None = None,
        gateway: ModelGateway | None = None,
        computer_provider: ComputerProvider | None = None,
        browser_provider: BrowserProvider | None = None,
        checkpoint: Callable[[Task], None] | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> Agent:
        """An agent wired from configuration (Phase 1).

        The model provider is selected via ``MODEL_PROVIDER`` and reached
        through the :class:`~agent_core.providers.gateway.ModelGateway`.
        With default settings (``MODEL_PROVIDER=mock``) and no optional
        network-backed provider, the agent is fully offline and requires no
        API keys.

        Pass a pre-built ``gateway`` to reuse/inspect one (e.g. to log the
        active provider name); otherwise it is built from ``settings``.
        Registers the default tool set (demo tool + Phase 2 safe built-ins)
        plus the Phase 3 workspace tools bound to ``settings.workspace_root``,
        the Phase 4 document tools bound to the same boundary and an
        in-memory knowledge store whose limits come from the settings, and
        the Phase 5 memory tools bound to an in-memory memory store whose
        limits also come from the settings. Computer and browser tools are
        registered only when explicit providers are passed; they use the same
        permission manager and event bus, with limits from ``COMPUTER_*`` and
        ``BROWSER_*``. Browser network access is opt-in and requires deployment
        egress controls if private/local sites must be isolated.
        """
        resolved = settings if settings is not None else Settings.from_env()
        if gateway is None:
            gateway = build_gateway(resolved)
        registry = ToolRegistry()
        register_default_tools(registry)
        workspace = Workspace.from_settings(resolved)
        register_workspace_tools(registry, workspace)
        store = (
            knowledge_store
            if knowledge_store is not None
            else KnowledgeStore(DocumentLimits.from_settings(resolved))
        )
        register_document_tools(registry, workspace, store)
        memory = SQLiteMemoryStore(
            resolved.data_root / "memory.sqlite3",
            MemoryLimits.from_settings(resolved),
            clock=clock,
        )
        register_memory_tools(registry, memory, clock=clock)
        permissions = PermissionManager(approval=approval)
        events = EventBus(clock=clock)
        vision_runtime: VisionRuntime | None = None
        if computer_provider is not None or browser_provider is not None:
            vision_runtime = VisionRuntime(
                limits=VisionLimits.from_settings(resolved),
                events=events,
                clock=clock,
            )
        if computer_provider is not None:
            assert vision_runtime is not None
            computer_runtime = ComputerRuntime(
                provider=computer_provider,
                permissions=permissions,
                events=events,
                limits=ComputerLimits.from_settings(resolved),
                clock=clock,
                visual_verifier=vision_runtime,
            )
            register_computer_tools(registry, computer_runtime)
            register_vision_tools(registry, computer_runtime, vision_runtime)
        if browser_provider is not None:
            assert vision_runtime is not None
            browser_runtime = BrowserRuntime(
                provider=browser_provider,
                permissions=permissions,
                events=events,
                limits=BrowserLimits.from_settings(resolved),
                vision=vision_runtime,
                clock=clock,
            )
            register_browser_tools(registry, browser_runtime)
        return cls(
            planner=ModelPlanner(gateway),
            registry=registry,
            permissions=permissions,
            events=events,
            verifier=BasicVerifier(),
            clock=clock,
            knowledge_store=store,
            memory_store=memory,
            checkpoint=checkpoint,
        )

    def resume(self, task: Task, *, input_channel: Literal["text", "voice"] = "text") -> Task:
        """Resume a safely paused task without replanning or repeating completed steps."""
        if input_channel not in {"text", "voice"}:
            raise ValueError("unsupported input channel")
        return self._executor.resume(task, redact_sensitive_events=input_channel == "voice")

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def knowledge_store(self) -> KnowledgeStore:
        return self._knowledge_store

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def context_builder(self) -> ContextBuilder:
        return self._context_builder

    def run(
        self,
        request: str,
        *,
        input_channel: Literal["text", "voice"] = "text",
        task_id: str | None = None,
    ) -> Task:
        """Run one user request through the canonical agent task flow.

        Voice is only a transport: it enters the same planner, permission
        manager, executor, and verifier. For privacy, voice-origin task and
        tool lifecycle events redact transcript-derived inputs, outputs, and
        error details; the text-mode event shape remains backwards compatible.
        """
        if input_channel not in {"text", "voice"}:
            raise ValueError("unsupported input channel")
        now = self._clock()
        task = Task.create(request, now=now)
        if task_id is not None:
            task.id = task_id
        created_data = (
            {"input_channel": "voice", "request_chars": len(request)}
            if input_channel == "voice"
            else {"request": request}
        )
        self._events.emit(EventType.TASK_CREATED, task_id=task.id, data=created_data)

        task.transition(TaskState.PLANNING, now=now)
        available_tools = self._registry.list_tools()
        context_text: str | None = None
        try:
            context = self._context_builder.build(
                ContextRequest(memory_query=request, document_query=request)
            )
            if context.items:
                context_text = _format_context_for_planner(context)
        except Exception:
            # Retrieval is advisory; a retrieval failure must never block a
            # user task. It is deliberately not surfaced to the model.
            context_text = None
        try:
            planner_parameters = inspect.signature(self._planner.plan).parameters
            if "context" in planner_parameters:
                plan = self._planner.plan(request, available_tools, context=context_text)
            else:
                # Backward-compatible adapter for custom planners written
                # before contextual planning was introduced.
                plan = self._planner.plan(request, available_tools)
        except PlanningError as exc:
            task.error = f"planning failed: {exc}"
            task.transition(TaskState.FAILED, now=self._clock())
            self._events.emit(
                EventType.TASK_FAILED,
                task_id=task.id,
                data={"error": "voice request planning failed"}
                if input_channel == "voice"
                else {"error": task.error},
            )
            return task

        task.steps = [
            TaskStep(
                id=str(uuid.uuid4()),
                tool_name=step.tool_name,
                description=step.description,
                input=dict(step.input),
            )
            for step in plan.steps
        ]
        safe_tool_names = {tool.name for tool in available_tools}
        self._events.emit(
            EventType.PLAN_CREATED,
            task_id=task.id,
            data={
                "steps": [
                    {
                        "tool_name": step.tool_name
                        if step.tool_name in safe_tool_names
                        else "unavailable_tool"
                    }
                    if input_channel == "voice"
                    else {
                        "tool_name": step.tool_name,
                        "description": bounded_text(step.description),
                    }
                    for step in plan.steps
                ],
            },
        )
        return self._executor.execute(
            task,
            redact_sensitive_events=input_channel == "voice",
        )
