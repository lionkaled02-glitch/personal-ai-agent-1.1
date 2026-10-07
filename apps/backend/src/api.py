"""FastAPI application for the task manager.

The API exposes bounded task submission/status/events. It does not expose an
arbitrary shell, Python execution endpoint, or raw filesystem endpoint.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from agent_core import Agent, ApprovalBroker, Settings, TaskState, build_gateway
from agent_core.browser.playwright_provider import PlaywrightBrowserProvider
from agent_core.computer.windows import WindowsComputerProvider
from agent_core.document_tools._common import load_document
from agent_core.documents.chunking import chunk_document
from agent_core.documents.sqlite_retrieval import SQLiteKnowledgeStore
from agent_core.task_manager import TaskManager, TaskStore, WorkerRuntime
from agent_core.workspace import Workspace
from creation_agent import CreationService, ScriptRequest
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)
    resource_keys: list[str] = Field(default_factory=list, max_length=16)


class TaskResponse(BaseModel):
    task_id: str


def create_app(
    agent_factory: Callable[[], Agent] | None = None, store_path: Path | None = None
) -> FastAPI:
    settings = Settings.from_env()
    broker = ApprovalBroker(settings.approval_timeout_s, settings.approval_max_pending)
    store = TaskStore(store_path or (settings.workspace_root.parent / "tasks.sqlite3"))
    store.recover_incomplete()

    def approval_callback(request):
        # Mirror the approval wait into the durable task view so the UI can
        # show WAITING_FOR_USER while the worker is safely blocked.
        task = store.get(request.task_id)
        if task is not None and not task.is_terminal():
            try:
                task.transition(TaskState.WAITING_FOR_USER, now=task.updated_at)
                store.save(task)
            except Exception:
                pass
        approved = broker.request(request)
        task = store.get(request.task_id)
        if approved and task is not None and not task.is_terminal():
            try:
                task.transition(TaskState.RUNNING, now=task.updated_at)
                store.save(task)
            except Exception:
                pass
        return approved

    workspace = Workspace.from_settings(settings)
    knowledge_store = SQLiteKnowledgeStore(
        settings.data_root / "knowledge.sqlite3",
    )
    computer_provider = None
    browser_provider = None
    if settings.computer_provider == "windows":
        computer_provider = WindowsComputerProvider()
    if settings.browser_provider == "playwright":
        browser_provider = PlaywrightBrowserProvider()
        browser_provider.launch()

    if agent_factory is None:

        def agent_factory() -> Agent:
            gateway = build_gateway(settings)
            return Agent.create_configured(
                settings=settings,
                gateway=gateway,
                approval=approval_callback,
                checkpoint=store.save,
                computer_provider=computer_provider,
                browser_provider=browser_provider,
                knowledge_store=knowledge_store,
            )

    manager = TaskManager(agent_factory, store, approval_broker=broker)
    worker = WorkerRuntime(manager, max_workers=4)
    worker.start()
    creation = CreationService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        worker.shutdown(wait=True)
        if browser_provider is not None:
            browser_provider.close_browser()
        knowledge_store.close()

    app = FastAPI(title="Personal AI Agent", version="1.0.0", lifespan=lifespan)
    app.state.task_manager = manager
    app.state.approval_broker = broker
    app.state.worker_runtime = worker
    app.state.knowledge_store = knowledge_store
    app.state.computer_provider = computer_provider
    app.state.browser_provider = browser_provider

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parents[2] / "frontend" / "index.html")

    @app.get("/capabilities")
    def capabilities() -> dict:
        return {
            "model_provider": settings.model_provider,
            "computer": settings.computer_provider or "none",
            "browser": settings.browser_provider or "none",
            "persistent_memory": True,
            "persistent_knowledge": True,
            "voice_transport": "provider-neutral",
            "creation": True,
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents/index")
    def index_document(path: str) -> dict:
        try:
            document, _ = load_document(workspace, path, knowledge_store.limits)
            result = chunk_document(document, knowledge_store.limits)
            knowledge_store.add_document(document, result.chunks)
            return {
                "document_id": document.document_id,
                "source_path": document.source_path,
                "chunks": len(result.chunks),
                "truncated": document.truncated or result.truncated,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/documents/search")
    def search_documents(q: str, limit: int = 10) -> list[dict]:
        limit = max(1, min(limit, knowledge_store.limits.max_search_results))
        try:
            hits = knowledge_store.search(q, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "document_id": hit.chunk.document_id,
                "title": hit.document_title,
                "score": hit.score,
                "text": hit.chunk.text,
                "location": hit.chunk.location,
            }
            for hit in hits
        ]

    @app.post("/creation/script")
    def create_script(payload: ScriptRequest) -> dict:
        return creation.create_script(payload).model_dump(mode="json")

    @app.post("/creation/storyboard")
    def create_storyboard(payload: ScriptRequest, scenes: int = 5) -> dict:
        script = creation.create_script(payload)
        return creation.storyboard(script, scenes=scenes).model_dump(mode="json")

    @app.get("/approvals")
    def list_approvals() -> list[dict]:
        return [
            {
                "task_id": item.request.task_id,
                "step_id": item.request.step_id,
                "tool_name": item.request.tool_name,
                "permission_level": item.request.permission_level.name,
                "reason": item.request.reason,
            }
            for item in broker.list_pending()
        ]

    @app.post("/approvals/{task_id}/{step_id}")
    def decide_approval(task_id: str, step_id: str, approved: bool) -> dict[str, bool]:
        if manager.get(task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        if not broker.decide(task_id, step_id, approved):
            raise HTTPException(status_code=409, detail="approval is no longer pending")
        return {"approved": approved}

    @app.post("/tasks", response_model=TaskResponse, status_code=202)
    def create_task(
        payload: TaskRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> TaskResponse:
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip() or None
            if idempotency_key is not None and len(idempotency_key) > 200:
                raise HTTPException(status_code=400, detail="Idempotency-Key is too long")
        try:
            return TaskResponse(
                task_id=worker.submit_with_resources(
                    payload.request,
                    resource_keys=payload.resource_keys,
                    idempotency_key=idempotency_key,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/worker/status")
    def worker_status() -> dict:
        return worker.snapshot().__dict__

    @app.get("/tasks")
    def list_tasks(limit: int = 100) -> list[dict]:
        return [task.model_dump(mode="json") for task in manager.list(limit)]

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str) -> dict:
        task = manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task.model_dump(mode="json")

    @app.post("/tasks/{task_id}/resume")
    def resume_task(task_id: str) -> dict[str, bool]:
        if manager.get(task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        if not manager.resume(task_id):
            raise HTTPException(status_code=409, detail="task is not paused or is already running")
        return {"resumed": True}

    @app.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict[str, bool]:
        if manager.get(task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        if not manager.cancel(task_id):
            raise HTTPException(status_code=409, detail="task already started or finished")
        return {"cancelled": True}

    @app.get("/tasks/{task_id}/events")
    def get_events(task_id: str, limit: int = 200, after_event_id: str | None = None) -> list[dict]:
        if manager.get(task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        return manager.events(task_id, limit, after_event_id)

    @app.websocket("/ws/tasks/{task_id}")
    async def task_events(websocket: WebSocket, task_id: str) -> None:
        await websocket.accept()
        if manager.get(task_id) is None:
            await websocket.close(code=1008, reason="task not found")
            return
        sent = 0
        try:
            while True:
                events = manager.events(task_id, 200)
                for event in events[sent:]:
                    await websocket.send_json(event)
                sent = len(events)
                task = manager.get(task_id)
                if task is not None and task.is_terminal() and sent >= len(events):
                    await websocket.send_json({"type": "TASK_STREAM_COMPLETE", "task_id": task_id})
                    return
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
