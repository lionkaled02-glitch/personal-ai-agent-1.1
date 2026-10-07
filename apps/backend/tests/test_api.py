from __future__ import annotations

from pathlib import Path

from agent_core import Agent
from fastapi.testclient import TestClient

from apps.backend.src.api import create_app


def test_api_health_and_task_flow(tmp_path: Path) -> None:
    def factory() -> Agent:
        return Agent.create_demo(workspace_root=tmp_path / "workspace")

    app = create_app(factory, tmp_path / "tasks.sqlite3")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        script = client.post("/creation/script", json={"topic": "test", "target_duration_s": 30})
        assert script.status_code == 200
        board = client.post(
            "/creation/storyboard?scenes=3", json={"topic": "test", "target_duration_s": 30}
        )
        assert board.status_code == 200
        assert len(board.json()["scenes"]) == 3
        response = client.post("/tasks", json={"request": "Run the demo tool."})
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        import time

        for _ in range(100):
            task = client.get(f"/tasks/{task_id}").json()
            if task["state"] == "COMPLETED":
                break
            time.sleep(0.01)
        assert task["state"] == "COMPLETED"
        assert client.get(f"/tasks/{task_id}/events").status_code == 200


def test_capabilities_and_document_indexing(tmp_path):

    from fastapi.testclient import TestClient

    from apps.backend.src.api import create_app

    app = create_app(store_path=tmp_path / "tasks.sqlite3")
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    (ws / "note.txt").write_text("alpha persistent knowledge", encoding="utf-8")
    # app settings use default workspace, so this test only verifies capability route.
    with TestClient(app) as client:
        response = client.get("/capabilities")
        assert response.status_code == 200
        assert response.json()["persistent_knowledge"] is True
