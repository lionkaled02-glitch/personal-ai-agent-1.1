from pathlib import Path

from agent_core import Agent
from fastapi.testclient import TestClient

from apps.backend.src.api import create_app


def test_api_idempotency_header(tmp_path: Path) -> None:
    app = create_app(
        lambda: Agent.create_demo(workspace_root=tmp_path / "workspace"), tmp_path / "tasks.sqlite3"
    )
    with TestClient(app) as client:
        a = client.post(
            "/tasks", headers={"Idempotency-Key": "req-1"}, json={"request": "Run the demo tool."}
        )
        b = client.post(
            "/tasks", headers={"Idempotency-Key": "req-1"}, json={"request": "Run the demo tool."}
        )
        assert a.status_code == b.status_code == 202
        assert a.json()["task_id"] == b.json()["task_id"]
        c = client.post(
            "/tasks", headers={"Idempotency-Key": "req-1"}, json={"request": "Calculate 2+2."}
        )
        assert c.status_code == 409
