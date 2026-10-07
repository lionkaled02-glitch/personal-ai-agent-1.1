"""Opt-in live-provider test — SKIPPED unless credentials are present.

Run explicitly (never in the default offline suite):

    OPENAI_API_KEY=sk-... .venv/bin/python -m pytest -m live

No credentials in this environment ⇒ the test is skipped, so the default
suite stays offline, deterministic, and CI-safe.
"""

from __future__ import annotations

import os

import pytest
from agent_core import (
    Agent,
    EventBus,
    ModelPlanner,
    PermissionManager,
    Settings,
    ToolRegistry,
    build_gateway,
)
from agent_core.demo_tools import DemoTool
from agent_core.events import utc_now

LIVE = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — live provider test skipped (offline default)",
)


@LIVE
@pytest.mark.live
def test_live_openai_plans_the_demo_tool() -> None:
    pytest.importorskip("openai")
    settings = Settings(
        model_provider="openai",
        model_name=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        model_timeout_s=60.0,
        model_max_retries=1,
    )
    gateway = build_gateway(settings)
    registry = ToolRegistry()
    registry.register(DemoTool())
    agent = Agent(
        planner=ModelPlanner(gateway),
        registry=registry,
        permissions=PermissionManager(),
        events=EventBus(),
        clock=utc_now,
    )
    task = agent.run("Run the demo tool.")
    # A live model may occasionally refuse or misplan; assert the outcome is
    # either a completed demo run or a *controlled* planning failure.
    assert task.state.value in ("COMPLETED", "FAILED")
    if task.state.value == "FAILED":
        assert task.error is not None and "planning failed" in task.error
    else:
        assert task.result is not None
