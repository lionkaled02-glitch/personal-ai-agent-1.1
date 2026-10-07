"""Shared fixtures: deterministic clock, scripted plan, wired-up demo agent."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from agent_core import (
    Agent,
    DemoTool,
    EventBus,
    MockModelProvider,
    ModelPlanner,
    PermissionManager,
    ToolRegistry,
)

FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)

DEMO_PLAN_JSON = (
    '{"steps": [{"tool_name": "demo_tool", '
    '"description": "Run the demo tool", "input": {"message": "hi from test"}}]}'
)


Clock = Callable[[], datetime]


@pytest.fixture
def fixed_clock() -> Clock:
    """A clock that always returns FIXED_NOW (deterministic timestamps)."""
    return lambda: FIXED_NOW


@pytest.fixture
def demo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DemoTool())
    return registry


@pytest.fixture
def scripted_plan_provider() -> MockModelProvider:
    """Mock provider that answers the planning prompt with the demo plan."""
    return MockModelProvider(responses=[DEMO_PLAN_JSON])


@pytest.fixture
def demo_agent(
    demo_registry: ToolRegistry,
    scripted_plan_provider: MockModelProvider,
    fixed_clock: Clock,
) -> Agent:
    """Fully wired agent: scripted mock provider, demo tool, default policy."""
    return Agent(
        planner=ModelPlanner(scripted_plan_provider),
        registry=demo_registry,
        permissions=PermissionManager(),
        events=EventBus(clock=fixed_clock),
        clock=fixed_clock,
    )
