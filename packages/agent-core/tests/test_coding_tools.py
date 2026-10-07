from __future__ import annotations

from pathlib import Path

from agent_core import (
    Agent,
    PermissionLevel,
    ToolRegistry,
    Workspace,
)
from agent_core.coding import register_coding_tools


def test_coding_tools_are_registered_and_bounded(tmp_path: Path) -> None:
    registry = ToolRegistry()
    workspace = Workspace(tmp_path / "workspace")
    register_coding_tools(registry, workspace)
    names = registry.names()
    assert "coding_search_text" in names
    assert "coding_apply_patch" in names
    assert registry.require("coding_apply_patch").spec.permission_level is PermissionLevel.HIGH


def test_agent_exposes_coding_tools(tmp_path: Path) -> None:
    agent = Agent.create_demo(workspace_root=tmp_path / "workspace")
    assert "coding_search_text" in agent.registry.names()
