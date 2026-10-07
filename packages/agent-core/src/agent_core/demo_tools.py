"""The first concrete tool: a harmless, deterministic demo tool.

It exists to prove the end-to-end flow (plan -> permission -> execute ->
verify -> complete). Real tools (files, web, ...) arrive in later phases,
each declared with its own permission level.
"""

from __future__ import annotations

from typing import Any

from .permissions import PermissionLevel
from .tools import ToolResult, ToolSpec

DEMO_TOOL_NAME = "demo_tool"


class DemoTool:
    """Echoes a message; no side effects; LOW permission."""

    spec = ToolSpec(
        name=DEMO_TOOL_NAME,
        description="Demonstration tool: echoes a message to prove the tool pipeline works.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo back."},
            },
            "required": ["message"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["tool", "message"],
        },
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output={"tool": DEMO_TOOL_NAME, "message": input["message"]})
