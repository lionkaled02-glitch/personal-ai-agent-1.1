"""Safe, deterministic built-in tools (Phase 2).

Every tool here is local, side-effect-free, schema-validated, LOW-permission,
and bounded. This package deliberately contains NO tools that touch the
shell, filesystem, network, or anything else with side effects — those are
future phases with their own permission/approval work (see SECURITY.md).

:func:`register_default_tools` is the standard way to get the default agent
tool set (demo tool + built-ins).
"""

from __future__ import annotations

from ..demo_tools import DEMO_TOOL_NAME, DemoTool
from ..tools import ToolRegistry
from .calculator import CALCULATOR_TOOL_NAME, CalculatorTool
from .datetime_tool import DATETIME_TOOL_NAME, DateTimeTool
from .json_tool import JSON_TOOL_NAME, JsonUtilsTool
from .text_tool import TEXT_TOOL_NAME, TextUtilsTool

__all__ = [
    "CALCULATOR_TOOL_NAME",
    "DATETIME_TOOL_NAME",
    "DEMO_TOOL_NAME",
    "JSON_TOOL_NAME",
    "TEXT_TOOL_NAME",
    "CalculatorTool",
    "DateTimeTool",
    "DemoTool",
    "JsonUtilsTool",
    "TextUtilsTool",
    "register_default_tools",
]


def register_default_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register the default agent tool set (idempotent per fresh registry).

    Returns the registry for convenience. Duplicate registration raises
    ``ToolAlreadyRegisteredError`` — pass a fresh registry.
    """
    registry.register(DemoTool())
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(TextUtilsTool())
    registry.register(JsonUtilsTool())
    return registry
