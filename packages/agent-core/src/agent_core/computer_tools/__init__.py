"""Explicit computer tools registered against a provider-neutral runtime."""

from __future__ import annotations

from ..computer.runtime import ComputerRuntime
from ..tools import ToolRegistry
from .action_tools import (
    CLICK_TOOL_NAME,
    DOUBLE_CLICK_TOOL_NAME,
    FOCUS_WINDOW_TOOL_NAME,
    HOTKEY_TOOL_NAME,
    MOVE_MOUSE_TOOL_NAME,
    PRESS_KEY_TOOL_NAME,
    SELECT_UI_ELEMENT_TOOL_NAME,
    TYPE_TEXT_TOOL_NAME,
    ComputerClickTool,
    ComputerDoubleClickTool,
    ComputerFocusWindowTool,
    ComputerHotkeyTool,
    ComputerMoveMouseTool,
    ComputerPressKeyTool,
    ComputerSelectUIElementTool,
    ComputerTypeTextTool,
)
from .observation_tools import (
    ACTIVE_WINDOW_TOOL_NAME,
    CURSOR_POSITION_TOOL_NAME,
    INSPECT_UI_TOOL_NAME,
    LIST_WINDOWS_TOOL_NAME,
    SCREEN_INFO_TOOL_NAME,
    SCREENSHOT_TOOL_NAME,
    ComputerActiveWindowTool,
    ComputerCursorPositionTool,
    ComputerInspectUITool,
    ComputerListWindowsTool,
    ComputerScreenInfoTool,
    ComputerScreenshotTool,
)

COMPUTER_TOOL_NAMES: tuple[str, ...] = (
    ACTIVE_WINDOW_TOOL_NAME,
    CLICK_TOOL_NAME,
    CURSOR_POSITION_TOOL_NAME,
    DOUBLE_CLICK_TOOL_NAME,
    FOCUS_WINDOW_TOOL_NAME,
    HOTKEY_TOOL_NAME,
    INSPECT_UI_TOOL_NAME,
    LIST_WINDOWS_TOOL_NAME,
    MOVE_MOUSE_TOOL_NAME,
    PRESS_KEY_TOOL_NAME,
    SCREEN_INFO_TOOL_NAME,
    SCREENSHOT_TOOL_NAME,
    SELECT_UI_ELEMENT_TOOL_NAME,
    TYPE_TEXT_TOOL_NAME,
)

__all__ = [
    "ACTIVE_WINDOW_TOOL_NAME",
    "CLICK_TOOL_NAME",
    "COMPUTER_TOOL_NAMES",
    "CURSOR_POSITION_TOOL_NAME",
    "DOUBLE_CLICK_TOOL_NAME",
    "FOCUS_WINDOW_TOOL_NAME",
    "HOTKEY_TOOL_NAME",
    "INSPECT_UI_TOOL_NAME",
    "LIST_WINDOWS_TOOL_NAME",
    "MOVE_MOUSE_TOOL_NAME",
    "PRESS_KEY_TOOL_NAME",
    "SCREENSHOT_TOOL_NAME",
    "SCREEN_INFO_TOOL_NAME",
    "SELECT_UI_ELEMENT_TOOL_NAME",
    "TYPE_TEXT_TOOL_NAME",
    "ComputerActiveWindowTool",
    "ComputerClickTool",
    "ComputerCursorPositionTool",
    "ComputerDoubleClickTool",
    "ComputerFocusWindowTool",
    "ComputerHotkeyTool",
    "ComputerInspectUITool",
    "ComputerListWindowsTool",
    "ComputerMoveMouseTool",
    "ComputerPressKeyTool",
    "ComputerScreenInfoTool",
    "ComputerScreenshotTool",
    "ComputerSelectUIElementTool",
    "ComputerTypeTextTool",
    "register_computer_tools",
]


def register_computer_tools(registry: ToolRegistry, runtime: ComputerRuntime) -> None:
    """Register explicit schemas; no generic arbitrary-action tool is exposed."""
    registry.register(ComputerScreenInfoTool(runtime))
    registry.register(ComputerCursorPositionTool(runtime))
    registry.register(ComputerActiveWindowTool(runtime))
    registry.register(ComputerListWindowsTool(runtime))
    registry.register(ComputerInspectUITool(runtime))
    registry.register(ComputerScreenshotTool(runtime))
    registry.register(ComputerMoveMouseTool(runtime))
    registry.register(ComputerClickTool(runtime))
    registry.register(ComputerDoubleClickTool(runtime))
    registry.register(ComputerFocusWindowTool(runtime))
    registry.register(ComputerSelectUIElementTool(runtime))
    registry.register(ComputerPressKeyTool(runtime))
    registry.register(ComputerHotkeyTool(runtime))
    registry.register(ComputerTypeTextTool(runtime))
