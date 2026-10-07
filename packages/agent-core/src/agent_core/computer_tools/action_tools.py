"""Explicit MEDIUM-permission computer interaction tools over the runtime."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..computer.models import (
    FocusWindowAction,
    KeyboardAction,
    KeyboardActionKind,
    KeyboardKey,
    KeyboardModifier,
    MouseAction,
    MouseActionKind,
    MouseButton,
    SelectUIElementAction,
)
from ..computer.runtime import ComputerRuntime
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ._common import (
    ACTION_OUTPUT_SCHEMA,
    VISUAL_VERIFICATION_SCHEMA,
    action_tool_result,
    has_only_keys,
    invalid_input_result,
    parse_verification,
    parse_visual_verification,
)

MOVE_MOUSE_TOOL_NAME = "computer_move_mouse"
CLICK_TOOL_NAME = "computer_click"
DOUBLE_CLICK_TOOL_NAME = "computer_double_click"
FOCUS_WINDOW_TOOL_NAME = "computer_focus_window"
SELECT_UI_ELEMENT_TOOL_NAME = "computer_select_ui_element"
PRESS_KEY_TOOL_NAME = "computer_press_key"
HOTKEY_TOOL_NAME = "computer_hotkey"
TYPE_TEXT_TOOL_NAME = "computer_type_text"

_BUTTON_SCHEMA = {"type": "string", "enum": [button.value for button in MouseButton]}
_MOUSE_PROPERTIES: dict[str, Any] = {
    "x": {"type": "integer", "description": "Screen x coordinate."},
    "y": {"type": "integer", "description": "Screen y coordinate."},
    "button": _BUTTON_SCHEMA,
    "action_id": {"type": "string"},
    "verification": {"type": "object", "description": "Optional deterministic postcondition."},
    "visual_verification": VISUAL_VERIFICATION_SCHEMA,
}
_VERIFICATION_KIND_VALUES = [
    "active_window_changed",
    "active_window_is",
    "cursor_at",
    "ui_element_focused",
    "ui_element_selected",
    "ui_element_presence",
    "screenshot_changed",
    "window_title_changed",
    "window_state_changed",
]


def _input_schema(*, required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


def _mouse_action(
    input: dict[str, Any], kind: MouseActionKind, *, allow_duration: bool
) -> MouseAction:
    payload: dict[str, Any] = {
        "kind": kind,
        "point": {"x": input.get("x"), "y": input.get("y")},
        "button": input.get("button", MouseButton.LEFT.value),
    }
    if "action_id" in input:
        payload["action_id"] = input["action_id"]
    if allow_duration and "duration_s" in input:
        payload["duration_s"] = input["duration_s"]
    return MouseAction.model_validate(payload)


def _run_mouse(
    runtime: ComputerRuntime,
    input: dict[str, Any],
    tool_name: str,
    kind: MouseActionKind,
    *,
    allow_duration: bool,
) -> ToolResult:
    allowed = {"x", "y", "action_id", "verification", "visual_verification"}
    if kind is not MouseActionKind.MOVE:
        allowed.add("button")
    if allow_duration:
        allowed.add("duration_s")
    if not has_only_keys(input, allowed):
        return invalid_input_result(tool_name)
    try:
        action = _mouse_action(input, kind, allow_duration=allow_duration)
        verification = parse_verification(input)
        visual_verification = parse_visual_verification(input)
    except ValidationError:
        return invalid_input_result(tool_name)
    if kind is MouseActionKind.MOVE:
        return action_tool_result(runtime.move_mouse(action, verification, visual_verification))
    if kind is MouseActionKind.CLICK:
        return action_tool_result(runtime.click(action, verification, visual_verification))
    return action_tool_result(runtime.double_click(action, verification, visual_verification))


class ComputerMoveMouseTool:
    spec = ToolSpec(
        name=MOVE_MOUSE_TOOL_NAME,
        description=(
            "Moves the cursor within the observed primary display. This MEDIUM "
            "interaction requires explicit approval by default and is verified "
            "against the resulting cursor position."
        ),
        input_schema=_input_schema(
            required=["x", "y"],
            properties={
                "x": _MOUSE_PROPERTIES["x"],
                "y": _MOUSE_PROPERTIES["y"],
                "duration_s": {"type": "number"},
                "action_id": _MOUSE_PROPERTIES["action_id"],
                "verification": _MOUSE_PROPERTIES["verification"],
                "visual_verification": _MOUSE_PROPERTIES["visual_verification"],
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        return _run_mouse(
            self._runtime,
            input,
            MOVE_MOUSE_TOOL_NAME,
            MouseActionKind.MOVE,
            allow_duration=True,
        )


class ComputerClickTool:
    spec = ToolSpec(
        name=CLICK_TOOL_NAME,
        description=(
            "Performs one MEDIUM-permission click at observed screen coordinates. "
            "Explicit approval is required by default. The result is not successful "
            "unless its supplied postcondition verifies."
        ),
        input_schema=_input_schema(
            required=["x", "y"],
            properties={
                "x": _MOUSE_PROPERTIES["x"],
                "y": _MOUSE_PROPERTIES["y"],
                "button": _MOUSE_PROPERTIES["button"],
                "action_id": _MOUSE_PROPERTIES["action_id"],
                "verification": _MOUSE_PROPERTIES["verification"],
                "visual_verification": _MOUSE_PROPERTIES["visual_verification"],
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        return _run_mouse(
            self._runtime,
            input,
            CLICK_TOOL_NAME,
            MouseActionKind.CLICK,
            allow_duration=False,
        )


class ComputerDoubleClickTool:
    spec = ToolSpec(
        name=DOUBLE_CLICK_TOOL_NAME,
        description=(
            "Performs one MEDIUM-permission double click at observed screen "
            "coordinates. Approval is required by default; it is never retried "
            "automatically."
        ),
        input_schema=_input_schema(
            required=["x", "y"],
            properties={
                "x": _MOUSE_PROPERTIES["x"],
                "y": _MOUSE_PROPERTIES["y"],
                "button": _MOUSE_PROPERTIES["button"],
                "action_id": _MOUSE_PROPERTIES["action_id"],
                "verification": _MOUSE_PROPERTIES["verification"],
                "visual_verification": _MOUSE_PROPERTIES["visual_verification"],
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        return _run_mouse(
            self._runtime,
            input,
            DOUBLE_CLICK_TOOL_NAME,
            MouseActionKind.DOUBLE_CLICK,
            allow_duration=False,
        )


class ComputerFocusWindowTool:
    spec = ToolSpec(
        name=FOCUS_WINDOW_TOOL_NAME,
        description=(
            "Focuses one currently visible window (MEDIUM permission) and "
            "verifies the active window identifier afterward."
        ),
        input_schema=_input_schema(
            required=["window_identifier"],
            properties={
                "window_identifier": {"type": "string"},
                "action_id": {"type": "string"},
                "verification": {"type": "object"},
                "visual_verification": VISUAL_VERIFICATION_SCHEMA,
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        if not has_only_keys(
            input, {"window_identifier", "action_id", "verification", "visual_verification"}
        ):
            return invalid_input_result(FOCUS_WINDOW_TOOL_NAME)
        payload: dict[str, Any] = {
            "window_identifier": input.get("window_identifier"),
        }
        if "action_id" in input:
            payload["action_id"] = input["action_id"]
        try:
            action = FocusWindowAction.model_validate(payload)
            verification = parse_verification(input)
            visual_verification = parse_visual_verification(input)
        except ValidationError:
            return invalid_input_result(FOCUS_WINDOW_TOOL_NAME)
        return action_tool_result(
            self._runtime.focus_window(action, verification, visual_verification)
        )


class ComputerSelectUIElementTool:
    spec = ToolSpec(
        name=SELECT_UI_ELEMENT_TOOL_NAME,
        description=(
            "Selects one enabled accessibility element in a visible window "
            "(MEDIUM permission), then verifies its selected state."
        ),
        input_schema=_input_schema(
            required=["window_identifier", "automation_id"],
            properties={
                "window_identifier": {"type": "string"},
                "automation_id": {"type": "string"},
                "action_id": {"type": "string"},
                "verification": {"type": "object"},
                "visual_verification": VISUAL_VERIFICATION_SCHEMA,
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        if not has_only_keys(
            input,
            {
                "window_identifier",
                "automation_id",
                "action_id",
                "verification",
                "visual_verification",
            },
        ):
            return invalid_input_result(SELECT_UI_ELEMENT_TOOL_NAME)
        payload: dict[str, Any] = {
            "window_identifier": input.get("window_identifier"),
            "automation_id": input.get("automation_id"),
        }
        if "action_id" in input:
            payload["action_id"] = input["action_id"]
        try:
            action = SelectUIElementAction.model_validate(payload)
            verification = parse_verification(input)
            visual_verification = parse_visual_verification(input)
        except ValidationError:
            return invalid_input_result(SELECT_UI_ELEMENT_TOOL_NAME)
        return action_tool_result(
            self._runtime.select_ui_element(action, verification, visual_verification)
        )


class ComputerPressKeyTool:
    spec = ToolSpec(
        name=PRESS_KEY_TOOL_NAME,
        description=(
            "Presses one key from the explicit supported-key set (MEDIUM "
            "permission). It does not capture or monitor input."
        ),
        input_schema=_input_schema(
            required=["key"],
            properties={
                "key": {"type": "string", "enum": [key.value for key in KeyboardKey]},
                "action_id": {"type": "string"},
                "verification": {"type": "object"},
                "visual_verification": VISUAL_VERIFICATION_SCHEMA,
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        if not has_only_keys(input, {"key", "action_id", "verification", "visual_verification"}):
            return invalid_input_result(PRESS_KEY_TOOL_NAME)
        payload = {"kind": KeyboardActionKind.PRESS_KEY, "key": input.get("key")}
        if "action_id" in input:
            payload["action_id"] = input["action_id"]
        try:
            action = KeyboardAction.model_validate(payload)
            verification = parse_verification(input)
            visual_verification = parse_visual_verification(input)
        except ValidationError:
            return invalid_input_result(PRESS_KEY_TOOL_NAME)
        return action_tool_result(
            self._runtime.keyboard_action(action, verification, visual_verification)
        )


class ComputerHotkeyTool:
    spec = ToolSpec(
        name=HOTKEY_TOOL_NAME,
        description=(
            "Performs one allow-listed keyboard shortcut (MEDIUM permission). "
            "Windows-key shortcuts, key hooks, and arbitrary virtual keys are not supported."
        ),
        input_schema=_input_schema(
            required=["key", "modifiers"],
            properties={
                "key": {"type": "string", "enum": [key.value for key in KeyboardKey]},
                "modifiers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [modifier.value for modifier in KeyboardModifier],
                    },
                },
                "action_id": {"type": "string"},
                "verification": {"type": "object"},
                "visual_verification": VISUAL_VERIFICATION_SCHEMA,
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        if not has_only_keys(
            input, {"key", "modifiers", "action_id", "verification", "visual_verification"}
        ):
            return invalid_input_result(HOTKEY_TOOL_NAME)
        payload = {
            "kind": KeyboardActionKind.HOTKEY,
            "key": input.get("key"),
            "modifiers": input.get("modifiers"),
        }
        if "action_id" in input:
            payload["action_id"] = input["action_id"]
        try:
            action = KeyboardAction.model_validate(payload)
            verification = parse_verification(input)
            visual_verification = parse_visual_verification(input)
        except ValidationError:
            return invalid_input_result(HOTKEY_TOOL_NAME)
        return action_tool_result(
            self._runtime.keyboard_action(action, verification, visual_verification)
        )


class ComputerTypeTextTool:
    spec = ToolSpec(
        name=TYPE_TEXT_TOOL_NAME,
        description=(
            "Types bounded literal text into the currently focused visible "
            "application (MEDIUM permission). Typed text is sensitive: it is "
            "redacted from events and is never captured back from the keyboard."
        ),
        input_schema=_input_schema(
            required=["text"],
            properties={
                "text": {"type": "string", "maxLength": 1_024},
                "action_id": {"type": "string"},
                "verification": {"type": "object"},
                "visual_verification": VISUAL_VERIFICATION_SCHEMA,
            },
        ),
        output_schema=ACTION_OUTPUT_SCHEMA,
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
        sensitive_input=True,
        sensitive_output=True,
    )

    def __init__(self, runtime: ComputerRuntime) -> None:
        self._runtime = runtime

    def run(self, input: dict[str, Any]) -> ToolResult:
        if not has_only_keys(input, {"text", "action_id", "verification", "visual_verification"}):
            return invalid_input_result(TYPE_TEXT_TOOL_NAME)
        payload: dict[str, Any] = {
            "kind": KeyboardActionKind.TYPE_TEXT,
            "text": input.get("text"),
        }
        if "action_id" in input:
            payload["action_id"] = input["action_id"]
        try:
            action = KeyboardAction.model_validate(payload)
            verification = parse_verification(input)
            visual_verification = parse_visual_verification(input)
        except ValidationError:
            return invalid_input_result(TYPE_TEXT_TOOL_NAME)
        return action_tool_result(
            self._runtime.keyboard_action(action, verification, visual_verification)
        )
