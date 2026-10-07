"""Optional Windows UI Automation adapter for the computer provider seam.

The module itself imports no Windows-only packages. Dependencies are loaded
only when a ``WindowsComputerProvider`` is constructed on Windows, so core
imports and fake-provider tests remain portable.
"""

from __future__ import annotations

import io
import math
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import islice
from typing import Any

from .errors import (
    ComputerValidationError,
    ProviderUnavailableError,
    UnsupportedPlatformError,
)
from .models import (
    Bounds,
    KeyboardKey,
    KeyboardModifier,
    MouseButton,
    Point,
    ScreenInfo,
    ScreenshotObservation,
    UIElement,
    WindowInfo,
)

_KEY_TOKENS: dict[KeyboardKey, str] = {
    KeyboardKey.ENTER: "{ENTER}",
    KeyboardKey.TAB: "{TAB}",
    KeyboardKey.ESC: "{ESC}",
    KeyboardKey.SPACE: "{SPACE}",
    KeyboardKey.BACKSPACE: "{BACKSPACE}",
    KeyboardKey.DELETE: "{DELETE}",
    KeyboardKey.INSERT: "{INSERT}",
    KeyboardKey.HOME: "{HOME}",
    KeyboardKey.END: "{END}",
    KeyboardKey.PAGE_UP: "{PGUP}",
    KeyboardKey.PAGE_DOWN: "{PGDN}",
    KeyboardKey.LEFT: "{LEFT}",
    KeyboardKey.RIGHT: "{RIGHT}",
    KeyboardKey.UP: "{UP}",
    KeyboardKey.DOWN: "{DOWN}",
    KeyboardKey.F1: "{F1}",
    KeyboardKey.F2: "{F2}",
    KeyboardKey.F3: "{F3}",
    KeyboardKey.F4: "{F4}",
    KeyboardKey.F5: "{F5}",
    KeyboardKey.F6: "{F6}",
    KeyboardKey.F7: "{F7}",
    KeyboardKey.F8: "{F8}",
    KeyboardKey.F9: "{F9}",
    KeyboardKey.F10: "{F10}",
    KeyboardKey.F11: "{F11}",
    KeyboardKey.F12: "{F12}",
}


class WindowsComputerProvider:
    """Windows-only implementation backed by pywinauto UIA + pywin32.

    UI inspection uses the UI Automation backend. Input uses explicit
    pywinauto mouse/keyboard calls, window focus/select operations, and
    screenshots use an in-memory Pillow buffer that is discarded after the
    returned observation. It never starts an application or process.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise UnsupportedPlatformError()
        try:
            import win32api
            from PIL import ImageGrab
            from pywinauto import Desktop, keyboard, mouse
        except ImportError as exc:
            raise ProviderUnavailableError(
                "Windows computer provider requires the optional agent-core[computer-windows] extra"
            ) from exc

        # Optional vendor/platform types are deliberately isolated to this
        # adapter. The rest of agent_core sees only ComputerProvider models.
        self._desktop: Any = Desktop(backend="uia")
        self._keyboard: Any = keyboard
        self._mouse: Any = mouse
        self._win32api: Any = win32api
        self._image_grab: Any = ImageGrab

    def get_screen_info(self) -> ScreenInfo:
        width = int(self._win32api.GetSystemMetrics(0))
        height = int(self._win32api.GetSystemMetrics(1))
        dpi = self._system_dpi()
        return ScreenInfo(
            width=width,
            height=height,
            display="primary",
            dpi_x=dpi,
            dpi_y=dpi,
            scale_x=dpi / 96.0 if dpi is not None else None,
            scale_y=dpi / 96.0 if dpi is not None else None,
        )

    def get_cursor_position(self) -> Point:
        x, y = self._win32api.GetCursorPos()
        return Point(x=int(x), y=int(y))

    def list_windows(self, *, limit: int) -> list[WindowInfo]:
        wrappers = self._desktop.windows(visible_only=True, top_level_only=True)
        active_handle = self._active_handle()
        # One overflow record is a sentinel used by the runtime to report
        # that the configured result cap truncated the list.
        return [
            self._window_info(wrapper, focused=self._handle(wrapper) == active_handle)
            for wrapper in wrappers[: limit + 1]
        ]

    def get_active_window(self) -> WindowInfo | None:
        try:
            wrapper = self._desktop.get_active()
        except Exception:
            return None
        if wrapper is None:
            return None
        return self._window_info(wrapper, focused=True)

    def focus_window(self, identifier: str) -> None:
        wrapper = self._find_visible_window(identifier)
        wrapper.set_focus()

    def inspect_ui(self, window_identifier: str | None, *, limit: int) -> list[UIElement]:
        if window_identifier is None:
            active = self._desktop.get_active()
            if active is None:
                return []
            window = active
        else:
            window = self._find_visible_window(window_identifier)

        window_id = self._identifier(self._handle(window))
        controls = islice(window.iter_descendants(), limit + 1)
        items: list[UIElement] = []
        # Consume at most one overflow sentinel so the runtime can accurately
        # mark truncation without materializing an unbounded UI tree.
        for index, control in enumerate(controls):
            info = control.element_info
            automation_id = self._optional_text(info, "automation_id", 128)
            sensitive = self._optional_bool(info, "is_password") is True
            rect = self._optional_bounds(info)
            identifier = automation_id or f"{window_id}:uia:{index}"
            name = "[sensitive control]" if sensitive else self._optional_text(info, "name", 256)
            items.append(
                UIElement(
                    element_id=identifier,
                    role=self._optional_text(info, "control_type", 64) or "Unknown",
                    name=name,
                    bounds=rect,
                    enabled=self._optional_bool(info, "enabled"),
                    visible=self._optional_bool(info, "visible"),
                    focused=self._optional_bool(info, "has_keyboard_focus"),
                    selected=self._optional_bool(info, "is_selected"),
                    automation_id=automation_id or None,
                    sensitive=sensitive,
                )
            )
        return items

    def select_ui_element(self, window_identifier: str, automation_id: str) -> None:
        window = self._find_visible_window(window_identifier)
        control = window.child_window(auto_id=automation_id).wrapper_object()
        if self._optional_bool(control, "is_enabled") is False:
            raise ComputerValidationError("element_disabled", "target UI element is disabled")
        if self._optional_bool(control, "is_visible") is False:
            raise ComputerValidationError("element_not_visible", "target UI element is not visible")
        select = getattr(control, "select", None)
        if not callable(select):
            raise ComputerValidationError(
                "element_not_selectable", "target UI element does not support selection"
            )
        select()

    def screenshot(self, *, max_bytes: int) -> ScreenshotObservation:
        image = self._image_grab.grab()
        timestamp = datetime.now(UTC)
        current = image.convert("RGB")
        for _ in range(10):
            buffer = io.BytesIO()
            current.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
            if len(payload) <= max_bytes:
                return ScreenshotObservation(
                    width=current.width,
                    height=current.height,
                    timestamp=timestamp,
                    source="primary_display",
                    payload=payload,
                )
            if current.width <= 64 or current.height <= 64:
                break
            current = current.resize(
                (max(1, current.width // 2), max(1, current.height // 2)),
                resample=self._image_resampling_lanczos(),
            )
        raise ComputerValidationError(
            "screenshot_too_large", "screenshot cannot be reduced under the configured byte limit"
        )

    def move_mouse(self, point: Point, *, duration_s: float) -> None:
        if duration_s <= 0:
            self._mouse.move(coords=(point.x, point.y))
            return
        start_x, start_y = self._win32api.GetCursorPos()
        steps = max(1, min(20, math.ceil(duration_s / 0.05)))
        pause = duration_s / steps
        for step in range(1, steps + 1):
            fraction = step / steps
            x = round(start_x + (point.x - start_x) * fraction)
            y = round(start_y + (point.y - start_y) * fraction)
            self._win32api.SetCursorPos((x, y))
            time.sleep(pause)

    def click(self, point: Point, *, button: MouseButton) -> None:
        self._mouse.click(button=button.value, coords=(point.x, point.y))

    def double_click(self, point: Point, *, button: MouseButton) -> None:
        self._mouse.double_click(button=button.value, coords=(point.x, point.y))

    def press_key(self, key: KeyboardKey) -> None:
        self._keyboard.send_keys(
            self._key_token(key),
            pause=0,
            with_spaces=True,
            with_tabs=True,
            with_newlines=True,
            vk_packet=True,
        )

    def hotkey(self, modifiers: Sequence[KeyboardModifier], key: KeyboardKey) -> None:
        prefixes = {
            KeyboardModifier.CTRL: "^",
            KeyboardModifier.SHIFT: "+",
        }
        sequence = "".join(prefixes[modifier] for modifier in modifiers)
        sequence += self._key_token(key)
        self._keyboard.send_keys(
            sequence,
            pause=0,
            with_spaces=True,
            with_tabs=True,
            vk_packet=True,
        )

    def type_text(self, text: str) -> None:
        escaped = self._escape_text(text)
        self._keyboard.send_keys(
            escaped,
            pause=0,
            with_spaces=True,
            with_tabs=True,
            with_newlines=True,
            vk_packet=True,
        )

    def _find_visible_window(self, identifier: str) -> Any:
        handle = self._parse_identifier(identifier)
        for wrapper in self._desktop.windows(visible_only=True, top_level_only=True):
            if self._handle(wrapper) == handle:
                return wrapper
        raise ComputerValidationError("window_not_found", "target window is not visible")

    def _window_info(self, wrapper: Any, *, focused: bool) -> WindowInfo:
        info = wrapper.element_info
        handle = self._handle(wrapper)
        return WindowInfo(
            identifier=self._identifier(handle),
            title=self._optional_text(info, "name", 512),
            application=self._optional_text(info, "class_name", 128) or None,
            bounds=self._required_bounds(info),
            visible=self._optional_bool(wrapper, "is_visible") is True,
            focused=focused,
            minimized=self._optional_bool(wrapper, "is_minimized"),
            maximized=self._optional_bool(wrapper, "is_maximized"),
        )

    def _active_handle(self) -> int | None:
        try:
            active = self._desktop.get_active()
        except Exception:
            return None
        return self._handle(active) if active is not None else None

    @staticmethod
    def _handle(wrapper: Any) -> int:
        handle = getattr(wrapper.element_info, "handle", None)
        if not isinstance(handle, int) or handle <= 0:
            raise ComputerValidationError("invalid_window", "window handle is unavailable")
        return handle

    @staticmethod
    def _identifier(handle: int) -> str:
        return f"hwnd:{handle:x}"

    @staticmethod
    def _parse_identifier(identifier: str) -> int:
        prefix, separator, value = identifier.partition(":")
        if prefix != "hwnd" or not separator or not value or len(value) > 16:
            raise ComputerValidationError("invalid_window", "window identifier is invalid")
        try:
            handle = int(value, 16)
        except ValueError as exc:
            raise ComputerValidationError("invalid_window", "window identifier is invalid") from exc
        if handle <= 0:
            raise ComputerValidationError("invalid_window", "window identifier is invalid")
        return handle

    @staticmethod
    def _optional_text(target: Any, attribute: str, max_length: int) -> str:
        try:
            value = getattr(target, attribute)
            if callable(value):
                value = value()
        except Exception:
            return ""
        if not isinstance(value, str):
            return ""
        return value[:max_length]

    @staticmethod
    def _optional_bool(target: Any, attribute: str) -> bool | None:
        try:
            value = getattr(target, attribute)
            if callable(value):
                value = value()
        except Exception:
            return None
        return value if isinstance(value, bool) else None

    @classmethod
    def _optional_bounds(cls, target: Any) -> Bounds | None:
        try:
            rectangle = target.rectangle
            if callable(rectangle):
                rectangle = rectangle()
            return Bounds(
                left=int(rectangle.left),
                top=int(rectangle.top),
                right=int(rectangle.right),
                bottom=int(rectangle.bottom),
            )
        except Exception:
            return None

    @classmethod
    def _required_bounds(cls, target: Any) -> Bounds:
        bounds = cls._optional_bounds(target)
        if bounds is None:
            raise ComputerValidationError("invalid_window", "window bounds are unavailable")
        return bounds

    def _system_dpi(self) -> float | None:
        function = getattr(self._win32api, "GetDpiForSystem", None)
        if not callable(function):
            return None
        try:
            dpi = function()
        except Exception:
            return None
        if isinstance(dpi, int) and dpi > 0:
            return float(dpi)
        return None

    @staticmethod
    def _key_token(key: KeyboardKey) -> str:
        if key in _KEY_TOKENS:
            return _KEY_TOKENS[key]
        return key.value.lower()

    @staticmethod
    def _escape_text(text: str) -> str:
        escaped: list[str] = []
        for char in text:
            if char in "^+%~()[]":
                escaped.append("{" + char + "}")
            elif char == "{":
                escaped.append("{{}")
            elif char == "}":
                escaped.append("{}}")
            else:
                escaped.append(char)
        return "".join(escaped)

    @staticmethod
    def _image_resampling_lanczos() -> Any:
        from PIL import Image

        return Image.Resampling.LANCZOS
