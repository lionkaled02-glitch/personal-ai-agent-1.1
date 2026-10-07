"""Provider-neutral contracts for desktop observation and primitive input."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import (
    KeyboardKey,
    KeyboardModifier,
    MouseButton,
    Point,
    ScreenInfo,
    ScreenshotObservation,
    UIElement,
    WindowInfo,
)


class ComputerProvider(Protocol):
    """Boundary implemented by a platform adapter or a deterministic fake.

    Providers expose only bounded observations and explicit input primitives.
    They must not execute commands, launch processes, access the network, or
    persist screenshots. The runtime validates and verifies all operations.
    """

    def get_screen_info(self) -> ScreenInfo: ...

    def get_cursor_position(self) -> Point: ...

    def list_windows(self, *, limit: int) -> Sequence[WindowInfo]: ...

    def get_active_window(self) -> WindowInfo | None: ...

    def focus_window(self, identifier: str) -> None: ...

    def inspect_ui(self, window_identifier: str | None, *, limit: int) -> Sequence[UIElement]: ...

    def select_ui_element(self, window_identifier: str, automation_id: str) -> None: ...

    def screenshot(self, *, max_bytes: int) -> ScreenshotObservation: ...

    def move_mouse(self, point: Point, *, duration_s: float) -> None: ...

    def click(self, point: Point, *, button: MouseButton) -> None: ...

    def double_click(self, point: Point, *, button: MouseButton) -> None: ...

    def press_key(self, key: KeyboardKey) -> None: ...

    def hotkey(self, modifiers: Sequence[KeyboardModifier], key: KeyboardKey) -> None: ...

    def type_text(self, text: str) -> None: ...
