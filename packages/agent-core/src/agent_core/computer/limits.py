"""Conservative, validated bounds for computer observations and actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Settings
from .models import HARD_MAX_SCREENSHOT_BYTES, HARD_MAX_TEXT_INPUT_CHARS


@dataclass(frozen=True)
class ComputerLimits:
    """Runtime limits; environment configuration cannot exceed hard caps."""

    max_actions_per_task: int = 20
    action_timeout_s: float = 5.0
    max_text_input_chars: int = 256
    max_screenshot_bytes: int = 1_048_576
    max_windows: int = 50
    max_ui_elements: int = 100
    max_retries: int = 1
    mouse_move_duration_s: float = 0.5
    cursor_tolerance_px: int = 2

    def __post_init__(self) -> None:
        integer_bounds = {
            "max_actions_per_task": (1, 500),
            "max_text_input_chars": (1, HARD_MAX_TEXT_INPUT_CHARS),
            "max_screenshot_bytes": (1, HARD_MAX_SCREENSHOT_BYTES),
            "max_windows": (1, 500),
            "max_ui_elements": (1, 1_000),
            "max_retries": (0, 3),
            "cursor_tolerance_px": (0, 10),
        }
        for name, (minimum, maximum) in integer_bounds.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")

        if (
            not isinstance(self.action_timeout_s, (int, float))
            or isinstance(self.action_timeout_s, bool)
            or not math.isfinite(self.action_timeout_s)
            or not 0.1 <= self.action_timeout_s <= 30.0
        ):
            raise ValueError("action_timeout_s must be finite and between 0.1 and 30 seconds")
        if (
            not isinstance(self.mouse_move_duration_s, (int, float))
            or isinstance(self.mouse_move_duration_s, bool)
            or not math.isfinite(self.mouse_move_duration_s)
            or not 0.0 <= self.mouse_move_duration_s <= 2.0
        ):
            raise ValueError("mouse_move_duration_s must be finite and between 0 and 2 seconds")

    @classmethod
    def from_settings(cls, settings: Settings) -> ComputerLimits:
        """Build validated computer limits from the shared Settings object."""
        return cls(
            max_actions_per_task=settings.computer_max_actions_per_task,
            action_timeout_s=settings.computer_action_timeout_s,
            max_text_input_chars=settings.computer_max_text_input_chars,
            max_screenshot_bytes=settings.computer_max_screenshot_bytes,
            max_windows=settings.computer_max_windows,
            max_ui_elements=settings.computer_max_ui_elements,
            max_retries=settings.computer_max_retries,
            mouse_move_duration_s=settings.computer_mouse_move_duration_s,
            cursor_tolerance_px=settings.computer_cursor_tolerance_px,
        )
