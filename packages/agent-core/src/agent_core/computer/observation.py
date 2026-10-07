"""Bounded normalization of provider observations into public data models."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .errors import ComputerValidationError
from .models import (
    ComputerObservation,
    Point,
    ScreenInfo,
    ScreenshotObservation,
    UIElement,
    WindowInfo,
    safe_external_text,
    stable_identifier,
)


def normalize_screen_info(value: object) -> ScreenInfo:
    """Validate and copy screen metadata without provider-specific objects."""
    return ScreenInfo.model_validate(value)


def normalize_point(value: object) -> Point:
    """Validate and copy a provider cursor position."""
    return Point.model_validate(value)


def normalize_window(value: object) -> WindowInfo:
    """Normalize a top-level window; title/application remain untrusted data."""
    raw: dict[str, Any]
    if isinstance(value, WindowInfo):
        raw = value.model_dump(mode="python")
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("provider returned an invalid window record")

    raw["identifier"] = stable_identifier(raw.get("identifier"), max_length=128)
    raw["title"] = safe_external_text(raw.get("title", ""), max_length=512)
    if raw.get("application") is not None:
        raw["application"] = safe_external_text(raw["application"], max_length=128)
    return WindowInfo.model_validate(raw)


def normalize_windows(values: Sequence[object], *, limit: int) -> tuple[list[WindowInfo], bool]:
    """Validate, sanitize, and cap a provider window sequence."""
    truncated = len(values) > limit
    windows = [normalize_window(value) for value in values[:limit]]
    return windows, truncated


def normalize_ui_element(value: object) -> UIElement:
    """Normalize safe accessibility properties; redact sensitive controls."""
    raw: dict[str, Any]
    if isinstance(value, UIElement):
        raw = value.model_dump(mode="python")
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("provider returned an invalid UI element")

    raw["element_id"] = stable_identifier(raw.get("element_id"), max_length=192)
    raw["role"] = safe_external_text(raw.get("role"), max_length=64) or "Unknown"
    raw["name"] = safe_external_text(raw.get("name", ""), max_length=256)
    if raw.get("automation_id") is not None:
        raw["automation_id"] = safe_external_text(raw["automation_id"], max_length=128)
    return UIElement.model_validate(raw)


def normalize_ui_elements(values: Sequence[object], *, limit: int) -> tuple[list[UIElement], bool]:
    """Validate, sanitize, and cap a provider UI element sequence."""
    truncated = len(values) > limit
    elements = [normalize_ui_element(value) for value in values[:limit]]
    return elements, truncated


def normalize_screenshot(value: object, *, max_bytes: int) -> ScreenshotObservation:
    """Validate an ephemeral screenshot and enforce the configured payload cap."""
    screenshot = ScreenshotObservation.model_validate(value)
    if screenshot.payload_bytes > max_bytes:
        raise ComputerValidationError(
            "screenshot_too_large", "screenshot payload exceeds configured byte limit"
        )
    return screenshot


def make_observation(
    *,
    observed_at: datetime,
    screen_info: ScreenInfo | None = None,
    cursor_position: Point | None = None,
    active_window: WindowInfo | None = None,
    windows: list[WindowInfo] | None = None,
    ui_elements: list[UIElement] | None = None,
    screenshot: ScreenshotObservation | None = None,
    windows_truncated: bool = False,
    ui_elements_truncated: bool = False,
) -> ComputerObservation:
    """Create the common explicit observation envelope."""
    return ComputerObservation(
        observed_at=observed_at,
        screen_info=screen_info,
        cursor_position=cursor_position,
        active_window=active_window,
        windows=windows,
        ui_elements=ui_elements,
        screenshot=screenshot,
        windows_truncated=windows_truncated,
        ui_elements_truncated=ui_elements_truncated,
    )
