"""Explicit JSON projection for computer observations and action results."""

from __future__ import annotations

import base64
from typing import Any

from .models import (
    ComputerActionResult,
    ComputerObservation,
    ComputerOperationResult,
    ScreenshotObservation,
)


def screenshot_to_dict(
    screenshot: ScreenshotObservation, *, include_payload: bool
) -> dict[str, object]:
    """Serialize screenshot metadata and optionally base64 bytes for the caller.

    The payload is intentionally excluded from event payloads and action
    history. Only the explicit screenshot tool includes it.
    """
    value: dict[str, object] = {
        "width": screenshot.width,
        "height": screenshot.height,
        "timestamp": screenshot.timestamp.isoformat(),
        "source": screenshot.source,
        "media_type": screenshot.media_type,
        "payload_bytes": screenshot.payload_bytes,
        "payload_sha256": screenshot.payload_sha256,
    }
    if include_payload and screenshot.payload is not None:
        value["image_base64"] = base64.b64encode(screenshot.payload).decode("ascii")
    return value


def observation_to_dict(
    observation: ComputerObservation, *, include_screenshot_payload: bool = False
) -> dict[str, Any]:
    """Serialize an observation without exposing provider-specific objects."""
    value = observation.model_dump(mode="json", exclude={"screenshot"}, exclude_none=True)
    if observation.screenshot is not None:
        value["screenshot"] = screenshot_to_dict(
            observation.screenshot, include_payload=include_screenshot_payload
        )
    return value


def operation_result_to_dict(
    result: ComputerOperationResult, *, include_screenshot_payload: bool = False
) -> dict[str, Any]:
    """Serialize one read-only operation, including structured failure data."""
    value: dict[str, Any] = {"ok": result.ok, "operation": result.operation}
    if result.error_code is not None:
        value["error_code"] = result.error_code
    if result.error is not None:
        value["error"] = result.error
    if result.observation is not None:
        value["observation"] = observation_to_dict(
            result.observation,
            include_screenshot_payload=include_screenshot_payload,
        )
    return value


def action_result_to_dict(result: ComputerActionResult) -> dict[str, Any]:
    """Serialize a structured action result; action observations hold no bytes."""
    value = result.model_dump(mode="json", exclude_none=True)
    value["risk_level"] = result.risk_level.name
    return value
