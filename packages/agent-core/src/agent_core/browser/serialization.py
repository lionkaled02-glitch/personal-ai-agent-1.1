"""Explicit safe serialization for browser observations, results, and events."""

from __future__ import annotations

from typing import Any

from .models import BrowserActionResult, BrowserObservation


def observation_to_dict(observation: BrowserObservation) -> dict[str, Any]:
    """Serialize bounded untrusted page data for a requesting tool/agent."""
    return observation.model_dump(mode="json")


def action_result_to_dict(result: BrowserActionResult) -> dict[str, Any]:
    """Serialize an action receipt; values submitted to forms are not retained."""
    return result.model_dump(mode="json")


def action_event_payload(result: BrowserActionResult) -> dict[str, Any]:
    """Return operational metadata only; never include page text or screenshot data."""
    return {
        "action_id": result.action_id,
        "action": result.action.value,
        "status": result.status.value,
        "permission_level": result.permission_level.name,
        "attempts": result.attempts,
        "verification": result.verification.status.value,
        "verification_reason": result.verification.reason_code,
        "recovery": result.recovery.action.value,
        "error_code": result.error_code,
    }
