"""Shared serializer and validation helpers for computer tools."""

from __future__ import annotations

from typing import Any

from ..computer.models import (
    ComputerActionResult,
    ComputerOperationResult,
    VerificationCondition,
)
from ..computer.runtime import classify_computer_operation
from ..computer.serialization import action_result_to_dict, operation_result_to_dict
from ..tools import ToolResult
from ..vision.models import VisualVerificationCondition, VisualVerificationKind

VISUAL_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": [kind.value for kind in VisualVerificationKind],
        },
        "region": {
            "type": "object",
            "required": ["x", "y", "width", "height"],
            "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 8192},
                "y": {"type": "integer", "minimum": 0, "maximum": 8192},
                "width": {"type": "integer", "minimum": 1, "maximum": 8192},
                "height": {"type": "integer", "minimum": 1, "maximum": 8192},
            },
            "additionalProperties": False,
        },
        "minimum_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "minimum_change_ratio": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
        },
    },
    "additionalProperties": False,
}

ACTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "action_id",
        "action",
        "risk_level",
        "status",
        "attempted",
        "success",
        "provider_completed",
        "verified",
        "attempts",
        "verification",
        "recovery",
    ],
    "properties": {
        "action_id": {"type": "string"},
        "action": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "status": {
            "type": "string",
            "enum": [
                "verified",
                "unverified",
                "verification_failed",
                "denied",
                "failed",
                "timed_out",
                "invalid",
            ],
        },
        "attempted": {"type": "boolean"},
        "success": {"type": "boolean"},
        "provider_completed": {"type": "boolean"},
        "verified": {"type": "boolean"},
        "attempts": {"type": "integer"},
        "observed_before": {"type": "object"},
        "observed_after": {"type": "object"},
        "recovery_observations": {"type": "array", "items": {"type": "object"}},
        "verification": {
            "type": "object",
            "required": ["status", "reason", "expected", "observed"],
            "properties": {
                "status": {"type": "string", "enum": ["passed", "failed", "not_run"]},
                "condition": {"type": "string"},
                "reason": {"type": "string"},
                "expected": {"type": "object"},
                "observed": {"type": "object"},
            },
        },
        "visual_verification": {
            "type": "object",
            "required": ["condition", "status", "method", "reason"],
            "properties": {
                "condition": VISUAL_VERIFICATION_SCHEMA,
                "status": {"type": "string", "enum": ["verified", "failed", "uncertain"]},
                "method": {"type": "string"},
                "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
                "evidence_ref": {"type": "string"},
                "match": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "error_code": {"type": "string"},
        "error": {"type": "string"},
        "retryable": {"type": "boolean"},
        "recovery": {
            "type": "object",
            "required": ["action", "reason", "retries_used", "retries_remaining"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "retry",
                        "refresh_observation",
                        "requery_active_window",
                        "requery_ui_element",
                        "stop",
                    ],
                },
                "reason": {"type": "string"},
                "retries_used": {"type": "integer"},
                "retries_remaining": {"type": "integer"},
            },
        },
    },
}

OBSERVATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ok", "operation"],
    "properties": {
        "ok": {"type": "boolean"},
        "operation": {"type": "string"},
        "observation": {"type": "object"},
        "error_code": {"type": "string"},
        "error": {"type": "string"},
    },
}


def has_only_keys(input: dict[str, Any], allowed: set[str]) -> bool:
    """Reject undeclared tool inputs rather than silently ignoring them."""
    return set(input).issubset(allowed)


def parse_verification(input: dict[str, Any]) -> VerificationCondition | None:
    """Parse an optional explicit verification condition without echoing data."""
    raw = input.get("verification")
    if raw is None:
        return None
    return VerificationCondition.model_validate(raw)


def parse_visual_verification(
    input: dict[str, Any],
) -> VisualVerificationCondition | None:
    """Parse one strict pixel-level condition without echoing its contents."""
    raw = input.get("visual_verification")
    if raw is None:
        return None
    return VisualVerificationCondition.model_validate(raw)


def action_tool_result(result: ComputerActionResult) -> ToolResult:
    """Return the structured outcome; ``success`` is the verified action flag."""
    return ToolResult(ok=True, output=action_result_to_dict(result))


def observation_tool_result(
    result: ComputerOperationResult, *, include_screenshot_payload: bool = False
) -> ToolResult:
    """Return the structured observation/error result to the explicit caller."""
    return ToolResult(
        ok=True,
        output=operation_result_to_dict(
            result, include_screenshot_payload=include_screenshot_payload
        ),
    )


def invalid_input_result(tool_name: str) -> ToolResult:
    """Sanitize Pydantic errors (never echo submitted text or model input)."""
    return ToolResult(
        ok=True,
        output={
            "action_id": "invalid",
            "action": tool_name.removeprefix("computer_"),
            "risk_level": classify_computer_operation(tool_name).name,
            "status": "invalid",
            "attempted": False,
            "success": False,
            "provider_completed": False,
            "verified": False,
            "attempts": 0,
            "verification": {
                "status": "not_run",
                "reason": "action_validation_failed",
                "expected": {},
                "observed": {},
            },
            "error_code": "invalid_action",
            "error": "computer action input failed validation",
            "retryable": False,
            "recovery": {
                "action": "stop",
                "reason": "invalid_action_must_not_be_retried",
                "retries_used": 0,
                "retries_remaining": 0,
            },
        },
    )
