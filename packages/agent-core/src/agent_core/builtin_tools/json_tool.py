"""JSON utility tool: safely parse/validate JSON text.

Uses the stdlib ``json`` module only (no code execution of any kind).
Input is bounded to :data:`MAX_JSON_LENGTH` characters. Invalid JSON is a
*successful, structured answer* (``valid: false`` with a bounded error
message), not a tool fault — validating bad input is the tool's job.
"""

from __future__ import annotations

import json
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec

JSON_TOOL_NAME = "json_utils"

#: Maximum accepted JSON text length (rejects oversized model output).
MAX_JSON_LENGTH = 100_000

_TYPE_NAMES = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "number",
    float: "number",
    type(None): "null",
}


def _describe(value: Any) -> tuple[str, int]:
    """(type name, top-level size) of a parsed JSON value."""
    if isinstance(value, (dict, list)):
        return _TYPE_NAMES[type(value)], len(value)
    return _TYPE_NAMES[type(value)], 0


class JsonUtilsTool:
    """Parses bounded JSON text and reports its shape (deterministic, LOW)."""

    spec = ToolSpec(
        name=JSON_TOOL_NAME,
        description=(
            "Validates/parses a JSON document and returns whether it is valid, "
            "its top-level JSON type, and its top-level size."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "json_text": {
                    "type": "string",
                    "description": "The JSON document to validate/parse.",
                },
            },
            "required": ["json_text"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "valid": {"type": "boolean"},
                "type": {"type": "string"},
                "count": {"type": "integer"},
                "error": {"type": "string"},
            },
            "required": ["valid", "type", "count", "error"],
        },
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        json_text = str(input["json_text"])
        if len(json_text) > MAX_JSON_LENGTH:
            return ToolResult(
                ok=False,
                error=f"json text too long ({len(json_text)} > {MAX_JSON_LENGTH} chars)",
                error_code="input_too_long",
            )
        try:
            value = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return ToolResult(
                ok=True,
                output={
                    "valid": False,
                    "type": "",
                    "count": 0,
                    "error": f"{exc.msg} (at position {exc.pos})",
                },
            )
        type_name, count = _describe(value)
        return ToolResult(
            ok=True,
            output={"valid": True, "type": type_name, "count": count, "error": ""},
        )
