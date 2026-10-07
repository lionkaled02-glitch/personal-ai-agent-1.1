"""Text utility tool: small, safe, bounded text operations.

Supported actions: ``length`` (characters), ``word_count``, ``line_count``.
Input is bounded to :data:`MAX_TEXT_LENGTH` characters (structured failure
beyond that). No filesystem, network, or code execution.
"""

from __future__ import annotations

from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec

TEXT_TOOL_NAME = "text_utils"

#: Maximum accepted text length (rejects oversized model output).
MAX_TEXT_LENGTH = 10_000

_ACTIONS = {"length", "word_count", "line_count"}


def _measure(text: str, action: str) -> int:
    if action == "length":
        return len(text)
    if action == "word_count":
        return len(text.split())
    return len(text.splitlines())  # line_count


class TextUtilsTool:
    """Counts characters/words/lines of a bounded text (deterministic, LOW)."""

    spec = ToolSpec(
        name=TEXT_TOOL_NAME,
        description=(
            "Performs a text counting operation (length, word_count, or line_count) "
            "on a bounded string and returns the integer result."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to measure."},
                "action": {
                    "type": "string",
                    "enum": ["length", "word_count", "line_count"],
                    "description": "Which count to compute.",
                },
            },
            "required": ["text", "action"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["action", "value"],
        },
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        action = str(input["action"])
        if action not in _ACTIONS:
            return ToolResult(
                ok=False,
                error=f"unsupported action {action!r}; supported: {sorted(_ACTIONS)}",
                error_code="invalid_action",
            )
        text = str(input["text"])
        if len(text) > MAX_TEXT_LENGTH:
            return ToolResult(
                ok=False,
                error=f"text too long ({len(text)} > {MAX_TEXT_LENGTH} chars)",
                error_code="input_too_long",
            )
        return ToolResult(ok=True, output={"action": action, "value": _measure(text, action)})
