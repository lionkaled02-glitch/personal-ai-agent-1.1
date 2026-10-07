"""Optional, narrowly-scoped normalization tool; no audio tool is exposed."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..permissions import PermissionLevel
from ..tools import ToolRegistry, ToolResult, ToolSpec
from .errors import VoiceError
from .limits import VoiceLimits
from .models import MAX_VOICE_TEXT_CHARS
from .normalization import normalize_transcription_text

VOICE_NORMALIZE_TOOL_NAME = "voice_normalize"
VOICE_TOOL_NAMES: tuple[str, ...] = (VOICE_NORMALIZE_TOOL_NAME,)


class _NormalizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(min_length=1, max_length=MAX_VOICE_TEXT_CHARS)


class VoiceNormalizeTool:
    """Normalize bounded text only; accepts no audio, path, or provider input."""

    def __init__(self, limits: VoiceLimits | None = None) -> None:
        self._limits = limits or VoiceLimits()
        self.spec = ToolSpec(
            name=VOICE_NORMALIZE_TOOL_NAME,
            description=(
                "Normalize whitespace and Unicode NFC in bounded user text. "
                "This has no audio, storage, network, or execution capability."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": self._limits.max_text_chars,
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": self._limits.max_text_chars}
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            permission_level=PermissionLevel.LOW,
            deterministic=True,
            sensitive_input=True,
            sensitive_output=True,
        )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            validated = _NormalizeInput.model_validate(input)
            normalized = normalize_transcription_text(
                validated.text,
                max_chars=self._limits.max_text_chars,
            )
        except ValidationError:
            return ToolResult(
                ok=False,
                error="voice normalization input failed validation",
                error_code="invalid_input",
            )
        except VoiceError as exc:
            return ToolResult(
                ok=False,
                error=exc.public_message,
                error_code=exc.code,
            )
        return ToolResult(ok=True, output={"text": normalized})


def register_voice_tools(
    registry: ToolRegistry,
    limits: VoiceLimits | None = None,
) -> None:
    """Opt-in registration of the pure, LOW-permission text normalizer."""
    registry.register(VoiceNormalizeTool(limits))
