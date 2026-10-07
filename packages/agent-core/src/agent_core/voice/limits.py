"""Hard-clamped voice input/output and operation limits."""

from __future__ import annotations

from pydantic import Field

from ..config import Settings
from .models import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_S,
    MAX_LANGUAGE_CHARS,
    MAX_SYNTHESIS_TEXT_CHARS,
    MAX_VOICE_NAME_CHARS,
    MAX_VOICE_TEXT_CHARS,
    VoiceModel,
)


class VoiceLimits(VoiceModel):
    """Configurable limits that may be tightened, never widened past hard caps."""

    max_audio_bytes: int = Field(default=1_048_576, ge=1, le=MAX_AUDIO_BYTES, strict=True)
    max_duration_s: float = Field(
        default=30.0,
        gt=0.0,
        le=MAX_AUDIO_DURATION_S,
        allow_inf_nan=False,
        strict=True,
    )
    max_text_chars: int = Field(
        default=MAX_VOICE_TEXT_CHARS,
        ge=1,
        le=MAX_VOICE_TEXT_CHARS,
        strict=True,
    )
    max_language_chars: int = Field(
        default=MAX_LANGUAGE_CHARS,
        ge=2,
        le=MAX_LANGUAGE_CHARS,
        strict=True,
    )
    max_voice_name_chars: int = Field(
        default=MAX_VOICE_NAME_CHARS,
        ge=1,
        le=MAX_VOICE_NAME_CHARS,
        strict=True,
    )
    max_synthesis_text_chars: int = Field(
        default=MAX_SYNTHESIS_TEXT_CHARS,
        ge=1,
        le=MAX_SYNTHESIS_TEXT_CHARS,
        strict=True,
    )
    max_transcription_time_s: float = Field(
        default=15.0,
        gt=0.0,
        le=60.0,
        allow_inf_nan=False,
        strict=True,
    )
    max_synthesis_time_s: float = Field(
        default=15.0,
        gt=0.0,
        le=60.0,
        allow_inf_nan=False,
        strict=True,
    )
    max_retries: int = Field(default=1, ge=0, le=3, strict=True)
    minimum_transcription_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        strict=True,
    )

    @classmethod
    def from_settings(cls, settings: Settings) -> VoiceLimits:
        """Build validated voice limits from shared non-secret settings."""
        return cls(
            max_audio_bytes=settings.voice_max_audio_bytes,
            max_duration_s=settings.voice_max_duration_s,
            max_text_chars=settings.voice_max_text_chars,
            max_language_chars=settings.voice_max_language_chars,
            max_voice_name_chars=settings.voice_max_voice_name_chars,
            max_synthesis_text_chars=settings.voice_max_synthesis_text_chars,
            max_transcription_time_s=settings.voice_max_transcription_time_s,
            max_synthesis_time_s=settings.voice_max_synthesis_time_s,
            max_retries=settings.voice_max_retries,
            minimum_transcription_confidence=settings.voice_min_transcription_confidence,
        )
