"""Explicit voice serialization projections; raw audio is never serialized."""

from __future__ import annotations

from typing import Any

from .models import (
    AudioMetadata,
    SynthesisResult,
    TranscriptionResult,
    VoiceObservation,
    VoiceResponse,
)


def audio_metadata_to_dict(metadata: AudioMetadata) -> dict[str, Any]:
    """Serialize audio metadata without any audio payload."""
    return metadata.model_dump(mode="json", exclude_none=True)


def transcription_to_dict(result: TranscriptionResult) -> dict[str, Any]:
    """Serialize a bounded transcript and its validated quality metadata."""
    return result.model_dump(mode="json", exclude_none=True)


def synthesis_result_to_dict(result: SynthesisResult) -> dict[str, Any]:
    """Serialize synthesis metadata only; omit opaque references and audio bytes."""
    return result.model_dump(mode="json", exclude_none=True)


def observation_to_dict(observation: VoiceObservation) -> dict[str, Any]:
    """Serialize a voice observation; it contains no raw microphone buffer."""
    return observation.model_dump(mode="json", exclude_none=True)


def response_to_dict(response: VoiceResponse) -> dict[str, Any]:
    """Serialize response text and metadata without raw input/output audio."""
    return response.model_dump(mode="json", exclude_none=True)
