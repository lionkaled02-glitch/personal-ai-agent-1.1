"""Strict, bounded models for provider-neutral voice input and output.

Audio bytes are in-process-only values. Payload fields are hidden from repr
and model serialization; event serializers use explicit metadata projections.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_AUDIO_DURATION_S = 300.0
MAX_VOICE_TEXT_CHARS = 4_000
MAX_LANGUAGE_CHARS = 35
MAX_VOICE_NAME_CHARS = 64
MAX_SYNTHESIS_TEXT_CHARS = 2_000
MAX_PROVIDER_METADATA_ITEMS = 16
MAX_PROVIDER_METADATA_KEY_CHARS = 32
MAX_PROVIDER_METADATA_VALUE_CHARS = 128

_LANGUAGE_PATTERN = r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$"
_VOICE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$"
_AUDIO_REFERENCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_METADATA_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SECRET_METADATA_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:api_?key|token|secret|password|credential|authorization)(?:_|$)", re.I
)


class VoiceModel(BaseModel):
    """Strict immutable base for public voice data models."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class AudioEncoding(StrEnum):
    """Supported uncompressed PCM encodings; containers/codecs are out of scope."""

    PCM_U8 = "pcm_u8"
    PCM_S16LE = "pcm_s16le"
    PCM_S24LE = "pcm_s24le"
    PCM_S32LE = "pcm_s32le"
    PCM_F32LE = "pcm_f32le"


class AudioFormat(VoiceModel):
    """Bounded raw PCM format description."""

    sample_rate: int = Field(gt=0, le=192_000, strict=True)
    channels: int = Field(ge=1, le=8, strict=True)
    sample_width: int = Field(ge=1, le=4, strict=True)
    encoding: AudioEncoding

    @model_validator(mode="after")
    def validate_sample_width(self) -> AudioFormat:
        expected_width = {
            AudioEncoding.PCM_U8: 1,
            AudioEncoding.PCM_S16LE: 2,
            AudioEncoding.PCM_S24LE: 3,
            AudioEncoding.PCM_S32LE: 4,
            AudioEncoding.PCM_F32LE: 4,
        }[self.encoding]
        if self.sample_width != expected_width:
            raise ValueError("sample width does not match the declared PCM encoding")
        return self


class AudioMetadata(VoiceModel):
    """Non-payload metadata for one bounded audio buffer."""

    duration: float = Field(gt=0.0, le=MAX_AUDIO_DURATION_S, allow_inf_nan=False, strict=True)
    format: AudioFormat
    byte_size: int = Field(ge=1, le=MAX_AUDIO_BYTES, strict=True)

    @model_validator(mode="after")
    def validate_duration_matches_pcm_size(self) -> AudioMetadata:
        frame_bytes = self.format.channels * self.format.sample_width
        if self.byte_size % frame_bytes:
            raise ValueError("audio byte size contains a partial PCM frame")
        measured_duration = self.byte_size / (
            self.format.sample_rate * self.format.channels * self.format.sample_width
        )
        if abs(measured_duration - self.duration) > 1.0 / self.format.sample_rate:
            raise ValueError("audio duration does not match its PCM metadata")
        return self


class AudioInput(VoiceModel):
    """Ephemeral audio buffer paired with validated metadata.

    This model deliberately has no path, URL, persistence, or upload field.
    """

    metadata: AudioMetadata
    payload: bytes = Field(
        min_length=1,
        max_length=MAX_AUDIO_BYTES,
        strict=True,
        repr=False,
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_payload_size(self) -> AudioInput:
        if len(self.payload) != self.metadata.byte_size:
            raise ValueError("audio payload length does not match its metadata")
        return self


class TranscriptionStatus(StrEnum):
    """Provider confidence state; uncertainty must never be treated as success."""

    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"


class TranscriptionResult(VoiceModel):
    """Bounded transcription text and provider-neutral quality metadata."""

    text: str = Field(max_length=MAX_VOICE_TEXT_CHARS, strict=True)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    language: str | None = Field(
        default=None,
        max_length=MAX_LANGUAGE_CHARS,
        pattern=_LANGUAGE_PATTERN,
        strict=True,
    )
    duration: float = Field(gt=0.0, le=MAX_AUDIO_DURATION_S, allow_inf_nan=False, strict=True)
    status: TranscriptionStatus = TranscriptionStatus.CONFIDENT
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        max_length=MAX_PROVIDER_METADATA_ITEMS,
    )

    @model_validator(mode="after")
    def validate_confidence_and_metadata(self) -> TranscriptionResult:
        if self.status is TranscriptionStatus.CONFIDENT and self.confidence is None:
            raise ValueError("a confident transcription requires a confidence score")
        for key, value in self.provider_metadata.items():
            if (
                len(key) > MAX_PROVIDER_METADATA_KEY_CHARS
                or _METADATA_KEY_PATTERN.fullmatch(key) is None
                or _SECRET_METADATA_KEY_PATTERN.search(key) is not None
            ):
                raise ValueError("provider metadata key is invalid")
            if isinstance(value, str) and len(value) > MAX_PROVIDER_METADATA_VALUE_CHARS:
                raise ValueError("provider metadata string exceeds its bound")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("provider metadata numbers must be finite")
        return self


class SynthesisRequest(VoiceModel):
    """Validated speech output options; text is still ordinary agent output."""

    text: str = Field(min_length=1, max_length=MAX_SYNTHESIS_TEXT_CHARS, strict=True)
    language: str | None = Field(
        default=None,
        max_length=MAX_LANGUAGE_CHARS,
        pattern=_LANGUAGE_PATTERN,
        strict=True,
    )
    voice: str | None = Field(
        default=None,
        max_length=MAX_VOICE_NAME_CHARS,
        pattern=_VOICE_NAME_PATTERN,
        strict=True,
    )
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False, strict=True)
    pitch: float = Field(default=0.0, ge=-12.0, le=12.0, allow_inf_nan=False, strict=True)


class SynthesisResult(VoiceModel):
    """Bounded synthesized audio or an opaque, non-path reference.

    The payload is never shown in repr or serialized by Pydantic. A reference,
    when used by a future provider, is an opaque token rather than a path/URL.
    """

    audio_metadata: AudioMetadata
    duration: float = Field(gt=0.0, le=MAX_AUDIO_DURATION_S, allow_inf_nan=False, strict=True)
    payload: bytes | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_AUDIO_BYTES,
        strict=True,
        repr=False,
        exclude=True,
    )
    reference: str | None = Field(
        default=None,
        pattern=_AUDIO_REFERENCE_PATTERN,
        strict=True,
        repr=False,
        exclude=True,
    )
    provider_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        max_length=MAX_PROVIDER_METADATA_ITEMS,
    )

    @model_validator(mode="after")
    def validate_output(self) -> SynthesisResult:
        if (self.payload is None) == (self.reference is None):
            raise ValueError("synthesis result requires exactly one bounded payload or reference")
        if self.payload is not None and len(self.payload) != self.audio_metadata.byte_size:
            raise ValueError("synthesized payload length does not match its metadata")
        if not math.isclose(self.duration, self.audio_metadata.duration, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("synthesis duration does not match audio metadata")
        for key, value in self.provider_metadata.items():
            if (
                len(key) > MAX_PROVIDER_METADATA_KEY_CHARS
                or _METADATA_KEY_PATTERN.fullmatch(key) is None
                or _SECRET_METADATA_KEY_PATTERN.search(key) is not None
            ):
                raise ValueError("provider metadata key is invalid")
            if isinstance(value, str) and len(value) > MAX_PROVIDER_METADATA_VALUE_CHARS:
                raise ValueError("provider metadata string exceeds its bound")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("provider metadata numbers must be finite")
        return self


class VoiceObservation(VoiceModel):
    """Metadata and transcription for one ephemeral input operation."""

    transcription: TranscriptionResult
    audio_metadata: AudioMetadata | None = None
    started_at: datetime = Field(strict=True)
    completed_at: datetime = Field(strict=True)

    @model_validator(mode="after")
    def validate_timestamps(self) -> VoiceObservation:
        for value in (self.started_at, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("voice observation timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("voice observation completion precedes its start")
        return self


class VoiceResponseStatus(StrEnum):
    """Result of the voice transport, separate from the agent's task state."""

    RESPONDED = "RESPONDED"
    NO_RESPONSE = "NO_RESPONSE"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"


class VoiceResponse(VoiceModel):
    """Bounded response for a voice exchange; any audio bytes remain ephemeral."""

    response_text: str = Field(default="", max_length=MAX_VOICE_TEXT_CHARS, strict=True)
    synthesis_result: SynthesisResult | None = None
    status: VoiceResponseStatus
    observation: VoiceObservation | None = None
    task_id: str | None = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9-]{1,64}$", strict=True
    )
    error_code: str | None = Field(
        default=None, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$", strict=True
    )
    error_message: str | None = Field(default=None, max_length=256, strict=True)

    @model_validator(mode="after")
    def validate_uncertainty(self) -> VoiceResponse:
        if self.status is VoiceResponseStatus.UNCERTAIN and (
            self.observation is None
            or self.observation.transcription.status is not TranscriptionStatus.UNCERTAIN
        ):
            raise ValueError("an uncertain voice response requires an uncertain transcription")
        return self
