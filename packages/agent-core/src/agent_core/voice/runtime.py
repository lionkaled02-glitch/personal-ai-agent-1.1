"""Provider-neutral synchronous runtime for ephemeral voice transport.

The runtime performs one STT operation, normalizes the resulting text, and
hands it to the existing Agent.run path exactly once. It never interprets
transcription as code or grants permissions; the normal planner, executor,
and PermissionManager remain authoritative. Provider time limits are
cooperative because synchronous calls cannot be forcibly interrupted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..events import Clock, EventBus, EventType, utc_now
from ..tasks import Task, TaskState
from .errors import (
    VoiceError,
    VoiceLimitError,
    VoiceProviderError,
    VoiceTimeoutError,
    VoiceValidationError,
)
from .interfaces import AgentHandoff, STTProvider, TTSProvider
from .limits import VoiceLimits
from .models import (
    AudioInput,
    SynthesisRequest,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionStatus,
    VoiceObservation,
    VoiceResponse,
    VoiceResponseStatus,
)
from .normalization import normalize_transcription_text

MonotonicClock = Callable[[], float]


class VoiceRuntime:
    """Bounded voice transport over pluggable providers and the existing Agent."""

    def __init__(
        self,
        stt_provider: STTProvider,
        *,
        tts_provider: TTSProvider | None = None,
        limits: VoiceLimits | None = None,
        events: EventBus | None = None,
        clock: Clock | None = None,
        monotonic: MonotonicClock | None = None,
    ) -> None:
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._limits = limits or VoiceLimits()
        self._events = events
        self._clock = clock or utc_now
        self._monotonic = monotonic or time.monotonic

    @classmethod
    def from_settings(
        cls,
        stt_provider: STTProvider,
        *,
        settings: Settings | None = None,
        tts_provider: TTSProvider | None = None,
        events: EventBus | None = None,
        clock: Clock | None = None,
        monotonic: MonotonicClock | None = None,
    ) -> VoiceRuntime:
        """Construct a runtime with validated ``VOICE_*`` bounds."""
        resolved = settings if settings is not None else Settings.from_env()
        return cls(
            stt_provider,
            tts_provider=tts_provider,
            limits=VoiceLimits.from_settings(resolved),
            events=events,
            clock=clock,
            monotonic=monotonic,
        )

    @property
    def limits(self) -> VoiceLimits:
        """Effective hard-clamped runtime limits."""
        return self._limits

    def transcribe(self, audio: AudioInput) -> VoiceObservation:
        """Validate audio and return normalized text plus metadata only.

        Payload bytes are handed to the provider only for the duration of the
        call. Neither the returned observation nor any emitted event contains
        those bytes.
        """
        started_at = self._clock()
        try:
            validated_audio = AudioInput.model_validate(audio)
            self._validate_audio_limits(validated_audio)
        except ValidationError:
            error = VoiceValidationError("invalid_audio")
            self._emit_error("validation", error)
            # Validation errors can embed input values, including raw bytes.
            raise error from None
        except VoiceError as exc:
            self._emit_error("validation", exc)
            raise

        self._emit(
            EventType.VOICE_TRANSCRIPTION_STARTED,
            {
                "audio_byte_size": validated_audio.metadata.byte_size,
                "duration": validated_audio.metadata.duration,
                "sample_rate": validated_audio.metadata.format.sample_rate,
                "channels": validated_audio.metadata.format.channels,
                "encoding": validated_audio.metadata.format.encoding.value,
            },
        )
        try:
            raw_result = self._call_provider(
                lambda: self._stt_provider.transcribe(validated_audio),
                timeout_s=self._limits.max_transcription_time_s,
            )
            try:
                result = TranscriptionResult.model_validate(raw_result)
            except ValidationError:
                raise VoiceValidationError("invalid_transcription") from None
            if len(result.text) > self._limits.max_text_chars:
                raise VoiceLimitError("transcription_limit_exceeded")
            if (
                result.language is not None
                and len(result.language) > self._limits.max_language_chars
            ):
                raise VoiceLimitError("language_limit_exceeded")
            duration_tolerance = max(0.05, 1.0 / validated_audio.metadata.format.sample_rate)
            if abs(result.duration - validated_audio.metadata.duration) > duration_tolerance:
                raise VoiceValidationError("invalid_transcription")
            normalized_text = normalize_transcription_text(
                result.text,
                max_chars=self._limits.max_text_chars,
            )
            status = result.status
            if (
                result.confidence is None
                or result.confidence < self._limits.minimum_transcription_confidence
            ):
                status = TranscriptionStatus.UNCERTAIN
            result = result.model_copy(update={"text": normalized_text, "status": status})
            completed_at = self._clock()
            observation = VoiceObservation(
                transcription=result,
                audio_metadata=validated_audio.metadata,
                started_at=started_at,
                completed_at=completed_at,
            )
        except VoiceError as exc:
            self._emit_error("transcription", exc)
            raise
        except Exception:
            provider_error = VoiceProviderError()
            self._emit_error("transcription", provider_error)
            raise provider_error from None

        self._emit(
            EventType.VOICE_TRANSCRIPTION_COMPLETED,
            {
                "status": result.status.value,
                "confidence": result.confidence,
                "language": result.language,
                "duration": result.duration,
                "text_chars": len(result.text),
            },
        )
        return observation

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize one bounded response; generated bytes stay ephemeral."""
        provider = self._tts_provider
        if provider is None:
            error = VoiceError("synthesis_unavailable")
            self._emit_error("synthesis", error)
            raise error
        try:
            validated_request = SynthesisRequest.model_validate(request)
            validated_request = self._validate_synthesis_request(validated_request)
        except ValidationError:
            error = VoiceValidationError("invalid_synthesis_request")
            self._emit_error("synthesis_validation", error)
            raise error from None
        except VoiceError as exc:
            self._emit_error("synthesis_validation", exc)
            raise

        self._emit(
            EventType.VOICE_SYNTHESIS_STARTED,
            {
                "text_chars": len(validated_request.text),
                "language": validated_request.language,
            },
        )
        try:
            raw_result = self._call_provider(
                lambda: provider.synthesize(
                    validated_request.text,
                    language=validated_request.language,
                    voice=validated_request.voice,
                    speaking_rate=validated_request.speaking_rate,
                    pitch=validated_request.pitch,
                ),
                timeout_s=self._limits.max_synthesis_time_s,
            )
            try:
                result = SynthesisResult.model_validate(raw_result)
            except ValidationError:
                raise VoiceValidationError("invalid_synthesis_result") from None
            if (
                result.audio_metadata.byte_size > self._limits.max_audio_bytes
                or result.audio_metadata.duration > self._limits.max_duration_s
            ):
                raise VoiceLimitError("synthesis_limit_exceeded")
        except VoiceError as exc:
            self._emit_error("synthesis", exc)
            raise
        except Exception:
            provider_error = VoiceProviderError()
            self._emit_error("synthesis", provider_error)
            raise provider_error from None

        self._emit(
            EventType.VOICE_SYNTHESIS_COMPLETED,
            {
                "audio_byte_size": result.audio_metadata.byte_size,
                "duration": result.duration,
                "encoding": result.audio_metadata.format.encoding.value,
                "has_payload": result.payload is not None,
                "has_reference": result.reference is not None,
            },
        )
        return result

    def process_audio(
        self,
        audio: AudioInput,
        agent: AgentHandoff,
        *,
        synthesize_response: bool = True,
        language: str | None = None,
        voice: str | None = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ) -> VoiceResponse:
        """Run the canonical voice → Agent.run → optional TTS transport once."""
        try:
            observation = self.transcribe(audio)
        except VoiceError as exc:
            return VoiceResponse(
                status=VoiceResponseStatus.FAILED,
                error_code=exc.code,
                error_message=exc.public_message,
            )

        transcription = observation.transcription
        if transcription.status is TranscriptionStatus.UNCERTAIN:
            return VoiceResponse(
                status=VoiceResponseStatus.UNCERTAIN,
                observation=observation,
            )

        try:
            task = agent.run(transcription.text, input_channel="voice")
            if not isinstance(task, Task):
                raise TypeError("agent handoff did not return a Task")
        except Exception:
            error = VoiceError("agent_handoff_failed")
            self._emit_error("agent_handoff", error)
            return VoiceResponse(
                status=VoiceResponseStatus.FAILED,
                observation=observation,
                error_code=error.code,
                error_message=error.public_message,
            )

        if task.state is not TaskState.COMPLETED:
            error = VoiceError("agent_task_not_completed")
            self._emit_error("agent_handoff", error)
            return VoiceResponse(
                status=VoiceResponseStatus.FAILED,
                observation=observation,
                task_id=task.id,
                error_code=error.code,
                error_message=error.public_message,
            )

        try:
            response_text = self._extract_response_text(task)
        except VoiceError as exc:
            self._emit_error("response_validation", exc)
            return VoiceResponse(
                status=VoiceResponseStatus.FAILED,
                observation=observation,
                task_id=task.id,
                error_code=exc.code,
                error_message=exc.public_message,
            )
        if not response_text:
            return VoiceResponse(
                status=VoiceResponseStatus.NO_RESPONSE,
                observation=observation,
                task_id=task.id,
            )

        if not synthesize_response or self._tts_provider is None:
            return VoiceResponse(
                response_text=response_text,
                status=VoiceResponseStatus.RESPONDED,
                observation=observation,
                task_id=task.id,
            )

        synthesis_language = language if language is not None else transcription.language
        try:
            synthesis_request = SynthesisRequest(
                text=response_text,
                language=synthesis_language,
                voice=voice,
                speaking_rate=speaking_rate,
                pitch=pitch,
            )
            synthesis = self.synthesize(synthesis_request)
        except ValidationError:
            error = VoiceValidationError("invalid_synthesis_request")
            self._emit_error("synthesis_validation", error)
            return self._response_with_synthesis_error(response_text, observation, task.id, error)
        except VoiceError as exc:
            return self._response_with_synthesis_error(response_text, observation, task.id, exc)

        return VoiceResponse(
            response_text=response_text,
            synthesis_result=synthesis,
            status=VoiceResponseStatus.RESPONDED,
            observation=observation,
            task_id=task.id,
        )

    def _validate_audio_limits(self, audio: AudioInput) -> None:
        if audio.metadata.byte_size > self._limits.max_audio_bytes:
            raise VoiceLimitError("audio_limit_exceeded")
        if audio.metadata.duration > self._limits.max_duration_s:
            raise VoiceLimitError("duration_limit_exceeded")

    def _validate_synthesis_request(self, request: SynthesisRequest) -> SynthesisRequest:
        if len(request.text) > self._limits.max_synthesis_text_chars:
            raise VoiceLimitError("synthesis_limit_exceeded")
        if request.language is not None and len(request.language) > self._limits.max_language_chars:
            raise VoiceLimitError("language_limit_exceeded")
        if request.voice is not None and len(request.voice) > self._limits.max_voice_name_chars:
            raise VoiceLimitError("voice_name_limit_exceeded")
        try:
            text = normalize_transcription_text(
                request.text,
                max_chars=self._limits.max_synthesis_text_chars,
            )
        except VoiceError as exc:
            if exc.code == "empty_transcription":
                raise VoiceValidationError("invalid_synthesis_request") from None
            raise
        return request.model_copy(update={"text": text})

    def _call_provider(self, operation: Callable[[], object], *, timeout_s: float) -> object:
        """Bound attempts and elapsed time; only explicit transient errors retry."""
        started = self._monotonic()
        for attempt in range(self._limits.max_retries + 1):
            if self._monotonic() - started >= timeout_s:
                raise VoiceTimeoutError()
            try:
                result = operation()
            except VoiceTimeoutError:
                raise
            except TimeoutError:
                raise VoiceTimeoutError() from None
            except VoiceProviderError as exc:
                if exc.retryable and attempt < self._limits.max_retries:
                    if self._monotonic() - started >= timeout_s:
                        raise VoiceTimeoutError() from None
                    continue
                raise VoiceProviderError() from None
            except Exception:
                raise VoiceProviderError() from None
            if self._monotonic() - started >= timeout_s:
                raise VoiceTimeoutError()
            return result
        raise VoiceProviderError()

    def _extract_response_text(self, task: Task) -> str:
        result: Any = task.result
        candidate: str | None = None
        if isinstance(result, str):
            candidate = result
        elif isinstance(result, dict):
            for key in ("response_text", "message"):
                value = result.get(key)
                if isinstance(value, str):
                    candidate = value
                    break
        if candidate is None:
            return ""
        try:
            return normalize_transcription_text(
                candidate,
                max_chars=self._limits.max_text_chars,
                allow_empty=True,
            )
        except VoiceLimitError:
            raise VoiceLimitError("response_limit_exceeded") from None

    def _response_with_synthesis_error(
        self,
        response_text: str,
        observation: VoiceObservation,
        task_id: str,
        error: VoiceError,
    ) -> VoiceResponse:
        return VoiceResponse(
            response_text=response_text,
            status=VoiceResponseStatus.RESPONDED,
            observation=observation,
            task_id=task_id,
            error_code=error.code,
            error_message=error.public_message,
        )

    def _emit_error(self, stage: str, error: VoiceError) -> None:
        self._emit(EventType.VOICE_ERROR, {"stage": stage, "error_code": error.code})

    def _emit(self, event_type: EventType, data: dict[str, object]) -> None:
        if self._events is not None:
            self._events.emit(event_type, data=data)
