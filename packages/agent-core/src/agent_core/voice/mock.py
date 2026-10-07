"""Deterministic, offline-only STT/TTS providers for tests and examples."""

from __future__ import annotations

from .errors import VoiceProviderError
from .models import (
    AudioEncoding,
    AudioFormat,
    AudioInput,
    AudioMetadata,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionStatus,
)


class MockSTTProvider:
    """Canned local transcription provider; it never retains payload bytes."""

    def __init__(
        self,
        text: str = "hello from the voice agent",
        *,
        confidence: float | None = 0.99,
        language: str | None = "en",
        status: TranscriptionStatus = TranscriptionStatus.CONFIDENT,
        retryable_failures: int = 0,
        always_fail: bool = False,
        timeout: bool = False,
    ) -> None:
        self._text = text
        self._confidence = confidence
        self._language = language
        self._status = status
        self._retryable_failures = retryable_failures
        self._always_fail = always_fail
        self._timeout = timeout
        self.calls = 0

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        self.calls += 1
        if self._timeout:
            raise TimeoutError("simulated local STT timeout")
        if self._retryable_failures > 0:
            self._retryable_failures -= 1
            raise VoiceProviderError(retryable=True)
        if self._always_fail:
            raise VoiceProviderError()
        return TranscriptionResult(
            text=self._text,
            confidence=self._confidence,
            language=self._language,
            duration=audio.metadata.duration,
            status=self._status,
            provider_metadata={"provider": "mock"},
        )


class MockTTSProvider:
    """Deterministic silent-PCM placeholder; this does not synthesize speech."""

    def __init__(
        self,
        *,
        retryable_failures: int = 0,
        always_fail: bool = False,
        timeout: bool = False,
    ) -> None:
        self._retryable_failures = retryable_failures
        self._always_fail = always_fail
        self._timeout = timeout
        self.calls = 0

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ) -> SynthesisResult:
        del language, voice, speaking_rate, pitch
        self.calls += 1
        if self._timeout:
            raise TimeoutError("simulated local TTS timeout")
        if self._retryable_failures > 0:
            self._retryable_failures -= 1
            raise VoiceProviderError(retryable=True)
        if self._always_fail:
            raise VoiceProviderError()

        duration = min(30.0, max(0.1, len(text) * 0.01))
        sample_rate = 8_000
        payload = bytes([128]) * round(duration * sample_rate)
        audio_metadata = AudioMetadata(
            duration=duration,
            format=AudioFormat(
                sample_rate=sample_rate,
                channels=1,
                sample_width=1,
                encoding=AudioEncoding.PCM_U8,
            ),
            byte_size=len(payload),
        )
        return SynthesisResult(
            audio_metadata=audio_metadata,
            duration=duration,
            payload=payload,
            provider_metadata={"provider": "mock", "output": "silence_placeholder"},
        )
