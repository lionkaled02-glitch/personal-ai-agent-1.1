"""Stable, content-free errors for voice input and synthesis boundaries."""

from __future__ import annotations

_ERROR_MESSAGES: dict[str, str] = {
    "voice_error": "The voice operation could not be completed.",
    "invalid_audio": "Audio input is malformed or unsupported.",
    "audio_limit_exceeded": "Audio input exceeds the configured size limit.",
    "duration_limit_exceeded": "Audio duration exceeds the configured limit.",
    "invalid_transcription": "Transcription output failed validation.",
    "transcription_limit_exceeded": "Transcription exceeds the configured text limit.",
    "language_limit_exceeded": "Language metadata exceeds the configured limit.",
    "empty_transcription": "The transcription did not contain usable text.",
    "transcription_uncertain": "The transcription is uncertain and was not handed to the agent.",
    "provider_failed": "The voice provider could not complete the operation.",
    "provider_timeout": "The voice provider exceeded its configured time limit.",
    "synthesis_limit_exceeded": "Synthesis input or output exceeds the configured limit.",
    "voice_name_limit_exceeded": "Voice metadata exceeds the configured limit.",
    "synthesis_unavailable": "No speech synthesis provider is configured.",
    "agent_handoff_failed": "The existing agent could not complete the voice handoff.",
    "agent_task_not_completed": "The existing agent did not complete the request.",
    "response_limit_exceeded": "The agent response exceeds the configured voice text limit.",
    "invalid_synthesis_request": "Synthesis request failed validation.",
    "invalid_synthesis_result": "Synthesis provider output failed validation.",
}


class VoiceError(Exception):
    """Voice error with a stable code and a safe public message."""

    def __init__(self, code: str = "voice_error") -> None:
        self.code = code if code in _ERROR_MESSAGES else "voice_error"
        self.public_message = _ERROR_MESSAGES[self.code]
        super().__init__(self.public_message)


class VoiceValidationError(VoiceError):
    """Malformed audio or provider output rejected at the boundary."""


class VoiceLimitError(VoiceError):
    """Input or output exceeded a hard or configured bound."""


class VoiceProviderError(VoiceError):
    """Provider failure; retryable only when explicitly marked safe."""

    def __init__(self, *, retryable: bool = False) -> None:
        self.retryable = retryable
        super().__init__("provider_failed")


class VoiceTimeoutError(VoiceError):
    """Provider exceeded the cooperative synchronous operation deadline."""

    def __init__(self) -> None:
        super().__init__("provider_timeout")
