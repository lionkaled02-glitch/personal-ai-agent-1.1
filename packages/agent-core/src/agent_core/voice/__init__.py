"""Provider-neutral, bounded voice transport foundation (Phase 8).

This package has no microphone, cloud service, network, storage, or API-key
integration. Voice input is normalized to ordinary text and handed to the
existing agent loop, whose permissions remain authoritative.
"""

from .errors import (
    VoiceError,
    VoiceLimitError,
    VoiceProviderError,
    VoiceTimeoutError,
    VoiceValidationError,
)
from .interfaces import AgentHandoff, STTProvider, TTSProvider
from .limits import VoiceLimits
from .mock import MockSTTProvider, MockTTSProvider
from .models import (
    AudioEncoding,
    AudioFormat,
    AudioInput,
    AudioMetadata,
    SynthesisRequest,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionStatus,
    VoiceObservation,
    VoiceResponse,
    VoiceResponseStatus,
)
from .normalization import normalize_transcription_text
from .runtime import VoiceRuntime
from .tools import (
    VOICE_NORMALIZE_TOOL_NAME,
    VOICE_TOOL_NAMES,
    VoiceNormalizeTool,
    register_voice_tools,
)

__all__ = [
    "VOICE_NORMALIZE_TOOL_NAME",
    "VOICE_TOOL_NAMES",
    "AgentHandoff",
    "AudioEncoding",
    "AudioFormat",
    "AudioInput",
    "AudioMetadata",
    "MockSTTProvider",
    "MockTTSProvider",
    "STTProvider",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSProvider",
    "TranscriptionResult",
    "TranscriptionStatus",
    "VoiceError",
    "VoiceLimitError",
    "VoiceLimits",
    "VoiceNormalizeTool",
    "VoiceObservation",
    "VoiceProviderError",
    "VoiceResponse",
    "VoiceResponseStatus",
    "VoiceRuntime",
    "VoiceTimeoutError",
    "VoiceValidationError",
    "normalize_transcription_text",
    "register_voice_tools",
]
