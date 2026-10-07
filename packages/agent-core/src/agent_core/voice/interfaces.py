"""Provider-neutral contracts for voice and the existing agent handoff."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from .models import AudioInput, SynthesisResult, TranscriptionResult

if TYPE_CHECKING:
    from ..tasks import Task


class STTProvider(Protocol):
    """Transcribe an ephemeral bounded audio buffer into structured text."""

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Return text and explicit confidence/uncertainty metadata."""
        ...


class TTSProvider(Protocol):
    """Synthesize one bounded text response into structured audio output."""

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ) -> SynthesisResult:
        """Return bounded ephemeral audio or a bounded opaque reference."""
        ...


class AgentHandoff(Protocol):
    """The existing agent loop; voice adds no planner or executor of its own."""

    def run(self, request: str, *, input_channel: Literal["text", "voice"] = "text") -> Task:
        """Pass normalized user text through the canonical Agent.run flow."""
        ...
