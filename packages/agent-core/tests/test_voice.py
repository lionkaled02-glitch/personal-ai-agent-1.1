"""Deterministic Phase 8 voice foundation and security-boundary tests."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_core import (
    Agent,
    AudioEncoding,
    AudioFormat,
    AudioInput,
    AudioMetadata,
    EventBus,
    EventType,
    MockSTTProvider,
    MockTTSProvider,
    PermissionLevel,
    PermissionManager,
    Plan,
    PlanStep,
    Settings,
    SynthesisRequest,
    Task,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    TranscriptionResult,
    TranscriptionStatus,
    VoiceLimits,
    VoiceNormalizeTool,
    VoiceResponseStatus,
    VoiceRuntime,
    VoiceValidationError,
    normalize_transcription_text,
    register_voice_tools,
)
from agent_core.permissions import ApprovalRequest
from agent_core.voice.errors import (
    VoiceError,
    VoiceLimitError,
    VoiceProviderError,
    VoiceTimeoutError,
)
from agent_core.voice.models import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_S,
    SynthesisResult,
    VoiceObservation,
    VoiceResponse,
)
from agent_core.voice.serialization import response_to_dict
from agent_core.voice.tools import VOICE_NORMALIZE_TOOL_NAME
from pydantic import ValidationError

FIXED_NOW = datetime(2026, 10, 4, tzinfo=UTC)


def _pcm_format(*, sample_rate: int = 16_000) -> AudioFormat:
    return AudioFormat(
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        encoding=AudioEncoding.PCM_S16LE,
    )


def _audio(*, payload: bytes = b"\x01\x02\x03\x04", duration: float = 0.1) -> AudioInput:
    # Tiny deterministic fixtures use a low PCM sample rate so format, byte
    # size, and duration remain internally consistent without large buffers.
    if len(payload) % 2 == 0:
        audio_format = _pcm_format(sample_rate=max(1, round(len(payload) / (duration * 2))))
    else:
        audio_format = AudioFormat(
            sample_rate=max(1, round(len(payload) / duration)),
            channels=1,
            sample_width=1,
            encoding=AudioEncoding.PCM_U8,
        )
    return AudioInput(
        metadata=AudioMetadata(
            duration=duration,
            format=audio_format,
            byte_size=len(payload),
        ),
        payload=payload,
    )


def _demo_agent(tmp_path: Path, *, events: EventBus | None = None) -> Agent:
    agent = Agent.create_demo(workspace_root=tmp_path)
    if events is not None:
        # Use the normal event bus property for assertions; Agent owns its bus.
        assert events is agent.events
    return agent


class _HighRiskTool:
    spec = ToolSpec(
        name="delete_file",
        description="Delete one selected file.",
        input_schema={"type": "object", "properties": {}, "required": []},
        output_schema={
            "type": "object",
            "properties": {"done": {"type": "boolean"}},
            "required": ["done"],
        },
        permission_level=PermissionLevel.HIGH,
    )

    def __init__(self) -> None:
        self.calls = 0

    def run(self, input: dict[str, Any]) -> ToolResult:
        del input
        self.calls += 1
        return ToolResult(ok=True, output={"done": True})


class _HighRiskPlanner:
    def plan(self, request: str, available_tools: Sequence[ToolSpec]) -> Plan:
        del request
        assert any(spec.name == "delete_file" for spec in available_tools)
        return Plan(
            steps=[
                PlanStep(
                    tool_name="delete_file",
                    description="Delete the selected file.",
                    input={},
                )
            ]
        )


class _SlowSTT:
    def __init__(self, elapsed: list[float]) -> None:
        self.elapsed = elapsed
        self.calls = 0

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        self.calls += 1
        self.elapsed[0] += 2.0
        return TranscriptionResult(
            text="hello",
            confidence=0.99,
            language="en",
            duration=audio.metadata.duration,
        )


class _SlowTTS:
    def __init__(self, elapsed: list[float], result: SynthesisResult) -> None:
        self.elapsed = elapsed
        self.result = result
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
        del text, language, voice, speaking_rate, pitch
        self.calls += 1
        self.elapsed[0] += 2.0
        return self.result


class TestVoiceModelsAndLimits:
    def test_valid_audio_format_metadata_and_ephemeral_payload(self) -> None:
        audio = _audio()
        assert audio.metadata.format.encoding is AudioEncoding.PCM_S16LE
        assert audio.metadata.byte_size == len(audio.payload)
        assert "payload" not in repr(audio)
        assert "payload" not in audio.model_dump()
        assert "audio" not in audio.model_dump_json()

    @pytest.mark.parametrize(
        "values",
        [
            {"sample_rate": 0, "channels": 1, "sample_width": 2, "encoding": "pcm_s16le"},
            {"sample_rate": 16_000, "channels": 0, "sample_width": 2, "encoding": "pcm_s16le"},
            {"sample_rate": 16_000, "channels": 1, "sample_width": 1, "encoding": "pcm_s16le"},
            {"sample_rate": 16_000, "channels": 1, "sample_width": 2, "encoding": "mp3"},
        ],
    )
    def test_malformed_audio_format_is_rejected(self, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            AudioFormat.model_validate(values)

    @pytest.mark.parametrize(
        "duration",
        [0, -1, float("inf"), MAX_AUDIO_DURATION_S + 1],
    )
    def test_malformed_audio_metadata_is_rejected(self, duration: float) -> None:
        with pytest.raises(ValidationError):
            AudioMetadata(duration=duration, format=_pcm_format(), byte_size=4)

    def test_numeric_voice_fields_reject_string_coercion(self) -> None:
        with pytest.raises(ValidationError):
            AudioMetadata(duration="0.1", format=_pcm_format(), byte_size=4)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hello", speaking_rate="1.0")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            VoiceLimits(max_duration_s="30.0")  # type: ignore[arg-type]

    def test_audio_metadata_rejects_extra_fields_and_mismatched_payload_size(self) -> None:
        with pytest.raises(ValidationError):
            AudioMetadata(duration=0.1, format=_pcm_format(), byte_size=4)
        with pytest.raises(ValidationError):
            AudioMetadata.model_validate(
                {
                    "duration": 0.1,
                    "format": _pcm_format(),
                    "byte_size": 3_200,
                    "filename": "recording.wav",
                }
            )
        with pytest.raises(ValidationError):
            AudioInput(
                metadata=AudioMetadata(
                    duration=0.000125,
                    format=_pcm_format(),
                    byte_size=4,
                ),
                payload=b"12345",
            )

    def test_oversized_audio_is_rejected_at_model_boundary(self) -> None:
        with pytest.raises(ValidationError):
            AudioMetadata(
                duration=1.0,
                format=_pcm_format(),
                byte_size=MAX_AUDIO_BYTES + 1,
            )
        with pytest.raises(ValidationError):
            AudioInput(
                metadata=AudioMetadata(
                    duration=MAX_AUDIO_BYTES / (16_000 * 2),
                    format=_pcm_format(),
                    byte_size=MAX_AUDIO_BYTES,
                ),
                payload=b"x" * (MAX_AUDIO_BYTES + 1),
            )

    def test_synthesis_models_bound_output_and_hide_bytes(self) -> None:
        result = MockTTSProvider().synthesize("hello")
        assert result.payload is not None
        assert len(result.payload) == result.audio_metadata.byte_size
        assert "payload" not in repr(result)
        assert "payload" not in result.model_dump()
        reference_result = SynthesisResult(
            audio_metadata=result.audio_metadata,
            duration=result.duration,
            reference="opaque-reference-token",
        )
        assert reference_result.reference == "opaque-reference-token"
        assert "reference" not in reference_result.model_dump()
        assert "opaque-reference-token" not in repr(reference_result)
        with pytest.raises(ValidationError):
            SynthesisResult(
                audio_metadata=result.audio_metadata,
                duration=result.duration,
                payload=result.payload,
                reference="also-present",
            )
        with pytest.raises(ValidationError):
            SynthesisResult(
                audio_metadata=result.audio_metadata,
                duration=result.duration,
                reference="../recording.wav",
            )

    def test_synthesis_request_rejects_oversized_or_malformed_fields(self) -> None:
        with pytest.raises(ValidationError):
            SynthesisRequest(text="x" * 2_001)
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hello", language="en/../../secret")
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hello", voice="bad/name")
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hello", speaking_rate=3.0)

    def test_voice_limits_are_safe_and_from_settings(self) -> None:
        defaults = VoiceLimits()
        assert defaults.max_audio_bytes == 1_048_576
        assert defaults.max_duration_s == 30.0
        assert defaults.max_retries == 1
        settings = Settings.from_env(
            env={
                "VOICE_MAX_AUDIO_BYTES": "2048",
                "VOICE_MAX_DURATION_S": "12.5",
                "VOICE_MAX_TEXT_CHARS": "300",
                "VOICE_MAX_LANGUAGE_CHARS": "12",
                "VOICE_MAX_VOICE_NAME_CHARS": "24",
                "VOICE_MAX_SYNTHESIS_TEXT_CHARS": "150",
                "VOICE_MAX_TRANSCRIPTION_TIME_S": "4.5",
                "VOICE_MAX_SYNTHESIS_TIME_S": "3",
                "VOICE_MAX_RETRIES": "2",
                "VOICE_MIN_TRANSCRIPTION_CONFIDENCE": "0.75",
            }
        )
        limits = VoiceLimits.from_settings(settings)
        assert limits.max_audio_bytes == 2048
        assert limits.max_duration_s == 12.5
        assert limits.max_text_chars == 300
        assert limits.max_language_chars == 12
        assert limits.max_voice_name_chars == 24
        assert limits.max_synthesis_text_chars == 150
        assert limits.max_transcription_time_s == 4.5
        assert limits.max_synthesis_time_s == 3.0
        assert limits.max_retries == 2
        assert limits.minimum_transcription_confidence == 0.75
        assert (
            VoiceRuntime.from_settings(
                MockSTTProvider(),
                settings=settings,
            ).limits
            == limits
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("voice_max_audio_bytes", MAX_AUDIO_BYTES + 1),
            ("voice_max_duration_s", MAX_AUDIO_DURATION_S + 1),
            ("voice_max_text_chars", 4_001),
            ("voice_max_retries", 4),
        ],
    )
    def test_voice_limits_cannot_exceed_hard_caps(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            VoiceLimits.model_validate({field: value})


class TestVoiceNormalization:
    def test_normalization_preserves_unicode_and_punctuation(self) -> None:
        assert normalize_transcription_text("  Cafe\u0301\t—  yes?  ") == "Café — yes?"
        arabic = "  مرحبًا\tبالعالم؟  "  # noqa: RUF001
        assert normalize_transcription_text(arabic) == "مرحبًا بالعالم؟"
        assert normalize_transcription_text("Go! 👩🏽‍💻") == "Go! 👩🏽‍💻"

    def test_empty_and_oversized_transcripts_are_rejected(self) -> None:
        with pytest.raises(VoiceValidationError) as empty:
            normalize_transcription_text(" \n\t ")
        assert empty.value.code == "empty_transcription"
        with pytest.raises(VoiceLimitError) as oversized:
            normalize_transcription_text("a" * 11, max_chars=10)
        assert oversized.value.code == "transcription_limit_exceeded"

    @pytest.mark.parametrize("text", ["bad\x00text", "bad\ud800text", "bad\u202etext"])
    def test_invalid_unicode_controls_are_rejected_without_silent_cleanup(self, text: str) -> None:
        with pytest.raises(VoiceValidationError):
            normalize_transcription_text(text)

    def test_optional_empty_response_does_not_change_text_semantics(self) -> None:
        assert normalize_transcription_text(" \t\n", allow_empty=True) == ""


class TestVoiceRuntime:
    def test_stt_success_normalizes_before_handoff_to_existing_agent(self, tmp_path: Path) -> None:
        agent = _demo_agent(tmp_path)
        provider = MockSTTProvider(text="  Run   the demo tool. ")
        tts = MockTTSProvider()
        runtime = VoiceRuntime(
            provider,
            tts_provider=tts,
            events=agent.events,
            clock=lambda: FIXED_NOW,
        )
        response = runtime.process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.RESPONDED
        assert response.response_text == "hello from the agent core"
        assert response.observation is not None
        assert response.observation.transcription.text == "Run the demo tool."
        assert response.synthesis_result is not None
        assert response.synthesis_result.payload is not None
        assert provider.calls == 1
        assert tts.calls == 1
        created = agent.events.events_of_type(EventType.TASK_CREATED)[0]
        assert created.data == {"input_channel": "voice", "request_chars": 18}
        assert EventType.PLAN_CREATED in [event.type for event in agent.events.history]

    def test_no_tts_provider_returns_text_only(self, tmp_path: Path) -> None:
        agent = _demo_agent(tmp_path)
        response = VoiceRuntime(MockSTTProvider(text="Run the demo tool.")).process_audio(
            _audio(), agent
        )
        assert response.status is VoiceResponseStatus.RESPONDED
        assert response.synthesis_result is None
        assert response.response_text == "hello from the agent core"

    @pytest.mark.parametrize(
        ("status", "confidence"),
        [
            (TranscriptionStatus.UNCERTAIN, 0.9),
            (TranscriptionStatus.CONFIDENT, 0.2),
            (TranscriptionStatus.UNCERTAIN, None),
        ],
    )
    def test_uncertain_transcription_is_exposed_without_agent_handoff(
        self, tmp_path: Path, status: TranscriptionStatus, confidence: float | None
    ) -> None:
        agent = _demo_agent(tmp_path)
        provider = MockSTTProvider(
            text="Run the demo tool.",
            status=status,
            confidence=confidence,
        )
        response = VoiceRuntime(provider, events=agent.events).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.UNCERTAIN
        assert response.observation is not None
        assert response.observation.transcription.status is TranscriptionStatus.UNCERTAIN
        assert agent.events.events_of_type(EventType.TASK_CREATED) == []
        assert provider.calls == 1

    def test_empty_transcription_fails_closed_without_agent_handoff(self, tmp_path: Path) -> None:
        agent = _demo_agent(tmp_path)
        response = VoiceRuntime(MockSTTProvider(text=" \n "), events=agent.events).process_audio(
            _audio(), agent
        )
        assert response.status is VoiceResponseStatus.FAILED
        assert response.error_code == "empty_transcription"
        assert agent.events.events_of_type(EventType.TASK_CREATED) == []

    def test_transcription_provider_failure_is_sanitized(self, tmp_path: Path) -> None:
        agent = _demo_agent(tmp_path)

        class _SensitiveFailure:
            def transcribe(self, audio: AudioInput) -> TranscriptionResult:
                del audio
                raise RuntimeError("synthetic secret: sk-test-voice-secret")

        response = VoiceRuntime(_SensitiveFailure(), events=agent.events).process_audio(
            _audio(), agent
        )
        assert response.status is VoiceResponseStatus.FAILED
        assert response.error_code == "provider_failed"
        assert "sk-test-voice-secret" not in (response.error_message or "")
        assert "sk-test-voice-secret" not in repr(agent.events.history)
        assert agent.events.events_of_type(EventType.TASK_CREATED) == []

    def test_stt_timeout_is_structured_and_not_retried(self) -> None:
        provider = MockSTTProvider(timeout=True)
        runtime = VoiceRuntime(provider, limits=VoiceLimits(max_retries=3))
        with pytest.raises(VoiceTimeoutError) as exc:
            runtime.transcribe(_audio())
        assert exc.value.code == "provider_timeout"
        assert provider.calls == 1

    def test_stt_retry_is_bounded_and_successful_only_for_transient_failures(self) -> None:
        provider = MockSTTProvider(text="ready", retryable_failures=1)
        observation = VoiceRuntime(provider, limits=VoiceLimits(max_retries=1)).transcribe(_audio())
        assert observation.transcription.text == "ready"
        assert provider.calls == 2

        exhausted = MockSTTProvider(retryable_failures=4)
        with pytest.raises(VoiceProviderError):
            VoiceRuntime(exhausted, limits=VoiceLimits(max_retries=1)).transcribe(_audio())
        assert exhausted.calls == 2

    def test_elapsed_stt_time_limit_is_cooperative_and_no_retry_occurs(self) -> None:
        elapsed = [0.0]
        provider = _SlowSTT(elapsed)
        runtime = VoiceRuntime(
            provider,
            limits=VoiceLimits(max_transcription_time_s=1.0, max_retries=3),
            monotonic=lambda: elapsed[0],
        )
        with pytest.raises(VoiceTimeoutError):
            runtime.transcribe(_audio())
        assert provider.calls == 1

    def test_configured_audio_bytes_duration_and_transcript_limits_are_enforced(
        self, tmp_path: Path
    ) -> None:
        agent = _demo_agent(tmp_path)
        oversized_runtime = VoiceRuntime(
            MockSTTProvider(),
            limits=VoiceLimits(max_audio_bytes=3),
            events=agent.events,
        )
        response = oversized_runtime.process_audio(_audio(), agent)
        assert response.error_code == "audio_limit_exceeded"
        assert agent.events.events_of_type(EventType.TASK_CREATED) == []

        duration_runtime = VoiceRuntime(
            MockSTTProvider(),
            limits=VoiceLimits(max_duration_s=0.05),
        )
        response = duration_runtime.process_audio(_audio(duration=0.1), agent)
        assert response.error_code == "duration_limit_exceeded"
        assert agent.events.events_of_type(EventType.TASK_CREATED) == []

        text_provider = MockSTTProvider(text="transcript too long")
        text_runtime = VoiceRuntime(text_provider, limits=VoiceLimits(max_text_chars=8))
        with pytest.raises(VoiceLimitError):
            text_runtime.transcribe(_audio())

    def test_provider_result_confidence_and_language_are_validated(self) -> None:
        class _InvalidConfidence:
            def transcribe(self, audio: AudioInput) -> dict[str, object]:
                return {
                    "text": "hello",
                    "confidence": 1.5,
                    "language": "en",
                    "duration": audio.metadata.duration,
                }

        with pytest.raises(VoiceValidationError) as confidence:
            VoiceRuntime(_InvalidConfidence()).transcribe(_audio())  # type: ignore[arg-type]
        assert confidence.value.code == "invalid_transcription"

        class _LongLanguage:
            def transcribe(self, audio: AudioInput) -> dict[str, object]:
                return {
                    "text": "hello",
                    "confidence": 0.9,
                    "language": "en-US",
                    "duration": audio.metadata.duration,
                }

        with pytest.raises(VoiceLimitError) as language:
            VoiceRuntime(
                _LongLanguage(),  # type: ignore[arg-type]
                limits=VoiceLimits(max_language_chars=3),
            ).transcribe(_audio())
        assert language.value.code == "language_limit_exceeded"

    def test_synthesis_success_failure_timeout_and_limits(self, tmp_path: Path) -> None:
        result = MockTTSProvider().synthesize("response")
        successful = VoiceRuntime(MockSTTProvider(), tts_provider=MockTTSProvider())
        synthesized = successful.synthesize(SynthesisRequest(text="response", voice="local"))
        assert synthesized.audio_metadata.byte_size > 0
        assert synthesized.payload is not None

        failed_provider = MockTTSProvider(always_fail=True)
        with pytest.raises(VoiceProviderError):
            VoiceRuntime(MockSTTProvider(), tts_provider=failed_provider).synthesize(
                SynthesisRequest(text="response")
            )
        assert failed_provider.calls == 1

        timeout_provider = MockTTSProvider(timeout=True)
        with pytest.raises(VoiceTimeoutError):
            VoiceRuntime(MockSTTProvider(), tts_provider=timeout_provider).synthesize(
                SynthesisRequest(text="response")
            )
        assert timeout_provider.calls == 1

        limited_provider = MockTTSProvider()
        limited_runtime = VoiceRuntime(
            MockSTTProvider(),
            tts_provider=limited_provider,
            limits=VoiceLimits(max_synthesis_text_chars=4),
        )
        with pytest.raises(VoiceLimitError):
            limited_runtime.synthesize(SynthesisRequest(text="response"))
        assert limited_provider.calls == 0

        unavailable_events = EventBus(clock=lambda: FIXED_NOW)
        with pytest.raises(VoiceError) as unavailable:
            VoiceRuntime(MockSTTProvider(), events=unavailable_events).synthesize(
                SynthesisRequest(text="response")
            )
        assert unavailable.value.code == "synthesis_unavailable"
        assert unavailable_events.events_of_type(EventType.VOICE_ERROR)[0].data == {
            "stage": "synthesis",
            "error_code": "synthesis_unavailable",
        }
        assert result.payload is not None

    def test_tts_elapsed_timeout_is_cooperative(self) -> None:
        elapsed = [0.0]
        output = MockTTSProvider().synthesize("ok")
        provider = _SlowTTS(elapsed, output)
        runtime = VoiceRuntime(
            MockSTTProvider(),
            tts_provider=provider,
            limits=VoiceLimits(max_synthesis_time_s=1.0, max_retries=3),
            monotonic=lambda: elapsed[0],
        )
        with pytest.raises(VoiceTimeoutError):
            runtime.synthesize(SynthesisRequest(text="hello"))
        assert provider.calls == 1

    def test_tts_retry_is_bounded(self) -> None:
        provider = MockTTSProvider(retryable_failures=1)
        runtime = VoiceRuntime(
            MockSTTProvider(),
            tts_provider=provider,
            limits=VoiceLimits(max_retries=1),
        )
        result = runtime.synthesize(SynthesisRequest(text="response"))
        assert result.payload is not None
        assert provider.calls == 2

    def test_response_text_bounds_and_no_agent_replay_after_tts_failure(
        self, tmp_path: Path
    ) -> None:
        agent = _demo_agent(tmp_path)
        stt = MockSTTProvider(text="Run the demo tool.")
        tts = MockTTSProvider(always_fail=True)
        runtime = VoiceRuntime(stt, tts_provider=tts, events=agent.events)
        response = runtime.process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.RESPONDED
        assert response.response_text == "hello from the agent core"
        assert response.error_code == "provider_failed"
        assert stt.calls == 1
        assert tts.calls == 1
        assert len(agent.events.events_of_type(EventType.TASK_CREATED)) == 1

        too_long = _demo_agent(tmp_path / "second")
        response = VoiceRuntime(
            MockSTTProvider(text="Run the demo tool."),
            events=too_long.events,
            limits=VoiceLimits(max_text_chars=4),
        ).process_audio(_audio(), too_long)
        assert response.error_code == "transcription_limit_exceeded"
        assert too_long.events.events_of_type(EventType.TASK_CREATED) == []

    def test_agent_failure_is_not_retried_or_treated_as_voice_success(self) -> None:
        class _FailingAgent:
            calls = 0

            def run(self, request: str, *, input_channel: str = "text") -> Task:
                del request, input_channel
                self.calls += 1
                raise RuntimeError("private provider detail")

        agent = _FailingAgent()
        response = VoiceRuntime(MockSTTProvider()).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.FAILED
        assert response.error_code == "agent_handoff_failed"
        assert "private provider detail" not in (response.error_message or "")
        assert agent.calls == 1


class TestVoicePermissionAndTools:
    def test_voice_high_risk_command_uses_existing_approval_path(self, tmp_path: Path) -> None:
        tool = _HighRiskTool()
        registry = ToolRegistry()
        registry.register(tool)
        approvals: list[ApprovalRequest] = []

        def deny(request: ApprovalRequest) -> bool:
            approvals.append(request)
            return False

        events = EventBus(clock=lambda: FIXED_NOW)
        agent = Agent(
            planner=_HighRiskPlanner(),
            registry=registry,
            permissions=PermissionManager(approval=deny),
            events=events,
            clock=lambda: FIXED_NOW,
        )
        response = VoiceRuntime(
            MockSTTProvider(text="delete this file"),
            events=events,
            clock=lambda: FIXED_NOW,
        ).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.FAILED
        assert response.error_code == "agent_task_not_completed"
        assert len(approvals) == 1
        assert approvals[0].permission_level is PermissionLevel.HIGH
        assert tool.calls == 0
        approval_events = events.events_of_type(EventType.APPROVAL_REQUIRED)
        assert len(approval_events) == 1
        assert approval_events[0].data["permission_level"] == "HIGH"
        assert len(events.events_of_type(EventType.TOOL_DENIED)) == 1

    def test_voice_normalize_tool_has_explicit_safe_schema_and_permissions(self) -> None:
        tool = VoiceNormalizeTool(VoiceLimits(max_text_chars=32))
        assert tool.spec.name == VOICE_NORMALIZE_TOOL_NAME
        assert tool.spec.permission_level is PermissionLevel.LOW
        assert tool.spec.sensitive_input is True
        assert tool.spec.sensitive_output is True
        assert tool.spec.input_schema["required"] == ["text"]
        assert tool.spec.input_schema["properties"]["text"]["maxLength"] == 32
        assert tool.spec.output_schema["required"] == ["text"]
        result = tool.run({"text": "  hello   world!  "})
        assert result.ok is True
        assert result.output == {"text": "hello world!"}
        invalid = tool.run({"text": "x" * 33})
        assert invalid.ok is False
        assert invalid.error_code == "transcription_limit_exceeded"

    def test_voice_tool_registration_is_explicit_and_audio_free(self) -> None:
        registry = ToolRegistry()
        register_voice_tools(registry, VoiceLimits())
        assert registry.names() == [VOICE_NORMALIZE_TOOL_NAME]
        spec = registry.require(VOICE_NORMALIZE_TOOL_NAME).spec
        assert "audio" not in spec.input_schema["properties"]
        assert spec.permission_level is PermissionLevel.LOW

    def test_tool_runtime_redacts_transcripts_from_voice_tool_events(self) -> None:
        events = EventBus(clock=lambda: FIXED_NOW)
        registry = ToolRegistry()
        register_voice_tools(registry)
        from agent_core import ToolInvocation, ToolRuntime
        from agent_core.permissions import PermissionDecision

        result = ToolRuntime(registry, events, clock=lambda: FIXED_NOW).execute(
            ToolInvocation(
                task_id="task-1",
                step_id="step-1",
                tool_name=VOICE_NORMALIZE_TOOL_NAME,
                input={"text": "private transcript text"},
            ),
            decision=PermissionDecision.ALLOWED,
        )
        assert result.ok is True
        assert "private transcript text" not in repr(events.history)
        assert events.events_of_type(EventType.TOOL_STARTED)[0].data["input"] == {"redacted": True}
        assert events.events_of_type(EventType.TOOL_COMPLETED)[0].data["output"] == {
            "redacted": True
        }


class TestVoiceEventsAndPrivacy:
    def test_voice_events_are_metadata_only_and_never_contain_audio_payloads(
        self, tmp_path: Path
    ) -> None:
        marker = b"VOICE-RAW-AUDIO-MARKER"
        audio = _audio(payload=marker, duration=0.1)
        agent = _demo_agent(tmp_path)
        events = agent.events
        response = VoiceRuntime(
            MockSTTProvider(text="Run the demo tool."),
            tts_provider=MockTTSProvider(),
            events=events,
            clock=lambda: FIXED_NOW,
        ).process_audio(audio, agent)
        assert response.synthesis_result is not None
        assert response.observation is not None
        assert response.status is VoiceResponseStatus.RESPONDED
        voice_events = [event for event in events.history if event.type.value.startswith("VOICE_")]
        assert {event.type for event in voice_events} >= {
            EventType.VOICE_TRANSCRIPTION_STARTED,
            EventType.VOICE_TRANSCRIPTION_COMPLETED,
            EventType.VOICE_SYNTHESIS_STARTED,
            EventType.VOICE_SYNTHESIS_COMPLETED,
        }
        for event in voice_events:
            assert "text" not in event.data
            assert "payload" not in event.data
            assert "audio" not in event.data
            assert marker.decode() not in repr(event.data)
        assert "VOICE-RAW-AUDIO-MARKER" not in repr(events.history)
        assert "VOICE-RAW-AUDIO-MARKER" not in response_to_dict(response).__repr__()
        assert "payload" not in response_to_dict(response)["synthesis_result"]
        created = events.events_of_type(EventType.TASK_CREATED)[0]
        assert "request" not in created.data
        assert created.data["input_channel"] == "voice"

    def test_audio_payload_cannot_be_serialized_through_observation_or_events(self) -> None:
        marker = b"PRIVATE-MIC-BUFFER"
        audio = _audio(payload=marker)
        provider = MockSTTProvider(text="safe transcript")
        events = EventBus(clock=lambda: FIXED_NOW)
        observation = VoiceRuntime(
            provider,
            events=events,
            clock=lambda: FIXED_NOW,
        ).transcribe(audio)
        assert marker not in repr(observation).encode()
        assert marker.decode() not in observation.model_dump_json()
        assert marker.decode() not in repr(events.history)
        assert "payload" not in observation.model_dump()

    def test_invalid_audio_errors_suppress_raw_payload_validation_details(self) -> None:
        marker = b"PRIVATE-VOICE-AUDIO"
        invalid_audio: Any = {
            "metadata": {
                "duration": 0.1,
                "format": {
                    "sample_rate": len(marker) * 10,
                    "channels": 1,
                    "sample_width": 1,
                    "encoding": "pcm_u8",
                },
                "byte_size": len(marker),
            },
            "payload": marker + b"x",
        }
        events = EventBus(clock=lambda: FIXED_NOW)
        with pytest.raises(VoiceValidationError) as captured:
            VoiceRuntime(MockSTTProvider(), events=events).transcribe(invalid_audio)
        assert captured.value.__cause__ is None
        assert captured.value.__suppress_context__ is True
        assert marker.decode() not in repr(events.history)

    def test_no_network_or_voice_credentials_are_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-voice-test-key")
        provider = MockSTTProvider(text="offline")
        observation = VoiceRuntime(provider).transcribe(_audio())
        synthesized = VoiceRuntime(
            MockSTTProvider(),
            tts_provider=MockTTSProvider(),
        ).synthesize(SynthesisRequest(text="offline response"))
        assert observation.transcription.text == "offline"
        assert synthesized.payload is not None
        assert provider.calls == 1
        assert "synthetic-voice-test-key" not in repr(observation)

    def test_voice_mode_redacts_transcript_from_all_task_and_tool_events(self) -> None:
        transcript = "spoken secret phrase synthetic-sensitive-value"

        class _EchoTool:
            spec = ToolSpec(
                name="echo_voice_text",
                description="Return the input text for a privacy test.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                permission_level=PermissionLevel.LOW,
            )

            def run(self, input: dict[str, Any]) -> ToolResult:
                return ToolResult(ok=True, output={"text": input["text"]})

        class _EchoPlanner:
            def plan(self, request: str, available_tools: Sequence[ToolSpec]) -> Plan:
                assert any(spec.name == "echo_voice_text" for spec in available_tools)
                return Plan(
                    steps=[
                        PlanStep(
                            tool_name="echo_voice_text",
                            description="Echo the request.",
                            input={"text": request},
                        )
                    ]
                )

        events = EventBus(clock=lambda: FIXED_NOW)
        registry = ToolRegistry()
        registry.register(_EchoTool())
        agent = Agent(
            planner=_EchoPlanner(),
            registry=registry,
            permissions=PermissionManager(),
            events=events,
            clock=lambda: FIXED_NOW,
        )
        response = VoiceRuntime(
            MockSTTProvider(text=transcript),
            events=events,
            clock=lambda: FIXED_NOW,
        ).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.NO_RESPONSE
        assert transcript not in repr(events.history)
        for event in events.history:
            if event.type in {EventType.TOOL_STARTED, EventType.TOOL_COMPLETED}:
                assert event.data.get("input", {"redacted": True}) == {"redacted": True}
                assert event.data.get("output", {"redacted": True}) == {"redacted": True}
            if event.type is EventType.TASK_COMPLETED:
                assert "result" not in event.data

    def test_voice_plan_events_sanitize_unregistered_planner_output(self) -> None:
        transcript = "private spoken planner marker"

        class _UnregisteredPlanner:
            def plan(self, request: str, available_tools: Sequence[ToolSpec]) -> Plan:
                assert not available_tools
                return Plan(
                    steps=[
                        PlanStep(
                            tool_name=f"unavailable {request}",
                            description=request,
                            input={"text": request},
                        )
                    ]
                )

        events = EventBus(clock=lambda: FIXED_NOW)
        agent = Agent(
            planner=_UnregisteredPlanner(),
            registry=ToolRegistry(),
            permissions=PermissionManager(),
            events=events,
            clock=lambda: FIXED_NOW,
        )
        response = VoiceRuntime(
            MockSTTProvider(text=transcript),
            events=events,
            clock=lambda: FIXED_NOW,
        ).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.FAILED
        plan_event = events.events_of_type(EventType.PLAN_CREATED)[0]
        assert plan_event.data == {"steps": [{"tool_name": "unavailable_tool"}]}
        assert transcript not in repr(events.history)

    def test_voice_tool_failures_redact_untrusted_error_details_and_codes(self) -> None:
        transcript = "private spoken tool failure code"

        class _FailingTool:
            spec = ToolSpec(
                name="voice_failure_test",
                description="Fail with untrusted details.",
                input_schema={"type": "object", "properties": {}, "required": []},
                output_schema={"type": "object", "properties": {}, "required": []},
                permission_level=PermissionLevel.LOW,
            )

            def run(self, input: dict[str, Any]) -> ToolResult:
                del input
                return ToolResult(ok=False, error=transcript, error_code=transcript)

        class _FailingPlanner:
            def plan(self, request: str, available_tools: Sequence[ToolSpec]) -> Plan:
                del request
                assert any(spec.name == "voice_failure_test" for spec in available_tools)
                return Plan(
                    steps=[
                        PlanStep(
                            tool_name="voice_failure_test",
                            description="Fail.",
                            input={},
                        )
                    ]
                )

        events = EventBus(clock=lambda: FIXED_NOW)
        registry = ToolRegistry()
        registry.register(_FailingTool())
        agent = Agent(
            planner=_FailingPlanner(),
            registry=registry,
            permissions=PermissionManager(),
            events=events,
            clock=lambda: FIXED_NOW,
        )
        response = VoiceRuntime(
            MockSTTProvider(text=transcript),
            events=events,
            clock=lambda: FIXED_NOW,
        ).process_audio(_audio(), agent)
        assert response.status is VoiceResponseStatus.FAILED
        assert transcript not in repr(events.history)
        failure = events.events_of_type(EventType.TOOL_FAILED)[0]
        assert failure.data["error"] == "voice tool failed"
        assert failure.data["error_code"] == "voice_tool_failed"

    def test_provider_failures_never_emit_provider_details(self) -> None:
        class _SensitiveProvider:
            def transcribe(self, audio: AudioInput) -> TranscriptionResult:
                del audio
                raise RuntimeError("token=synthetic-secret-value")

        events = EventBus(clock=lambda: FIXED_NOW)
        with pytest.raises(VoiceProviderError):
            VoiceRuntime(_SensitiveProvider(), events=events).transcribe(_audio())
        errors = events.events_of_type(EventType.VOICE_ERROR)
        assert len(errors) == 1
        assert errors[0].data == {"stage": "transcription", "error_code": "provider_failed"}
        assert "synthetic-secret-value" not in repr(errors[0].data)

    def test_audio_is_never_written_to_persistent_files_by_voice_runtime(
        self, tmp_path: Path
    ) -> None:
        marker = b"MIC-NEVER-PERSIST"
        audio = _audio(payload=marker)
        runtime = VoiceRuntime(MockSTTProvider(text="hello"))
        observation = runtime.transcribe(audio)
        assert observation.transcription.text == "hello"
        assert list(tmp_path.iterdir()) == []

    def test_provider_modules_need_no_platform_audio_or_optional_dependencies(self) -> None:
        assert issubclass(type(MockSTTProvider()), object)
        assert issubclass(type(MockTTSProvider()), object)
        assert "pyaudio" not in sys.modules
        assert "sounddevice" not in sys.modules


def test_voice_observation_requires_aware_ordered_timestamps() -> None:
    transcription = TranscriptionResult(
        text="ok",
        confidence=0.9,
        language="en",
        duration=0.1,
    )
    with pytest.raises(ValidationError):
        VoiceObservation(
            transcription=transcription,
            started_at=datetime(2026, 10, 4),
            completed_at=FIXED_NOW,
        )
    with pytest.raises(ValidationError):
        VoiceObservation(
            transcription=transcription,
            started_at=FIXED_NOW,
            completed_at=datetime(2026, 10, 3, tzinfo=UTC),
        )


def test_voice_response_does_not_claim_uncertain_transcription_succeeded() -> None:
    with pytest.raises(ValidationError):
        VoiceResponse(status=VoiceResponseStatus.UNCERTAIN)


def test_voice_runtime_uses_no_hidden_provider_state() -> None:
    first = MockSTTProvider(text="first")
    second = MockSTTProvider(text="second")
    assert VoiceRuntime(first).transcribe(_audio()).transcription.text == "first"
    assert VoiceRuntime(second).transcribe(_audio()).transcription.text == "second"
