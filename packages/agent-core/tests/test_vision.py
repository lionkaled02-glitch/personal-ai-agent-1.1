"""Deterministic Phase 7 visual observation and verification tests."""

from __future__ import annotations

import struct
import sys
import zlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, NoReturn

import pytest
from agent_core import (
    BoundingBox,
    Bounds,
    ComputerActionResult,
    ComputerActionStatus,
    ComputerRuntime,
    DeterministicVisionProvider,
    EventBus,
    EventType,
    ImageLimitError,
    ImageSize,
    InvalidImageError,
    InvalidRegionError,
    KeyboardKey,
    KeyboardModifier,
    MouseAction,
    MouseActionKind,
    MouseButton,
    OptionalImageSupportError,
    PermissionManager,
    Point,
    ScreenInfo,
    ScreenshotObservation,
    Settings,
    VerificationKind,
    VerificationStatus,
    VisionAnalyzeScreenshotTool,
    VisionLimits,
    VisionRuntime,
    VisionTimeoutError,
    VisualAnalysis,
    VisualMatchStatus,
    VisualRegion,
    VisualVerificationCondition,
    VisualVerificationKind,
    VisualVerificationResult,
    VisualVerificationStatus,
    WindowInfo,
)
from agent_core.vision.errors import ComparisonLimitError, VisionProviderError
from agent_core.vision.models import ImageFrame
from pydantic import ValidationError

FIXED_NOW = datetime(2026, 10, 4, tzinfo=UTC)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _rgb_png(width: int, height: int, pixels: Sequence[tuple[int, int, int]]) -> bytes:
    """Encode a tiny RGB PNG using stdlib only, for platform-independent tests."""
    if len(pixels) != width * height:
        raise ValueError("pixel count does not match PNG dimensions")
    scanlines = bytearray()
    for y in range(height):
        scanlines.append(0)  # PNG filter type: None.
        for red, green, blue in pixels[y * width : (y + 1) * width]:
            scanlines.extend((red, green, blue))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _screenshot(
    payload: bytes,
    *,
    width: int = 2,
    height: int = 2,
) -> ScreenshotObservation:
    return ScreenshotObservation(
        width=width,
        height=height,
        timestamp=FIXED_NOW,
        payload=payload,
    )


def _black_png(width: int = 2, height: int = 2) -> bytes:
    return _rgb_png(width, height, [(0, 0, 0)] * (width * height))


def _changed_png() -> bytes:
    return _rgb_png(2, 2, [(0, 0, 0), (255, 255, 255), (0, 0, 0), (0, 0, 0)])


class _FakeComputerProvider:
    """Small deterministic provider that swaps a synthetic screenshot on click."""

    def __init__(self, before: bytes | None = None, after: bytes | None = None) -> None:
        self.before = before or _black_png()
        self.after = after or _changed_png()
        self.current = self.before
        self.click_calls = 0
        self.screenshot_calls = 0

    def get_screen_info(self) -> ScreenInfo:
        return ScreenInfo(width=2, height=2)

    def get_cursor_position(self) -> Point:
        return Point(x=0, y=0)

    def list_windows(self, *, limit: int) -> Sequence[WindowInfo]:
        return []

    def get_active_window(self) -> WindowInfo | None:
        return WindowInfo(
            identifier="window:1",
            title="Synthetic test window",
            bounds=Bounds(left=0, top=0, right=2, bottom=2),
            visible=True,
            focused=True,
        )

    def focus_window(self, identifier: str) -> None:
        del identifier

    def inspect_ui(self, window_identifier: str | None, *, limit: int) -> Sequence[Any]:
        del window_identifier, limit
        return []

    def select_ui_element(self, window_identifier: str, automation_id: str) -> None:
        del window_identifier, automation_id

    def screenshot(self, *, max_bytes: int) -> ScreenshotObservation:
        del max_bytes
        self.screenshot_calls += 1
        return _screenshot(self.current)

    def move_mouse(self, point: Point, *, duration_s: float) -> None:
        del point, duration_s

    def click(self, point: Point, *, button: MouseButton) -> None:
        del point, button
        self.click_calls += 1
        self.current = self.after

    def double_click(self, point: Point, *, button: MouseButton) -> None:
        self.click(point, button=button)

    def press_key(self, key: KeyboardKey) -> None:
        del key

    def hotkey(self, modifiers: Sequence[KeyboardModifier], key: KeyboardKey) -> None:
        del modifiers, key

    def type_text(self, text: str) -> None:
        del text


class _UncertainOnceVerifier:
    """Return one uncertainty, then delegate to the local deterministic verifier."""

    max_observation_retries = 1

    def __init__(self) -> None:
        self.delegate = VisionRuntime()
        self.calls = 0

    def validate_screenshot(self, screenshot: ScreenshotObservation) -> ImageFrame:
        return self.delegate.validate_screenshot(screenshot)

    def verify_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        self.calls += 1
        if self.calls == 1:
            return VisualVerificationResult(
                condition=condition,
                status=VisualVerificationStatus.UNCERTAIN,
                method="test_uncertain_once",
                reason="insufficient_evidence",
            )
        return self.delegate.verify_screenshots(before, after, condition)


class _AlwaysUncertainVerifier(_UncertainOnceVerifier):
    def verify_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        self.calls += 1
        return VisualVerificationResult(
            condition=condition,
            status=VisualVerificationStatus.UNCERTAIN,
            method="test_uncertain",
            reason="insufficient_evidence",
        )


class _SlowUncertainVerifier(_AlwaysUncertainVerifier):
    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self.clock = clock

    def verify_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        result = super().verify_screenshots(before, after, condition)
        self.clock[0] = 10.0
        return result


class _SensitiveErrorVerifier(_UncertainOnceVerifier):
    def validate_screenshot(self, screenshot: ScreenshotObservation) -> ImageFrame:
        del screenshot
        raise VisionProviderError(code="password_leak", message="do not expose this detail")


class _ProviderWithSensitiveErrors:
    def analyze_image(self, image: ImageFrame) -> NoReturn:
        del image
        raise VisionProviderError(code="password_leak", message="do not expose this detail")

    def compare_images(
        self,
        before: ImageFrame,
        after: ImageFrame,
        *,
        region: BoundingBox | None,
        max_comparison_pixels: int,
    ) -> NoReturn:
        del before, after, region, max_comparison_pixels
        raise VisionProviderError(code="password_leak", message="do not expose this detail")


class _StaticAnalysisProvider:
    def __init__(self, analysis: VisualAnalysis) -> None:
        self.analysis = analysis

    def analyze_image(self, image: ImageFrame) -> VisualAnalysis:
        del image
        return self.analysis

    def compare_images(
        self,
        before: ImageFrame,
        after: ImageFrame,
        *,
        region: BoundingBox | None,
        max_comparison_pixels: int,
    ) -> NoReturn:
        del before, after, region, max_comparison_pixels
        raise AssertionError("comparison is not part of this test")


class TestVisualModelsAndLimits:
    def test_bounding_box_and_image_size_are_bounded(self) -> None:
        assert ImageSize(width=2, height=2).pixels == 4
        assert BoundingBox(x=1, y=1, width=1, height=1).fits_within(ImageSize(width=2, height=2))
        with pytest.raises(ValidationError):
            BoundingBox(x=8_192, y=0, width=1, height=1)
        with pytest.raises(ValidationError):
            ImageSize(width=0, height=1)

    def test_visual_conditions_require_only_their_supported_region_shape(self) -> None:
        with pytest.raises(ValidationError):
            VisualVerificationCondition(kind=VisualVerificationKind.REGION_CHANGED)
        with pytest.raises(ValidationError):
            VisualVerificationCondition(
                kind=VisualVerificationKind.SCREENSHOT_CHANGED,
                region=BoundingBox(x=0, y=0, width=1, height=1),
            )
        condition = VisualVerificationCondition(
            kind=VisualVerificationKind.REGION_CHANGED,
            region=BoundingBox(x=0, y=0, width=1, height=1),
            minimum_change_ratio=0.25,
        )
        assert condition.minimum_change_ratio == 0.25
        with pytest.raises(ValidationError):
            VisualVerificationResult(
                condition=VisualVerificationCondition(
                    kind=VisualVerificationKind.SCREENSHOT_CHANGED
                ),
                status=VisualVerificationStatus.VERIFIED,
                method="pixel_comparison",
                reason="pixel_change_observed",
            )

    def test_limits_are_environment_configurable_and_fail_closed(self) -> None:
        settings = Settings.from_env(
            {
                "VISION_MAX_IMAGE_BYTES": "4096",
                "VISION_MAX_COMPARISON_PIXELS": "256",
                "VISION_MAX_OBSERVATION_RETRIES": "2",
            }
        )
        limits = VisionLimits.from_settings(settings)
        assert limits.max_image_bytes == 4_096
        assert limits.max_comparison_pixels == 256
        assert limits.max_observation_retries == 2
        with pytest.raises(ValidationError):
            VisionLimits.from_settings(Settings(vision_max_observation_retries=99))


class TestEphemeralVisionRuntime:
    def test_analysis_returns_bounded_metadata_without_image_bytes_or_event_payload(self) -> None:
        payload = _black_png()
        events = EventBus(clock=lambda: FIXED_NOW)
        runtime = VisionRuntime(events=events, clock=lambda: FIXED_NOW)
        screenshot = _screenshot(payload)

        frame = runtime.validate_screenshot(screenshot)
        observation = runtime.analyze_screenshot(screenshot)

        assert payload not in repr(screenshot).encode()
        assert frame.payload == payload
        assert "payload" not in frame.model_dump(mode="json")
        assert payload not in repr(frame).encode()
        assert observation.image_size == ImageSize(width=2, height=2)
        assert observation.source == "DeterministicVisionProvider"
        assert observation.untrusted_content is True
        assert observation.regions == []
        assert "semantic content is not interpreted" in (observation.summary or "")
        assert "payload" not in observation.model_dump(mode="json")
        assert payload not in repr(observation).encode()
        assert [event.type for event in events.history] == [EventType.VISION_ANALYZED]
        assert all(payload not in repr(event.data).encode() for event in events.history)
        assert all("image_base64" not in event.data for event in events.history)

    def test_bad_png_and_image_limits_are_rejected(self) -> None:
        runtime = VisionRuntime()
        with pytest.raises(InvalidImageError):
            runtime.validate_screenshot(_screenshot(b"not a png"))

        corrupt_header = bytearray(_black_png())
        corrupt_header[29] ^= 1  # Corrupt the IHDR CRC.
        with pytest.raises(InvalidImageError):
            runtime.validate_screenshot(_screenshot(bytes(corrupt_header)))

        payload = _black_png()
        limited = VisionRuntime(limits=VisionLimits(max_image_bytes=len(payload) - 1))
        with pytest.raises(ImageLimitError):
            limited.validate_screenshot(_screenshot(payload))

        wrong_size = ScreenshotObservation(
            width=3,
            height=2,
            timestamp=FIXED_NOW,
            payload=payload,
        )
        with pytest.raises(InvalidImageError):
            runtime.validate_screenshot(wrong_size)

    def test_deterministic_compare_and_region_metrics(self) -> None:
        pytest.importorskip("PIL")
        runtime = VisionRuntime()
        before = _screenshot(_black_png())
        after = _screenshot(_changed_png())
        before_frame = runtime.validate_screenshot(before)
        provider = DeterministicVisionProvider()

        identical = provider.compare_images(
            before_frame,
            before_frame,
            region=None,
            max_comparison_pixels=1,
        )
        assert identical.status is VisualMatchStatus.IDENTICAL
        assert identical.similarity == 1.0
        assert identical.changed_ratio == 0.0
        unchanged_result = runtime.verify_screenshots(
            before,
            before,
            VisualVerificationCondition(kind=VisualVerificationKind.SCREENSHOT_CHANGED),
        )
        assert unchanged_result.status is VisualVerificationStatus.FAILED

        changed = runtime.compare_screenshots(before, after)
        assert changed.status is VisualMatchStatus.CHANGED
        assert changed.changed_ratio == 0.25
        assert changed.similarity == 0.75
        assert changed.changed_region == BoundingBox(x=1, y=0, width=1, height=1)

        region_condition = VisualVerificationCondition(
            kind=VisualVerificationKind.REGION_CHANGED,
            region=BoundingBox(x=1, y=0, width=1, height=1),
        )
        region_result = runtime.verify_screenshots(before, after, region_condition)
        assert region_result.status is VisualVerificationStatus.VERIFIED
        assert region_result.match is not None
        assert region_result.match.changed_ratio == 1.0

    def test_nonidentical_comparison_reports_optional_pillow_support(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = VisionRuntime()
        monkeypatch.setitem(sys.modules, "PIL", None)
        with pytest.raises(OptionalImageSupportError):
            runtime.compare_screenshots(_screenshot(_black_png()), _screenshot(_changed_png()))

    def test_comparison_region_and_work_caps_fail_closed(self) -> None:
        pytest.importorskip("PIL")
        runtime = VisionRuntime()
        before = _screenshot(_black_png())
        after = _screenshot(_changed_png())
        before_frame = runtime.validate_screenshot(before)
        after_frame = runtime.validate_screenshot(after)
        provider = DeterministicVisionProvider()

        with pytest.raises(InvalidRegionError):
            provider.compare_images(
                before_frame,
                after_frame,
                region=BoundingBox(x=1, y=0, width=2, height=1),
                max_comparison_pixels=4,
            )
        with pytest.raises(ComparisonLimitError):
            provider.compare_images(
                before_frame,
                after_frame,
                region=None,
                max_comparison_pixels=3,
            )
        uncertain = VisionRuntime(limits=VisionLimits(max_comparison_pixels=3))
        result = uncertain.verify_screenshots(
            before,
            after,
            VisualVerificationCondition(kind=VisualVerificationKind.SCREENSHOT_CHANGED),
        )
        assert result.status is VisualVerificationStatus.UNCERTAIN
        assert result.reason == "comparison_limit_exceeded"

    def test_provider_region_label_and_summary_limits_are_enforced(self) -> None:
        region = VisualRegion(
            region_id="test-region",
            bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
            label="button-label",
        )
        cases = (
            (
                VisionLimits(max_regions=0),
                VisualAnalysis(method="test_analysis", regions=[region]),
                "analysis_region_limit_exceeded",
            ),
            (
                VisionLimits(max_label_chars=3),
                VisualAnalysis(method="test_analysis", regions=[region]),
                "analysis_label_limit_exceeded",
            ),
            (
                VisionLimits(max_summary_chars=3),
                VisualAnalysis(method="test_analysis", summary="too-long"),
                "analysis_summary_limit_exceeded",
            ),
        )
        for limits, analysis, expected_code in cases:
            runtime = VisionRuntime(
                provider=_StaticAnalysisProvider(analysis),
                limits=limits,
            )
            with pytest.raises(VisionProviderError) as error:
                runtime.analyze_screenshot(_screenshot(_black_png()))
            assert error.value.code == expected_code

    def test_metadata_analysis_does_not_require_optional_pixel_decoder(self) -> None:
        runtime = VisionRuntime()
        observation = runtime.analyze_screenshot(_screenshot(_black_png()))
        assert observation.image_size == ImageSize(width=2, height=2)

    def test_provider_errors_are_stable_and_do_not_leak_messages(self) -> None:
        runtime = VisionRuntime(provider=_ProviderWithSensitiveErrors())
        screenshot = _screenshot(_black_png())
        with pytest.raises(VisionProviderError) as analysis_error:
            runtime.analyze_screenshot(screenshot)
        assert analysis_error.value.code == "analysis_failed"
        assert "do not expose" not in analysis_error.value.public_message

        with pytest.raises(VisionProviderError) as comparison_error:
            runtime.compare_screenshots(screenshot, screenshot)
        assert comparison_error.value.code == "comparison_failed"
        assert "password_leak" not in comparison_error.value.code
        assert "do not expose" not in comparison_error.value.public_message

    def test_operation_time_limit_uses_injected_monotonic_clock(self) -> None:
        ticks = iter((0.0, 2.0))
        runtime = VisionRuntime(
            limits=VisionLimits(max_operation_seconds=1.0),
            monotonic=lambda: next(ticks),
        )
        with pytest.raises(VisionTimeoutError):
            runtime.analyze_screenshot(_screenshot(_black_png()))


class TestVisionToolAndComputerLifecycle:
    def test_vision_tool_reuses_computer_screenshot_and_never_returns_bytes(self) -> None:
        payload = _black_png()
        provider = _FakeComputerProvider(before=payload, after=payload)
        events = EventBus(clock=lambda: FIXED_NOW)
        computer = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(),
            events=events,
            clock=lambda: FIXED_NOW,
        )
        tool = VisionAnalyzeScreenshotTool(
            computer, VisionRuntime(events=events, clock=lambda: FIXED_NOW)
        )

        result = tool.run({})

        assert result.ok is True
        assert isinstance(result.output, dict)
        output = result.output
        assert output["ok"] is True
        assert "observation" in output
        serialized = repr(output).encode()
        assert payload not in serialized
        assert "image_base64" not in repr(output)
        assert provider.screenshot_calls == 1
        assert all(payload not in repr(event.data).encode() for event in events.history)

    def test_visual_pixel_change_alone_does_not_verify_a_click(self) -> None:
        pytest.importorskip("PIL")
        provider = _FakeComputerProvider()
        events = EventBus(clock=lambda: FIXED_NOW)
        runtime = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(approval=lambda _request: True),
            events=events,
            clock=lambda: FIXED_NOW,
            visual_verifier=VisionRuntime(events=events, clock=lambda: FIXED_NOW),
        )
        condition = VisualVerificationCondition(kind=VisualVerificationKind.SCREENSHOT_CHANGED)

        result = runtime.click(
            MouseAction(kind=MouseActionKind.CLICK, point=Point(x=1, y=1)),
            visual_verification=condition,
        )

        assert result.visual_verification is not None
        assert result.visual_verification.status is VisualVerificationStatus.VERIFIED
        assert result.status is ComputerActionStatus.UNVERIFIED
        assert result.success is False
        assert result.attempts == 1
        assert provider.click_calls == 1
        assert result.observed_before is not None
        assert result.observed_before.screenshot is not None
        assert result.observed_before.screenshot.payload is None
        assert result.observed_after is not None
        assert result.observed_after.screenshot is not None
        assert result.observed_after.screenshot.payload is None

        forged = result.model_dump(mode="python")
        forged["status"] = ComputerActionStatus.VERIFIED
        forged["success"] = True
        forged["verified"] = True
        verification = forged["verification"]
        assert isinstance(verification, dict)
        verification["status"] = VerificationStatus.PASSED
        verification["condition"] = VerificationKind.SCREENSHOT_CHANGED
        with pytest.raises(ValidationError):
            ComputerActionResult.model_validate(forged)

    def test_elapsed_visual_verification_timeout_does_not_report_action_success(self) -> None:
        provider = _FakeComputerProvider()
        clock = [0.0]
        runtime = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(approval=lambda _request: True),
            events=EventBus(clock=lambda: FIXED_NOW),
            clock=lambda: FIXED_NOW,
            monotonic=lambda: clock[0],
            visual_verifier=_SlowUncertainVerifier(clock),
        )
        result = runtime.click(
            MouseAction(kind=MouseActionKind.CLICK, point=Point(x=1, y=1)),
            visual_verification=VisualVerificationCondition(
                kind=VisualVerificationKind.SCREENSHOT_CHANGED
            ),
        )

        assert result.status is ComputerActionStatus.TIMED_OUT
        assert result.success is False
        assert result.attempts == 1
        assert provider.click_calls == 1
        assert result.visual_verification is not None
        assert result.visual_verification.status is VisualVerificationStatus.UNCERTAIN

    def test_visual_uncertainty_refreshes_only_observation_without_replaying_action(self) -> None:
        provider = _FakeComputerProvider()
        verifier = _AlwaysUncertainVerifier()
        runtime = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(approval=lambda _request: True),
            events=EventBus(clock=lambda: FIXED_NOW),
            clock=lambda: FIXED_NOW,
            visual_verifier=verifier,
        )
        condition = VisualVerificationCondition(kind=VisualVerificationKind.SCREENSHOT_CHANGED)

        result = runtime.click(
            MouseAction(kind=MouseActionKind.CLICK, point=Point(x=1, y=1)),
            visual_verification=condition,
        )

        assert result.visual_verification is not None
        assert result.visual_verification.status is VisualVerificationStatus.UNCERTAIN
        assert result.success is False
        assert result.status is ComputerActionStatus.UNVERIFIED
        assert result.attempts == 1
        assert provider.click_calls == 1
        assert verifier.calls == 2  # Initial comparison plus one bounded refresh.
        assert len(result.recovery_observations) == 1
        assert result.recovery_observations[0].screenshot is not None
        assert result.recovery_observations[0].screenshot.payload is None
        unsafe_payload = result.model_dump(mode="python")
        unsafe_payload["success"] = True
        unsafe_payload["verified"] = True
        unsafe_payload["status"] = ComputerActionStatus.VERIFIED
        with pytest.raises(ValidationError):
            ComputerActionResult.model_validate(unsafe_payload)

    def test_visual_condition_without_configured_verifier_fails_before_action(self) -> None:
        provider = _FakeComputerProvider()
        runtime = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(approval=lambda _request: True),
            events=EventBus(clock=lambda: FIXED_NOW),
            clock=lambda: FIXED_NOW,
        )

        result = runtime.click(
            MouseAction(kind=MouseActionKind.CLICK, point=Point(x=1, y=1)),
            visual_verification=VisualVerificationCondition(
                kind=VisualVerificationKind.SCREENSHOT_CHANGED
            ),
        )

        assert result.status is ComputerActionStatus.INVALID
        assert result.error_code == "visual_verifier_unavailable"
        assert result.success is False
        assert provider.click_calls == 0

    def test_visual_provider_errors_are_sanitized_before_action(self) -> None:
        provider = _FakeComputerProvider()
        runtime = ComputerRuntime(
            provider=provider,
            permissions=PermissionManager(approval=lambda _request: True),
            events=EventBus(clock=lambda: FIXED_NOW),
            clock=lambda: FIXED_NOW,
            visual_verifier=_SensitiveErrorVerifier(),
        )

        result = runtime.click(
            MouseAction(kind=MouseActionKind.CLICK, point=Point(x=1, y=1)),
            visual_verification=VisualVerificationCondition(
                kind=VisualVerificationKind.SCREENSHOT_CHANGED
            ),
        )

        assert result.status is ComputerActionStatus.INVALID
        assert result.error_code == "visual_provider_failed"
        assert result.error == "Visual verification could not be completed."
        assert result.visual_verification is not None
        assert result.visual_verification.reason == "visual_provider_failed"
        assert "password_leak" not in repr(result)
        assert provider.click_calls == 0


class TestVisionProviderBoundary:
    def test_no_semantic_model_or_remote_provider_is_registered_by_default(self) -> None:
        runtime = VisionRuntime()
        assert isinstance(runtime._provider, DeterministicVisionProvider)
        with pytest.raises(VisionProviderError):
            runtime.compare_screenshots(
                _screenshot(_black_png()),
                _screenshot(_changed_png()),
                region="not-a-bounding-box",  # type: ignore[arg-type]
            )
