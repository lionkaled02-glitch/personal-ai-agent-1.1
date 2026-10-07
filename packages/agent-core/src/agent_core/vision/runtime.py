"""Ephemeral screenshot analysis and bounded visual verification runtime."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from ..computer.models import ScreenshotObservation
from ..events import Clock, EventBus, EventType, utc_now
from .comparison import DeterministicVisionProvider
from .errors import (
    ComparisonLimitError,
    InvalidRegionError,
    OptionalImageSupportError,
    VisionComparisonError,
    VisionError,
    VisionProviderError,
    VisionTimeoutError,
)
from .interfaces import VisionProvider
from .limits import VisionLimits
from .models import (
    BoundingBox,
    ImageFrame,
    VisualAnalysis,
    VisualMatch,
    VisualMatchStatus,
    VisualObservation,
    VisualVerificationCondition,
    VisualVerificationResult,
    VisualVerificationStatus,
)
from .normalization import normalize_screenshot_image
from .verification import verify_visual_match

_SAFE_SOURCE = re.compile(r"[^A-Za-z0-9._-]")


class VisionRuntime:
    """Orchestrate validated, in-memory visual analysis over Phase 6 images.

    The runtime never acquires screenshots itself and never retains frames or
    image bytes after a call. Provider outputs are revalidated and event data
    contains only bounded operational metadata.
    """

    def __init__(
        self,
        provider: VisionProvider | None = None,
        limits: VisionLimits | None = None,
        events: EventBus | None = None,
        clock: Clock | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._provider: VisionProvider = provider or DeterministicVisionProvider()
        self._limits = limits or VisionLimits()
        self._events = events
        self._clock: Clock = clock or utc_now
        self._monotonic = monotonic or time.monotonic

    @property
    def limits(self) -> VisionLimits:
        return self._limits

    @property
    def max_observation_retries(self) -> int:
        return self._limits.max_observation_retries

    def validate_screenshot(self, screenshot: ScreenshotObservation) -> ImageFrame:
        """Validate the existing screenshot before permitting a visual action."""
        return normalize_screenshot_image(screenshot, self._limits)

    def analyze_screenshot(self, screenshot: ScreenshotObservation) -> VisualObservation:
        """Return a bounded metadata/region observation without retaining bytes."""
        frame = self.validate_screenshot(screenshot)
        started = self._monotonic()
        try:
            raw_analysis = self._provider.analyze_image(frame)
            analysis = VisualAnalysis.model_validate(raw_analysis)
        except Exception as exc:
            # Providers are an untrusted extension boundary. Do not expose
            # arbitrary exception types, codes, or messages to callers.
            raise VisionProviderError(code="analysis_failed") from exc
        self._check_deadline(started)
        self._validate_analysis(analysis, frame)
        analysis = analysis.model_copy(
            update={"method": self._safe_method(analysis.method, "provider_analysis")}
        )
        observation = VisualObservation(
            image_size=frame.size,
            timestamp=self._clock(),
            regions=analysis.regions,
            source=self._provider_source(),
            summary=analysis.summary,
            image_sha256=frame.sha256,
            metadata={"method": analysis.method, "confidence": analysis.confidence},
        )
        self._emit(
            EventType.VISION_ANALYZED,
            {
                "source": observation.source,
                "method": analysis.method,
                "region_count": len(analysis.regions),
                "image_width": frame.size.width,
                "image_height": frame.size.height,
            },
        )
        return observation

    def compare_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        *,
        region: BoundingBox | None = None,
    ) -> VisualMatch:
        """Compare two existing screenshots using a bounded provider method."""
        if region is not None and not isinstance(region, BoundingBox):
            raise VisionProviderError(code="invalid_comparison_region")
        before_frame = self.validate_screenshot(before)
        after_frame = self.validate_screenshot(after)
        started = self._monotonic()
        try:
            raw_match = self._provider.compare_images(
                before_frame,
                after_frame,
                region=region,
                max_comparison_pixels=self._limits.max_comparison_pixels,
            )
            match = VisualMatch.model_validate(raw_match)
        except VisionError as exc:
            raise self._canonical_comparison_error(exc) from exc
        except Exception as exc:
            raise VisionProviderError(code="comparison_failed") from exc
        self._check_deadline(started)
        match = match.model_copy(
            update={
                "method": self._safe_method(match.method, "provider_comparison"),
                "reason": self._safe_reason(match.reason, "provider_result"),
            }
        )
        if match.before_sha256 != before_frame.sha256 or match.after_sha256 != after_frame.sha256:
            raise VisionProviderError(code="comparison_evidence_mismatch")
        if match.changed_region is not None and (
            not match.changed_region.fits_within(before_frame.size)
            or not match.changed_region.fits_within(after_frame.size)
            or (
                region is not None
                and (
                    match.changed_region.x < region.x
                    or match.changed_region.y < region.y
                    or match.changed_region.x + match.changed_region.width > region.x + region.width
                    or match.changed_region.y + match.changed_region.height
                    > region.y + region.height
                )
            )
        ):
            raise VisionProviderError(code="comparison_region_invalid")
        self._emit(
            EventType.VISION_COMPARISON_COMPLETED,
            {
                "method": match.method,
                "status": match.status.value,
                "similarity": match.similarity,
                "changed_ratio": match.changed_ratio,
            },
        )
        return match

    def verify_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        """Return VERIFIED/FAILED/UNCERTAIN for an explicit pixel predicate."""
        try:
            condition = VisualVerificationCondition.model_validate(condition)
        except Exception as exc:
            raise VisionProviderError(code="invalid_visual_condition") from exc
        try:
            match = self.compare_screenshots(before, after, region=condition.region)
            result = verify_visual_match(match, condition)
        except VisionError as exc:
            result = VisualVerificationResult(
                condition=condition,
                status=VisualVerificationStatus.UNCERTAIN,
                method="unavailable",
                confidence=None,
                reason=exc.code[:256],
                evidence_ref=(
                    f"sha256:{after.payload_sha256}" if after.payload_sha256 is not None else None
                ),
                match=VisualMatch(
                    status=VisualMatchStatus.UNCERTAIN,
                    similarity=None,
                    changed_ratio=None,
                    reason=exc.code[:256],
                    method="unavailable",
                    before_sha256=before.payload_sha256 or "0" * 64,
                    after_sha256=after.payload_sha256 or "0" * 64,
                ),
            )
        self._emit(
            EventType.VISION_VERIFICATION_COMPLETED,
            {
                "status": result.status.value,
                "method": result.method,
                "reason": result.reason,
            },
        )
        return result

    def _validate_analysis(self, analysis: VisualAnalysis, frame: ImageFrame) -> None:
        if len(analysis.regions) > self._limits.max_regions:
            raise VisionProviderError(code="analysis_region_limit_exceeded")
        if analysis.summary is not None and len(analysis.summary) > self._limits.max_summary_chars:
            raise VisionProviderError(code="analysis_summary_limit_exceeded")
        for region in analysis.regions:
            if region.label is not None and len(region.label) > self._limits.max_label_chars:
                raise VisionProviderError(code="analysis_label_limit_exceeded")
            if not region.bounding_box.fits_within(frame.size):
                raise VisionProviderError(code="analysis_region_out_of_bounds")

    def _check_deadline(self, started: float) -> None:
        if self._monotonic() - started > self._limits.max_operation_seconds:
            raise VisionTimeoutError()

    @staticmethod
    def _safe_method(value: str, fallback: str) -> str:
        return value if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) else fallback

    @staticmethod
    def _safe_reason(value: str, fallback: str) -> str:
        return value if re.fullmatch(r"[a-z][a-z0-9_]{0,255}", value) else fallback

    @staticmethod
    def _canonical_comparison_error(error: VisionError) -> VisionError:
        """Rebuild only known provider errors with canonical public messages."""
        if error.code == "invalid_region":
            return InvalidRegionError()
        if error.code == "comparison_limit_exceeded":
            return ComparisonLimitError()
        if error.code == "optional_image_support_unavailable":
            return OptionalImageSupportError()
        if error.code == "vision_comparison_failed":
            return VisionComparisonError()
        return VisionProviderError(code="comparison_failed")

    def _provider_source(self) -> str:
        source = type(self._provider).__name__
        source = _SAFE_SOURCE.sub("_", source)[:64]
        return source or "vision_provider"

    def _emit(self, event_type: EventType, data: dict[str, object]) -> None:
        if self._events is not None:
            self._events.emit(event_type, data=data)
