"""Provider-neutral interfaces for ephemeral visual data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .models import (
    BoundingBox,
    ImageFrame,
    VisualAnalysis,
    VisualMatch,
    VisualVerificationCondition,
    VisualVerificationResult,
)

if TYPE_CHECKING:
    from ..computer.models import ScreenshotObservation


class VisionProvider(Protocol):
    """Local/mock implementation contract; receives bytes only in process."""

    def analyze_image(self, image: ImageFrame) -> VisualAnalysis:
        """Return bounded, non-authoritative visual labels or metadata."""
        ...

    def compare_images(
        self,
        before: ImageFrame,
        after: ImageFrame,
        *,
        region: BoundingBox | None,
        max_comparison_pixels: int,
    ) -> VisualMatch:
        """Compare two images deterministically or report uncertainty."""
        ...


class VisualVerifier(Protocol):
    """Action-runtime seam over already-acquired Phase 6 screenshots."""

    @property
    def max_observation_retries(self) -> int:
        """Maximum screenshot-only verification refreshes, bounded by policy."""
        ...

    def validate_screenshot(self, screenshot: ScreenshotObservation) -> ImageFrame:
        """Fail closed before an action if the screenshot cannot be processed."""
        ...

    def verify_screenshots(
        self,
        before: ScreenshotObservation,
        after: ScreenshotObservation,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        """Return an explicit verdict without changing action success policy."""
        ...
