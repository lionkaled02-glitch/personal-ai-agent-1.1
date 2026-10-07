"""Deterministic, bounded local image analysis and pixel comparison."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from .errors import (
    ComparisonLimitError,
    InvalidRegionError,
    OptionalImageSupportError,
    VisionComparisonError,
)
from .interfaces import VisionProvider
from .models import (
    BoundingBox,
    ImageFrame,
    RegionSource,
    VisualAnalysis,
    VisualMatch,
    VisualMatchStatus,
    VisualRegion,
)

if TYPE_CHECKING:
    from PIL import Image as PILImage


class DeterministicVisionProvider:
    """Offline metadata analyser and exact RGB comparator.

    This implementation intentionally does not recognize objects, read text,
    infer clicks, or attach meaning to regions. Pixel differences are evidence
    of changed pixels only. Pillow is loaded lazily and is required only when
    a non-identical image pair must be decoded.
    """

    name = "deterministic-local"

    def analyze_image(self, image: ImageFrame) -> VisualAnalysis:
        """Return metadata only; no semantic interpretation or OCR is performed."""
        return VisualAnalysis(
            summary="PNG header and dimensions validated; semantic content is not interpreted.",
            regions=[],
            confidence=None,
            method="metadata_only",
        )

    def compare_images(
        self,
        before: ImageFrame,
        after: ImageFrame,
        *,
        region: BoundingBox | None,
        max_comparison_pixels: int,
    ) -> VisualMatch:
        """Compare equal-coordinate RGB pixels under a strict work bound."""
        if region is not None:
            if not region.fits_within(before.size) or not region.fits_within(after.size):
                raise InvalidRegionError()
            pixels = region.width * region.height
        elif before.size == after.size:
            pixels = before.size.pixels
        else:
            return VisualMatch(
                status=VisualMatchStatus.CHANGED,
                similarity=0.0,
                changed_ratio=1.0,
                reason="image_dimensions_changed",
                method="dimension_comparison",
                before_sha256=before.sha256,
                after_sha256=after.sha256,
            )

        if (
            before.size == after.size
            and before.sha256 == after.sha256
            and before.payload == after.payload
        ):
            return VisualMatch(
                status=VisualMatchStatus.IDENTICAL,
                similarity=1.0,
                changed_ratio=0.0,
                reason="image_payloads_identical",
                method="payload_exact",
                before_sha256=before.sha256,
                after_sha256=after.sha256,
            )

        if pixels > max_comparison_pixels:
            raise ComparisonLimitError()

        first = self._decode_rgb(before)
        second = self._decode_rgb(after)
        if region is None:
            crop = (0, 0, before.size.width, before.size.height)
            origin_x = origin_y = 0
        else:
            crop = (
                region.x,
                region.y,
                region.x + region.width,
                region.y + region.height,
            )
            origin_x, origin_y = region.x, region.y

        first_bytes = first.crop(crop).tobytes()
        second_bytes = second.crop(crop).tobytes()
        if len(first_bytes) != len(second_bytes):
            raise VisionComparisonError()

        changed_pixels = 0
        absolute_error = 0
        min_x = pixels
        min_y = pixels
        max_x = max_y = -1
        width = region.width if region is not None else before.size.width
        for pixel_index in range(pixels):
            offset = pixel_index * 3
            red_delta = abs(first_bytes[offset] - second_bytes[offset])
            green_delta = abs(first_bytes[offset + 1] - second_bytes[offset + 1])
            blue_delta = abs(first_bytes[offset + 2] - second_bytes[offset + 2])
            pixel_error = red_delta + green_delta + blue_delta
            absolute_error += pixel_error
            if pixel_error:
                changed_pixels += 1
                x = pixel_index % width
                y = pixel_index // width
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        changed_ratio = changed_pixels / pixels
        similarity = 1.0 - absolute_error / (pixels * 3 * 255)
        changed_region = (
            BoundingBox(
                x=origin_x + min_x,
                y=origin_y + min_y,
                width=max_x - min_x + 1,
                height=max_y - min_y + 1,
            )
            if changed_pixels
            else None
        )
        is_changed = changed_pixels > 0
        return VisualMatch(
            status=VisualMatchStatus.CHANGED if is_changed else VisualMatchStatus.IDENTICAL,
            similarity=max(0.0, min(1.0, similarity)),
            changed_ratio=changed_ratio,
            changed_region=changed_region,
            reason="pixel_values_changed" if is_changed else "decoded_pixels_identical",
            method="rgb_pixel_difference",
            before_sha256=before.sha256,
            after_sha256=after.sha256,
        )

    @staticmethod
    def _decode_rgb(frame: ImageFrame) -> PILImage.Image:
        try:
            from PIL import Image
        except ImportError as exc:
            raise OptionalImageSupportError() from exc
        try:
            with Image.open(BytesIO(frame.payload)) as decoded:
                if decoded.format != "PNG" or decoded.size != (frame.size.width, frame.size.height):
                    raise VisionComparisonError()
                decoded.load()
                return decoded.convert("RGB").copy()
        except VisionComparisonError:
            raise
        except Exception as exc:
            # Do not forward decoder errors: they may contain provider details.
            raise VisionComparisonError() from exc


# A narrow structural check useful to callers without importing an adapter.
DeterministicVisionBackend: type[VisionProvider] = DeterministicVisionProvider


def difference_region(match: VisualMatch) -> VisualRegion | None:
    """Adapt a pixel-derived bounding region without assigning semantic labels."""
    if match.changed_region is None:
        return None
    return VisualRegion(
        region_id="pixel-difference",
        bounding_box=match.changed_region,
        label=None,
        confidence=None,
        source=RegionSource.PIXEL_COMPARISON,
    )
