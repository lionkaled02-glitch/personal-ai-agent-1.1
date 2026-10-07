"""Hard caps for image observation and deterministic comparison."""

from __future__ import annotations

from pydantic import Field

from ..config import Settings
from .models import MAX_IMAGE_BYTES, MAX_IMAGE_DIMENSION, VisionModel


class VisionLimits(VisionModel):
    """Runtime caps; callers may tighten them but cannot exceed model limits.

    The elapsed-time ceiling is cooperative: synchronous decoding/provider
    calls are checked after returning and cannot be forcibly interrupted.
    """

    max_image_bytes: int = Field(default=MAX_IMAGE_BYTES, ge=1, le=MAX_IMAGE_BYTES, strict=True)
    max_image_width: int = Field(default=4_096, ge=1, le=MAX_IMAGE_DIMENSION, strict=True)
    max_image_height: int = Field(default=4_096, ge=1, le=MAX_IMAGE_DIMENSION, strict=True)
    max_image_pixels: int = Field(default=16_777_216, ge=1, le=16_777_216, strict=True)
    max_regions: int = Field(default=100, ge=0, le=100, strict=True)
    max_label_chars: int = Field(default=128, ge=1, le=128, strict=True)
    max_summary_chars: int = Field(default=512, ge=1, le=512, strict=True)
    max_comparison_pixels: int = Field(default=1_048_576, ge=1, le=8_388_608, strict=True)
    max_observation_retries: int = Field(default=1, ge=0, le=2, strict=True)
    max_operation_seconds: float = Field(
        default=5.0, gt=0.0, le=30.0, strict=True, allow_inf_nan=False
    )

    @classmethod
    def from_settings(cls, settings: Settings) -> VisionLimits:
        """Build validated visual limits from the shared non-secret settings."""
        return cls(
            max_image_bytes=settings.vision_max_image_bytes,
            max_image_width=settings.vision_max_image_width,
            max_image_height=settings.vision_max_image_height,
            max_image_pixels=settings.vision_max_image_pixels,
            max_regions=settings.vision_max_regions,
            max_label_chars=settings.vision_max_label_chars,
            max_summary_chars=settings.vision_max_summary_chars,
            max_comparison_pixels=settings.vision_max_comparison_pixels,
            max_observation_retries=settings.vision_max_observation_retries,
            max_operation_seconds=settings.vision_max_operation_seconds,
        )
