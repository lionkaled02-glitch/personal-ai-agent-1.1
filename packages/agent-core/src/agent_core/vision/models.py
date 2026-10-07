"""Bounded provider-neutral models for visual observation and comparison.

Visual labels and summaries are untrusted external data, never instructions.
Image payloads are carried only by the ephemeral ``ImageFrame`` passed to a
provider and are explicitly excluded from its representation/serialization.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_IMAGE_DIMENSION = 8_192
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_VISUAL_REGIONS = 500
MAX_VISUAL_LABEL_CHARS = 256
MAX_VISUAL_SUMMARY_CHARS = 1_024
SHA256_PATTERN = r"^[0-9a-f]{64}$"
SAFE_TOKEN_PATTERN = r"^[a-z][a-z0-9_]{0,255}$"
SAFE_METHOD_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
REGION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"


class VisionModel(BaseModel):
    """Strict, immutable base for public vision data models."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class ImageSize(VisionModel):
    """Bounded pixel dimensions for an image."""

    width: int = Field(gt=0, le=MAX_IMAGE_DIMENSION, strict=True)
    height: int = Field(gt=0, le=MAX_IMAGE_DIMENSION, strict=True)

    @property
    def pixels(self) -> int:
        return self.width * self.height


class BoundingBox(VisionModel):
    """An integer rectangle in image pixel coordinates."""

    x: int = Field(ge=0, le=MAX_IMAGE_DIMENSION, strict=True)
    y: int = Field(ge=0, le=MAX_IMAGE_DIMENSION, strict=True)
    width: int = Field(gt=0, le=MAX_IMAGE_DIMENSION, strict=True)
    height: int = Field(gt=0, le=MAX_IMAGE_DIMENSION, strict=True)

    @model_validator(mode="after")
    def check_maximum_extent(self) -> BoundingBox:
        if self.x + self.width > MAX_IMAGE_DIMENSION:
            raise ValueError("bounding box exceeds maximum image width")
        if self.y + self.height > MAX_IMAGE_DIMENSION:
            raise ValueError("bounding box exceeds maximum image height")
        return self

    def fits_within(self, image_size: ImageSize) -> bool:
        """Whether the rectangle lies wholly within ``image_size``."""
        return self.x + self.width <= image_size.width and self.y + self.height <= image_size.height


class RegionSource(StrEnum):
    PROVIDER = "provider"
    PIXEL_COMPARISON = "pixel_comparison"
    USER_SPECIFIED = "user_specified"


class VisualRegion(VisionModel):
    """One bounded screenshot region; labels remain untrusted data."""

    region_id: str = Field(pattern=REGION_ID_PATTERN)
    bounding_box: BoundingBox
    label: str | None = Field(default=None, max_length=MAX_VISUAL_LABEL_CHARS)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    source: RegionSource = RegionSource.PROVIDER


class ImageFrame(VisionModel):
    """Ephemeral normalized image passed to a vision provider.

    The image bytes are deliberately excluded from repr and model dumps. A
    frame is an in-process value only; it has no persistence or upload path.
    """

    size: ImageSize
    media_type: Literal["image/png"] = "image/png"
    payload_bytes: int = Field(ge=1, le=MAX_IMAGE_BYTES, strict=True)
    sha256: str = Field(pattern=SHA256_PATTERN)
    payload: bytes = Field(repr=False, exclude=True)

    @model_validator(mode="after")
    def validate_payload(self) -> ImageFrame:
        if len(self.payload) != self.payload_bytes:
            raise ValueError("image payload length does not match metadata")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("image payload digest does not match")
        return self


class VisualAnalysis(VisionModel):
    """A provider's bounded visual-analysis output; it is not trusted policy."""

    untrusted_content: Literal[True] = True
    summary: str | None = Field(default=None, max_length=MAX_VISUAL_SUMMARY_CHARS)
    regions: list[VisualRegion] = Field(default_factory=list, max_length=MAX_VISUAL_REGIONS)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    method: str = Field(pattern=SAFE_METHOD_PATTERN)


class VisualObservation(VisionModel):
    """Screenshot-derived metadata; it never retains screenshot bytes."""

    untrusted_content: Literal[True] = True
    observation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), pattern=REGION_ID_PATTERN
    )
    image_size: ImageSize
    timestamp: datetime
    regions: list[VisualRegion] = Field(default_factory=list, max_length=MAX_VISUAL_REGIONS)
    source: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    summary: str | None = Field(default=None, max_length=MAX_VISUAL_SUMMARY_CHARS)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=32
    )

    @model_validator(mode="after")
    def check_timestamp_and_metadata(self) -> VisualObservation:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("visual observation timestamp must be timezone-aware")
        for region in self.regions:
            if not region.bounding_box.fits_within(self.image_size):
                raise ValueError("visual region lies outside the image dimensions")
        for key, value in self.metadata.items():
            if len(key) > 64 or (isinstance(value, str) and len(value) > 256):
                raise ValueError("visual metadata value exceeds its configured bound")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("visual metadata numbers must be finite")
        return self


class VisualMatchStatus(StrEnum):
    IDENTICAL = "identical"
    CHANGED = "changed"
    UNCERTAIN = "uncertain"


class VisualMatch(VisionModel):
    """Explicit pixel-comparison result; it does not imply semantic meaning."""

    status: VisualMatchStatus
    similarity: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    changed_ratio: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    changed_region: BoundingBox | None = None
    reason: str = Field(pattern=SAFE_TOKEN_PATTERN)
    method: str = Field(pattern=SAFE_METHOD_PATTERN)
    before_sha256: str = Field(pattern=SHA256_PATTERN)
    after_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_pixel_metrics(self) -> VisualMatch:
        if self.status is VisualMatchStatus.UNCERTAIN:
            return self
        if self.similarity is None or self.changed_ratio is None:
            raise ValueError("definitive pixel comparison requires both metrics")
        if self.status is VisualMatchStatus.IDENTICAL and (
            self.similarity != 1.0 or self.changed_ratio != 0.0
        ):
            raise ValueError("identical pixels must have exact identity metrics")
        if self.status is VisualMatchStatus.CHANGED and self.changed_ratio <= 0.0:
            raise ValueError("changed pixels require a non-zero changed ratio")
        return self


class VisualVerificationKind(StrEnum):
    SCREENSHOT_CHANGED = "screenshot_changed"
    REGION_CHANGED = "region_changed"
    REGION_UNCHANGED = "region_unchanged"
    SIMILARITY_AT_LEAST = "similarity_at_least"


class VisualVerificationCondition(VisionModel):
    """Structured visual expectation evaluated against two fresh screenshots."""

    kind: VisualVerificationKind
    region: BoundingBox | None = None
    minimum_similarity: float = Field(
        default=0.98, ge=0.0, le=1.0, strict=True, allow_inf_nan=False
    )
    minimum_change_ratio: float = Field(
        default=0.001, gt=0.0, le=1.0, strict=True, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_expectation(self) -> VisualVerificationCondition:
        needs_region = self.kind in {
            VisualVerificationKind.REGION_CHANGED,
            VisualVerificationKind.REGION_UNCHANGED,
        }
        if needs_region and self.region is None:
            raise ValueError(f"{self.kind.value} requires a bounding region")
        if not needs_region and self.region is not None:
            raise ValueError("region is supported only for region-based verification")
        return self


class VisualVerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class VisualVerificationResult(VisionModel):
    """Verdict plus explicit method and non-sensitive evidence reference."""

    condition: VisualVerificationCondition
    status: VisualVerificationStatus
    method: str = Field(pattern=SAFE_METHOD_PATTERN)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str = Field(pattern=SAFE_TOKEN_PATTERN)
    evidence_ref: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    match: VisualMatch | None = None

    @model_validator(mode="after")
    def require_evidence_for_definitive_verdict(self) -> VisualVerificationResult:
        if self.status is VisualVerificationStatus.UNCERTAIN:
            return self
        if (
            self.match is None
            or self.match.status is VisualMatchStatus.UNCERTAIN
            or self.match.similarity is None
            or self.match.changed_ratio is None
            or self.evidence_ref != f"sha256:{self.match.after_sha256}"
        ):
            raise ValueError("definitive visual verdict requires matching comparison evidence")
        return self
