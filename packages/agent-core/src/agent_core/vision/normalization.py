"""Validate and normalize Phase 6 screenshot data for in-process vision use."""

from __future__ import annotations

import hashlib
import struct
import zlib

from ..computer.models import ScreenshotObservation
from .errors import ImageLimitError, InvalidImageError
from .limits import VisionLimits
from .models import ImageFrame, ImageSize

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR = b"IHDR"
_PNG_IHDR_LENGTH = 13
_PNG_HEADER_LENGTH = 33


def normalize_screenshot_image(
    screenshot: ScreenshotObservation,
    limits: VisionLimits,
) -> ImageFrame:
    """Build an ephemeral bounded frame from an existing Phase 6 screenshot.

    No acquisition occurs here. The caller must have obtained the screenshot
    through ``ComputerRuntime`` and must not persist the resulting frame.
    """
    payload = screenshot.payload
    if payload is None:
        raise InvalidImageError(code="screenshot_payload_missing")
    if len(payload) > limits.max_image_bytes:
        raise ImageLimitError()
    if screenshot.media_type != "image/png" or len(payload) < _PNG_HEADER_LENGTH:
        raise InvalidImageError()
    if (
        payload[:8] != _PNG_SIGNATURE
        or payload[8:12] != struct.pack(">I", _PNG_IHDR_LENGTH)
        or payload[12:16] != _PNG_IHDR
        or (zlib.crc32(payload[12:29]) & 0xFFFFFFFF) != struct.unpack_from(">I", payload, 29)[0]
    ):
        raise InvalidImageError()

    width, height = struct.unpack_from(">II", payload, 16)
    if width < 1 or height < 1:
        raise InvalidImageError()
    if (
        width > limits.max_image_width
        or height > limits.max_image_height
        or width > screenshot.width
        or height > screenshot.height
        or width * height > limits.max_image_pixels
    ):
        raise ImageLimitError()
    # Phase 6 metadata is sourced from the same provider call. A mismatch is
    # rejected rather than silently changing the coordinate space.
    if (width, height) != (screenshot.width, screenshot.height):
        raise InvalidImageError(code="screenshot_dimension_mismatch")

    digest = hashlib.sha256(payload).hexdigest()
    if screenshot.payload_sha256 is not None and screenshot.payload_sha256 != digest:
        raise InvalidImageError(code="screenshot_digest_mismatch")
    return ImageFrame(
        size=ImageSize(width=width, height=height),
        media_type="image/png",
        payload_bytes=len(payload),
        sha256=digest,
        payload=payload,
    )
