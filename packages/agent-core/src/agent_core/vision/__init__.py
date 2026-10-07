"""Provider-neutral visual observation and verification models.

Image processing backends are optional and imported lazily. Screenshot
acquisition remains owned by ``agent_core.computer``.
"""

from .comparison import DeterministicVisionProvider
from .errors import (
    ComparisonLimitError,
    ImageLimitError,
    InvalidImageError,
    InvalidRegionError,
    OptionalImageSupportError,
    VisionComparisonError,
    VisionError,
    VisionProviderError,
    VisionTimeoutError,
)
from .interfaces import VisionProvider, VisualVerifier
from .limits import VisionLimits
from .models import (
    BoundingBox,
    ImageFrame,
    ImageSize,
    RegionSource,
    VisualAnalysis,
    VisualMatch,
    VisualMatchStatus,
    VisualObservation,
    VisualRegion,
    VisualVerificationCondition,
    VisualVerificationKind,
    VisualVerificationResult,
    VisualVerificationStatus,
)

__all__ = [
    "BoundingBox",
    "ComparisonLimitError",
    "DeterministicVisionProvider",
    "ImageFrame",
    "ImageLimitError",
    "ImageSize",
    "InvalidImageError",
    "InvalidRegionError",
    "OptionalImageSupportError",
    "RegionSource",
    "VisionComparisonError",
    "VisionError",
    "VisionLimits",
    "VisionProvider",
    "VisionProviderError",
    "VisionTimeoutError",
    "VisualAnalysis",
    "VisualMatch",
    "VisualMatchStatus",
    "VisualObservation",
    "VisualRegion",
    "VisualVerificationCondition",
    "VisualVerificationKind",
    "VisualVerificationResult",
    "VisualVerificationStatus",
    "VisualVerifier",
]
