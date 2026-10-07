"""Stable, content-free errors from the visual observation boundary."""


class VisionError(Exception):
    """Base error carrying a stable code and safe public message."""

    code = "vision_error"
    public_message = "Visual observation could not be completed."

    def __init__(self, *, code: str | None = None, message: str | None = None) -> None:
        self.code = code or type(self).code
        self.public_message = message or type(self).public_message
        super().__init__(self.public_message)


class InvalidImageError(VisionError):
    code = "invalid_image"
    public_message = "Screenshot data is not a supported bounded PNG image."


class ImageLimitError(VisionError):
    code = "image_limit_exceeded"
    public_message = "Screenshot exceeds the configured visual-observation limits."


class InvalidRegionError(VisionError):
    code = "invalid_region"
    public_message = "The requested visual region is outside the screenshot bounds."


class ComparisonLimitError(VisionError):
    code = "comparison_limit_exceeded"
    public_message = "Visual comparison exceeds the configured pixel limit."


class VisionProviderError(VisionError):
    code = "vision_provider_unavailable"
    public_message = "The configured visual provider is unavailable."


class OptionalImageSupportError(VisionError):
    code = "optional_image_support_unavailable"
    public_message = "Optional local image comparison support is not installed."


class VisionTimeoutError(VisionError):
    code = "vision_operation_timeout"
    public_message = "Visual observation exceeded its configured time limit."


class VisionComparisonError(VisionError):
    code = "vision_comparison_failed"
    public_message = "Visual comparison could not be completed."
