from __future__ import annotations


class MediaError(Exception):
    """Base media error."""


class MediaValidationError(MediaError):
    """Input failed media validation."""


class MediaProviderError(MediaError):
    """Provider or renderer failed."""
