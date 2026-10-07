"""Structured errors for the provider-neutral computer layer."""

from __future__ import annotations


class ComputerError(Exception):
    """Base class for computer-agent errors with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UnsupportedPlatformError(ComputerError):
    """The selected computer provider is unsupported on this platform."""

    def __init__(self, message: str = "computer control is supported only on Windows") -> None:
        super().__init__("unsupported_platform", message)


class ProviderUnavailableError(ComputerError):
    """A platform provider's optional implementation dependency is absent."""

    def __init__(self, message: str) -> None:
        super().__init__("provider_unavailable", message)


class ComputerValidationError(ComputerError):
    """A computer action or observation failed safe validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)


class ComputerProviderError(ComputerError):
    """A provider operation failed; exception detail is intentionally hidden."""

    def __init__(self, operation: str) -> None:
        super().__init__("provider_error", f"computer provider failed during {operation}")
