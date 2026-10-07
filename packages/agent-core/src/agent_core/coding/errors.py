"""Stable content-free errors for the bounded coding foundation."""

from __future__ import annotations

from ..errors import AgentCoreError

_ERROR_MESSAGES: dict[str, str] = {
    "coding_error": "The coding operation could not be completed.",
    "coding_validation_failed": "Coding input or provider output failed validation.",
    "coding_limit_exceeded": "The coding operation exceeds a configured limit.",
    "coding_workspace_violation": "The coding path is outside the configured project workspace.",
    "coding_permission_denied": "The coding operation was denied by permission policy.",
    "coding_provider_failed": "The coding provider could not complete the operation.",
    "coding_provider_timeout": "The coding provider exceeded its configured time limit.",
    "coding_operation_unsupported": "The coding provider does not support this operation.",
}


class CodingError(AgentCoreError):
    """Coding failure with a stable code and a content-free public message."""

    def __init__(self, code: str = "coding_error") -> None:
        self.code = code if code in _ERROR_MESSAGES else "coding_error"
        self.public_message = _ERROR_MESSAGES[self.code]
        super().__init__(self.public_message)


class CodingValidationError(CodingError):
    """Malformed coding input or provider output rejected at the boundary."""

    def __init__(self) -> None:
        super().__init__("coding_validation_failed")


class CodingLimitError(CodingError):
    """Input or output exceeded a hard or configured coding limit."""

    def __init__(self) -> None:
        super().__init__("coding_limit_exceeded")


class CodingWorkspaceError(CodingError):
    """A project/file path failed the existing workspace containment checks."""

    def __init__(self) -> None:
        super().__init__("coding_workspace_violation")


class CodingPermissionError(CodingError):
    """A later permission-gated coding operation was not authorized."""

    def __init__(self) -> None:
        super().__init__("coding_permission_denied")


class CodingProviderError(CodingError):
    """Provider failure; retry is opt-in and only safe when explicitly marked."""

    def __init__(self, *, retryable: bool = False) -> None:
        self.retryable = retryable
        super().__init__("coding_provider_failed")


class CodingTimeoutError(CodingError):
    """A provider exceeded the cooperative configured operation deadline."""

    def __init__(self) -> None:
        super().__init__("coding_provider_timeout")


class CodingUnsupportedOperationError(CodingError):
    """The requested bounded provider capability is unavailable."""

    def __init__(self) -> None:
        super().__init__("coding_operation_unsupported")
