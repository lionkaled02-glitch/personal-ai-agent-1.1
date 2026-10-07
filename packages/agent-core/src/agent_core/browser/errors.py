"""Stable, content-free errors for the bounded browser foundation."""

from __future__ import annotations

_ERROR_MESSAGES: dict[str, str] = {
    "browser_error": "The browser operation could not be completed.",
    "invalid_url": "The browser URL is malformed or unsupported.",
    "unsupported_scheme": "Only HTTP and HTTPS browser URLs are supported.",
    "host_not_allowed": "The browser URL is outside the configured host policy.",
    "url_limit_exceeded": "The browser URL exceeds the configured length limit.",
    "invalid_input": "The browser request failed validation.",
    "limit_exceeded": "The browser operation exceeds a configured limit.",
    "timeout_limit_exceeded": "The requested browser timeout exceeds its configured limit.",
    "browser_not_started": "The browser provider has not been started.",
    "provider_unavailable": "The optional browser provider is unavailable.",
    "provider_failed": "The browser provider could not complete the operation.",
    "provider_timeout": "The browser provider exceeded its configured time limit.",
    "permission_denied": "The browser operation was denied by permission policy.",
    "approval_denied": "The browser operation did not receive the required approval.",
    "session_not_found": "The requested browser session does not exist.",
    "page_not_found": "The requested browser page does not exist.",
    "page_closed": "The requested browser page is closed.",
    "page_limit_exceeded": "The browser session has reached its page limit.",
    "session_limit_exceeded": "The browser runtime has reached its session limit.",
    "observation_required": "Observe the current page before acting on an element.",
    "element_not_found": "The requested page element was not found in the latest observation.",
    "stale_element": "The element reference is stale; observe the page again.",
    "element_not_actionable": "The requested page element is not visible or enabled.",
    "sensitive_field_blocked": (
        "Sensitive browser fields cannot be filled or selected in this phase."
    ),
    "verification_uncertain": "The browser action outcome could not be verified.",
    "verification_failed": "The browser action did not meet its verification condition.",
    "screenshot_unavailable": "A validated browser screenshot is unavailable.",
    "invalid_snapshot": "The browser provider returned invalid page metadata.",
    "wait_state_unavailable": "The requested page state could not be observed.",
}


class BrowserError(Exception):
    """Browser error with a stable code and a safe public message."""

    def __init__(self, code: str = "browser_error") -> None:
        self.code = code if code in _ERROR_MESSAGES else "browser_error"
        self.public_message = _ERROR_MESSAGES[self.code]
        super().__init__(self.public_message)


class BrowserValidationError(BrowserError):
    """Malformed URL, provider data, or action input rejected at the boundary."""


class BrowserLimitError(BrowserError):
    """Input or provider output exceeded a hard or configured limit."""


class BrowserProviderUnavailableError(BrowserError):
    """Optional browser automation dependency is not installed or ready."""

    def __init__(self) -> None:
        super().__init__("provider_unavailable")


class BrowserProviderError(BrowserError):
    """Provider failure; transient retry is opt-in and limited to safe navigation."""

    def __init__(self, *, retryable: bool = False) -> None:
        self.retryable = retryable
        super().__init__("provider_failed")


class BrowserTimeoutError(BrowserError):
    """A bounded browser operation exceeded its timeout."""

    def __init__(self) -> None:
        super().__init__("provider_timeout")


class BrowserPermissionError(BrowserError):
    """Permission policy or confirmation did not authorize this operation."""

    def __init__(self, code: str = "permission_denied") -> None:
        super().__init__(code)
