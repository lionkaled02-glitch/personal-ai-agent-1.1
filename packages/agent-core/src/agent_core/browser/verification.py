"""Deterministic, fail-closed verification for browser actions."""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote, urlsplit

from .errors import BrowserError
from .models import (
    BrowserActionType,
    BrowserObservation,
    BrowserVerificationResult,
    BrowserVerificationStatus,
)
from .url_safety import hostname_from_url, safe_observed_url, validate_http_url


def _verification_time(value: datetime | None) -> datetime:
    return value if value is not None else datetime.now(UTC)


def verify_navigation(
    expected_url: str,
    after: BrowserObservation | None,
    *,
    actual_url: str | None = None,
    verified_at: datetime | None = None,
) -> BrowserVerificationResult:
    """Verify the exact URL privately while exposing only redacted projections."""
    expected_safe = safe_observed_url(expected_url)
    observed_safe = after.url if after is not None else None
    if after is None:
        status = BrowserVerificationStatus.UNCERTAIN
        reason = "navigation_observation_missing"
    elif observed_safe is None:
        status = BrowserVerificationStatus.UNCERTAIN
        reason = "navigation_url_unavailable"
    elif actual_url is None:
        status = BrowserVerificationStatus.UNCERTAIN
        reason = "navigation_raw_url_unavailable"
    else:
        try:
            actual_safe = safe_observed_url(actual_url)
            matches = (
                actual_safe is not None
                and observed_safe is not None
                and expected_safe is not None
                and _navigation_url_identity(observed_safe)
                == _navigation_url_identity(expected_safe)
                and _navigation_url_identity(actual_safe) == _navigation_url_identity(expected_safe)
                and _navigation_url_identity(expected_url) == _navigation_url_identity(actual_url)
            )
        except BrowserError:
            matches = False
        status = BrowserVerificationStatus.VERIFIED if matches else BrowserVerificationStatus.FAILED
        reason = "navigation_url_matches" if matches else "navigation_url_mismatch"
    return BrowserVerificationResult(
        action=BrowserActionType.NAVIGATE,
        status=status,
        reason_code=reason,
        expected={"url": expected_safe},
        observed={"url": observed_safe},
        verified_at=_verification_time(verified_at),
    )


def _navigation_url_identity(url: str) -> tuple[str, str, int | None, str, str, str]:
    """Normalize URL structure for exact checking without exposing its values."""
    validated = validate_http_url(url)
    parts = urlsplit(validated)
    hostname = hostname_from_url(validated)
    port = parts.port
    if port == {"http": 80, "https": 443}[parts.scheme.lower()]:
        port = None
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="/?@!$&'()*+,;=:%-._~")
    fragment = quote(parts.fragment, safe="/?@!$&'()*+,;=:%-._~")
    return parts.scheme.lower(), hostname, port, path, query, fragment


def verify_observed_state_change(
    action: BrowserActionType,
    before: BrowserObservation | None,
    after: BrowserObservation | None,
    *,
    verified_at: datetime | None = None,
) -> BrowserVerificationResult:
    """Verify that an interaction changed a bounded, non-secret observation."""
    if before is None or after is None:
        status = BrowserVerificationStatus.UNCERTAIN
        reason = "state_observation_missing"
        changed = None
    else:
        changed = (
            before.url != after.url
            or before.title != after.title
            or before.visible_text != after.visible_text
            or before.ready_state != after.ready_state
            or tuple(item.element_id for item in before.elements)
            != tuple(item.element_id for item in after.elements)
        )
        status = (
            BrowserVerificationStatus.VERIFIED if changed else BrowserVerificationStatus.UNCERTAIN
        )
        reason = "page_state_changed" if changed else "page_state_unchanged"
    return BrowserVerificationResult(
        action=action,
        status=status,
        reason_code=reason,
        expected={"state_changed": True},
        observed={"state_changed": changed},
        verified_at=_verification_time(verified_at),
    )


def verify_value_action(
    action: BrowserActionType,
    *,
    matches: bool | None,
    expected_chars: int,
    verified_at: datetime | None = None,
) -> BrowserVerificationResult:
    """Verify fill/select without ever retaining or returning the supplied value."""
    if matches is None:
        status = BrowserVerificationStatus.UNCERTAIN
        reason = "provider_verification_unavailable"
    elif matches:
        status = BrowserVerificationStatus.VERIFIED
        reason = "value_action_matches"
    else:
        status = BrowserVerificationStatus.FAILED
        reason = "value_action_mismatch"
    return BrowserVerificationResult(
        action=action,
        status=status,
        reason_code=reason,
        expected={"value_chars": expected_chars, "matches": True},
        observed={"matches": matches},
        verified_at=_verification_time(verified_at),
    )


def uncertain_verification(
    action: BrowserActionType,
    *,
    reason_code: str,
    verified_at: datetime | None = None,
) -> BrowserVerificationResult:
    """Construct an explicit uncertain result when no safe proof is available."""
    return BrowserVerificationResult(
        action=action,
        status=BrowserVerificationStatus.UNCERTAIN,
        reason_code=reason_code,
        verified_at=_verification_time(verified_at),
    )
