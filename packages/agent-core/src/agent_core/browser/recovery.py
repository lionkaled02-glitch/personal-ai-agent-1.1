"""Bounded browser recovery recommendations; no permission is granted here."""

from __future__ import annotations

from dataclasses import dataclass

from ..permissions import PermissionLevel
from .models import (
    BrowserActionType,
    BrowserRecoveryAction,
    BrowserRecoveryResult,
    BrowserVerificationStatus,
)


@dataclass(frozen=True)
class BrowserRecoveryPolicy:
    """Recommend safe recovery while keeping all retries bounded and explicit."""

    max_retries: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or not 0 <= self.max_retries <= 3
        ):
            raise ValueError("max_retries must be an integer between 0 and 3")

    def recommend(
        self,
        *,
        action: BrowserActionType,
        risk_level: PermissionLevel,
        status: BrowserVerificationStatus,
        attempts: int,
        provider_retryable: bool = False,
    ) -> BrowserRecoveryResult:
        """Return one next-step hint, never retry high-risk or non-navigation work."""
        retries_used = max(0, attempts - 1)
        remaining = max(0, self.max_retries - retries_used)
        if status is BrowserVerificationStatus.VERIFIED:
            next_action = BrowserRecoveryAction.STOP
            reason = "verified_no_recovery_needed"
            remaining = 0
        elif risk_level is PermissionLevel.HIGH:
            next_action = BrowserRecoveryAction.STOP
            reason = "high_risk_actions_are_never_retried"
            remaining = 0
        elif status is BrowserVerificationStatus.UNCERTAIN:
            next_action = BrowserRecoveryAction.REOBSERVE
            reason = "uncertain_outcome_requires_fresh_observation"
            remaining = 0
        elif action is BrowserActionType.NAVIGATE and provider_retryable and remaining > 0:
            next_action = BrowserRecoveryAction.RETRY
            reason = "bounded_idempotent_navigation_retry"
        else:
            next_action = BrowserRecoveryAction.STOP
            reason = "no_safe_retry_available"
            remaining = 0
        return BrowserRecoveryResult(
            action=next_action,
            reason_code=reason,
            attempts=attempts,
            retries_remaining=remaining,
        )
