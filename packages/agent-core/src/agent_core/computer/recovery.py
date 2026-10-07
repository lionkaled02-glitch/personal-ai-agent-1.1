"""Small bounded recovery policy for computer actions."""

from __future__ import annotations

from dataclasses import dataclass

from ..permissions import PermissionLevel
from .models import (
    ComputerActionStatus,
    RecoveryAction,
    RecoveryRecommendation,
    VerificationCondition,
    VerificationKind,
)


@dataclass(frozen=True)
class RecoveryPolicy:
    """Recommend bounded recovery; never authorizes a new action by itself."""

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
        action: str,
        risk_level: PermissionLevel,
        status: ComputerActionStatus,
        attempts: int,
        retry_safe: bool,
        condition: VerificationCondition | None,
    ) -> RecoveryRecommendation:
        """Return a deterministic recovery hint for an observed action result."""
        retries_used = max(0, attempts - 1)
        remaining = max(0, self.max_retries - retries_used)

        if status is ComputerActionStatus.VERIFIED:
            return RecoveryRecommendation(
                action=RecoveryAction.STOP,
                reason="verified_action_no_recovery_needed",
                retries_used=retries_used,
                retries_remaining=0,
            )
        if risk_level is PermissionLevel.HIGH:
            return RecoveryRecommendation(
                action=RecoveryAction.STOP,
                reason="high_risk_actions_are_never_retried_automatically",
                retries_used=retries_used,
                retries_remaining=remaining,
            )
        if status is ComputerActionStatus.TIMED_OUT:
            return RecoveryRecommendation(
                action=RecoveryAction.STOP,
                reason="timeout_may_leave_action_outcome_uncertain",
                retries_used=retries_used,
                retries_remaining=0,
            )
        if status is ComputerActionStatus.DENIED:
            return RecoveryRecommendation(
                action=RecoveryAction.STOP,
                reason="permission_denied_no_retry",
                retries_used=retries_used,
                retries_remaining=0,
            )
        if (
            retry_safe
            and status in {ComputerActionStatus.FAILED, ComputerActionStatus.VERIFICATION_FAILED}
            and remaining > 0
        ):
            return RecoveryRecommendation(
                action=RecoveryAction.RETRY,
                reason="bounded_idempotent_retry_available",
                retries_used=retries_used,
                retries_remaining=remaining,
            )
        if status in {
            ComputerActionStatus.UNVERIFIED,
            ComputerActionStatus.VERIFICATION_FAILED,
            ComputerActionStatus.FAILED,
        }:
            if condition is not None and condition.kind in {
                VerificationKind.UI_ELEMENT_FOCUSED,
                VerificationKind.UI_ELEMENT_SELECTED,
                VerificationKind.UI_ELEMENT_PRESENCE,
            }:
                action_hint = RecoveryAction.REQUERY_UI_ELEMENT
            elif action == "focus_window":
                action_hint = RecoveryAction.REQUERY_ACTIVE_WINDOW
            else:
                action_hint = RecoveryAction.REFRESH_OBSERVATION
            return RecoveryRecommendation(
                action=action_hint,
                reason="refresh_observation_before_any_non_idempotent_retry",
                retries_used=retries_used,
                retries_remaining=remaining,
            )
        return RecoveryRecommendation(
            action=RecoveryAction.STOP,
            reason="no_safe_recovery_available",
            retries_used=retries_used,
            retries_remaining=0,
        )
