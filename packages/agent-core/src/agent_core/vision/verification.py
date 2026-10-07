"""Evaluate declared pixel-level expectations without semantic inference."""

from __future__ import annotations

from .models import (
    VisualMatch,
    VisualMatchStatus,
    VisualVerificationCondition,
    VisualVerificationKind,
    VisualVerificationResult,
    VisualVerificationStatus,
)


def verify_visual_match(
    match: VisualMatch,
    condition: VisualVerificationCondition,
) -> VisualVerificationResult:
    """Evaluate one schema-validated pixel predicate.

    ``VERIFIED`` here means only that the requested pixel-level predicate was
    met. It is not evidence that a particular user-interface action occurred.
    """
    if match.status is VisualMatchStatus.UNCERTAIN:
        return VisualVerificationResult(
            status=VisualVerificationStatus.UNCERTAIN,
            condition=condition,
            method=match.method,
            confidence=None,
            reason="visual_comparison_uncertain",
            evidence_ref=f"sha256:{match.after_sha256}",
            match=match,
        )

    if condition.kind in {
        VisualVerificationKind.SCREENSHOT_CHANGED,
        VisualVerificationKind.REGION_CHANGED,
    }:
        changed_ratio = match.changed_ratio
        passed = changed_ratio is not None and changed_ratio >= condition.minimum_change_ratio
        reason = "pixel_change_observed" if passed else "expected_pixel_change_not_observed"
    elif condition.kind is VisualVerificationKind.REGION_UNCHANGED:
        similarity = match.similarity
        passed = similarity is not None and similarity >= condition.minimum_similarity
        reason = "region_pixels_unchanged" if passed else "region_pixels_changed"
    else:
        similarity = match.similarity
        passed = similarity is not None and similarity >= condition.minimum_similarity
        reason = "pixel_similarity_threshold_met" if passed else "pixel_similarity_below_threshold"

    if match.similarity is None or match.changed_ratio is None:
        status = VisualVerificationStatus.UNCERTAIN
        reason = "visual_comparison_metrics_missing"
    else:
        status = VisualVerificationStatus.VERIFIED if passed else VisualVerificationStatus.FAILED

    return VisualVerificationResult(
        status=status,
        condition=condition,
        method=match.method,
        confidence=None,
        reason=reason,
        evidence_ref=f"sha256:{match.after_sha256}",
        match=match,
    )
