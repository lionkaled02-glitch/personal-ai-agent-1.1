"""Explicit JSON projections that exclude all ephemeral image bytes."""

from __future__ import annotations

from typing import Any

from .models import VisualMatch, VisualObservation, VisualVerificationResult


def observation_to_dict(observation: VisualObservation) -> dict[str, Any]:
    """Serialize validated observation metadata only."""
    return observation.model_dump(mode="json", exclude_none=True)


def match_to_dict(match: VisualMatch) -> dict[str, Any]:
    """Serialize comparison metrics and bounded pixel-region metadata."""
    return match.model_dump(mode="json", exclude_none=True)


def verification_result_to_dict(result: VisualVerificationResult) -> dict[str, Any]:
    """Serialize a structured visual verdict without screenshot payloads."""
    return result.model_dump(mode="json", exclude_none=True)
