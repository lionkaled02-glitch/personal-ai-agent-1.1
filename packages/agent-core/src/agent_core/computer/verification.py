"""Provider-neutral deterministic verification over before/after observations."""

from __future__ import annotations

from .models import (
    ComputerObservation,
    UIElement,
    VerificationCondition,
    VerificationKind,
    VerificationResult,
    VerificationStatus,
    WindowInfo,
)


def not_run(reason: str = "no_verification_condition") -> VerificationResult:
    """An explicit non-verification result; provider completion is not success."""
    return VerificationResult(status=VerificationStatus.NOT_RUN, reason=reason)


def _window(observation: ComputerObservation, identifier: str | None) -> WindowInfo | None:
    if observation.active_window is not None and (
        identifier is None or observation.active_window.identifier == identifier
    ):
        return observation.active_window
    if observation.windows is None:
        return None
    for window in observation.windows:
        if identifier is not None and window.identifier == identifier:
            return window
    return None


def _element(observation: ComputerObservation, automation_id: str) -> UIElement | None:
    if observation.ui_elements is None:
        return None
    for element in observation.ui_elements:
        if element.automation_id == automation_id:
            return element
    return None


def verify_condition(
    condition: VerificationCondition,
    before: ComputerObservation,
    after: ComputerObservation,
) -> VerificationResult:
    """Evaluate one explicit postcondition without inferring intent.

    A condition passes only when its expected state is supported by the
    supplied observations. Missing evidence is a failure, never a success.
    """
    kind = condition.kind
    expected: dict[str, str | int | float | bool | None] = {}
    observed: dict[str, str | int | float | bool | None] = {}
    passed = False
    reason = "verification_failed"

    if kind is VerificationKind.ACTIVE_WINDOW_CHANGED:
        old = before.active_window.identifier if before.active_window is not None else None
        new = after.active_window.identifier if after.active_window is not None else None
        expected = {"changed": True}
        observed = {"before_identifier": old, "after_identifier": new}
        passed = old is not None and new is not None and old != new
        reason = "active_window_changed" if passed else "active_window_did_not_change"

    elif kind is VerificationKind.ACTIVE_WINDOW_IS:
        actual = after.active_window.identifier if after.active_window is not None else None
        expected = {"window_identifier": condition.window_identifier}
        observed = {"window_identifier": actual}
        passed = actual == condition.window_identifier
        reason = "active_window_matches" if passed else "active_window_mismatch"

    elif kind is VerificationKind.CURSOR_AT:
        expected_point = condition.point
        actual_point = after.cursor_position
        expected = {
            "x": expected_point.x if expected_point is not None else None,
            "y": expected_point.y if expected_point is not None else None,
            "tolerance_px": condition.tolerance_px,
        }
        observed = {
            "x": actual_point.x if actual_point is not None else None,
            "y": actual_point.y if actual_point is not None else None,
        }
        passed = (
            expected_point is not None
            and actual_point is not None
            and abs(expected_point.x - actual_point.x) <= condition.tolerance_px
            and abs(expected_point.y - actual_point.y) <= condition.tolerance_px
        )
        reason = "cursor_within_tolerance" if passed else "cursor_outside_tolerance_or_missing"

    elif kind in {
        VerificationKind.UI_ELEMENT_FOCUSED,
        VerificationKind.UI_ELEMENT_SELECTED,
    }:
        automation_id = condition.automation_id or ""
        old_element = _element(before, automation_id)
        new_element = _element(after, automation_id)
        field_name = "focused" if kind is VerificationKind.UI_ELEMENT_FOCUSED else "selected"
        old_value = getattr(old_element, field_name) if old_element is not None else None
        new_value = getattr(new_element, field_name) if new_element is not None else None
        expected = {"automation_id": automation_id, field_name: True}
        observed = {"before": old_value, "after": new_value}
        passed = new_value is True and old_value is not True
        reason = (
            f"ui_element_became_{field_name}" if passed else f"ui_element_not_verified_{field_name}"
        )

    elif kind is VerificationKind.UI_ELEMENT_PRESENCE:
        automation_id = condition.automation_id or ""
        old_element = _element(before, automation_id)
        new_element = _element(after, automation_id)
        expected = {
            "automation_id": automation_id,
            "present": condition.expected_present,
        }
        observed = {
            "before_present": old_element is not None,
            "after_present": new_element is not None,
        }
        passed = (
            condition.expected_present is True and old_element is None and new_element is not None
        ) or (
            condition.expected_present is False and old_element is not None and new_element is None
        )
        reason = (
            "ui_element_presence_changed_as_expected"
            if passed
            else "ui_element_presence_not_verified"
        )

    elif kind is VerificationKind.SCREENSHOT_CHANGED:
        old_digest = before.screenshot.payload_sha256 if before.screenshot is not None else None
        new_digest = after.screenshot.payload_sha256 if after.screenshot is not None else None
        expected = {"changed": True}
        observed = {
            "before_sha256": old_digest,
            "after_sha256": new_digest,
        }
        passed = old_digest is not None and new_digest is not None and old_digest != new_digest
        reason = "screenshot_content_changed" if passed else "screenshot_change_not_observed"

    elif kind is VerificationKind.WINDOW_TITLE_CHANGED:
        old_window = _window(before, condition.window_identifier)
        new_window = _window(after, condition.window_identifier)
        old_title = old_window.title if old_window is not None else None
        new_title = new_window.title if new_window is not None else None
        expected = {
            "changed": True,
            "expected_title": condition.expected_title,
        }
        observed = {"before_title": old_title, "after_title": new_title}
        passed = (
            old_title is not None
            and new_title is not None
            and old_title != new_title
            and (condition.expected_title is None or new_title == condition.expected_title)
        )
        reason = (
            "window_title_changed_as_expected" if passed else "window_title_change_not_verified"
        )

    elif kind is VerificationKind.WINDOW_STATE_CHANGED:
        identifier = condition.window_identifier
        old_window = _window(before, identifier)
        new_window = _window(after, identifier)
        old_state = _state(old_window)
        new_state = _state(new_window)
        expected = {
            "minimized": condition.expected_minimized,
            "maximized": condition.expected_maximized,
        }
        observed = {
            "before_minimized": old_state.get("minimized") if old_state is not None else None,
            "before_maximized": old_state.get("maximized") if old_state is not None else None,
            "after_minimized": new_state.get("minimized") if new_state is not None else None,
            "after_maximized": new_state.get("maximized") if new_state is not None else None,
        }
        changed = old_state is not None and new_state is not None and old_state != new_state
        expected_matches = new_state is not None and all(
            wanted is None or new_state[key] == wanted
            for key, wanted in (
                ("minimized", condition.expected_minimized),
                ("maximized", condition.expected_maximized),
            )
        )
        passed = changed and expected_matches
        reason = (
            "window_state_changed_as_expected" if passed else "window_state_change_not_verified"
        )

    return VerificationResult(
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        condition=kind,
        reason=reason,
        expected=expected,
        observed=observed,
    )


def _state(window: WindowInfo | None) -> dict[str, bool | None] | None:
    if window is None:
        return None
    return {"minimized": window.minimized, "maximized": window.maximized}
