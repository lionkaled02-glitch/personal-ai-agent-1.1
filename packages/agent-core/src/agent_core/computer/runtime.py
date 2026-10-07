"""Permission-enforced computer runtime with observation and verification."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from ..events import Clock, EventBus, EventType, utc_now
from ..permissions import (
    ApprovalRequest,
    PermissionDecision,
    PermissionLevel,
    PermissionManager,
    _active_permission_authorization,
    _permission_authorization_scope,
)
from ..vision.errors import VisionError
from ..vision.interfaces import VisualVerifier
from ..vision.models import (
    VisualVerificationCondition,
    VisualVerificationResult,
    VisualVerificationStatus,
)
from ..vision.verification import verify_visual_match
from .errors import ComputerError, ComputerProviderError, ComputerValidationError
from .interfaces import ComputerProvider
from .limits import ComputerLimits
from .models import (
    ComputerActionResult,
    ComputerActionStatus,
    ComputerObservation,
    ComputerOperationResult,
    FocusWindowAction,
    KeyboardAction,
    KeyboardActionKind,
    MouseAction,
    MouseActionKind,
    Point,
    RecoveryAction,
    RecoveryRecommendation,
    ScreenInfo,
    ScreenshotObservation,
    SelectUIElementAction,
    UIElement,
    VerificationCondition,
    VerificationKind,
    VerificationResult,
    WindowInfo,
)
from .observation import (
    make_observation,
    normalize_point,
    normalize_screen_info,
    normalize_screenshot,
    normalize_ui_elements,
    normalize_window,
    normalize_windows,
)
from .recovery import RecoveryPolicy
from .verification import not_run, verify_condition

_SAFE_VISUAL_REASONS = frozenset(
    {
        "visual_verification_not_run",
        "visual_verifier_unavailable",
        "visual_preflight_failed",
        "visual_region_out_of_bounds",
        "visual_screenshot_unavailable",
        "visual_refresh_failed",
        "visual_refresh_screenshot_unavailable",
        "visual_verification_failed",
        "pre_action_screenshot_unavailable",
        "visual_condition_mismatch",
        "invalid_visual_result",
        "visual_evidence_uncertain",
        "visual_result_inconsistent",
        "visual_provider_failed",
        "invalid_image",
        "image_limit_exceeded",
        "invalid_region",
        "comparison_limit_exceeded",
        "optional_image_support_unavailable",
        "vision_operation_timeout",
        "vision_provider_unavailable",
        "vision_comparison_failed",
    }
)
_VISION_ERROR_MESSAGES = {
    "invalid_image": (
        "invalid_image",
        "Screenshot data is not a supported bounded PNG image.",
    ),
    "screenshot_payload_missing": (
        "invalid_image",
        "Screenshot data is not available for visual verification.",
    ),
    "screenshot_dimension_mismatch": (
        "invalid_image",
        "Screenshot dimensions do not match its PNG header.",
    ),
    "screenshot_digest_mismatch": (
        "invalid_image",
        "Screenshot integrity validation failed.",
    ),
    "image_limit_exceeded": (
        "image_limit_exceeded",
        "Screenshot exceeds the configured visual-observation limits.",
    ),
    "invalid_region": (
        "invalid_region",
        "The requested visual region is outside the screenshot bounds.",
    ),
    "comparison_limit_exceeded": (
        "comparison_limit_exceeded",
        "Visual comparison exceeds the configured pixel limit.",
    ),
    "optional_image_support_unavailable": (
        "optional_image_support_unavailable",
        "Optional local image comparison support is not installed.",
    ),
    "vision_operation_timeout": (
        "vision_operation_timeout",
        "Visual observation exceeded its configured time limit.",
    ),
    "vision_comparison_failed": (
        "vision_comparison_failed",
        "Visual comparison could not be completed.",
    ),
    "vision_provider_unavailable": (
        "vision_provider_unavailable",
        "The configured visual provider is unavailable.",
    ),
}

# Operation classification fails closed: an unknown computer operation is HIGH.
_LOW_OPERATIONS = frozenset(
    {
        "computer_screen_info",
        "computer_cursor_position",
        "computer_active_window",
        "computer_list_windows",
        "computer_inspect_ui",
        "computer_screenshot",
    }
)
_MEDIUM_OPERATIONS = frozenset(
    {
        "computer_move_mouse",
        "computer_click",
        "computer_double_click",
        "computer_focus_window",
        "computer_select_ui_element",
        "computer_press_key",
        "computer_hotkey",
        "computer_type_text",
    }
)
_HIGH_OPERATIONS = frozenset(
    {
        "computer_delete",
        "computer_publish",
        "computer_purchase",
        "computer_change_security",
        "computer_privileged_action",
    }
)


class _PermissionDescriptor:
    """Structural descriptor consumed by the existing PermissionManager."""

    def __init__(self, name: str, permission_level: PermissionLevel) -> None:
        self.name = name
        self.permission_level = permission_level


@dataclass(frozen=True)
class _Authorization:
    allowed: bool
    task_id: str
    step_id: str
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ObservationRequest:
    screen: bool = False
    cursor: bool = False
    active_window: bool = False
    windows: bool = False
    ui_elements: bool = False
    screenshot: bool = False
    window_identifier: str | None = None


ActionMetadata = dict[str, int | float | str | bool]


def classify_computer_operation(operation: str) -> PermissionLevel:
    """Map an explicit computer operation to the shared permission levels.

    Current pointer movement and observation are LOW; clicks, focus, element
    selection, and keyboard input are MEDIUM. Known destructive/external
    categories are HIGH. Unknown names fail closed to HIGH.
    """
    if operation in _LOW_OPERATIONS:
        return PermissionLevel.LOW
    if operation in _MEDIUM_OPERATIONS:
        return PermissionLevel.MEDIUM
    if operation in _HIGH_OPERATIONS:
        return PermissionLevel.HIGH
    return PermissionLevel.HIGH


class ComputerRuntime:
    """Orchestrates bounded observation → permission → action → verification.

    ``PermissionManager`` is checked at this boundary even when a caller uses
    the runtime or registry directly. Calls made by the Agent's Executor reuse
    only the exact, already-approved tool scope. Provider calls are
    synchronous: the runtime enforces a cooperative elapsed-time deadline and
    stops retries on timeout, but cannot forcibly interrupt a misbehaving OS
    API call.
    """

    _MAX_TRACKED_TASKS = 512

    def __init__(
        self,
        provider: ComputerProvider,
        permissions: PermissionManager,
        events: EventBus,
        limits: ComputerLimits | None = None,
        clock: Clock | None = None,
        monotonic: Callable[[], float] | None = None,
        visual_verifier: VisualVerifier | None = None,
    ) -> None:
        self._provider = provider
        self._permissions = permissions
        self._events = events
        self._limits = limits or ComputerLimits()
        self._clock: Clock = clock or utc_now
        self._monotonic = monotonic or time.monotonic
        self._visual_verifier = visual_verifier
        self._recovery = RecoveryPolicy(max_retries=self._limits.max_retries)
        self._action_counts: OrderedDict[str, int] = OrderedDict()

    @property
    def limits(self) -> ComputerLimits:
        return self._limits

    def screen_info(self) -> ComputerOperationResult:
        return self._read_observation(
            "computer_screen_info",
            "screen_info",
            _ObservationRequest(screen=True),
        )

    def cursor_position(self) -> ComputerOperationResult:
        return self._read_observation(
            "computer_cursor_position",
            "cursor_position",
            _ObservationRequest(cursor=True),
        )

    def active_window(self) -> ComputerOperationResult:
        return self._read_observation(
            "computer_active_window",
            "active_window",
            _ObservationRequest(active_window=True),
        )

    def list_windows(self) -> ComputerOperationResult:
        return self._read_observation(
            "computer_list_windows",
            "list_windows",
            _ObservationRequest(windows=True),
        )

    def inspect_ui(self, window_identifier: str | None = None) -> ComputerOperationResult:
        return self._read_observation(
            "computer_inspect_ui",
            "inspect_ui",
            _ObservationRequest(
                active_window=window_identifier is None,
                windows=window_identifier is not None,
                ui_elements=True,
                window_identifier=window_identifier,
            ),
        )

    def screenshot(self) -> ComputerOperationResult:
        return self._read_observation(
            "computer_screenshot",
            "screenshot",
            _ObservationRequest(screenshot=True),
        )

    def invalid_observation(self, operation: str, error_code: str) -> ComputerOperationResult:
        """Return a sanitized validation failure without contacting a provider."""
        self._emit_observed(operation, False, error_code)
        return ComputerOperationResult(
            ok=False,
            operation=operation,
            error_code=error_code,
            error="computer observation input failed validation",
        )

    def move_mouse(
        self,
        action: MouseAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        if action.kind is not MouseActionKind.MOVE:
            return self._invalid_action(
                "computer_move_mouse", action.action_id, "invalid_action_kind"
            )
        condition = verification or VerificationCondition(
            kind=VerificationKind.CURSOR_AT,
            point=action.point,
            tolerance_px=self._limits.cursor_tolerance_px,
        )
        duration = (
            action.duration_s
            if action.duration_s is not None
            else self._limits.mouse_move_duration_s
        )
        return self._execute_action(
            tool_name="computer_move_mouse",
            action_id=action.action_id,
            action="move_mouse",
            permission_level=classify_computer_operation("computer_move_mouse"),
            operation=partial(self._provider.move_mouse, action.point, duration_s=duration),
            verification=condition,
            visual_verification=visual_verification,
            retry_safe=True,
            metadata={
                "x": action.point.x,
                "y": action.point.y,
                "duration_s": duration,
            },
        )

    def click(
        self,
        action: MouseAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        if action.kind is not MouseActionKind.CLICK:
            return self._invalid_action("computer_click", action.action_id, "invalid_action_kind")
        return self._execute_action(
            tool_name="computer_click",
            action_id=action.action_id,
            action="click",
            permission_level=classify_computer_operation("computer_click"),
            operation=partial(self._provider.click, action.point, button=action.button),
            verification=verification,
            visual_verification=visual_verification,
            retry_safe=False,
            metadata={"x": action.point.x, "y": action.point.y, "button": action.button.value},
        )

    def double_click(
        self,
        action: MouseAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        if action.kind is not MouseActionKind.DOUBLE_CLICK:
            return self._invalid_action(
                "computer_double_click", action.action_id, "invalid_action_kind"
            )
        return self._execute_action(
            tool_name="computer_double_click",
            action_id=action.action_id,
            action="double_click",
            permission_level=classify_computer_operation("computer_double_click"),
            operation=partial(self._provider.double_click, action.point, button=action.button),
            verification=verification,
            visual_verification=visual_verification,
            retry_safe=False,
            metadata={"x": action.point.x, "y": action.point.y, "button": action.button.value},
        )

    def focus_window(
        self,
        action: FocusWindowAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        condition = verification or VerificationCondition(
            kind=VerificationKind.ACTIVE_WINDOW_IS,
            window_identifier=action.window_identifier,
        )
        return self._execute_action(
            tool_name="computer_focus_window",
            action_id=action.action_id,
            action="focus_window",
            permission_level=classify_computer_operation("computer_focus_window"),
            operation=partial(self._provider.focus_window, action.window_identifier),
            verification=condition,
            visual_verification=visual_verification,
            retry_safe=True,
            target_window=action.window_identifier,
        )

    def select_ui_element(
        self,
        action: SelectUIElementAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        condition = verification or VerificationCondition(
            kind=VerificationKind.UI_ELEMENT_SELECTED,
            window_identifier=action.window_identifier,
            automation_id=action.automation_id,
        )
        return self._execute_action(
            tool_name="computer_select_ui_element",
            action_id=action.action_id,
            action="select_ui_element",
            permission_level=classify_computer_operation("computer_select_ui_element"),
            operation=partial(
                self._provider.select_ui_element,
                action.window_identifier,
                action.automation_id,
            ),
            verification=condition,
            visual_verification=visual_verification,
            retry_safe=False,
            target_window=action.window_identifier,
            metadata={
                "window_identifier": action.window_identifier,
                "automation_id": action.automation_id,
            },
        )

    def keyboard_action(
        self,
        action: KeyboardAction,
        verification: VerificationCondition | None = None,
        visual_verification: VisualVerificationCondition | None = None,
    ) -> ComputerActionResult:
        if action.kind is KeyboardActionKind.PRESS_KEY:
            if action.key is None:
                return self._invalid_action(
                    "computer_press_key", action.action_id, "invalid_keyboard_action"
                )
            key = action.key
            operation = partial(self._provider.press_key, key)
            tool_name = "computer_press_key"
            action_name = "press_key"
        elif action.kind is KeyboardActionKind.HOTKEY:
            if action.key is None:
                return self._invalid_action(
                    "computer_hotkey", action.action_id, "invalid_keyboard_action"
                )
            key = action.key
            modifiers = tuple(action.modifiers)
            operation = partial(self._provider.hotkey, modifiers, key)
            tool_name = "computer_hotkey"
            action_name = "hotkey"
        else:
            text = action.text or ""
            if len(text) > self._limits.max_text_input_chars:
                return self._invalid_action(
                    "computer_type_text", action.action_id, "text_input_limit_exceeded"
                )
            operation = partial(self._provider.type_text, text)
            tool_name = "computer_type_text"
            action_name = "type_text"
        return self._execute_action(
            tool_name=tool_name,
            action_id=action.action_id,
            action=action_name,
            permission_level=classify_computer_operation(tool_name),
            operation=operation,
            verification=verification,
            visual_verification=visual_verification,
            retry_safe=False,
            metadata={"character_count": len(action.text or "")}
            if action.kind is KeyboardActionKind.TYPE_TEXT
            else {},
        )

    def _read_observation(
        self,
        tool_name: str,
        operation: str,
        request: _ObservationRequest,
    ) -> ComputerOperationResult:
        action_id = str(uuid.uuid4())
        authorization = self._authorize(
            tool_name,
            classify_computer_operation(tool_name),
            action_id=action_id,
            reason=f"Read computer observation: {operation}.",
        )
        if not authorization.allowed:
            self._emit_observed(operation, False, authorization.error_code)
            return ComputerOperationResult(
                ok=False,
                operation=operation,
                error_code=authorization.error_code,
                error=authorization.error,
            )
        try:
            with _permission_authorization_scope(
                task_id=authorization.task_id,
                step_id=authorization.step_id,
                tool_name=tool_name,
                permission_level=classify_computer_operation(tool_name),
            ):
                observation = self._capture_observation(
                    request,
                    metadata_only_screenshot=False,
                    permission_action_id=action_id,
                )
        except ComputerError as exc:
            self._emit_observed(operation, False, exc.code)
            return ComputerOperationResult(
                ok=False,
                operation=operation,
                error_code=exc.code,
                error=exc.message,
            )
        except Exception as exc:
            mapped = self._provider_error(operation, exc)
            self._emit_observed(operation, False, mapped.code)
            return ComputerOperationResult(
                ok=False,
                operation=operation,
                error_code=mapped.code,
                error=mapped.message,
            )
        self._emit_observed(operation, True, None)
        return ComputerOperationResult(ok=True, operation=operation, observation=observation)

    def _capture_observation(
        self,
        request: _ObservationRequest,
        *,
        metadata_only_screenshot: bool,
        permission_action_id: str,
    ) -> ComputerObservation:
        screen_info: ScreenInfo | None = None
        cursor_position: Point | None = None
        active_window: WindowInfo | None = None
        windows: list[WindowInfo] | None = None
        ui_elements: list[UIElement] | None = None
        screenshot: ScreenshotObservation | None = None
        windows_truncated = False
        ui_elements_truncated = False

        if request.screen:
            self._require_observation_permission("computer_screen_info", permission_action_id)
            screen_info = normalize_screen_info(self._provider.get_screen_info())
        if request.cursor:
            self._require_observation_permission("computer_cursor_position", permission_action_id)
            cursor_position = normalize_point(self._provider.get_cursor_position())
        if request.active_window:
            self._require_observation_permission("computer_active_window", permission_action_id)
            active = self._provider.get_active_window()
            active_window = normalize_window(active) if active is not None else None
        if request.windows:
            self._require_observation_permission("computer_list_windows", permission_action_id)
            raw_windows = self._provider.list_windows(limit=self._limits.max_windows)
            windows, windows_truncated = normalize_windows(
                raw_windows, limit=self._limits.max_windows
            )
        if request.ui_elements:
            if request.window_identifier is not None:
                if windows is None:
                    self._require_observation_permission(
                        "computer_list_windows", permission_action_id
                    )
                    raw_windows = self._provider.list_windows(limit=self._limits.max_windows)
                    windows, windows_truncated = normalize_windows(
                        raw_windows, limit=self._limits.max_windows
                    )
                target = self._visible_window(windows, request.window_identifier)
                if target is None:
                    raise ComputerValidationError(
                        "window_not_found", "target window is not currently visible"
                    )
            self._require_observation_permission("computer_inspect_ui", permission_action_id)
            raw_elements = self._provider.inspect_ui(
                request.window_identifier, limit=self._limits.max_ui_elements
            )
            ui_elements, ui_elements_truncated = normalize_ui_elements(
                raw_elements, limit=self._limits.max_ui_elements
            )
        if request.screenshot:
            self._require_observation_permission("computer_screenshot", permission_action_id)
            raw_screenshot = self._provider.screenshot(max_bytes=self._limits.max_screenshot_bytes)
            screenshot = normalize_screenshot(
                raw_screenshot, max_bytes=self._limits.max_screenshot_bytes
            )
            if metadata_only_screenshot:
                screenshot = screenshot.metadata_only()

        return make_observation(
            observed_at=self._clock(),
            screen_info=screen_info,
            cursor_position=cursor_position,
            active_window=active_window,
            windows=windows,
            ui_elements=ui_elements,
            screenshot=screenshot,
            windows_truncated=windows_truncated,
            ui_elements_truncated=ui_elements_truncated,
        )

    def _execute_action(
        self,
        *,
        tool_name: str,
        action_id: str,
        action: str,
        permission_level: PermissionLevel,
        operation: Callable[[], None],
        verification: VerificationCondition | None,
        visual_verification: VisualVerificationCondition | None,
        retry_safe: bool,
        target_window: str | None = None,
        metadata: ActionMetadata | None = None,
    ) -> ComputerActionResult:
        task_id, step_id = self._execution_ids(action_id)
        prior_count = self._action_counts.get(task_id, 0)
        if prior_count >= self._limits.max_actions_per_task:
            return self._invalid_action(
                tool_name,
                action_id,
                "action_limit_exceeded",
                task_id=task_id,
                step_id=step_id,
                risk_level=permission_level,
            )
        self._action_counts[task_id] = prior_count + 1
        self._action_counts.move_to_end(task_id)
        if len(self._action_counts) > self._MAX_TRACKED_TASKS:
            self._action_counts.popitem(last=False)

        visual_result = self._visual_not_run(visual_verification)
        visual_verifier = self._visual_verifier
        if visual_verification is not None and visual_verifier is None:
            return self._finish_action(
                tool_name=tool_name,
                action_id=action_id,
                action=action,
                permission_level=permission_level,
                status=ComputerActionStatus.INVALID,
                attempted=False,
                provider_completed=False,
                attempts=0,
                before=None,
                after=None,
                verification_result=not_run("visual_verifier_unavailable"),
                visual_verification=self._visual_uncertain(
                    visual_verification, "visual_verifier_unavailable"
                ),
                error_code="visual_verifier_unavailable",
                error="visual verification is not configured",
                retry_safe=False,
                verification=verification,
                task_id=task_id,
                step_id=step_id,
                metadata=metadata,
            )

        request = _ObservationRequest(
            screen=True,
            cursor=True,
            active_window=True,
            windows=target_window is not None
            or (
                verification is not None
                and verification.kind
                in {
                    VerificationKind.WINDOW_STATE_CHANGED,
                    VerificationKind.WINDOW_TITLE_CHANGED,
                }
                and verification.window_identifier is not None
            ),
            ui_elements=verification is not None
            and verification.kind
            in {
                VerificationKind.UI_ELEMENT_FOCUSED,
                VerificationKind.UI_ELEMENT_SELECTED,
                VerificationKind.UI_ELEMENT_PRESENCE,
            },
            screenshot=(
                visual_verification is not None
                or (
                    verification is not None
                    and verification.kind is VerificationKind.SCREENSHOT_CHANGED
                )
            ),
            window_identifier=(
                target_window
                or (verification.window_identifier if verification is not None else None)
            ),
        )
        try:
            observed_before_raw = self._capture_observation(
                request,
                metadata_only_screenshot=visual_verification is None,
                permission_action_id=action_id,
            )
            visual_before_screenshot = observed_before_raw.screenshot
            observed_before = self._metadata_only_observation(observed_before_raw)
        except ComputerError as exc:
            pre_status = (
                ComputerActionStatus.DENIED
                if exc.code in {"permission_denied", "approval_denied"}
                else ComputerActionStatus.FAILED
            )
            return self._finish_action(
                tool_name=tool_name,
                action_id=action_id,
                action=action,
                permission_level=permission_level,
                status=pre_status,
                attempted=False,
                provider_completed=False,
                attempts=0,
                before=None,
                after=None,
                verification_result=not_run("pre_action_observation_failed"),
                visual_verification=visual_result,
                error_code=exc.code,
                error=exc.message,
                retry_safe=False,
                verification=verification,
                task_id=task_id,
                step_id=step_id,
                metadata=metadata,
            )
        except Exception as exc:
            mapped = self._provider_error("pre_action_observation", exc)
            return self._finish_action(
                tool_name=tool_name,
                action_id=action_id,
                action=action,
                permission_level=permission_level,
                status=ComputerActionStatus.FAILED,
                attempted=False,
                provider_completed=False,
                attempts=0,
                before=None,
                after=None,
                verification_result=not_run("pre_action_observation_failed"),
                visual_verification=visual_result,
                error_code=mapped.code,
                error=mapped.message,
                retry_safe=False,
                verification=verification,
                task_id=task_id,
                step_id=step_id,
                metadata=metadata,
            )

        initial_observation = observed_before
        recovery_observations: list[ComputerObservation] = []
        self._emit_action_event(
            EventType.COMPUTER_ACTION_REQUESTED,
            task_id,
            step_id,
            action_id,
            action,
            permission_level,
            metadata=metadata,
        )
        if visual_verification is not None:
            assert visual_verifier is not None
            if visual_before_screenshot is None:
                visual_result = self._visual_uncertain(
                    visual_verification, "pre_action_screenshot_unavailable"
                )
                return self._finish_action(
                    tool_name=tool_name,
                    action_id=action_id,
                    action=action,
                    permission_level=permission_level,
                    status=ComputerActionStatus.INVALID,
                    attempted=False,
                    provider_completed=False,
                    attempts=0,
                    before=observed_before,
                    after=None,
                    verification_result=not_run("pre_action_screenshot_unavailable"),
                    visual_verification=visual_result,
                    error_code="pre_action_screenshot_unavailable",
                    error="visual verification requires a bounded pre-action screenshot",
                    retry_safe=False,
                    verification=verification,
                    task_id=task_id,
                    step_id=step_id,
                    metadata=metadata,
                )
            try:
                before_frame = visual_verifier.validate_screenshot(visual_before_screenshot)
                if (
                    visual_verification.region is not None
                    and not visual_verification.region.fits_within(before_frame.size)
                ):
                    visual_result = self._visual_uncertain(
                        visual_verification, "visual_region_out_of_bounds"
                    )
                    return self._finish_action(
                        tool_name=tool_name,
                        action_id=action_id,
                        action=action,
                        permission_level=permission_level,
                        status=ComputerActionStatus.INVALID,
                        attempted=False,
                        provider_completed=False,
                        attempts=0,
                        before=observed_before,
                        after=None,
                        verification_result=not_run("visual_region_out_of_bounds"),
                        visual_verification=visual_result,
                        error_code="visual_region_out_of_bounds",
                        error="visual verification region exceeds screenshot bounds",
                        retry_safe=False,
                        verification=verification,
                        task_id=task_id,
                        step_id=step_id,
                        metadata=metadata,
                    )
            except VisionError as exc:
                error_code, error_message = self._safe_vision_error(exc)
                visual_result = self._visual_uncertain(visual_verification, error_code)
                return self._finish_action(
                    tool_name=tool_name,
                    action_id=action_id,
                    action=action,
                    permission_level=permission_level,
                    status=ComputerActionStatus.INVALID,
                    attempted=False,
                    provider_completed=False,
                    attempts=0,
                    before=observed_before,
                    after=None,
                    verification_result=not_run("visual_preflight_failed"),
                    visual_verification=visual_result,
                    error_code=error_code,
                    error=error_message,
                    retry_safe=False,
                    verification=verification,
                    task_id=task_id,
                    step_id=step_id,
                    metadata=metadata,
                )
            except Exception:
                visual_result = self._visual_uncertain(
                    visual_verification, "visual_preflight_failed"
                )
                return self._finish_action(
                    tool_name=tool_name,
                    action_id=action_id,
                    action=action,
                    permission_level=permission_level,
                    status=ComputerActionStatus.INVALID,
                    attempted=False,
                    provider_completed=False,
                    attempts=0,
                    before=observed_before,
                    after=None,
                    verification_result=not_run("visual_preflight_failed"),
                    visual_verification=visual_result,
                    error_code="visual_preflight_failed",
                    error="visual verification preflight failed",
                    retry_safe=False,
                    verification=verification,
                    task_id=task_id,
                    step_id=step_id,
                    metadata=metadata,
                )

        validation_error = self._validate_action_bounds(
            action=action,
            observation=observed_before,
            target_window=target_window,
            verification=verification,
            metadata=metadata,
        )
        if validation_error is not None:
            code, message = validation_error
            return self._finish_action(
                tool_name=tool_name,
                action_id=action_id,
                action=action,
                permission_level=permission_level,
                status=ComputerActionStatus.INVALID,
                attempted=False,
                provider_completed=False,
                attempts=0,
                before=observed_before,
                after=None,
                verification_result=not_run("action_validation_failed"),
                visual_verification=visual_result,
                error_code=code,
                error=message,
                retry_safe=False,
                verification=verification,
                task_id=task_id,
                step_id=step_id,
                metadata=metadata,
            )

        authorization = self._authorize(
            tool_name,
            permission_level,
            action_id=action_id,
            reason=f"Computer action: {action}. Explicit approval may be required.",
        )
        if not authorization.allowed:
            return self._finish_action(
                tool_name=tool_name,
                action_id=action_id,
                action=action,
                permission_level=permission_level,
                status=ComputerActionStatus.DENIED,
                attempted=False,
                provider_completed=False,
                attempts=0,
                before=observed_before,
                after=None,
                verification_result=not_run("permission_denied"),
                visual_verification=visual_result,
                error_code=authorization.error_code or "permission_denied",
                error=authorization.error or "computer action denied by permission policy",
                retry_safe=False,
                verification=verification,
                task_id=authorization.task_id,
                step_id=authorization.step_id,
                metadata=metadata,
            )

        self._emit_action_event(
            EventType.COMPUTER_ACTION_STARTED,
            authorization.task_id,
            authorization.step_id,
            action_id,
            action,
            permission_level,
            metadata=metadata,
        )
        start = self._monotonic()
        deadline = start + self._limits.action_timeout_s
        attempts = 0
        final_after: ComputerObservation | None = None
        final_verification = not_run()
        final_status = ComputerActionStatus.FAILED
        final_error_code: str | None = None
        final_error: str | None = None
        provider_completed = False

        while True:
            if self._monotonic() >= deadline:
                final_status = ComputerActionStatus.TIMED_OUT
                final_error_code = "action_timeout"
                final_error = "computer action exceeded its configured timeout"
                break

            attempts += 1
            attempt_error: ComputerError | None = None
            provider_completed = False
            try:
                operation()
                provider_completed = True
            except ComputerError as exc:
                attempt_error = exc
            except TimeoutError:
                attempt_error = ComputerError(
                    "action_timeout", "computer provider operation timed out"
                )
            except Exception as exc:
                attempt_error = self._provider_error(action, exc)

            timed_out = self._monotonic() > deadline or (
                attempt_error is not None and attempt_error.code == "action_timeout"
            )
            try:
                final_after_raw = self._capture_observation(
                    request,
                    metadata_only_screenshot=visual_verification is None,
                    permission_action_id=action_id,
                )
                final_after = self._metadata_only_observation(final_after_raw)
            except ComputerError as exc:
                final_error_code = exc.code
                final_error = exc.message
                final_status = ComputerActionStatus.FAILED
                final_verification = not_run("post_action_observation_failed")
                break
            except Exception as exc:
                mapped = self._provider_error("post_action_observation", exc)
                final_error_code = mapped.code
                final_error = mapped.message
                final_status = ComputerActionStatus.FAILED
                final_verification = not_run("post_action_observation_failed")
                break

            if verification is None:
                final_verification = not_run()
            else:
                final_verification = verify_condition(verification, observed_before, final_after)

            if timed_out:
                final_status = ComputerActionStatus.TIMED_OUT
                final_error_code = "action_timeout"
                final_error = (
                    "computer action exceeded its configured timeout; outcome may be uncertain"
                )
                break

            if visual_verification is not None:
                assert visual_verifier is not None
                if visual_before_screenshot is None or final_after_raw.screenshot is None:
                    visual_result = self._visual_uncertain(
                        visual_verification, "visual_screenshot_unavailable"
                    )
                else:
                    try:
                        visual_result = self._validate_visual_result(
                            visual_verifier.verify_screenshots(
                                visual_before_screenshot,
                                final_after_raw.screenshot,
                                visual_verification,
                            ),
                            visual_verification,
                        )
                    except VisionError as exc:
                        error_code, _ = self._safe_vision_error(exc)
                        visual_result = self._visual_uncertain(visual_verification, error_code)
                    except Exception:
                        visual_result = self._visual_uncertain(
                            visual_verification, "visual_verification_failed"
                        )

                refresh_count = min(max(0, visual_verifier.max_observation_retries), 2)
                for _ in range(
                    refresh_count
                    if visual_result.status is VisualVerificationStatus.UNCERTAIN
                    else 0
                ):
                    if self._monotonic() >= deadline:
                        break
                    try:
                        refreshed = self._capture_observation(
                            _ObservationRequest(screenshot=True),
                            metadata_only_screenshot=False,
                            permission_action_id=action_id,
                        )
                    except Exception:
                        visual_result = self._visual_uncertain(
                            visual_verification, "visual_refresh_failed"
                        )
                        break
                    recovery_observations.append(self._metadata_only_observation(refreshed))
                    final_after = self._metadata_only_observation(refreshed)
                    if verification is not None:
                        final_verification = verify_condition(
                            verification, observed_before, final_after
                        )
                    if refreshed.screenshot is None or visual_before_screenshot is None:
                        visual_result = self._visual_uncertain(
                            visual_verification, "visual_refresh_screenshot_unavailable"
                        )
                        break
                    try:
                        visual_result = self._validate_visual_result(
                            visual_verifier.verify_screenshots(
                                visual_before_screenshot,
                                refreshed.screenshot,
                                visual_verification,
                            ),
                            visual_verification,
                        )
                    except VisionError as exc:
                        error_code, _ = self._safe_vision_error(exc)
                        visual_result = self._visual_uncertain(visual_verification, error_code)
                    except Exception:
                        visual_result = self._visual_uncertain(
                            visual_verification, "visual_verification_failed"
                        )
                    if visual_result.status is not VisualVerificationStatus.UNCERTAIN:
                        break

                if self._monotonic() > deadline:
                    final_status = ComputerActionStatus.TIMED_OUT
                    final_error_code = "action_timeout"
                    final_error = (
                        "computer action exceeded its configured timeout; outcome may be uncertain"
                    )
                    break

                if visual_result.status is VisualVerificationStatus.FAILED:
                    final_status = ComputerActionStatus.VERIFICATION_FAILED
                    final_error_code = "visual_verification_failed"
                    final_error = visual_result.reason
                elif visual_result.status is VisualVerificationStatus.UNCERTAIN:
                    final_status = (
                        ComputerActionStatus.VERIFICATION_FAILED
                        if verification is not None
                        else ComputerActionStatus.UNVERIFIED
                    )
                    if attempt_error is not None:
                        final_error_code = attempt_error.code
                        final_error = attempt_error.message
                    else:
                        final_error_code = "visual_verification_uncertain"
                        final_error = "visual evidence remained uncertain after bounded refreshes"
                elif (
                    verification is None or verification.kind is VerificationKind.SCREENSHOT_CHANGED
                ):
                    # Pixel differences prove only that pixels changed; they
                    # cannot prove that the requested click/action occurred.
                    final_status = ComputerActionStatus.UNVERIFIED
                    final_error_code = "semantic_verification_required"
                    final_error = (
                        "pixel-level visual evidence is not semantic action proof; "
                        "supply a non-screenshot deterministic postcondition"
                    )
                elif final_verification.ok:
                    final_status = ComputerActionStatus.VERIFIED
                    if attempt_error is not None:
                        final_error_code = attempt_error.code
                        final_error = attempt_error.message
                else:
                    final_status = ComputerActionStatus.VERIFICATION_FAILED
                    final_error_code = "verification_failed"
                    final_error = final_verification.reason
                break

            if (
                final_verification.ok
                and verification is not None
                and verification.kind is VerificationKind.SCREENSHOT_CHANGED
            ):
                # A changed screenshot is a pixel fact, not evidence that the
                # requested click or other semantic action occurred.
                final_status = ComputerActionStatus.UNVERIFIED
                final_error_code = "semantic_verification_required"
                final_error = (
                    "screenshot change is not semantic action proof; "
                    "supply a non-screenshot deterministic postcondition"
                )
                break

            if final_verification.ok:
                final_status = ComputerActionStatus.VERIFIED
                if attempt_error is not None:
                    # The provider fault is retained as operational detail,
                    # but the observed postcondition is the source of truth.
                    final_error_code = attempt_error.code
                    final_error = attempt_error.message
                break

            if attempt_error is not None:
                final_status = ComputerActionStatus.FAILED
                final_error_code = attempt_error.code
                final_error = attempt_error.message
            elif verification is None:
                final_status = ComputerActionStatus.UNVERIFIED
                final_error_code = "verification_required"
                final_error = "provider returned but no deterministic postcondition was supplied"
            else:
                final_status = ComputerActionStatus.VERIFICATION_FAILED
                final_error_code = "verification_failed"
                final_error = final_verification.reason

            recommendation = self._recovery.recommend(
                action=action,
                risk_level=permission_level,
                status=final_status,
                attempts=attempts,
                retry_safe=retry_safe,
                condition=verification,
            )
            if recommendation.action is not RecoveryAction.RETRY:
                break
            # A retry gets a fresh pre-action observation. Only idempotent
            # LOW/MEDIUM actions can reach this branch; HIGH never retries.
            try:
                observed_before = self._capture_observation(
                    request,
                    metadata_only_screenshot=True,
                    permission_action_id=action_id,
                )
                recovery_observations.append(observed_before)
            except Exception as exc:
                mapped = self._provider_error("recovery_observation", exc)
                final_status = ComputerActionStatus.FAILED
                final_error_code = mapped.code
                final_error = mapped.message
                final_verification = not_run("recovery_observation_failed")
                break

        return self._finish_action(
            tool_name=tool_name,
            action_id=action_id,
            action=action,
            permission_level=permission_level,
            status=final_status,
            attempted=attempts > 0,
            provider_completed=provider_completed,
            attempts=attempts,
            before=initial_observation,
            after=final_after,
            verification_result=final_verification,
            visual_verification=visual_result,
            error_code=final_error_code,
            error=final_error,
            retry_safe=retry_safe,
            verification=verification,
            task_id=authorization.task_id,
            step_id=authorization.step_id,
            metadata=metadata,
            recovery_observations=recovery_observations,
        )

    @staticmethod
    def _metadata_only_observation(observation: ComputerObservation) -> ComputerObservation:
        """Strip ephemeral screenshot bytes before any result/history path."""
        if observation.screenshot is None or observation.screenshot.payload is None:
            return observation
        return observation.model_copy(update={"screenshot": observation.screenshot.metadata_only()})

    @staticmethod
    def _visual_not_run(
        condition: VisualVerificationCondition | None,
    ) -> VisualVerificationResult | None:
        if condition is None:
            return None
        return ComputerRuntime._visual_uncertain(condition, "visual_verification_not_run")

    @staticmethod
    def _safe_vision_error(error: VisionError) -> tuple[str, str]:
        """Return a known, content-free visual error code and message."""
        code = error.code if isinstance(error.code, str) else ""
        return _VISION_ERROR_MESSAGES.get(
            code,
            ("visual_provider_failed", "Visual verification could not be completed."),
        )

    @staticmethod
    def _validate_visual_result(
        value: object,
        condition: VisualVerificationCondition,
    ) -> VisualVerificationResult:
        """Revalidate a provider verdict and recompute its pixel predicate."""
        try:
            result = VisualVerificationResult.model_validate(value)
        except Exception:
            return ComputerRuntime._visual_uncertain(condition, "invalid_visual_result")
        if result.condition != condition:
            return ComputerRuntime._visual_uncertain(condition, "visual_condition_mismatch")
        if result.status is VisualVerificationStatus.UNCERTAIN:
            # Uncertainty is monotone: never upgrade it from provider-supplied
            # metrics or text, even if the result also contains a partial match.
            return ComputerRuntime._visual_uncertain(condition, "visual_evidence_uncertain")
        assert result.match is not None
        recomputed = verify_visual_match(result.match, condition)
        if recomputed.status is not result.status:
            return ComputerRuntime._visual_uncertain(condition, "visual_result_inconsistent")
        return recomputed

    @staticmethod
    def _visual_uncertain(
        condition: VisualVerificationCondition,
        reason: str,
    ) -> VisualVerificationResult:
        """Build a content-free UNCERTAIN result for a failed visual boundary."""
        safe_reason = (
            reason
            if isinstance(reason, str) and reason in _SAFE_VISUAL_REASONS
            else "visual_provider_failed"
        )
        return VisualVerificationResult(
            condition=condition,
            status=VisualVerificationStatus.UNCERTAIN,
            method="unavailable",
            confidence=None,
            reason=safe_reason,
        )

    def _validate_action_bounds(
        self,
        *,
        action: str,
        observation: ComputerObservation,
        target_window: str | None,
        verification: VerificationCondition | None,
        metadata: ActionMetadata | None,
    ) -> tuple[str, str] | None:
        if action == "move_mouse" or action in {"click", "double_click"}:
            # The tool adapters supply a screen-bounded Point; verify it here
            # against the observed screen before any provider input occurs.
            point = self._point_from_action_metadata(metadata)
            if point is None:
                return "invalid_action", "mouse target coordinates are missing"
            screen = observation.screen_info
            if screen is None or point.x >= screen.width or point.y >= screen.height:
                return "out_of_bounds", "mouse target is outside the observed screen bounds"
            if action == "move_mouse":
                duration = self._duration_from_metadata(metadata)
                if duration is None or duration > self._limits.mouse_move_duration_s:
                    return (
                        "duration_limit_exceeded",
                        "mouse movement duration exceeds configured limit",
                    )
        if action in {"press_key", "hotkey", "type_text"}:
            active_window = observation.active_window
            if active_window is None or not active_window.visible or not active_window.focused:
                return (
                    "active_window_unavailable",
                    "keyboard input requires an observed visible active window",
                )
        if action == "type_text":
            char_count = (metadata or {}).get("character_count", 0)
            if not isinstance(char_count, int) or char_count > self._limits.max_text_input_chars:
                return "text_input_limit_exceeded", "text input exceeds configured character limit"
        if target_window is not None:
            windows = observation.windows or []
            if self._visible_window(windows, target_window) is None:
                return "window_not_found", "target window is not currently visible"
        if action == "select_ui_element":
            automation_id = (metadata or {}).get("automation_id")
            if not isinstance(automation_id, str) or observation.ui_elements is None:
                return "ui_element_unavailable", "target UI element could not be observed"
            target_element = next(
                (
                    element
                    for element in observation.ui_elements
                    if element.automation_id == automation_id
                ),
                None,
            )
            if target_element is None:
                return "element_not_found", "target UI element is not in the observed UI"
            if target_element.enabled is not True:
                return "element_not_enabled", "target UI element is not confirmed enabled"
            if target_element.visible is not True:
                return "element_not_visible", "target UI element is not confirmed visible"
        if (
            verification is not None
            and verification.kind
            in {
                VerificationKind.UI_ELEMENT_FOCUSED,
                VerificationKind.UI_ELEMENT_SELECTED,
                VerificationKind.UI_ELEMENT_PRESENCE,
            }
            and observation.ui_elements is None
        ):
            return "verification_observation_missing", "UI verification data is unavailable"
        return None

    @staticmethod
    def _point_from_action_metadata(
        metadata: ActionMetadata | None,
    ) -> Point | None:
        if metadata is None:
            return None
        x = metadata.get("x")
        y = metadata.get("y")
        if (
            isinstance(x, int)
            and not isinstance(x, bool)
            and isinstance(y, int)
            and not isinstance(y, bool)
        ):
            return Point(x=x, y=y)
        return None

    @staticmethod
    def _duration_from_metadata(
        metadata: ActionMetadata | None,
    ) -> float | None:
        if metadata is None:
            return None
        duration = metadata.get("duration_s")
        return float(duration) if isinstance(duration, (int, float)) else None

    def _require_observation_permission(self, tool_name: str, action_id: str) -> None:
        """Enforce LOW permission on each provider observation boundary."""
        authorization = self._authorize(
            tool_name,
            PermissionLevel.LOW,
            action_id=action_id,
            reason=f"Read computer observation: {tool_name.removeprefix('computer_')}.",
        )
        if not authorization.allowed:
            raise ComputerValidationError(
                authorization.error_code or "permission_denied",
                authorization.error or "computer observation denied by permission policy",
            )

    def _authorize(
        self,
        tool_name: str,
        level: PermissionLevel,
        *,
        action_id: str,
        reason: str,
    ) -> _Authorization:
        scope = _active_permission_authorization()
        if scope is not None and scope.tool_name == tool_name and scope.permission_level >= level:
            return _Authorization(True, scope.task_id, scope.step_id)

        task_id, step_id = self._execution_ids(action_id)
        decision = self._permissions.check(_PermissionDescriptor(tool_name, level))
        if decision is PermissionDecision.DENIED:
            return _Authorization(
                False,
                task_id,
                step_id,
                "permission_denied",
                "computer operation denied by permission policy",
            )
        if decision is PermissionDecision.REQUIRES_APPROVAL:
            self._events.emit(
                EventType.APPROVAL_REQUIRED,
                task_id=task_id,
                step_id=step_id,
                data={
                    "tool_name": tool_name,
                    "permission_level": level.name,
                    "reason": reason,
                },
            )
            try:
                approved = self._permissions.request_approval(
                    ApprovalRequest(
                        task_id=task_id,
                        step_id=step_id,
                        tool_name=tool_name,
                        permission_level=level,
                        reason=reason,
                    )
                )
            except Exception:
                approved = False
            if not approved:
                return _Authorization(
                    False,
                    task_id,
                    step_id,
                    "approval_denied",
                    "computer operation approval was not granted",
                )
        return _Authorization(True, task_id, step_id)

    def _finish_action(
        self,
        *,
        tool_name: str,
        action_id: str,
        action: str,
        permission_level: PermissionLevel,
        status: ComputerActionStatus,
        attempted: bool,
        provider_completed: bool,
        attempts: int,
        before: ComputerObservation | None,
        after: ComputerObservation | None,
        verification_result: VerificationResult,
        error_code: str | None,
        error: str | None,
        retry_safe: bool,
        verification: VerificationCondition | None,
        task_id: str,
        step_id: str,
        metadata: ActionMetadata | None,
        recovery_observations: list[ComputerObservation] | None = None,
        visual_verification: VisualVerificationResult | None = None,
    ) -> ComputerActionResult:
        verified = (
            status is ComputerActionStatus.VERIFIED
            and verification_result.ok
            and (
                visual_verification is None
                or visual_verification.status is VisualVerificationStatus.VERIFIED
            )
        )
        recovery = self._recovery.recommend(
            action=action,
            risk_level=permission_level,
            status=status,
            attempts=attempts,
            retry_safe=retry_safe and visual_verification is None,
            condition=verification,
        )
        result = ComputerActionResult(
            action_id=action_id,
            action=action,
            risk_level=permission_level,
            status=status,
            attempted=attempted,
            success=verified,
            provider_completed=provider_completed,
            verified=verified,
            attempts=attempts,
            observed_before=before,
            observed_after=after,
            recovery_observations=recovery_observations or [],
            verification=verification_result,
            visual_verification=visual_verification,
            error_code=error_code,
            error=error,
            retryable=recovery.action is RecoveryAction.RETRY,
            recovery=recovery,
        )
        event_type = (
            EventType.COMPUTER_ACTION_DENIED
            if status is ComputerActionStatus.DENIED
            else EventType.COMPUTER_ACTION_COMPLETED
            if verified
            else EventType.COMPUTER_ACTION_FAILED
        )
        self._emit_action_event(
            event_type,
            task_id,
            step_id,
            action_id,
            action,
            permission_level,
            status=status,
            success=verified,
            attempts=attempts,
            error_code=error_code,
            metadata=metadata,
        )
        return result

    def _invalid_action(
        self,
        tool_name: str,
        action_id: str,
        error_code: str,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        risk_level: PermissionLevel | None = None,
    ) -> ComputerActionResult:
        current_task_id, current_step_id = self._execution_ids(action_id)
        task = task_id or current_task_id
        step = step_id or current_step_id
        level = risk_level or classify_computer_operation(tool_name)
        action = tool_name.removeprefix("computer_")
        result = ComputerActionResult(
            action_id=action_id,
            action=action,
            risk_level=level,
            status=ComputerActionStatus.INVALID,
            attempted=False,
            success=False,
            provider_completed=False,
            verified=False,
            attempts=0,
            verification=not_run("action_validation_failed"),
            error_code=error_code,
            error="computer action was rejected before provider execution",
            retryable=False,
            recovery=RecoveryRecommendation(
                action=RecoveryAction.STOP,
                reason="invalid_action_must_not_be_retried",
                retries_used=0,
                retries_remaining=0,
            ),
        )
        self._emit_action_event(
            EventType.COMPUTER_ACTION_FAILED,
            task,
            step,
            action_id,
            action,
            level,
            status=ComputerActionStatus.INVALID,
            success=False,
            attempts=0,
            error_code=error_code,
        )
        return result

    def _visible_window(
        self, windows: Sequence[WindowInfo] | None, identifier: str
    ) -> WindowInfo | None:
        if windows is None:
            return None
        for window in windows:
            if window.identifier == identifier and window.visible:
                return window
        return None

    def _execution_ids(self, action_id: str) -> tuple[str, str]:
        scope = _active_permission_authorization()
        if scope is not None:
            return scope.task_id, scope.step_id
        return "computer-direct", action_id

    def _provider_error(self, operation: str, exc: Exception) -> ComputerProviderError:
        # Never echo provider exception details; they may contain UI text,
        # filesystem paths, or other untrusted/sensitive implementation data.
        if isinstance(exc, ComputerProviderError):
            return exc
        return ComputerProviderError(operation)

    def _emit_observed(self, operation: str, ok: bool, error_code: str | None) -> None:
        scope = _active_permission_authorization()
        self._events.emit(
            EventType.COMPUTER_OBSERVED,
            task_id=scope.task_id if scope is not None else None,
            step_id=scope.step_id if scope is not None else None,
            data={"operation": operation, "ok": ok, "error_code": error_code},
        )

    def _emit_action_event(
        self,
        event_type: EventType,
        task_id: str,
        step_id: str,
        action_id: str,
        action: str,
        permission_level: PermissionLevel,
        *,
        metadata: ActionMetadata | None = None,
        status: ComputerActionStatus | None = None,
        success: bool | None = None,
        attempts: int | None = None,
        error_code: str | None = None,
    ) -> None:
        data: dict[str, object] = {
            "action_id": action_id,
            "action": action,
            "permission_level": permission_level.name,
        }
        if metadata:
            # Text input content is never in metadata; only its character
            # count is recorded by KeyboardAction.
            data.update(metadata)
        if status is not None:
            data["status"] = status.value
        if success is not None:
            data["success"] = success
        if attempts is not None:
            data["attempts"] = attempts
        if error_code is not None:
            data["error_code"] = error_code
        self._events.emit(
            event_type,
            task_id=task_id,
            step_id=step_id,
            data=data,
        )
