"""Provider-neutral computer interaction and observation models.

Window titles, UI names, application labels, and screenshot content originate
outside the agent and are untrusted data. They must not be interpreted as
instructions or used to change permissions.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..permissions import PermissionLevel
from ..vision.models import VisualVerificationResult, VisualVerificationStatus

HARD_MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024
HARD_MAX_TEXT_INPUT_CHARS = 1_024
ACTION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"


class ComputerModel(BaseModel):
    """Shared strict-shape configuration for public computer models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Point(ComputerModel):
    """A non-negative pixel coordinate in the primary display coordinate space."""

    x: int = Field(ge=0, le=32_767, strict=True)
    y: int = Field(ge=0, le=32_767, strict=True)


class Bounds(ComputerModel):
    """A screen-space rectangle; windows may straddle displays and be negative."""

    left: int = Field(ge=-100_000, le=100_000, strict=True)
    top: int = Field(ge=-100_000, le=100_000, strict=True)
    right: int = Field(ge=-100_000, le=100_000, strict=True)
    bottom: int = Field(ge=-100_000, le=100_000, strict=True)

    @model_validator(mode="after")
    def check_order(self) -> Bounds:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("bounds right/bottom must not precede left/top")
        return self


class ScreenInfo(ComputerModel):
    """Current primary-display dimensions; DPI is optional when unavailable."""

    width: int = Field(gt=0, le=32_768, strict=True)
    height: int = Field(gt=0, le=32_768, strict=True)
    display: str = Field(default="primary", min_length=1, max_length=64)
    dpi_x: float | None = Field(default=None, gt=0)
    dpi_y: float | None = Field(default=None, gt=0)
    scale_x: float | None = Field(default=None, gt=0)
    scale_y: float | None = Field(default=None, gt=0)


class WindowInfo(ComputerModel):
    """Bounded metadata for a top-level window.

    ``title`` is external UI content. ``application`` is an optional display
    label (for the Windows provider, the window class name rather than an
    executable path or process details).
    """

    identifier: str = Field(min_length=1, max_length=128)
    title: str = Field(default="", max_length=512)
    application: str | None = Field(default=None, max_length=128)
    bounds: Bounds
    visible: bool = Field(strict=True)
    focused: bool = Field(default=False, strict=True)
    minimized: bool | None = Field(default=None, strict=True)
    maximized: bool | None = Field(default=None, strict=True)


class UIElement(ComputerModel):
    """Safe accessibility metadata for one UI element; name remains untrusted."""

    element_id: str = Field(min_length=1, max_length=192)
    role: str = Field(min_length=1, max_length=64)
    name: str = Field(default="", max_length=256)
    bounds: Bounds | None = None
    enabled: bool | None = Field(default=None, strict=True)
    visible: bool | None = Field(default=None, strict=True)
    focused: bool | None = Field(default=None, strict=True)
    selected: bool | None = Field(default=None, strict=True)
    automation_id: str | None = Field(default=None, max_length=128)
    sensitive: bool = False

    @model_validator(mode="after")
    def redact_sensitive_name(self) -> UIElement:
        if self.sensitive and self.name != "[sensitive control]":
            object.__setattr__(self, "name", "[sensitive control]")
        return self


class MouseButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class MouseActionKind(StrEnum):
    MOVE = "move"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"


def _new_action_id() -> str:
    return str(uuid.uuid4())


class MouseAction(ComputerModel):
    """An explicit, bounded mouse primitive. Clicks are never inferred as safe."""

    action_id: str = Field(default_factory=_new_action_id, pattern=ACTION_ID_PATTERN)
    kind: MouseActionKind
    point: Point
    button: MouseButton = MouseButton.LEFT
    duration_s: float | None = Field(default=None, ge=0.0, le=2.0, strict=True)

    @model_validator(mode="after")
    def check_duration_use(self) -> MouseAction:
        if self.duration_s is not None and self.kind is not MouseActionKind.MOVE:
            raise ValueError("duration_s is supported only for mouse movement")
        return self


class KeyboardKey(StrEnum):
    """Explicitly supported keys; arbitrary virtual-key codes are not accepted."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    LETTER_I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    LETTER_O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"
    DIGIT_0 = "0"
    DIGIT_1 = "1"
    DIGIT_2 = "2"
    DIGIT_3 = "3"
    DIGIT_4 = "4"
    DIGIT_5 = "5"
    DIGIT_6 = "6"
    DIGIT_7 = "7"
    DIGIT_8 = "8"
    DIGIT_9 = "9"
    ENTER = "ENTER"
    TAB = "TAB"
    ESC = "ESC"
    SPACE = "SPACE"
    BACKSPACE = "BACKSPACE"
    DELETE = "DELETE"
    INSERT = "INSERT"
    HOME = "HOME"
    END = "END"
    PAGE_UP = "PAGE_UP"
    PAGE_DOWN = "PAGE_DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"
    F7 = "F7"
    F8 = "F8"
    F9 = "F9"
    F10 = "F10"
    F11 = "F11"
    F12 = "F12"


class KeyboardModifier(StrEnum):
    CTRL = "CTRL"
    SHIFT = "SHIFT"


class KeyboardActionKind(StrEnum):
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    TYPE_TEXT = "type_text"


_SAFE_HOTKEYS: frozenset[tuple[frozenset[KeyboardModifier], KeyboardKey]] = frozenset(
    {
        (frozenset({KeyboardModifier.CTRL}), key)
        for key in (
            KeyboardKey.A,
            KeyboardKey.C,
            KeyboardKey.F,
            KeyboardKey.N,
            KeyboardKey.LETTER_O,
            KeyboardKey.P,
            KeyboardKey.S,
            KeyboardKey.V,
            KeyboardKey.W,
            KeyboardKey.X,
            KeyboardKey.Y,
            KeyboardKey.Z,
            KeyboardKey.HOME,
            KeyboardKey.END,
            KeyboardKey.TAB,
        )
    }
    | {
        (frozenset({KeyboardModifier.SHIFT}), KeyboardKey.TAB),
        (frozenset({KeyboardModifier.CTRL, KeyboardModifier.SHIFT}), KeyboardKey.S),
        (frozenset({KeyboardModifier.CTRL, KeyboardModifier.SHIFT}), KeyboardKey.TAB),
    }
)


class KeyboardAction(ComputerModel):
    """A bounded key press, allow-listed shortcut, or literal text input."""

    action_id: str = Field(default_factory=_new_action_id, pattern=ACTION_ID_PATTERN)
    kind: KeyboardActionKind
    key: KeyboardKey | None = None
    modifiers: tuple[KeyboardModifier, ...] = Field(default_factory=tuple, max_length=2)
    text: str | None = Field(default=None, max_length=HARD_MAX_TEXT_INPUT_CHARS)

    @model_validator(mode="after")
    def validate_action_shape(self) -> KeyboardAction:
        if self.kind is KeyboardActionKind.PRESS_KEY:
            if self.key is None or self.modifiers or self.text is not None:
                raise ValueError("press_key requires only a supported key")
        elif self.kind is KeyboardActionKind.HOTKEY:
            if self.key is None or not self.modifiers or self.text is not None:
                raise ValueError("hotkey requires an allow-listed key and modifier set")
            if len(set(self.modifiers)) != len(self.modifiers):
                raise ValueError("hotkey modifiers must be unique")
            if (frozenset(self.modifiers), self.key) not in _SAFE_HOTKEYS:
                raise ValueError("hotkey is not in the safe shortcut allow-list")
        elif self.kind is KeyboardActionKind.TYPE_TEXT:
            if self.key is not None or self.modifiers or self.text is None or self.text == "":
                raise ValueError("type_text requires non-empty text only")
            if any(ord(char) < 32 and char not in "\t\n" for char in self.text):
                raise ValueError("type_text contains unsupported control characters")
            if "\x7f" in self.text:
                raise ValueError("type_text contains unsupported control characters")
        return self


class FocusWindowAction(ComputerModel):
    action_id: str = Field(default_factory=_new_action_id, pattern=ACTION_ID_PATTERN)
    window_identifier: str = Field(min_length=1, max_length=128)


class SelectUIElementAction(ComputerModel):
    action_id: str = Field(default_factory=_new_action_id, pattern=ACTION_ID_PATTERN)
    window_identifier: str = Field(min_length=1, max_length=128)
    automation_id: str = Field(min_length=1, max_length=128)


class VerificationKind(StrEnum):
    ACTIVE_WINDOW_CHANGED = "active_window_changed"
    ACTIVE_WINDOW_IS = "active_window_is"
    CURSOR_AT = "cursor_at"
    UI_ELEMENT_FOCUSED = "ui_element_focused"
    UI_ELEMENT_SELECTED = "ui_element_selected"
    UI_ELEMENT_PRESENCE = "ui_element_presence"
    SCREENSHOT_CHANGED = "screenshot_changed"
    WINDOW_TITLE_CHANGED = "window_title_changed"
    WINDOW_STATE_CHANGED = "window_state_changed"


class VerificationCondition(ComputerModel):
    """A deterministic postcondition evaluated only against observations."""

    kind: VerificationKind
    window_identifier: str | None = Field(default=None, max_length=128)
    automation_id: str | None = Field(default=None, max_length=128)
    point: Point | None = None
    tolerance_px: int = Field(default=2, ge=0, le=10, strict=True)
    expected_present: bool | None = Field(default=None, strict=True)
    expected_title: str | None = Field(default=None, max_length=256)
    expected_minimized: bool | None = Field(default=None, strict=True)
    expected_maximized: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def validate_condition_fields(self) -> VerificationCondition:
        if self.kind is VerificationKind.ACTIVE_WINDOW_IS and not self.window_identifier:
            raise ValueError("active_window_is requires window_identifier")
        if self.kind is VerificationKind.CURSOR_AT and self.point is None:
            raise ValueError("cursor_at requires point")
        if (
            self.kind
            in {
                VerificationKind.UI_ELEMENT_FOCUSED,
                VerificationKind.UI_ELEMENT_SELECTED,
                VerificationKind.UI_ELEMENT_PRESENCE,
            }
            and not self.automation_id
        ):
            raise ValueError(f"{self.kind.value} requires automation_id")
        if self.kind is VerificationKind.UI_ELEMENT_PRESENCE and self.expected_present is None:
            raise ValueError("ui_element_presence requires expected_present")
        if self.kind is VerificationKind.WINDOW_STATE_CHANGED:
            if not self.window_identifier:
                raise ValueError("window_state_changed requires window_identifier")
            if self.expected_minimized is None and self.expected_maximized is None:
                raise ValueError("window_state_changed requires an expected state")
        if self.kind is not VerificationKind.CURSOR_AT and self.point is not None:
            raise ValueError("point is supported only for cursor_at")
        return self


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class VerificationResult(ComputerModel):
    """Deterministic verification verdict, including an explicit reason."""

    status: VerificationStatus
    condition: VerificationKind | None = None
    reason: str = Field(min_length=1, max_length=256)
    expected: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    observed: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.PASSED


class RecoveryAction(StrEnum):
    RETRY = "retry"
    REFRESH_OBSERVATION = "refresh_observation"
    REQUERY_ACTIVE_WINDOW = "requery_active_window"
    REQUERY_UI_ELEMENT = "requery_ui_element"
    STOP = "stop"


class RecoveryRecommendation(ComputerModel):
    action: RecoveryAction
    reason: str = Field(min_length=1, max_length=256)
    retries_used: int = Field(default=0, ge=0, strict=True)
    retries_remaining: int = Field(default=0, ge=0, strict=True)


class ComputerActionStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    VERIFICATION_FAILED = "verification_failed"
    DENIED = "denied"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INVALID = "invalid"


class ScreenshotObservation(ComputerModel):
    """Ephemeral screenshot bytes plus bounded metadata; never persisted by core."""

    width: int = Field(gt=0, le=32_768, strict=True)
    height: int = Field(gt=0, le=32_768, strict=True)
    timestamp: datetime
    source: str = Field(default="primary_display", min_length=1, max_length=64)
    media_type: Literal["image/png"] = "image/png"
    payload: bytes | None = Field(default=None, repr=False)
    payload_bytes: int = Field(default=0, ge=0, le=HARD_MAX_SCREENSHOT_BYTES, strict=True)
    payload_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_payload(self) -> ScreenshotObservation:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("screenshot timestamp must be timezone-aware")
        if self.payload is not None:
            length = len(self.payload)
            if length > HARD_MAX_SCREENSHOT_BYTES:
                raise ValueError("screenshot payload exceeds the hard size limit")
            digest = hashlib.sha256(self.payload).hexdigest()
            if self.payload_sha256 is not None and self.payload_sha256 != digest:
                raise ValueError("screenshot payload digest does not match")
            object.__setattr__(self, "payload_bytes", length)
            object.__setattr__(self, "payload_sha256", digest)
        return self

    def metadata_only(self) -> ScreenshotObservation:
        """Return a screenshot observation without retaining its image bytes."""
        return self.model_copy(update={"payload": None})


class ComputerObservation(ComputerModel):
    """A single, explicitly timestamped provider observation.

    ``untrusted_content`` is fixed true because UI/window labels come from
    external applications. It is a trust annotation, not an instruction.
    """

    observation_id: str = Field(default_factory=_new_action_id, pattern=ACTION_ID_PATTERN)
    observed_at: datetime
    screen_info: ScreenInfo | None = None
    cursor_position: Point | None = None
    active_window: WindowInfo | None = None
    windows: list[WindowInfo] | None = None
    ui_elements: list[UIElement] | None = None
    screenshot: ScreenshotObservation | None = None
    windows_truncated: bool = Field(default=False, strict=True)
    ui_elements_truncated: bool = Field(default=False, strict=True)
    untrusted_content: Literal[True] = True

    @model_validator(mode="after")
    def require_aware_time(self) -> ComputerObservation:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observation timestamp must be timezone-aware")
        return self


class ComputerActionResult(ComputerModel):
    """Structured action outcome; provider completion alone is never success."""

    action_id: str = Field(pattern=ACTION_ID_PATTERN)
    action: str = Field(min_length=1, max_length=64)
    risk_level: PermissionLevel
    status: ComputerActionStatus
    attempted: bool = Field(strict=True)
    success: bool = Field(default=False, strict=True)
    provider_completed: bool = Field(default=False, strict=True)
    verified: bool = Field(default=False, strict=True)
    attempts: int = Field(ge=0, strict=True)
    observed_before: ComputerObservation | None = None
    observed_after: ComputerObservation | None = None
    recovery_observations: list[ComputerObservation] = Field(default_factory=list, max_length=3)
    verification: VerificationResult
    visual_verification: VisualVerificationResult | None = None
    error_code: str | None = Field(default=None, max_length=64)
    error: str | None = Field(default=None, max_length=256)
    retryable: bool = False
    recovery: RecoveryRecommendation

    @model_validator(mode="after")
    def successful_action_requires_semantic_postcondition(self) -> ComputerActionResult:
        if not self.success:
            return self
        if self.status is not ComputerActionStatus.VERIFIED or not self.verified:
            raise ValueError("successful actions must be explicitly verified")
        if self.verification.condition is VerificationKind.SCREENSHOT_CHANGED:
            raise ValueError("a screenshot change is not semantic action proof")
        if self.visual_verification is not None:
            if self.visual_verification.status is not VisualVerificationStatus.VERIFIED:
                raise ValueError("uncertain or failed visual verification cannot be successful")
            if (
                self.verification.status is not VerificationStatus.PASSED
                or self.verification.condition is None
            ):
                raise ValueError(
                    "visual verification also requires a passed semantic postcondition"
                )
        return self


class ComputerOperationResult(ComputerModel):
    """Successful or failed read-only operation result without external text."""

    ok: bool
    operation: str = Field(min_length=1, max_length=64)
    observation: ComputerObservation | None = None
    error_code: str | None = Field(default=None, max_length=64)
    error: str | None = Field(default=None, max_length=256)


def safe_external_text(value: object, *, max_length: int) -> str:
    """Normalize untrusted display text without interpreting its contents."""
    if not isinstance(value, str):
        return ""
    # Strip non-printing control characters and NULs; keep ordinary Unicode
    # and whitespace. Truncation is deterministic and never changes trust.
    cleaned = "".join(char for char in value if char.isprintable() or char in "\t\n")
    cleaned = cleaned.replace("\x00", "")
    return cleaned[:max_length]


def stable_identifier(value: object, *, max_length: int = 128) -> str:
    """Return a bounded provider identifier or a stable safe fallback."""
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._:-]", "_", value.strip())
    return cleaned[:max_length]
