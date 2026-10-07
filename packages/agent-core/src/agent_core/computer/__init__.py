"""Provider-neutral computer-agent foundation (Phase 6).

Importing this package is safe on every platform. The optional Windows
adapter imports pywinauto/pywin32/Pillow only when instantiated on Windows.
"""

from .errors import (
    ComputerError,
    ComputerProviderError,
    ComputerValidationError,
    ProviderUnavailableError,
    UnsupportedPlatformError,
)
from .interfaces import ComputerProvider
from .limits import ComputerLimits
from .models import (
    HARD_MAX_SCREENSHOT_BYTES,
    HARD_MAX_TEXT_INPUT_CHARS,
    Bounds,
    ComputerActionResult,
    ComputerActionStatus,
    ComputerObservation,
    ComputerOperationResult,
    FocusWindowAction,
    KeyboardAction,
    KeyboardActionKind,
    KeyboardKey,
    KeyboardModifier,
    MouseAction,
    MouseActionKind,
    MouseButton,
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
    VerificationStatus,
    WindowInfo,
)
from .recovery import RecoveryPolicy
from .runtime import ComputerRuntime, classify_computer_operation
from .verification import not_run, verify_condition
from .windows import WindowsComputerProvider

__all__ = [
    "HARD_MAX_SCREENSHOT_BYTES",
    "HARD_MAX_TEXT_INPUT_CHARS",
    "Bounds",
    "ComputerActionResult",
    "ComputerActionStatus",
    "ComputerError",
    "ComputerLimits",
    "ComputerObservation",
    "ComputerOperationResult",
    "ComputerProvider",
    "ComputerProviderError",
    "ComputerRuntime",
    "ComputerValidationError",
    "FocusWindowAction",
    "KeyboardAction",
    "KeyboardActionKind",
    "KeyboardKey",
    "KeyboardModifier",
    "MouseAction",
    "MouseActionKind",
    "MouseButton",
    "Point",
    "ProviderUnavailableError",
    "RecoveryAction",
    "RecoveryPolicy",
    "RecoveryRecommendation",
    "ScreenInfo",
    "ScreenshotObservation",
    "SelectUIElementAction",
    "UIElement",
    "UnsupportedPlatformError",
    "VerificationCondition",
    "VerificationKind",
    "VerificationResult",
    "VerificationStatus",
    "WindowInfo",
    "WindowsComputerProvider",
    "classify_computer_operation",
    "not_run",
    "verify_condition",
]
