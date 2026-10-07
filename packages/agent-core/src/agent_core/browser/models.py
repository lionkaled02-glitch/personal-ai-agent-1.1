"""Bounded browser models. All page-derived values are untrusted data."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..permissions import PermissionLevel
from .url_safety import safe_observed_url, validate_http_url

MAX_BROWSER_URL_CHARS = 4_096
MAX_BROWSER_TITLE_CHARS = 512
MAX_BROWSER_TEXT_CHARS = 20_000
MAX_BROWSER_ELEMENTS = 500
MAX_BROWSER_ELEMENT_TEXT_CHARS = 1_024
MAX_BROWSER_ATTRIBUTES = 32
MAX_BROWSER_ATTRIBUTE_CHARS = 512
MAX_BROWSER_FILL_CHARS = 2_048
MAX_BROWSER_SCREENSHOT_BYTES = 4 * 1_024 * 1_024
MAX_BROWSER_SESSIONS = 32
MAX_BROWSER_PAGES_PER_SESSION = 50
MAX_BROWSER_TIMEOUT_SECONDS = 30.0

BROWSER_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
BROWSER_REASON_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
BROWSER_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SAFE_ATTRIBUTE_NAMES = frozenset(
    {
        "type",
        "name",
        "id",
        "class",
        "role",
        "autocomplete",
        "placeholder",
        "aria-label",
        "disabled",
        "required",
        "checked",
        "selected",
    }
)


def _require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("browser timestamps must be timezone-aware")
    return value


class BrowserModel(BaseModel):
    """Immutable public base with strict shape and no unexpected keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class BrowserSessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class BrowserPageStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class BrowserReadyState(StrEnum):
    LOADING = "loading"
    DOMCONTENTLOADED = "domcontentloaded"
    LOAD = "load"
    UNKNOWN = "unknown"


class BrowserWaitUntil(StrEnum):
    DOMCONTENTLOADED = "domcontentloaded"
    LOAD = "load"


class BrowserWaitState(StrEnum):
    DOCUMENT_LOADED = "document_loaded"
    ELEMENT_ATTACHED = "element_attached"
    ELEMENT_DETACHED = "element_detached"
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_HIDDEN = "element_hidden"


class BrowserKey(StrEnum):
    """Small explicit keyboard allowlist; text entry uses the bounded fill tool."""

    ENTER = "Enter"
    TAB = "Tab"
    ESCAPE = "Escape"
    ARROW_UP = "ArrowUp"
    ARROW_DOWN = "ArrowDown"
    ARROW_LEFT = "ArrowLeft"
    ARROW_RIGHT = "ArrowRight"
    HOME = "Home"
    END = "End"
    BACKSPACE = "Backspace"
    DELETE = "Delete"
    SPACE = " "
    PAGE_UP = "PageUp"
    PAGE_DOWN = "PageDown"


class BrowserActionType(StrEnum):
    NAVIGATE = "navigate"
    BACK = "back"
    FORWARD = "forward"
    RELOAD = "reload"
    CLICK = "click"
    FILL = "fill"
    SELECT_OPTION = "select_option"
    PRESS_KEY = "press_key"


class BrowserVerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class BrowserRecoveryAction(StrEnum):
    RETRY = "retry"
    REOBSERVE = "reobserve"
    STOP = "stop"


class BrowserSession(BrowserModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    status: BrowserSessionStatus = BrowserSessionStatus.ACTIVE
    created_at: datetime
    page_ids: list[str] = Field(default_factory=list, max_length=MAX_BROWSER_PAGES_PER_SESSION)
    current_page_id: str | None = Field(default=None, max_length=128, pattern=BROWSER_ID_PATTERN)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @field_validator("page_ids")
    @classmethod
    def validate_page_ids(cls, values: list[str]) -> list[str]:
        if any(re.fullmatch(BROWSER_ID_PATTERN, value) is None for value in values):
            raise ValueError("invalid browser page identifier")
        if len(values) != len(set(values)):
            raise ValueError("browser page identifiers must be unique")
        return values


class BrowserPage(BrowserModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    page_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    status: BrowserPageStatus = BrowserPageStatus.OPEN
    url: str | None = Field(default=None, max_length=MAX_BROWSER_URL_CHARS, strict=True)
    title: str = Field(default="", max_length=MAX_BROWSER_TITLE_CHARS, strict=True)
    untrusted_content: Literal[True] = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return safe_observed_url(value)


class BrowserBoundingBox(BrowserModel):
    x: float = Field(ge=0, le=1_000_000, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0, le=1_000_000, allow_inf_nan=False, strict=True)
    width: float = Field(gt=0, le=1_000_000, allow_inf_nan=False, strict=True)
    height: float = Field(gt=0, le=1_000_000, allow_inf_nan=False, strict=True)


class BrowserElement(BrowserModel):
    """A bounded, ephemeral element reference; attributes exclude live values."""

    element_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    role: str = Field(default="unknown", min_length=1, max_length=64, strict=True)
    tag_name: str = Field(default="unknown", min_length=1, max_length=32, strict=True)
    accessible_name: str = Field(default="", max_length=MAX_BROWSER_ELEMENT_TEXT_CHARS, strict=True)
    text: str = Field(default="", max_length=MAX_BROWSER_ELEMENT_TEXT_CHARS, strict=True)
    attributes: dict[str, str] = Field(default_factory=dict, max_length=MAX_BROWSER_ATTRIBUTES)
    bounds: BrowserBoundingBox | None = None
    visible: bool = Field(default=True, strict=True)
    enabled: bool = Field(default=True, strict=True)
    focused: bool = Field(default=False, strict=True)
    sensitive: bool = Field(default=False, strict=True)
    is_form_submit: bool = Field(default=False, strict=True)
    requires_high_confirmation: bool = Field(default=False, strict=True)
    untrusted_content: Literal[True] = True

    @field_validator("attributes")
    @classmethod
    def restrict_attribute_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_BROWSER_ATTRIBUTES:
            raise ValueError("too many browser attributes")
        normalized: dict[str, str] = {}
        for key, item in value.items():
            name = key.lower()
            if name not in _SAFE_ATTRIBUTE_NAMES:
                raise ValueError("unsupported browser attribute metadata")
            if name in normalized:
                raise ValueError("duplicate browser attribute metadata")
            if len(item) > MAX_BROWSER_ATTRIBUTE_CHARS:
                raise ValueError("browser attribute metadata exceeds its bound")
            normalized[name] = item
        return normalized

    @model_validator(mode="after")
    def redact_sensitive_control_metadata(self) -> BrowserElement:
        if self.sensitive:
            safe_attributes = {
                key: value
                for key, value in self.attributes.items()
                if key.lower() in {"type", "autocomplete"}
            }
            object.__setattr__(self, "accessible_name", "[sensitive control]")
            object.__setattr__(self, "text", "")
            object.__setattr__(self, "attributes", safe_attributes)
        return self


class BrowserPageSnapshot(BrowserModel):
    """Provider-side page metadata before runtime redaction and limits."""

    url: str | None = Field(default=None, max_length=MAX_BROWSER_URL_CHARS, strict=True)
    title: str = Field(default="", max_length=MAX_BROWSER_TITLE_CHARS, strict=True)
    visible_text: str = Field(default="", max_length=MAX_BROWSER_TEXT_CHARS, strict=True)
    elements: list[BrowserElement] = Field(default_factory=list, max_length=MAX_BROWSER_ELEMENTS)
    ready_state: BrowserReadyState = BrowserReadyState.UNKNOWN
    untrusted_content: Literal[True] = True

    @field_validator("url")
    @classmethod
    def validate_provider_url(cls, value: str | None) -> str | None:
        if value is None or value == "about:blank":
            return value
        return validate_http_url(value)


class BrowserScreenshotMetadata(BrowserModel):
    """Non-persistent screenshot facts; encoded image bytes are never included."""

    width: int = Field(gt=0, le=8_192, strict=True)
    height: int = Field(gt=0, le=8_192, strict=True)
    payload_bytes: int = Field(gt=0, le=MAX_BROWSER_SCREENSHOT_BYTES, strict=True)
    sha256: str = Field(pattern=BROWSER_SHA256_PATTERN, strict=True)
    captured_at: datetime

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)


class BrowserObservation(BrowserModel):
    """Bounded page facts for reasoning; content remains explicitly untrusted."""

    observation_id: str = Field(
        min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True
    )
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    page_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    observed_at: datetime
    url: str | None = Field(default=None, max_length=MAX_BROWSER_URL_CHARS, strict=True)
    title: str = Field(default="", max_length=MAX_BROWSER_TITLE_CHARS, strict=True)
    visible_text: str = Field(default="", max_length=MAX_BROWSER_TEXT_CHARS, strict=True)
    elements: list[BrowserElement] = Field(default_factory=list, max_length=MAX_BROWSER_ELEMENTS)
    ready_state: BrowserReadyState = BrowserReadyState.UNKNOWN
    screenshot: BrowserScreenshotMetadata | None = None
    untrusted_content: Literal[True] = True

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @field_validator("url")
    @classmethod
    def validate_observed_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return safe_observed_url(value)


class BrowserVerificationResult(BrowserModel):
    action: BrowserActionType
    status: BrowserVerificationStatus
    reason_code: str = Field(pattern=BROWSER_REASON_PATTERN, strict=True)
    expected: dict[str, str | int | bool | None] = Field(default_factory=dict, max_length=8)
    observed: dict[str, str | int | bool | None] = Field(default_factory=dict, max_length=8)
    verified_at: datetime

    @field_validator("expected", "observed")
    @classmethod
    def bound_verification_metadata(
        cls, values: dict[str, str | int | bool | None]
    ) -> dict[str, str | int | bool | None]:
        if any(len(key) > 64 for key in values):
            raise ValueError("verification metadata key exceeds its bound")
        if any(
            isinstance(value, str) and len(value) > MAX_BROWSER_URL_CHARS
            for value in values.values()
        ):
            raise ValueError("verification metadata value exceeds its bound")
        return values

    @field_validator("verified_at")
    @classmethod
    def validate_verified_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)


class BrowserRecoveryResult(BrowserModel):
    action: BrowserRecoveryAction
    reason_code: str = Field(pattern=BROWSER_REASON_PATTERN, strict=True)
    attempts: int = Field(ge=1, le=4, strict=True)
    retries_remaining: int = Field(ge=0, le=3, strict=True)


class BrowserActionResult(BrowserModel):
    """Safe action receipt. It never carries raw input text or page HTML."""

    action_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    action: BrowserActionType
    status: BrowserVerificationStatus
    permission_level: PermissionLevel
    attempts: int = Field(ge=1, le=4, strict=True)
    before: BrowserObservation | None = None
    after: BrowserObservation | None = None
    verification: BrowserVerificationResult
    recovery: BrowserRecoveryResult
    error_code: str | None = Field(default=None, pattern=BROWSER_REASON_PATTERN, strict=True)


class BrowserEmptyRequest(BrowserModel):
    """Explicit empty input for bounded list operations."""


class BrowserPageRef(BrowserModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    page_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)


class BrowserHistoryRequest(BrowserPageRef):
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )


class BrowserOpenSessionRequest(BrowserModel):
    url: str | None = Field(default=None, max_length=MAX_BROWSER_URL_CHARS, strict=True)
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )

    @field_validator("url")
    @classmethod
    def validate_start_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value)

    @model_validator(mode="after")
    def require_url_for_start_timeout(self) -> BrowserOpenSessionRequest:
        if self.timeout_s is not None and self.url is None:
            raise ValueError("timeout_s only applies to a startup URL")
        return self


class BrowserCreatePageRequest(BrowserModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)


class BrowserObserveRequest(BrowserPageRef):
    include_screenshot: bool = Field(default=False, strict=True)


class BrowserNavigateRequest(BrowserPageRef):
    url: str = Field(min_length=1, max_length=MAX_BROWSER_URL_CHARS, strict=True)
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )
    wait_until: BrowserWaitUntil = BrowserWaitUntil.DOMCONTENTLOADED

    @field_validator("url")
    @classmethod
    def validate_navigation_url(cls, value: str) -> str:
        return validate_http_url(value)


class BrowserElementRequest(BrowserPageRef):
    element_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )


class BrowserFillRequest(BrowserElementRequest):
    text: str = Field(max_length=MAX_BROWSER_FILL_CHARS, strict=True)


class BrowserSelectRequest(BrowserElementRequest):
    option: str = Field(min_length=1, max_length=256, strict=True)


class BrowserPressKeyRequest(BrowserPageRef):
    key: BrowserKey
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )


class BrowserFindElementsRequest(BrowserPageRef):
    role: str | None = Field(default=None, min_length=1, max_length=64, strict=True)
    name: str | None = Field(default=None, max_length=MAX_BROWSER_ELEMENT_TEXT_CHARS, strict=True)
    limit: int = Field(default=50, ge=1, le=MAX_BROWSER_ELEMENTS, strict=True)


class BrowserWaitRequest(BrowserPageRef):
    state: BrowserWaitState
    element_id: str | None = Field(
        default=None, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True
    )
    timeout_s: float | None = Field(
        default=None, gt=0, le=MAX_BROWSER_TIMEOUT_SECONDS, allow_inf_nan=False, strict=True
    )


class BrowserClosePageRequest(BrowserPageRef):
    pass


class BrowserCloseSessionRequest(BrowserModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=BROWSER_ID_PATTERN, strict=True)


def utc_now() -> datetime:
    """Return timezone-aware UTC time for browser records."""
    return datetime.now(UTC)


def is_finite_timeout(value: float) -> bool:
    """Small helper used at runtime boundaries after strict model validation."""
    return math.isfinite(value)
