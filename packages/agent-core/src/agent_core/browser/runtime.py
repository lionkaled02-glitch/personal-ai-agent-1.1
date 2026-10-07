"""Permissioned browser runtime with bounded observe/action/verify behavior."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..computer.models import ScreenshotObservation
from ..events import Clock, EventBus, EventType, utc_now
from ..permissions import (
    ApprovalRequest,
    PermissionDecision,
    PermissionLevel,
    PermissionManager,
    _active_permission_authorization,
)
from ..vision.errors import VisionError
from ..vision.runtime import VisionRuntime
from .errors import (
    BrowserError,
    BrowserLimitError,
    BrowserPermissionError,
    BrowserProviderError,
    BrowserTimeoutError,
    BrowserValidationError,
)
from .interfaces import BrowserProvider
from .limits import BrowserLimits
from .models import (
    BrowserActionResult,
    BrowserActionType,
    BrowserClosePageRequest,
    BrowserCloseSessionRequest,
    BrowserCreatePageRequest,
    BrowserElement,
    BrowserFillRequest,
    BrowserFindElementsRequest,
    BrowserKey,
    BrowserNavigateRequest,
    BrowserObservation,
    BrowserObserveRequest,
    BrowserOpenSessionRequest,
    BrowserPage,
    BrowserPageRef,
    BrowserPageSnapshot,
    BrowserPageStatus,
    BrowserPressKeyRequest,
    BrowserRecoveryAction,
    BrowserRecoveryResult,
    BrowserScreenshotMetadata,
    BrowserSelectRequest,
    BrowserSession,
    BrowserSessionStatus,
    BrowserVerificationResult,
    BrowserVerificationStatus,
    BrowserWaitRequest,
    BrowserWaitState,
    BrowserWaitUntil,
    is_finite_timeout,
)
from .recovery import BrowserRecoveryPolicy
from .sanitization import redact_sensitive_text
from .serialization import action_event_payload
from .url_safety import safe_observed_url, validate_http_url
from .verification import (
    uncertain_verification,
    verify_navigation,
    verify_observed_state_change,
    verify_value_action,
)

BROWSER_OPEN_SESSION = "browser_open_session"
BROWSER_CREATE_PAGE = "browser_create_page"
BROWSER_CLOSE_PAGE = "browser_close_page"
BROWSER_CLOSE_SESSION = "browser_close_session"
BROWSER_LIST_SESSIONS = "browser_list_sessions"
BROWSER_GET_PAGE = "browser_get_page"
BROWSER_GET_URL = "browser_get_url"
BROWSER_GET_TITLE = "browser_get_title"
BROWSER_OBSERVE = "browser_observe"
BROWSER_FIND_ELEMENTS = "browser_find_elements"
BROWSER_WAIT_FOR_STATE = "browser_wait_for_state"
BROWSER_NAVIGATE = "browser_navigate"
BROWSER_GO_BACK = "browser_go_back"
BROWSER_GO_FORWARD = "browser_go_forward"
BROWSER_RELOAD = "browser_reload"
BROWSER_CLICK = "browser_click"
BROWSER_FILL = "browser_fill"
BROWSER_SELECT_OPTION = "browser_select_option"
BROWSER_PRESS_KEY = "browser_press_key"

_SENSITIVE_FIELD = re.compile(
    r"password|passcode|passwd|authorization|set.?cookie|cookie|api[_ -]?key|"
    r"access[_ -]?token|refresh[_ -]?token|session(?:[_ -]?(?:id|token|key))?|"
    r"auth(?:[_ -]?(?:token|key))|\bauth\b|oauth|csrf|secret|credential|"
    r"credit|card|cc[_ -]?(?:number|exp|csc|type|name)|cvv|cvc|security.?code|"
    r"one-time-code|e.?mail|phone|telephone|\btel\b|address|postal.?code|"
    r"ssn|social.?security|passport|tax-id|birth|bday|government.?id|driver.?license|"
    r"payment|billing|iban|routing|bank|"
    r"account|username|user.?name|login|\bpin\b",
    re.IGNORECASE,
)

_HIGH_RISK_LABEL = re.compile(
    r"\b(buy|purchase|pay|send|submit|publish|post|delete|remove|transfer|authorize|"
    r"place order|confirm payment|security|password|permission|access|authentication|"
    r"two[- ]factor|2fa|mfa)\b",
    re.IGNORECASE,
)


class _PermissionDescriptor:
    """Structural descriptor consumed by the shared PermissionManager."""

    def __init__(self, name: str, permission_level: PermissionLevel) -> None:
        self.name = name
        self.permission_level = permission_level


@dataclass
class _RuntimeSession:
    session: BrowserSession
    context_id: str
    pages: dict[str, BrowserPage] = field(default_factory=dict)
    observations: dict[str, BrowserObservation] = field(default_factory=dict)


@dataclass(frozen=True)
class _Authorization:
    task_id: str
    step_id: str


class BrowserRuntime:
    """Provider-neutral browser operations with explicit session/page scope.

    Every action names both a session and page. Element actions require a fresh
    observed element reference. Page-derived strings are bounded and marked as
    untrusted; event payloads contain only operational metadata.
    """

    def __init__(
        self,
        provider: BrowserProvider,
        permissions: PermissionManager,
        *,
        events: EventBus | None = None,
        limits: BrowserLimits | None = None,
        vision: VisionRuntime | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._provider = provider
        self._permissions = permissions
        self._events = events if events is not None else EventBus()
        self._limits = limits or BrowserLimits()
        self._vision = vision
        self._clock: Clock = clock or utc_now
        self._recovery = BrowserRecoveryPolicy(max_retries=self._limits.max_retries)
        self._browser_started = False
        self._sessions: dict[str, _RuntimeSession] = {}

    @property
    def limits(self) -> BrowserLimits:
        """Validated immutable runtime bounds."""
        return self._limits

    @property
    def events(self) -> EventBus:
        """The shared in-process event bus used by this runtime."""
        return self._events

    def open_session(
        self,
        url: str | None = None,
        *,
        timeout_s: float | None = None,
    ) -> BrowserSession:
        """Launch the provider lazily and create a fresh isolated context/page."""
        try:
            request = BrowserOpenSessionRequest(url=url, timeout_s=timeout_s)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        if request.url is not None:
            validate_http_url(request.url, max_chars=self._limits.max_url_chars)
            self._timeout_ms(
                request.timeout_s,
                default_s=self._limits.max_navigation_time_s,
                maximum_s=self._limits.max_navigation_time_s,
            )
        action_id = self._new_id("session")
        authorization = self._authorize(
            BROWSER_OPEN_SESSION,
            PermissionLevel.MEDIUM,
            action_id=action_id,
            reason="Open an isolated ephemeral browser session.",
        )
        active_sessions = sum(
            value.session.status is BrowserSessionStatus.ACTIVE for value in self._sessions.values()
        )
        if active_sessions >= self._limits.max_sessions:
            raise BrowserLimitError("session_limit_exceeded")
        if len(self._sessions) >= self._limits.max_sessions:
            closed_session_id = next(
                (
                    session_id
                    for session_id, value in self._sessions.items()
                    if value.session.status is BrowserSessionStatus.CLOSED
                ),
                None,
            )
            if closed_session_id is None:
                raise BrowserLimitError("session_limit_exceeded")
            del self._sessions[closed_session_id]
        self._ensure_started()
        context_id: str | None = None
        try:
            context_id = self._provider.create_context()
            page_id = self._provider.create_page(context_id)
            self._validate_provider_id(context_id)
            self._validate_provider_id(page_id)
        except BrowserError:
            if context_id is not None:
                self._close_context_best_effort(context_id)
            raise
        except Exception:
            if context_id is not None:
                self._close_context_best_effort(context_id)
            raise BrowserProviderError() from None

        session_id = self._new_id("bs")
        page = BrowserPage(session_id=session_id, page_id=page_id)
        session = BrowserSession(
            session_id=session_id,
            created_at=self._clock(),
            page_ids=[page_id],
            current_page_id=page_id,
        )
        state = _RuntimeSession(session=session, context_id=context_id, pages={page_id: page})
        self._sessions[session_id] = state
        self._emit(
            EventType.BROWSER_SESSION_OPENED,
            authorization,
            {"session_id": session_id, "page_count": 1},
        )
        self._emit(
            EventType.BROWSER_PAGE_OPENED,
            authorization,
            {"session_id": session_id, "page_id": page_id},
        )

        if request.url is not None:
            try:
                self.navigate(
                    session_id,
                    page_id,
                    request.url,
                    timeout_s=request.timeout_s,
                )
            except BrowserError:
                try:
                    self._provider.close_context(state.context_id)
                except Exception:
                    raise BrowserProviderError() from None
                self._mark_session_closed(state)
                self._emit(
                    EventType.BROWSER_SESSION_CLOSED,
                    authorization,
                    {"session_id": session_id, "page_count": len(state.pages)},
                )
                raise
        return state.session

    def create_page(self, session_id: str) -> BrowserPage:
        """Open an additional page in the named session; no active-tab lookup."""
        try:
            request = BrowserCreatePageRequest(session_id=session_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state = self._require_session(request.session_id)
        action_id = self._new_id("page")
        authorization = self._authorize(
            BROWSER_CREATE_PAGE,
            PermissionLevel.LOW,
            action_id=action_id,
            reason="Create a blank page in the named browser session.",
        )
        if len(state.pages) >= self._limits.max_pages_per_session:
            raise BrowserLimitError("page_limit_exceeded")
        try:
            page_id = self._provider.create_page(state.context_id)
            self._validate_provider_id(page_id)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        page = BrowserPage(session_id=request.session_id, page_id=page_id)
        state.pages[page_id] = page
        state.session = state.session.model_copy(
            update={
                "page_ids": [*state.session.page_ids, page_id],
                "current_page_id": page_id,
            }
        )
        self._emit(
            EventType.BROWSER_PAGE_OPENED,
            authorization,
            {"session_id": request.session_id, "page_id": page_id},
        )
        return page

    def close_page(self, session_id: str, page_id: str) -> BrowserPage:
        """Close exactly the named page and invalidate its element references."""
        try:
            request = BrowserClosePageRequest(session_id=session_id, page_id=page_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, page = self._require_page(request.session_id, request.page_id)
        action_id = self._new_id("close")
        authorization = self._authorize(
            BROWSER_CLOSE_PAGE,
            PermissionLevel.MEDIUM,
            action_id=action_id,
            reason="Close the explicitly named browser page.",
        )
        try:
            self._provider.close_page(state.context_id, request.page_id)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        closed = page.model_copy(update={"status": BrowserPageStatus.CLOSED})
        state.pages[request.page_id] = closed
        state.observations.pop(request.page_id, None)
        next_page_id = next(
            (
                item.page_id
                for item in reversed(list(state.pages.values()))
                if item.status is BrowserPageStatus.OPEN
            ),
            None,
        )
        state.session = state.session.model_copy(update={"current_page_id": next_page_id})
        self._emit(
            EventType.BROWSER_PAGE_CLOSED,
            authorization,
            {"session_id": request.session_id, "page_id": request.page_id},
        )
        return closed

    def close_session(self, session_id: str) -> BrowserSession:
        """Close the named isolated context and all its pages."""
        try:
            request = BrowserCloseSessionRequest(session_id=session_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state = self._require_session(request.session_id)
        action_id = self._new_id("close")
        authorization = self._authorize(
            BROWSER_CLOSE_SESSION,
            PermissionLevel.MEDIUM,
            action_id=action_id,
            reason="Close the explicitly named isolated browser session.",
        )
        try:
            self._provider.close_context(state.context_id)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        self._mark_session_closed(state)
        self._emit(
            EventType.BROWSER_SESSION_CLOSED,
            authorization,
            {"session_id": request.session_id, "page_count": len(state.pages)},
        )
        return state.session

    def list_sessions(self) -> list[BrowserSession]:
        """Return bounded session/page identifiers after a LOW permission check."""
        authorization = self._authorize(
            BROWSER_LIST_SESSIONS,
            PermissionLevel.LOW,
            action_id=self._new_id("list"),
            reason="List browser session identifiers and statuses.",
        )
        del authorization
        return [state.session for state in self._sessions.values()]

    def get_page(self, session_id: str, page_id: str) -> BrowserPage:
        """Return safe metadata for one explicitly named page."""
        try:
            request = BrowserPageRef(session_id=session_id, page_id=page_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, page = self._require_page(request.session_id, request.page_id)
        self._authorize(
            BROWSER_GET_PAGE,
            PermissionLevel.LOW,
            action_id=self._new_id("get"),
            reason="Read bounded metadata for the named browser page.",
        )
        return self._refresh_page_metadata(state, page)

    def get_url(self, session_id: str, page_id: str) -> str | None:
        """Return a validated page URL without query or fragment values."""
        state, page = self._checked_page_ref(session_id, page_id, BROWSER_GET_URL)
        del page
        try:
            raw_url = self._provider.get_url(state.context_id, page_id)
            return safe_observed_url(raw_url, max_chars=self._limits.max_url_chars)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None

    def get_title(self, session_id: str, page_id: str) -> str:
        """Return bounded, credential-redacted title metadata."""
        state, _ = self._checked_page_ref(session_id, page_id, BROWSER_GET_TITLE)
        try:
            raw_title = self._provider.get_title(state.context_id, page_id)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        return redact_sensitive_text(raw_title, max_chars=self._limits.max_title_chars)

    def observe(
        self,
        session_id: str,
        page_id: str,
        *,
        include_screenshot: bool = False,
    ) -> BrowserObservation:
        """Observe the named page; content is bounded untrusted data."""
        try:
            request = BrowserObserveRequest(
                session_id=session_id,
                page_id=page_id,
                include_screenshot=include_screenshot,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, _ = self._require_page(request.session_id, request.page_id)
        authorization = self._authorize(
            BROWSER_OBSERVE,
            PermissionLevel.LOW,
            action_id=self._new_id("observe"),
            reason="Read a bounded browser observation for the named page.",
        )
        return self._observe_page(
            state,
            request.page_id,
            include_screenshot=request.include_screenshot,
            authorization=authorization,
            emit_event=True,
        )

    def find_elements(
        self,
        session_id: str,
        page_id: str,
        *,
        role: str | None = None,
        name: str | None = None,
        limit: int = 50,
    ) -> list[BrowserElement]:
        """Locate explicit element references from the latest bounded observation."""
        try:
            request = BrowserFindElementsRequest(
                session_id=session_id,
                page_id=page_id,
                role=role,
                name=name,
                limit=limit,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, _ = self._require_page(request.session_id, request.page_id)
        self._authorize(
            BROWSER_FIND_ELEMENTS,
            PermissionLevel.LOW,
            action_id=self._new_id("find"),
            reason="Locate page elements within the latest bounded observation.",
        )
        previous = state.observations.get(request.page_id)
        if previous is None:
            raise BrowserError("observation_required")
        try:
            matches = self._provider.locate_elements(
                state.context_id,
                request.page_id,
                role=request.role,
                name=request.name,
                limit=min(request.limit, self._limits.max_elements),
            )
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        known = {element.element_id: element for element in previous.elements}
        results: list[BrowserElement] = []
        for raw_match in matches[: self._limits.max_elements]:
            try:
                candidate = BrowserElement.model_validate(raw_match)
            except ValidationError:
                raise BrowserProviderError() from None
            element = known.get(candidate.element_id)
            if element is not None:
                results.append(element)
        return results[: request.limit]

    def wait_for_state(
        self,
        session_id: str,
        page_id: str,
        state: BrowserWaitState,
        *,
        element_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Wait for one allowlisted state without returning page text."""
        try:
            request = BrowserWaitRequest(
                session_id=session_id,
                page_id=page_id,
                state=state,
                element_id=element_id,
                timeout_s=timeout_s,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state_data, _ = self._require_page(request.session_id, request.page_id)
        timeout_ms = self._timeout_ms(
            request.timeout_s,
            default_s=self._limits.max_wait_time_s,
            maximum_s=self._limits.max_wait_time_s,
        )
        self._authorize(
            BROWSER_WAIT_FOR_STATE,
            PermissionLevel.LOW,
            action_id=self._new_id("wait"),
            reason="Wait for a bounded document or observed-element state.",
        )
        if request.element_id is not None:
            observation = state_data.observations.get(request.page_id)
            if observation is None:
                raise BrowserError("observation_required")
            if not any(item.element_id == request.element_id for item in observation.elements):
                raise BrowserError("stale_element")
        try:
            self._provider.wait_for_state(
                state_data.context_id,
                request.page_id,
                request.state,
                element_id=request.element_id,
                timeout_ms=timeout_ms,
            )
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        return {"state": request.state.value, "observed": True}

    def navigate(
        self,
        session_id: str,
        page_id: str,
        url: str,
        *,
        timeout_s: float | None = None,
        wait_until: BrowserWaitUntil = BrowserWaitUntil.DOMCONTENTLOADED,
    ) -> BrowserActionResult:
        """Navigate to one explicitly supplied HTTP(S) URL with bounded retries."""
        try:
            request = BrowserNavigateRequest(
                session_id=session_id,
                page_id=page_id,
                url=url,
                timeout_s=timeout_s,
                wait_until=wait_until,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        expected_url = validate_http_url(request.url, max_chars=self._limits.max_url_chars)
        state, _ = self._require_page(request.session_id, request.page_id)
        timeout_ms = self._timeout_ms(
            request.timeout_s,
            default_s=self._limits.max_navigation_time_s,
            maximum_s=self._limits.max_navigation_time_s,
        )
        before = self._observe_page(
            state, request.page_id, include_screenshot=False, authorization=None, emit_event=False
        )
        action_id = self._new_id("action")
        authorization = self._authorize(
            BROWSER_NAVIGATE,
            PermissionLevel.MEDIUM,
            action_id=action_id,
            reason="Navigate the explicitly named page to a validated HTTP(S) URL.",
            action=BrowserActionType.NAVIGATE,
        )
        self._emit_action_started(
            authorization, action_id, BrowserActionType.NAVIGATE, PermissionLevel.MEDIUM
        )
        attempts = 0
        provider_error: BrowserError | None = None
        after: BrowserObservation | None = None
        verification = uncertain_verification(
            BrowserActionType.NAVIGATE,
            reason_code="navigation_not_observed",
            verified_at=self._clock(),
        )
        recovery = BrowserRecoveryResult(
            action=BrowserRecoveryAction.STOP,
            reason_code="no_safe_retry_available",
            attempts=1,
            retries_remaining=0,
        )
        while True:
            attempts += 1
            provider_error = None
            try:
                self._provider.navigate(
                    state.context_id,
                    request.page_id,
                    expected_url,
                    timeout_ms=timeout_ms,
                    wait_until=request.wait_until,
                )
            except BrowserError as exc:
                provider_error = exc
            except Exception:
                provider_error = BrowserProviderError()
            try:
                after = self._observe_page(
                    state,
                    request.page_id,
                    include_screenshot=False,
                    authorization=authorization,
                    emit_event=False,
                )
            except BrowserError:
                after = None
            try:
                actual_url = self._provider.get_url(state.context_id, request.page_id)
            except Exception:
                actual_url = None
            verification = verify_navigation(
                expected_url,
                after,
                actual_url=actual_url,
                verified_at=self._clock(),
            )
            if verification.status is BrowserVerificationStatus.VERIFIED:
                provider_error = None
                recovery = self._recovery.recommend(
                    action=BrowserActionType.NAVIGATE,
                    risk_level=PermissionLevel.MEDIUM,
                    status=verification.status,
                    attempts=attempts,
                )
                break
            if isinstance(provider_error, BrowserTimeoutError):
                verification = uncertain_verification(
                    BrowserActionType.NAVIGATE,
                    reason_code="navigation_timeout_outcome_uncertain",
                    verified_at=self._clock(),
                )
                recovery = self._recovery.recommend(
                    action=BrowserActionType.NAVIGATE,
                    risk_level=PermissionLevel.MEDIUM,
                    status=verification.status,
                    attempts=attempts,
                )
                break
            unchanged = after is not None and self._observations_match(before, after)
            status = (
                BrowserVerificationStatus.UNCERTAIN
                if after is None
                else BrowserVerificationStatus.FAILED
                if unchanged or verification.status is BrowserVerificationStatus.FAILED
                else BrowserVerificationStatus.UNCERTAIN
            )
            verification = verification.model_copy(update={"status": status})
            retryable = (
                isinstance(provider_error, BrowserProviderError) and provider_error.retryable
            )
            recovery = self._recovery.recommend(
                action=BrowserActionType.NAVIGATE,
                risk_level=PermissionLevel.MEDIUM,
                status=status,
                attempts=attempts,
                provider_retryable=retryable,
            )
            self._emit_recovery(authorization, action_id, BrowserActionType.NAVIGATE, recovery)
            if recovery.action.value == "retry":
                continue
            break
        result = self._make_action_result(
            action_id=action_id,
            action=BrowserActionType.NAVIGATE,
            permission_level=PermissionLevel.MEDIUM,
            attempts=attempts,
            before=before,
            after=after,
            verification=verification,
            recovery=recovery,
            error_code=provider_error.code if provider_error is not None else None,
        )
        self._emit_action_result(authorization, result)
        return result

    def go_back(
        self, session_id: str, page_id: str, *, timeout_s: float | None = None
    ) -> BrowserActionResult:
        return self._run_history_action(
            session_id,
            page_id,
            action=BrowserActionType.BACK,
            tool_name=BROWSER_GO_BACK,
            operation="go_back",
            timeout_s=timeout_s,
        )

    def go_forward(
        self, session_id: str, page_id: str, *, timeout_s: float | None = None
    ) -> BrowserActionResult:
        return self._run_history_action(
            session_id,
            page_id,
            action=BrowserActionType.FORWARD,
            tool_name=BROWSER_GO_FORWARD,
            operation="go_forward",
            timeout_s=timeout_s,
        )

    def reload(
        self, session_id: str, page_id: str, *, timeout_s: float | None = None
    ) -> BrowserActionResult:
        return self._run_history_action(
            session_id,
            page_id,
            action=BrowserActionType.RELOAD,
            tool_name=BROWSER_RELOAD,
            operation="reload",
            timeout_s=timeout_s,
        )

    def click(
        self,
        session_id: str,
        page_id: str,
        element_id: str,
        *,
        timeout_s: float | None = None,
    ) -> BrowserActionResult:
        return self._run_element_action(
            session_id,
            page_id,
            element_id,
            action=BrowserActionType.CLICK,
            tool_name=BROWSER_CLICK,
            timeout_s=timeout_s,
            operation=lambda context, timeout: self._provider.click(
                context, page_id, element_id, timeout_ms=timeout
            ),
        )

    def fill(
        self,
        session_id: str,
        page_id: str,
        element_id: str,
        text: str,
        *,
        timeout_s: float | None = None,
    ) -> BrowserActionResult:
        try:
            request = BrowserFillRequest(
                session_id=session_id,
                page_id=page_id,
                element_id=element_id,
                text=text,
                timeout_s=timeout_s,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        if len(request.text) > self._limits.max_fill_chars:
            raise BrowserLimitError("limit_exceeded")
        return self._run_element_action(
            request.session_id,
            request.page_id,
            request.element_id,
            action=BrowserActionType.FILL,
            tool_name=BROWSER_FILL,
            timeout_s=request.timeout_s,
            operation=lambda context, timeout: self._provider.fill(
                context, request.page_id, request.element_id, request.text, timeout_ms=timeout
            ),
            verify_value=lambda context, timeout: self._provider.verify_filled(
                context,
                request.page_id,
                request.element_id,
                request.text,
                timeout_ms=timeout,
            ),
            expected_chars=len(request.text),
        )

    def select_option(
        self,
        session_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_s: float | None = None,
    ) -> BrowserActionResult:
        try:
            request = BrowserSelectRequest(
                session_id=session_id,
                page_id=page_id,
                element_id=element_id,
                option=option,
                timeout_s=timeout_s,
            )
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        if len(request.option) > self._limits.max_fill_chars:
            raise BrowserLimitError("limit_exceeded")
        return self._run_element_action(
            request.session_id,
            request.page_id,
            request.element_id,
            action=BrowserActionType.SELECT_OPTION,
            tool_name=BROWSER_SELECT_OPTION,
            timeout_s=request.timeout_s,
            operation=lambda context, timeout: self._provider.select_option(
                context,
                request.page_id,
                request.element_id,
                request.option,
                timeout_ms=timeout,
            ),
            verify_value=lambda context, timeout: self._provider.verify_selected(
                context,
                request.page_id,
                request.element_id,
                request.option,
                timeout_ms=timeout,
            ),
            expected_chars=len(request.option),
        )

    def press_key(
        self,
        session_id: str,
        page_id: str,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> BrowserActionResult:
        try:
            normalized_key = BrowserKey(key)
            request = BrowserPressKeyRequest(
                session_id=session_id,
                page_id=page_id,
                key=normalized_key,
                timeout_s=timeout_s,
            )
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                raise self._validation_error(exc) from None
            raise BrowserValidationError("invalid_input") from None
        state, _ = self._require_page(request.session_id, request.page_id)
        before = self._observe_page(
            state, request.page_id, include_screenshot=False, authorization=None, emit_event=False
        )
        timeout_ms = self._timeout_ms(
            request.timeout_s,
            default_s=self._limits.max_action_time_s,
            maximum_s=self._limits.max_action_time_s,
        )
        level = PermissionLevel.HIGH if request.key is BrowserKey.ENTER else PermissionLevel.MEDIUM
        action_id = self._new_id("action")
        authorization = self._authorize(
            BROWSER_PRESS_KEY,
            level,
            action_id=action_id,
            reason=(
                "Press the allowlisted Enter key, which may submit a form."
                if level is PermissionLevel.HIGH
                else "Press one allowlisted non-submitting keyboard key."
            ),
            action=BrowserActionType.PRESS_KEY,
        )
        self._emit_action_started(authorization, action_id, BrowserActionType.PRESS_KEY, level)
        provider_error: BrowserError | None = None
        try:
            self._provider.press_key(
                state.context_id, request.page_id, request.key.value, timeout_ms=timeout_ms
            )
        except BrowserError as exc:
            provider_error = exc
        except Exception:
            provider_error = BrowserProviderError()
        try:
            after = self._observe_page(
                state,
                request.page_id,
                include_screenshot=False,
                authorization=authorization,
                emit_event=False,
            )
        except BrowserError:
            after = None
        if isinstance(provider_error, BrowserTimeoutError):
            verification = uncertain_verification(
                BrowserActionType.PRESS_KEY,
                reason_code="key_action_outcome_uncertain",
                verified_at=self._clock(),
            )
        else:
            verification = verify_observed_state_change(
                BrowserActionType.PRESS_KEY,
                before,
                after,
                verified_at=self._clock(),
            )
        recovery = self._recovery.recommend(
            action=BrowserActionType.PRESS_KEY,
            risk_level=level,
            status=verification.status,
            attempts=1,
        )
        result = self._make_action_result(
            action_id=action_id,
            action=BrowserActionType.PRESS_KEY,
            permission_level=level,
            attempts=1,
            before=before,
            after=after,
            verification=verification,
            recovery=recovery,
            error_code=provider_error.code if provider_error is not None else None,
        )
        self._emit_action_result(authorization, result)
        return result

    def _run_history_action(
        self,
        session_id: str,
        page_id: str,
        *,
        action: BrowserActionType,
        tool_name: str,
        operation: str,
        timeout_s: float | None,
    ) -> BrowserActionResult:
        try:
            request = BrowserPageRef(session_id=session_id, page_id=page_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, _ = self._require_page(request.session_id, request.page_id)
        timeout_ms = self._timeout_ms(
            timeout_s,
            default_s=self._limits.max_navigation_time_s,
            maximum_s=self._limits.max_navigation_time_s,
        )
        before = self._observe_page(
            state, request.page_id, include_screenshot=False, authorization=None, emit_event=False
        )
        action_id = self._new_id("action")
        authorization = self._authorize(
            tool_name,
            PermissionLevel.MEDIUM,
            action_id=action_id,
            reason=f"Perform the explicitly requested {action.value} operation on this page.",
            action=action,
        )
        self._emit_action_started(authorization, action_id, action, PermissionLevel.MEDIUM)
        provider_error: BrowserError | None = None
        try:
            if operation == "go_back":
                self._provider.go_back(state.context_id, request.page_id, timeout_ms=timeout_ms)
            elif operation == "go_forward":
                self._provider.go_forward(state.context_id, request.page_id, timeout_ms=timeout_ms)
            else:
                self._provider.reload(state.context_id, request.page_id, timeout_ms=timeout_ms)
        except BrowserError as exc:
            provider_error = exc
        except Exception:
            provider_error = BrowserProviderError()
        try:
            after = self._observe_page(
                state,
                request.page_id,
                include_screenshot=False,
                authorization=authorization,
                emit_event=False,
            )
        except BrowserError:
            after = None
        if isinstance(provider_error, BrowserTimeoutError):
            verification = uncertain_verification(
                action,
                reason_code="history_action_outcome_uncertain",
                verified_at=self._clock(),
            )
        else:
            verification = verify_observed_state_change(
                action,
                before,
                after,
                verified_at=self._clock(),
            )
        recovery = self._recovery.recommend(
            action=action,
            risk_level=PermissionLevel.MEDIUM,
            status=verification.status,
            attempts=1,
        )
        result = self._make_action_result(
            action_id=action_id,
            action=action,
            permission_level=PermissionLevel.MEDIUM,
            attempts=1,
            before=before,
            after=after,
            verification=verification,
            recovery=recovery,
            error_code=provider_error.code if provider_error is not None else None,
        )
        self._emit_action_result(authorization, result)
        return result

    def _run_element_action(
        self,
        session_id: str,
        page_id: str,
        element_id: str,
        *,
        action: BrowserActionType,
        tool_name: str,
        timeout_s: float | None,
        operation: Callable[[str, int], None],
        verify_value: Callable[[str, int], bool | None] | None = None,
        expected_chars: int = 0,
    ) -> BrowserActionResult:
        state, _ = self._require_page(session_id, page_id)
        previous = state.observations.get(page_id)
        if previous is None:
            raise BrowserError("observation_required")
        # Refresh bounded metadata before checking permission and resolving the
        # target. This prevents acting on an arbitrary/stale active-page handle.
        before = self._observe_page(
            state, page_id, include_screenshot=False, authorization=None, emit_event=False
        )
        element = next((item for item in before.elements if item.element_id == element_id), None)
        previous_element = next(
            (item for item in previous.elements if item.element_id == element_id), None
        )
        if element is None or previous_element is None:
            raise BrowserError("stale_element")
        if not self._same_element_identity(previous_element, element):
            raise BrowserError("stale_element")
        if element.sensitive:
            raise BrowserError("sensitive_field_blocked")
        if not element.visible or not element.enabled:
            raise BrowserError("element_not_actionable")
        high_risk = (
            element.is_form_submit
            or element.requires_high_confirmation
            or _HIGH_RISK_LABEL.search(element.accessible_name) is not None
        )
        timeout_ms = self._timeout_ms(
            timeout_s,
            default_s=self._limits.max_action_time_s,
            maximum_s=self._limits.max_action_time_s,
        )
        level = PermissionLevel.HIGH if high_risk else PermissionLevel.MEDIUM
        action_id = self._new_id("action")
        authorization = self._authorize(
            tool_name,
            level,
            action_id=action_id,
            reason=(
                "Confirm this form submission or externally consequential page action."
                if high_risk
                else f"Perform one explicit {action.value} on a fresh observed element."
            ),
            action=action,
        )
        self._emit_action_started(authorization, action_id, action, level)
        provider_error: BrowserError | None = None
        try:
            operation(state.context_id, timeout_ms)
        except BrowserError as exc:
            provider_error = exc
        except Exception:
            provider_error = BrowserProviderError()
        try:
            after = self._observe_page(
                state,
                page_id,
                include_screenshot=False,
                authorization=authorization,
                emit_event=False,
            )
        except BrowserError:
            after = None
        if verify_value is not None:
            try:
                matches = verify_value(state.context_id, timeout_ms)
            except BrowserError:
                matches = None
            except Exception:
                matches = None
            verification = verify_value_action(
                action,
                matches=matches,
                expected_chars=expected_chars,
                verified_at=self._clock(),
            )
        elif isinstance(provider_error, BrowserTimeoutError):
            verification = uncertain_verification(
                action,
                reason_code="action_timeout_outcome_uncertain",
                verified_at=self._clock(),
            )
        else:
            verification = verify_observed_state_change(
                action,
                before,
                after,
                verified_at=self._clock(),
            )
        recovery = self._recovery.recommend(
            action=action,
            risk_level=level,
            status=verification.status,
            attempts=1,
        )
        result = self._make_action_result(
            action_id=action_id,
            action=action,
            permission_level=level,
            attempts=1,
            before=before,
            after=after,
            verification=verification,
            recovery=recovery,
            error_code=provider_error.code if provider_error is not None else None,
        )
        self._emit_action_result(authorization, result)
        return result

    def _observe_page(
        self,
        state: _RuntimeSession,
        page_id: str,
        *,
        include_screenshot: bool,
        authorization: _Authorization | None,
        emit_event: bool,
    ) -> BrowserObservation:
        self._ensure_started()
        try:
            raw_snapshot = self._provider.inspect_page(
                state.context_id, page_id, limits=self._limits
            )
            snapshot = BrowserPageSnapshot.model_validate(raw_snapshot)
        except BrowserError:
            raise
        except ValidationError:
            raise BrowserProviderError() from None
        except Exception:
            raise BrowserProviderError() from None

        safe_url = safe_observed_url(snapshot.url, max_chars=self._limits.max_url_chars)
        title = redact_sensitive_text(snapshot.title, max_chars=self._limits.max_title_chars)
        visible_text = redact_sensitive_text(
            snapshot.visible_text, max_chars=self._limits.max_text_chars
        )
        elements = [
            self._sanitize_element(element)
            for element in snapshot.elements[: self._limits.max_elements]
        ]
        screenshot_metadata = None
        if include_screenshot:
            screenshot_metadata = self._observe_screenshot(state.context_id, page_id)
        observation = BrowserObservation(
            observation_id=self._new_id("obs"),
            session_id=state.session.session_id,
            page_id=page_id,
            observed_at=self._clock(),
            url=safe_url,
            title=title,
            visible_text=visible_text,
            elements=elements,
            ready_state=snapshot.ready_state,
            screenshot=screenshot_metadata,
        )
        state.observations[page_id] = observation
        current_page = state.pages[page_id]
        state.pages[page_id] = current_page.model_copy(
            update={"url": safe_url, "title": title, "status": BrowserPageStatus.OPEN}
        )
        if emit_event:
            self._emit(
                EventType.BROWSER_OBSERVED,
                authorization,
                {
                    "session_id": state.session.session_id,
                    "page_id": page_id,
                    "visible_text_chars": len(visible_text),
                    "element_count": len(elements),
                    "screenshot_metadata_present": screenshot_metadata is not None,
                },
            )
        return observation

    def _observe_screenshot(self, context_id: str, page_id: str) -> BrowserScreenshotMetadata:
        if self._vision is None:
            raise BrowserError("screenshot_unavailable")
        try:
            screenshot: ScreenshotObservation = self._provider.screenshot(
                context_id,
                page_id,
                max_bytes=self._limits.max_screenshot_bytes,
            )
            if screenshot.payload is None or screenshot.payload_bytes < 1:
                raise BrowserError("screenshot_unavailable")
            if screenshot.payload_bytes > self._limits.max_screenshot_bytes:
                raise BrowserLimitError("limit_exceeded")
            frame = self._vision.validate_screenshot(screenshot)
            metadata = BrowserScreenshotMetadata(
                width=frame.size.width,
                height=frame.size.height,
                payload_bytes=frame.payload_bytes,
                sha256=frame.sha256,
                captured_at=screenshot.timestamp,
            )
        except BrowserError:
            raise
        except VisionError:
            raise BrowserError("screenshot_unavailable") from None
        except Exception:
            raise BrowserError("screenshot_unavailable") from None
        return metadata

    def _sanitize_element(self, element: BrowserElement) -> BrowserElement:
        raw_attributes = dict(element.attributes)
        identity = " ".join(
            [
                *(raw_attributes.get(key, "") for key in ("type", "name", "id", "autocomplete")),
                element.accessible_name,
                element.text,
            ]
        )
        is_form_control = element.tag_name.casefold() in {"input", "textarea", "select"} or (
            element.role.casefold() in {"textbox", "combobox", "searchbox"}
        )
        sensitive = element.sensitive or (
            is_form_control and _SENSITIVE_FIELD.search(identity) is not None
        )
        attributes = {
            key: redact_sensitive_text(value, max_chars=self._limits.max_attribute_chars)
            for key, value in list(raw_attributes.items())[: self._limits.max_attributes]
        }
        safe_name = redact_sensitive_text(
            element.accessible_name, max_chars=self._limits.max_element_text_chars
        )
        safe_text = redact_sensitive_text(
            element.text, max_chars=self._limits.max_element_text_chars
        )
        if sensitive:
            safe_name = "[sensitive control]"
            safe_text = ""
        return BrowserElement.model_validate(
            {
                "element_id": element.element_id,
                "role": element.role,
                "tag_name": element.tag_name,
                "accessible_name": safe_name,
                "text": safe_text,
                "attributes": attributes,
                "bounds": element.bounds,
                "visible": element.visible,
                "enabled": element.enabled,
                "focused": element.focused,
                "sensitive": sensitive,
                "is_form_submit": element.is_form_submit,
                "requires_high_confirmation": element.requires_high_confirmation,
            }
        )

    def _refresh_page_metadata(self, state: _RuntimeSession, page: BrowserPage) -> BrowserPage:
        try:
            raw_url = self._provider.get_url(state.context_id, page.page_id)
            raw_title = self._provider.get_title(state.context_id, page.page_id)
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        safe_url = safe_observed_url(raw_url, max_chars=self._limits.max_url_chars)
        safe_title = redact_sensitive_text(raw_title, max_chars=self._limits.max_title_chars)
        updated = page.model_copy(update={"url": safe_url, "title": safe_title})
        state.pages[page.page_id] = updated
        return updated

    def _checked_page_ref(
        self, session_id: str, page_id: str, tool_name: str
    ) -> tuple[_RuntimeSession, BrowserPage]:
        try:
            request = BrowserPageRef(session_id=session_id, page_id=page_id)
        except ValidationError as exc:
            raise self._validation_error(exc) from None
        state, page = self._require_page(request.session_id, request.page_id)
        self._authorize(
            tool_name,
            PermissionLevel.LOW,
            action_id=self._new_id("get"),
            reason="Read bounded metadata for the explicitly named browser page.",
        )
        return state, page

    def _require_session(self, session_id: str) -> _RuntimeSession:
        state = self._sessions.get(session_id)
        if state is None or state.session.status is not BrowserSessionStatus.ACTIVE:
            raise BrowserError("session_not_found")
        return state

    def _require_page(self, session_id: str, page_id: str) -> tuple[_RuntimeSession, BrowserPage]:
        state = self._require_session(session_id)
        page = state.pages.get(page_id)
        if page is None:
            raise BrowserError("page_not_found")
        if page.status is not BrowserPageStatus.OPEN:
            raise BrowserError("page_closed")
        return state, page

    def _authorize(
        self,
        tool_name: str,
        level: PermissionLevel,
        *,
        action_id: str,
        reason: str,
        action: BrowserActionType | None = None,
    ) -> _Authorization:
        scope = _active_permission_authorization()
        if scope is not None:
            task_id, step_id = scope.task_id, scope.step_id
        else:
            task_id, step_id = "browser-direct", action_id
        self._events.emit(
            EventType.BROWSER_ACTION_REQUESTED,
            task_id=task_id,
            step_id=step_id,
            data={
                "action_id": action_id,
                "tool_name": tool_name,
                "action": action.value if action is not None else tool_name,
                "permission_level": level.name,
            },
        )
        if scope is not None and scope.tool_name == tool_name and scope.permission_level >= level:
            return _Authorization(scope.task_id, scope.step_id)
        decision = self._permissions.check(_PermissionDescriptor(tool_name, level))
        if decision is PermissionDecision.DENIED:
            self._deny_authorization(task_id, step_id, action_id, tool_name, "permission_denied")
            raise BrowserPermissionError("permission_denied")
        if decision is PermissionDecision.REQUIRES_APPROVAL:
            self._events.emit(
                EventType.APPROVAL_REQUIRED,
                task_id=task_id,
                step_id=step_id,
                data={
                    "action_id": action_id,
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
                self._deny_authorization(task_id, step_id, action_id, tool_name, "approval_denied")
                raise BrowserPermissionError("approval_denied")
        return _Authorization(task_id, step_id)

    def _deny_authorization(
        self,
        task_id: str,
        step_id: str,
        action_id: str,
        tool_name: str,
        error_code: str,
    ) -> None:
        self._events.emit(
            EventType.BROWSER_ACTION_DENIED,
            task_id=task_id,
            step_id=step_id,
            data={"action_id": action_id, "tool_name": tool_name, "error_code": error_code},
        )

    def _emit_action_started(
        self,
        authorization: _Authorization,
        action_id: str,
        action: BrowserActionType,
        level: PermissionLevel,
    ) -> None:
        self._emit(
            EventType.BROWSER_ACTION_STARTED,
            authorization,
            {
                "action_id": action_id,
                "action": action.value,
                "permission_level": level.name,
            },
        )

    def _emit_action_result(
        self, authorization: _Authorization, result: BrowserActionResult
    ) -> None:
        event_type = (
            EventType.BROWSER_ACTION_COMPLETED
            if result.status is BrowserVerificationStatus.VERIFIED
            else EventType.BROWSER_ACTION_FAILED
        )
        self._emit(event_type, authorization, action_event_payload(result))
        if result.recovery.action.value != "stop":
            self._emit_recovery(authorization, result.action_id, result.action, result.recovery)

    def _emit_recovery(
        self,
        authorization: _Authorization,
        action_id: str,
        action: BrowserActionType,
        recovery: BrowserRecoveryResult,
    ) -> None:
        self._emit(
            EventType.BROWSER_RECOVERY,
            authorization,
            {
                "action_id": action_id,
                "action": action.value,
                "recovery": recovery.action.value,
                "reason": recovery.reason_code,
                "attempts": recovery.attempts,
                "retries_remaining": recovery.retries_remaining,
            },
        )

    def _emit(
        self,
        event_type: EventType,
        authorization: _Authorization | None,
        data: dict[str, Any],
    ) -> None:
        self._events.emit(
            event_type,
            task_id=authorization.task_id if authorization is not None else None,
            step_id=authorization.step_id if authorization is not None else None,
            data=data,
        )

    def _make_action_result(
        self,
        *,
        action_id: str,
        action: BrowserActionType,
        permission_level: PermissionLevel,
        attempts: int,
        before: BrowserObservation | None,
        after: BrowserObservation | None,
        verification: BrowserVerificationResult,
        recovery: BrowserRecoveryResult,
        error_code: str | None,
    ) -> BrowserActionResult:
        return BrowserActionResult(
            action_id=action_id,
            action=action,
            status=verification.status,
            permission_level=permission_level,
            attempts=attempts,
            before=before,
            after=after,
            verification=verification,
            recovery=recovery,
            error_code=error_code,
        )

    @staticmethod
    def _observations_match(before: BrowserObservation, after: BrowserObservation) -> bool:
        """Compare only bounded, redacted page state to prove no navigation occurred."""
        return (
            before.url == after.url
            and before.title == after.title
            and before.visible_text == after.visible_text
            and before.ready_state == after.ready_state
            and [item.element_id for item in before.elements]
            == [item.element_id for item in after.elements]
        )

    @staticmethod
    def _same_element_identity(old: BrowserElement, new: BrowserElement) -> bool:
        """Reject a refreshed element reference if its bounded identity changed."""
        return (
            old.role == new.role
            and old.tag_name == new.tag_name
            and old.accessible_name == new.accessible_name
            and old.attributes == new.attributes
            and old.sensitive == new.sensitive
            and old.is_form_submit == new.is_form_submit
            and old.requires_high_confirmation == new.requires_high_confirmation
        )

    def _timeout_ms(
        self,
        timeout_s: float | None,
        *,
        default_s: float,
        maximum_s: float,
    ) -> int:
        value = default_s if timeout_s is None else timeout_s
        if not is_finite_timeout(value) or value <= 0 or value > maximum_s:
            raise BrowserLimitError("timeout_limit_exceeded")
        return max(1, int(value * 1_000))

    def _ensure_started(self) -> None:
        if self._browser_started:
            return
        try:
            self._provider.launch()
        except BrowserError:
            raise
        except Exception:
            raise BrowserProviderError() from None
        self._browser_started = True

    @staticmethod
    def _mark_session_closed(state: _RuntimeSession) -> None:
        for page_id, page in list(state.pages.items()):
            state.pages[page_id] = page.model_copy(update={"status": BrowserPageStatus.CLOSED})
        state.observations.clear()
        state.session = state.session.model_copy(
            update={"status": BrowserSessionStatus.CLOSED, "current_page_id": None}
        )

    def _close_context_best_effort(self, context_id: str) -> None:
        try:
            self._provider.close_context(context_id)
        except Exception:
            return

    def _close_session_provider(self, state: _RuntimeSession) -> None:
        try:
            self._provider.close_context(state.context_id)
        except Exception:
            return

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _validate_provider_id(value: str) -> None:
        try:
            BrowserPageRef(session_id="provider-validation", page_id=value)
        except ValidationError:
            raise BrowserProviderError() from None

    @staticmethod
    def _validation_error(exc: ValidationError) -> BrowserError:
        del exc
        return BrowserValidationError("invalid_input")

    def shutdown(self) -> None:
        """Close all ephemeral sessions/browser resources; not an agent tool."""
        for state in self._sessions.values():
            if state.session.status is BrowserSessionStatus.ACTIVE:
                self._close_session_provider(state)
                self._mark_session_closed(state)
        try:
            if self._browser_started:
                self._provider.close_browser()
        except Exception:
            pass
        self._browser_started = False
