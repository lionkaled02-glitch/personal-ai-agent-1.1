"""Deterministic in-memory browser provider for offline tests and examples."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..computer.models import ScreenshotObservation
from .errors import (
    BrowserError,
    BrowserLimitError,
    BrowserProviderError,
    BrowserTimeoutError,
)
from .limits import BrowserLimits
from .models import (
    BrowserBoundingBox,
    BrowserElement,
    BrowserPageSnapshot,
    BrowserReadyState,
    BrowserWaitState,
    BrowserWaitUntil,
)
from .url_safety import validate_http_url

# A deterministic 1x1 PNG used only by opt-in screenshot tests.
_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j0ioAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class MockElementDefinition:
    """Page fixture metadata. No live form value can be configured here."""

    role: str = "button"
    tag_name: str = "button"
    accessible_name: str = ""
    text: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0
    width: float = 80.0
    height: float = 24.0
    visible: bool = True
    enabled: bool = True
    focused: bool = False
    sensitive: bool = False
    is_form_submit: bool = False
    requires_high_confirmation: bool = False
    click_url: str | None = None
    click_text: str | None = None


@dataclass
class _MockPage:
    page_id: str
    urls: list[str] = field(default_factory=lambda: ["about:blank"])
    history_index: int = 0
    title: str = ""
    visible_text: str = ""
    elements: list[MockElementDefinition] = field(default_factory=list)
    ready_state: BrowserReadyState = BrowserReadyState.DOMCONTENTLOADED
    is_open: bool = True

    @property
    def url(self) -> str:
        return self.urls[self.history_index]


@dataclass
class _MockContext:
    context_id: str
    pages: dict[str, _MockPage] = field(default_factory=dict)
    is_open: bool = True


@dataclass(frozen=True)
class _Failure:
    retryable: bool
    timeout: bool


class MockBrowserProvider:
    """Offline provider with explicit fixtures and injectable deterministic errors."""

    def __init__(self) -> None:
        self._launched = False
        self._next_context_id = 1
        self._next_page_id = 1
        self._contexts: dict[str, _MockContext] = {}
        self._element_cache: dict[
            tuple[str, str], tuple[BrowserElement, MockElementDefinition]
        ] = {}
        self._field_values: dict[tuple[str, str], str] = {}
        self._selected_values: dict[tuple[str, str], str] = {}
        self._failures: dict[str, list[_Failure]] = {}
        self.calls: list[str] = []

    def fail_next(
        self,
        operation: str,
        *,
        retryable: bool = False,
        timeout: bool = False,
    ) -> None:
        """Inject one safe, content-free provider failure for an operation."""
        self._failures.setdefault(operation, []).append(_Failure(retryable, timeout))

    def launch(self) -> None:
        self.calls.append("launch")
        self._launched = True

    def close_browser(self) -> None:
        self.calls.append("close_browser")
        for context in self._contexts.values():
            context.is_open = False
            for page in context.pages.values():
                page.is_open = False
        self._contexts.clear()
        self._element_cache.clear()
        self._field_values.clear()
        self._selected_values.clear()
        self._launched = False

    def create_context(self) -> str:
        self._require_launched()
        context_id = f"context-{self._next_context_id}"
        self._next_context_id += 1
        self._contexts[context_id] = _MockContext(context_id=context_id)
        self.calls.append("create_context")
        return context_id

    def close_context(self, context_id: str) -> None:
        context = self._context(context_id)
        context.is_open = False
        page_ids = set(context.pages)
        for page in context.pages.values():
            page.is_open = False
        for key in [key for key in self._element_cache if key[0] in page_ids]:
            self._element_cache.pop(key, None)
        for key in [key for key in self._field_values if key[0] in page_ids]:
            self._field_values.pop(key, None)
        for key in [key for key in self._selected_values if key[0] in page_ids]:
            self._selected_values.pop(key, None)
        self._contexts.pop(context_id, None)
        self.calls.append("close_context")

    def create_page(self, context_id: str) -> str:
        context = self._context(context_id)
        page_id = f"page-{self._next_page_id}"
        self._next_page_id += 1
        context.pages[page_id] = _MockPage(page_id=page_id)
        self.calls.append("create_page")
        return page_id

    def close_page(self, context_id: str, page_id: str) -> None:
        page = self._page(context_id, page_id)
        page.is_open = False
        for key in [key for key in self._element_cache if key[0] == page_id]:
            self._element_cache.pop(key, None)
        for key in [key for key in self._field_values if key[0] == page_id]:
            self._field_values.pop(key, None)
        for key in [key for key in self._selected_values if key[0] == page_id]:
            self._selected_values.pop(key, None)
        self.calls.append("close_page")

    def configure_page(
        self,
        page_id: str,
        *,
        url: str = "about:blank",
        title: str = "",
        visible_text: str = "",
        elements: list[MockElementDefinition] | None = None,
        ready_state: BrowserReadyState = BrowserReadyState.DOMCONTENTLOADED,
    ) -> None:
        """Set deterministic page fixture data without network access."""
        context_id = self._context_id_for_page(page_id)
        page = self._page(context_id, page_id)
        if url != "about:blank":
            validate_http_url(url)
        page.urls = [url]
        page.history_index = 0
        page.title = title
        page.visible_text = visible_text
        page.elements = list(elements or [])
        page.ready_state = ready_state
        self.calls.append("configure_page")

    def get_url(self, context_id: str, page_id: str) -> str | None:
        self.calls.append("get_url")
        url = self._page(context_id, page_id).url
        return None if url == "about:blank" else url

    def get_title(self, context_id: str, page_id: str) -> str:
        self.calls.append("get_title")
        return self._page(context_id, page_id).title

    def inspect_page(
        self,
        context_id: str,
        page_id: str,
        *,
        limits: BrowserLimits,
    ) -> BrowserPageSnapshot:
        self._maybe_fail("inspect_page")
        page = self._page(context_id, page_id)
        for key in [key for key in self._element_cache if key[0] == page_id]:
            self._element_cache.pop(key, None)
        elements: list[BrowserElement] = []
        for index, definition in enumerate(page.elements[: limits.max_elements]):
            element_id = f"{page_id}-el-{index}"
            allowed_attributes = {
                key: value[: limits.max_attribute_chars]
                for key, value in definition.attributes.items()
                if key.lower()
                in {
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
            }
            element = BrowserElement(
                element_id=element_id,
                role=definition.role[:64] or "unknown",
                tag_name=definition.tag_name[:32] or "unknown",
                accessible_name=definition.accessible_name[: limits.max_element_text_chars],
                text=definition.text[: limits.max_element_text_chars],
                attributes=allowed_attributes,
                bounds=BrowserBoundingBox(
                    x=definition.x,
                    y=definition.y,
                    width=max(definition.width, 0.01),
                    height=max(definition.height, 0.01),
                ),
                visible=definition.visible,
                enabled=definition.enabled,
                focused=definition.focused,
                sensitive=definition.sensitive
                or definition.attributes.get("type", "").lower() == "password",
                is_form_submit=definition.is_form_submit,
                requires_high_confirmation=definition.requires_high_confirmation,
            )
            elements.append(element)
            self._element_cache[(page_id, element_id)] = (element, definition)
        self.calls.append("inspect_page")
        return BrowserPageSnapshot(
            url=page.url,
            title=page.title[: limits.max_title_chars],
            visible_text=page.visible_text[: limits.max_text_chars],
            elements=elements,
            ready_state=page.ready_state,
        )

    def locate_elements(
        self,
        context_id: str,
        page_id: str,
        *,
        role: str | None,
        name: str | None,
        limit: int,
    ) -> list[BrowserElement]:
        self._page(context_id, page_id)
        values = [
            element
            for (cached_page_id, _), (element, _) in self._element_cache.items()
            if cached_page_id == page_id
            and (role is None or element.role.casefold() == role.casefold())
            and (name is None or name.casefold() in element.accessible_name.casefold())
        ]
        self.calls.append("locate_elements")
        return values[:limit]

    def navigate(
        self,
        context_id: str,
        page_id: str,
        url: str,
        *,
        timeout_ms: int,
        wait_until: BrowserWaitUntil,
    ) -> None:
        del timeout_ms, wait_until
        self._maybe_fail("navigate")
        validated = validate_http_url(url)
        page = self._page(context_id, page_id)
        page.urls = page.urls[: page.history_index + 1]
        page.urls.append(validated)
        page.history_index = len(page.urls) - 1
        page.ready_state = BrowserReadyState.DOMCONTENTLOADED
        self.calls.append("navigate")

    def go_back(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        del timeout_ms
        self._maybe_fail("go_back")
        page = self._page(context_id, page_id)
        if page.history_index > 0:
            page.history_index -= 1
        self.calls.append("go_back")

    def go_forward(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        del timeout_ms
        self._maybe_fail("go_forward")
        page = self._page(context_id, page_id)
        if page.history_index + 1 < len(page.urls):
            page.history_index += 1
        self.calls.append("go_forward")

    def reload(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        del timeout_ms
        self._maybe_fail("reload")
        self._page(context_id, page_id).ready_state = BrowserReadyState.LOAD
        self.calls.append("reload")

    def click(self, context_id: str, page_id: str, element_id: str, *, timeout_ms: int) -> None:
        del timeout_ms
        self._maybe_fail("click")
        page = self._page(context_id, page_id)
        element, definition = self._element(context_id, page_id, element_id)
        self._require_actionable(element)
        if definition.click_url is not None:
            validated = validate_http_url(definition.click_url)
            page.urls = [*page.urls[: page.history_index + 1], validated]
            page.history_index = len(page.urls) - 1
        if definition.click_text is not None:
            page.visible_text = definition.click_text
        self.calls.append("click")

    def fill(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        text: str,
        *,
        timeout_ms: int,
    ) -> None:
        del timeout_ms
        self._maybe_fail("fill")
        element, _ = self._element(context_id, page_id, element_id)
        self._require_actionable(element)
        if element.sensitive:
            raise BrowserError("sensitive_field_blocked")
        self._field_values[(page_id, element_id)] = text
        self.calls.append("fill")

    def verify_filled(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        expected_text: str,
        *,
        timeout_ms: int,
    ) -> bool | None:
        del timeout_ms
        self._maybe_fail("verify_filled")
        self._page(context_id, page_id)
        value = self._field_values.pop((page_id, element_id), None)
        return value == expected_text

    def press_key(self, context_id: str, page_id: str, key: str, *, timeout_ms: int) -> None:
        del timeout_ms
        self._maybe_fail("press_key")
        self._page(context_id, page_id)
        if key not in {
            "Enter",
            "Tab",
            "Escape",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End",
            "Backspace",
            "Delete",
            " ",
            "PageUp",
            "PageDown",
        }:
            raise BrowserError("invalid_input")
        self.calls.append("press_key")

    def select_option(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_ms: int,
    ) -> None:
        del timeout_ms
        self._maybe_fail("select_option")
        element, _ = self._element(context_id, page_id, element_id)
        self._require_actionable(element)
        if element.sensitive:
            raise BrowserError("sensitive_field_blocked")
        self._selected_values[(page_id, element_id)] = option
        self.calls.append("select_option")

    def verify_selected(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_ms: int,
    ) -> bool | None:
        del timeout_ms
        self._maybe_fail("verify_selected")
        self._page(context_id, page_id)
        selected = self._selected_values.pop((page_id, element_id), None)
        return selected == option

    def wait_for_state(
        self,
        context_id: str,
        page_id: str,
        state: BrowserWaitState,
        *,
        element_id: str | None,
        timeout_ms: int,
    ) -> None:
        del timeout_ms
        self._maybe_fail("wait_for_state")
        page = self._page(context_id, page_id)
        if state is BrowserWaitState.DOCUMENT_LOADED:
            if page.ready_state not in {BrowserReadyState.DOMCONTENTLOADED, BrowserReadyState.LOAD}:
                raise BrowserError("wait_state_unavailable")
            return
        if element_id is None:
            raise BrowserError("invalid_input")
        element, _ = self._element(context_id, page_id, element_id)
        passed = {
            BrowserWaitState.ELEMENT_ATTACHED: True,
            BrowserWaitState.ELEMENT_DETACHED: False,
            BrowserWaitState.ELEMENT_VISIBLE: element.visible,
            BrowserWaitState.ELEMENT_HIDDEN: not element.visible,
        }[state]
        if not passed:
            raise BrowserError("wait_state_unavailable")

    def screenshot(
        self,
        context_id: str,
        page_id: str,
        *,
        max_bytes: int,
    ) -> ScreenshotObservation:
        self._maybe_fail("screenshot")
        self._page(context_id, page_id)
        if len(_TEST_PNG) > max_bytes:
            raise BrowserLimitError("limit_exceeded")
        self.calls.append("screenshot")
        return ScreenshotObservation(
            width=1,
            height=1,
            timestamp=datetime.now(UTC),
            source="mock_browser",
            payload=_TEST_PNG,
        )

    def _require_launched(self) -> None:
        if not self._launched:
            raise BrowserError("browser_not_started")

    def _context_id_for_page(self, page_id: str) -> str:
        for context_id, context in self._contexts.items():
            if context.is_open and page_id in context.pages:
                return context_id
        raise BrowserError("page_not_found")

    def _context(self, context_id: str) -> _MockContext:
        self._require_launched()
        context = self._contexts.get(context_id)
        if context is None or not context.is_open:
            raise BrowserError("session_not_found")
        return context

    def _page(self, context_id: str, page_id: str) -> _MockPage:
        context = self._context(context_id)
        page = context.pages.get(page_id)
        if page is None:
            raise BrowserError("page_not_found")
        if not page.is_open:
            raise BrowserError("page_closed")
        return page

    def _element(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
    ) -> tuple[BrowserElement, MockElementDefinition]:
        self._page(context_id, page_id)
        result = self._element_cache.get((page_id, element_id))
        if result is None:
            raise BrowserError("stale_element")
        return result

    @staticmethod
    def _require_actionable(element: BrowserElement) -> None:
        if not element.visible or not element.enabled:
            raise BrowserError("element_not_actionable")

    def _maybe_fail(self, operation: str) -> None:
        failures = self._failures.get(operation)
        if not failures:
            return
        failure = failures.pop(0)
        if failure.timeout:
            raise BrowserTimeoutError()
        raise BrowserProviderError(retryable=failure.retryable)
