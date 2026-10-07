"""Optional synchronous Playwright implementation of the browser boundary.

The Playwright package is imported only when ``launch`` is called. Browser
binaries are neither downloaded nor required by core installation or the
offline test suite. No browser profile is reused or persisted.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import struct
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from ..computer.models import ScreenshotObservation
from .errors import (
    BrowserError,
    BrowserLimitError,
    BrowserProviderError,
    BrowserProviderUnavailableError,
    BrowserTimeoutError,
    BrowserValidationError,
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

_ELEMENT_SELECTOR = "a,button,input,textarea,select,[role],[contenteditable]"

_PAGE_TEXT_SCRIPT = """(element, maximum) => {
    if (!element) return '';
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let output = '';
    let node;
    while (output.length < maximum && (node = walker.nextNode())) {
      const parent = node.parentElement;
      const controlSelector = 'input,textarea,select,option,[contenteditable]';
      if (!parent || parent.closest(controlSelector)) continue;
      const style = window.getComputedStyle(parent);
      if (style.display === 'none' || style.visibility === 'hidden') continue;
      const remaining = maximum - output.length;
      output += String(node.nodeValue || '').slice(0, remaining);
    }
    return output;
}"""
_ELEMENT_METADATA_SCRIPT = """(root, args) => {
    const [maximumElements, maximumText] = args;
    const selector = 'a,button,input,textarea,select,[role],[contenteditable]';
    const nodeList = root.querySelectorAll(selector);
    const nodes = [];
    for (let index = 0; index < nodeList.length && nodes.length < maximumElements; index += 1) {
      nodes.push({ element: nodeList[index], index });
    }
    const sensitiveTerms = [
      'password', 'passcode', 'passwd', 'authorization', 'cookie', 'api key', 'api-key',
      'api_key', 'access token', 'access-token', 'access_token', 'refresh token',
      'refresh-token', 'refresh_token', 'session', 'auth token', 'auth-token', 'auth_token',
      'oauth', 'csrf', 'secret', 'credential', 'credit', 'card', 'cc-number', 'cc-name',
      'cc-exp', 'cc-csc', 'cvv', 'cvc', 'one-time-code', 'email', 'e-mail', 'phone',
      'telephone', 'address', 'postal-code', 'ssn', 'social security', 'passport',
      'tax-id', 'birth', 'bday', 'government-id', 'driver-license', 'payment', 'billing', 'iban',
      'routing', 'bank', 'account', 'security code', 'username', 'user name', 'user-name', 'login'
    ];
    const highRiskTerms = [
      'buy', 'purchase', 'pay', 'send', 'submit', 'publish', 'post', 'delete', 'remove',
      'transfer', 'authorize', 'place order', 'confirm payment', 'security', 'password',
      'permission', 'access', 'authentication', 'two-factor', 'two factor', '2fa', 'mfa'
    ];
    return nodes.map(({ element: el, index }) => {
      const tag = (el.tagName || 'unknown').toLowerCase();
      const type = (el.getAttribute('type') || '').toLowerCase();
      const autocomplete = (el.getAttribute('autocomplete') || '').toLowerCase();
      const identity = [
        el.getAttribute('name') || '', el.id || '', autocomplete, type,
        el.getAttribute('placeholder') || '', el.getAttribute('aria-label') || ''
      ].join(' ').toLowerCase();
      const roleMap = {
        a: 'link', button: 'button', textarea: 'textbox', select: 'combobox',
        input: type === 'checkbox' ? 'checkbox' :
          (type === 'radio' ? 'radio' : (type === 'submit' ? 'button' : 'textbox'))
      };
      const role = el.getAttribute('role') || roleMap[tag] || tag || 'unknown';
      const associatedLabels = el.labels
        ? Array.from(el.labels).slice(0, 10)
          .map(item => String(item.innerText || '').slice(0, maximumText)).join(' ') : '';
      const isFormControl = tag === 'input' || tag === 'textarea' || tag === 'select';
      const fallbackLabel = isFormControl
        ? associatedLabels : String(el.innerText || el.getAttribute('title') || '');
      const label = el.getAttribute('aria-label') ||
        el.getAttribute('placeholder') || fallbackLabel;
      const sensitiveIdentity = `${identity} ${String(label || '').toLowerCase()}`;
      const sensitiveField = isFormControl ||
        ['textbox', 'combobox', 'searchbox'].includes(String(role).toLowerCase());
      const sensitive = Boolean(el.isContentEditable) ||
        (sensitiveField && (
          sensitiveTerms.some(term => sensitiveIdentity.includes(term)) ||
          /\\b(?:pin|auth|tel)\\b|\\bsocial[\\s_-]*security\\b/.test(sensitiveIdentity)));
      const text = isFormControl || sensitive || el.isContentEditable
        ? '' : String(el.innerText || '').slice(0, maximumText);
      const rect = el.getBoundingClientRect();
      const style = window.getComputedStyle(el);
      const visible = rect.width > 0 && rect.height > 0 &&
        style.visibility !== 'hidden' && style.display !== 'none';
      const attributes = {};
      const attributeNames = [
        'type', 'name', 'id', 'class', 'role', 'autocomplete', 'placeholder',
        'aria-label', 'disabled', 'required', 'checked', 'selected'
      ];
      for (const name of attributeNames) {
        const value = el.getAttribute(name);
        if (value !== null) attributes[name] = String(value).slice(0, 512);
      }
      const buttonType = (el.getAttribute('type') || '').toLowerCase();
      const formSubmit = type === 'submit' ||
        (tag === 'button' && Boolean(el.form) &&
         buttonType !== 'button' && buttonType !== 'reset');
      const riskName = `${label} ${text} ${el.getAttribute('name') || ''}`.toLowerCase();
      const requiresHigh = highRiskTerms.some(term => riskName.includes(term));
      return {
        index, role, tagName: tag,
        accessibleName: sensitive ? '[sensitive control]' :
          String(label || '').slice(0, maximumText),
        text, attributes, x: Math.max(0, rect.x), y: Math.max(0, rect.y),
        width: Math.max(0.01, rect.width), height: Math.max(0.01, rect.height),
        visible, enabled: !Boolean(el.disabled), focused: document.activeElement === el,
        sensitive, isFormSubmit: formSubmit, requiresHighConfirmation: requiresHigh
      };
    });
}"""

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


def _is_sensitive_identity(value: str) -> bool:
    return _SENSITIVE_FIELD.search(value) is not None


_HIGH_RISK_LABEL = re.compile(
    r"\b(buy|purchase|pay|send|submit|publish|post|delete|remove|transfer|authorize|"
    r"place order|confirm payment|security|password|permission|access|authentication|"
    r"two[- ]factor|2fa|mfa)\b",
    re.IGNORECASE,
)


class PlaywrightBrowserProvider:
    """High-level Playwright adapter with isolated ephemeral contexts."""

    def __init__(self) -> None:
        self._manager: Any | None = None
        self._browser: Any | None = None
        self._contexts: dict[str, Any] = {}
        self._pages: dict[tuple[str, str], Any] = {}
        self._element_locators: dict[tuple[str, str, str], Any] = {}
        self._element_metadata: dict[tuple[str, str], list[BrowserElement]] = {}
        self._next_context_id = 1
        self._next_page_id = 1

    def launch(self) -> None:
        if self._browser is not None:
            return
        try:
            module = importlib.import_module("playwright.sync_api")
        except ImportError:
            raise BrowserProviderUnavailableError() from None
        try:
            manager = module.sync_playwright()
            browser = manager.start().chromium.launch(headless=True)
        except Exception as exc:
            try:
                if "manager" in locals():
                    manager.stop()
            except Exception:
                pass
            if exc.__class__.__name__ == "Error" and "Executable doesn't exist" in str(exc):
                raise BrowserProviderUnavailableError() from None
            raise BrowserProviderError() from None
        self._manager = manager
        self._browser = browser

    def close_browser(self) -> None:
        for context_id in list(self._contexts):
            try:
                self.close_context(context_id)
            except BrowserError:
                continue
        if self._browser is not None:
            with suppress(Exception):
                self._browser.close()
        if self._manager is not None:
            with suppress(Exception):
                self._manager.stop()
        self._contexts.clear()
        self._pages.clear()
        self._element_locators.clear()
        self._element_metadata.clear()
        self._browser = None
        self._manager = None

    def create_context(self) -> str:
        browser = self._require_browser()
        context: Any | None = None
        try:
            context = browser.new_context(accept_downloads=False, service_workers="block")
            context.route("**/*", self._guard_navigation)
        except Exception:
            if context is not None:
                with suppress(Exception):
                    context.close()
            raise BrowserProviderError() from None
        context_id = f"context-{self._next_context_id}"
        self._next_context_id += 1
        self._contexts[context_id] = context
        return context_id

    def close_context(self, context_id: str) -> None:
        context = self._contexts.get(context_id)
        if context is None:
            raise BrowserError("session_not_found")
        try:
            context.close()
        except Exception:
            raise BrowserProviderError() from None
        self._contexts.pop(context_id, None)
        for page_key in [page_key for page_key in self._pages if page_key[0] == context_id]:
            self._pages.pop(page_key, None)
        for locator_key in [
            locator_key for locator_key in self._element_locators if locator_key[0] == context_id
        ]:
            self._element_locators.pop(locator_key, None)
        for metadata_key in [
            metadata_key for metadata_key in self._element_metadata if metadata_key[0] == context_id
        ]:
            self._element_metadata.pop(metadata_key, None)

    def create_page(self, context_id: str) -> str:
        context = self._require_context(context_id)
        page: Any | None = None
        try:
            page = context.new_page()
            page.set_default_timeout(5_000)
            page.on("popup", lambda popup: popup.close())
        except Exception:
            if page is not None:
                with suppress(Exception):
                    page.close()
            raise BrowserProviderError() from None
        page_id = f"page-{self._next_page_id}"
        self._next_page_id += 1
        self._pages[(context_id, page_id)] = page
        return page_id

    def close_page(self, context_id: str, page_id: str) -> None:
        page = self._require_page(context_id, page_id)
        try:
            page.close()
        except Exception:
            raise BrowserProviderError() from None
        self._pages.pop((context_id, page_id), None)
        for key in [key for key in self._element_locators if key[:2] == (context_id, page_id)]:
            self._element_locators.pop(key, None)
        self._element_metadata.pop((context_id, page_id), None)

    def get_url(self, context_id: str, page_id: str) -> str | None:
        page = self._require_page(context_id, page_id)
        url = str(page.url)
        if url == "about:blank":
            return None
        return self._safe_current_url(url)

    def get_title(self, context_id: str, page_id: str) -> str:
        page = self._require_page(context_id, page_id)
        try:
            title = page.title()
        except Exception as exc:
            raise self._canonical_error(exc) from None
        return str(title)[:512]

    def inspect_page(
        self,
        context_id: str,
        page_id: str,
        *,
        limits: BrowserLimits,
    ) -> BrowserPageSnapshot:
        page = self._require_page(context_id, page_id)
        try:
            raw_url = str(page.url)
            url = None if raw_url == "about:blank" else self._safe_current_url(raw_url)
            title = str(page.title())[: limits.max_title_chars]
            body = page.locator("body")
            text = body.evaluate(_PAGE_TEXT_SCRIPT, limits.max_text_chars) if body.count() else ""
            raw_elements = (
                body.evaluate(
                    _ELEMENT_METADATA_SCRIPT, [limits.max_elements, limits.max_element_text_chars]
                )
                if body.count()
                else []
            )
            ready_state = self._ready_state(page)
        except BrowserError:
            raise
        except Exception as exc:
            raise self._canonical_error(exc) from None

        self._clear_page_elements(context_id, page_id)
        elements: list[BrowserElement] = []
        for item in raw_elements[: limits.max_elements]:
            try:
                locator = body.locator(_ELEMENT_SELECTOR).nth(int(item["index"]))
                element_handle = locator.element_handle()
                if element_handle is None:
                    continue
                element_id = f"{page_id}-el-{int(item['index'])}"
                element = BrowserElement(
                    element_id=element_id,
                    role=str(item["role"])[:64] or "unknown",
                    tag_name=str(item["tagName"])[:32] or "unknown",
                    accessible_name=str(item["accessibleName"])[: limits.max_element_text_chars],
                    text=str(item["text"])[: limits.max_element_text_chars],
                    attributes={
                        str(key): str(value)[: limits.max_attribute_chars]
                        for key, value in dict(item["attributes"]).items()
                        if key
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
                    },
                    bounds=BrowserBoundingBox(
                        x=max(0.0, float(item["x"])),
                        y=max(0.0, float(item["y"])),
                        width=max(0.01, float(item["width"])),
                        height=max(0.01, float(item["height"])),
                    ),
                    visible=bool(item["visible"]),
                    enabled=bool(item["enabled"]),
                    focused=bool(item["focused"]),
                    sensitive=bool(item["sensitive"]),
                    is_form_submit=bool(item["isFormSubmit"]),
                    requires_high_confirmation=bool(item["requiresHighConfirmation"])
                    or _HIGH_RISK_LABEL.search(str(item["accessibleName"])) is not None,
                )
            except Exception:
                raise BrowserProviderError() from None
            elements.append(element)
            self._element_locators[(context_id, page_id, element_id)] = element_handle
        self._element_metadata[(context_id, page_id)] = elements
        return BrowserPageSnapshot(
            url=url,
            title=title,
            visible_text=str(text)[: limits.max_text_chars],
            elements=elements,
            ready_state=ready_state,
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
        self._require_page(context_id, page_id)
        elements = self._element_metadata.get((context_id, page_id), [])
        matches = [
            element
            for element in elements
            if (role is None or element.role.casefold() == role.casefold())
            and (name is None or name.casefold() in element.accessible_name.casefold())
        ]
        return matches[: max(0, limit)]

    def navigate(
        self,
        context_id: str,
        page_id: str,
        url: str,
        *,
        timeout_ms: int,
        wait_until: BrowserWaitUntil,
    ) -> None:
        validated = validate_http_url(url)
        page = self._require_page(context_id, page_id)
        try:
            page.goto(validated, timeout=timeout_ms, wait_until=wait_until.value)
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def go_back(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        page = self._require_page(context_id, page_id)
        try:
            page.go_back(timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def go_forward(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        page = self._require_page(context_id, page_id)
        try:
            page.go_forward(timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def reload(self, context_id: str, page_id: str, *, timeout_ms: int) -> None:
        page = self._require_page(context_id, page_id)
        try:
            page.reload(timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def click(self, context_id: str, page_id: str, element_id: str, *, timeout_ms: int) -> None:
        page = self._require_page(context_id, page_id)
        element = self._require_element_locator(context_id, page_id, element_id)
        try:
            tag_name = str(element.evaluate("el => (el.tagName || '').toLowerCase()"))
            href = element.get_attribute("href") if tag_name == "a" else None
            if href is not None and urlsplit(href).scheme.lower() in {"javascript", "data", "file"}:
                raise BrowserValidationError("unsupported_scheme")
            element.click(timeout=timeout_ms)
        except BrowserError:
            raise
        except Exception as exc:
            raise self._canonical_error(exc) from None
        if page.is_closed():
            raise BrowserError("page_closed")

    def fill(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        text: str,
        *,
        timeout_ms: int,
    ) -> None:
        element = self._require_element_locator(context_id, page_id, element_id)
        try:
            input_type = (element.get_attribute("type") or "").lower()
            autocomplete = (element.get_attribute("autocomplete") or "").lower()
            identity = " ".join(
                [
                    input_type,
                    autocomplete,
                    element.get_attribute("name") or "",
                    element.get_attribute("id") or "",
                ]
            ).lower()
            if _is_sensitive_identity(identity):
                raise BrowserError("sensitive_field_blocked")
            element.fill(text, timeout=timeout_ms)
        except BrowserError:
            raise
        except Exception as exc:
            raise self._canonical_error(exc) from None

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
        element = self._require_element_locator(context_id, page_id, element_id)
        try:
            current = element.input_value()
        except Exception as exc:
            raise self._canonical_error(exc) from None
        return bool(current == expected_text)

    def press_key(self, context_id: str, page_id: str, key: str, *, timeout_ms: int) -> None:
        page = self._require_page(context_id, page_id)
        try:
            page.keyboard.press(key, timeout=timeout_ms)
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def select_option(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_ms: int,
    ) -> None:
        element = self._require_element_locator(context_id, page_id, element_id)
        try:
            identity = " ".join(
                [
                    element.get_attribute("autocomplete") or "",
                    element.get_attribute("name") or "",
                    element.get_attribute("id") or "",
                ]
            ).lower()
            if _is_sensitive_identity(identity):
                raise BrowserError("sensitive_field_blocked")
            element.select_option(label=option, timeout=timeout_ms)
        except BrowserError:
            raise
        except Exception as exc:
            raise self._canonical_error(exc) from None

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
        element = self._require_element_locator(context_id, page_id, element_id)
        try:
            selected = element.locator("option:checked").first
            label = selected.inner_text(timeout=1_000)
        except Exception as exc:
            raise self._canonical_error(exc) from None
        return bool(label == option)

    def wait_for_state(
        self,
        context_id: str,
        page_id: str,
        state: BrowserWaitState,
        *,
        element_id: str | None,
        timeout_ms: int,
    ) -> None:
        page = self._require_page(context_id, page_id)
        try:
            if state is BrowserWaitState.DOCUMENT_LOADED:
                page.wait_for_load_state("load", timeout=timeout_ms)
                return
            if element_id is None:
                raise BrowserError("invalid_input")
            element = self._require_element_locator(context_id, page_id, element_id)
            if state is BrowserWaitState.ELEMENT_ATTACHED:
                if not bool(element.evaluate("el => el.isConnected")):
                    raise BrowserError("wait_state_unavailable")
                return
            if state is BrowserWaitState.ELEMENT_DETACHED:
                page.wait_for_function("el => !el.isConnected", element, timeout=timeout_ms)
                return
            wait_state = "visible" if state is BrowserWaitState.ELEMENT_VISIBLE else "hidden"
            element.wait_for_element_state(wait_state, timeout=timeout_ms)
        except BrowserError:
            raise
        except Exception as exc:
            raise self._canonical_error(exc) from None

    def screenshot(
        self,
        context_id: str,
        page_id: str,
        *,
        max_bytes: int,
    ) -> ScreenshotObservation:
        page = self._require_page(context_id, page_id)
        try:
            payload = page.screenshot(
                type="png",
                full_page=False,
                animations="disabled",
                timeout=5_000,
            )
        except Exception as exc:
            raise self._canonical_error(exc) from None
        if len(payload) > max_bytes or len(payload) > 4 * 1_024 * 1_024:
            raise BrowserLimitError("limit_exceeded")
        if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
            raise BrowserProviderError()
        width, height = struct.unpack(">II", payload[16:24])
        if not width or not height or width > 8_192 or height > 8_192:
            raise BrowserLimitError("limit_exceeded")
        return ScreenshotObservation(
            width=width,
            height=height,
            timestamp=datetime.now(UTC),
            source="playwright_browser",
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _guard_navigation(self, route: Any) -> None:
        try:
            request = route.request
            url = str(request.url)
            if url != "about:blank":
                validate_http_url(url)
            route.continue_()
        except BrowserError:
            with suppress(Exception):
                route.abort()
        except Exception:
            with suppress(Exception):
                route.abort()

    def _safe_current_url(self, url: str) -> str:
        if url == "about:blank":
            return url
        return validate_http_url(url)

    def _ready_state(self, page: Any) -> BrowserReadyState:
        try:
            state = page.evaluate("() => document.readyState")
        except Exception:
            return BrowserReadyState.UNKNOWN
        return {
            "loading": BrowserReadyState.LOADING,
            "interactive": BrowserReadyState.DOMCONTENTLOADED,
            "complete": BrowserReadyState.LOAD,
        }.get(str(state), BrowserReadyState.UNKNOWN)

    def _require_browser(self) -> Any:
        if self._browser is None:
            raise BrowserError("browser_not_started")
        return self._browser

    def _require_context(self, context_id: str) -> Any:
        self._require_browser()
        context = self._contexts.get(context_id)
        if context is None:
            raise BrowserError("session_not_found")
        return context

    def _require_page(self, context_id: str, page_id: str) -> Any:
        self._require_context(context_id)
        page = self._pages.get((context_id, page_id))
        if page is None:
            raise BrowserError("page_not_found")
        try:
            if page.is_closed():
                raise BrowserError("page_closed")
        except BrowserError:
            raise
        except Exception:
            raise BrowserError("page_closed") from None
        return page

    def _clear_page_elements(self, context_id: str, page_id: str) -> None:
        for key in [key for key in self._element_locators if key[:2] == (context_id, page_id)]:
            self._element_locators.pop(key, None)
        self._element_metadata.pop((context_id, page_id), None)

    def _require_element_locator(self, context_id: str, page_id: str, element_id: str) -> Any:
        self._require_page(context_id, page_id)
        locator = self._element_locators.get((context_id, page_id, element_id))
        if locator is None:
            raise BrowserError("stale_element")
        return locator

    @staticmethod
    def _canonical_error(exc: Exception) -> BrowserError:
        name = exc.__class__.__name__.lower()
        if "timeout" in name:
            return BrowserTimeoutError()
        if "closed" in str(exc).lower():
            return BrowserError("page_closed")
        return BrowserProviderError()
