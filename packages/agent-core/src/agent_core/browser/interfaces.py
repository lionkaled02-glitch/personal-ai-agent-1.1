"""Provider-neutral, high-level browser automation contracts.

The interface exposes only named navigation, observation, and form actions.
It has no arbitrary selector execution, JavaScript evaluation, cookie/storage
API, shell, process, or filesystem capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..computer.models import ScreenshotObservation
from .limits import BrowserLimits
from .models import (
    BrowserElement,
    BrowserPageSnapshot,
    BrowserWaitState,
    BrowserWaitUntil,
)


class BrowserProvider(Protocol):
    """Browser boundary implemented by Playwright or a deterministic fake."""

    def launch(self) -> None: ...

    def close_browser(self) -> None: ...

    def create_context(self) -> str: ...

    def close_context(self, context_id: str) -> None: ...

    def create_page(self, context_id: str) -> str: ...

    def close_page(self, context_id: str, page_id: str) -> None: ...

    def get_url(self, context_id: str, page_id: str) -> str | None: ...

    def get_title(self, context_id: str, page_id: str) -> str: ...

    def inspect_page(
        self,
        context_id: str,
        page_id: str,
        *,
        limits: BrowserLimits,
    ) -> BrowserPageSnapshot: ...

    def locate_elements(
        self,
        context_id: str,
        page_id: str,
        *,
        role: str | None,
        name: str | None,
        limit: int,
    ) -> Sequence[BrowserElement]: ...

    def navigate(
        self,
        context_id: str,
        page_id: str,
        url: str,
        *,
        timeout_ms: int,
        wait_until: BrowserWaitUntil,
    ) -> None: ...

    def go_back(self, context_id: str, page_id: str, *, timeout_ms: int) -> None: ...

    def go_forward(self, context_id: str, page_id: str, *, timeout_ms: int) -> None: ...

    def reload(self, context_id: str, page_id: str, *, timeout_ms: int) -> None: ...

    def click(self, context_id: str, page_id: str, element_id: str, *, timeout_ms: int) -> None: ...

    def fill(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        text: str,
        *,
        timeout_ms: int,
    ) -> None: ...

    def verify_filled(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        expected_text: str,
        *,
        timeout_ms: int,
    ) -> bool | None: ...

    def press_key(self, context_id: str, page_id: str, key: str, *, timeout_ms: int) -> None: ...

    def select_option(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_ms: int,
    ) -> None: ...

    def verify_selected(
        self,
        context_id: str,
        page_id: str,
        element_id: str,
        option: str,
        *,
        timeout_ms: int,
    ) -> bool | None: ...

    def wait_for_state(
        self,
        context_id: str,
        page_id: str,
        state: BrowserWaitState,
        *,
        element_id: str | None,
        timeout_ms: int,
    ) -> None: ...

    def screenshot(
        self,
        context_id: str,
        page_id: str,
        *,
        max_bytes: int,
    ) -> ScreenshotObservation: ...
