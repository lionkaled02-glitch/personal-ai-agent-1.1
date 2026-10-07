"""Hard-clamped limits for browser navigation, observations, and actions."""

from __future__ import annotations

from pydantic import Field

from ..config import Settings
from .models import (
    MAX_BROWSER_ATTRIBUTE_CHARS,
    MAX_BROWSER_ATTRIBUTES,
    MAX_BROWSER_ELEMENT_TEXT_CHARS,
    MAX_BROWSER_ELEMENTS,
    MAX_BROWSER_FILL_CHARS,
    MAX_BROWSER_PAGES_PER_SESSION,
    MAX_BROWSER_SCREENSHOT_BYTES,
    MAX_BROWSER_SESSIONS,
    MAX_BROWSER_TEXT_CHARS,
    MAX_BROWSER_TITLE_CHARS,
    MAX_BROWSER_URL_CHARS,
    BrowserModel,
)


class BrowserLimits(BrowserModel):
    """Bounds callers may tighten but cannot widen past model hard caps."""

    max_url_chars: int = Field(default=2_048, ge=1, le=MAX_BROWSER_URL_CHARS, strict=True)
    max_title_chars: int = Field(default=256, ge=1, le=MAX_BROWSER_TITLE_CHARS, strict=True)
    max_text_chars: int = Field(default=4_000, ge=1, le=MAX_BROWSER_TEXT_CHARS, strict=True)
    max_elements: int = Field(default=100, ge=0, le=MAX_BROWSER_ELEMENTS, strict=True)
    max_element_text_chars: int = Field(
        default=256, ge=1, le=MAX_BROWSER_ELEMENT_TEXT_CHARS, strict=True
    )
    max_attributes: int = Field(default=12, ge=0, le=MAX_BROWSER_ATTRIBUTES, strict=True)
    max_attribute_chars: int = Field(default=128, ge=1, le=MAX_BROWSER_ATTRIBUTE_CHARS, strict=True)
    max_fill_chars: int = Field(default=1_024, ge=1, le=MAX_BROWSER_FILL_CHARS, strict=True)
    max_screenshot_bytes: int = Field(
        default=1_048_576, ge=1, le=MAX_BROWSER_SCREENSHOT_BYTES, strict=True
    )
    max_sessions: int = Field(default=5, ge=1, le=MAX_BROWSER_SESSIONS, strict=True)
    max_pages_per_session: int = Field(
        default=10, ge=1, le=MAX_BROWSER_PAGES_PER_SESSION, strict=True
    )
    max_navigation_time_s: float = Field(default=10.0, gt=0, le=30, strict=True)
    max_action_time_s: float = Field(default=5.0, gt=0, le=30, strict=True)
    max_wait_time_s: float = Field(default=5.0, gt=0, le=30, strict=True)
    max_retries: int = Field(default=1, ge=0, le=3, strict=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> BrowserLimits:
        """Construct validated bounds from the shared BROWSER_* settings."""
        return cls(
            max_url_chars=settings.browser_max_url_chars,
            max_title_chars=settings.browser_max_title_chars,
            max_text_chars=settings.browser_max_text_chars,
            max_elements=settings.browser_max_elements,
            max_element_text_chars=settings.browser_max_element_text_chars,
            max_attributes=settings.browser_max_attributes,
            max_attribute_chars=settings.browser_max_attribute_chars,
            max_fill_chars=settings.browser_max_fill_chars,
            max_screenshot_bytes=settings.browser_max_screenshot_bytes,
            max_sessions=settings.browser_max_sessions,
            max_pages_per_session=settings.browser_max_pages_per_session,
            max_navigation_time_s=settings.browser_max_navigation_time_s,
            max_action_time_s=settings.browser_max_action_time_s,
            max_wait_time_s=settings.browser_max_wait_time_s,
            max_retries=settings.browser_max_retries,
        )
