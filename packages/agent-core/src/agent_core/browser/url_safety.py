"""Strict URL validation and credential-safe browser URL projections."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .errors import BrowserValidationError

HARD_MAX_URL_CHARS = 4_096
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_INVALID_DNS_LABEL = re.compile(r"[^A-Za-z0-9-]")


def validate_http_url(url: str, *, max_chars: int = HARD_MAX_URL_CHARS) -> str:
    """Accept only bounded, syntactically valid HTTP(S) URLs without userinfo.

    This rejects filesystem paths and browser-special schemes before a
    provider sees them. No URL is fetched or resolved by this helper.
    """
    if not isinstance(url, str) or not url:
        raise BrowserValidationError("invalid_url")
    if max_chars < 1 or max_chars > HARD_MAX_URL_CHARS:
        raise ValueError("max_chars must be within the hard URL bound")
    if len(url) > min(max_chars, HARD_MAX_URL_CHARS):
        raise BrowserValidationError("url_limit_exceeded")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in url
    ):
        raise BrowserValidationError("invalid_url")
    if "\\" in url or _INVALID_PERCENT_ESCAPE.search(url):
        raise BrowserValidationError("invalid_url")

    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise BrowserValidationError("unsupported_scheme")
        hostname = parts.hostname
        if not parts.netloc or hostname is None or not hostname:
            raise BrowserValidationError("invalid_url")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise BrowserValidationError("invalid_url")
        if parts.netloc.endswith(":"):
            raise BrowserValidationError("invalid_url")
        port = parts.port
    except BrowserValidationError:
        raise
    except ValueError:
        raise BrowserValidationError("invalid_url") from None

    if port is not None and not 1 <= port <= 65_535:
        raise BrowserValidationError("invalid_url")
    _validate_hostname(hostname)
    return url


def hostname_from_url(url: str) -> str:
    """Return the canonical lower-case hostname after HTTP(S) validation."""
    validated = validate_http_url(url)
    hostname = urlsplit(validated).hostname
    if hostname is None:  # Guarded by validation; keeps the return type explicit.
        raise BrowserValidationError("invalid_url")
    try:
        return hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError:
        raise BrowserValidationError("invalid_url") from None


def safe_observed_url(url: str | None, *, max_chars: int = HARD_MAX_URL_CHARS) -> str | None:
    """Return an HTTP(S) URL without query/fragment values that can hold tokens."""
    if url is None or url == "about:blank":
        return None
    validated = validate_http_url(url, max_chars=max_chars)
    parts = urlsplit(validated)
    safe_parts = SplitResult(parts.scheme.lower(), parts.netloc, parts.path, "", "")
    safe_url = urlunsplit(safe_parts)
    if len(safe_url) > max_chars:
        raise BrowserValidationError("url_limit_exceeded")
    return safe_url


def _validate_hostname(hostname: str) -> None:
    if "%" in hostname:
        raise BrowserValidationError("invalid_url")
    try:
        ipaddress.ip_address(hostname)
        return
    except ValueError:
        pass
    try:
        ascii_host = hostname.encode("idna").decode("ascii").rstrip(".")
    except UnicodeError:
        raise BrowserValidationError("invalid_url") from None
    if not ascii_host or len(ascii_host) > 253:
        raise BrowserValidationError("invalid_url")
    labels = ascii_host.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or _INVALID_DNS_LABEL.search(label) is not None
        for label in labels
    ):
        raise BrowserValidationError("invalid_url")
