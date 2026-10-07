"""Conservative secret-content guard (Phase 5).

A *heuristic*, not a guarantee: it rejects memory content that obviously
looks like credentials (key/value assignment patterns for well-known
secret names, common token prefixes, private key blocks, credentials in
URLs). Arbitrary secrets cannot be detected reliably by any heuristic —
the real guarantees are structural:

- memory creation is explicit (permission-gated tool or a clearly defined
  trusted internal pathway); conversation text is never auto-persisted;
- memory content is data only (never executed, never instructions).

See SECURITY.md.
"""

from __future__ import annotations

import re

#: Conservative patterns for obvious credential material.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # name/assignment style: api_key=..., password: ..., token = ...
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|secret|secret[_-]?key|access[_-]?token|"
        r"auth[_-]?token|password|passwd|client[_-]?secret|private[_-]?key)\b"
        r"\s*[:=]\s*\S+"
    ),
    # Authorization header / bearer tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-_.~+/]{16,}"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*\S+"),
    # Provider token shapes (OpenAI/Anthropic-style, AWS keys, GitHub tokens)
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9]{20,}\b"),
    # Private key blocks
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # Credentials embedded in URLs: scheme://user:pass@host
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@]+:[^\s/@]+@[^\s]+"),
)


def contains_secret_like_content(text: str) -> bool:
    """True when ``text`` matches an obvious credential pattern.

    Deliberately conservative (favors false negatives over false
    positives): legitimate prose mentioning secret *concepts* (without an
    assignment or a real token shape) passes. This is a heuristic and
    must never be presented as a complete secret-detection guarantee.
    """
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)
