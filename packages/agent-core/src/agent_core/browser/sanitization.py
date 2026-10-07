"""Deterministic redaction for visible browser text and safe attributes."""

from __future__ import annotations

import re

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passcode|passwd|authorization|cookie|set-cookie|api[_ -]?key|"
    r"access[_ -]?token|refresh[_ -]?token|session[_ -]?(?:id|token|key)?|"
    r"auth(?:[_ -]?(?:token|key))?|oauth|csrf|secret|credential|client[_ -]?secret|"
    r"cvv|cvc|security[_ -]?code|pin|cc[_ -]?(?:number|name|exp|csc|type)|"
    r"credit[_ -]?card(?:[_ -]?(?:number|no))?|card[_ -]?(?:number|no)|"
    r"payment[_ -]?(?:token|data)|iban|routing[_ -]?(?:number|no)|"
    r"account[_ -]?(?:number|no)|email|e-mail|phone|telephone|tel|address|postal[_ -]?code|"
    r"ssn|social[_ -]?security|passport|tax[_ -]?id|bday|government[_ -]?id|driver[_ -]?license|"
    r"username|user[_ -]?name|login|pin)\b\s*[:=]\s*([^\s,;<>]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_COMMON_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def redact_sensitive_text(text: str, *, max_chars: int) -> str:
    """Redact recognizable credentials/cards, then return bounded text.

    This is a defense-in-depth content filter, not a credential detector.
    Sensitive form-control values are never read by the browser provider.
    """
    redacted = _BEARER_TOKEN.sub("Bearer [redacted]", text)
    redacted = _COMMON_TOKEN.sub("[redacted token]", redacted)
    redacted = _SENSITIVE_ASSIGNMENT.sub(r"\1=[redacted]", redacted)
    redacted = _CARD_CANDIDATE.sub(_redact_card_if_valid, redacted)
    return redacted[:max_chars]


def _redact_card_if_valid(match: re.Match[str]) -> str:
    digits = "".join(character for character in match.group(0) if character.isdigit())
    if not 13 <= len(digits) <= 19:
        return match.group(0)
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return "[redacted card]" if total % 10 == 0 else match.group(0)
