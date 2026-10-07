"""Deterministic Unicode and whitespace normalization for voice text."""

from __future__ import annotations

import unicodedata

from .errors import VoiceLimitError, VoiceValidationError
from .models import MAX_VOICE_TEXT_CHARS

_BIDI_OVERRIDE_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))


def normalize_transcription_text(
    text: str,
    *,
    max_chars: int = MAX_VOICE_TEXT_CHARS,
    allow_empty: bool = False,
) -> str:
    """NFC-normalize Unicode and collapse whitespace without rewriting words.

    Canonically equivalent code points and whitespace are normalized only;
    punctuation and lexical content are preserved. Invalid Unicode scalar
    values, non-whitespace controls, and bidirectional override controls are
    rejected rather than silently deleted. The raw and normalized text are
    both checked against the caller's configured bound.
    """
    if not isinstance(text, str):
        raise VoiceValidationError("invalid_transcription")
    if max_chars < 1 or max_chars > MAX_VOICE_TEXT_CHARS:
        raise ValueError("max_chars must be within the hard voice text bound")
    if len(text) > max_chars:
        raise VoiceLimitError("transcription_limit_exceeded")

    normalized = unicodedata.normalize("NFC", text)
    for character in normalized:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if category == "Cs" or (category == "Cc" and not character.isspace()):
            raise VoiceValidationError("invalid_transcription")
        if any(start <= codepoint <= end for start, end in _BIDI_OVERRIDE_RANGES):
            raise VoiceValidationError("invalid_transcription")

    normalized = " ".join(normalized.split())
    if len(normalized) > max_chars:
        raise VoiceLimitError("transcription_limit_exceeded")
    if not normalized and not allow_empty:
        raise VoiceValidationError("empty_transcription")
    return normalized
