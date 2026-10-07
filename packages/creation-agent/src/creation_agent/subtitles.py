from __future__ import annotations


def _timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_srt(text: str, duration_s: float, max_chars: int = 84) -> str:
    """Create deterministic, bounded SRT cues from narration text."""
    clean = " ".join(text.split())
    if not clean:
        return ""
    max_chars = max(20, min(max_chars, 200))
    words = clean.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            length = 0
        current.append(word)
        length += len(word) + (1 if length else 0)
    if current:
        chunks.append(" ".join(current))
    duration = max(0.1, float(duration_s))
    step = duration / len(chunks)
    cues: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        start = i - 1
        cues.append(f"{i}\n{_timestamp(start * step)} --> {_timestamp(i * step)}\n{chunk}\n")
    return "\n".join(cues)
