"""Memory errors (Phase 5).

Stable, machine-readable error codes for the memory domain. Tool layers map
them into structured :class:`~agent_core.tools.ToolResult` failures; the
agent process never crashes on a memory operation failure.

:class:`MemoryStoreError` is the error type for the whole memory domain
(store, retrieval, policies, tools) — named ``MemoryStoreError`` (not
``MemoryError``) deliberately, so it never shadows the Python built-in.
"""

from __future__ import annotations

from ..errors import AgentCoreError

# ---------------------------------------------------------------------------
# Stable memory error codes (Phase 5)
# ---------------------------------------------------------------------------

MEMORY_NOT_FOUND = "memory_not_found"
MEMORY_INVALID_INPUT = "memory_invalid_input"
MEMORY_TYPE_NOT_ALLOWED = "memory_type_not_allowed"
MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
MEMORY_FIELD_IMMUTABLE = "memory_field_immutable"
#: Conservative secret heuristic tripped (extension code — a heuristic, not a
#: guarantee; see SECURITY.md).
SECRET_LIKE_CONTENT = "secret_like_content"


class MemoryStoreError(AgentCoreError):
    """A structured memory-domain failure with a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
