"""list_memories: bounded listing with filters (Phase 5). LOW permission.

Filters: memory type, active status (default: active-and-not-expired
only), result count (bounded by the configured recall cap). Output is the
stable public memory fields only — no storage implementation details.
"""

from __future__ import annotations

from typing import Any

from ..events import Clock
from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.limits import MemoryLimits
from ..memory.store import MemoryStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import fail
from ._common import memory_view

LIST_MEMORIES_TOOL_NAME = "list_memories"


class ListMemoriesTool:
    """Deterministic, bounded listing of memories."""

    def __init__(self, store: MemoryStore, limits: MemoryLimits, clock: Clock) -> None:
        self._store = store
        self._limits = limits
        self._clock = clock

    spec = ToolSpec(
        name=LIST_MEMORIES_TOOL_NAME,
        description=(
            "Lists memories in deterministic order with optional memory_type "
            "filter, active-only filter (default true — active and not "
            "expired), and a bounded result count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_type": {"type": "string"},
                "active_only": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array", "items": {"type": "object"}},
                "count": {"type": "integer"},
            },
            "required": ["results", "count"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,  # expiration is evaluated against the clock
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            memory_type = input.get("memory_type")
            if memory_type is not None and not isinstance(memory_type, str):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "memory_type must be a string")
            active_only = input.get("active_only", True)
            if not isinstance(active_only, bool):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "active_only must be a boolean")
            limit = input.get("limit")
            if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            ):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "limit must be a positive integer")
            memories = self._store.list(
                memory_type=memory_type,
                active_only=active_only,
                limit=limit,
                now=self._clock(),
            )
            results = [memory_view(m) for m in memories]
            return ToolResult(ok=True, output={"results": results, "count": len(results)})
        except MemoryStoreError as exc:
            return fail(exc.code, str(exc))
