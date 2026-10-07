"""recall: retrieve relevant memories (Phase 5). LOW permission.

Deterministic lexical matching with optional type filter and bounded
result count. Results carry stable ids and full provenance. Expired and
soft-forgotten memories are never returned.
"""

from __future__ import annotations

from typing import Any

from ..events import Clock
from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.limits import MemoryLimits
from ..memory.retrieval import MemoryRetriever
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import fail
from ._common import memory_view

RECALL_TOOL_NAME = "recall"


class RecallTool:
    """Bounded, provenance-labeled lexical recall over active memories."""

    def __init__(self, retriever: MemoryRetriever, limits: MemoryLimits, clock: Clock) -> None:
        self._retriever = retriever
        self._limits = limits
        self._clock = clock

    spec = ToolSpec(
        name=RECALL_TOOL_NAME,
        description=(
            "Recalls memories relevant to a lexical query from active, "
            "non-expired memories. Optional memory_type filter and limit. "
            "Results are deterministically ranked and include provenance "
            "(source category + optional source_ref) and stable ids."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "memory_type": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "count": {"type": "integer"},
            },
            "required": ["query", "results", "count"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,  # expiration is evaluated against the clock
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            query = input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "query must be a non-empty string")
            if len(query) > self._limits.max_content_chars:
                raise MemoryStoreError(
                    MEMORY_INVALID_INPUT,
                    f"query is too long (max {self._limits.max_content_chars} characters)",
                )
            memory_type = input.get("memory_type")
            if memory_type is not None and not isinstance(memory_type, str):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "memory_type must be a string")
            limit = input.get("limit")
            if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            ):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "limit must be a positive integer")
            memories = self._retriever.recall(
                query,
                memory_type=memory_type,
                limit=limit,
                now=self._clock(),
            )
            results = [memory_view(m) for m in memories]
            return ToolResult(
                ok=True,
                output={"query": query, "results": results, "count": len(results)},
            )
        except MemoryStoreError as exc:
            return fail(exc.code, str(exc))
