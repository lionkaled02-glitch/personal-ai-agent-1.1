"""update_memory: modify an existing memory (Phase 5). MEDIUM permission.

Mutable fields only: content, metadata (replace), confidence, expires_at,
active. Identity fields (memory_id, memory_type, source, source_ref,
created_at) are immutable — present in the input at all, they are a
structured ``memory_field_immutable`` failure (never silently changed,
never silently ignored).
"""

from __future__ import annotations

from typing import Any

from ..events import Clock
from ..memory.errors import MEMORY_FIELD_IMMUTABLE, MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.limits import MemoryLimits
from ..memory.store import MemoryStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import fail
from ._common import iso, parse_expires_at

UPDATE_MEMORY_TOOL_NAME = "update_memory"

#: Identity fields that cannot be changed (memory_id is the lookup key —
#: the API has no field to change it, by design).
_IMMUTABLE_FIELDS = ("memory_type", "source", "source_ref", "created_at")


class UpdateMemoryTool:
    """Field-wise update of mutable memory fields."""

    def __init__(self, store: MemoryStore, limits: MemoryLimits, clock: Clock) -> None:
        self._store = store
        self._limits = limits
        self._clock = clock

    spec = ToolSpec(
        name=UPDATE_MEMORY_TOOL_NAME,
        description=(
            "Updates an existing memory by id. Mutable fields: content, "
            "metadata (replaces the whole mapping), confidence, expires_at "
            "(ISO 8601), active. Identity fields (memory_type, source, "
            "source_ref, created_at) are immutable and rejected if present."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string"},
                "metadata": {"type": "object"},
                "confidence": {"type": "number"},
                "expires_at": {"type": "string"},
                "active": {"type": "boolean"},
            },
            "required": ["memory_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "memory_type": {"type": "string"},
                "source": {"type": "string"},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
                "expires_at": {"type": "string"},
                "active": {"type": "boolean"},
            },
            "required": [
                "memory_id",
                "memory_type",
                "source",
                "created_at",
                "updated_at",
                "active",
            ],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,  # reads the clock for updated_at
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            for field_name in _IMMUTABLE_FIELDS:
                if field_name in input:
                    raise MemoryStoreError(
                        MEMORY_FIELD_IMMUTABLE,
                        f"{field_name} is immutable and cannot be updated",
                    )
            memory_id = input.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id.strip():
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "memory_id must be a non-empty string")
            metadata = input.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "metadata must be an object")
            confidence = input.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise MemoryStoreError(
                    MEMORY_INVALID_INPUT, "confidence must be a number in [0, 1]"
                )
            memory = self._store.update(
                memory_id,
                content=input.get("content"),
                metadata=metadata,
                confidence=float(confidence) if confidence is not None else None,
                expires_at=parse_expires_at(input.get("expires_at")),
                active=input.get("active"),
                now=self._clock(),
            )
            output = {
                "memory_id": memory.memory_id,
                "memory_type": str(memory.memory_type),
                "source": str(memory.source),
                "created_at": iso(memory.created_at),
                "updated_at": iso(memory.updated_at),
                "expires_at": iso(memory.expires_at),
                "active": memory.active,
            }
            return ToolResult(ok=True, output=output)
        except MemoryStoreError as exc:
            return fail(exc.code, str(exc))
