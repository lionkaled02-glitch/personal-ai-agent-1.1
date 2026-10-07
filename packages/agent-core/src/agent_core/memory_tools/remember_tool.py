"""remember: create a memory (Phase 5). MEDIUM permission.

Explicit, permission-gated creation — the only supported pathway for new
memories (plus clearly defined trusted internal code paths). Never
automatic: no conversation text is ever captured implicitly.
"""

from __future__ import annotations

from typing import Any

from ..events import Clock
from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.limits import MemoryLimits
from ..memory.models import MemoryType, SourceCategory
from ..memory.store import MemoryStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import fail
from ._common import memory_meta_view, parse_expires_at

REMEMBER_TOOL_NAME = "remember"

_TYPE_ENUM = [t.value for t in MemoryType]
_SOURCE_ENUM = [s.value for s in SourceCategory]


class RememberTool:
    """Creates one memory after validation and limit enforcement."""

    def __init__(self, store: MemoryStore, limits: MemoryLimits, clock: Clock) -> None:
        self._store = store
        self._limits = limits
        self._clock = clock

    spec = ToolSpec(
        name=REMEMBER_TOOL_NAME,
        description=(
            "Stores an explicit memory about the user, task, project, or a "
            "prior decision. Requires approval (MEDIUM). Input: memory_type "
            f"({'/'.join(_TYPE_ENUM)}), content, source "
            f"({'/'.join(_SOURCE_ENUM)}), and optional source_ref, metadata, "
            "confidence, expires_at (ISO 8601). Short-term and working "
            "memories expire automatically after their configured TTL; "
            "long-term/knowledge do not. Content that looks like "
            "credential material is rejected."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_type": {"type": "string", "enum": _TYPE_ENUM},
                "content": {"type": "string"},
                "source": {"type": "string", "enum": _SOURCE_ENUM},
                "source_ref": {"type": "string"},
                "metadata": {"type": "object"},
                "confidence": {"type": "number"},
                "expires_at": {"type": "string"},
            },
            "required": ["memory_type", "content", "source"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "memory_type": {"type": "string"},
                "source": {"type": "string"},
                "source_ref": {"type": "string"},
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
        deterministic=False,  # reads the clock for created_at/TTL
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            memory_type = input.get("memory_type")
            if not isinstance(memory_type, str):
                raise MemoryStoreError(
                    MEMORY_INVALID_INPUT,
                    "memory_type must be one of: " + ", ".join(_TYPE_ENUM),
                )
            content = input.get("content")
            if not isinstance(content, str):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "content must be a string")
            source = input.get("source")
            if not isinstance(source, str):
                raise MemoryStoreError(
                    MEMORY_INVALID_INPUT,
                    "source must be one of: " + ", ".join(_SOURCE_ENUM),
                )
            metadata = input.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise MemoryStoreError("memory_invalid_input", "metadata must be an object")
            confidence = input.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise MemoryStoreError(
                    "memory_invalid_input", "confidence must be a number in [0, 1]"
                )
            source_ref = input.get("source_ref")
            if source_ref is not None and not isinstance(source_ref, str):
                raise MemoryStoreError("memory_invalid_input", "source_ref must be a string")
            memory = self._store.remember(
                memory_type=memory_type,
                content=content,
                source=source,
                source_ref=source_ref,
                metadata=metadata,
                confidence=float(confidence) if confidence is not None else None,
                expires_at=parse_expires_at(input.get("expires_at")),
                now=self._clock(),
            )
            return ToolResult(ok=True, output=memory_meta_view(memory))
        except MemoryStoreError as exc:
            return fail(exc.code, str(exc))
