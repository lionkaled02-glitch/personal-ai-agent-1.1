"""forget: deactivate (default) or delete a memory (Phase 5). HIGH.

Safe by default: ``forget`` soft-deactivates (the memory is hidden from
recall and active listings but remains auditable); ``hard: true`` deletes
it. Either way exactly ONE memory is affected — never recursive, never
batch. HIGH permission means mandatory approval; a denial performs no
mutation.
"""

from __future__ import annotations

from typing import Any

from ..events import Clock
from ..memory.errors import MEMORY_INVALID_INPUT, MemoryStoreError
from ..memory.store import MemoryStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import fail

FORGET_TOOL_NAME = "forget"


class ForgetTool:
    """Soft-deactivates (default) or hard-deletes a single memory."""

    def __init__(self, store: MemoryStore, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    spec = ToolSpec(
        name=FORGET_TOOL_NAME,
        description=(
            "Forgets ONE memory by id. Default is safe soft-deactivation "
            "(hidden from recall, kept for audit); hard: true deletes it. "
            "Requires explicit approval (HIGH)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "hard": {"type": "boolean"},
            },
            "required": ["memory_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "hard": {"type": "boolean"},
                "active": {"type": "boolean"},
                "deleted": {"type": "boolean"},
            },
            "required": ["memory_id", "hard", "active", "deleted"],
        },
        permission_level=PermissionLevel.HIGH,
        deterministic=False,  # reads the clock for updated_at
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            memory_id = input.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id.strip():
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "memory_id must be a non-empty string")
            hard = input.get("hard", False)
            if not isinstance(hard, bool):
                raise MemoryStoreError(MEMORY_INVALID_INPUT, "hard must be a boolean")
            memory = self._store.forget(memory_id, hard=hard, now=self._clock())
            return ToolResult(
                ok=True,
                output={
                    "memory_id": memory.memory_id,
                    "hard": hard,
                    "active": memory.active and not hard,
                    "deleted": hard,
                },
            )
        except MemoryStoreError as exc:
            return fail(exc.code, str(exc))
