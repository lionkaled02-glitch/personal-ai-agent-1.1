"""Memory tools (Phase 5).

Five provider-agnostic tools over the memory layer, registered against an
explicit :class:`~agent_core.memory.store.MemoryStore` (and the derived
retriever). There is no implicit memory state anywhere else:

- LOW:    recall, list_memories
- MEDIUM: remember, update_memory (explicit creation/mutation — approval)
- HIGH:   forget (deactivation/deletion — approval)

Registration is explicit; the store, limits, and clock are injected so
the tools stay deterministic and testable.
"""

from __future__ import annotations

from ..events import Clock, utc_now
from ..memory.limits import MemoryLimits
from ..memory.retrieval import LexicalMemoryRetriever
from ..memory.store import MemoryStore
from ..tools import ToolRegistry
from .forget_tool import FORGET_TOOL_NAME, ForgetTool
from .list_tool import LIST_MEMORIES_TOOL_NAME, ListMemoriesTool
from .recall_tool import RECALL_TOOL_NAME, RecallTool
from .remember_tool import REMEMBER_TOOL_NAME, RememberTool
from .update_tool import UPDATE_MEMORY_TOOL_NAME, UpdateMemoryTool

__all__ = [
    "FORGET_TOOL_NAME",
    "LIST_MEMORIES_TOOL_NAME",
    "MEMORY_TOOL_NAMES",
    "RECALL_TOOL_NAME",
    "REMEMBER_TOOL_NAME",
    "UPDATE_MEMORY_TOOL_NAME",
    "ForgetTool",
    "ListMemoriesTool",
    "RecallTool",
    "RememberTool",
    "UpdateMemoryTool",
    "register_memory_tools",
]

MEMORY_TOOL_NAMES: tuple[str, ...] = (
    FORGET_TOOL_NAME,
    LIST_MEMORIES_TOOL_NAME,
    RECALL_TOOL_NAME,
    REMEMBER_TOOL_NAME,
    UPDATE_MEMORY_TOOL_NAME,
)


def register_memory_tools(
    registry: ToolRegistry,
    store: MemoryStore,
    limits: MemoryLimits | None = None,
    clock: Clock | None = None,
) -> None:
    """Registers all five memory tools against the given store + limits."""
    if limits is None:
        limits = store.limits
    ck = clock or utc_now
    retriever = LexicalMemoryRetriever(store)
    registry.register(RememberTool(store, limits, ck))
    registry.register(RecallTool(retriever, limits, ck))
    registry.register(UpdateMemoryTool(store, limits, ck))
    registry.register(ForgetTool(store, ck))
    registry.register(ListMemoriesTool(store, limits, ck))
