"""Document tools (Phase 4).

Four provider-agnostic tools bridging the workspace boundary (Phase 3),
the document layer (parsing/chunking), and the knowledge store
(lexical retrieval):

- LOW:    inspect_document, extract_document, search_documents
- MEDIUM: index_document (internal knowledge mutation)

Registration requires an explicit ``Workspace`` and ``KnowledgeStore``;
there is no implicit filesystem or index state.
"""

from __future__ import annotations

from ..documents.retrieval import KnowledgeStore
from ..tools import ToolRegistry
from ..workspace import Workspace
from .extract_tool import EXTRACT_TOOL_NAME, ExtractDocumentTool
from .index_tool import INDEX_TOOL_NAME, IndexDocumentTool
from .inspect_tool import INSPECT_TOOL_NAME, InspectDocumentTool
from .search_tool import SEARCH_TOOL_NAME, SearchDocumentsTool

__all__ = [
    "DOCUMENT_TOOL_NAMES",
    "EXTRACT_TOOL_NAME",
    "INDEX_TOOL_NAME",
    "INSPECT_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "ExtractDocumentTool",
    "IndexDocumentTool",
    "InspectDocumentTool",
    "SearchDocumentsTool",
    "register_document_tools",
]

DOCUMENT_TOOL_NAMES: tuple[str, ...] = (
    EXTRACT_TOOL_NAME,
    INDEX_TOOL_NAME,
    INSPECT_TOOL_NAME,
    SEARCH_TOOL_NAME,
)


def register_document_tools(
    registry: ToolRegistry, workspace: Workspace, store: KnowledgeStore
) -> None:
    """Registers all four document tools against the given workspace + store."""
    limits = store.limits
    registry.register(InspectDocumentTool(workspace, limits))
    registry.register(ExtractDocumentTool(workspace, limits))
    registry.register(IndexDocumentTool(workspace, store))
    registry.register(SearchDocumentsTool(store))
