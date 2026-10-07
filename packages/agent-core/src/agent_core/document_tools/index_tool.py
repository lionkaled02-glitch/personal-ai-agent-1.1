"""index_document: parse + chunk + index into the knowledge store (Phase 4).

MEDIUM permission — indexing is an internal knowledge mutation (analogous
to workspace file mutations in Phase 3). A denied operation never mutates
the store (the permission system runs before the tool body).
"""

from __future__ import annotations

from typing import Any

from ..documents.chunking import chunk_document
from ..documents.errors import INVALID_INPUT, DocumentError
from ..documents.limits import DocumentLimits
from ..documents.retrieval import KnowledgeStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ..workspace_tools._common import as_os_error, fail
from ._common import load_document

INDEX_TOOL_NAME = "index_document"


class IndexDocumentTool:
    """Parses, chunks, and indexes a document (replacing prior chunks)."""

    def __init__(self, workspace: Workspace, store: KnowledgeStore) -> None:
        self._ws = workspace
        self._store = store

    spec = ToolSpec(
        name=INDEX_TOOL_NAME,
        description=(
            "Parses a document inside the workspace (workspace-relative "
            "path), splits it into bounded chunks, and indexes it for "
            "search_documents. Re-indexing a path replaces its chunks. "
            "Optional chunk_size / chunk_overlap override the defaults."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative document path.",
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Optional maximum chunk size in characters.",
                },
                "chunk_overlap": {
                    "type": "integer",
                    "description": "Optional overlap in characters between chunks.",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "source_path": {"type": "string"},
                "document_type": {"type": "string"},
                "sections": {"type": "integer"},
                "chunks": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "skipped_chunks": {"type": "integer"},
            },
            "required": [
                "document_id",
                "source_path",
                "document_type",
                "sections",
                "chunks",
                "truncated",
                "skipped_chunks",
            ],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
    )

    def _effective_limits(self, input: dict[str, Any]) -> DocumentLimits:
        """Apply validated optional chunking overrides to the store limits."""
        base = self._store.limits
        chunk_size = input.get("chunk_size")
        chunk_overlap = input.get("chunk_overlap")
        if chunk_size is not None and (
            not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1
        ):
            raise DocumentError(INVALID_INPUT, "chunk_size must be a positive integer")
        if chunk_overlap is not None and (
            not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_overlap < 0
        ):
            raise DocumentError(INVALID_INPUT, "chunk_overlap must be a non-negative integer")
        size = chunk_size if chunk_size is not None else base.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else base.chunk_overlap
        if overlap >= size:
            raise DocumentError(INVALID_INPUT, "chunk_overlap must be smaller than chunk_size")
        return DocumentLimits(
            max_input_bytes=base.max_input_bytes,
            max_extracted_chars=base.max_extracted_chars,
            max_pages=base.max_pages,
            max_slides=base.max_slides,
            max_sheets=base.max_sheets,
            max_sections=base.max_sections,
            max_chunks=base.max_chunks,
            chunk_size=size,
            chunk_overlap=overlap,
            max_search_results=base.max_search_results,
            max_query_chars=base.max_query_chars,
        )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            limits = self._effective_limits(input)
            document, _size = load_document(self._ws, input.get("path"), limits)
            result = chunk_document(document, limits)
            self._store.add_document(document, list(result.chunks))
            return ToolResult(
                ok=True,
                output={
                    "document_id": document.document_id,
                    "source_path": document.source_path,
                    "document_type": document.document_type,
                    "sections": len(document.sections),
                    "chunks": len(result.chunks),
                    "truncated": document.truncated or result.truncated,
                    "skipped_chunks": result.skipped_chunks,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except DocumentError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
