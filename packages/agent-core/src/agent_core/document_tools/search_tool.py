"""search_documents: grounded lexical search over the knowledge store.

LOW permission (read-only). Returns matched chunks with their score,
matched terms, source document metadata, and source location — grounded
results, bounded count, deterministic ranking.
"""

from __future__ import annotations

from typing import Any

from ..documents.errors import INVALID_QUERY, DocumentError
from ..documents.retrieval import KnowledgeStore
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace_tools._common import as_os_error, fail

SEARCH_TOOL_NAME = "search_documents"


class SearchDocumentsTool:
    """Deterministic lexical search with grounded, metadata-rich results."""

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    spec = ToolSpec(
        name=SEARCH_TOOL_NAME,
        description=(
            "Searches indexed documents (via index_document) with a lexical "
            "query. Returns ranked chunk matches with document/chunk "
            "metadata and source location. Results are bounded in count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "document_id": {
                    "type": "string",
                    "description": "Optional: restrict the search to one document.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional max number of results.",
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "document_id": {"type": "string"},
                            "section_id": {"type": "string"},
                            "index": {"type": "integer"},
                            "score": {"type": "number"},
                            "matched_terms": {"type": "array", "items": {"type": "string"}},
                            "source_path": {"type": "string"},
                            "document_type": {"type": "string"},
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": [
                            "chunk_id",
                            "document_id",
                            "index",
                            "score",
                            "matched_terms",
                            "text",
                        ],
                    },
                },
                "count": {"type": "integer"},
                "truncated": {"type": "boolean"},
            },
            "required": ["query", "results", "count", "truncated"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        query = input.get("query")
        if not isinstance(query, str) or not query.strip():
            return fail(INVALID_QUERY, "query must be a non-empty string")
        limits = self._store.limits
        if len(query) > limits.max_query_chars:
            return fail(
                INVALID_QUERY,
                f"query is too long (max {limits.max_query_chars} characters)",
            )
        limit_raw = input.get("limit")
        if limit_raw is not None and (
            not isinstance(limit_raw, int) or isinstance(limit_raw, bool) or limit_raw < 1
        ):
            return fail(INVALID_QUERY, "limit must be a positive integer")
        limit = min(
            limit_raw if limit_raw is not None else limits.max_search_results,
            limits.max_search_results,
        )
        document_id = input.get("document_id")
        if document_id is not None and not isinstance(document_id, str):
            return fail(INVALID_QUERY, "document_id must be a string")
        try:
            # Fetch one extra to detect truncation beyond the requested limit.
            hits = self._store.search(query, document_id=document_id, limit=limit + 1)
            truncated = len(hits) > limit
            hits = hits[:limit]
            results = []
            for hit in hits:
                entry: dict[str, Any] = {
                    "chunk_id": hit.chunk.chunk_id,
                    "document_id": hit.chunk.document_id,
                    "index": hit.chunk.index,
                    "score": hit.score,
                    "matched_terms": list(hit.matched_terms),
                    "source_path": hit.chunk.metadata.get("source_path", ""),
                    "document_type": hit.chunk.metadata.get("document_type", ""),
                    "text": hit.chunk.text,
                }
                if hit.chunk.section_id is not None:
                    entry["section_id"] = hit.chunk.section_id
                if hit.document_title is not None:
                    entry["title"] = hit.document_title
                if hit.chunk.location is not None:
                    entry["location"] = hit.chunk.location
                results.append(entry)
            return ToolResult(
                ok=True,
                output={
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "truncated": truncated,
                },
            )
        except DocumentError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:  # defensive: store is in-memory; keep the shape
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
