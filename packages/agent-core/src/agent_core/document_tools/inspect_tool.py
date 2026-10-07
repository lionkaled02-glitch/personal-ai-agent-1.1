"""inspect_document: metadata + extraction stats, no full text (Phase 4).

LOW permission. Returns identity, type, title, size, section/char counts,
warnings, and the truncation flag — deliberately not the document text.
"""

from __future__ import annotations

from typing import Any

from ..documents.errors import DocumentError
from ..documents.limits import DocumentLimits
from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ..workspace_tools._common import as_os_error, fail
from ._common import load_document

INSPECT_TOOL_NAME = "inspect_document"


class InspectDocumentTool:
    """Reports document metadata and extraction stats without the text."""

    def __init__(self, workspace: Workspace, limits: DocumentLimits) -> None:
        self._ws = workspace
        self._limits = limits

    spec = ToolSpec(
        name=INSPECT_TOOL_NAME,
        description=(
            "Inspects a document inside the workspace (workspace-relative "
            "path) and returns its metadata: type, title, size, section and "
            "character counts, extraction warnings, and whether extraction "
            "was truncated. Does not return document text."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative document path.",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "source_path": {"type": "string"},
                "filename": {"type": "string"},
                "document_type": {"type": "string"},
                "media_type": {"type": "string"},
                "title": {"type": "string"},
                "size_bytes": {"type": "integer"},
                "sections": {"type": "integer"},
                "chars": {"type": "integer"},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "truncated": {"type": "boolean"},
            },
            "required": [
                "document_id",
                "source_path",
                "filename",
                "document_type",
                "media_type",
                "size_bytes",
                "sections",
                "chars",
                "warnings",
                "truncated",
            ],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            document, size = load_document(self._ws, input.get("path"), self._limits)
            output: dict[str, Any] = {
                "document_id": document.document_id,
                "source_path": document.source_path,
                "filename": document.filename,
                "document_type": document.document_type,
                "media_type": document.media_type,
                "size_bytes": size,
                "sections": len(document.sections),
                "chars": document.stats.get("chars", 0),
                "warnings": list(document.warnings),
                "truncated": document.truncated,
            }
            if document.title is not None:
                output["title"] = document.title
            return ToolResult(ok=True, output=output)
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except DocumentError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
