"""extract_document: normalized text + structure, bounded (Phase 4).

LOW permission. Returns the normalized sections (heading, type, ordering,
page/slide/sheet location) with text already limited by the configured
extraction caps; truncation is always reported via the ``truncated`` flag
and warnings.
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

EXTRACT_TOOL_NAME = "extract_document"


class ExtractDocumentTool:
    """Returns the normalized, bounded representation of a document."""

    def __init__(self, workspace: Workspace, limits: DocumentLimits) -> None:
        self._ws = workspace
        self._limits = limits

    spec = ToolSpec(
        name=EXTRACT_TOOL_NAME,
        description=(
            "Extracts the normalized text and structure of a document inside "
            "the workspace (workspace-relative path): sections in document "
            "order with headings, page/slide/sheet locations, warnings, and "
            "statistics. Extraction is bounded; truncation is reported."
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
                "title": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_id": {"type": "string"},
                            "section_type": {"type": "string"},
                            "heading": {"type": "string"},
                            "text": {"type": "string"},
                            "index": {"type": "integer"},
                        },
                        "required": ["section_id", "section_type", "text", "index"],
                    },
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
                "stats": {"type": "object"},
                "truncated": {"type": "boolean"},
            },
            "required": [
                "document_id",
                "source_path",
                "filename",
                "document_type",
                "sections",
                "warnings",
                "stats",
                "truncated",
            ],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            document, _size = load_document(self._ws, input.get("path"), self._limits)
            sections = []
            for section in document.sections:
                entry: dict[str, Any] = {
                    "section_id": section.section_id,
                    "section_type": section.section_type,
                    "text": section.text,
                    "index": section.index,
                }
                if section.heading is not None:
                    entry["heading"] = section.heading
                if section.location is not None:
                    entry["location"] = section.location
                sections.append(entry)
            output: dict[str, Any] = {
                "document_id": document.document_id,
                "source_path": document.source_path,
                "filename": document.filename,
                "document_type": document.document_type,
                "sections": sections,
                "warnings": list(document.warnings),
                "stats": dict(document.stats),
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
