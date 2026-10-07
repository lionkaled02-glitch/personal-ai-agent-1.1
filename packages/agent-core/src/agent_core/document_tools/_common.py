"""Shared helpers for document tools (Phase 4).

Every document tool resolves its path through the Phase 3
:class:`~agent_core.workspace.Workspace` boundary first — the workspace
security guarantees (no absolute paths, no traversal, no symlink escape,
fail-closed) apply to all document operations. Workspace errors keep their
stable codes (``path_outside_workspace``, ``invalid_path``,
``security_violation``); document-domain failures use the Phase 4
:class:`~agent_core.documents.errors.DocumentError` codes.

Tool messages contain only workspace-relative paths — never host paths and
never document content.
"""

from __future__ import annotations

from typing import Any

from ..documents.errors import (
    DOCUMENT_NOT_FOUND,
    DOCUMENT_TOO_LARGE,
    INVALID_DOCUMENT,
    UNSUPPORTED_DOCUMENT_TYPE,
    DocumentError,
)
from ..documents.limits import DocumentLimits
from ..documents.models import Document, make_document_id
from ..documents.parsers import ParserRegistry, default_registry
from ..workspace import Workspace

_DEFAULT_REGISTRY: ParserRegistry | None = None


def get_default_registry() -> ParserRegistry:
    """Lazily build (and cache) the built-in parser registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = default_registry()
    return _DEFAULT_REGISTRY


def load_document(
    workspace: Workspace, raw_path: Any, limits: DocumentLimits
) -> tuple[Document, int]:
    """Resolve a workspace path, size-check, parse, and normalize.

    Returns ``(document, size_bytes)``. Raises :class:`WorkspaceError`
    (boundary violations keep their codes) or :class:`DocumentError`
    (document-domain failures).
    """
    if not isinstance(raw_path, str):
        raise DocumentError(INVALID_DOCUMENT, "path must be a string")
    resolved = workspace.resolve(raw_path)
    rel = workspace.relative_to_root(resolved)
    if not resolved.exists():
        raise DocumentError(DOCUMENT_NOT_FOUND, f"document not found: {rel}")
    if not resolved.is_file():
        raise DocumentError("not_a_file", f"path is not a file: {rel}")
    size = resolved.stat().st_size
    if size > limits.max_input_bytes:
        raise DocumentError(
            DOCUMENT_TOO_LARGE,
            f"document is {size} bytes; limit is {limits.max_input_bytes} bytes",
        )
    data = resolved.read_bytes()
    parser = get_default_registry().for_filename(resolved.name)
    if parser is None:
        suffix = resolved.suffix.lstrip(".").lower() or "<no extension>"
        raise DocumentError(UNSUPPORTED_DOCUMENT_TYPE, f"unsupported document type: .{suffix}")
    document = parser.parse(
        data,
        source_path=rel,
        filename=resolved.name,
        document_id=make_document_id(rel),
        limits=limits,
    )
    return document, size
