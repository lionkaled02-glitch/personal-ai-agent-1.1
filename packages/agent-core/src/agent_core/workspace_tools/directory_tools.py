"""Directory tools: list_directory, create_directory.

Listings are sorted (deterministic order) and bounded by the configured
``max_list_entries``. Directory creation is nested-capable but stays inside
the workspace boundary; creating an already-existing directory is an
idempotent success.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ._common import as_os_error, fail, require_directory, require_str_field

LIST_TOOL_NAME = "list_directory"
CREATE_DIR_TOOL_NAME = "create_directory"


def _entry_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


class ListDirectoryTool:
    """Lists a workspace directory with bounded, sorted entries (LOW)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=LIST_TOOL_NAME,
        description=(
            "Lists the entries of a workspace directory (workspace-relative "
            "path; empty = workspace root) with name, type, and size. Results "
            "are sorted and limited in count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path (empty = workspace root).",
                },
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "type": {"enum": ["file", "directory", "symlink", "other"]},
                            "size_bytes": {"type": "integer"},
                        },
                        "required": ["name", "path", "type", "size_bytes"],
                    },
                },
                "count": {"type": "integer"},
                "truncated": {"type": "boolean"},
            },
            "required": ["path", "entries", "count", "truncated"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            path_value = input.get("path")
            if path_value is not None and not isinstance(path_value, str):
                raise WorkspaceError("invalid_path", "path must be a string")
            resolved, rel = require_directory(self._ws, path_value or "")
            limit = self._ws.limits.max_list_entries
            with os.scandir(resolved) as it:
                children = sorted(it, key=lambda e: e.name)
            entries: list[dict[str, Any]] = []
            for child in children:
                if len(entries) >= limit:
                    break
                child_path = Path(os.path.normpath(child.path))
                size = 0
                try:
                    if not child_path.is_symlink() and child_path.is_file():
                        size = child_path.stat().st_size
                except OSError:
                    size = 0
                entries.append(
                    {
                        "name": child.name,
                        "path": self._ws.relative_to_root(child_path),
                        "type": _entry_type(child_path),
                        "size_bytes": size,
                    }
                )
            return ToolResult(
                ok=True,
                output={
                    "path": rel,
                    "entries": entries,
                    "count": len(entries),
                    "truncated": len(children) > limit,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)


class CreateDirectoryTool:
    """Creates (nested) directories inside the workspace (MEDIUM permission).

    Idempotent: creating an existing directory succeeds with created=false.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=CREATE_DIR_TOOL_NAME,
        description=(
            "Creates a directory (nested paths allowed) inside the workspace. "
            "If the directory already exists, this is a no-op success."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path to create.",
                },
                "parents": {
                    "type": "boolean",
                    "description": "Create missing parent directories (default true).",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "created": {"type": "boolean"},
            },
            "required": ["path", "created"],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            parents = bool(input.get("parents", True))
            resolved = self._ws.resolve(require_str_field(input, "path"))
            rel = self._ws.relative_to_root(resolved)
            if resolved == self._ws.root:
                raise WorkspaceError(
                    "unsupported_operation", "cannot (re)create the workspace root"
                )
            if resolved.exists():
                if not resolved.is_dir():
                    raise WorkspaceError("not_a_directory", f"path is not a directory: {rel}")
                return ToolResult(ok=True, output={"path": rel, "created": False})
            try:
                resolved.mkdir(parents=parents)
            except FileNotFoundError as exc:
                raise WorkspaceError(
                    "path_not_found", "parent directory does not exist (parents=false)"
                ) from exc
            return ToolResult(ok=True, output={"path": rel, "created": True})
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
