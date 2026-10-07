"""Transfer tools: copy_file, move_file, delete_file.

All three operate strictly inside the workspace boundary. Copy/move read
and write bounded byte streams (never shell out, never follow symlinks out
of the workspace); delete is files-only with no recursive mode. Delete is
HIGH permission and is gated by the approval mechanism before any
filesystem action occurs.
"""

from __future__ import annotations

import os
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ._common import as_os_error, atomic_write_bytes, fail, require_source, require_str_field

COPY_TOOL_NAME = "copy_file"
MOVE_TOOL_NAME = "move_file"
DELETE_TOOL_NAME = "delete_file"


class CopyFileTool:
    """Copies a file to a new location inside the workspace (MEDIUM)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=COPY_TOOL_NAME,
        description=(
            "Copies a file to a new location inside the workspace "
            "(workspace-relative paths). The destination file must not exist "
            "unless overwrite=true. Destination directory must exist."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Workspace-relative source file."},
                "destination": {
                    "type": "string",
                    "description": "Workspace-relative destination file path.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite the destination if it exists (default false).",
                },
            },
            "required": ["source", "destination"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "bytes_copied": {"type": "integer"},
                "overwritten": {"type": "boolean"},
            },
            "required": ["source", "destination", "bytes_copied", "overwritten"],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            source, src_rel = require_source(self._ws, require_str_field(input, "source"))
            dest = self._ws.resolve(require_str_field(input, "destination"))
            dest_rel = self._ws.relative_to_root(dest)
            overwrite = bool(input.get("overwrite", False))
            if source == dest:
                raise WorkspaceError("invalid_path", "source and destination are identical")
            if source.stat().st_size > self._ws.limits.max_read_bytes:
                raise WorkspaceError(
                    "file_too_large",
                    f"source is {source.stat().st_size} bytes; limit is "
                    f"{self._ws.limits.max_read_bytes}",
                )
            existed_before = dest.exists()
            if existed_before:
                if not overwrite:
                    raise WorkspaceError(
                        "target_exists",
                        f"destination already exists and overwrite is false: {dest_rel}",
                    )
                if not dest.is_file():
                    raise WorkspaceError(
                        "not_a_file", f"destination is not a regular file: {dest_rel}"
                    )
            else:
                if not dest.parent.is_dir():
                    raise WorkspaceError(
                        "path_not_found", f"destination directory does not exist: {dest_rel}"
                    )
            data = source.read_bytes()
            atomic_write_bytes(dest, data)
            return ToolResult(
                ok=True,
                output={
                    "source": src_rel,
                    "destination": dest_rel,
                    "bytes_copied": len(data),
                    "overwritten": existed_before,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)


class MoveFileTool:
    """Moves a file to a new location inside the workspace (MEDIUM)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=MOVE_TOOL_NAME,
        description=(
            "Moves (renames) a file to a new location inside the workspace "
            "(workspace-relative paths). The destination file must not exist "
            "unless overwrite=true."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Workspace-relative source file."},
                "destination": {
                    "type": "string",
                    "description": "Workspace-relative destination file path.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite the destination if it exists (default false).",
                },
            },
            "required": ["source", "destination"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "bytes_moved": {"type": "integer"},
                "overwritten": {"type": "boolean"},
            },
            "required": ["source", "destination", "bytes_moved", "overwritten"],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            source, src_rel = require_source(self._ws, require_str_field(input, "source"))
            dest = self._ws.resolve(require_str_field(input, "destination"))
            dest_rel = self._ws.relative_to_root(dest)
            overwrite = bool(input.get("overwrite", False))
            if source == dest:
                raise WorkspaceError("invalid_path", "source and destination are identical")
            existed_before = dest.exists()
            if existed_before:
                if not overwrite:
                    raise WorkspaceError(
                        "target_exists",
                        f"destination already exists and overwrite is false: {dest_rel}",
                    )
                if not dest.is_file():
                    raise WorkspaceError(
                        "not_a_file", f"destination is not a regular file: {dest_rel}"
                    )
            elif not dest.parent.is_dir():
                raise WorkspaceError(
                    "path_not_found", f"destination directory does not exist: {dest_rel}"
                )
            size = source.stat().st_size
            os.replace(source, dest)
            return ToolResult(
                ok=True,
                output={
                    "source": src_rel,
                    "destination": dest_rel,
                    "bytes_moved": size,
                    "overwritten": existed_before,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)


class DeleteFileTool:
    """Deletes a single file inside the workspace (HIGH permission).

    Files only — directories are never deleted (recursive or not). Requires
    explicit approval through the existing mechanism; the runtime rejects
    unapproved calls before this class is ever invoked.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=DELETE_TOOL_NAME,
        description=(
            "Deletes a single file inside the workspace (workspace-relative "
            "path). Directories are never deleted. Requires explicit approval."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to delete.",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "bytes_deleted": {"type": "integer"},
            },
            "required": ["path", "bytes_deleted"],
        },
        permission_level=PermissionLevel.HIGH,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            resolved = self._ws.resolve(require_str_field(input, "path"))
            rel = self._ws.relative_to_root(resolved)
            if resolved == self._ws.root:
                raise WorkspaceError(
                    "unsupported_operation", "the workspace root cannot be deleted"
                )
            if not resolved.exists():
                raise WorkspaceError("path_not_found", f"path does not exist: {rel}")
            if resolved.is_dir():
                raise WorkspaceError(
                    "unsupported_operation",
                    "directories cannot be deleted (no recursive deletion)",
                )
            if not resolved.is_file():
                raise WorkspaceError("not_a_file", f"path is not a regular file: {rel}")
            size = resolved.stat().st_size
            resolved.unlink()
            return ToolResult(ok=True, output={"path": rel, "bytes_deleted": size})
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
