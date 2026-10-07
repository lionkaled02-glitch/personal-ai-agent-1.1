"""File tools: read_text_file, write_text_file, file_info.

All paths are workspace-relative and resolved through the
:class:`~agent_core.workspace.Workspace` boundary. Reads and writes are
byte-based (no platform newline translation) so round-trips are
deterministic. Writes are atomic (temp file + rename) and only create or
overwrite per the explicit ``overwrite`` input.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ._common import (
    as_os_error,
    atomic_write_bytes,
    check_encoding,
    ensure_parent,
    fail,
    require_file,
    require_str_field,
)

READ_TOOL_NAME = "read_text_file"
WRITE_TOOL_NAME = "write_text_file"
INFO_TOOL_NAME = "file_info"

_DEFAULT_ENCODING = "utf-8"


def _validate_encoding(input: Mapping[str, Any]) -> str:
    value = input.get("encoding")
    if value is None:
        value = _DEFAULT_ENCODING
    if not isinstance(value, str):
        raise WorkspaceError("invalid_encoding", "encoding must be a string")
    return check_encoding(value)


class ReadTextFileTool:
    """Reads a bounded text file inside the workspace (LOW permission)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=READ_TOOL_NAME,
        description=(
            "Reads a text file from the workspace (workspace-relative path) and "
            "returns its content plus metadata. Files larger than the configured "
            f"limit are rejected; default encoding is {_DEFAULT_ENCODING}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path (no absolute paths, no ../).",
                },
                "encoding": {
                    "type": "string",
                    "description": f"Text encoding (default '{_DEFAULT_ENCODING}').",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string"},
                "size_bytes": {"type": "integer"},
            },
            "required": ["path", "content", "encoding", "size_bytes"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            encoding = _validate_encoding(input)
            resolved, rel = require_file(self._ws, require_str_field(input, "path"))
            size = os.stat(resolved).st_size
            if size > self._ws.limits.max_read_bytes:
                raise WorkspaceError(
                    "file_too_large",
                    f"file is {size} bytes; limit is {self._ws.limits.max_read_bytes}",
                )
            data = resolved.read_bytes()
            try:
                content = data.decode(encoding)
            except UnicodeDecodeError as exc:
                raise WorkspaceError(
                    "decode_error", f"content is not valid {encoding}: {exc.reason}"
                ) from exc
            return ToolResult(
                ok=True,
                output={
                    "path": rel,
                    "content": content,
                    "encoding": encoding,
                    "size_bytes": size,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)


class WriteTextFileTool:
    """Writes a bounded text file inside the workspace (MEDIUM permission).

    Overwrites only when ``overwrite`` is explicitly true. Creates parent
    directories? No — the parent must already exist (use create_directory).
    """

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=WRITE_TOOL_NAME,
        description=(
            "Writes a text file inside the workspace (workspace-relative path). "
            "Creates the file, or overwrites it only when overwrite=true. The "
            "parent directory must already exist. Content size is limited."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to create or overwrite.",
                },
                "content": {"type": "string", "description": "File content to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite an existing file (default false).",
                },
                "encoding": {
                    "type": "string",
                    "description": f"Text encoding (default '{_DEFAULT_ENCODING}').",
                },
            },
            "required": ["path", "content"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "bytes_written": {"type": "integer"},
                "created": {"type": "boolean"},
            },
            "required": ["path", "bytes_written", "created"],
        },
        permission_level=PermissionLevel.MEDIUM,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            encoding = _validate_encoding(input)
            content = input.get("content")
            if not isinstance(content, str):
                raise WorkspaceError("invalid_content", "content must be a string")
            overwrite = bool(input.get("overwrite", False))
            resolved = self._ws.resolve(require_str_field(input, "path"))
            rel = self._ws.relative_to_root(resolved)
            if resolved.exists():
                if not resolved.is_file():
                    raise WorkspaceError("not_a_file", f"path is not a regular file: {rel}")
                if not overwrite:
                    raise WorkspaceError(
                        "target_exists",
                        f"target already exists and overwrite is false: {rel}",
                    )
            else:
                ensure_parent(resolved)
            data = content.encode(encoding)
            if len(data) > self._ws.limits.max_write_bytes:
                raise WorkspaceError(
                    "content_too_large",
                    f"content is {len(data)} bytes; limit is {self._ws.limits.max_write_bytes}",
                )
            created = not resolved.exists()
            atomic_write_bytes(resolved, data)
            return ToolResult(
                ok=True,
                output={"path": rel, "bytes_written": len(data), "created": created},
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)


class FileInfoTool:
    """Reports metadata for a workspace path (LOW permission).

    A missing path is a *successful* structured answer (``exists: false``),
    not an error.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=INFO_TOOL_NAME,
        description=(
            "Returns metadata for a workspace path (exists, is_file, "
            "is_directory, size_bytes). A missing path returns exists=false."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path (empty = workspace root).",
                },
            },
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "exists": {"type": "boolean"},
                "is_file": {"type": "boolean"},
                "is_directory": {"type": "boolean"},
                "size_bytes": {"type": "integer"},
            },
            "required": ["path", "exists", "is_file", "is_directory", "size_bytes"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            resolved = self._ws.resolve(require_str_field(input, "path"))
            rel = self._ws.relative_to_root(resolved)
            exists = resolved.exists()
            is_file = resolved.is_file()
            is_directory = resolved.is_dir()
            size = resolved.stat().st_size if is_file else 0
            return ToolResult(
                ok=True,
                output={
                    "path": rel,
                    "exists": exists,
                    "is_file": is_file,
                    "is_directory": is_directory,
                    "size_bytes": size,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
        except OSError as exc:
            mapped = as_os_error(exc)
            return fail(mapped.code, mapped.message)
