"""Shared helpers for workspace tools.

Workspace tools never raise: every failure becomes a structured
:class:`~agent_core.tools.ToolResult` with a stable ``error_code`` and a
concise message that contains only workspace-relative paths (never the
absolute host path or other filesystem internals).
"""

from __future__ import annotations

import codecs
import contextlib
import errno
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..tools import ToolResult
from ..workspace import Workspace, WorkspaceError


def fail(code: str, message: str) -> ToolResult:
    return ToolResult(ok=False, error=message, error_code=code)


def as_os_error(exc: OSError) -> WorkspaceError:
    """Map an OS error to a stable workspace error (bounded, path-free)."""
    if exc.errno in (errno.EACCES, errno.EPERM):
        return WorkspaceError(
            "permission_denied", "operation not permitted by the operating system"
        )
    return WorkspaceError(
        "filesystem_error",
        f"filesystem operation failed: {type(exc).__name__} (errno={exc.errno})",
    )


def check_encoding(encoding: str) -> str:
    """Validate an encoding name (raises WorkspaceError when unknown)."""
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise WorkspaceError("invalid_encoding", f"unknown encoding {encoding!r}") from exc
    return encoding


def require_str_field(input: Mapping[str, Any], key: str, *, code: str = "invalid_path") -> str:
    """Fetch a required string field with NO silent coercion of other types.

    Missing, non-string, or otherwise malformed input fails closed with a
    structured error (the code depends on the field; ``invalid_path`` for
    path-like fields).
    """
    value = input.get(key)
    if not isinstance(value, str):
        raise WorkspaceError(code, f"{key} must be a string")
    return value


def require_file(workspace: Workspace, raw: str) -> tuple[Path, str]:
    """Resolve ``raw`` and require an existing regular file.

    Returns ``(resolved_path, relative_posix_path)``.
    """
    resolved = workspace.resolve(raw)
    rel = workspace.relative_to_root(resolved)
    if not resolved.exists():
        raise WorkspaceError("path_not_found", f"path not found: {rel or '<workspace root>'}")
    if not resolved.is_file():
        raise WorkspaceError(
            "not_a_file", f"path is not a regular file: {rel or '<workspace root>'}"
        )
    return resolved, rel


def require_source(workspace: Workspace, raw: str) -> tuple[Path, str]:
    """Resolve ``raw`` and require an existing regular file, reporting a
    missing file as ``source_not_found`` (copy/move semantics)."""
    resolved = workspace.resolve(raw)
    rel = workspace.relative_to_root(resolved)
    if not resolved.exists():
        raise WorkspaceError("source_not_found", f"source not found: {rel or '<workspace root>'}")
    if not resolved.is_file():
        raise WorkspaceError(
            "not_a_file", f"source is not a regular file: {rel or '<workspace root>'}"
        )
    return resolved, rel


def require_directory(workspace: Workspace, raw: str) -> tuple[Path, str]:
    """Resolve ``raw`` and require an existing directory."""
    resolved = workspace.resolve(raw)
    rel = workspace.relative_to_root(resolved)
    if not resolved.exists():
        raise WorkspaceError("path_not_found", f"path not found: {rel or '<workspace root>'}")
    if not resolved.is_dir():
        raise WorkspaceError(
            "not_a_directory", f"path is not a directory: {rel or '<workspace root>'}"
        )
    return resolved, rel


def ensure_parent(resolved: Path) -> None:
    """Require the parent directory of a new file to exist (write/copy/move
    never create directories implicitly)."""
    if not resolved.parent.is_dir():
        raise WorkspaceError("path_not_found", "parent directory does not exist")


def atomic_write_bytes(resolved: Path, data: bytes) -> None:
    """Write bytes atomically: a uniquely-named temp file in the same
    directory, then ``os.replace`` (atomic on POSIX and Windows)."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, resolved)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
