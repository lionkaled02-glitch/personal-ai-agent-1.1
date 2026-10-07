"""Search tool: search_files.

Case-sensitive glob-style matching (``*``, ``?``, ``[seq]``) against the
*relative POSIX path* of each file in the workspace — no regular
expressions, no shell expansion, no path components beyond the relative
path. Results are deterministic (sorted walk), limited to regular files
(symlinks are never followed), and capped by the configured
``max_search_results``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec
from ..workspace import Workspace, WorkspaceError
from ._common import fail

SEARCH_TOOL_NAME = "search_files"

_MAX_PATTERN_LENGTH = 200


class SearchFilesTool:
    """Finds files by glob-style pattern inside the workspace (LOW)."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    spec = ToolSpec(
        name=SEARCH_TOOL_NAME,
        description=(
            "Finds files inside the workspace whose workspace-relative POSIX "
            "path matches a glob-style pattern (*, ?, [seq]). Case-sensitive. "
            "Results are sorted, capped in count, and include size metadata."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob-style pattern, e.g. 'docs/**/*.md' or 'src/*.py'.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Optional workspace-relative directory to search in "
                        "(default: whole workspace)."
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search subdirectories (default true).",
                },
            },
            "required": ["pattern"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "size_bytes": {"type": "integer"},
                            "type": {"enum": ["file"]},
                        },
                        "required": ["path", "size_bytes", "type"],
                    },
                },
                "count": {"type": "integer"},
                "truncated": {"type": "boolean"},
            },
            "required": ["pattern", "path", "entries", "count", "truncated"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        pattern = input.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return fail("invalid_path", "pattern is required and must be a non-empty string")
        if len(pattern) > _MAX_PATTERN_LENGTH:
            return fail(
                "invalid_path", f"pattern is too long (max {_MAX_PATTERN_LENGTH} characters)"
            )
        try:
            base_rel = input.get("path")
            if base_rel is not None and not isinstance(base_rel, str):
                return fail("invalid_path", "path must be a string")
            resolved = self._ws.resolve(base_rel or "")
            if not resolved.is_dir():
                return fail(
                    "path_not_found" if not resolved.exists() else "not_a_directory",
                    f"search path is not a directory: {self._ws.relative_to_root(resolved)}",
                )
            recursive = bool(input.get("recursive", True))
            limit = self._ws.limits.max_search_results
            files = (
                self._ws.walk_files(resolved)
                if recursive
                else self._ws.walk_files(resolved, max_depth=1)
            )
            entries: list[dict[str, Any]] = []
            total_matches = 0
            for file_path in files:
                rel = self._ws.relative_to_root(file_path)
                if not fnmatchcase(rel, pattern):
                    continue
                total_matches += 1
                if len(entries) < limit:
                    try:
                        size = file_path.stat().st_size
                    except OSError:
                        size = 0
                    entries.append({"path": rel, "size_bytes": size, "type": "file"})
            return ToolResult(
                ok=True,
                output={
                    "pattern": pattern,
                    "path": self._ws.relative_to_root(resolved),
                    "entries": entries,
                    "count": len(entries),
                    "truncated": total_matches > limit,
                },
            )
        except WorkspaceError as exc:
            return fail(exc.code, exc.message)
