"""Workspace boundary: safe path resolution inside a configured root (Phase 3).

The :class:`Workspace` is the single place that knows how to turn a
model-supplied path into a real filesystem path **without** allowing escape
from the configured workspace root:

- Tool paths are **workspace-relative only**. Absolute paths are rejected
  outright — a model must never be able to address host paths directly.
- Empty paths, NUL bytes, and over-long paths are rejected (``invalid_path``).
- The final path is resolved with :meth:`pathlib.Path.resolve`, which
  canonicalizes the whole chain **and follows symlinks / Windows junctions
  (reparse points)**, then it must equal the root or live under it. A
  symlink/junction inside the workspace that points outside therefore
  resolves outside and is rejected (``path_outside_workspace``).
- No string prefix checks: containment is decided on the fully resolved
  path, so ``../`` tricks, case tricks, and redirected directories all fail
  the same way.
- If a case cannot be proven safe (resolution error, unexpected OS failure),
  the operation is rejected with a structured security error rather than
  attempted.

Error messages deliberately contain only the *relative* path the caller
supplied — never the absolute host path or other filesystem details.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .errors import AgentCoreError


class WorkspaceError(AgentCoreError):
    """A workspace operation was rejected.

    ``code`` is a stable machine-readable error class (see ARCHITECTURE.md /
    SECURITY.md for the full set); ``message`` is concise and never leaks
    host-absolute paths.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class WorkspaceLimits:
    """Configurable safety limits for workspace operations."""

    max_read_bytes: int = 1_048_576  # 1 MiB
    max_write_bytes: int = 1_048_576  # 1 MiB
    max_list_entries: int = 500
    max_search_results: int = 200
    max_path_length: int = 512


class Workspace:
    """An explicitly configured, inescapable workspace root + limits."""

    def __init__(
        self,
        root: Path | str,
        limits: WorkspaceLimits | None = None,
    ) -> None:
        # Canonicalize the root itself (follows a symlinked root too).
        self._root = Path(root).expanduser().resolve()
        self._limits = limits or WorkspaceLimits()

    @classmethod
    def from_settings(cls, settings: Settings) -> Workspace:
        return cls(
            settings.workspace_root,
            WorkspaceLimits(
                max_read_bytes=settings.workspace_max_read_bytes,
                max_write_bytes=settings.workspace_max_write_bytes,
                max_list_entries=settings.workspace_max_list_entries,
                max_search_results=settings.workspace_max_search_results,
                max_path_length=settings.workspace_max_path_length,
            ),
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def limits(self) -> WorkspaceLimits:
        return self._limits

    def resolve(self, raw: str) -> Path:
        """Resolve a workspace-relative path safely.

        Returns the fully resolved absolute path. Raises
        :class:`WorkspaceError` with a stable ``code`` on any violation.
        ``""`` (or whitespace) resolves to the workspace root.
        """
        if not isinstance(raw, str):
            raise WorkspaceError("invalid_path", "path must be a string")
        stripped = raw.strip()
        if not stripped:
            return self._root
        if "\x00" in stripped:
            raise WorkspaceError("invalid_path", "path contains a NUL byte")
        if len(stripped) > self._limits.max_path_length:
            raise WorkspaceError(
                "invalid_path",
                f"path longer than {self._limits.max_path_length} characters",
            )
        # Tool paths are workspace-relative only; absolute paths are always
        # outside the contract, even when they would land inside the root.
        if Path(stripped).is_absolute() or _is_windows_absolute(stripped):
            raise WorkspaceError(
                "path_outside_workspace",
                "absolute paths are not allowed; use workspace-relative paths",
            )
        try:
            candidate = self._root / stripped
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            # Unsafe to determine containment -> fail closed.
            raise WorkspaceError("security_violation", "path could not be safely resolved") from exc
        if resolved != self._root and self._root not in resolved.parents:
            raise WorkspaceError("path_outside_workspace", "path resolves outside the workspace")
        return resolved

    def relative_to_root(self, resolved: Path) -> str:
        """POSIX-style relative path for output/events (never leaks host
        separators or the absolute host location)."""
        rel = resolved.relative_to(self._root)
        return "" if rel == Path(".") else rel.as_posix()

    def walk_files(
        self,
        top: Path,
        max_depth: int | None = None,
        *,
        skip_directories: Collection[str] = (),
        max_files: int | None = None,
        max_entries: int | None = None,
    ) -> Iterator[Path]:
        """Deterministically walk regular directory trees inside this workspace.

        Directory symlinks and reparse points are pruned before descent.
        ``skip_directories`` matches child directory names case-insensitively;
        the explicitly selected ``top`` directory is never skipped. ``max_files``
        bounds yielded file entries (use one extra sentinel to detect overflow).
        ``max_entries`` bounds discovered child directory/file entries and raises
        a safe ``WorkspaceError`` if traversal would exceed that budget.

        ``max_depth`` bounds how far below ``top`` to descend (1 = files
        directly in ``top`` only); ``None`` walks the whole subtree. Every
        yielded path remains under the canonical ``top`` boundary.
        """
        if max_files is not None and max_files < 0:
            raise ValueError("max_files must be non-negative")
        if max_entries is not None and max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        if max_files == 0:
            return
        try:
            base = top.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspaceError(
                "security_violation",
                "directory could not be safely resolved",
            ) from exc
        if base != self._root and self._root not in base.parents:
            raise WorkspaceError("path_outside_workspace", "directory resolves outside workspace")

        excluded = {name.casefold() for name in skip_directories}
        yielded = 0
        discovered_entries = 0

        def fail_walk(error: OSError) -> None:
            raise WorkspaceError("filesystem_error", "workspace traversal failed") from error

        for dirpath, dirnames, filenames in os.walk(
            base,
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        ):
            current = Path(dirpath)
            discovered_entries += len(dirnames) + len(filenames)
            if max_entries is not None and discovered_entries > max_entries:
                raise WorkspaceError(
                    "traversal_limit",
                    "workspace traversal exceeded its configured entry limit",
                )
            depth = len(current.relative_to(base).parts)
            if max_depth is not None:
                if depth + 1 >= max_depth:
                    dirnames[:] = []  # do not descend further
                if depth >= max_depth:
                    continue

            safe_directories: list[str] = []
            for name in sorted(dirnames):
                if name.casefold() in excluded:
                    continue
                child = current / name
                if _is_link_or_reparse_point(child):
                    continue
                try:
                    resolved = child.resolve()
                except (OSError, RuntimeError, ValueError):
                    continue
                if resolved != base and base not in resolved.parents:
                    continue
                safe_directories.append(name)
            dirnames[:] = safe_directories

            for name in sorted(filenames):
                yield current / name
                yielded += 1
                if max_files is not None and yielded >= max_files:
                    return


def _is_link_or_reparse_point(path: Path) -> bool:
    """Fail closed for directory links before a recursive walk descends."""
    try:
        entry_stat = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(entry_stat.st_mode):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if callable(is_junction):
        try:
            if is_junction(path):
                return True
        except OSError:
            return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(entry_stat, "st_file_attributes", 0) & reparse_flag)


def _is_windows_absolute(path: str) -> bool:
    """Detect Windows absolute forms (``C:\\...``, UNC) as plain strings, so
    the rule also holds on POSIX (a model might still send ``C:\\temp``)."""
    if path.startswith("\\\\"):  # UNC: \\server\share
        return True
    # Drive letter: C:\... or C:/...
    return len(path) >= 2 and path[1] == ":" and path[0].isalpha()
