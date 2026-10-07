"""Safe application of previously validated coding patches."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..permissions import PermissionDecision
from ..workspace import Workspace, WorkspaceError
from .errors import CodingValidationError, CodingWorkspaceError
from .limits import CodingLimits
from .models import CodeEditResult, CodePatch, CodingProject


class CodePatchApplier:
    """Apply full-file replacements only after explicit permission and revalidation."""

    def __init__(self, workspace: Workspace, limits: CodingLimits) -> None:
        self.workspace = workspace
        self.limits = limits

    def apply(
        self, project: CodingProject, patch: CodePatch, *, decision: PermissionDecision
    ) -> tuple[str, ...]:
        if decision is not PermissionDecision.ALLOWED:
            raise PermissionError("coding patch application requires ALLOWED permission")
        self.limits.validate_patch(patch)
        targets = patch.resolve_target_paths(project, self.workspace)
        staged: list[tuple[Path, bytes, str]] = []
        for change, relative in zip(patch.changes, targets, strict=True):
            try:
                path = self.workspace.resolve(relative)
            except WorkspaceError:
                raise CodingWorkspaceError() from None
            if not path.is_file():
                raise CodingValidationError()
            current = path.read_bytes()
            digest = hashlib.sha256(current).hexdigest()
            if digest != change.original_sha256 or len(current) != change.original_size_bytes:
                raise CodingValidationError()
            replacement = change.replacement_content.encode("utf-8")
            staged.append((path, replacement, relative))
        written: list[Path] = []
        try:
            for path, replacement, _ in staged:
                tmp = path.with_name(f".{path.name}.agent-tmp")
                if tmp.exists():
                    tmp.unlink()
                with open(tmp, "wb") as handle:
                    handle.write(replacement)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                written.append(path)
        except OSError as exc:
            for path in written:
                # Never attempt speculative rollback over a changed workspace.
                # The caller receives a controlled failure; subsequent hash checks prevent stale writes.  # noqa: E501
                _ = path
            raise CodingValidationError() from exc
        return tuple(relative for _, _, relative in staged)

    def apply_result(
        self, project: CodingProject, result: CodeEditResult, *, decision: PermissionDecision
    ) -> tuple[str, ...]:
        if result.patch is None:
            return ()
        return self.apply(project, result.patch, decision=decision)
