import hashlib
from pathlib import Path

import pytest
from agent_core.coding.editing import CodePatchApplier
from agent_core.coding.limits import CodingLimits
from agent_core.coding.models import CodeChange, CodePatch, CodingProject
from agent_core.permissions import PermissionDecision
from agent_core.workspace import Workspace


def test_patch_application_requires_allowed_and_rechecks_hash(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    project_root = root / "project"
    project_root.mkdir(parents=True)
    path = project_root / "main.py"
    original = b"print('old')\n"
    path.write_bytes(original)
    project = CodingProject(project_id="p", root_path="project")
    patch = CodePatch(
        project_id="p",
        summary="update",
        changes=(
            CodeChange(
                target_path="project/main.py",
                original_sha256=hashlib.sha256(original).hexdigest(),
                original_size_bytes=len(original),
                replacement_content="print('new')\n",
            ),
        ),
    )
    applier = CodePatchApplier(Workspace(root), CodingLimits(max_changed_files=2))
    with pytest.raises(PermissionError):
        applier.apply(project, patch, decision=PermissionDecision.REQUIRES_APPROVAL)
    assert path.read_bytes() == original
    assert applier.apply(project, patch, decision=PermissionDecision.ALLOWED) == (
        "project/main.py",
    )
    assert path.read_text() == "print('new')\n"
