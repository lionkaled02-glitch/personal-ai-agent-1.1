from pathlib import Path

from agent_core.coding.limits import CodingLimits
from agent_core.coding.models import CodingProject
from agent_core.coding.search import CodeSearchRuntime
from agent_core.workspace import Workspace


def _runtime(tmp_path: Path) -> tuple[Workspace, CodingProject, CodeSearchRuntime]:
    root = tmp_path / "workspace"
    project_root = root / "project"
    project_root.mkdir(parents=True)
    (project_root / "main.py").write_text("def hello():\n    return 'needle'\n", encoding="utf-8")
    (project_root / "notes.txt").write_text("needle here\n", encoding="utf-8")
    workspace = Workspace(root)
    project = CodingProject(project_id="p1", root_path="project")
    return workspace, project, CodeSearchRuntime(workspace, CodingLimits(max_project_files=20))


def test_text_search_is_bounded_and_read_only(tmp_path: Path) -> None:
    workspace, project, runtime = _runtime(tmp_path)
    before = (workspace.root / "project" / "main.py").read_bytes()
    result = runtime.text_search(project, "needle")
    assert [h.path for h in result.hits] == ["main.py", "notes.txt"]
    assert (workspace.root / "project" / "main.py").read_bytes() == before


def test_symbol_and_definition_search(tmp_path: Path) -> None:
    _, project, runtime = _runtime(tmp_path)
    symbols = runtime.symbol_search(project, "hello")
    definitions = runtime.find_definition(project, "hello")
    assert symbols.hits[0].symbol == "hello"
    assert definitions.hits[0].line == 1


def test_file_search_does_not_escape_project(tmp_path: Path) -> None:
    _, project, runtime = _runtime(tmp_path)
    result = runtime.file_search(project, "main")
    assert [h.path for h in result.hits] == ["main.py"]
