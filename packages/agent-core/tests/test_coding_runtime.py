"""Deterministic offline tests for the Phase 10 Step 2 read-only analyzer."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from agent_core import (
    PermissionDecision,
    PermissionLevel,
    PermissionManager,
    Workspace,
    WorkspaceLimits,
)
from agent_core.coding import (
    CodeAnalysisLimitReason,
    CodeAnalysisMethod,
    CodeAnalysisResult,
    CodeAnalysisStatus,
    CodeDiagnosticCategory,
    CodeLanguage,
    CodeSkipReason,
    CodeSymbolKind,
    CodingAnalysisRuntime,
    CodingLimits,
    CodingProject,
)


def _project_tree(
    tmp_path: Path,
    *,
    root_name: str = "project",
) -> tuple[Workspace, CodingProject, Path]:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    project_root = workspace_root / root_name
    project_root.mkdir()
    return (
        Workspace(workspace_root),
        CodingProject(project_id="analysis-test", root_path=root_name),
        project_root,
    )


def _write(project_root: Path, relative_path: str, content: str | bytes) -> Path:
    target = project_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        # Preserve LF bytes so source-character expectations are platform-neutral.
        target.write_text(content, encoding="utf-8", newline="")
    return target


def _analyze(
    workspace: Workspace,
    project: CodingProject,
    limits: CodingLimits | None = None,
) -> CodeAnalysisResult:
    return CodingAnalysisRuntime(workspace, limits).analyze_project(project)


class TestCodingAnalysisRuntime:
    def test_python_ast_is_read_only_and_reports_bounded_symbols(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        marker = tmp_path / "must-not-be-created.txt"
        marker_expression = " +\n    ".join(
            repr(str(marker)[index : index + 40]) for index in range(0, len(str(marker)), 40)
        )
        source = (
            "import pathlib\n"
            "CONSTANT = 4\n"
            "def calculate(value):\n"
            "    return value + CONSTANT\n"
            "class Calculator:\n"
            "    def add(self, left, right):\n"
            "        return left + right\n"
            "pathlib.Path(\n"
            f"    {marker_expression}\n"
            ").write_text('executed')\n"
        )
        _write(project_root, "src/math_tools.py", source)

        runtime = CodingAnalysisRuntime(workspace)
        result = runtime.analyze_project(project)

        assert result.status is CodeAnalysisStatus.COMPLETE
        assert result.truncated is False
        assert result.limit_reasons == ()
        assert not marker.exists()
        assert [(item.name, item.kind) for item in result.symbols] == [
            ("pathlib", CodeSymbolKind.OTHER),
            ("CONSTANT", CodeSymbolKind.VARIABLE),
            ("calculate", CodeSymbolKind.FUNCTION),
            ("Calculator", CodeSymbolKind.CLASS),
            ("add", CodeSymbolKind.METHOD),
        ]
        assert result.analyzed_files[0].method is CodeAnalysisMethod.PYTHON_AST
        assert result.analyzed_files[0].path == "project/src/math_tools.py"
        assert result.analyzed_files[0].line_count == len(source.splitlines())
        assert result.observation.total_source_chars == len(source)
        assert runtime.name == "coding_analyze"
        assert runtime.permission_level is PermissionLevel.LOW
        assert PermissionManager().check(runtime) is PermissionDecision.ALLOWED
        result.validate_workspace(project, workspace)

    def test_analysis_facts_are_stable_across_runs(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "b.py", "def second():\n    return 2\n")
        _write(project_root, "a.py", "def first():\n    return 1\n")

        first = _analyze(workspace, project)
        second = _analyze(workspace, project)

        assert first.symbols == second.symbols
        assert first.diagnostics == second.diagnostics
        assert first.analyzed_files == second.analyzed_files
        assert first.skipped_files == second.skipped_files
        assert first.observation == second.observation
        assert [item.path for item in first.analyzed_files] == [
            "project/a.py",
            "project/b.py",
        ]

    def test_javascript_and_typescript_use_only_a_shallow_line_scan(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        source = (
            'import defaultThing, { usefulThing, other as renamed } from "pkg";\n'
            "export async function run() {}\n"
            "export class Widget {}\n"
            "const calculate = (input) => input;\n"
            "export { calculate as publicCalculate };\n"
        )
        _write(project_root, "src/app.ts", source)

        result = _analyze(workspace, project)

        assert result.status is CodeAnalysisStatus.COMPLETE
        assert result.analyzed_files[0].language is CodeLanguage.TYPESCRIPT
        assert result.analyzed_files[0].method is CodeAnalysisMethod.ECMASCRIPT_LINE_SCAN
        assert [item.name for item in result.symbols] == [
            "defaultThing",
            "usefulThing",
            "renamed",
            "run",
            "Widget",
            "calculate",
            "publicCalculate",
        ]
        assert all(item.region.start_line >= 1 for item in result.symbols)

    def test_metadata_formats_are_not_executed_or_returned_as_source(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        marker = tmp_path / "package-script-must-not-run"
        package = (
            '{"scripts":{"postinstall":"create marker"},"homepage":"https://example.invalid/"}\n'
        )
        _write(project_root, "package.json", package)
        _write(project_root, "README.md", "# Title\nText\n## Details\n")
        _write(project_root, "settings.yaml", "name: demo\ncount: 2\n# comment\n")
        _write(project_root, "settings.toml", "name = 'demo'\n[tool]\nenabled = true\n")

        result = _analyze(workspace, project)

        assert not marker.exists()
        assert all(
            item.method is CodeAnalysisMethod.METADATA_ONLY for item in result.analyzed_files
        )
        by_path = {item.path: item for item in result.analyzed_files}
        assert by_path["project/package.json"].metrics[0].value == 2
        assert by_path["project/README.md"].metrics[0].value == 2
        assert by_path["project/settings.yaml"].metrics[0].value == 2
        assert by_path["project/settings.toml"].metrics[0].value == 2
        serialized = result.model_dump_json()
        assert "create marker" not in serialized
        assert "https://example.invalid/" not in serialized
        assert package not in serialized

    def test_sensitive_and_unsupported_files_are_skipped_before_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, ".env", "PRIVATE_SENTINEL=never-open\n")
        _write(project_root, "archive.png", b"\x89PNG\r\n\x1a\n")
        _write(project_root, ".env.example", "PUBLIC_EXAMPLE=true\n")
        _write(project_root, "main.py", "answer = 42\n")
        opened: list[str] = []
        original_open = Path.open

        def tracking_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            opened.append(path.name)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", tracking_open)
        result = _analyze(workspace, project)

        skipped = {item.path: item.reason for item in result.skipped_files}
        assert skipped["project/.env"] is CodeSkipReason.SENSITIVE_FILE_NAME
        assert skipped["project/archive.png"] is CodeSkipReason.UNSUPPORTED_FILE_TYPE
        assert ".env" not in opened
        assert "archive.png" not in opened
        assert ".env.example" in opened
        assert "PRIVATE_SENTINEL" not in result.model_dump_json()
        assert result.status is CodeAnalysisStatus.PARTIAL

    def test_invalid_utf8_and_nul_binary_are_skipped_without_exposure(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "bad.py", b"\xff\xfe")
        _write(project_root, "contains-nul.txt", b"text\x00binary")

        result = _analyze(workspace, project)

        reasons = {item.path: item.reason for item in result.skipped_files}
        assert reasons["project/bad.py"] is CodeSkipReason.DECODE_FAILED
        assert reasons["project/contains-nul.txt"] is CodeSkipReason.BINARY_FILE
        assert result.observation.files == ()

    def test_excluded_directories_are_pruned_but_selected_root_is_not(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path, root_name="build")
        _write(project_root, "src/main.py", "answer = 42\n")
        _write(project_root, "vendor/library.py", "vendor = True\n")
        _write(project_root, "node_modules/pkg/index.js", "export const x = 1;\n")
        _write(project_root, ".git/config", "ignored = true\n")

        result = _analyze(workspace, project)

        assert [item.path for item in result.analyzed_files] == ["build/src/main.py"]
        assert result.skipped_files == ()

    def test_project_and_workspace_escape_paths_are_never_opened(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        outside = tmp_path / "outside-secret.py"
        outside.write_text("OUTSIDE_SENTINEL = 'do not read'\n", encoding="utf-8")
        link = project_root / "outside.py"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks are unavailable")

        opened: list[Path] = []
        original_open = Path.open

        def tracking_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            opened.append(path)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", tracking_open)
        result = _analyze(workspace, project)

        assert result.status is CodeAnalysisStatus.PARTIAL
        assert any(
            item.path == "project/outside.py" and item.reason is CodeSkipReason.UNSAFE_PATH
            for item in result.skipped_files
        )
        assert outside not in opened
        assert "OUTSIDE_SENTINEL" not in result.model_dump_json()
        result.validate_workspace(project, workspace)

    def test_internal_file_symlink_aliases_are_deduplicated_without_overlap(
        self,
        tmp_path: Path,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        target = _write(project_root, "target.py", "value = 1\n")
        try:
            os.symlink(target, project_root / "alias.py")
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("symlinks are unavailable")

        result = _analyze(workspace, project)

        assert [item.path for item in result.analyzed_files] == ["project/target.py"]
        assert result.skipped_files == ()
        assert any(item.code == "duplicate_project_path" for item in result.diagnostics)
        result.validate_workspace(project, workspace)

    def test_symlinked_directories_are_pruned_before_descent(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        outside_dir = tmp_path / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "secret.py").write_text("payload = True\n", encoding="utf-8")
        try:
            os.symlink(outside_dir, project_root / "linked", target_is_directory=True)
        except (OSError, NotImplementedError, ValueError):
            pytest.skip("directory symlinks are unavailable")

        result = _analyze(workspace, project)

        assert result.analyzed_files == ()
        assert result.skipped_files == ()
        assert "secret" not in result.model_dump_json()

    @pytest.mark.parametrize("root_path", ["../outside", "/absolute/project", "C:\\outside"])
    def test_invalid_project_roots_fail_closed(self, tmp_path: Path, root_path: str) -> None:
        workspace, _, _ = _project_tree(tmp_path)
        project = CodingProject(project_id="invalid-root", root_path=root_path)

        result = _analyze(workspace, project)

        assert result.status is CodeAnalysisStatus.PARTIAL
        assert result.analyzed_files == ()
        assert result.observation.files == ()
        assert result.diagnostics[0].code == "project_boundary_invalid"

    def test_workspace_read_limit_cannot_be_exceeded_by_coding_limits(
        self,
        tmp_path: Path,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "source.py", "answer = 42\n")
        restricted_workspace = Workspace(
            workspace.root,
            WorkspaceLimits(max_read_bytes=3),
        )

        result = _analyze(
            restricted_workspace,
            project,
            CodingLimits(max_file_size_bytes=100),
        )

        assert result.analyzed_files == ()
        assert result.skipped_files[0].reason is CodeSkipReason.FILE_TOO_LARGE

    def test_project_file_and_discovery_entry_limits_are_reported(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        for index in range(3):
            _write(project_root, f"{index}.py", f"value_{index} = {index}\n")
        result = _analyze(
            workspace,
            project,
            CodingLimits(max_project_files=2),
        )
        assert len(result.analyzed_files) == 2
        assert CodeAnalysisLimitReason.PROJECT_FILE_LIMIT in result.limit_reasons

        workspace2, project2, project_root2 = _project_tree(tmp_path / "second")
        for index in range(321):
            (project_root2 / f"empty-{index:03d}").mkdir()
        limited = _analyze(workspace2, project2, CodingLimits(max_project_files=10))
        assert limited.analyzed_files == ()
        assert CodeAnalysisLimitReason.DISCOVERY_ENTRY_LIMIT in limited.limit_reasons
        assert limited.truncated is True

    def test_file_source_symbol_region_and_diagnostic_limits_are_explicit(
        self,
        tmp_path: Path,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "large.py", "value = 12345\n")
        per_file = _analyze(
            workspace,
            project,
            CodingLimits(max_file_size_bytes=4),
        )
        assert per_file.skipped_files[0].reason is CodeSkipReason.FILE_TOO_LARGE
        assert CodeAnalysisLimitReason.FILE_SIZE_LIMIT in per_file.limit_reasons

        workspace2, project2, project_root2 = _project_tree(tmp_path / "aggregate")
        _write(project_root2, "a.py", "first = 1\n")
        _write(project_root2, "b.py", "second = 2\n")
        aggregate = _analyze(
            workspace2,
            project2,
            CodingLimits(max_source_chars=12),
        )
        assert CodeAnalysisLimitReason.SOURCE_CHAR_LIMIT in aggregate.limit_reasons
        assert any(
            item.reason is CodeSkipReason.SOURCE_LIMIT_EXCEEDED for item in aggregate.skipped_files
        )

        symbol_limited = _analyze(
            workspace2,
            project2,
            CodingLimits(max_symbols=1),
        )
        assert len(symbol_limited.symbols) == 1
        assert CodeAnalysisLimitReason.SYMBOL_LIMIT in symbol_limited.limit_reasons

        region_limited = _analyze(
            workspace2,
            project2,
            CodingLimits(max_regions=0),
        )
        assert region_limited.symbols == ()
        assert CodeAnalysisLimitReason.REGION_LIMIT in region_limited.limit_reasons

        _write(project_root2, "ignored.bin", b"not source")
        diagnostics_limited = _analyze(
            workspace2,
            project2,
            CodingLimits(max_diagnostics=0),
        )
        assert diagnostics_limited.diagnostics == ()
        assert CodeAnalysisLimitReason.DIAGNOSTIC_LIMIT in diagnostics_limited.limit_reasons

    def test_output_and_cooperative_time_limits_are_bounded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(
            project_root,
            "many.py",
            "\n".join(f"value_{index} = {index}" for index in range(12)),
        )
        output_limited = _analyze(
            workspace,
            project,
            CodingLimits(max_output_bytes=1_400),
        )
        assert CodeAnalysisLimitReason.OUTPUT_LIMIT in output_limited.limit_reasons
        assert len(output_limited.model_dump_json().encode("utf-8")) <= 1_400

        tick_values = iter([0.0, 1.1, 1.2, 1.3])
        monkeypatch.setattr(time, "perf_counter", lambda: next(tick_values))
        time_limited = _analyze(
            workspace,
            project,
            CodingLimits(max_analysis_time_s=1.0),
        )
        assert time_limited.analyzed_files == ()
        assert CodeAnalysisLimitReason.ANALYSIS_TIME_LIMIT in time_limited.limit_reasons
        assert time_limited.elapsed_time_s == 1.0

    def test_python_javascript_and_typescript_diagnostics_are_workspace_scoped(
        self,
        tmp_path: Path,
    ) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "src/broken.py", "def broken(:\n    pass\n")
        _write(project_root, "src/broken.js", "const result = values[0;\n")
        _write(project_root, "src/style.ts", "const value = 1  \n")

        result = _analyze(workspace, project)

        by_code = {(item.code, item.path): item for item in result.diagnostics}
        expected = (
            ("PY001", "project/src/broken.py"),
            ("ECMA001", "project/src/broken.js"),
            ("STYLE001", "project/src/style.ts"),
        )
        for key in expected:
            diagnostic = by_code[key]
            assert diagnostic.region is not None
            assert diagnostic.region.path == key[1]
            assert diagnostic.region.start_line == 1
        assert result.status is CodeAnalysisStatus.PARTIAL
        assert result.observation.files
        result.validate_workspace(project, workspace)

    def test_diagnostics_obey_shared_diagnostic_and_region_limits(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "style.py", "first = 1  \nsecond = 2  \n")

        diagnostic_limited = _analyze(
            workspace,
            project,
            CodingLimits(max_diagnostics=1),
        )
        assert len(diagnostic_limited.diagnostics) == 1
        assert CodeAnalysisLimitReason.DIAGNOSTIC_LIMIT in diagnostic_limited.limit_reasons
        assert diagnostic_limited.truncated is True

        region_limited = _analyze(
            workspace,
            project,
            CodingLimits(max_regions=0),
        )
        assert len(region_limited.diagnostics) == 2
        assert all(item.region is None for item in region_limited.diagnostics)
        assert CodeAnalysisLimitReason.REGION_LIMIT in region_limited.limit_reasons
        assert region_limited.truncated is True
        region_limited.validate_workspace(project, workspace)

    def test_malformed_python_is_reported_without_execution(self, tmp_path: Path) -> None:
        workspace, project, project_root = _project_tree(tmp_path)
        _write(project_root, "broken.py", "def broken(:\n    pass\n")

        result = _analyze(workspace, project)

        assert result.analyzed_files[0].method is CodeAnalysisMethod.PYTHON_AST
        assert result.symbols == ()
        assert result.status is CodeAnalysisStatus.PARTIAL
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "PY001"
        assert diagnostic.category is CodeDiagnosticCategory.SYNTAX
        assert diagnostic.severity.value == "error"
        assert diagnostic.path == "project/broken.py"
        assert diagnostic.region is not None
        assert diagnostic.region.start_line == 1
        result.validate_workspace(project, workspace)
