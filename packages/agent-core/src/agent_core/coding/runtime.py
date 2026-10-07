"""Read-only, deterministic analysis of bounded workspace coding projects.

Project text is untrusted data. Python is parsed to an AST without execution;
JavaScript/TypeScript receive a shallow line scan; other supported formats
produce metadata only. Paths are always resolved by the shared ``Workspace``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import stat
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

from ..permissions import PermissionLevel
from ..workspace import Workspace, WorkspaceError
from .diagnostics import CodeDiagnosticsEngine
from .errors import CodingLimitError, CodingWorkspaceError
from .interfaces import CodingOperation
from .limits import CodingLimits
from .models import (
    MAX_CODING_PATH_CHARS,
    AnalyzedCodeFile,
    CodeAnalysisLimitReason,
    CodeAnalysisMethod,
    CodeAnalysisResult,
    CodeAnalysisStatus,
    CodeDiagnostic,
    CodeDiagnosticCategory,
    CodeFile,
    CodeFileMetric,
    CodeLanguage,
    CodeRegion,
    CodeSkipReason,
    CodeSymbol,
    CodeSymbolKind,
    CodingObservation,
    CodingProject,
    DiagnosticSeverity,
    SkippedCodeFile,
)

# These are exact child directory names. The explicit project root is never
# pruned, even if it is named "build", "vendor", or another excluded name.
_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".bzr",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "virtualenv",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        "build",
        "dist",
        "target",
        "out",
        "coverage",
        "site-packages",
        "vendor",
        ".ssh",
        ".secrets",
        "credentials",
        "secrets",
    }
)
_SAFE_ENV_EXAMPLES = frozenset({".env.example", ".env.sample", ".env.template"})
_SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".der"})
_SENSITIVE_NAME_RE = re.compile(
    r"(?i)(^|[._-])(credential|credentials|secret|secrets|password|passwd|token|"
    r"api[-_]?key|access[-_]?token|auth[-_]?token|id_rsa|id_ed25519|private[-_]?key)([._-]|$)"
)

_LANGUAGE_BY_EXTENSION: dict[str, CodeLanguage] = {
    ".py": CodeLanguage.PYTHON,
    ".pyi": CodeLanguage.PYTHON,
    ".js": CodeLanguage.JAVASCRIPT,
    ".mjs": CodeLanguage.JAVASCRIPT,
    ".cjs": CodeLanguage.JAVASCRIPT,
    ".jsx": CodeLanguage.JAVASCRIPT,
    ".ts": CodeLanguage.TYPESCRIPT,
    ".mts": CodeLanguage.TYPESCRIPT,
    ".cts": CodeLanguage.TYPESCRIPT,
    ".tsx": CodeLanguage.TYPESCRIPT,
    ".json": CodeLanguage.JSON,
    ".yaml": CodeLanguage.YAML,
    ".yml": CodeLanguage.YAML,
    ".toml": CodeLanguage.TOML,
    ".md": CodeLanguage.MARKDOWN,
    ".markdown": CodeLanguage.MARKDOWN,
    ".html": CodeLanguage.HTML,
    ".htm": CodeLanguage.HTML,
    ".css": CodeLanguage.CSS,
    ".scss": CodeLanguage.CSS,
    ".less": CodeLanguage.CSS,
    ".txt": CodeLanguage.TEXT,
    ".text": CodeLanguage.TEXT,
    ".cfg": CodeLanguage.TEXT,
    ".ini": CodeLanguage.TEXT,
    ".conf": CodeLanguage.TEXT,
    ".properties": CodeLanguage.TEXT,
    ".xml": CodeLanguage.TEXT,
}
_TEXT_BASENAMES = frozenset(
    {
        ".dockerignore",
        ".editorconfig",
        ".gitignore",
        ".npmrc",
        ".nvmrc",
        ".python-version",
        ".tool-versions",
        "dockerfile",
        "gnumakefile",
        "justfile",
        "license",
        "makefile",
        "notice",
        "procfile",
    }
)
_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]{0,255}"
_IMPORT_FROM_RE = re.compile(
    r"^\s*import\s+(?:type\s+)?(?P<bindings>[^;\n]{1,512}?)\s+from\s*['\"][^'\"]{1,256}['\"]"
)
_SIDE_EFFECT_IMPORT_RE = re.compile(r"^\s*import\s*['\"][^'\"]{1,256}['\"]")
_FUNCTION_RE = re.compile(
    rf"^\s*(?:(?:export|default|declare)\s+)*(?:async\s+)?function\s*\*?\s*({_IDENTIFIER})"
)
_CLASS_RE = re.compile(rf"^\s*(?:(?:export|default|declare|abstract)\s+)*class\s+({_IDENTIFIER})")
_VARIABLE_RE = re.compile(
    rf"^\s*(?:(?:export|default|declare)\s+)*(?:const|let|var)\s+({_IDENTIFIER})"
)
_ARROW_FUNCTION_RE = re.compile(
    rf"^\s*(?:(?:export|default)\s+)*(?:const|let|var)\s+({_IDENTIFIER})\s*=\s*(?:async\s+)?(?:function\b|\([^\n)]{{0,256}}\)\s*=>|{_IDENTIFIER}\s*=>)"
)
_TS_TYPE_RE = re.compile(rf"^\s*export\s+(?:declare\s+)?(?:interface|type|enum)\s+({_IDENTIFIER})")
_EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{\s*([^}\n]{1,512})\s*\}")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+")
_YAML_MAPPING_RE = re.compile(r"(?m)^[ \t]*[^#\s][^:\n]{0,256}:")


class _SymbolLimitReached(Exception):
    """Internal non-error signal that the configured structural cap was hit."""


_SymbolEmitter = Callable[[str, CodeSymbolKind, int, int, str | None], bool]


class _PythonStructureVisitor(ast.NodeVisitor):
    """Extract a small allow-listed set of Python AST nodes without execution."""

    def __init__(self, emit: _SymbolEmitter) -> None:
        self._emit = emit
        self._classes: list[str] = []
        self._function_depth = 0

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_body(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = ".".join((*self._classes, node.name))
        self._emit_node(node, node.name, CodeSymbolKind.CLASS, qualified)
        self._classes.append(node.name)
        self._visit_body(node.body)
        self._classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._emit_node(node, name, CodeSymbolKind.OTHER, alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            qualified = f"{module}.{alias.name}" if module else alias.name
            self._emit_node(node, name, CodeSymbolKind.OTHER, qualified)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._classes and self._function_depth == 0:
            for target in node.targets:
                for name in _assignment_names(target):
                    self._emit_node(node, name, CodeSymbolKind.VARIABLE, name)
        # Expression trees cannot declare module/class/function symbols; do not
        # recursively walk untrusted right-hand-side expressions.

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._classes and self._function_depth == 0:
            for name in _assignment_names(node.target):
                self._emit_node(node, name, CodeSymbolKind.VARIABLE, name)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not self._classes and self._function_depth == 0:
            for name in _assignment_names(node.target):
                self._emit_node(node, name, CodeSymbolKind.VARIABLE, name)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self._classes) and self._function_depth == 0
        kind = CodeSymbolKind.METHOD if is_method else CodeSymbolKind.FUNCTION
        qualified_parts = [*self._classes, node.name] if self._classes else [node.name]
        self._emit_node(node, node.name, kind, ".".join(qualified_parts))
        self._function_depth += 1
        self._visit_body(node.body)
        self._function_depth -= 1

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for statement in body:
            self.visit(statement)

    def _emit_node(
        self,
        node: ast.AST,
        name: str,
        kind: CodeSymbolKind,
        qualified_name: str | None,
    ) -> None:
        line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", line) or line
        if not isinstance(line, int) or not isinstance(end_line, int):
            return
        if not self._emit(name, kind, line, end_line, qualified_name):
            raise _SymbolLimitReached


def _assignment_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(_assignment_names(item))
        return tuple(names)
    if isinstance(target, ast.Starred):
        return _assignment_names(target.value)
    return ()


class _AnalysisAccumulator:
    """Mutable, request-local bounded collector; it retains no state after return."""

    def __init__(self, limits: CodingLimits) -> None:
        self.limits = limits
        self.symbols: list[CodeSymbol] = []
        self.diagnostics: list[CodeDiagnostic] = []
        self.analyzed_files: list[AnalyzedCodeFile] = []
        self.skipped_files: list[SkippedCodeFile] = []
        self.source_files: list[CodeFile] = []
        self.limit_reasons: list[CodeAnalysisLimitReason] = []
        self.region_count = 0
        self.total_source_chars = 0

    @property
    def truncated(self) -> bool:
        return bool(self.limit_reasons)

    def mark_limit(self, reason: CodeAnalysisLimitReason) -> None:
        if reason not in self.limit_reasons:
            self.limit_reasons.append(reason)

    def add_diagnostic(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
        category: CodeDiagnosticCategory = CodeDiagnosticCategory.ANALYSIS,
        region: CodeRegion | None = None,
    ) -> bool:
        if len(self.diagnostics) >= self.limits.max_diagnostics:
            self.mark_limit(CodeAnalysisLimitReason.DIAGNOSTIC_LIMIT)
            return False
        if region is not None:
            if self.region_count >= self.limits.max_regions:
                self.mark_limit(CodeAnalysisLimitReason.REGION_LIMIT)
                region = None
            else:
                self.region_count += 1
        self.diagnostics.append(
            CodeDiagnostic(
                code=code,
                category=category,
                severity=severity,
                message=message,
                path=path,
                region=region,
            )
        )
        return True

    def add_skipped(
        self,
        path: str,
        reason: CodeSkipReason,
        *,
        language: CodeLanguage | None = None,
        size_bytes: int | None = None,
    ) -> None:
        self.skipped_files.append(
            SkippedCodeFile(
                path=path,
                reason=reason,
                language=language,
                size_bytes=size_bytes,
            )
        )

    def add_symbol(
        self,
        path: str,
        name: str,
        kind: CodeSymbolKind,
        start_line: int,
        end_line: int,
        qualified_name: str | None,
    ) -> bool:
        if len(name) > 256 or (qualified_name is not None and len(qualified_name) > 1_024):
            self.mark_limit(CodeAnalysisLimitReason.SYMBOL_LIMIT)
            return False
        if len(self.symbols) >= self.limits.max_symbols:
            self.mark_limit(CodeAnalysisLimitReason.SYMBOL_LIMIT)
            return False
        if self.region_count >= self.limits.max_regions:
            self.mark_limit(CodeAnalysisLimitReason.REGION_LIMIT)
            return False
        region = CodeRegion(
            path=path,
            start_line=max(1, start_line),
            end_line=max(max(1, start_line), end_line),
        )
        self.symbols.append(
            CodeSymbol(
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                region=region,
            )
        )
        self.region_count += 1
        return True


class CodingAnalysisRuntime:
    """Read-only project analyzer bound to one existing ``Workspace``.

    No provider is constructed or called. Only bounded UTF-8 snapshots are
    opened in binary read mode; parsed text is neither imported nor executed.
    Analysis uses the existing LOW ``coding_analyze`` permission operation.
    """

    name: str = CodingOperation.ANALYZE.value
    permission_level: PermissionLevel = CodingOperation.ANALYZE.required_permission

    def __init__(self, workspace: Workspace, limits: CodingLimits | None = None) -> None:
        self._workspace = workspace
        self._limits = limits or CodingLimits()
        self._diagnostics_engine = CodeDiagnosticsEngine(self._limits)

    @property
    def limits(self) -> CodingLimits:
        return self._limits

    def analyze_project(self, project: CodingProject) -> CodeAnalysisResult:
        """Discover and analyze a bounded workspace-scoped project snapshot."""
        started = time.perf_counter()
        collector = _AnalysisAccumulator(self._limits)
        root = self._resolve_project_root(project)
        if root is None:
            collector.add_diagnostic(
                "project_boundary_invalid",
                "Project path failed workspace validation; no files were read.",
                severity=DiagnosticSeverity.ERROR,
            )
            return self._build_result(project, collector, started, force_partial=True)

        candidates: list[Path] = []
        try:
            for candidate in self._workspace.walk_files(
                root,
                skip_directories=_EXCLUDED_DIRECTORIES,
                max_files=self._limits.max_project_files + 1,
                max_entries=max(256, self._limits.max_project_files * 32),
            ):
                candidates.append(candidate)
        except WorkspaceError as exc:
            if exc.code == "traversal_limit":
                collector.mark_limit(CodeAnalysisLimitReason.DISCOVERY_ENTRY_LIMIT)
                collector.add_diagnostic(
                    "discovery_entry_limit",
                    "Workspace discovery stopped at its bounded directory-entry limit.",
                )
            else:
                collector.add_diagnostic(
                    "project_discovery_failed",
                    "Workspace file discovery failed safely; analysis is incomplete.",
                    severity=DiagnosticSeverity.ERROR,
                )

        if len(candidates) > self._limits.max_project_files:
            collector.mark_limit(CodeAnalysisLimitReason.PROJECT_FILE_LIMIT)
            collector.add_diagnostic(
                "project_file_limit",
                "File discovery stopped at the configured project file limit.",
            )
            candidates = candidates[: self._limits.max_project_files]
        candidates.sort(key=self._workspace.relative_to_root)

        for candidate in candidates:
            if self._deadline_reached(started):
                collector.mark_limit(CodeAnalysisLimitReason.ANALYSIS_TIME_LIMIT)
                collector.add_diagnostic(
                    "analysis_time_limit",
                    "Analysis stopped at the configured cooperative time limit.",
                )
                break
            self._analyze_candidate(candidate, project, collector)

        if time.perf_counter() - started > self._limits.max_analysis_time_s:
            collector.mark_limit(CodeAnalysisLimitReason.ANALYSIS_TIME_LIMIT)
        return self._build_result(project, collector, started)

    def _resolve_project_root(self, project: CodingProject) -> Path | None:
        if _contains_parent_traversal(project.root_path):
            return None
        try:
            relative_root = project.resolve_root(self._workspace)
            root = self._workspace.resolve(relative_root)
        except (CodingWorkspaceError, WorkspaceError):
            return None
        try:
            if not root.is_dir():
                return None
        except OSError:
            return None
        return root

    def _deadline_reached(self, started: float) -> bool:
        return time.perf_counter() - started >= self._limits.max_analysis_time_s

    def _analyze_candidate(
        self,
        candidate: Path,
        project: CodingProject,
        collector: _AnalysisAccumulator,
    ) -> None:
        relative_candidate = self._workspace.relative_to_root(candidate)
        if len(relative_candidate) > MAX_CODING_PATH_CHARS:
            collector.add_diagnostic(
                "path_too_long",
                "A discovered path exceeds the hard coding path bound and was not read.",
                severity=DiagnosticSeverity.ERROR,
            )
            return
        try:
            canonical_relative = project.resolve_file_path(self._workspace, relative_candidate)
            if len(canonical_relative) > MAX_CODING_PATH_CHARS:
                collector.add_diagnostic(
                    "path_too_long",
                    "A resolved path exceeds the hard coding path bound and was not read.",
                    severity=DiagnosticSeverity.ERROR,
                )
                return
            resolved = self._workspace.resolve(canonical_relative)
        except (CodingWorkspaceError, WorkspaceError):
            collector.add_skipped(relative_candidate, CodeSkipReason.UNSAFE_PATH)
            collector.add_diagnostic(
                "unsafe_project_path",
                "A discovered path failed workspace/project validation and was not read.",
                severity=DiagnosticSeverity.ERROR,
            )
            return

        candidate_is_sensitive = _is_sensitive_filename(candidate.name)
        target_is_sensitive = _is_sensitive_filename(resolved.name)
        if candidate_is_sensitive or target_is_sensitive:
            skipped_path = relative_candidate if candidate_is_sensitive else canonical_relative
            try:
                size_bytes = resolved.stat().st_size
            except OSError:
                size_bytes = None
            already_seen = {item.path for item in collector.analyzed_files}
            already_seen.update(item.path for item in collector.skipped_files)
            if skipped_path not in already_seen:
                collector.add_skipped(
                    skipped_path,
                    CodeSkipReason.SENSITIVE_FILE_NAME,
                    size_bytes=size_bytes,
                )
            collector.add_diagnostic(
                "sensitive_file_name",
                "Sensitive-looking file names are excluded without reading content.",
                path=skipped_path,
                severity=DiagnosticSeverity.INFO,
            )
            return

        already_seen = {item.path for item in collector.analyzed_files}
        already_seen.update(item.path for item in collector.skipped_files)
        if canonical_relative in already_seen:
            duplicate_path = relative_candidate
            if duplicate_path not in already_seen:
                collector.add_skipped(duplicate_path, CodeSkipReason.DUPLICATE_PATH)
            else:
                duplicate_path = canonical_relative
            collector.add_diagnostic(
                "duplicate_project_path",
                "A duplicate canonical file path was skipped.",
                path=duplicate_path,
            )
            return

        try:
            file_stat = resolved.stat()
        except OSError:
            collector.add_skipped(canonical_relative, CodeSkipReason.READ_FAILED)
            collector.add_diagnostic(
                "file_stat_failed",
                "File metadata could not be read safely.",
                path=canonical_relative,
            )
            return
        if not stat.S_ISREG(file_stat.st_mode):
            collector.add_skipped(
                canonical_relative,
                CodeSkipReason.NOT_REGULAR_FILE,
                size_bytes=file_stat.st_size,
            )
            collector.add_diagnostic(
                "not_regular_file",
                "Only regular files are included in coding analysis.",
                path=canonical_relative,
            )
            return

        candidate_language = _classify_file(candidate)
        target_language = _classify_file(resolved)
        if candidate_language is None or candidate_language is not target_language:
            skipped_path = canonical_relative
            if relative_candidate != canonical_relative:
                skipped_path = relative_candidate
            collector.add_skipped(
                skipped_path,
                CodeSkipReason.UNSUPPORTED_FILE_TYPE,
                size_bytes=file_stat.st_size,
            )
            collector.add_diagnostic(
                "unsupported_file_type",
                "File type is unsupported or ambiguous; file content was not read.",
                path=skipped_path,
                severity=DiagnosticSeverity.INFO,
            )
            return
        language = candidate_language

        file_limit = min(
            self._limits.max_file_size_bytes,
            max(0, self._workspace.limits.max_read_bytes),
        )
        if file_stat.st_size > file_limit:
            self._skip_limited_file(
                collector,
                canonical_relative,
                CodeSkipReason.FILE_TOO_LARGE,
                CodeAnalysisLimitReason.FILE_SIZE_LIMIT,
                "File exceeds the configured per-file read limit.",
                language,
                file_stat.st_size,
            )
            return

        remaining_chars = self._limits.max_source_chars - collector.total_source_chars
        byte_budget = min(file_limit, remaining_chars * 4 + 3)
        if file_stat.st_size > byte_budget:
            self._skip_limited_file(
                collector,
                canonical_relative,
                CodeSkipReason.SOURCE_LIMIT_EXCEEDED,
                CodeAnalysisLimitReason.SOURCE_CHAR_LIMIT,
                "File exceeds the remaining total source-text budget.",
                language,
                file_stat.st_size,
            )
            return

        try:
            with resolved.open("rb") as handle:
                data = handle.read(byte_budget + 1)
        except OSError:
            collector.add_skipped(
                canonical_relative,
                CodeSkipReason.READ_FAILED,
                language=language,
                size_bytes=file_stat.st_size,
            )
            collector.add_diagnostic(
                "file_read_failed",
                "File content could not be read safely.",
                path=canonical_relative,
            )
            return

        if len(data) > file_limit:
            self._skip_limited_file(
                collector,
                canonical_relative,
                CodeSkipReason.FILE_TOO_LARGE,
                CodeAnalysisLimitReason.FILE_SIZE_LIMIT,
                "File grew beyond the configured per-file read limit.",
                language,
                file_stat.st_size,
            )
            return
        if len(data) > byte_budget:
            self._skip_limited_file(
                collector,
                canonical_relative,
                CodeSkipReason.SOURCE_LIMIT_EXCEEDED,
                CodeAnalysisLimitReason.SOURCE_CHAR_LIMIT,
                "File grew beyond the remaining total source-text budget.",
                language,
                file_stat.st_size,
            )
            return
        if b"\x00" in data:
            collector.add_skipped(
                canonical_relative,
                CodeSkipReason.BINARY_FILE,
                language=language,
                size_bytes=file_stat.st_size,
            )
            collector.add_diagnostic(
                "binary_file",
                "File contains binary NUL data and was skipped.",
                path=canonical_relative,
            )
            return
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            collector.add_skipped(
                canonical_relative,
                CodeSkipReason.DECODE_FAILED,
                language=language,
                size_bytes=file_stat.st_size,
            )
            collector.add_diagnostic(
                "decode_failed",
                "File is not valid UTF-8 text and was skipped.",
                path=canonical_relative,
            )
            return
        if len(text) > remaining_chars:
            self._skip_limited_file(
                collector,
                canonical_relative,
                CodeSkipReason.SOURCE_LIMIT_EXCEEDED,
                CodeAnalysisLimitReason.SOURCE_CHAR_LIMIT,
                "Decoded file exceeds the remaining total source-text budget.",
                language,
                file_stat.st_size,
            )
            return

        source = CodeFile(path=canonical_relative, content=text)
        collector.total_source_chars += len(text)
        collector.source_files.append(source)
        method, metrics = self._extract_structure(
            language,
            canonical_relative,
            text,
            collector,
        )
        remaining_diagnostics = max(
            0,
            self._limits.max_diagnostics - len(collector.diagnostics),
        )
        diagnostic_batch = self._diagnostics_engine.diagnose_file(
            source,
            language,
            max_diagnostics=remaining_diagnostics,
        )
        for diagnostic in diagnostic_batch.diagnostics:
            collector.add_diagnostic(
                diagnostic.code,
                diagnostic.message,
                path=diagnostic.path,
                severity=diagnostic.severity,
                category=diagnostic.category,
                region=diagnostic.region,
            )
        if diagnostic_batch.truncated:
            collector.mark_limit(CodeAnalysisLimitReason.DIAGNOSTIC_LIMIT)
        collector.analyzed_files.append(
            AnalyzedCodeFile(
                path=source.path,
                size_bytes=source.size_bytes,
                source_chars=len(source.content),
                source_sha256=source.source_sha256,
                language=language,
                method=method,
                line_count=_line_count(text),
                metrics=metrics,
            )
        )

    def _skip_limited_file(
        self,
        collector: _AnalysisAccumulator,
        path: str,
        skip_reason: CodeSkipReason,
        limit_reason: CodeAnalysisLimitReason,
        message: str,
        language: CodeLanguage,
        size_bytes: int,
    ) -> None:
        collector.mark_limit(limit_reason)
        collector.add_skipped(
            path,
            skip_reason,
            language=language,
            size_bytes=size_bytes,
        )
        collector.add_diagnostic(
            skip_reason.value,
            message,
            path=path,
        )

    def _extract_structure(
        self,
        language: CodeLanguage,
        path: str,
        text: str,
        collector: _AnalysisAccumulator,
    ) -> tuple[CodeAnalysisMethod, tuple[CodeFileMetric, ...]]:
        if language is CodeLanguage.PYTHON:
            if CodeAnalysisLimitReason.SYMBOL_LIMIT in collector.limit_reasons or (
                CodeAnalysisLimitReason.REGION_LIMIT in collector.limit_reasons
            ):
                return CodeAnalysisMethod.METADATA_ONLY, ()
            try:
                tree = ast.parse(text, filename=path, type_comments=True)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                # ``CodeDiagnosticsEngine`` reports syntax/structural findings
                # from the same bounded snapshot after symbol extraction.
                return CodeAnalysisMethod.PYTHON_AST, ()
            visitor = _PythonStructureVisitor(
                lambda name, kind, start, end, qualified: collector.add_symbol(
                    path,
                    name,
                    kind,
                    start,
                    end,
                    qualified,
                )
            )
            try:
                with contextlib.suppress(_SymbolLimitReached):
                    visitor.visit(tree)
            except (RecursionError, MemoryError):
                collector.add_diagnostic(
                    "python_ast_limit",
                    "Python AST analysis exceeded a safe structural bound.",
                    path=path,
                    category=CodeDiagnosticCategory.LIMIT,
                )
            return CodeAnalysisMethod.PYTHON_AST, ()

        if language in {CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT}:
            if CodeAnalysisLimitReason.SYMBOL_LIMIT in collector.limit_reasons or (
                CodeAnalysisLimitReason.REGION_LIMIT in collector.limit_reasons
            ):
                return CodeAnalysisMethod.METADATA_ONLY, ()
            self._scan_ecmascript(path, text, collector)
            return CodeAnalysisMethod.ECMASCRIPT_LINE_SCAN, ()

        metrics = self._metadata_metrics(language, text, path, collector)
        return CodeAnalysisMethod.METADATA_ONLY, metrics

    def _scan_ecmascript(
        self,
        path: str,
        text: str,
        collector: _AnalysisAccumulator,
    ) -> None:
        for line_number, raw_line in enumerate(io.StringIO(text), start=1):
            line = raw_line.lstrip("\ufeff") if line_number == 1 else raw_line
            match = _IMPORT_FROM_RE.match(line)
            if match is not None:
                names = _import_binding_names(match.group("bindings"))
                if not names:
                    names = (f"import_line_{line_number}",)
                for name in names:
                    if not collector.add_symbol(
                        path,
                        name,
                        CodeSymbolKind.OTHER,
                        line_number,
                        line_number,
                        f"import:{name}",
                    ):
                        return
            elif _SIDE_EFFECT_IMPORT_RE.match(line) is not None:
                name = f"side_effect_import_{line_number}"
                if not collector.add_symbol(
                    path,
                    name,
                    CodeSymbolKind.OTHER,
                    line_number,
                    line_number,
                    name,
                ):
                    return

            function_match = _FUNCTION_RE.match(line)
            class_match = _CLASS_RE.match(line)
            variable_match = _VARIABLE_RE.match(line)
            arrow_match = _ARROW_FUNCTION_RE.match(line)
            type_match = _TS_TYPE_RE.match(line)
            if function_match is not None:
                if not collector.add_symbol(
                    path,
                    function_match.group(1),
                    CodeSymbolKind.FUNCTION,
                    line_number,
                    line_number,
                    function_match.group(1),
                ):
                    return
            elif class_match is not None:
                if not collector.add_symbol(
                    path,
                    class_match.group(1),
                    CodeSymbolKind.CLASS,
                    line_number,
                    line_number,
                    class_match.group(1),
                ):
                    return
            elif type_match is not None:
                if not collector.add_symbol(
                    path,
                    type_match.group(1),
                    CodeSymbolKind.OTHER,
                    line_number,
                    line_number,
                    type_match.group(1),
                ):
                    return
            elif arrow_match is not None:
                if not collector.add_symbol(
                    path,
                    arrow_match.group(1),
                    CodeSymbolKind.FUNCTION,
                    line_number,
                    line_number,
                    arrow_match.group(1),
                ):
                    return
            elif variable_match is not None and not collector.add_symbol(
                path,
                variable_match.group(1),
                CodeSymbolKind.VARIABLE,
                line_number,
                line_number,
                variable_match.group(1),
            ):
                return

            export_match = _EXPORT_LIST_RE.match(line)
            if export_match is not None:
                for name in _exported_names(export_match.group(1)):
                    if not collector.add_symbol(
                        path,
                        name,
                        CodeSymbolKind.OTHER,
                        line_number,
                        line_number,
                        f"export:{name}",
                    ):
                        return

    def _metadata_metrics(
        self,
        language: CodeLanguage,
        text: str,
        path: str,
        collector: _AnalysisAccumulator,
    ) -> tuple[CodeFileMetric, ...]:
        metric: CodeFileMetric | None = None
        try:
            if language is CodeLanguage.MARKDOWN:
                metric = CodeFileMetric(
                    name="heading_count",
                    value=sum(1 for _ in _MARKDOWN_HEADING_RE.finditer(text)),
                )
            elif language is CodeLanguage.YAML:
                metric = CodeFileMetric(
                    name="mapping_line_count",
                    value=sum(1 for _ in _YAML_MAPPING_RE.finditer(text)),
                )
            elif language is CodeLanguage.JSON:
                value = json.loads(text.removeprefix("\ufeff"))
                count = len(value) if isinstance(value, (dict, list)) else 1
                metric = CodeFileMetric(name="top_level_item_count", value=count)
            elif language is CodeLanguage.TOML:
                value = tomllib.loads(text.removeprefix("\ufeff"))
                metric = CodeFileMetric(name="top_level_item_count", value=len(value))
        except (
            json.JSONDecodeError,
            tomllib.TOMLDecodeError,
            RecursionError,
            MemoryError,
            ValueError,
        ):
            collector.add_diagnostic(
                f"malformed_{language.value}",
                (
                    f"{language.value.upper()} structure could not be parsed; "
                    "content remains data only."
                ),
                path=path,
            )
        return (metric,) if metric is not None else ()

    def _build_result(
        self,
        project: CodingProject,
        collector: _AnalysisAccumulator,
        started: float,
        *,
        force_partial: bool = False,
    ) -> CodeAnalysisResult:
        actual_elapsed = max(0.0, time.perf_counter() - started)
        if actual_elapsed > self._limits.max_analysis_time_s:
            collector.mark_limit(CodeAnalysisLimitReason.ANALYSIS_TIME_LIMIT)
        # Time checks are cooperative; report a bounded elapsed value and carry
        # the explicit truncation reason if a synchronous parse overran the cap.
        elapsed = min(actual_elapsed, self._limits.max_analysis_time_s)

        def make_result() -> CodeAnalysisResult:
            partial = force_partial or bool(
                collector.diagnostics or collector.skipped_files or collector.truncated
            )
            status = CodeAnalysisStatus.PARTIAL if partial else CodeAnalysisStatus.COMPLETE
            summary = (
                f"Partial local analysis: {len(collector.analyzed_files)} file(s) analyzed, "
                f"{len(collector.skipped_files)} skipped."
                if partial
                else f"Completed local analysis of {len(collector.analyzed_files)} file(s)."
            )
            return CodeAnalysisResult(
                project_id=project.project_id,
                status=status,
                summary=summary,
                symbols=tuple(collector.symbols),
                diagnostics=tuple(collector.diagnostics),
                elapsed_time_s=elapsed,
                observation=CodingObservation.from_files(project, collector.source_files),
                analyzed_files=tuple(collector.analyzed_files),
                skipped_files=tuple(collector.skipped_files),
                truncated=collector.truncated,
                limit_reasons=tuple(collector.limit_reasons),
            )

        result = make_result()
        while len(result.model_dump_json().encode("utf-8")) > self._limits.max_output_bytes:
            collector.mark_limit(CodeAnalysisLimitReason.OUTPUT_LIMIT)
            if collector.symbols:
                collector.symbols.pop()
            elif collector.skipped_files:
                collector.skipped_files.pop()
            elif collector.diagnostics:
                collector.diagnostics.pop()
            elif collector.analyzed_files:
                removed = collector.analyzed_files.pop()
                for index in range(len(collector.source_files) - 1, -1, -1):
                    if collector.source_files[index].path == removed.path:
                        collector.source_files.pop(index)
                        break
            else:
                raise CodingLimitError()
            result = make_result()

        self._limits.validate_analysis_result(result)
        return result


def _contains_parent_traversal(path: str) -> bool:
    """Reject explicit parent segments before Workspace canonicalization."""
    return any(part == ".." for part in re.split(r"[\\/]", path))


def _is_sensitive_filename(name: str) -> bool:
    folded = name.casefold()
    if folded == ".env" or (folded.startswith(".env.") and folded not in _SAFE_ENV_EXAMPLES):
        return True
    if Path(folded).suffix in _SENSITIVE_SUFFIXES:
        return True
    return _SENSITIVE_NAME_RE.search(folded) is not None


def _classify_file(path: Path) -> CodeLanguage | None:
    name = path.name.casefold()
    if name in _SAFE_ENV_EXAMPLES:
        return CodeLanguage.TEXT
    if name in _TEXT_BASENAMES:
        return CodeLanguage.TEXT
    return _LANGUAGE_BY_EXTENSION.get(path.suffix.casefold())


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _import_binding_names(bindings: str) -> tuple[str, ...]:
    value = bindings.strip()
    names: list[str] = []
    brace_start = value.find("{")
    brace_end = value.find("}", brace_start + 1) if brace_start >= 0 else -1
    if brace_start >= 0 and brace_end > brace_start:
        default_binding = value[:brace_start].strip().rstrip(",").split()
        if default_binding and re.fullmatch(_IDENTIFIER, default_binding[0]):
            names.append(default_binding[0])
        inside = value[brace_start + 1 : brace_end]
        for specifier in inside.split(","):
            tokens = specifier.strip().split()
            if tokens and tokens[0] == "type":
                tokens = tokens[1:]
            if not tokens:
                continue
            if "as" in tokens and tokens.index("as") + 1 < len(tokens):
                candidate = tokens[tokens.index("as") + 1]
            else:
                candidate = tokens[0]
            if re.fullmatch(_IDENTIFIER, candidate):
                names.append(candidate)
        return tuple(names)

    tokens = value.split()
    if (
        len(tokens) >= 3
        and tokens[0] == "*"
        and tokens[1] == "as"
        and re.fullmatch(_IDENTIFIER, tokens[2])
    ):
        return (tokens[2],)
    if tokens and re.fullmatch(_IDENTIFIER, tokens[0]):
        return (tokens[0],)
    return ()


def _exported_names(bindings: str) -> tuple[str, ...]:
    names: list[str] = []
    for specifier in bindings.split(","):
        tokens = specifier.strip().split()
        if not tokens:
            continue
        if "as" in tokens and tokens.index("as") + 1 < len(tokens):
            candidate = tokens[tokens.index("as") + 1]
        else:
            candidate = tokens[0]
            if candidate == "type" and len(tokens) > 1:
                candidate = tokens[1]
        if re.fullmatch(_IDENTIFIER, candidate):
            names.append(candidate)
    return tuple(names)
