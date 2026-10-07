"""Bounded provider-neutral models for the Phase 10 coding foundation.

Source and provider text are untrusted data. These models describe bounded
snapshots and proposals only; they do not read, write, execute, or validate
source code. Use :class:`CodingProject` with the existing :class:`Workspace`
before passing paths to a provider.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..workspace import Workspace, WorkspaceError
from .errors import CodingWorkspaceError

MAX_CODING_PATH_CHARS = 4_096
MAX_CODING_PROJECT_FILES = 500
MAX_CODING_FILE_SIZE_BYTES = 4 * 1_024 * 1_024
MAX_CODING_FILE_CHARS = 2_000_000
MAX_CODING_SOURCE_CHARS = 2_000_000
MAX_CODING_PATCH_SIZE_BYTES = 2 * 1_024 * 1_024
MAX_CODING_CHANGED_FILES = 100
MAX_CODING_SYMBOLS = 2_000
MAX_CODING_REGIONS = 1_000
MAX_CODING_DIAGNOSTICS = 2_000
MAX_CODING_TEST_CASES = 500
MAX_CODING_ANALYSIS_TIME_S = 120.0
MAX_CODING_TEST_DURATION_S = 3_600.0
MAX_CODING_OUTPUT_BYTES = 2 * 1_024 * 1_024
MAX_CODING_OUTPUT_TEXT_CHARS = 32_768
MAX_CODING_REQUEST_CHARS = 8_000

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_PROJECT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"


def _validate_utf8_text(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("coding text must be valid UTF-8") from exc


class CodingModel(BaseModel):
    """Strict immutable base for public coding-domain records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )

    @model_validator(mode="after")
    def validate_utf8_text_fields(self) -> CodingModel:
        for value in self.__dict__.values():
            if isinstance(value, str):
                _validate_utf8_text(value)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, str):
                        _validate_utf8_text(item)
        return self


class CodingProject(CodingModel):
    """A logical project rooted at a workspace-relative directory.

    The path is never treated as a host path. Resolution, symlink/junction
    handling, and workspace containment are delegated to ``Workspace``.
    """

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    root_path: str = Field(max_length=MAX_CODING_PATH_CHARS, strict=True)
    display_name: str | None = Field(default=None, max_length=256, strict=True)

    def resolve_root(self, workspace: Workspace) -> str:
        """Return the canonical workspace-relative project root, never a host path."""
        root = self._resolve_root(workspace)
        return workspace.relative_to_root(root)

    def resolve_project_paths(self, workspace: Workspace, paths: Iterable[str]) -> tuple[str, ...]:
        """Resolve unique workspace-relative files that stay inside this project.

        ``Workspace.resolve`` remains authoritative for absolute paths,
        traversal, symlinks/junctions, and workspace-root containment. This
        method adds only the logical project-root check after canonicalization
        and returns normalized relative paths, never absolute host paths. It
        does not inspect file contents or perform filesystem mutations.
        """
        root = self._resolve_root(workspace)
        resolved_paths: list[str] = []
        seen: set[Path] = set()
        for raw_path in paths:
            resolved = self._resolve_project_path(workspace, root, raw_path)
            if resolved in seen:
                raise CodingWorkspaceError()
            resolved_paths.append(workspace.relative_to_root(resolved))
            seen.add(resolved)
        return tuple(resolved_paths)

    def resolve_file_path(self, workspace: Workspace, path: str) -> str:
        """Return one canonical workspace-relative path contained in-project."""
        root = self._resolve_root(workspace)
        resolved = self._resolve_project_path(workspace, root, path)
        return workspace.relative_to_root(resolved)

    @staticmethod
    def _resolve_project_path(workspace: Workspace, root: Path, path: str) -> Path:
        try:
            resolved = workspace.resolve(path)
        except WorkspaceError:
            raise CodingWorkspaceError() from None
        if resolved == root or root not in resolved.parents:
            raise CodingWorkspaceError()
        return resolved

    def _resolve_root(self, workspace: Workspace) -> Path:
        try:
            return workspace.resolve(self.root_path)
        except WorkspaceError:
            raise CodingWorkspaceError() from None


class CodeFile(CodingModel):
    """One bounded UTF-8 source snapshot; ``content`` is untrusted data.

    ``path`` is workspace-relative and must be resolved through its project.
    """

    path: str = Field(min_length=1, max_length=MAX_CODING_PATH_CHARS, strict=True)
    content: str = Field(max_length=MAX_CODING_FILE_CHARS, strict=True, repr=False)

    @model_validator(mode="after")
    def validate_utf8_size(self) -> CodeFile:
        try:
            size = len(self.content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("source text is not valid UTF-8 text") from exc
        if size > MAX_CODING_FILE_SIZE_BYTES:
            raise ValueError("source file exceeds the hard byte limit")
        return self

    @property
    def size_bytes(self) -> int:
        """UTF-8 byte length, computed from the in-memory source snapshot."""
        return len(self.content.encode("utf-8"))

    @property
    def source_sha256(self) -> str:
        """Stable reference for the supplied snapshot, not a trust verdict."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class CodeRegion(CodingModel):
    """A bounded one-based source span in a workspace-relative file."""

    path: str = Field(min_length=1, max_length=MAX_CODING_PATH_CHARS, strict=True)
    start_line: int = Field(ge=1, le=MAX_CODING_FILE_CHARS, strict=True)
    end_line: int = Field(ge=1, le=MAX_CODING_FILE_CHARS, strict=True)
    start_column: int | None = Field(default=None, ge=1, le=16_384, strict=True)
    end_column: int | None = Field(default=None, ge=1, le=16_384, strict=True)

    @model_validator(mode="after")
    def validate_span_order(self) -> CodeRegion:
        if self.end_line < self.start_line:
            raise ValueError("region end line precedes its start line")
        if (
            self.end_line == self.start_line
            and self.start_column is not None
            and self.end_column is not None
            and self.end_column < self.start_column
        ):
            raise ValueError("region end column precedes its start column")
        return self


class CodeSymbolKind(StrEnum):
    """Provider-neutral symbol categories; no language parser is implied."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    OTHER = "other"


class CodeSymbol(CodingModel):
    """A bounded symbol reference and its source region."""

    name: str = Field(min_length=1, max_length=256, strict=True)
    qualified_name: str | None = Field(default=None, max_length=1_024, strict=True)
    kind: CodeSymbolKind
    region: CodeRegion
    signature: str | None = Field(default=None, max_length=4_096, strict=True, repr=False)


class DiagnosticSeverity(StrEnum):
    """Severity labels for deterministic or provider-produced diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CodeDiagnosticCategory(StrEnum):
    """Provider-neutral category labels for deterministic diagnostic rules."""

    ANALYSIS = "analysis"
    SYNTAX = "syntax"
    STYLE = "style"
    LIMIT = "limit"


class CodeDiagnostic(CodingModel):
    """A bounded diagnostic; its message remains untrusted output."""

    code: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN, strict=True)
    category: CodeDiagnosticCategory = CodeDiagnosticCategory.ANALYSIS
    severity: DiagnosticSeverity
    message: str = Field(min_length=1, max_length=4_096, strict=True, repr=False)
    path: str | None = Field(default=None, max_length=MAX_CODING_PATH_CHARS, strict=True)
    region: CodeRegion | None = None

    @model_validator(mode="after")
    def validate_path_region(self) -> CodeDiagnostic:
        if self.path is not None and self.region is not None and self.path != self.region.path:
            raise ValueError("diagnostic path and region path do not match")
        return self


class ObservedCodeFile(CodingModel):
    """Content-free metadata for one source snapshot."""

    path: str = Field(min_length=1, max_length=MAX_CODING_PATH_CHARS, strict=True)
    size_bytes: int = Field(ge=0, le=MAX_CODING_FILE_SIZE_BYTES, strict=True)
    source_chars: int = Field(ge=0, le=MAX_CODING_FILE_CHARS, strict=True)
    source_sha256: str = Field(pattern=_SHA256_PATTERN, strict=True)

    @classmethod
    def from_file(cls, source: CodeFile) -> ObservedCodeFile:
        """Project a code snapshot to bounded, content-free metadata."""
        return cls(
            path=source.path,
            size_bytes=source.size_bytes,
            source_chars=len(source.content),
            source_sha256=source.source_sha256,
        )


class CodeLanguage(StrEnum):
    """Deterministic file classifications supported by local analysis."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    HTML = "html"
    CSS = "css"
    TEXT = "text"


class CodeAnalysisMethod(StrEnum):
    """The shallow, non-executing method used for one file."""

    PYTHON_AST = "python_ast"
    ECMASCRIPT_LINE_SCAN = "ecmascript_line_scan"
    METADATA_ONLY = "metadata_only"


class CodeSkipReason(StrEnum):
    """Stable reasons a discovered file was not included as source text."""

    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    SENSITIVE_FILE_NAME = "sensitive_file_name"
    FILE_TOO_LARGE = "file_too_large"
    SOURCE_LIMIT_EXCEEDED = "source_limit_exceeded"
    DECODE_FAILED = "decode_failed"
    BINARY_FILE = "binary_file"
    UNSAFE_PATH = "unsafe_path"
    NOT_REGULAR_FILE = "not_regular_file"
    READ_FAILED = "read_failed"
    DUPLICATE_PATH = "duplicate_path"


class CodeAnalysisLimitReason(StrEnum):
    """Explicit limits that caused deterministic analysis truncation."""

    PROJECT_FILE_LIMIT = "project_file_limit"
    DISCOVERY_ENTRY_LIMIT = "discovery_entry_limit"
    FILE_SIZE_LIMIT = "file_size_limit"
    SOURCE_CHAR_LIMIT = "source_char_limit"
    SYMBOL_LIMIT = "symbol_limit"
    REGION_LIMIT = "region_limit"
    DIAGNOSTIC_LIMIT = "diagnostic_limit"
    ANALYSIS_TIME_LIMIT = "analysis_time_limit"
    OUTPUT_LIMIT = "output_limit"


class CodeFileMetric(CodingModel):
    """Small numeric metadata; metric names come from runtime constants."""

    name: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN, strict=True)
    value: int = Field(ge=0, le=MAX_CODING_FILE_CHARS, strict=True)


class AnalyzedCodeFile(ObservedCodeFile):
    """Content-free classification and counts for one analyzed source file."""

    language: CodeLanguage
    method: CodeAnalysisMethod
    line_count: int = Field(ge=0, le=MAX_CODING_FILE_CHARS, strict=True)
    metrics: tuple[CodeFileMetric, ...] = Field(default=(), max_length=8)


class SkippedCodeFile(CodingModel):
    """Bounded path/size metadata for a discovered file that was not read."""

    path: str = Field(min_length=1, max_length=MAX_CODING_PATH_CHARS, strict=True)
    reason: CodeSkipReason
    language: CodeLanguage | None = None
    size_bytes: int | None = Field(default=None, ge=0, le=(1 << 63) - 1, strict=True)


class CodingObservation(CodingModel):
    """A metadata-only view suitable for safe operational reporting."""

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    files: tuple[ObservedCodeFile, ...] = Field(max_length=MAX_CODING_PROJECT_FILES)
    total_source_chars: int = Field(ge=0, le=MAX_CODING_SOURCE_CHARS, strict=True)

    @model_validator(mode="after")
    def validate_file_metadata(self) -> CodingObservation:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("observation contains duplicate file paths")
        if sum(item.source_chars for item in self.files) != self.total_source_chars:
            raise ValueError("observation source character count does not match its files")
        return self

    def validate_workspace(self, project: CodingProject, workspace: Workspace) -> tuple[str, ...]:
        """Validate observed paths and return canonical workspace-relative paths."""
        if self.project_id != project.project_id:
            raise CodingWorkspaceError()
        return project.resolve_project_paths(workspace, (item.path for item in self.files))

    @classmethod
    def from_files(
        cls,
        project: CodingProject,
        files: Sequence[CodeFile],
    ) -> CodingObservation:
        """Build a deterministic observation without copying source contents."""
        ordered = sorted(files, key=lambda item: item.path)
        observed = tuple(ObservedCodeFile.from_file(item) for item in ordered)
        return cls(
            project_id=project.project_id,
            files=observed,
            total_source_chars=sum(item.source_chars for item in observed),
        )


class CodeAnalysisRequest(CodingModel):
    """Analysis inputs are bounded snapshots; source is always untrusted data."""

    project: CodingProject
    files: tuple[CodeFile, ...] = Field(min_length=1, max_length=MAX_CODING_PROJECT_FILES)
    question: str = Field(
        min_length=1,
        max_length=MAX_CODING_REQUEST_CHARS,
        strict=True,
        repr=False,
    )
    regions: tuple[CodeRegion, ...] = Field(default=(), max_length=MAX_CODING_REGIONS)

    @model_validator(mode="after")
    def validate_file_references(self) -> CodeAnalysisRequest:
        _validate_source_references(self.files, self.regions)
        return self

    def validate_workspace(self, workspace: Workspace) -> tuple[str, ...]:
        """Validate all supplied paths with ``Workspace``; return normalized files."""
        return _validate_workspace_references(self.project, self.files, self.regions, workspace)


class CodeAnalysisStatus(StrEnum):
    """Explicitly distinguishes complete from partial provider observations."""

    COMPLETE = "complete"
    PARTIAL = "partial"


class CodeAnalysisResult(CodingModel):
    """Bounded deterministic or provider-produced analysis facts."""

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    status: CodeAnalysisStatus
    summary: str = Field(max_length=MAX_CODING_OUTPUT_TEXT_CHARS, strict=True, repr=False)
    symbols: tuple[CodeSymbol, ...] = Field(max_length=MAX_CODING_SYMBOLS)
    diagnostics: tuple[CodeDiagnostic, ...] = Field(max_length=MAX_CODING_DIAGNOSTICS)
    elapsed_time_s: float = Field(
        ge=0.0,
        le=MAX_CODING_ANALYSIS_TIME_S,
        allow_inf_nan=False,
        strict=True,
    )
    observation: CodingObservation
    analyzed_files: tuple[AnalyzedCodeFile, ...] = Field(
        default=(),
        max_length=MAX_CODING_PROJECT_FILES,
    )
    skipped_files: tuple[SkippedCodeFile, ...] = Field(
        default=(),
        max_length=MAX_CODING_PROJECT_FILES,
    )
    truncated: bool = False
    limit_reasons: tuple[CodeAnalysisLimitReason, ...] = Field(default=(), max_length=9)

    @model_validator(mode="after")
    def validate_result_shape(self) -> CodeAnalysisResult:
        if self.observation.project_id != self.project_id:
            raise ValueError("analysis observation belongs to a different project")
        analyzed_paths = [item.path for item in self.analyzed_files]
        skipped_paths = [item.path for item in self.skipped_files]
        if len(analyzed_paths) != len(set(analyzed_paths)):
            raise ValueError("analysis contains duplicate analyzed file paths")
        if len(skipped_paths) != len(set(skipped_paths)):
            raise ValueError("analysis contains duplicate skipped file paths")
        if set(analyzed_paths) & set(skipped_paths):
            raise ValueError("a file cannot be both analyzed and skipped")
        if len(analyzed_paths) + len(skipped_paths) > MAX_CODING_PROJECT_FILES:
            raise ValueError("analysis file metadata exceeds the hard project file limit")
        if self.analyzed_files:
            observed = {item.path: item for item in self.observation.files}
            if set(observed) != set(analyzed_paths):
                raise ValueError("analyzed file metadata does not match the observation")
            for item in self.analyzed_files:
                snapshot = observed[item.path]
                if (
                    item.size_bytes != snapshot.size_bytes
                    or item.source_chars != snapshot.source_chars
                    or item.source_sha256 != snapshot.source_sha256
                ):
                    raise ValueError("analyzed file metadata does not match its observation")
        if len(self.limit_reasons) != len(set(self.limit_reasons)):
            raise ValueError("analysis contains duplicate limit reasons")
        if self.truncated != bool(self.limit_reasons):
            raise ValueError("truncation flag and limit reasons do not match")
        if self.status is CodeAnalysisStatus.COMPLETE and (self.skipped_files or self.truncated):
            raise ValueError("a partial analysis cannot be marked complete")
        if self.truncated and self.status is not CodeAnalysisStatus.PARTIAL:
            raise ValueError("a truncated analysis must be marked partial")
        return self

    def validate_workspace(self, project: CodingProject, workspace: Workspace) -> None:
        """Recheck returned paths against the canonical workspace/project boundary."""
        if project.project_id != self.project_id:
            raise CodingWorkspaceError()
        observed_paths = set(self.observation.validate_workspace(project, workspace))
        safe_skipped_paths: set[str] = set()
        unsafe_skipped_paths = {
            item.path for item in self.skipped_files if item.reason is CodeSkipReason.UNSAFE_PATH
        }
        for skipped_file in self.skipped_files:
            if skipped_file.reason is CodeSkipReason.UNSAFE_PATH:
                continue
            resolved = project.resolve_file_path(workspace, skipped_file.path)
            safe_skipped_paths.add(resolved)
        for analyzed_file in self.analyzed_files:
            resolved = project.resolve_file_path(workspace, analyzed_file.path)
            if resolved not in observed_paths:
                raise CodingWorkspaceError()
        for symbol in self.symbols:
            resolved = project.resolve_file_path(workspace, symbol.region.path)
            if resolved not in observed_paths:
                raise CodingWorkspaceError()
        for diagnostic in self.diagnostics:
            if diagnostic.path is not None and diagnostic.path not in unsafe_skipped_paths:
                resolved = project.resolve_file_path(workspace, diagnostic.path)
                if resolved not in observed_paths | safe_skipped_paths:
                    raise CodingWorkspaceError()
            if diagnostic.region is not None:
                resolved = project.resolve_file_path(workspace, diagnostic.region.path)
                if resolved not in observed_paths:
                    raise CodingWorkspaceError()


class CodeEditRequest(CodingModel):
    """Request a structured proposal only; it does not authorize file writes."""

    project: CodingProject
    files: tuple[CodeFile, ...] = Field(min_length=1, max_length=MAX_CODING_PROJECT_FILES)
    objective: str = Field(
        min_length=1,
        max_length=MAX_CODING_REQUEST_CHARS,
        strict=True,
        repr=False,
    )
    regions: tuple[CodeRegion, ...] = Field(default=(), max_length=MAX_CODING_REGIONS)

    @model_validator(mode="after")
    def validate_file_references(self) -> CodeEditRequest:
        _validate_source_references(self.files, self.regions)
        return self

    def validate_workspace(self, workspace: Workspace) -> tuple[str, ...]:
        """Validate all supplied paths with ``Workspace``; return normalized files."""
        return _validate_workspace_references(self.project, self.files, self.regions, workspace)


class PatchValidationStatus(StrEnum):
    """Advisory validation state; provider claims require core revalidation."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class PatchCheckStatus(StrEnum):
    """Status of one named, non-executable structural validation check."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


class PatchValidationCheck(CodingModel):
    """A bounded check label without command, output, or source-code fields."""

    name: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN, strict=True)
    status: PatchCheckStatus


class PatchValidationMetadata(CodingModel):
    """Advisory provider metadata; it is not proof that a patch is safe."""

    status: PatchValidationStatus = PatchValidationStatus.PENDING
    checks: tuple[PatchValidationCheck, ...] = Field(default=(), max_length=32)


class CodeChange(CodingModel):
    """Full-file replacement with a workspace-relative target and hash precondition."""

    target_path: str = Field(min_length=1, max_length=MAX_CODING_PATH_CHARS, strict=True)
    original_sha256: str = Field(pattern=_SHA256_PATTERN, strict=True)
    original_size_bytes: int = Field(ge=0, le=MAX_CODING_FILE_SIZE_BYTES, strict=True)
    replacement_content: str = Field(
        max_length=MAX_CODING_FILE_CHARS,
        strict=True,
        repr=False,
    )
    validation: PatchValidationMetadata = Field(default_factory=PatchValidationMetadata)

    @model_validator(mode="after")
    def validate_replacement_size(self) -> CodeChange:
        try:
            size = len(self.replacement_content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("replacement text is not valid UTF-8 text") from exc
        if size > MAX_CODING_FILE_SIZE_BYTES:
            raise ValueError("replacement file exceeds the hard byte limit")
        return self

    @property
    def replacement_size_bytes(self) -> int:
        return len(self.replacement_content.encode("utf-8"))

    @property
    def replacement_sha256(self) -> str:
        return hashlib.sha256(self.replacement_content.encode("utf-8")).hexdigest()


class CodePatch(CodingModel):
    """A bounded set of structured full-file replacements; never applied here."""

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    summary: str = Field(max_length=MAX_CODING_OUTPUT_TEXT_CHARS, strict=True, repr=False)
    changes: tuple[CodeChange, ...] = Field(
        min_length=1,
        max_length=MAX_CODING_CHANGED_FILES,
    )

    @model_validator(mode="after")
    def validate_patch_shape(self) -> CodePatch:
        paths = [change.target_path for change in self.changes]
        if len(paths) != len(set(paths)):
            raise ValueError("patch contains duplicate target paths")
        if self.patch_size_bytes > MAX_CODING_PATCH_SIZE_BYTES:
            raise ValueError("patch exceeds the hard byte limit")
        return self

    @property
    def patch_size_bytes(self) -> int:
        """UTF-8 byte size of the complete serialized structural patch."""
        return len(self.model_dump_json().encode("utf-8"))

    @property
    def patch_id(self) -> str:
        """Stable opaque identifier derived from structure and content hashes."""
        digest = hashlib.sha256()
        digest.update(self.project_id.encode("utf-8"))
        for change in self.changes:
            digest.update(change.target_path.encode("utf-8"))
            digest.update(change.original_sha256.encode("ascii"))
            digest.update(change.replacement_sha256.encode("ascii"))
        return f"patch-{digest.hexdigest()[:20]}"

    def resolve_target_paths(self, project: CodingProject, workspace: Workspace) -> tuple[str, ...]:
        """Validate proposed targets with the same workspace/project boundary."""
        if project.project_id != self.project_id:
            raise CodingWorkspaceError()
        return project.resolve_project_paths(
            workspace,
            (change.target_path for change in self.changes),
        )


class CodeEditStatus(StrEnum):
    """An edit proposal is distinct from applying any change."""

    PROPOSED = "proposed"
    NO_CHANGES = "no_changes"
    UNAVAILABLE = "unavailable"


class CodeEditResult(CodingModel):
    """Proposal-only result; no write or apply operation is represented."""

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    status: CodeEditStatus
    summary: str = Field(max_length=MAX_CODING_OUTPUT_TEXT_CHARS, strict=True, repr=False)
    patch: CodePatch | None = None

    @model_validator(mode="after")
    def validate_patch_status(self) -> CodeEditResult:
        if (self.status is CodeEditStatus.PROPOSED) != (self.patch is not None):
            raise ValueError("proposed edit status and patch presence do not match")
        if self.patch is not None and self.patch.project_id != self.project_id:
            raise ValueError("edit patch belongs to a different project")
        return self

    def validate_workspace(self, project: CodingProject, workspace: Workspace) -> tuple[str, ...]:
        """Recheck proposed targets with the existing workspace boundary."""
        if project.project_id != self.project_id:
            raise CodingWorkspaceError()
        if self.patch is None:
            return ()
        return self.patch.resolve_target_paths(project, workspace)


class CodeTestPlanRequest(CodingModel):
    """Inputs for planning tests; no test command or output is accepted."""

    project: CodingProject
    files: tuple[CodeFile, ...] = Field(min_length=1, max_length=MAX_CODING_PROJECT_FILES)
    target_paths: tuple[str, ...] = Field(min_length=1, max_length=MAX_CODING_CHANGED_FILES)
    objective: str = Field(
        min_length=1,
        max_length=MAX_CODING_REQUEST_CHARS,
        strict=True,
        repr=False,
    )

    @model_validator(mode="after")
    def validate_target_references(self) -> CodeTestPlanRequest:
        _validate_source_references(self.files, ())
        target_paths = list(self.target_paths)
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("test plan contains duplicate target paths")
        file_paths = {item.path for item in self.files}
        if any(path not in file_paths for path in self.target_paths):
            raise ValueError("test plan target must refer to a supplied source file")
        return self

    def validate_workspace(self, workspace: Workspace) -> tuple[str, ...]:
        """Validate source/target paths and return canonical target paths."""
        resolved_files = set(
            self.project.resolve_project_paths(workspace, (source.path for source in self.files))
        )
        resolved_targets = tuple(
            self.project.resolve_file_path(workspace, path) for path in self.target_paths
        )
        if len(resolved_targets) != len(set(resolved_targets)) or any(
            path not in resolved_files for path in resolved_targets
        ):
            raise CodingWorkspaceError()
        return resolved_targets


class TestCasePlan(CodingModel):
    """A non-executable test-design item; it has no command or runner field."""

    target_paths: tuple[str, ...] = Field(min_length=1, max_length=MAX_CODING_CHANGED_FILES)
    purpose: str = Field(min_length=1, max_length=4_096, strict=True, repr=False)
    expected_behavior: str = Field(min_length=1, max_length=4_096, strict=True, repr=False)


class CodeTestPlanStatus(StrEnum):
    """Planning outcome only; there is deliberately no executed/failed status."""

    PLANNED = "planned"
    UNAVAILABLE = "unavailable"


class CodeTestPlanResult(CodingModel):
    """A bounded plan; ``execution_performed`` can never be true in Step 1."""

    project_id: str = Field(min_length=1, max_length=64, pattern=_PROJECT_ID_PATTERN, strict=True)
    status: CodeTestPlanStatus
    test_cases: tuple[TestCasePlan, ...] = Field(max_length=MAX_CODING_TEST_CASES)
    estimated_duration_s: float = Field(
        ge=0.0,
        le=MAX_CODING_TEST_DURATION_S,
        allow_inf_nan=False,
        strict=True,
    )
    summary: str = Field(max_length=MAX_CODING_OUTPUT_TEXT_CHARS, strict=True, repr=False)
    execution_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan_status(self) -> CodeTestPlanResult:
        if self.status is CodeTestPlanStatus.PLANNED and not self.test_cases:
            raise ValueError("a planned test result must contain at least one test case")
        if self.status is CodeTestPlanStatus.UNAVAILABLE and self.test_cases:
            raise ValueError("an unavailable test plan cannot contain test cases")
        return self

    def validate_workspace(self, project: CodingProject, workspace: Workspace) -> tuple[str, ...]:
        """Recheck all provider-returned test-plan paths; no tests are run."""
        if project.project_id != self.project_id:
            raise CodingWorkspaceError()
        return tuple(
            project.resolve_file_path(workspace, path)
            for case in self.test_cases
            for path in case.target_paths
        )


def _validate_source_references(
    files: Sequence[CodeFile],
    regions: Sequence[CodeRegion],
) -> None:
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("request contains duplicate source file paths")
    if sum(len(item.content) for item in files) > MAX_CODING_SOURCE_CHARS:
        raise ValueError("request exceeds the hard total source character limit")
    file_paths = set(paths)
    if any(region.path not in file_paths for region in regions):
        raise ValueError("source region must refer to a supplied file")


def _validate_workspace_references(
    project: CodingProject,
    files: Sequence[CodeFile],
    regions: Sequence[CodeRegion],
    workspace: Workspace,
) -> tuple[str, ...]:
    resolved_files = project.resolve_project_paths(workspace, (source.path for source in files))
    allowed_paths = set(resolved_files)
    for region in regions:
        if project.resolve_file_path(workspace, region.path) not in allowed_paths:
            raise CodingWorkspaceError()
    return resolved_files
