"""Explicit coding tools exposed through the normal Tool Runtime."""

from __future__ import annotations

from typing import Any

from ..permissions import PermissionDecision, PermissionLevel, _active_permission_authorization
from ..tools import ToolRegistry, ToolResult, ToolSpec
from ..workspace import Workspace
from .editing import CodePatchApplier
from .limits import CodingLimits
from .models import CodePatch, CodingProject
from .runtime import CodingAnalysisRuntime
from .search import CodeSearchRuntime

ANALYZE = "coding_analyze_project"
SEARCH_TEXT = "coding_search_text"
SEARCH_SYMBOLS = "coding_search_symbols"
SEARCH_FILES = "coding_search_files"
DEFINITIONS = "coding_find_definitions"
APPLY_PATCH = "coding_apply_patch"


def _project(input: dict[str, Any]) -> CodingProject:
    return CodingProject(project_id=str(input["project_id"]), root_path=str(input["project_root"]))


class CodingAnalyzeTool:
    spec = ToolSpec(
        name=ANALYZE,
        description="Analyze source files in a bounded workspace project without executing them.",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}, "project_root": {"type": "string"}},
            "required": ["project_id", "project_root"],
        },
        output_schema={"type": "object"},
        permission_level=PermissionLevel.LOW,
        deterministic=True,
    )

    def __init__(self, workspace: Workspace, limits: CodingLimits) -> None:
        self.runtime = CodingAnalysisRuntime(workspace, limits)

    def run(self, input: dict[str, Any]) -> ToolResult:
        project = _project(input)
        result = self.runtime.analyze_project(project)
        return ToolResult(ok=True, output=result.model_dump(mode="json"))


class _SearchTool:
    operation = SEARCH_TEXT
    description = "Search source text in a bounded workspace project."
    method = "text_search"

    def __init__(self, workspace: Workspace, limits: CodingLimits) -> None:
        self.runtime = CodeSearchRuntime(workspace, limits)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.operation,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "project_root": {"type": "string"},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["project_id", "project_root", "query"],
            },
            output_schema={"type": "object"},
            permission_level=PermissionLevel.LOW,
            deterministic=True,
        )

    def run(self, input: dict[str, Any]) -> ToolResult:
        project = _project(input)
        result = getattr(self.runtime, self.method)(
            project, str(input["query"]), max_results=int(input.get("max_results", 50))
        )
        return ToolResult(ok=True, output=result.model_dump(mode="json"))


class CodingSymbolSearchTool(_SearchTool):
    operation = SEARCH_SYMBOLS
    description = "Search symbol names in a bounded workspace project."
    method = "symbol_search"


class CodingFileSearchTool(_SearchTool):
    operation = SEARCH_FILES
    description = "Search file and path names in a bounded workspace project."
    method = "file_search"


class CodingTextSearchTool(_SearchTool):
    operation = SEARCH_TEXT
    description = "Search literal text in a bounded workspace project."
    method = "text_search"


class CodingDefinitionsTool(_SearchTool):
    operation = DEFINITIONS
    description = "Find symbol definitions in a bounded workspace project."
    method = "find_definition"


class CodingApplyPatchTool:
    spec = ToolSpec(
        name=APPLY_PATCH,
        description="Apply a previously validated full-file coding patch. Requires explicit HIGH approval.",  # noqa: E501
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "project_root": {"type": "string"},
                "patch": {"type": "object"},
            },
            "required": ["project_id", "project_root", "patch"],
        },
        output_schema={"type": "object"},
        permission_level=PermissionLevel.HIGH,
        deterministic=False,
    )

    def __init__(self, workspace: Workspace, limits: CodingLimits) -> None:
        self.applier = CodePatchApplier(workspace, limits)

    def run(self, input: dict[str, Any]) -> ToolResult:
        try:
            project = _project(input)
            patch = CodePatch.model_validate(input["patch"])
            # Executor permission is the only authorization path for this write.
            scope = _active_permission_authorization()
            if (
                scope is None
                or scope.tool_name != APPLY_PATCH
                or scope.permission_level is not PermissionLevel.HIGH
            ):
                return ToolResult(
                    ok=False,
                    error="coding patch application requires HIGH authorization",
                    error_code="permission_denied",
                )
            paths = self.applier.apply(project, patch, decision=PermissionDecision.ALLOWED)
            return ToolResult(ok=True, output={"applied": list(paths), "patch_id": patch.patch_id})
        except Exception as exc:
            return ToolResult(
                ok=False, error=f"{type(exc).__name__}: {exc}", error_code="coding_patch_failed"
            )


def register_coding_tools(
    registry: ToolRegistry, workspace: Workspace, limits: CodingLimits | None = None
) -> None:
    limits = limits or CodingLimits()
    for tool in (
        CodingAnalyzeTool(workspace, limits),
        CodingTextSearchTool(workspace, limits),
        CodingSymbolSearchTool(workspace, limits),
        CodingFileSearchTool(workspace, limits),
        CodingDefinitionsTool(workspace, limits),
        CodingApplyPatchTool(workspace, limits),
    ):
        registry.register(tool)
