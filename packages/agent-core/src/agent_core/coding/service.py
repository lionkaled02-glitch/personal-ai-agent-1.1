"""High-level coding service combining analysis, search, proposal and safe apply."""

from __future__ import annotations

from .editing import CodePatchApplier
from .limits import CodingLimits
from .models import (
    CodeAnalysisRequest,
    CodeAnalysisResult,
    CodePatch,
    CodingProject,
)
from .runtime import CodingAnalysisRuntime
from .search import CodeSearchResult, CodeSearchRuntime


class CodingService:
    """Provider-neutral facade; external providers remain optional and isolated."""

    def __init__(self, workspace, limits: CodingLimits | None = None) -> None:
        self.limits = limits or CodingLimits()
        self.analysis = CodingAnalysisRuntime(workspace, self.limits)
        self.search = CodeSearchRuntime(workspace, self.limits)
        self.patcher = CodePatchApplier(workspace, self.limits)

    def analyze(self, request: CodeAnalysisRequest) -> CodeAnalysisResult:
        return self.analysis.analyze_project(request.project)

    def search_text(
        self, project: CodingProject, query: str, max_results: int = 50
    ) -> CodeSearchResult:
        return self.search.text_search(project, query, max_results=max_results)

    def search_symbols(
        self, project: CodingProject, query: str, max_results: int = 50
    ) -> CodeSearchResult:
        return self.search.symbol_search(project, query, max_results=max_results)

    def search_files(
        self, project: CodingProject, query: str, max_results: int = 50
    ) -> CodeSearchResult:
        return self.search.file_search(project, query, max_results=max_results)

    def definitions(
        self, project: CodingProject, name: str, max_results: int = 20
    ) -> CodeSearchResult:
        return self.search.find_definition(project, name, max_results=max_results)

    def apply(self, project: CodingProject, patch: CodePatch, *, decision):
        return self.patcher.apply(project, patch, decision=decision)
