# ruff: noqa: F401
"""Phase 10 coding foundation: bounded data, read-only analysis, diagnostics.

The local runtime reads validated workspace files and returns bounded facts;
the diagnostics engine consumes only those source snapshots. Edit proposals
and test plans remain data. This package adds no file writes, code execution,
command runner, or real model provider. Resolve paths through ``Workspace``.
"""

from .diagnostics import CodeDiagnosticBatch, CodeDiagnosticsEngine
from .errors import (
    CodingError,
    CodingLimitError,
    CodingPermissionError,
    CodingProviderError,
    CodingTimeoutError,
    CodingUnsupportedOperationError,
    CodingValidationError,
    CodingWorkspaceError,
)
from .interfaces import CodingOperation, CodingProvider
from .limits import CodingLimits
from .mock import MockCodingProvider
from .models import (
    AnalyzedCodeFile,
    CodeAnalysisLimitReason,
    CodeAnalysisMethod,
    CodeAnalysisRequest,
    CodeAnalysisResult,
    CodeAnalysisStatus,
    CodeChange,
    CodeDiagnostic,
    CodeDiagnosticCategory,
    CodeEditRequest,
    CodeEditResult,
    CodeEditStatus,
    CodeFile,
    CodeFileMetric,
    CodeLanguage,
    CodePatch,
    CodeRegion,
    CodeSkipReason,
    CodeSymbol,
    CodeSymbolKind,
    CodeTestPlanRequest,
    CodeTestPlanResult,
    CodeTestPlanStatus,
    CodingObservation,
    CodingProject,
    DiagnosticSeverity,
    ObservedCodeFile,
    PatchCheckStatus,
    PatchValidationCheck,
    PatchValidationMetadata,
    PatchValidationStatus,
    SkippedCodeFile,
    TestCasePlan,
)
from .runtime import CodingAnalysisRuntime

__all__ = [
    "AnalyzedCodeFile",
    "CodeAnalysisLimitReason",
    "CodeAnalysisMethod",
    "CodeAnalysisRequest",
    "CodeAnalysisResult",
    "CodeAnalysisStatus",
    "CodeChange",
    "CodeDiagnostic",
    "CodeDiagnosticBatch",
    "CodeDiagnosticCategory",
    "CodeDiagnosticsEngine",
    "CodeEditRequest",
    "CodeEditResult",
    "CodeEditStatus",
    "CodeFile",
    "CodeFileMetric",
    "CodeLanguage",
    "CodePatch",
    "CodeRegion",
    "CodeSkipReason",
    "CodeSymbol",
    "CodeSymbolKind",
    "CodeTestPlanRequest",
    "CodeTestPlanResult",
    "CodeTestPlanStatus",
    "CodingAnalysisRuntime",
    "CodingError",
    "CodingLimitError",
    "CodingLimits",
    "CodingObservation",
    "CodingOperation",
    "CodingPermissionError",
    "CodingProject",
    "CodingProvider",
    "CodingProviderError",
    "CodingTimeoutError",
    "CodingUnsupportedOperationError",
    "CodingValidationError",
    "CodingWorkspaceError",
    "DiagnosticSeverity",
    "MockCodingProvider",
    "ObservedCodeFile",
    "PatchCheckStatus",
    "PatchValidationCheck",
    "PatchValidationMetadata",
    "PatchValidationStatus",
    "SkippedCodeFile",
    "TestCasePlan",
]

from .editing import CodePatchApplier
from .search import CodeSearchHit, CodeSearchResult, CodeSearchRuntime
from .service import CodingService
from .tools import (
    ANALYZE,
    APPLY_PATCH,
    DEFINITIONS,
    SEARCH_FILES,
    SEARCH_SYMBOLS,
    SEARCH_TEXT,
    register_coding_tools,
)
