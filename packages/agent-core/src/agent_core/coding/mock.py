"""Deterministic, offline-only mock coding provider."""

from __future__ import annotations

from .errors import CodingProviderError, CodingTimeoutError, CodingValidationError
from .interfaces import CodingOperation
from .limits import CodingLimits
from .models import (
    CodeAnalysisRequest,
    CodeAnalysisResult,
    CodeAnalysisStatus,
    CodeEditRequest,
    CodeEditResult,
    CodeEditStatus,
    CodePatch,
    CodeTestPlanRequest,
    CodeTestPlanResult,
    CodeTestPlanStatus,
    CodingObservation,
    TestCasePlan,
)


class MockCodingProvider:
    """Local deterministic provider with optional fixed failures for tests.

    It never reads a path, writes a patch, inspects repository text, executes
    code, runs tests, accesses a network, or retains source snapshots.
    """

    def __init__(
        self,
        *,
        proposed_patch: CodePatch | None = None,
        fail_operation: CodingOperation | None = None,
        timeout_operation: CodingOperation | None = None,
    ) -> None:
        if fail_operation is not None and fail_operation is timeout_operation:
            raise ValueError("one mock operation cannot both fail and time out")
        self._proposed_patch = proposed_patch
        self._fail_operation = fail_operation
        self._timeout_operation = timeout_operation
        self.calls: dict[CodingOperation, int] = dict.fromkeys(CodingOperation, 0)

    def analyze(
        self,
        request: CodeAnalysisRequest,
        *,
        limits: CodingLimits,
    ) -> CodeAnalysisResult:
        limits.validate_analysis_request(request)
        self._before(CodingOperation.ANALYZE)
        result = CodeAnalysisResult(
            project_id=request.project.project_id,
            status=CodeAnalysisStatus.COMPLETE,
            summary=f"Offline mock analyzed {len(request.files)} bounded source file(s).",
            symbols=(),
            diagnostics=(),
            elapsed_time_s=0.0,
            observation=CodingObservation.from_files(request.project, request.files),
        )
        limits.validate_analysis_result(result)
        return result

    def propose_edits(
        self,
        request: CodeEditRequest,
        *,
        limits: CodingLimits,
    ) -> CodeEditResult:
        limits.validate_edit_request(request)
        self._before(CodingOperation.PROPOSE_EDITS)
        patch = self._proposed_patch
        if patch is None:
            result = CodeEditResult(
                project_id=request.project.project_id,
                status=CodeEditStatus.NO_CHANGES,
                summary="Offline mock does not propose source changes by default.",
            )
        else:
            self._validate_patch_reference(request, patch)
            limits.validate_patch(patch)
            result = CodeEditResult(
                project_id=request.project.project_id,
                status=CodeEditStatus.PROPOSED,
                summary=patch.summary,
                patch=patch,
            )
        limits.validate_edit_result(result)
        return result

    def plan_tests(
        self,
        request: CodeTestPlanRequest,
        *,
        limits: CodingLimits,
    ) -> CodeTestPlanResult:
        limits.validate_test_plan_request(request)
        self._before(CodingOperation.PLAN_TESTS)
        cases = tuple(
            TestCasePlan(
                target_paths=(path,),
                purpose=f"Plan focused tests for {path}.",
                expected_behavior=(
                    "Cover the requested behavior and relevant boundaries with offline tests."
                ),
            )
            for path in sorted(request.target_paths)
        )
        result = CodeTestPlanResult(
            project_id=request.project.project_id,
            status=CodeTestPlanStatus.PLANNED,
            test_cases=cases,
            estimated_duration_s=min(limits.max_test_duration_s, 5.0 * len(cases)),
            summary=f"Planned {len(cases)} test case(s); no tests were executed.",
            execution_performed=False,
        )
        limits.validate_test_plan_result(result)
        return result

    def _before(self, operation: CodingOperation) -> None:
        self.calls[operation] += 1
        if operation is self._timeout_operation:
            raise CodingTimeoutError()
        if operation is self._fail_operation:
            raise CodingProviderError()

    @staticmethod
    def _validate_patch_reference(request: CodeEditRequest, patch: CodePatch) -> None:
        if patch.project_id != request.project.project_id:
            raise CodingValidationError()
        source_by_path = {source.path: source for source in request.files}
        for change in patch.changes:
            source = source_by_path.get(change.target_path)
            if (
                source is None
                or change.original_sha256 != source.source_sha256
                or change.original_size_bytes != source.size_bytes
            ):
                raise CodingValidationError()
