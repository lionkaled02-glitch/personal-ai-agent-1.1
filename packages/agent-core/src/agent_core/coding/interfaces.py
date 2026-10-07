"""Provider-neutral contracts for bounded coding analysis and planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..permissions import PermissionLevel
from .limits import CodingLimits
from .models import (
    CodeAnalysisRequest,
    CodeAnalysisResult,
    CodeEditRequest,
    CodeEditResult,
    CodeTestPlanRequest,
    CodeTestPlanResult,
)


class CodingOperation(StrEnum):
    """Step 1 operations are analysis/planning only and therefore LOW risk."""

    ANALYZE = "coding_analyze"
    PROPOSE_EDITS = "coding_propose_edits"
    PLAN_TESTS = "coding_plan_tests"

    @property
    def required_permission(self) -> PermissionLevel:
        """These operations return data/proposals and perform no mutations."""
        return PermissionLevel.LOW


@runtime_checkable
class CodingProvider(Protocol):
    """Synchronous, in-memory provider contract; it exposes no execution API.

    Implementations receive already-bounded source snapshots. Repository,
    provider, and any future test-output text are untrusted data, never policy
    or instructions.
    The caller remains responsible for validating all paths through
    ``Workspace`` and for enforcing configured limits on provider output.
    This protocol has no filesystem write, shell/process, network, compiler,
    package-installation, test-execution, or generic run-code operation.
    """

    def analyze(
        self,
        request: CodeAnalysisRequest,
        *,
        limits: CodingLimits,
    ) -> CodeAnalysisResult:
        """Return bounded symbols, diagnostics, timing, and safe metadata."""
        ...

    def propose_edits(
        self,
        request: CodeEditRequest,
        *,
        limits: CodingLimits,
    ) -> CodeEditResult:
        """Return a structured proposal; never apply it to disk."""
        ...

    def plan_tests(
        self,
        request: CodeTestPlanRequest,
        *,
        limits: CodingLimits,
    ) -> CodeTestPlanResult:
        """Return test-design metadata only; never run tests or build commands."""
        ...
