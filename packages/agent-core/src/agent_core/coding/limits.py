"""Hard-clamped limits and validation helpers for coding requests/results."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from ..config import Settings
from .errors import CodingLimitError
from .models import (
    MAX_CODING_ANALYSIS_TIME_S,
    MAX_CODING_CHANGED_FILES,
    MAX_CODING_DIAGNOSTICS,
    MAX_CODING_FILE_SIZE_BYTES,
    MAX_CODING_OUTPUT_BYTES,
    MAX_CODING_PATCH_SIZE_BYTES,
    MAX_CODING_PROJECT_FILES,
    MAX_CODING_REGIONS,
    MAX_CODING_SOURCE_CHARS,
    MAX_CODING_SYMBOLS,
    MAX_CODING_TEST_CASES,
    MAX_CODING_TEST_DURATION_S,
    CodeAnalysisRequest,
    CodeAnalysisResult,
    CodeEditRequest,
    CodeEditResult,
    CodeFile,
    CodePatch,
    CodeTestPlanRequest,
    CodeTestPlanResult,
    CodingModel,
    CodingObservation,
)


class CodingLimits(CodingModel):
    """Configurable coding bounds; settings can only tighten hard caps."""

    max_project_files: int = Field(default=100, ge=1, le=MAX_CODING_PROJECT_FILES, strict=True)
    max_file_size_bytes: int = Field(
        default=524_288,
        ge=1,
        le=MAX_CODING_FILE_SIZE_BYTES,
        strict=True,
    )
    max_source_chars: int = Field(ge=1, le=MAX_CODING_SOURCE_CHARS, default=500_000, strict=True)
    max_patch_size_bytes: int = Field(
        default=262_144,
        ge=1,
        le=MAX_CODING_PATCH_SIZE_BYTES,
        strict=True,
    )
    max_changed_files: int = Field(default=20, ge=0, le=MAX_CODING_CHANGED_FILES, strict=True)
    max_symbols: int = Field(default=200, ge=0, le=MAX_CODING_SYMBOLS, strict=True)
    max_regions: int = Field(default=100, ge=0, le=MAX_CODING_REGIONS, strict=True)
    max_diagnostics: int = Field(default=100, ge=0, le=MAX_CODING_DIAGNOSTICS, strict=True)
    max_analysis_time_s: float = Field(
        default=20.0,
        gt=0.0,
        le=MAX_CODING_ANALYSIS_TIME_S,
        allow_inf_nan=False,
        strict=True,
    )
    max_test_duration_s: float = Field(
        default=120.0,
        gt=0.0,
        le=MAX_CODING_TEST_DURATION_S,
        allow_inf_nan=False,
        strict=True,
    )
    max_output_bytes: int = Field(default=262_144, ge=1, le=MAX_CODING_OUTPUT_BYTES, strict=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> CodingLimits:
        """Build validated coding limits from the shared CODING_* settings."""
        return cls(
            max_project_files=settings.coding_max_project_files,
            max_file_size_bytes=settings.coding_max_file_size_bytes,
            max_source_chars=settings.coding_max_source_chars,
            max_patch_size_bytes=settings.coding_max_patch_size_bytes,
            max_changed_files=settings.coding_max_changed_files,
            max_symbols=settings.coding_max_symbols,
            max_regions=settings.coding_max_regions,
            max_diagnostics=settings.coding_max_diagnostics,
            max_analysis_time_s=settings.coding_max_analysis_time_s,
            max_test_duration_s=settings.coding_max_test_duration_s,
            max_output_bytes=settings.coding_max_output_bytes,
        )

    def validate_analysis_request(self, request: CodeAnalysisRequest) -> None:
        self._validate_sources(request.files)
        if len(request.regions) > self.max_regions:
            raise CodingLimitError()

    def validate_edit_request(self, request: CodeEditRequest) -> None:
        self._validate_sources(request.files)
        if len(request.regions) > self.max_regions:
            raise CodingLimitError()

    def validate_test_plan_request(self, request: CodeTestPlanRequest) -> None:
        self._validate_sources(request.files)
        if len(request.target_paths) > self.max_changed_files:
            raise CodingLimitError()

    def validate_observation(self, observation: CodingObservation) -> None:
        if len(observation.files) > self.max_project_files:
            raise CodingLimitError()
        if observation.total_source_chars > self.max_source_chars:
            raise CodingLimitError()
        if any(item.size_bytes > self.max_file_size_bytes for item in observation.files):
            raise CodingLimitError()

    def validate_analysis_result(self, result: CodeAnalysisResult) -> None:
        if len(result.symbols) > self.max_symbols:
            raise CodingLimitError()
        if len(result.diagnostics) > self.max_diagnostics:
            raise CodingLimitError()
        if len(result.analyzed_files) + len(result.skipped_files) > self.max_project_files:
            raise CodingLimitError()
        region_count = sum(symbol.region is not None for symbol in result.symbols)
        region_count += sum(diagnostic.region is not None for diagnostic in result.diagnostics)
        if region_count > self.max_regions:
            raise CodingLimitError()
        if result.elapsed_time_s > self.max_analysis_time_s:
            raise CodingLimitError()
        self.validate_observation(result.observation)
        self._validate_output_size_bytes(len(result.model_dump_json().encode("utf-8")))

    def validate_patch(self, patch: CodePatch) -> None:
        if len(patch.changes) > self.max_changed_files:
            raise CodingLimitError()
        if patch.patch_size_bytes > self.max_patch_size_bytes:
            raise CodingLimitError()
        if any(
            change.original_size_bytes > self.max_file_size_bytes
            or change.replacement_size_bytes > self.max_file_size_bytes
            for change in patch.changes
        ):
            raise CodingLimitError()

    def validate_edit_result(self, result: CodeEditResult) -> None:
        if result.patch is not None:
            self.validate_patch(result.patch)
        self._validate_output_size_bytes(len(result.model_dump_json().encode("utf-8")))

    def validate_test_plan_result(self, result: CodeTestPlanResult) -> None:
        if len(result.test_cases) > min(self.max_project_files, MAX_CODING_TEST_CASES):
            raise CodingLimitError()
        if result.estimated_duration_s > self.max_test_duration_s:
            raise CodingLimitError()
        self._validate_output_size_bytes(len(result.model_dump_json().encode("utf-8")))

    def _validate_sources(self, files: Iterable[CodeFile]) -> None:
        source_files = tuple(files)
        if len(source_files) > self.max_project_files:
            raise CodingLimitError()
        total_chars = sum(len(source.content) for source in source_files)
        if total_chars > self.max_source_chars:
            raise CodingLimitError()
        if any(source.size_bytes > self.max_file_size_bytes for source in source_files):
            raise CodingLimitError()

    def _validate_output_size_bytes(self, size: int) -> None:
        if size > self.max_output_bytes:
            raise CodingLimitError()
