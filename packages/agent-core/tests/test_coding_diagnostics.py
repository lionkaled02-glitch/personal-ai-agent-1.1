"""Deterministic offline tests for provider-neutral code diagnostics."""

from __future__ import annotations

import pytest
from agent_core.coding import (
    CodeDiagnosticCategory,
    CodeDiagnosticsEngine,
    CodeFile,
    CodeLanguage,
    CodingLimitError,
    CodingLimits,
    DiagnosticSeverity,
)


@pytest.fixture
def engine() -> CodeDiagnosticsEngine:
    return CodeDiagnosticsEngine()


class TestPythonDiagnostics:
    def test_syntax_diagnostic_has_stable_code_category_and_location(
        self,
        engine: CodeDiagnosticsEngine,
    ) -> None:
        source = CodeFile(path="project/broken.py", content="def broken(:\n    pass\n")

        batch = engine.diagnose_file(source, CodeLanguage.PYTHON)
        diagnostics = batch.diagnostics
        assert batch.truncated is False

        assert len(diagnostics) == 1
        diagnostic = diagnostics[0]
        assert diagnostic.code == "PY001"
        assert diagnostic.category is CodeDiagnosticCategory.SYNTAX
        assert diagnostic.severity is DiagnosticSeverity.ERROR
        assert diagnostic.path == source.path
        assert diagnostic.region is not None
        assert diagnostic.region.path == source.path
        assert diagnostic.region.start_line == 1
        assert diagnostic.region.start_column is not None
        assert "broken" not in diagnostic.message

    def test_common_style_diagnostics_are_ordered_and_repeatable(
        self,
        engine: CodeDiagnosticsEngine,
    ) -> None:
        source = CodeFile(
            path="project/style.py",
            content="value = 1  \n" + "x" * 121 + "\n",
        )

        first = engine.diagnose_file(source, CodeLanguage.PYTHON).diagnostics
        second = engine.diagnose_file(source, CodeLanguage.PYTHON).diagnostics

        assert first == second
        assert [item.code for item in first] == ["STYLE001", "STYLE002"]
        assert all(item.category is CodeDiagnosticCategory.STYLE for item in first)
        assert all(item.severity is DiagnosticSeverity.WARNING for item in first)
        assert [item.region.start_line for item in first if item.region is not None] == [1, 2]
        assert first[0].region is not None
        assert first[0].region.start_column == len("value = 1") + 1
        assert first[1].region is not None
        assert first[1].region.start_column == 121

    def test_coding_limits_bound_diagnostics_and_source_input(self) -> None:
        source = CodeFile(path="project/style.py", content="a = 1  \nb = 2  \n")
        limits = CodingLimits(max_diagnostics=1)
        batch = CodeDiagnosticsEngine(limits).diagnose_file(source, CodeLanguage.PYTHON)

        assert len(batch.diagnostics) == 1
        assert batch.truncated is True

        with pytest.raises(CodingLimitError):
            CodeDiagnosticsEngine(CodingLimits(max_source_chars=3)).diagnose_file(
                source,
                CodeLanguage.PYTHON,
            )


class TestEcmaScriptDiagnostics:
    @pytest.mark.parametrize("language", [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT])
    def test_unmatched_delimiters_are_reported_with_path_and_location(
        self,
        engine: CodeDiagnosticsEngine,
        language: CodeLanguage,
    ) -> None:
        extension = ".ts" if language is CodeLanguage.TYPESCRIPT else ".js"
        source = CodeFile(
            path=f"project/broken{extension}",
            content="const result = values[0;\n",
        )

        diagnostics = engine.diagnose_file(source, language).diagnostics

        assert diagnostics
        diagnostic = diagnostics[0]
        assert diagnostic.code == "ECMA001"
        assert diagnostic.category is CodeDiagnosticCategory.SYNTAX
        assert diagnostic.severity is DiagnosticSeverity.ERROR
        assert diagnostic.path == source.path
        assert diagnostic.region is not None
        assert diagnostic.region.path == source.path
        assert diagnostic.region.start_line == 1
        assert diagnostic.region.start_column is not None

    def test_delimiters_inside_strings_and_comments_are_ignored(
        self,
        engine: CodeDiagnosticsEngine,
    ) -> None:
        source = CodeFile(
            path="project/valid.js",
            content=(
                'const text = "([}"; // }])\n'
                "/* comment with unmatched-looking ({[ */\n"
                "function read() { return text; }\n"
            ),
        )

        diagnostics = engine.diagnose_file(source, CodeLanguage.JAVASCRIPT).diagnostics

        assert diagnostics == ()

    def test_unterminated_string_is_a_structured_syntax_diagnostic(
        self,
        engine: CodeDiagnosticsEngine,
    ) -> None:
        source = CodeFile(path="project/broken.ts", content='const value = "open\n')

        diagnostics = engine.diagnose_file(source, CodeLanguage.TYPESCRIPT).diagnostics

        assert diagnostics[0].code == "ECMA002"
        assert diagnostics[0].category is CodeDiagnosticCategory.SYNTAX
        assert diagnostics[0].path == source.path
        assert diagnostics[0].region is not None
        assert diagnostics[0].region.start_line == 1

    def test_metadata_only_language_has_no_source_diagnostics(
        self,
        engine: CodeDiagnosticsEngine,
    ) -> None:
        source = CodeFile(path="project/README.md", content="trailing spaces  \n")

        assert engine.diagnose_file(source, CodeLanguage.MARKDOWN).diagnostics == ()
