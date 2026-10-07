"""Bounded deterministic diagnostics for read-only coding source snapshots.

This module consumes already-loaded :class:`CodeFile` data only. It never
opens paths or executes source. Python syntax uses ``ast.parse``; JavaScript
and TypeScript use a conservative delimiter/string/comment scan, not a parser
or compiler. A small fixed set of whitespace checks applies to all three.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

from .errors import CodingLimitError
from .limits import CodingLimits
from .models import (
    MAX_CODING_FILE_CHARS,
    CodeDiagnostic,
    CodeDiagnosticCategory,
    CodeFile,
    CodeLanguage,
    CodeRegion,
    DiagnosticSeverity,
)

_MAX_LINE_LENGTH = 120
_MAX_ECMASCRIPT_DELIMITER_DEPTH = 256
_CLOSING_FOR = {"(": ")", "[": "]", "{": "}"}


@dataclass(frozen=True, slots=True)
class CodeDiagnosticBatch:
    """One per-file bounded diagnostic batch and its truncation state."""

    diagnostics: tuple[CodeDiagnostic, ...]
    truncated: bool


class CodeDiagnosticsEngine:
    """Run stable, local diagnostic rules over one bounded source snapshot."""

    def __init__(self, limits: CodingLimits | None = None) -> None:
        self._limits = limits or CodingLimits()

    def diagnose_file(
        self,
        source: CodeFile,
        language: CodeLanguage,
        *,
        max_diagnostics: int | None = None,
    ) -> CodeDiagnosticBatch:
        """Return deterministic diagnostics without accessing the filesystem.

        Syntax findings precede fixed style findings. The per-file batch is
        capped by ``CodingLimits``; callers may tighten that cap to account for
        diagnostics already accumulated elsewhere in a project result.
        """
        if (
            source.size_bytes > self._limits.max_file_size_bytes
            or len(source.content) > self._limits.max_source_chars
        ):
            raise CodingLimitError()
        if max_diagnostics is not None and max_diagnostics < 0:
            raise ValueError("max_diagnostics must be non-negative")
        limit = self._limits.max_diagnostics
        if max_diagnostics is not None:
            limit = min(limit, max_diagnostics)

        diagnostics: list[CodeDiagnostic] = []
        for diagnostic in _iter_file_diagnostics(source, language):
            if len(diagnostics) >= limit:
                return CodeDiagnosticBatch(tuple(diagnostics), truncated=True)
            diagnostics.append(diagnostic)
        return CodeDiagnosticBatch(tuple(diagnostics), truncated=False)


def _iter_file_diagnostics(
    source: CodeFile,
    language: CodeLanguage,
) -> Iterator[CodeDiagnostic]:
    if language is CodeLanguage.PYTHON:
        diagnostic = _python_syntax_diagnostic(source)
        if diagnostic is not None:
            yield diagnostic
    elif language in {CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT}:
        diagnostic = _ecmascript_syntax_diagnostic(source)
        if diagnostic is not None:
            yield diagnostic
    else:
        return

    yield from _style_diagnostics(source)


def _python_syntax_diagnostic(source: CodeFile) -> CodeDiagnostic | None:
    try:
        ast.parse(source.content, filename=source.path, type_comments=True)
    except SyntaxError as exc:
        return _diagnostic(
            "PY001",
            CodeDiagnosticCategory.SYNTAX,
            DiagnosticSeverity.ERROR,
            "Python syntax is invalid; source was not executed.",
            source.path,
            exc.lineno,
            exc.offset,
        )
    except (ValueError, RecursionError, MemoryError):
        return _diagnostic(
            "PY002",
            CodeDiagnosticCategory.LIMIT,
            DiagnosticSeverity.WARNING,
            "Python syntax analysis reached a safe structural limit.",
            source.path,
        )
    return None


def _ecmascript_syntax_diagnostic(source: CodeFile) -> CodeDiagnostic | None:
    text = source.content
    stack: list[tuple[str, int]] = []
    quote: str | None = None
    quote_start = 0
    escaped = False
    in_line_comment = False
    in_block_comment = False
    block_comment_start = 0
    index = 0

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
            else:
                index += 1
                continue

        if in_block_comment:
            if char == "*" and following == "/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            elif char in "\r\n" and quote != "`":
                return _diagnostic(
                    "ECMA002",
                    CodeDiagnosticCategory.SYNTAX,
                    DiagnosticSeverity.ERROR,
                    "String literal is not terminated.",
                    source.path,
                    *_line_column(text, quote_start),
                )
            index += 1
            continue

        if char == "/" and following == "/":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            in_block_comment = True
            block_comment_start = index
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            quote_start = index
            index += 1
            continue
        if char in _CLOSING_FOR:
            if len(stack) >= _MAX_ECMASCRIPT_DELIMITER_DEPTH:
                line, column = _line_column(text, index)
                return _diagnostic(
                    "ECMA003",
                    CodeDiagnosticCategory.LIMIT,
                    DiagnosticSeverity.WARNING,
                    "JavaScript/TypeScript delimiter nesting exceeded the analysis bound.",
                    source.path,
                    line,
                    column,
                )
            stack.append((char, index))
        elif char in _CLOSING_FOR.values():
            if not stack or _CLOSING_FOR[stack[-1][0]] != char:
                line, column = _line_column(text, index)
                return _diagnostic(
                    "ECMA001",
                    CodeDiagnosticCategory.SYNTAX,
                    DiagnosticSeverity.ERROR,
                    "JavaScript/TypeScript contains an unmatched closing delimiter.",
                    source.path,
                    line,
                    column,
                )
            stack.pop()
        index += 1

    if quote is not None:
        return _diagnostic(
            "ECMA002",
            CodeDiagnosticCategory.SYNTAX,
            DiagnosticSeverity.ERROR,
            "String or template literal is not terminated.",
            source.path,
            *_line_column(text, quote_start),
        )
    if in_block_comment:
        return _diagnostic(
            "ECMA002",
            CodeDiagnosticCategory.SYNTAX,
            DiagnosticSeverity.ERROR,
            "Block comment is not terminated.",
            source.path,
            *_line_column(text, block_comment_start),
        )
    if stack:
        _, position = stack[-1]
        line, column = _line_column(text, position)
        return _diagnostic(
            "ECMA001",
            CodeDiagnosticCategory.SYNTAX,
            DiagnosticSeverity.ERROR,
            "JavaScript/TypeScript opening delimiter is not closed.",
            source.path,
            line,
            column,
        )
    return None


def _source_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield lines without building a source-sized list; recognize CR/LF forms."""
    line_number = 1
    start = 0
    index = 0
    while index < len(text):
        if text[index] in "\r\n":
            yield line_number, text[start:index]
            if text[index] == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            index += 1
            line_number += 1
            start = index
        else:
            index += 1
    yield line_number, text[start:]


def _style_diagnostics(source: CodeFile) -> Iterator[CodeDiagnostic]:
    for line_number, line in _source_lines(source.content):
        trimmed = line.rstrip(" \t")
        if len(trimmed) != len(line):
            column = len(trimmed) + 1
            yield _diagnostic(
                "STYLE001",
                CodeDiagnosticCategory.STYLE,
                DiagnosticSeverity.WARNING,
                "Line has trailing whitespace.",
                source.path,
                line_number,
                column,
            )
        if len(line) > _MAX_LINE_LENGTH:
            yield _diagnostic(
                "STYLE002",
                CodeDiagnosticCategory.STYLE,
                DiagnosticSeverity.WARNING,
                f"Line exceeds the fixed {_MAX_LINE_LENGTH}-character analysis threshold.",
                source.path,
                line_number,
                _MAX_LINE_LENGTH + 1,
            )


def _diagnostic(
    code: str,
    category: CodeDiagnosticCategory,
    severity: DiagnosticSeverity,
    message: str,
    path: str,
    line: int | None = None,
    column: int | None = None,
) -> CodeDiagnostic:
    region: CodeRegion | None = None
    if line is not None and 1 <= line <= MAX_CODING_FILE_CHARS:
        safe_column = column if column is not None and 1 <= column <= 16_384 else None
        region = CodeRegion(
            path=path,
            start_line=line,
            end_line=line,
            start_column=safe_column,
            end_column=safe_column,
        )
    return CodeDiagnostic(
        code=code,
        category=category,
        severity=severity,
        message=message,
        path=path,
        region=region,
    )


def _line_column(text: str, position: int) -> tuple[int, int]:
    line = 1
    column = 1
    index = 0
    while index < position:
        if text[index] == "\r":
            line += 1
            column = 1
            if index + 1 < position and text[index + 1] == "\n":
                index += 1
        elif text[index] == "\n":
            line += 1
            column = 1
        else:
            column += 1
        index += 1
    return line, column
