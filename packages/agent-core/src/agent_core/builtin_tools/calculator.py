"""Calculator tool: safe basic arithmetic, no code execution.

Safety design:

- **No ``eval``/``exec``, no shell, no imports.** Expressions are processed
  by a hand-written tokenizer + recursive-descent parser over an explicit
  allow-list: integers/floats (including ``1e308``-style exponent literals),
  ``+ - * / // %``, unary ``+ -`` and parentheses. Anything else
  (identifiers, ``**``, comparison, assignment, attribute access, ...) is
  rejected as an invalid expression.
- Expressions longer than :data:`MAX_EXPRESSION_LENGTH` are rejected before
  parsing (bounded work).
- Division by zero and non-finite results are structured failures
  (``ok=False`` with a domain ``error_code``), never exceptions.
"""

from __future__ import annotations

import math
from typing import Any

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec

CALCULATOR_TOOL_NAME = "calculator"

#: Maximum accepted expression length (rejects oversized model output).
MAX_EXPRESSION_LENGTH = 200

_NUMBER_CHARS = "0123456789."


def _is_number_literal(token: str) -> bool:
    return "." in token or "e" in token or "E" in token


_BINARY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
}


class _CalcError(Exception):
    """Internal: a malformed or unsupported expression."""


def _tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(expression)
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
        elif ch in _NUMBER_CHARS:
            start = i
            while i < n and expression[i] in _NUMBER_CHARS:
                i += 1
            # Optional exponent part: e/E, optional sign, at least one digit.
            if i < n and expression[i] in "eE":
                j = i + 1
                if j < n and expression[j] in "+-":
                    j += 1
                if j < n and expression[j].isdigit():
                    j += 1
                    while j < n and expression[j].isdigit():
                        j += 1
                    i = j
            tokens.append(expression[start:i])
        elif expression[i : i + 2] == "//":
            tokens.append("//")
            i += 2
        elif ch in "+-*/%()":
            tokens.append(ch)
            i += 1
        else:
            raise _CalcError(f"unsupported character {ch!r}")
    return tokens


class _Parser:
    """Recursive-descent parser/evaluator for the supported expression grammar.

    grammar:  expr    := term (("+" | "-") term)*
              term    := factor (("*" | "/" | "//" | "%") factor)*
              factor  := ("+" | "-") factor | primary
              primary := NUMBER | "(" expr ")"
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> float | int:
        if not self._tokens:
            raise _CalcError("empty expression")
        value = self._expr()
        if self._pos != len(self._tokens):
            raise _CalcError(f"unexpected token {self._tokens[self._pos]!r} at end")
        return value

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> str:
        token = self._peek()
        if token is None:
            raise _CalcError("unexpected end of expression")
        self._pos += 1
        return token

    def _expr(self) -> float | int:
        value = self._term()
        while (token := self._peek()) in ("+", "-"):
            self._next()
            value = _BINARY_OPS[token](value, self._term())
        return value

    def _term(self) -> float | int:
        value = self._factor()
        while (token := self._peek()) in ("*", "/", "//", "%"):
            self._next()
            value = _BINARY_OPS[token](value, self._factor())
        return value

    def _factor(self) -> float | int:
        token = self._peek()
        if token == "+":
            self._next()
            return self._factor()
        if token == "-":
            self._next()
            return -self._factor()
        return self._primary()

    def _primary(self) -> float | int:
        token = self._next()
        if token == "(":
            value = self._expr()
            if self._next() != ")":
                raise _CalcError("missing closing parenthesis")
            return value
        try:
            return float(token) if _is_number_literal(token) else int(token)
        except ValueError:
            raise _CalcError(f"invalid number {token!r}") from None


def calculate(expression: str) -> float | int:
    """Evaluate a safe arithmetic expression. Raises :class:`_CalcError` on
    malformed/unsupported input and :class:`ZeroDivisionError` on /0."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise _CalcError(f"expression too long ({len(expression)} > {MAX_EXPRESSION_LENGTH} chars)")
    return _Parser(_tokenize(expression)).parse()


class CalculatorTool:
    """Evaluates basic arithmetic expressions (deterministic, LOW permission)."""

    spec = ToolSpec(
        name=CALCULATOR_TOOL_NAME,
        description=(
            "Evaluates a basic arithmetic expression (+, -, *, /, //, %, parentheses; "
            "no other operations) and returns the numeric result."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 3 * 4'.",
                },
            },
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "result": {"type": "number"},
                "type": {"enum": ["integer", "float"]},
            },
            "required": ["expression", "result", "type"],
        },
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, Any]) -> ToolResult:
        expression = str(input["expression"])
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return ToolResult(
                ok=False,
                error=f"expression too long ({len(expression)} > {MAX_EXPRESSION_LENGTH} chars)",
                error_code="input_too_long",
            )
        try:
            result = calculate(expression)
        except ZeroDivisionError:
            return ToolResult(ok=False, error="division by zero", error_code="division_by_zero")
        except _CalcError as exc:
            return ToolResult(
                ok=False,
                error=f"invalid expression: {exc}",
                error_code="invalid_expression",
            )
        if isinstance(result, float) and not math.isfinite(result):
            return ToolResult(
                ok=False,
                error="result is not finite",
                error_code="non_finite_result",
            )
        return ToolResult(
            ok=True,
            output={
                "expression": expression,
                "result": result,
                "type": "integer" if isinstance(result, int) else "float",
            },
        )
