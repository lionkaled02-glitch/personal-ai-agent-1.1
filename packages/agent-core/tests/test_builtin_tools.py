"""Built-in tool tests: calculator, date/time, text utils, JSON utils.

All deterministic: the date/time tool gets an injected fixed "now" source.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from agent_core import (
    CalculatorTool,
    DateTimeTool,
    JsonUtilsTool,
    PermissionLevel,
    TextUtilsTool,
    Tool,
    ToolRegistry,
    register_default_tools,
)
from agent_core.builtin_tools.calculator import MAX_EXPRESSION_LENGTH
from agent_core.builtin_tools.json_tool import MAX_JSON_LENGTH
from agent_core.builtin_tools.text_tool import MAX_TEXT_LENGTH

FIXED_NOW = datetime(2026, 10, 3, 12, 30, 0, tzinfo=UTC)


def run(tool: Tool, input: dict[str, object]) -> dict[str, object]:
    result = tool.run(input)
    assert result.ok, f"expected ok, got error={result.error} code={result.error_code}"
    assert isinstance(result.output, dict)
    return result.output


class TestCalculator:
    tool = CalculatorTool()

    @pytest.mark.parametrize(
        ("expression", "expected", "type"),
        [
            ("2 + 3", 5, "integer"),
            ("2*3+4", 10, "integer"),
            ("10 / 4", 2.5, "float"),
            ("10 // 4", 2, "integer"),
            ("10 % 3", 1, "integer"),
            ("-(2 + 3)", -5, "integer"),
            ("+7", 7, "integer"),
            ("2 * (3 + 4)", 14, "integer"),
            ("((1))", 1, "integer"),
            ("1.5 + 2.25", 3.75, "float"),
            ("  42  ", 42, "integer"),
            ("-2 * -3", 6, "integer"),
            ("7 - 10", -3, "integer"),
            ("1e2 + 1", 101.0, "float"),
            ("2E+3 * 1.5", 3000.0, "float"),
            ("1.5e-1", 0.15, "float"),
        ],
    )
    def test_valid_expressions(self, expression: str, expected: int | float, type: str) -> None:
        out = run(self.tool, {"expression": expression})
        assert out["result"] == expected
        assert out["type"] == type
        assert out["expression"] == expression

    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "   ",
            "+",
            "2 +",
            "2 3",
            "(2",
            "2)",
            "(2)(3)",
            "2 ** 8",  # exponentiation not supported (DoS-safe)
            "2 ^ 2",
            "x + 1",  # identifiers not supported
            "__import__('os')",
            "().__class__",
            "'a' + 'b'",
            "1 == 1",
            "a = 5",
        ],
    )
    def test_invalid_expressions_rejected(self, expression: str) -> None:
        result = self.tool.run({"expression": expression})
        assert not result.ok
        assert result.error_code == "invalid_expression"
        assert result.error is not None and "invalid expression" in result.error

    def test_expression_too_long_rejected(self) -> None:
        result = self.tool.run({"expression": "1 + " * (MAX_EXPRESSION_LENGTH)})
        assert not result.ok
        assert result.error_code == "input_too_long"

    def test_division_by_zero_is_structured(self) -> None:
        result = self.tool.run({"expression": "1 / 0"})
        assert not result.ok
        assert result.error_code == "division_by_zero"

    def test_floor_division_by_zero_is_structured(self) -> None:
        result = self.tool.run({"expression": "1 // 0"})
        assert not result.ok
        assert result.error_code == "division_by_zero"

    def test_modulo_by_zero_is_structured(self) -> None:
        result = self.tool.run({"expression": "1 % 0"})
        assert not result.ok
        assert result.error_code == "division_by_zero"

    def test_non_finite_result_rejected(self) -> None:
        result = self.tool.run({"expression": "1e308 * 10"})
        assert not result.ok
        assert result.error_code == "non_finite_result"


class TestDateTime:
    def test_utc_default(self) -> None:
        out = run(DateTimeTool(now=lambda: FIXED_NOW), {})
        assert out["timezone"] == "UTC"
        assert out["now"] == "2026-10-03T12:30:00+00:00"
        assert out["date"] == "2026-10-03"
        assert out["time"] == "12:30:00"
        assert out["utc_offset"] == "+00:00"

    def test_named_timezone(self) -> None:
        out = run(DateTimeTool(now=lambda: FIXED_NOW), {"timezone": "Asia/Riyadh"})
        assert out["timezone"] == "Asia/Riyadh"
        assert out["now"] == "2026-10-03T15:30:00+03:00"
        assert out["utc_offset"] == "+03:00"

    def test_unknown_timezone_is_structured(self) -> None:
        result = DateTimeTool(now=lambda: FIXED_NOW).run({"timezone": "Not/AZone"})
        assert not result.ok
        assert result.error_code == "invalid_timezone"

    def test_naive_now_treated_as_utc(self) -> None:
        naive = datetime(2026, 1, 1, 8, 0, 0)
        out = run(DateTimeTool(now=lambda: naive), {})
        assert out["now"] == "2026-01-01T08:00:00+00:00"

    def test_declared_non_deterministic(self) -> None:
        assert DateTimeTool.spec.deterministic is False
        assert DateTimeTool.spec.permission_level is PermissionLevel.LOW


class TestTextUtils:
    tool = TextUtilsTool()

    def test_length(self) -> None:
        assert run(self.tool, {"text": "hello", "action": "length"}) == {
            "action": "length",
            "value": 5,
        }

    def test_word_count(self) -> None:
        out = run(self.tool, {"text": "one two  three", "action": "word_count"})
        assert out["value"] == 3

    def test_line_count(self) -> None:
        out = run(self.tool, {"text": "a\nb\nc", "action": "line_count"})
        assert out["value"] == 3

    def test_empty_text(self) -> None:
        assert run(self.tool, {"text": "", "action": "word_count"}) == {
            "action": "word_count",
            "value": 0,
        }

    def test_text_too_long_rejected(self) -> None:
        result = self.tool.run({"text": "x" * (MAX_TEXT_LENGTH + 1), "action": "length"})
        assert not result.ok
        assert result.error_code == "input_too_long"

    def test_unsupported_action_rejected(self) -> None:
        result = self.tool.run({"text": "hi", "action": "reverse"})
        assert not result.ok
        assert result.error_code == "invalid_action"


class TestJsonUtils:
    tool = JsonUtilsTool()

    def test_valid_object(self) -> None:
        out = run(self.tool, {"json_text": '{"a": 1, "b": [1, 2]}'})
        assert out == {"valid": True, "type": "object", "count": 2, "error": ""}

    def test_valid_array(self) -> None:
        out = run(self.tool, {"json_text": "[1, 2, 3]"})
        assert out == {"valid": True, "type": "array", "count": 3, "error": ""}

    @pytest.mark.parametrize(
        ("text", "type"),
        [
            ('"hi"', "string"),
            ("42", "number"),
            ("true", "boolean"),
            ("null", "null"),
        ],
    )
    def test_scalars(self, text: str, type: str) -> None:
        out = run(self.tool, {"json_text": text})
        assert out["valid"] is True
        assert out["type"] == type
        assert out["count"] == 0

    def test_invalid_json_is_a_structured_answer(self) -> None:
        out = run(self.tool, {"json_text": "{not json"})
        assert out["valid"] is False
        assert out["type"] == ""
        assert out["count"] == 0
        assert out["error"]  # bounded JSONDecodeError detail

    def test_json_too_long_rejected(self) -> None:
        result = self.tool.run({"json_text": "1" * (MAX_JSON_LENGTH + 1)})
        assert not result.ok
        assert result.error_code == "input_too_long"


class TestDefaultToolSet:
    def test_register_default_tools_registers_all_five(self) -> None:
        registry: ToolRegistry = register_default_tools(ToolRegistry())
        assert registry.names() == [
            "calculator",
            "datetime",
            "demo_tool",
            "json_utils",
            "text_utils",
        ]

    def test_default_tools_are_low_permission_and_bounded(self) -> None:
        registry: ToolRegistry = register_default_tools(ToolRegistry())
        for spec in registry.list_tools():
            assert spec.permission_level is PermissionLevel.LOW
            assert spec.version

    def test_determinism_flags(self) -> None:
        registry: ToolRegistry = register_default_tools(ToolRegistry())
        flags = {spec.name: spec.deterministic for spec in registry.list_tools()}
        assert flags["calculator"] is True
        assert flags["datetime"] is False
        assert flags["text_utils"] is True
        assert flags["json_utils"] is True
        assert flags["demo_tool"] is True
