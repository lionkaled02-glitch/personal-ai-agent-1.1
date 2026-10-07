"""Tool registry, controlled execution, and JSON-Schema subset validation."""

from __future__ import annotations

import pytest
from agent_core import (
    DemoTool,
    Tool,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    validate_against_schema,
)
from agent_core.permissions import PermissionLevel


class IntegerTool:
    """Test tool with an integer field (to check the bool/int edge case)."""

    spec = ToolSpec(
        name="int_tool",
        description="takes an integer",
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"n": input["n"]})


class ExplodingTool:
    spec = ToolSpec(
        name="exploding_tool",
        description="always raises",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        raise RuntimeError("kaboom")


class LyingTool:
    """Declares one output schema but returns a different shape."""

    spec = ToolSpec(
        name="lying_tool",
        description="output does not match schema",
        input_schema={"type": "object"},
        output_schema={"type": "object", "required": ["missing_key"]},
        permission_level=PermissionLevel.LOW,
    )

    def run(self, input: dict[str, object]) -> ToolResult:
        return ToolResult(ok=True, output={"other": 1})


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(DemoTool())
    reg.register(IntegerTool())
    reg.register(ExplodingTool())
    reg.register(LyingTool())
    return reg


class TestRegistration:
    def test_register_get_require(self, registry: ToolRegistry) -> None:
        assert registry.get("demo_tool") is not None
        assert registry.require("demo_tool").spec.name == "demo_tool"
        assert registry.get("nope") is None

    def test_require_unknown_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError, match="nope"):
            registry.require("nope")

    def test_duplicate_registration_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DemoTool())

    def test_list_is_deterministic(self, registry: ToolRegistry) -> None:
        names = registry.names()
        assert names == sorted(names)
        assert "demo_tool" in names and "int_tool" in names


class TestControlledExecution:
    def test_valid_input_runs(self, registry: ToolRegistry) -> None:
        result = registry.execute("demo_tool", {"message": "hello"})
        assert result.ok
        assert result.output == {"tool": "demo_tool", "message": "hello"}

    def test_missing_required_field_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolInputError, match="missing required property 'message'"):
            registry.execute("demo_tool", {})

    def test_wrong_type_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolInputError, match="expected type 'string'"):
            registry.execute("demo_tool", {"message": 42})

    def test_bool_not_accepted_as_integer(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolInputError, match="got boolean"):
            registry.execute("int_tool", {"n": True})

    def test_unknown_tool_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError):
            registry.execute("ghost_tool", {})

    def test_tool_exception_contained_in_result(self, registry: ToolRegistry) -> None:
        result = registry.execute("exploding_tool", {})
        assert not result.ok
        assert result.error is not None and "kaboom" in result.error

    def test_output_schema_enforced(self, registry: ToolRegistry) -> None:
        result = registry.execute("lying_tool", {})
        assert not result.ok
        assert result.error is not None and "output schema" in result.error


class TestSchemaSubset:
    def test_valid_nested_object(self) -> None:
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "object", "required": ["b"], "properties": {"b": {"type": "string"}}}
            },
        }
        assert validate_against_schema({"a": {"b": "x"}}, schema) == []

    def test_nested_error_paths(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}},
        }
        errors = validate_against_schema({"a": {"b": 1}}, schema)
        assert len(errors) == 1
        assert errors[0].startswith("$.a.b")

    def test_array_items(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate_against_schema([1, 2], schema) == []
        errors = validate_against_schema([1, "x"], schema)
        assert len(errors) == 1
        assert errors[0].startswith("$[1]")

    def test_enum(self) -> None:
        errors = validate_against_schema("yellow", {"enum": ["red", "green"]})
        assert len(errors) == 1
        assert "not in enum" in errors[0]

    def test_null_type(self) -> None:
        assert validate_against_schema(None, {"type": "null"}) == []
        assert validate_against_schema(0, {"type": "null"}) != []

    def test_unknown_type_keyword_ignored(self) -> None:
        assert validate_against_schema(42, {"type": "weird-future-type"}) == []

    def test_extra_properties_allowed_by_default(self) -> None:
        assert validate_against_schema({"a": 1, "extra": 2}, {"type": "object"}) == []

    def test_protocol_conformance(self) -> None:
        tool: Tool = DemoTool()
        assert tool.spec.name == "demo_tool"
