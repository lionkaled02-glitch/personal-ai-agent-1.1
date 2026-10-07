"""Minimal JSON-Schema validation (documented subset).

Supports the subset of JSON Schema that agent-core tools use today:
``type``, ``properties``, ``required``, ``items``, ``enum``. Unrecognized
keywords are ignored (forward-compatible, never guessed at).

If tool schemas outgrow this subset, switch to a dedicated validation
library — that is a deliberate future decision, not a Phase 0 one.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return human-readable validation errors (empty list when valid)."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum {schema['enum']!r}")
        return errors

    expected = schema.get("type")
    if expected is not None:
        types = _TYPE_CHECKS.get(expected)
        if types is None:
            return errors  # unknown type keyword: skip, do not guess
        # bool is a subclass of int in Python; reject it for numeric types.
        if isinstance(value, bool) and expected in ("integer", "number"):
            errors.append(f"{path}: expected {expected}, got boolean")
            return errors
        if not isinstance(value, types):
            errors.append(f"{path}: expected type {expected!r}, got {type(value).__name__!r}")
            return errors

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_against_schema(value[key], subschema, f"{path}.{key}"))

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_against_schema(item, schema["items"], f"{path}[{index}]"))

    return errors
