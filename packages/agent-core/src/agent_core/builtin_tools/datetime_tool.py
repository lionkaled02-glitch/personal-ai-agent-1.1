"""Date/time tool: current date and time in a named timezone.

- Timezones are IANA names (``UTC``, ``Asia/Riyadh``, ...) resolved with the
  stdlib ``zoneinfo`` — unknown names are structured failures, never guesses.
- The "now" source is injectable (``now`` callable) so tests are
  deterministic; with the default it reads the system clock, which is why
  the tool is declared ``deterministic=False`` in its spec.
- No filesystem, network, or code execution.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..permissions import PermissionLevel
from ..tools import ToolResult, ToolSpec

DATETIME_TOOL_NAME = "datetime"

DEFAULT_TIMEZONE = "UTC"


class DateTimeTool:
    """Returns the current date/time in a timezone (LOW permission)."""

    spec = ToolSpec(
        name=DATETIME_TOOL_NAME,
        description=(
            "Returns the current date and time in an IANA timezone "
            f"(default {DEFAULT_TIMEZONE}). Non-deterministic: reads the current time."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        f"IANA timezone name, e.g. 'UTC' or 'Asia/Riyadh' "
                        f"(default '{DEFAULT_TIMEZONE}')"
                    ),
                },
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "now": {"type": "string", "description": "ISO-8601 timestamp with offset."},
                "date": {"type": "string", "description": "Local date (YYYY-MM-DD)."},
                "time": {"type": "string", "description": "Local time (HH:MM:SS)."},
                "timezone": {"type": "string"},
                "utc_offset": {"type": "string"},
            },
            "required": ["now", "date", "time", "timezone", "utc_offset"],
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
    )

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        # Injectable clock source for deterministic tests.
        self._now: Callable[[], datetime] = now or (lambda: datetime.now(UTC))

    def run(self, input: dict[str, Any]) -> ToolResult:
        tz_name = str(input.get("timezone") or DEFAULT_TIMEZONE)
        try:
            tz = ZoneInfo(tz_name)
        except (ValueError, KeyError):
            return ToolResult(
                ok=False,
                error=f"unknown timezone {tz_name!r}",
                error_code="invalid_timezone",
            )
        current = self._now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        local = current.astimezone(tz)
        raw_offset = local.strftime("%z")  # e.g. "+0300"
        utc_offset = f"{raw_offset[:3]}:{raw_offset[3:]}" if len(raw_offset) == 5 else "+00:00"
        return ToolResult(
            ok=True,
            output={
                "now": local.isoformat(),
                "date": local.date().isoformat(),
                "time": local.time().isoformat(),
                "timezone": tz_name,
                "utc_offset": utc_offset,
            },
        )
