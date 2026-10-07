"""Explicit LOW-permission tool for ephemeral screenshot metadata analysis."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..computer.runtime import ComputerRuntime
from ..permissions import PermissionLevel
from ..tools import ToolRegistry, ToolResult, ToolSpec
from .errors import VisionError
from .runtime import VisionRuntime
from .serialization import observation_to_dict

VISION_ANALYZE_SCREENSHOT_TOOL_NAME = "vision_analyze_screenshot"
VISION_TOOL_NAMES: tuple[str, ...] = (VISION_ANALYZE_SCREENSHOT_TOOL_NAME,)


class VisionAnalyzeScreenshotTool:
    """Analyze one screenshot obtained through the Phase 6 computer runtime."""

    spec = ToolSpec(
        name=VISION_ANALYZE_SCREENSHOT_TOOL_NAME,
        description=(
            "Validates one fresh Phase 6 screenshot and returns bounded visual "
            "metadata. Image bytes remain ephemeral and are never returned, "
            "stored, uploaded, or interpreted as instructions."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["ok"],
            "properties": {
                "ok": {"type": "boolean"},
                "observation": {"type": "object"},
                "error_code": {"type": "string"},
                "error": {"type": "string"},
            },
        },
        permission_level=PermissionLevel.LOW,
        deterministic=False,
        sensitive_output=True,
    )

    def __init__(self, computer: ComputerRuntime, vision: VisionRuntime) -> None:
        self._computer = computer
        self._vision = vision

    def run(self, input: dict[str, Any]) -> ToolResult:
        if input:
            return ToolResult(
                ok=True,
                output={
                    "ok": False,
                    "error_code": "invalid_input",
                    "error": "visual observation input failed validation",
                },
            )
        screenshot_result = self._computer.screenshot()
        screenshot = (
            screenshot_result.observation.screenshot
            if screenshot_result.ok and screenshot_result.observation is not None
            else None
        )
        if screenshot is None:
            return ToolResult(
                ok=True,
                output={
                    "ok": False,
                    "error_code": screenshot_result.error_code or "screenshot_unavailable",
                    "error": "screenshot observation could not be acquired",
                },
            )
        try:
            observation = self._vision.analyze_screenshot(screenshot)
        except VisionError as exc:
            return ToolResult(
                ok=True,
                output={"ok": False, "error_code": exc.code, "error": exc.public_message},
            )
        except (ValidationError, ValueError):
            return ToolResult(
                ok=True,
                output={
                    "ok": False,
                    "error_code": "invalid_visual_analysis",
                    "error": "visual analysis output failed validation",
                },
            )
        return ToolResult(
            ok=True,
            output={"ok": True, "observation": observation_to_dict(observation)},
        )


def register_vision_tools(
    registry: ToolRegistry,
    computer: ComputerRuntime,
    vision: VisionRuntime,
) -> None:
    """Register visual analysis only when an existing computer provider exists."""
    registry.register(VisionAnalyzeScreenshotTool(computer, vision))
