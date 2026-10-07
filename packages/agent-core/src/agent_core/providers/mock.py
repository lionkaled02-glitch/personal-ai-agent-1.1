"""Deterministic in-memory model provider (test double + demo default).

Two usage modes:

1. **Scripted** — pass ``responses``; each :meth:`complete` pops the next
   canned response (the last one repeats). This is what tests use, which
   keeps them deterministic and offline.
2. **Keyword planner** — with no scripted responses, the provider emulates a
   very small planner: it looks for known tool keywords in the user message
   and emits a strict-JSON plan. The demo entry point relies on this.

This provider never touches the network and never requires an API key.
``self.requests`` records every request for test assertions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..errors import ProviderError
from .base import Capability, ModelProvider, ModelRequest, ModelResponse

# keyword -> (tool_name, tool_input, step_description)
_KNOWN_TOOL_KEYWORDS: dict[str, tuple[str, dict[str, Any], str]] = {
    "مرحبا": (
        "demo_tool",
        {"message": "مرحباً! أنا مساعدك الشخصي الذكي. كيف يمكنني مساعدتك؟"},
        "Respond to the greeting",
    ),
    "مربحا": (
        "demo_tool",
        {"message": "مرحباً! أنا مساعدك الشخصي الذكي. كيف يمكنني مساعدتك؟"},
        "Respond to the greeting",
    ),
    "hello": (
        "demo_tool",
        {"message": "Hello! I am your personal AI assistant. How can I help?"},
        "Respond to the greeting",
    ),
    "كيف انت": ("demo_tool", {"message": "أنا بخير وجاهز لمساعدتك."}, "Respond to the user"),
    "demo tool": ("demo_tool", {"message": "hello from the agent core"}, "Run the demo tool"),
    "demo_tool": ("demo_tool", {"message": "hello from the agent core"}, "Run the demo tool"),
}


class MockModelProvider(ModelProvider):
    name = "mock"
    capabilities = frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT})

    def __init__(self, responses: Sequence[str] | None = None, model: str = "mock-1") -> None:
        self._responses = list(responses) if responses is not None else []
        self._model = model
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._responses:
            # Pop the next canned response; the last one repeats forever.
            content = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        else:
            content = self._keyword_plan(request)
        return ModelResponse(content=content, model=self._model)

    def _keyword_plan(self, request: ModelRequest) -> str:
        user_message = request.messages[-1].content.lower() if request.messages else ""
        for keyword, (tool_name, tool_input, description) in _KNOWN_TOOL_KEYWORDS.items():
            if keyword in user_message:
                plan = {
                    "steps": [
                        {"tool_name": tool_name, "description": description, "input": tool_input}
                    ]
                }
                return json.dumps(plan)
        raise ProviderError(
            "MockModelProvider: no scripted response and no keyword match for the request; "
            "pass `responses` for deterministic tests"
        )
