from __future__ import annotations

import re

from .models import CreationLimits, ScriptRequest, ScriptResult, Storyboard, StoryboardScene


class CreationService:
    """Deterministic planning layer for scripts/storyboards.

    It deliberately does not call external model APIs. Applications may feed
    these bounded structures to a configured provider later.
    """

    def __init__(self, limits: CreationLimits | None = None) -> None:
        self.limits = (limits or CreationLimits()).bounded()

    def create_script(self, request: ScriptRequest) -> ScriptResult:
        topic = request.topic.strip()
        if len(topic) > self.limits.max_topic_chars:
            raise ValueError("topic exceeds creation limit")
        words = max(1, round(request.target_duration_s * 2.2))
        narration = (
            f"Today we will explore {topic}. "
            f"We will introduce the key idea, explain the important points, "
            f"and finish with a concise takeaway."
        )
        narration = (narration + " ") * max(1, words // max(1, len(narration.split())))
        narration = " ".join(narration.split())[: self.limits.max_script_chars]
        sections = ["Hook", "Main explanation", "Key takeaway"]
        return ScriptResult(
            title=topic[:500],
            sections=sections,
            narration=narration,
            estimated_duration_s=max(1, min(request.target_duration_s, 3600)),
        )

    def storyboard(self, script: ScriptResult, scenes: int = 5) -> Storyboard:
        count = max(1, min(scenes, self.limits.max_scenes))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script.narration) if s.strip()]
        if not sentences:
            sentences = [script.narration]
        result: list[StoryboardScene] = []
        for i in range(count):
            narration = sentences[i % len(sentences)][:4000]
            result.append(
                StoryboardScene(
                    index=i + 1,
                    narration=narration,
                    visual_prompt=f"Visual illustration for: {narration}",
                    duration_s=max(1.0, script.estimated_duration_s / count),
                )
            )
        return Storyboard(title=script.title, scenes=result)
