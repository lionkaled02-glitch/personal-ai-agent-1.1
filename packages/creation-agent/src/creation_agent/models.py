from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CreationKind(StrEnum):
    SCRIPT = "script"
    IMAGE = "image"
    VIDEO = "video"
    SUBTITLE = "subtitle"


class ScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=1, max_length=10_000)
    language: str = Field(default="en", min_length=2, max_length=20)
    target_duration_s: int = Field(default=60, ge=1, le=3600)
    tone: str = Field(default="clear", min_length=1, max_length=100)


class ScriptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    sections: list[str]
    narration: str
    estimated_duration_s: int = Field(ge=0)


class StoryboardScene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=1)
    narration: str = Field(max_length=4000)
    visual_prompt: str = Field(max_length=4000)
    duration_s: float = Field(gt=0, le=3600)


class Storyboard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(max_length=500)
    scenes: list[StoryboardScene] = Field(min_length=1, max_length=100)


class CreationLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_topic_chars: int = 10_000
    max_scenes: int = 100
    max_script_chars: int = 100_000

    def bounded(self) -> CreationLimits:
        return CreationLimits(
            max_topic_chars=max(1, min(self.max_topic_chars, 50_000)),
            max_scenes=max(1, min(self.max_scenes, 500)),
            max_script_chars=max(1, min(self.max_script_chars, 500_000)),
        )
