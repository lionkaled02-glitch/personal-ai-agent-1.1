"""Bounded provider-neutral media models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


class MediaFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"
    MP4 = "mp4"
    WAV = "wav"
    SRT = "srt"


class MediaLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_prompt_chars: int = 4000
    max_duration_s: int = 900
    max_frames: int = 108000
    max_output_bytes: int = 200_000_000
    max_assets: int = 100

    def bounded(self) -> MediaLimits:
        return MediaLimits(
            max_prompt_chars=max(1, min(self.max_prompt_chars, 20_000)),
            max_duration_s=max(1, min(self.max_duration_s, 3600)),
            max_frames=max(1, min(self.max_frames, 216_000)),
            max_output_bytes=max(1, min(self.max_output_bytes, 1_000_000_000)),
            max_assets=max(1, min(self.max_assets, 500)),
        )


class MediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=20_000)
    output_format: MediaFormat = MediaFormat.PNG
    width: int = Field(default=1024, ge=1, le=4096)
    height: int = Field(default=1024, ge=1, le=4096)
    duration_s: float | None = Field(default=None, gt=0, le=3600)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MediaAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: MediaKind
    format: MediaFormat
    relative_path: str
    bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MediaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    assets: list[MediaAsset] = Field(default_factory=list)
    error: str | None = None


class VideoScene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: str
    duration_s: float = Field(gt=0, le=3600)
    caption: str = Field(default="", max_length=4000)


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenes: list[VideoScene] = Field(min_length=1, max_length=100)
    output_name: str = Field(default="video.mp4", min_length=1, max_length=200)
    fps: int = Field(default=24, ge=1, le=60)
