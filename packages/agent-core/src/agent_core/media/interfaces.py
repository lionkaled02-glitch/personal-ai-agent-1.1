"""Provider interfaces for media generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import MediaRequest, MediaResult, VideoRequest


class ImageGenerationProvider(Protocol):
    def generate_image(self, request: MediaRequest, output_dir: Path) -> MediaResult: ...


class VideoGenerationProvider(Protocol):
    def render_video(self, request: VideoRequest, output_dir: Path) -> MediaResult: ...
