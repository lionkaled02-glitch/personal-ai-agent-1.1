"""Safe local media implementations for offline operation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from ..workspace import Workspace
from .errors import MediaProviderError, MediaValidationError
from .models import (
    MediaAsset,
    MediaFormat,
    MediaKind,
    MediaLimits,
    MediaRequest,
    MediaResult,
    VideoRequest,
)


class LocalImageGenerator:
    """Deterministic placeholder generator; never interprets the prompt as code."""

    def __init__(self, workspace: Workspace, limits: MediaLimits | None = None) -> None:
        self.workspace = workspace
        self.limits = (limits or MediaLimits()).bounded()

    def generate_image(self, request: MediaRequest, output_dir: Path) -> MediaResult:
        if len(request.prompt) > self.limits.max_prompt_chars:
            raise MediaValidationError("prompt exceeds media limit")
        if request.output_format not in {MediaFormat.PNG, MediaFormat.JPEG}:
            raise MediaValidationError("local image generator supports PNG/JPEG only")
        root = self.workspace.resolve(str(output_dir))
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:16]
        suffix = "jpg" if request.output_format == MediaFormat.JPEG else "png"
        path = root / f"generated-{digest}.{suffix}"
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise MediaProviderError("Pillow is required for local image generation") from exc
        image = Image.new("RGB", (request.width, request.height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, request.width - 1, request.height - 1), outline="black", width=4)
        draw.text((24, 24), "Local placeholder", fill="black")
        image.save(path, format="JPEG" if suffix == "jpg" else "PNG")
        size = path.stat().st_size
        if size > self.limits.max_output_bytes:
            path.unlink(missing_ok=True)
            raise MediaValidationError("generated output exceeds media limit")
        return MediaResult(
            success=True,
            assets=[
                MediaAsset(
                    id=str(uuid4()),
                    kind=MediaKind.IMAGE,
                    format=request.output_format,
                    relative_path=path.relative_to(self.workspace.root).as_posix(),
                    bytes=size,
                )
            ],
        )


class LocalVideoRenderer:
    """Placeholder boundary for an explicitly approved external video renderer.

    The agent-core package deliberately does not launch subprocesses. A real
    renderer belongs in an application/worker adapter that supplies its own
    sandbox and egress policy.
    """

    def __init__(self, workspace: Workspace, limits: MediaLimits | None = None) -> None:
        self.workspace = workspace
        self.limits = (limits or MediaLimits()).bounded()

    def render_video(self, request: VideoRequest, output_dir: Path) -> MediaResult:
        _ = output_dir
        if len(request.scenes) > self.limits.max_assets:
            raise MediaValidationError("too many video scenes")
        raise MediaProviderError("video rendering requires an application-level sandboxed provider")
