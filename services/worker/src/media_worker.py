from __future__ import annotations

import os
import subprocess
from pathlib import Path


class RenderError(RuntimeError):
    pass


class FFmpegRenderer:
    """Explicit argv-only FFmpeg adapter for the application worker.

    This is intentionally outside agent-core. The executable path is supplied
    by deployment configuration and shell execution is disabled.
    """

    def __init__(self, executable: str | None = None, timeout_s: int = 900) -> None:
        self.executable = executable or os.environ.get("FFMPEG_PATH", "ffmpeg")
        self.timeout_s = max(1, min(timeout_s, 3600))

    def render_image_sequence(self, input_pattern: str, output_path: Path, fps: int = 24) -> Path:
        if not input_pattern or "\x00" in input_pattern:
            raise RenderError("invalid input pattern")
        if output_path.suffix.lower() != ".mp4":
            raise RenderError("output must be an mp4 path")
        fps = max(1, min(fps, 60))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            self.executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RenderError("renderer unavailable or timed out") from exc
        if completed.returncode != 0:
            raise RenderError("renderer failed")
        if not output_path.is_file():
            raise RenderError("renderer produced no output")
        return output_path
