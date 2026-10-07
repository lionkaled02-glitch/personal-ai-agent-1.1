from __future__ import annotations

from pathlib import Path

import pytest
from agent_core import (
    LocalImageGenerator,
    LocalVideoRenderer,
    MediaProviderError,
    MediaRequest,
    VideoRequest,
    VideoScene,
    Workspace,
)


def test_local_image_generation_stays_in_workspace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    result = LocalImageGenerator(workspace).generate_image(
        MediaRequest(prompt="hello"), Path("data/generated")
    )
    assert result.success
    assert result.assets[0].relative_path.startswith("data/generated/")
    assert (workspace.root / result.assets[0].relative_path).is_file()


def test_video_renderer_fails_closed_in_core(tmp_path: Path) -> None:
    renderer = LocalVideoRenderer(Workspace(tmp_path / "workspace"))
    with pytest.raises(MediaProviderError):
        renderer.render_video(
            VideoRequest(scenes=[VideoScene(image_path="x.png", duration_s=1)]),
            Path("data/generated"),
        )
