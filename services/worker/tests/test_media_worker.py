from pathlib import Path

from media_worker import FFmpegRenderer, RenderError


def test_renderer_rejects_non_mp4(tmp_path: Path) -> None:
    renderer = FFmpegRenderer()
    try:
        renderer.render_image_sequence("frames/%04d.png", tmp_path / "out.txt")
    except RenderError as exc:
        assert "mp4" in str(exc)
    else:
        raise AssertionError("expected validation failure")
