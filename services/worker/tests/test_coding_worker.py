from pathlib import Path

import pytest
from coding_worker import run_verification


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_verification("shell", tmp_path)


def test_unsupported_option_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_verification("pytest", tmp_path, ["--capture=no"])
