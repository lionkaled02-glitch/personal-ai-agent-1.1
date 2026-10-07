"""Smoke test for the demo entry point (mock provider, no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BACKEND_MAIN = Path(__file__).resolve().parents[1] / "src" / "main.py"


def _load_main() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backend_main", BACKEND_MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend_main"] = module
    spec.loader.exec_module(module)
    return module


def test_demo_request_completes(capsys: pytest.CaptureFixture[str]) -> None:
    main_mod = _load_main()
    main_fn = main_mod.main
    rc = main_fn(["Run the demo tool."])
    out = capsys.readouterr().out
    assert rc == 0
    assert "COMPLETED" in out
    assert "demo_tool" in out
    assert "hello from the agent core" in out


def test_provider_misconfiguration_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # openai selected but no key: controlled core error, exit code 2, logged
    # as a single clean error line (no traceback).
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main_mod = _load_main()
    main_fn = main_mod.main
    with caplog.at_level("ERROR"):
        rc = main_fn(["Run the demo tool."])
    error_records = [r for r in caplog.records if r.levelno >= 40]
    assert rc == 2
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert "ProviderConfigurationError" in message
    assert "OPENAI_API_KEY" in message
