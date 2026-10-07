"""Settings tests: environment-based configuration with safe defaults."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from agent_core import Settings


class TestFromEnv:
    def test_defaults_when_env_empty(self) -> None:
        settings = Settings.from_env(env={})
        assert settings.agent_name == "personal-agent"
        assert settings.log_level == "INFO"
        assert settings.data_root == Path("data")
        # Phase 1 model-gateway defaults (safe: offline mock provider).
        assert settings.model_provider == "mock"
        assert settings.model_name == ""
        assert settings.model_timeout_s == 60.0
        assert settings.model_max_retries == 2

    def test_overrides(self) -> None:
        settings = Settings.from_env(
            env={
                "AGENT_NAME": "my-agent",
                "LOG_LEVEL": "debug",
                "DATA_ROOT": "/tmp/agent-data",
            }
        )
        assert settings.agent_name == "my-agent"
        assert settings.log_level == "debug"
        assert settings.data_root == Path("/tmp/agent-data")

    def test_partial_env_falls_back_to_defaults(self) -> None:
        settings = Settings.from_env(env={"AGENT_NAME": "only-name"})
        assert settings.agent_name == "only-name"
        assert settings.log_level == "INFO"
        assert settings.data_root == Path("data")

    def test_reads_os_environ_when_no_mapping_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_NAME", "from-os")
        settings = Settings.from_env()
        assert settings.agent_name == "from-os"

    def test_configure_logging_sets_level(self) -> None:
        settings = Settings(log_level="WARNING")
        settings.configure_logging()
        assert logging.getLogger().level == logging.WARNING


class TestMemorySettings:
    def test_memory_defaults(self) -> None:
        settings = Settings.from_env(env={})
        assert settings.memory_max_items == 1_000
        assert settings.memory_max_content_chars == 4_000
        assert settings.memory_max_metadata_bytes == 4_096
        assert settings.memory_max_recall_results == 10
        assert settings.memory_max_context_chars == 8_000
        assert settings.memory_max_context_items == 20
        assert settings.memory_short_term_ttl_s == 3_600
        assert settings.memory_working_ttl_s == 86_400

    def test_memory_env_overrides(self) -> None:
        settings = Settings.from_env(
            env={
                "MEMORY_MAX_ITEMS": "500",
                "MEMORY_MAX_CONTENT_CHARS": "2000",
                "MEMORY_MAX_METADATA_BYTES": "2048",
                "MEMORY_MAX_RECALL_RESULTS": "5",
                "MEMORY_MAX_CONTEXT_CHARS": "4000",
                "MEMORY_MAX_CONTEXT_ITEMS": "10",
                "MEMORY_SHORT_TERM_TTL_S": "1800",
                "MEMORY_WORKING_TTL_S": "43200",
            }
        )
        assert settings.memory_max_items == 500
        assert settings.memory_max_content_chars == 2000
        assert settings.memory_max_metadata_bytes == 2048
        assert settings.memory_max_recall_results == 5
        assert settings.memory_max_context_chars == 4000
        assert settings.memory_max_context_items == 10
        assert settings.memory_short_term_ttl_s == 1800
        assert settings.memory_working_ttl_s == 43200

    def test_memory_limits_built_from_settings(self) -> None:
        from agent_core import MemoryLimits

        settings = Settings.from_env(env={"MEMORY_MAX_ITEMS": "7"})
        limits = MemoryLimits.from_settings(settings)
        assert limits.max_items == 7
        # Unset variables keep their defaults.
        assert limits.max_content_chars == 4_000


def test_optional_local_provider_settings() -> None:
    settings = Settings.from_env({"COMPUTER_PROVIDER": "WINDOWS", "BROWSER_PROVIDER": "Playwright"})
    assert settings.computer_provider == "windows"
    assert settings.browser_provider == "playwright"
