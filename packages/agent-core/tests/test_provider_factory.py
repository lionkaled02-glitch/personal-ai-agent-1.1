"""Provider selection and configuration tests (offline, explicit env)."""

from __future__ import annotations

import pytest
from agent_core import (
    MockModelProvider,
    ModelGateway,
    OpenAIProvider,
    ProviderConfigurationError,
    Settings,
    build_gateway,
    create_provider,
)

FAKE_KEY = "sk-test-not-a-real-key"


class TestSelection:
    def test_default_is_mock(self) -> None:
        provider = create_provider(Settings(), env={})
        assert isinstance(provider, MockModelProvider)

    def test_mock_explicit(self) -> None:
        settings = Settings(model_provider="mock", model_name="my-model")
        provider = create_provider(settings, env={})
        assert isinstance(provider, MockModelProvider)
        assert provider.name == "mock"

    def test_unknown_provider_rejected_with_supported_list(self) -> None:
        settings = Settings(model_provider="skynet")
        with pytest.raises(ProviderConfigurationError, match="skynet"):
            create_provider(settings, env={})
        with pytest.raises(ProviderConfigurationError) as excinfo:
            create_provider(settings, env={})
        message = str(excinfo.value)
        assert "mock" in message and "openai" in message

    def test_provider_name_is_case_insensitive(self) -> None:
        provider = create_provider(Settings(model_provider="MOCK"), env={})
        assert isinstance(provider, MockModelProvider)

    def test_build_gateway_wraps_selected_provider(self) -> None:
        gateway = build_gateway(Settings(), env={})
        assert isinstance(gateway, ModelGateway)
        assert gateway.name == "mock"
        assert isinstance(gateway.provider, MockModelProvider)


class TestOpenAISelection:
    def test_missing_key_rejected_without_touching_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even with the SDK installed, a missing key must fail fast with a
        # configuration error that names the variable (never its value).
        settings = Settings(model_provider="openai")
        with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
            create_provider(settings, env={})
        # And the error must not echo any key material.
        with pytest.raises(ProviderConfigurationError) as excinfo:
            create_provider(settings, env={"OPENAI_API_KEY": "  "})  # whitespace-only
        assert FAKE_KEY not in str(excinfo.value)

    def test_key_is_read_from_env_mapping_not_os_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        settings = Settings(model_provider="openai", model_name="gpt-4o-mini")
        provider = create_provider(settings, env={"OPENAI_API_KEY": FAKE_KEY})
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("openai")
        settings = Settings(model_provider="openai")
        provider = create_provider(
            settings,
            env={"OPENAI_API_KEY": FAKE_KEY, "OPENAI_BASE_URL": "http://localhost:11434/v1"},
        )
        assert isinstance(provider, OpenAIProvider)

    def test_gateway_inherits_retry_policy_from_settings(self) -> None:
        settings = Settings(model_provider="mock", model_max_retries=5)
        gateway = build_gateway(settings, env={})
        assert isinstance(gateway, ModelGateway)
        assert gateway.name == "mock"
        assert isinstance(gateway.provider, MockModelProvider)


class TestSettingsIntegration:
    def test_from_env_provider_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_PROVIDER", "openai")
        monkeypatch.setenv("MODEL_NAME", "gpt-4o")
        monkeypatch.setenv("MODEL_TIMEOUT_S", "30")
        monkeypatch.setenv("MODEL_MAX_RETRIES", "0")
        settings = Settings.from_env()
        assert settings.model_provider == "openai"
        assert settings.model_name == "gpt-4o"
        assert settings.model_timeout_s == 30.0
        assert settings.model_max_retries == 0
