"""Tests for environment-based configuration."""

import os
import json
import pytest
from unittest.mock import patch

from llm_rate_guard.env_config import (
    get_env,
    get_env_bool,
    get_env_int,
    get_env_float,
    load_providers_from_env,
    load_config_from_env,
)
from llm_rate_guard.config import ProviderType


class TestEnvHelpers:
    """Tests for environment helper functions."""

    def test_get_env(self):
        """get_env returns environment value or default."""
        with patch.dict(os.environ, {"TEST_VAR": "value"}):
            assert get_env("TEST_VAR") == "value"
            assert get_env("MISSING_VAR") is None
            assert get_env("MISSING_VAR", "default") == "default"

    def test_get_env_bool(self):
        """get_env_bool parses boolean values."""
        with patch.dict(os.environ, {
            "TRUE_VAR": "true",
            "FALSE_VAR": "false",
            "YES_VAR": "yes",
            "ONE_VAR": "1",
        }):
            assert get_env_bool("TRUE_VAR") is True
            assert get_env_bool("FALSE_VAR") is False
            assert get_env_bool("YES_VAR") is True
            assert get_env_bool("ONE_VAR") is True
            assert get_env_bool("MISSING_VAR") is False
            assert get_env_bool("MISSING_VAR", True) is True

    def test_get_env_int(self):
        """get_env_int parses integer values."""
        with patch.dict(os.environ, {
            "INT_VAR": "42",
            "INVALID_VAR": "not_a_number",
        }):
            assert get_env_int("INT_VAR") == 42
            assert get_env_int("INVALID_VAR") == 0
            assert get_env_int("MISSING_VAR") == 0
            assert get_env_int("MISSING_VAR", 100) == 100

    def test_get_env_float(self):
        """get_env_float parses float values."""
        with patch.dict(os.environ, {
            "FLOAT_VAR": "3.14",
            "INVALID_VAR": "not_a_number",
        }):
            assert get_env_float("FLOAT_VAR") == 3.14
            assert get_env_float("INVALID_VAR") == 0.0
            assert get_env_float("MISSING_VAR", 1.5) == 1.5


class TestLoadProvidersFromEnv:
    """Tests for provider loading from environment."""

    def test_load_from_json(self):
        """Load providers from JSON environment variable."""
        providers_json = json.dumps([
            {"type": "openai", "model": "gpt-4", "api_key": "sk-test"},
            {"type": "bedrock", "model": "claude-3", "region": "us-east-1"},
        ])

        with patch.dict(os.environ, {"LLM_RATE_GUARD_PROVIDERS": providers_json}):
            providers = load_providers_from_env()

            assert len(providers) == 2
            assert providers[0].type == ProviderType.OPENAI
            assert providers[0].model == "gpt-4"
            assert providers[1].type == ProviderType.BEDROCK
            assert providers[1].region == "us-east-1"

    def test_load_from_numbered_vars(self):
        """Load providers from numbered environment variables."""
        env_vars = {
            "LLM_RATE_GUARD_PROVIDER_1_TYPE": "openai",
            "LLM_RATE_GUARD_PROVIDER_1_MODEL": "gpt-4",
            "LLM_RATE_GUARD_PROVIDER_1_API_KEY": "sk-test",
            "LLM_RATE_GUARD_PROVIDER_2_TYPE": "anthropic",
            "LLM_RATE_GUARD_PROVIDER_2_MODEL": "claude-3-opus",
            "LLM_RATE_GUARD_PROVIDER_2_API_KEY": "sk-ant-test",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            providers = load_providers_from_env()

            assert len(providers) == 2
            assert providers[0].type == ProviderType.OPENAI
            assert providers[1].type == ProviderType.ANTHROPIC

    def test_skip_incomplete_providers(self):
        """Skip providers with missing required fields."""
        env_vars = {
            "LLM_RATE_GUARD_PROVIDER_1_TYPE": "openai",
            # Missing MODEL
            "LLM_RATE_GUARD_PROVIDER_2_TYPE": "anthropic",
            "LLM_RATE_GUARD_PROVIDER_2_MODEL": "claude-3",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            providers = load_providers_from_env()

            assert len(providers) == 1
            assert providers[0].type == ProviderType.ANTHROPIC

    def test_empty_when_no_config(self):
        """Return empty list when no providers configured."""
        with patch.dict(os.environ, {}, clear=True):
            providers = load_providers_from_env()
            assert len(providers) == 0


class TestLoadConfigFromEnv:
    """Tests for full config loading from environment."""

    def test_returns_none_when_no_providers(self):
        """Return None when no providers configured."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config_from_env()
            assert config is None

    def test_load_full_config(self):
        """Load full config with all options."""
        env_vars = {
            "LLM_RATE_GUARD_PROVIDER_1_TYPE": "openai",
            "LLM_RATE_GUARD_PROVIDER_1_MODEL": "gpt-4",
            "LLM_RATE_GUARD_CACHE_ENABLED": "true",
            "LLM_RATE_GUARD_CACHE_TTL": "7200",
            "LLM_RATE_GUARD_FAILOVER_ENABLED": "true",
            "LLM_RATE_GUARD_COOLDOWN_SECONDS": "120",
            "LLM_RATE_GUARD_RETRY_MAX": "5",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = load_config_from_env()

            assert config is not None
            assert len(config.providers) == 1
            assert config.cache.enabled is True
            assert config.cache.ttl_seconds == 7200
            assert config.cooldown_seconds == 120.0
            assert config.retry.max_retries == 5

    def test_default_values_used(self):
        """Default values are used when env vars not set."""
        env_vars = {
            "LLM_RATE_GUARD_PROVIDER_1_TYPE": "openai",
            "LLM_RATE_GUARD_PROVIDER_1_MODEL": "gpt-4",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = load_config_from_env()

            assert config.cache.enabled is True  # Default
            assert config.failover_enabled is True  # Default
            assert config.queue_enabled is True  # Default
