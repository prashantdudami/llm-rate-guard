"""Tests for RateGuardClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from llm_rate_guard.client import RateGuardClient
from llm_rate_guard.config import ProviderConfig, ProviderType, RateGuardConfig
from llm_rate_guard.exceptions import ConfigurationError, AllProvidersExhausted
from llm_rate_guard.providers.base import CompletionResponse, Message, Usage


class TestMessageValidation:
    """Tests for message validation."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked router."""
        def create_mock_provider(config):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            mock.complete = AsyncMock(return_value=CompletionResponse(
                content="Hello!",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="mock",
                latency_ms=50.0,
            ))
            return mock

        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[
                    ProviderConfig(type=ProviderType.OPENAI, model="gpt-4"),
                ],
                cache_enabled=False,
            )
            return client

    @pytest.mark.asyncio
    async def test_empty_messages_rejected(self, mock_client):
        """Empty messages list raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            await mock_client.complete([])

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_role_rejected(self, mock_client):
        """Message without role raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            await mock_client.complete([{"content": "Hello"}])

        assert "role" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_content_rejected(self, mock_client):
        """Message without content raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            await mock_client.complete([{"role": "user"}])

        assert "content" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_role_rejected(self, mock_client):
        """Invalid role raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            await mock_client.complete([{"role": "invalid", "content": "Hello"}])

        assert "role" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_valid_roles_accepted(self, mock_client):
        """Valid roles are accepted."""
        # Test all valid roles
        for role in ["system", "user", "assistant"]:
            response = await mock_client.complete([{"role": role, "content": "Test"}])
            assert response.content == "Hello!"

    @pytest.mark.asyncio
    async def test_message_object_accepted(self, mock_client):
        """Message objects are accepted."""
        response = await mock_client.complete([
            Message(role="user", content="Hello")
        ])
        assert response.content == "Hello!"

    @pytest.mark.asyncio
    async def test_too_many_messages_rejected(self):
        """Too many messages raises error."""
        def create_mock_provider(config):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            return mock

        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            config = RateGuardConfig(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                max_messages_per_request=5,
            )
            client = RateGuardClient(config=config)

            with pytest.raises(ConfigurationError) as exc_info:
                await client.complete([
                    {"role": "user", "content": f"Message {i}"}
                    for i in range(10)
                ])

            assert "too many" in str(exc_info.value).lower()


class TestSecureApiKey:
    """Tests for secure API key handling."""

    def test_api_key_not_in_repr(self):
        """API key should not appear in repr."""
        config = ProviderConfig(
            type=ProviderType.OPENAI,
            model="gpt-4",
            api_key="sk-secret-key-12345",
        )

        repr_str = repr(config)
        assert "sk-secret-key-12345" not in repr_str
        assert "secret" not in repr_str.lower()

    def test_api_key_accessible_via_method(self):
        """API key can be retrieved via get_api_key method."""
        config = ProviderConfig(
            type=ProviderType.OPENAI,
            model="gpt-4",
            api_key="sk-secret-key-12345",
        )

        assert config.get_api_key() == "sk-secret-key-12345"

    def test_none_api_key(self):
        """None API key returns None from get_api_key."""
        config = ProviderConfig(
            type=ProviderType.OPENAI,
            model="gpt-4",
        )

        assert config.get_api_key() is None


class TestCircuitBreaker:
    """Tests for circuit breaker configuration."""

    def test_circuit_breaker_config_defaults(self):
        """Circuit breaker has sensible defaults."""
        config = RateGuardConfig(
            providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")]
        )

        assert config.circuit_breaker.enabled is True
        assert config.circuit_breaker.failure_threshold == 5
        assert config.circuit_breaker.success_threshold == 2
        assert config.circuit_breaker.half_open_timeout == 30.0

    def test_circuit_breaker_config_custom(self):
        """Circuit breaker can be customized."""
        from llm_rate_guard.config import CircuitBreakerConfig

        config = RateGuardConfig(
            providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            circuit_breaker=CircuitBreakerConfig(
                enabled=True,
                failure_threshold=10,
                success_threshold=3,
                half_open_timeout=60.0,
            ),
        )

        assert config.circuit_breaker.failure_threshold == 10
        assert config.circuit_breaker.success_threshold == 3
        assert config.circuit_breaker.half_open_timeout == 60.0


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_extra_fields_rejected(self):
        """Extra/typo fields in config are rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4",
                modell="gpt-4",  # Typo - should be rejected
            )

    def test_model_too_long_rejected(self):
        """Model name too long is rejected."""
        with pytest.raises(Exception):
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="x" * 300,  # Too long
            )

    def test_timeout_configurable(self):
        """Timeout is configurable per provider."""
        config = ProviderConfig(
            type=ProviderType.OPENAI,
            model="gpt-4",
            timeout_seconds=30.0,
        )

        assert config.timeout_seconds == 30.0

    def test_timeout_has_bounds(self):
        """Timeout must be within bounds."""
        with pytest.raises(Exception):
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4",
                timeout_seconds=0.5,  # Too low
            )

        with pytest.raises(Exception):
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4",
                timeout_seconds=1000,  # Too high
            )
