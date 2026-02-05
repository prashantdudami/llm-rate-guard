"""Tests for router module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from llm_rate_guard.config import ProviderConfig, ProviderType, RateGuardConfig
from llm_rate_guard.exceptions import AllProvidersExhausted, RateLimitExceeded
from llm_rate_guard.providers.base import CompletionResponse, Message, Usage
from llm_rate_guard.router import MultiRouter, ProviderHealth, ManagedProvider


class TestProviderHealth:
    """Tests for ProviderHealth class."""

    def test_initial_state(self):
        """Health starts healthy with zero counters."""
        health = ProviderHealth(provider_id="test")
        
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_requests == 0

    def test_record_success(self):
        """Success resets failures and updates stats."""
        health = ProviderHealth(provider_id="test")
        health.consecutive_failures = 5
        health.is_healthy = False
        
        health.record_success(latency_ms=100.0)
        
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_requests == 1
        assert health.avg_latency_ms == 100.0

    def test_record_failure(self):
        """Failure increments counters."""
        health = ProviderHealth(provider_id="test")
        
        health.record_failure()
        health.record_failure(is_rate_limit=True)
        
        assert health.consecutive_failures == 2
        assert health.total_failures == 2
        assert health.total_rate_limits == 1

    def test_cooldown(self):
        """Cooldown sets provider unhealthy."""
        health = ProviderHealth(provider_id="test")
        
        health.set_cooldown(60.0)
        
        assert health.is_healthy is False
        assert health.cooldown_until is not None

    def test_cooldown_expiry(self):
        """Cooldown expires correctly."""
        health = ProviderHealth(provider_id="test")
        
        # Set very short cooldown
        health.set_cooldown(0.01)
        
        import time
        time.sleep(0.02)
        
        # Check cooldown should restore health
        result = health.check_cooldown()
        
        assert result is True
        assert health.is_healthy is True
        assert health.cooldown_until is None


class TestMultiRouter:
    """Tests for MultiRouter class."""

    @pytest.fixture
    def mock_provider_class(self):
        """Create a mock provider class."""
        def create_mock():
            mock_instance = MagicMock()
            mock_instance.provider_name = "mock"
            mock_instance.config = MagicMock()
            mock_instance.config.rpm_limit = None
            mock_instance.config.tpm_limit = None
            type(mock_instance).rpm_limit = PropertyMock(return_value=100)
            type(mock_instance).tpm_limit = PropertyMock(return_value=10000)
            mock_instance.complete = AsyncMock(return_value=CompletionResponse(
                content="Hello!",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="mock",
                latency_ms=50.0,
            ))
            return mock_instance
        
        mock_class = MagicMock(side_effect=lambda config: create_mock())
        return mock_class

    @pytest.mark.asyncio
    async def test_route_success(self, mock_provider_class):
        """Successful routing to provider."""
        config = RateGuardConfig(
            providers=[
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4"),
            ]
        )
        
        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_provider_class):
            router = MultiRouter(config)
            
            response = await router.route(
                messages=[Message(role="user", content="Hi")],
            )
            
            assert response.content == "Hello!"
            assert router.healthy_provider_count == 1

    @pytest.mark.asyncio
    async def test_failover_on_rate_limit(self, mock_provider_class):
        """Failover to next provider on rate limit."""
        config = RateGuardConfig(
            providers=[
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4"),
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4-backup"),
            ],
            failover_enabled=True,
        )
        
        # Create mocks that track calls
        call_count = [0]
        
        def create_mock_provider(config):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            
            async def mock_complete(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RateLimitExceeded("Rate limited")
                return CompletionResponse(
                    content="Success!",
                    model="gpt-4-backup",
                    usage=Usage(input_tokens=10, output_tokens=5),
                    provider="mock",
                    latency_ms=50.0,
                )
            
            mock.complete = mock_complete
            return mock
        
        mock_class = MagicMock(side_effect=create_mock_provider)
        
        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            router = MultiRouter(config)
            
            response = await router.route(
                messages=[Message(role="user", content="Hi")],
            )
            
            assert response.content == "Success!"
            assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_all_providers_exhausted(self, mock_provider_class):
        """Raises when all providers fail."""
        config = RateGuardConfig(
            providers=[
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4"),
            ],
            failover_enabled=True,
        )
        
        def create_failing_mock(cfg):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            mock.complete = AsyncMock(side_effect=RateLimitExceeded("Rate limited"))
            return mock
        
        mock_class = MagicMock(side_effect=create_failing_mock)
        
        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            router = MultiRouter(config)
            
            with pytest.raises(AllProvidersExhausted):
                await router.route(
                    messages=[Message(role="user", content="Hi")],
                )

    @pytest.mark.asyncio
    async def test_cooldown_skips_provider(self, mock_provider_class):
        """Providers in cooldown are skipped."""
        config = RateGuardConfig(
            providers=[
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4"),
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4-backup"),
            ],
            failover_enabled=True,
            cooldown_seconds=60.0,
        )
        
        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_provider_class):
            router = MultiRouter(config)
            
            # Put first provider in cooldown
            router._providers[0].health.set_cooldown(60.0)
            
            # Should only have one healthy provider
            assert router.healthy_provider_count == 1

    def test_get_provider_stats(self, mock_provider_class):
        """Provider stats are returned correctly."""
        config = RateGuardConfig(
            providers=[
                ProviderConfig(type=ProviderType.OPENAI, model="gpt-4", region="us"),
            ]
        )
        
        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_provider_class):
            router = MultiRouter(config)
            
            stats = router.get_provider_stats()
            
            assert len(stats) == 1
            assert stats[0]["type"] == "openai"
            assert stats[0]["model"] == "gpt-4"
            assert stats[0]["is_healthy"] is True
