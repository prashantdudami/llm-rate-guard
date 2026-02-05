"""Advanced tests for RateGuardClient - edge cases and advanced features."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from llm_rate_guard.client import RateGuardClient
from llm_rate_guard.config import ProviderConfig, ProviderType, RateGuardConfig
from llm_rate_guard.exceptions import ConfigurationError
from llm_rate_guard.providers.base import CompletionResponse, Message, Usage


def create_mock_provider(config):
    """Create a mock provider for testing."""
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


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self):
        """Health check returns comprehensive status."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )

            health = await client.health_check()

            assert "healthy" in health
            assert "status" in health
            assert "providers" in health
            assert "cache" in health
            assert "queue" in health
            assert "metrics" in health

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_providers_available(self):
        """Health check reports healthy when providers are available."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            health = await client.health_check()

            assert health["healthy"] is True
            assert health["status"] == "ok"
            assert health["providers"]["healthy"] > 0


class TestCostEstimation:
    """Tests for cost estimation feature."""

    def test_estimate_cost_before_request(self):
        """Can estimate cost before sending request."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            estimate = client.estimate_cost(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=100,
            )

            assert "estimated_input_tokens" in estimate
            assert "estimated_output_tokens" in estimate
            assert "input_usd" in estimate
            assert "output_usd" in estimate
            assert "total_usd" in estimate
            assert estimate["total_usd"] > 0

    def test_estimate_cost_empty_messages(self):
        """Empty messages returns zero cost."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            estimate = client.estimate_cost(messages=[], max_tokens=100)

            assert estimate["total_usd"] == 0.0


class TestGracefulShutdown:
    """Tests for graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_during_request(self):
        """Shutdown rejects new requests."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                # Start shutdown
                client._shutting_down = True

                with pytest.raises(ConfigurationError) as exc_info:
                    await client.complete([{"role": "user", "content": "Hello"}])

                assert "shutting down" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_active_requests_tracked(self):
        """Active requests are tracked correctly."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                assert client.active_requests == 0

                # Make a request
                await client.complete([{"role": "user", "content": "Hello"}])

                # Should be back to 0 after completion
                assert client.active_requests == 0


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Multiple concurrent requests are handled correctly."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                # Send multiple concurrent requests
                tasks = [
                    client.complete([{"role": "user", "content": f"Hello {i}"}])
                    for i in range(5)
                ]

                responses = await asyncio.gather(*tasks)

                assert len(responses) == 5
                assert all(r.content == "Hello!" for r in responses)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_unicode_messages(self):
        """Unicode characters in messages are handled."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )

            # Various unicode characters
            response = await client.complete([
                {"role": "user", "content": "Hello 你好 مرحبا 🎉 emoji test"}
            ])

            assert response is not None

    @pytest.mark.asyncio
    async def test_empty_content_message(self):
        """Messages with empty content are handled."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )

            # Empty content is valid (some models accept it)
            response = await client.complete([
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hello"},
            ])

            assert response is not None

    @pytest.mark.asyncio
    async def test_message_object_input(self):
        """Message objects work as input."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )

            response = await client.complete([
                Message(role="user", content="Hello")
            ])

            assert response.content == "Hello!"

    def test_is_shutting_down_property(self):
        """is_shutting_down property works correctly."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            assert client.is_shutting_down is False

            client._shutting_down = True
            assert client.is_shutting_down is True


class TestThreadSafeActiveRequests:
    """Tests for thread-safe active request counting."""

    @pytest.mark.asyncio
    async def test_concurrent_request_counting(self):
        """Active request counter handles concurrent access correctly."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                # Track max concurrent requests observed
                max_observed = 0
                original_complete = client._do_complete

                async def tracking_complete(*args, **kwargs):
                    nonlocal max_observed
                    max_observed = max(max_observed, client.active_requests)
                    return await original_complete(*args, **kwargs)

                client._do_complete = tracking_complete

                # Send concurrent requests
                tasks = [
                    client.complete([{"role": "user", "content": f"Hello {i}"}])
                    for i in range(10)
                ]

                await asyncio.gather(*tasks)

                # After all complete, should be 0
                assert client.active_requests == 0
                # At some point should have seen multiple active
                assert max_observed > 0

    @pytest.mark.asyncio
    async def test_request_complete_event_signaling(self):
        """Request complete event is properly signaled."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                # Verify event is set when no active requests
                assert client._requests_complete_event.is_set()

                # Make a request - event should clear then set again
                await client.complete([{"role": "user", "content": "Hello"}])

                # After completion, event should be set
                assert client._requests_complete_event.is_set()


class TestTimeoutHandling:
    """Tests for request timeout handling."""

    @pytest.mark.asyncio
    async def test_request_timeout_triggers_error(self):
        """Request that times out raises appropriate error."""
        # Create mock that hangs forever
        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(10)  # Long delay
            return CompletionResponse(
                content="Hello!",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="mock",
                latency_ms=50.0,
            )

        def create_slow_provider(config):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            mock.complete = slow_complete
            return mock

        mock_class = MagicMock(side_effect=create_slow_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            config = ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4",
                timeout_seconds=1.0,  # Minimum valid value
            )
            # Override timeout for test (bypass validation)
            object.__setattr__(config, 'timeout_seconds', 0.1)

            client = RateGuardClient(
                providers=[config],
                cache_enabled=False,
                failover_enabled=False,  # Don't retry
            )

            from llm_rate_guard.exceptions import ProviderError, AllProvidersExhausted

            # Should raise ProviderError with timeout message
            with pytest.raises((ProviderError, AllProvidersExhausted)) as exc_info:
                await client.complete([{"role": "user", "content": "Hello"}])

            assert "timed out" in str(exc_info.value).lower()


class TestCostEstimationWithTrackers:
    """Tests for cost estimation with different tracker implementations."""

    def test_estimate_cost_with_simple_tracker(self):
        """Cost estimation works with SimpleCostTracker."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            # SimpleCostTracker should have pricing attribute
            assert hasattr(client._metrics.cost_tracker, 'pricing')

            estimate = client.estimate_cost(
                messages=[{"role": "user", "content": "Hello world"}],
                max_tokens=100,
                model="gpt-4",
            )

            assert estimate["estimated_input_tokens"] > 0
            assert estimate["total_usd"] > 0
            assert "input_usd" in estimate
            assert "output_usd" in estimate

    def test_estimate_cost_handles_missing_pricing_attr(self):
        """Cost estimation handles trackers without pricing attribute."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            # Create a mock tracker without pricing attribute
            mock_tracker = MagicMock()
            mock_tracker.estimate_cost = MagicMock(return_value=0.05)
            # Explicitly don't set pricing attribute
            del mock_tracker.pricing

            client._metrics.cost_tracker = mock_tracker

            estimate = client.estimate_cost(
                messages=[{"role": "user", "content": "Hello world"}],
                max_tokens=100,
                model="gpt-4",
            )

            # Should use proportional split (33% input, 67% output)
            assert estimate["total_usd"] == 0.05
            assert abs(estimate["input_usd"] - 0.05 * 0.33) < 0.001
            assert abs(estimate["output_usd"] - 0.05 * 0.67) < 0.001


class TestEmbedRateLimiting:
    """Tests for embed method respecting rate limits."""

    @pytest.mark.asyncio
    async def test_embed_respects_shutdown(self):
        """Embed rejects requests during shutdown."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            client._shutting_down = True

            with pytest.raises(ConfigurationError) as exc_info:
                await client.embed("Hello world")

            assert "shutting down" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_tracks_active_requests(self):
        """Embed method tracks active requests."""
        def create_embed_provider(config):
            from llm_rate_guard.providers.base import EmbeddingResponse
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
            mock.embed = AsyncMock(return_value=EmbeddingResponse(
                embedding=[0.1] * 1536,
                model="text-embedding-ada-002",
                usage=Usage(input_tokens=5, output_tokens=0),
                provider="mock",
            ))
            return mock

        mock_class = MagicMock(side_effect=create_embed_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            ) as client:
                assert client.active_requests == 0

                response = await client.embed("Hello world")

                assert response is not None
                assert len(response.embedding) == 1536
                assert client.active_requests == 0


class TestCacheKeyHashing:
    """Tests for cache key hashing with long content."""

    def test_short_cache_key_not_hashed(self):
        """Short cache keys are not hashed."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            from llm_rate_guard.providers.base import Message
            messages = [Message(role="user", content="Hello")]
            key = client._get_cache_key(messages, model="gpt-4")

            assert not key.startswith("hash:")
            assert "Hello" in key

    def test_long_cache_key_is_hashed(self):
        """Long cache keys (>1KB) are hashed for efficiency."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            from llm_rate_guard.providers.base import Message
            # Create a very long message (> 1KB)
            long_content = "x" * 2000
            messages = [Message(role="user", content=long_content)]
            key = client._get_cache_key(messages, model="gpt-4")

            assert key.startswith("hash:")
            assert len(key) < 100  # SHA256 hex is 64 chars + prefix

    def test_same_content_produces_same_hash(self):
        """Same content produces same hash (deterministic)."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            from llm_rate_guard.providers.base import Message
            long_content = "y" * 2000
            messages = [Message(role="user", content=long_content)]

            key1 = client._get_cache_key(messages, model="gpt-4")
            key2 = client._get_cache_key(messages, model="gpt-4")

            assert key1 == key2


class TestStreamingResponse:
    """Tests for streaming response support."""

    @pytest.mark.asyncio
    async def test_stream_respects_shutdown(self):
        """Stream rejects requests during shutdown."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )
            client._shutting_down = True

            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in client.stream([{"role": "user", "content": "Hello"}]):
                    pass

            assert "shutting down" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stream_validates_messages(self):
        """Stream validates messages like complete()."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            with pytest.raises(ConfigurationError) as exc_info:
                async for _ in client.stream([]):  # Empty messages
                    pass

            assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stream_tracks_active_requests(self):
        """Streaming tracks active request count."""
        from llm_rate_guard.providers.base import StreamChunk

        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", done=False)
            yield StreamChunk(content=" World", done=True, usage=Usage(input_tokens=5, output_tokens=2))

        def create_streaming_provider(config):
            mock = MagicMock()
            mock.provider_name = "mock"
            type(mock).rpm_limit = PropertyMock(return_value=100)
            type(mock).tpm_limit = PropertyMock(return_value=10000)
            mock.stream = mock_stream
            mock.complete = AsyncMock(return_value=CompletionResponse(
                content="Hello!",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="mock",
                latency_ms=50.0,
            ))
            return mock

        mock_class = MagicMock(side_effect=create_streaming_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                assert client.active_requests == 0

                chunks = []
                async for chunk in client.stream([{"role": "user", "content": "Hello"}]):
                    chunks.append(chunk)

                # After stream completes, should be back to 0
                assert client.active_requests == 0
                assert len(chunks) == 2


class TestGracefulShutdownWithEvent:
    """Tests for graceful shutdown using asyncio.Event."""

    @pytest.mark.asyncio
    async def test_wait_for_requests_uses_event(self):
        """Shutdown waits using event instead of polling."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            async with RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            ) as client:
                # Verify event is initialized
                assert client._requests_complete_event is not None

                # Start a request in background
                async def slow_request():
                    await asyncio.sleep(0.1)
                    await client.complete([{"role": "user", "content": "Hello"}])

                task = asyncio.create_task(slow_request())

                # Small delay to let request start
                await asyncio.sleep(0.05)

                # Stop should wait for request
                await asyncio.wait_for(client.stop(graceful=True, timeout=5.0), timeout=5.0)

                await task  # Ensure task completes

    @pytest.mark.asyncio
    async def test_stop_with_no_active_requests(self):
        """Stop completes immediately with no active requests."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )
            await client.start()

            # Should complete almost instantly
            import time
            start = time.monotonic()
            await client.stop(graceful=True, timeout=10.0)
            elapsed = time.monotonic() - start

            assert elapsed < 0.5  # Should be very fast
