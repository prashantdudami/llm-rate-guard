"""Tests for sync wrappers and batch processing."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from llm_rate_guard import (
    RateGuardClient,
    RateGuardConfig,
    ProviderConfig,
    ProviderType,
    CacheConfig,
)
from llm_rate_guard.providers.base import CompletionResponse, EmbeddingResponse, Usage


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "test-provider"
    provider.model = "test-model"
    provider.priority = 1
    provider.weight = 1.0
    provider.rpm_limit = 100
    provider.tpm_limit = 100000
    type(provider).is_healthy = PropertyMock(return_value=True)
    provider.complete = AsyncMock(
        return_value=CompletionResponse(
            content="Test response",
            model="test-model",
            usage=Usage(input_tokens=10, output_tokens=20),
        )
    )
    return provider


@pytest.fixture
def config():
    """Create a test configuration."""
    return RateGuardConfig(
        providers=[
            ProviderConfig(
                type=ProviderType.BEDROCK,
                model="anthropic.claude-v2",
                region="us-east-1",
            )
        ],
        cache=CacheConfig(enabled=False),
    )


class TestSyncWrappers:
    """Tests for synchronous wrapper methods."""

    def test_complete_sync_basic(self, config, mock_provider):
        """Test basic complete_sync functionality."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            response = client.complete_sync(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )

            assert response.content == "Test response"
            assert response.model == "test-model"

    def test_complete_sync_with_parameters(self, config, mock_provider):
        """Test complete_sync with various parameters."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            response = client.complete_sync(
                messages=[
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                max_tokens=256,
                temperature=0.5,
            )

            assert response.content == "Test response"
            # Verify parameters were passed
            call_args = mock_provider.complete.call_args
            assert call_args.kwargs.get("max_tokens") == 256
            assert call_args.kwargs.get("temperature") == 0.5

    def test_embed_sync_basic(self, config, mock_provider):
        """Test basic embed_sync functionality."""
        mock_provider.embed = AsyncMock(
            return_value=EmbeddingResponse(
                embedding=[0.1, 0.2, 0.3],
                model="test-embed-model",
                usage=Usage(input_tokens=5, output_tokens=0),
            )
        )

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            response = client.embed_sync(text="Hello world")

            assert response.embedding == [0.1, 0.2, 0.3]

    def test_health_check_sync(self, config, mock_provider):
        """Test health_check_sync functionality."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            health = client.health_check_sync()

            assert "healthy" in health
            assert "status" in health
            assert "providers" in health


class TestBatchProcessing:
    """Tests for batch processing methods."""

    @pytest.mark.asyncio
    async def test_complete_batch_basic(self, config, mock_provider):
        """Test basic batch processing."""
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return CompletionResponse(
                content=f"Response {call_count}",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=20),
            )

        mock_provider.complete = mock_complete

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            prompts = [
                [{"role": "user", "content": "What is 2+2?"}],
                [{"role": "user", "content": "What is 3+3?"}],
                [{"role": "user", "content": "What is 4+4?"}],
            ]

            responses = await client.complete_batch(prompts)

            assert len(responses) == 3
            # All should be CompletionResponse
            for resp in responses:
                assert isinstance(resp, CompletionResponse)

    @pytest.mark.asyncio
    async def test_complete_batch_empty(self, config, mock_provider):
        """Test batch with empty input."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            responses = await client.complete_batch([])

            assert responses == []

    @pytest.mark.asyncio
    async def test_complete_batch_concurrency_limit(self, config, mock_provider):
        """Test that concurrency is properly limited."""
        concurrent_count = 0
        max_concurrent = 0

        async def mock_complete(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)  # Simulate latency
            concurrent_count -= 1
            return CompletionResponse(
                content="Response",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=20),
            )

        mock_provider.complete = mock_complete

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            # Create 20 prompts but limit concurrency to 5
            prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(20)]

            responses = await client.complete_batch(prompts, max_concurrency=5)

            assert len(responses) == 20
            # Max concurrent should not exceed 5
            assert max_concurrent <= 5

    @pytest.mark.asyncio
    async def test_complete_batch_preserves_order(self, config, mock_provider):
        """Test that batch results maintain input order."""
        import random

        async def mock_complete(*args, messages, **kwargs):
            # Random delay to simulate varying response times
            await asyncio.sleep(random.uniform(0.01, 0.1))
            content = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
            return CompletionResponse(
                content=f"Answer to: {content}",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=20),
            )

        mock_provider.complete = mock_complete

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            prompts = [[{"role": "user", "content": f"Question {i}"}] for i in range(10)]

            responses = await client.complete_batch(prompts, max_concurrency=10)

            # Verify order is preserved
            for i, resp in enumerate(responses):
                assert f"Question {i}" in resp.content

    @pytest.mark.asyncio
    async def test_complete_batch_with_exceptions_returned(self, config, mock_provider):
        """Test batch with return_exceptions=True."""
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError("Simulated error")
            return CompletionResponse(
                content="Success",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=20),
            )

        mock_provider.complete = mock_complete

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(4)]

            responses = await client.complete_batch(prompts, return_exceptions=True)

            assert len(responses) == 4
            # Check mix of responses and exceptions
            success_count = sum(1 for r in responses if isinstance(r, CompletionResponse))
            error_count = sum(1 for r in responses if isinstance(r, Exception))
            assert success_count >= 1
            assert error_count >= 1

    @pytest.mark.asyncio
    async def test_complete_batch_raises_without_return_exceptions(self, config, mock_provider):
        """Test batch raises exception when return_exceptions=False."""

        async def mock_complete(*args, **kwargs):
            raise ValueError("Simulated error")

        mock_provider.complete = mock_complete

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            prompts = [[{"role": "user", "content": "Q1"}]]

            # Exception gets wrapped in AllProvidersExhausted
            from llm_rate_guard import AllProvidersExhausted
            with pytest.raises(AllProvidersExhausted, match="Simulated error"):
                await client.complete_batch(prompts, return_exceptions=False)

    def test_complete_batch_sync(self, config, mock_provider):
        """Test synchronous batch processing."""
        mock_provider.complete = AsyncMock(
            return_value=CompletionResponse(
                content="Batch response",
                model="test-model",
                usage=Usage(input_tokens=10, output_tokens=20),
            )
        )

        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            prompts = [
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
            ]

            responses = client.complete_batch_sync(prompts)

            assert len(responses) == 2
            assert all(isinstance(r, CompletionResponse) for r in responses)


class TestRunSyncHelper:
    """Tests for the _run_sync helper method."""

    def test_run_sync_no_loop(self, config, mock_provider):
        """Test _run_sync when no event loop is running."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            # Simple coroutine
            async def simple_coro():
                return 42

            result = client._run_sync(simple_coro())
            assert result == 42

    def test_run_sync_with_exception(self, config, mock_provider):
        """Test _run_sync propagates exceptions."""
        with patch("llm_rate_guard.router.get_provider_class") as mock_get_class:
            mock_class = MagicMock(return_value=mock_provider)
            mock_get_class.return_value = mock_class

            client = RateGuardClient(config=config)

            async def failing_coro():
                raise RuntimeError("Test error")

            with pytest.raises(RuntimeError, match="Test error"):
                client._run_sync(failing_coro())
