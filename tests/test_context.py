"""Tests for request context, middleware, and quota manager."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from llm_rate_guard.context import (
    MiddlewareChain,
    QuotaManager,
    RequestContext,
    get_current_context,
    set_current_context,
)
from llm_rate_guard.client import RateGuardClient
from llm_rate_guard.config import ProviderConfig, ProviderType
from llm_rate_guard.exceptions import ConfigurationError
from llm_rate_guard.providers.base import CompletionResponse, Usage


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


class TestRequestContext:
    """Tests for RequestContext class."""

    def test_context_creation(self):
        """Context is created with auto-generated request ID."""
        ctx = RequestContext()
        assert ctx.request_id is not None
        assert len(ctx.request_id) == 8

    def test_context_with_tenant(self):
        """Context with tenant and user IDs."""
        ctx = RequestContext(
            tenant_id="tenant-123",
            user_id="user-456",
        )
        assert ctx.tenant_id == "tenant-123"
        assert ctx.user_id == "user-456"

    def test_context_with_labels(self):
        """Context with custom labels."""
        ctx = RequestContext(
            labels={"project": "chatbot", "env": "prod"},
        )
        assert ctx.labels["project"] == "chatbot"
        assert ctx.labels["env"] == "prod"

    def test_context_to_dict(self):
        """Context can be converted to dictionary."""
        ctx = RequestContext(
            request_id="test-123",
            tenant_id="tenant-1",
            labels={"key": "value"},
        )
        d = ctx.to_dict()
        assert d["request_id"] == "test-123"
        assert d["tenant_id"] == "tenant-1"
        assert d["labels"] == {"key": "value"}

    def test_context_repr(self):
        """Context has readable repr."""
        ctx = RequestContext(
            request_id="test-123",
            tenant_id="tenant-1",
        )
        repr_str = repr(ctx)
        assert "test-123" in repr_str
        assert "tenant-1" in repr_str


class TestContextVariable:
    """Tests for context variable storage."""

    def test_context_variable_default(self):
        """Context variable is None by default."""
        assert get_current_context() is None

    def test_set_and_get_context(self):
        """Can set and get context variable."""
        ctx = RequestContext(tenant_id="test-tenant")
        token = set_current_context(ctx)

        assert get_current_context() is not None
        assert get_current_context().tenant_id == "test-tenant"

        # Reset
        set_current_context(None)
        assert get_current_context() is None


class TestMiddlewareChain:
    """Tests for MiddlewareChain class."""

    @pytest.mark.asyncio
    async def test_pre_middleware_passthrough(self):
        """Pre-middleware passes through data."""
        chain = MiddlewareChain()

        def passthrough(data, ctx):
            return data

        chain.add_pre(passthrough)

        result = await chain.process_pre({"key": "value"}, None)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_pre_middleware_modification(self):
        """Pre-middleware can modify data."""
        chain = MiddlewareChain()

        def modify(data, ctx):
            data["modified"] = True
            return data

        chain.add_pre(modify)

        result = await chain.process_pre({"key": "value"}, None)
        assert result["modified"] is True

    @pytest.mark.asyncio
    async def test_pre_middleware_blocking(self):
        """Pre-middleware can block requests."""
        chain = MiddlewareChain()

        def block(data, ctx):
            return None  # Block request

        chain.add_pre(block)

        result = await chain.process_pre({"key": "value"}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_middleware(self):
        """Async middleware is supported."""
        chain = MiddlewareChain()

        async def async_middleware(data, ctx):
            await asyncio.sleep(0.01)
            data["async_processed"] = True
            return data

        chain.add_pre(async_middleware)

        result = await chain.process_pre({"key": "value"}, None)
        assert result["async_processed"] is True

    @pytest.mark.asyncio
    async def test_middleware_chain_order(self):
        """Middleware executes in order added."""
        chain = MiddlewareChain()
        order = []

        def first(data, ctx):
            order.append("first")
            return data

        def second(data, ctx):
            order.append("second")
            return data

        chain.add_pre(first)
        chain.add_pre(second)

        await chain.process_pre({}, None)
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_post_middleware(self):
        """Post-middleware receives response data."""
        chain = MiddlewareChain()
        received = {}

        def capture(data, ctx):
            received.update(data)

        chain.add_post(capture)

        await chain.process_post({"content": "Hello"}, None)
        assert received["content"] == "Hello"

    def test_remove_middleware(self):
        """Can remove middleware."""
        chain = MiddlewareChain()

        def middleware(data, ctx):
            return data

        chain.add_pre(middleware)
        assert chain.pre_count == 1

        result = chain.remove_pre(middleware)
        assert result is True
        assert chain.pre_count == 0

    def test_clear_middleware(self):
        """Can clear all middleware."""
        chain = MiddlewareChain()

        chain.add_pre(lambda d, c: d)
        chain.add_post(lambda d, c: None)

        assert chain.pre_count == 1
        assert chain.post_count == 1

        chain.clear()

        assert chain.pre_count == 0
        assert chain.post_count == 0


class TestQuotaManager:
    """Tests for QuotaManager class."""

    def test_no_limit_allows_all(self):
        """Without limits set, all requests are allowed."""
        quota = QuotaManager()
        assert quota.check("tenant-1", tokens=1000000) is True

    def test_token_limit_enforcement(self):
        """Token limits are enforced."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", tokens_per_day=1000)

        # Within limit
        assert quota.check("tenant-1", tokens=500) is True

        # Record usage
        quota.record("tenant-1", tokens=500)

        # Still within limit
        assert quota.check("tenant-1", tokens=400) is True

        # Would exceed limit
        assert quota.check("tenant-1", tokens=600) is False

    def test_request_limit_enforcement(self):
        """Request limits are enforced."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", requests_per_day=10)

        # Make requests up to limit
        for _ in range(10):
            assert quota.check("tenant-1", requests=1) is True
            quota.record("tenant-1", requests=1)

        # Next request exceeds limit
        assert quota.check("tenant-1", requests=1) is False

    def test_cost_limit_enforcement(self):
        """Cost limits are enforced."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", cost_per_day_usd=10.0)

        # Within limit
        assert quota.check("tenant-1", cost_usd=5.0) is True
        quota.record("tenant-1", cost_usd=5.0)

        # Would exceed
        assert quota.check("tenant-1", cost_usd=6.0) is False

    def test_get_usage(self):
        """Can retrieve usage statistics."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", tokens_per_day=1000, requests_per_day=100)
        quota.record("tenant-1", tokens=250, requests=5)

        usage = quota.get_usage("tenant-1")
        assert usage["tokens_used"] == 250
        assert usage["tokens_limit"] == 1000
        assert usage["requests_used"] == 5
        assert usage["requests_limit"] == 100

    def test_reset_single_tenant(self):
        """Can reset usage for a single tenant."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", tokens_per_day=1000)
        quota.set_limit("tenant-2", tokens_per_day=1000)

        quota.record("tenant-1", tokens=500)
        quota.record("tenant-2", tokens=500)

        quota.reset("tenant-1")

        assert quota.get_usage("tenant-1")["tokens_used"] == 0
        assert quota.get_usage("tenant-2")["tokens_used"] == 500

    def test_reset_all_tenants(self):
        """Can reset usage for all tenants."""
        quota = QuotaManager()
        quota.set_limit("tenant-1", tokens_per_day=1000)
        quota.set_limit("tenant-2", tokens_per_day=1000)

        quota.record("tenant-1", tokens=500)
        quota.record("tenant-2", tokens=500)

        quota.reset()

        assert quota.get_usage("tenant-1")["tokens_used"] == 0
        assert quota.get_usage("tenant-2")["tokens_used"] == 0


class TestClientMiddlewareIntegration:
    """Tests for middleware integration with RateGuardClient."""

    @pytest.mark.asyncio
    async def test_pre_middleware_receives_request(self):
        """Pre-middleware receives request data."""
        mock_class = MagicMock(side_effect=create_mock_provider)
        received_data = {}

        def capture_request(data, ctx):
            received_data.update(data)
            return data

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )
            client.add_pre_middleware(capture_request)

            await client.complete([{"role": "user", "content": "Hello"}])

            assert "messages" in received_data
            assert received_data["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_post_middleware_receives_response(self):
        """Post-middleware receives response data."""
        mock_class = MagicMock(side_effect=create_mock_provider)
        received_data = {}

        def capture_response(data, ctx):
            received_data.update(data)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )
            client.add_post_middleware(capture_response)

            await client.complete([{"role": "user", "content": "Hello"}])

            assert received_data["content"] == "Hello!"
            assert "usage" in received_data

    @pytest.mark.asyncio
    async def test_middleware_can_block_request(self):
        """Middleware can block requests."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        def block_all(data, ctx):
            return None  # Block

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )
            client.add_pre_middleware(block_all)

            with pytest.raises(ConfigurationError) as exc_info:
                await client.complete([{"role": "user", "content": "Hello"}])

            assert "blocked by middleware" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_request_context_passed_to_middleware(self):
        """Request context is passed to middleware."""
        mock_class = MagicMock(side_effect=create_mock_provider)
        received_ctx = []

        def capture_context(data, ctx):
            received_ctx.append(ctx)
            return data

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
                cache_enabled=False,
            )
            client.add_pre_middleware(capture_context)

            ctx = RequestContext(tenant_id="my-tenant")
            await client.complete(
                [{"role": "user", "content": "Hello"}],
                context=ctx,
            )

            assert len(received_ctx) == 1
            assert received_ctx[0].tenant_id == "my-tenant"


class TestClientQuotaIntegration:
    """Tests for quota manager integration with RateGuardClient."""

    def test_set_quota_manager(self):
        """Can set quota manager on client."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )

            quota = QuotaManager()
            client.set_quota_manager(quota)

            assert client.quota_manager is quota

    @pytest.mark.asyncio
    async def test_quota_enforcement_via_middleware(self):
        """Can enforce quotas via middleware."""
        mock_class = MagicMock(side_effect=create_mock_provider)

        quota = QuotaManager()
        quota.set_limit("blocked-tenant", requests_per_day=0)  # Block all

        def enforce_quota(data, ctx):
            if ctx and not quota.check(ctx.tenant_id or "", requests=1):
                return None  # Block
            return data

        with patch("llm_rate_guard.router.get_provider_class", return_value=mock_class):
            client = RateGuardClient(
                providers=[ProviderConfig(type=ProviderType.OPENAI, model="gpt-4")],
            )
            client.add_pre_middleware(enforce_quota)

            ctx = RequestContext(tenant_id="blocked-tenant")

            with pytest.raises(ConfigurationError) as exc_info:
                await client.complete(
                    [{"role": "user", "content": "Hello"}],
                    context=ctx,
                )

            assert "blocked by middleware" in str(exc_info.value).lower()
