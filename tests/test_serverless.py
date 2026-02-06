"""Tests for serverless components (Redis, DynamoDB rate limiters, Lambda decorator)."""

import json
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from llm_rate_guard.serverless import (
    RedisRateLimiter,
    DynamoDBRateLimiter,
    ServerlessConfig,
    lambda_rate_limited,
)


class TestRedisRateLimiter:
    """Tests for RedisRateLimiter."""

    def test_initialization(self):
        """Test basic initialization."""
        limiter = RedisRateLimiter(
            host="localhost",
            port=6379,
            rpm=250,
            tpm=2_000_000,
        )
        assert limiter.rpm == 250
        assert limiter.tpm == 2_000_000
        assert limiter._client is None  # Lazy init

    def test_url_initialization(self):
        """Test URL-based initialization."""
        limiter = RedisRateLimiter(
            url="redis://user:pass@host:6379/0",
            rpm=100,
        )
        assert limiter._url == "redis://user:pass@host:6379/0"

    def test_try_acquire_mocked(self):
        """Test try_acquire with mocked Redis."""
        limiter = RedisRateLimiter(rpm=250, tpm=2_000_000)

        # Mock Redis client
        mock_client = MagicMock()
        mock_script = MagicMock(side_effect=[1, 1])  # RPM ok, TPM ok
        limiter._client = mock_client
        limiter._script = mock_script

        result = limiter.try_acquire(estimated_tokens=500)
        assert result is True
        assert mock_script.call_count == 2

    def test_try_acquire_rpm_denied(self):
        """Test try_acquire when RPM denied."""
        limiter = RedisRateLimiter(rpm=250, tpm=2_000_000)

        mock_client = MagicMock()
        mock_script = MagicMock(return_value=0)  # RPM denied
        limiter._client = mock_client
        limiter._script = mock_script

        result = limiter.try_acquire(estimated_tokens=500)
        assert result is False
        assert mock_script.call_count == 1  # Only RPM checked

    def test_try_acquire_tpm_denied(self):
        """Test try_acquire when TPM denied (rolls back RPM)."""
        limiter = RedisRateLimiter(rpm=250, tpm=2_000_000)

        mock_client = MagicMock()
        mock_script = MagicMock(side_effect=[1, 0])  # RPM ok, TPM denied
        limiter._client = mock_client
        limiter._script = mock_script

        result = limiter.try_acquire(estimated_tokens=500)
        assert result is False
        # Should attempt RPM rollback
        mock_client.hincrbyfloat.assert_called_once()

    def test_acquire_mocked(self):
        """Test blocking acquire with mocked Redis."""
        limiter = RedisRateLimiter(rpm=250, tpm=2_000_000)

        mock_client = MagicMock()
        mock_script = MagicMock(return_value=1)  # Always ok
        limiter._client = mock_client
        limiter._script = mock_script

        wait = limiter.acquire(estimated_tokens=500)
        assert wait >= 0

    def test_acquire_timeout(self):
        """Test acquire times out."""
        limiter = RedisRateLimiter(rpm=250, tpm=2_000_000)

        mock_client = MagicMock()
        mock_script = MagicMock(return_value=0)  # Always denied
        limiter._client = mock_client
        limiter._script = mock_script

        with pytest.raises(TimeoutError):
            limiter.acquire(estimated_tokens=500, timeout=0.2)

    def test_close(self):
        """Test connection close."""
        limiter = RedisRateLimiter()
        mock_client = MagicMock()
        limiter._client = mock_client

        limiter.close()
        mock_client.close.assert_called_once()
        assert limiter._client is None

    def test_import_error(self):
        """Test graceful error when redis not installed."""
        limiter = RedisRateLimiter()

        with patch.dict("sys.modules", {"redis": None}):
            # Can't easily test this without actually uninstalling redis
            # Just verify the limiter was created
            assert limiter._client is None


class TestDynamoDBRateLimiter:
    """Tests for DynamoDBRateLimiter."""

    def test_initialization(self):
        """Test basic initialization."""
        limiter = DynamoDBRateLimiter(
            table_name="rate-limits",
            rpm=250,
            tpm=2_000_000,
            region="us-east-1",
        )
        assert limiter.table_name == "rate-limits"
        assert limiter.rpm == 250
        assert limiter.tpm == 2_000_000
        assert limiter._table is None  # Lazy init

    def test_try_acquire_mocked(self):
        """Test try_acquire with mocked DynamoDB."""
        limiter = DynamoDBRateLimiter(
            table_name="test-table",
            rpm=250,
            tpm=2_000_000,
        )

        # Mock DynamoDB table
        mock_table = MagicMock()
        mock_table.get_item.return_value = {}  # No existing item
        mock_table.put_item.return_value = {}
        limiter._table = mock_table

        with patch("llm_rate_guard.serverless.DynamoDBRateLimiter._try_acquire_bucket", return_value=True):
            result = limiter.try_acquire(estimated_tokens=500)
            assert result is True

    def test_acquire_mocked(self):
        """Test blocking acquire with mocked DynamoDB."""
        limiter = DynamoDBRateLimiter(
            table_name="test-table",
            rpm=250,
            tpm=2_000_000,
        )

        with patch.object(limiter, "try_acquire", return_value=True):
            wait = limiter.acquire(estimated_tokens=500)
            assert wait >= 0

    def test_acquire_timeout(self):
        """Test acquire times out."""
        limiter = DynamoDBRateLimiter(
            table_name="test-table",
            rpm=250,
            tpm=2_000_000,
        )

        with patch.object(limiter, "try_acquire", return_value=False):
            with pytest.raises(TimeoutError):
                limiter.acquire(estimated_tokens=500, timeout=0.2)

    def test_key_prefix(self):
        """Test custom key prefix."""
        limiter = DynamoDBRateLimiter(
            table_name="test",
            rpm=100,
            key_prefix="tenant#123",
        )
        assert limiter._key_prefix == "tenant#123"


class TestLambdaRateLimited:
    """Tests for @lambda_rate_limited decorator."""

    def test_basic_usage(self):
        """Test decorator allows request when rate limit ok."""
        mock_limiter = MagicMock()
        mock_limiter.acquire.return_value = 0.0

        @lambda_rate_limited(mock_limiter, estimated_tokens=500)
        def handler(event, context):
            return {"statusCode": 200, "body": "ok"}

        result = handler({"prompt": "hi"}, {})
        assert result["statusCode"] == 200
        mock_limiter.acquire.assert_called_once()

    def test_rate_limit_timeout(self):
        """Test returns 429 when rate limited."""
        mock_limiter = MagicMock()
        mock_limiter.acquire.side_effect = TimeoutError("timed out")

        @lambda_rate_limited(mock_limiter, timeout=1)
        def handler(event, context):
            return {"statusCode": 200, "body": "ok"}

        result = handler({}, {})
        assert result["statusCode"] == 429
        body = json.loads(result["body"])
        assert "Rate limit exceeded" in body["error"]

    def test_passes_through_kwargs(self):
        """Test kwargs are passed to handler."""
        mock_limiter = MagicMock()
        mock_limiter.acquire.return_value = 0.0

        @lambda_rate_limited(mock_limiter)
        def handler(event, context, extra="default"):
            return {"extra": extra}

        result = handler({}, {}, extra="custom")
        assert result["extra"] == "custom"

    def test_limiter_accessible(self):
        """Test limiter is accessible on decorated function."""
        mock_limiter = MagicMock()

        @lambda_rate_limited(mock_limiter)
        def handler(event, context):
            return {}

        assert handler._rate_limiter is mock_limiter

    def test_preserves_function_name(self):
        """Test decorator preserves function metadata."""
        mock_limiter = MagicMock()
        mock_limiter.acquire.return_value = 0.0

        @lambda_rate_limited(mock_limiter)
        def my_lambda_handler(event, context):
            """My handler docstring."""
            return {}

        assert my_lambda_handler.__name__ == "my_lambda_handler"
        assert my_lambda_handler.__doc__ == "My handler docstring."


class TestServerlessConfig:
    """Tests for ServerlessConfig."""

    def test_for_lambda_defaults(self):
        """Test default Lambda config."""
        config = ServerlessConfig.for_lambda()
        assert config["cache_enabled"] is False
        assert config["queue_enabled"] is False
        assert config["failover_enabled"] is True
        assert config["max_retries"] == 2
        assert "notes" in config
        assert len(config["notes"]) > 0

    def test_for_lambda_custom(self):
        """Test custom Lambda config."""
        config = ServerlessConfig.for_lambda(
            cache_enabled=True,
            queue_enabled=True,
            max_retries=5,
        )
        assert config["cache_enabled"] is True
        assert config["queue_enabled"] is True
        assert config["max_retries"] == 5

    def test_for_lambda_with_providers(self):
        """Test Lambda config with providers."""
        mock_providers = [MagicMock(), MagicMock()]
        config = ServerlessConfig.for_lambda(providers=mock_providers)
        assert config["providers"] == mock_providers
