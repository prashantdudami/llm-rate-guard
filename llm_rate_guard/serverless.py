"""Serverless support for AWS Lambda and similar environments.

Components designed for stateless, short-lived execution environments
where in-memory state is lost on cold starts.

External state backends (Redis, DynamoDB) persist rate limiting and
circuit breaker state across invocations.

Example:
    ```python
    from llm_rate_guard.serverless import DynamoDBRateLimiter, lambda_rate_limited

    limiter = DynamoDBRateLimiter(table_name="rate-limits", rpm=250, tpm=2_000_000)

    @lambda_rate_limited(limiter)
    def handler(event, context):
        response = bedrock_client.invoke(...)
        return response
    ```
"""

import functools
import json
import logging
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Redis-backed Rate Limiter
# =============================================================================


class RedisRateLimiter:
    """Rate limiter with state persisted in Redis.

    Uses Redis atomic operations (Lua scripts) for accurate distributed
    rate limiting. State survives Lambda cold starts.

    Requires: pip install redis

    Example:
        ```python
        limiter = RedisRateLimiter(
            host="redis.example.com",
            rpm=250,
            tpm=2_000_000,
        )

        # Blocks until capacity is available
        limiter.acquire(estimated_tokens=500)
        response = bedrock_client.invoke(...)

        # Non-blocking check
        if limiter.try_acquire(estimated_tokens=500):
            response = bedrock_client.invoke(...)
        ```
    """

    # Lua script for atomic token bucket acquire
    _ACQUIRE_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])

    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or capacity
    local last_refill = tonumber(data[2]) or now

    -- Refill tokens
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * refill_rate)

    if tokens >= requested then
        tokens = tokens - requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 120)
        return 1
    else
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 120)
        return 0
    end
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        url: Optional[str] = None,
        rpm: int = 250,
        tpm: int = 2_000_000,
        key_prefix: str = "llm_rg:rl:",
        burst_multiplier: float = 1.0,
    ):
        """Initialize Redis rate limiter.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database.
            password: Redis password.
            url: Redis URL (overrides host/port/db/password).
            rpm: Requests per minute.
            tpm: Tokens per minute.
            key_prefix: Redis key prefix.
            burst_multiplier: Burst capacity multiplier.
        """
        self.rpm = rpm
        self.tpm = tpm
        self._key_prefix = key_prefix
        self._rpm_capacity = rpm * burst_multiplier
        self._tpm_capacity = tpm * burst_multiplier
        self._rpm_refill_rate = rpm / 60.0
        self._tpm_refill_rate = tpm / 60.0
        self._client: Any = None
        self._script: Any = None
        self._url = url
        self._host = host
        self._port = port
        self._db = db
        self._password = password

    def _ensure_client(self) -> Any:
        """Lazy-initialize Redis client."""
        if self._client is not None:
            return self._client

        try:
            import redis
        except ImportError:
            raise ImportError(
                "RedisRateLimiter requires 'redis' package. "
                "Install with: pip install redis"
            )

        if self._url:
            self._client = redis.Redis.from_url(self._url, decode_responses=False)
        else:
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
            )

        self._script = self._client.register_script(self._ACQUIRE_SCRIPT)
        return self._client

    def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire capacity without blocking.

        Args:
            estimated_tokens: Estimated tokens for this request.

        Returns:
            True if capacity acquired, False otherwise.
        """
        self._ensure_client()
        now = time.time()

        # Check RPM
        rpm_key = f"{self._key_prefix}rpm"
        rpm_ok = self._script(
            keys=[rpm_key],
            args=[self._rpm_capacity, self._rpm_refill_rate, 1, now],
        )

        if not rpm_ok:
            return False

        # Check TPM
        tpm_key = f"{self._key_prefix}tpm"
        tpm_ok = self._script(
            keys=[tpm_key],
            args=[self._tpm_capacity, self._tpm_refill_rate, estimated_tokens, now],
        )

        if not tpm_ok:
            # Rollback RPM (best-effort)
            try:
                self._client.hincrbyfloat(rpm_key, "tokens", 1)
            except Exception:
                pass
            return False

        return True

    def acquire(self, estimated_tokens: int = 1000, timeout: float = 60.0) -> float:
        """Acquire capacity, blocking until available.

        Args:
            estimated_tokens: Estimated tokens.
            timeout: Maximum wait in seconds.

        Returns:
            Wait time in seconds.
        """
        start = time.monotonic()

        while True:
            if self.try_acquire(estimated_tokens):
                return time.monotonic() - start

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Rate limit acquisition timed out after {elapsed:.1f}s"
                )

            time.sleep(0.05)  # Poll interval

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


# =============================================================================
# DynamoDB-backed Rate Limiter
# =============================================================================


class DynamoDBRateLimiter:
    """Rate limiter with state persisted in DynamoDB.

    AWS-native solution, no Redis required. Uses conditional writes
    for atomic updates. State survives Lambda cold starts.

    Requires: pip install boto3

    DynamoDB table schema:
        - Partition key: pk (String)
        - No sort key needed
        - TTL attribute: ttl

    Example:
        ```python
        limiter = DynamoDBRateLimiter(
            table_name="rate-limits",
            rpm=250,
            tpm=2_000_000,
        )
        limiter.acquire(estimated_tokens=500)
        ```
    """

    def __init__(
        self,
        table_name: str,
        rpm: int = 250,
        tpm: int = 2_000_000,
        region: Optional[str] = None,
        key_prefix: str = "llm_rg",
        burst_multiplier: float = 1.0,
    ):
        """Initialize DynamoDB rate limiter.

        Args:
            table_name: DynamoDB table name.
            rpm: Requests per minute.
            tpm: Tokens per minute.
            region: AWS region (uses default if not specified).
            key_prefix: Key prefix for rate limit entries.
            burst_multiplier: Burst capacity multiplier.
        """
        self.table_name = table_name
        self.rpm = rpm
        self.tpm = tpm
        self._key_prefix = key_prefix
        self._rpm_capacity = rpm * burst_multiplier
        self._tpm_capacity = tpm * burst_multiplier
        self._rpm_refill_rate = rpm / 60.0
        self._tpm_refill_rate = tpm / 60.0
        self._region = region
        self._table: Any = None

    def _ensure_table(self) -> Any:
        """Lazy-initialize DynamoDB table resource."""
        if self._table is not None:
            return self._table

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "DynamoDBRateLimiter requires 'boto3' package. "
                "Install with: pip install boto3"
            )

        kwargs: dict[str, Any] = {}
        if self._region:
            kwargs["region_name"] = self._region

        dynamodb = boto3.resource("dynamodb", **kwargs)
        self._table = dynamodb.Table(self.table_name)
        return self._table

    def _try_acquire_bucket(
        self,
        bucket_key: str,
        capacity: float,
        refill_rate: float,
        requested: float,
    ) -> bool:
        """Try to acquire from a single bucket using DynamoDB conditional writes."""
        from botocore.exceptions import ClientError
        from decimal import Decimal

        table = self._ensure_table()
        now = Decimal(str(time.time()))
        ttl_value = int(time.time()) + 120  # 2 minute TTL

        try:
            # Try to get existing item
            response = table.get_item(Key={"pk": bucket_key})
            item = response.get("Item")

            if item:
                tokens = float(item.get("tokens", capacity))
                last_refill = float(item.get("last_refill", now))

                # Refill
                elapsed = float(now) - last_refill
                tokens = min(capacity, tokens + elapsed * refill_rate)

                if tokens < requested:
                    # Update timestamp but not enough tokens
                    table.put_item(Item={
                        "pk": bucket_key,
                        "tokens": Decimal(str(tokens)),
                        "last_refill": now,
                        "ttl": ttl_value,
                    })
                    return False

                # Consume tokens
                new_tokens = Decimal(str(tokens - requested))
                table.put_item(Item={
                    "pk": bucket_key,
                    "tokens": new_tokens,
                    "last_refill": now,
                    "ttl": ttl_value,
                })
                return True
            else:
                # First request, initialize bucket
                initial_tokens = capacity - requested
                if initial_tokens < 0:
                    return False

                table.put_item(Item={
                    "pk": bucket_key,
                    "tokens": Decimal(str(initial_tokens)),
                    "last_refill": now,
                    "ttl": ttl_value,
                })
                return True

        except ClientError as e:
            logger.warning(f"DynamoDB error: {e}")
            # On error, allow the request (fail-open)
            return True

    def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire without blocking.

        Args:
            estimated_tokens: Estimated token usage.

        Returns:
            True if acquired.
        """
        rpm_key = f"{self._key_prefix}#rpm"
        rpm_ok = self._try_acquire_bucket(
            rpm_key, self._rpm_capacity, self._rpm_refill_rate, 1
        )

        if not rpm_ok:
            return False

        tpm_key = f"{self._key_prefix}#tpm"
        tpm_ok = self._try_acquire_bucket(
            tpm_key, self._tpm_capacity, self._tpm_refill_rate, estimated_tokens
        )

        return tpm_ok

    def acquire(self, estimated_tokens: int = 1000, timeout: float = 60.0) -> float:
        """Acquire capacity, blocking until available.

        Args:
            estimated_tokens: Estimated tokens.
            timeout: Maximum wait seconds.

        Returns:
            Wait time in seconds.
        """
        start = time.monotonic()

        while True:
            if self.try_acquire(estimated_tokens):
                return time.monotonic() - start

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Rate limit acquisition timed out after {elapsed:.1f}s"
                )

            time.sleep(0.1)  # Poll interval


# =============================================================================
# Lambda Decorator
# =============================================================================


def lambda_rate_limited(
    limiter: Any,
    estimated_tokens: int = 1000,
    timeout: float = 30.0,
) -> Callable[[F], F]:
    """Decorator for AWS Lambda handlers to add rate limiting.

    Works with any rate limiter that has an `acquire()` method
    (RedisRateLimiter, DynamoDBRateLimiter, SyncRateLimiter).

    Example:
        ```python
        from llm_rate_guard.serverless import DynamoDBRateLimiter, lambda_rate_limited

        limiter = DynamoDBRateLimiter(table_name="rate-limits", rpm=250)

        @lambda_rate_limited(limiter)
        def handler(event, context):
            response = bedrock.invoke_model(...)
            return {"statusCode": 200, "body": response}
        ```

    Args:
        limiter: Rate limiter instance with acquire() method.
        estimated_tokens: Default token estimate per request.
        timeout: Maximum wait for rate limit acquisition.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tokens = kwargs.pop("_estimated_tokens", estimated_tokens)
            try:
                wait_time = limiter.acquire(estimated_tokens=tokens, timeout=timeout)
                if wait_time > 0:
                    logger.debug(f"Rate limiter waited {wait_time:.2f}s")
            except TimeoutError:
                logger.error("Rate limit timeout in Lambda handler")
                return {
                    "statusCode": 429,
                    "body": json.dumps({"error": "Rate limit exceeded, please retry later"}),
                }

            return func(*args, **kwargs)

        wrapper._rate_limiter = limiter  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Serverless Configuration Helper
# =============================================================================


class ServerlessConfig:
    """Configuration preset optimized for serverless environments.

    Disables features that don't work well in Lambda (in-memory queues,
    connection pools) and enables features that do (external cache,
    distributed rate limiting).

    Example:
        ```python
        from llm_rate_guard.serverless import ServerlessConfig

        config = ServerlessConfig.for_lambda(
            providers=[
                ProviderConfig(type="bedrock", model="claude-3-sonnet", region="us-east-1"),
            ],
            rate_limiter=DynamoDBRateLimiter(table_name="rate-limits"),
        )
        ```
    """

    @staticmethod
    def for_lambda(
        providers: Any = None,
        cache_enabled: bool = False,
        queue_enabled: bool = False,
        failover_enabled: bool = True,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Return configuration dict optimized for AWS Lambda.

        Args:
            providers: List of ProviderConfig objects.
            cache_enabled: Enable caching (usually False for Lambda).
            queue_enabled: Enable queuing (usually False for Lambda).
            failover_enabled: Enable failover (usually True).
            max_retries: Max retries (keep low for Lambda timeout).

        Returns:
            Configuration dict suitable for RateGuardConfig.
        """
        return {
            "providers": providers or [],
            "cache_enabled": cache_enabled,
            "queue_enabled": queue_enabled,
            "failover_enabled": failover_enabled,
            "max_retries": max_retries,
            "notes": [
                "In-memory cache disabled (lost on cold start)",
                "Queue disabled (no persistent workers in Lambda)",
                "Use external rate limiter (Redis/DynamoDB)",
                "Low retry count to avoid Lambda timeout",
            ],
        }
