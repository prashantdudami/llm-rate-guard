"""Token bucket rate limiter for LLM Rate Guard."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    """Token bucket rate limiter implementation.

    Uses the token bucket algorithm to control request rates.
    Tokens are added at a fixed rate and consumed per request.
    """

    capacity: float
    """Maximum tokens (requests or LLM tokens) the bucket can hold."""

    refill_rate: float
    """Tokens added per second."""

    tokens: float = field(init=False)
    """Current token count."""

    last_refill: float = field(init=False)
    """Timestamp of last refill."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    """Lock for thread-safe operations."""

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds (0.0 if no wait was needed).
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time needed
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate

            # Wait for tokens to become available
            await asyncio.sleep(wait_time)

            # Refill and consume
            self._refill()
            self.tokens -= tokens
            return wait_time

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (without acquiring lock)."""
        self._refill()
        return self.tokens

    @property
    def utilization(self) -> float:
        """Get current utilization as a percentage (0-100)."""
        return (1 - (self.tokens / self.capacity)) * 100


class RateLimiter:
    """Combined rate limiter for RPM and TPM.

    Manages both requests per minute and tokens per minute limits
    using separate token buckets.
    """

    def __init__(
        self,
        rpm_limit: int,
        tpm_limit: int,
        burst_multiplier: float = 1.0,
    ):
        """Initialize the rate limiter.

        Args:
            rpm_limit: Maximum requests per minute.
            tpm_limit: Maximum tokens per minute.
            burst_multiplier: Allow bursting up to this multiple of the limit.
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        # RPM bucket: capacity = rpm_limit * burst_multiplier, refill = rpm_limit / 60
        self.rpm_bucket = TokenBucket(
            capacity=rpm_limit * burst_multiplier,
            refill_rate=rpm_limit / 60.0,
        )

        # TPM bucket: capacity = tpm_limit * burst_multiplier, refill = tpm_limit / 60
        self.tpm_bucket = TokenBucket(
            capacity=tpm_limit * burst_multiplier,
            refill_rate=tpm_limit / 60.0,
        )

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_wait_time = 0.0
        self._rate_limit_hits = 0

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """Acquire capacity for a request with estimated tokens.

        Args:
            estimated_tokens: Estimated token usage for the request.

        Returns:
            Total wait time in seconds.
        """
        # Acquire from both buckets
        rpm_wait = await self.rpm_bucket.acquire(1)
        tpm_wait = await self.tpm_bucket.acquire(estimated_tokens)

        total_wait = rpm_wait + tpm_wait

        # Update metrics
        self._total_requests += 1
        self._total_tokens += estimated_tokens
        self._total_wait_time += total_wait
        if total_wait > 0:
            self._rate_limit_hits += 1

        return total_wait

    async def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire capacity without waiting.

        Args:
            estimated_tokens: Estimated token usage for the request.

        Returns:
            True if capacity was acquired, False otherwise.
        """
        # Check RPM first
        if not await self.rpm_bucket.try_acquire(1):
            return False

        # Check TPM
        if not await self.tpm_bucket.try_acquire(estimated_tokens):
            # Rollback RPM acquisition
            self.rpm_bucket.tokens += 1
            return False

        self._total_requests += 1
        self._total_tokens += estimated_tokens
        return True

    def record_actual_tokens(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Adjust token bucket based on actual vs estimated usage.

        Call this after getting the actual token count from the response.

        Args:
            actual_tokens: Actual tokens used.
            estimated_tokens: Previously estimated tokens.
        """
        difference = estimated_tokens - actual_tokens
        if difference > 0:
            # We overestimated, add back the difference
            self.tpm_bucket.tokens = min(
                self.tpm_bucket.capacity,
                self.tpm_bucket.tokens + difference,
            )
        # If we underestimated, we accept the overage (already consumed)

    @property
    def rpm_utilization(self) -> float:
        """Current RPM utilization percentage."""
        return self.rpm_bucket.utilization

    @property
    def tpm_utilization(self) -> float:
        """Current TPM utilization percentage."""
        return self.tpm_bucket.utilization

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "rpm_available": self.rpm_bucket.available_tokens,
            "tpm_available": self.tpm_bucket.available_tokens,
            "rpm_utilization_pct": self.rpm_utilization,
            "tpm_utilization_pct": self.tpm_utilization,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_wait_time_seconds": self._total_wait_time,
            "rate_limit_hits": self._rate_limit_hits,
        }


class MultiProviderRateLimiter:
    """Manages rate limiters for multiple providers/regions.

    Each provider/region combination gets its own RateLimiter instance.
    """

    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        provider_id: str,
        rpm_limit: int,
        tpm_limit: int,
        burst_multiplier: float = 1.0,
    ) -> RateLimiter:
        """Get or create a rate limiter for a provider.

        Args:
            provider_id: Unique identifier for the provider (e.g., 'bedrock:us-east-1').
            rpm_limit: Requests per minute limit.
            tpm_limit: Tokens per minute limit.
            burst_multiplier: Burst multiplier for the limiter.

        Returns:
            RateLimiter for the provider.
        """
        if provider_id not in self._limiters:
            self._limiters[provider_id] = RateLimiter(
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                burst_multiplier=burst_multiplier,
            )
        return self._limiters[provider_id]

    def get(self, provider_id: str) -> Optional[RateLimiter]:
        """Get a rate limiter by provider ID.

        Args:
            provider_id: Unique identifier for the provider.

        Returns:
            RateLimiter if found, None otherwise.
        """
        return self._limiters.get(provider_id)

    async def acquire(
        self,
        provider_id: str,
        estimated_tokens: int = 1000,
    ) -> float:
        """Acquire capacity from a provider's rate limiter.

        Args:
            provider_id: Provider identifier.
            estimated_tokens: Estimated token usage.

        Returns:
            Wait time in seconds.

        Raises:
            KeyError: If provider not found.
        """
        limiter = self._limiters.get(provider_id)
        if limiter is None:
            raise KeyError(f"No rate limiter found for provider: {provider_id}")
        return await limiter.acquire(estimated_tokens)

    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all providers."""
        return {pid: limiter.get_stats() for pid, limiter in self._limiters.items()}
