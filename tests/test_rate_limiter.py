"""Tests for rate limiter module."""

import asyncio
import pytest
import time

from llm_rate_guard.rate_limiter import TokenBucket, RateLimiter, MultiProviderRateLimiter


class TestTokenBucket:
    """Tests for TokenBucket class."""

    @pytest.mark.asyncio
    async def test_initial_capacity(self):
        """Bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        """Acquiring tokens reduces available tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        wait_time = await bucket.acquire(3)
        
        assert wait_time == 0.0  # No wait needed
        assert bucket.tokens == 7

    @pytest.mark.asyncio
    async def test_acquire_waits_when_insufficient(self):
        """Acquiring more tokens than available causes wait."""
        bucket = TokenBucket(capacity=2, refill_rate=10.0)  # 10 tokens/sec
        
        # Drain the bucket
        await bucket.acquire(2)
        
        # Now acquire more - should wait
        start = time.monotonic()
        wait_time = await bucket.acquire(1)
        elapsed = time.monotonic() - start
        
        assert wait_time > 0
        assert elapsed >= 0.05  # At least 0.1 seconds for 1 token at 10/sec

    @pytest.mark.asyncio
    async def test_try_acquire_no_wait(self):
        """try_acquire returns immediately without waiting."""
        bucket = TokenBucket(capacity=1, refill_rate=0.1)
        
        # First acquire succeeds
        assert await bucket.try_acquire(1) is True
        
        # Second acquire fails (no tokens, won't wait)
        assert await bucket.try_acquire(1) is False

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # 100 tokens/sec
        
        # Drain tokens
        await bucket.acquire(10)
        assert bucket.tokens == 0
        
        # Wait a bit for refill
        await asyncio.sleep(0.05)  # 50ms = 5 tokens at 100/sec
        
        # Check refill happened
        bucket._refill()
        assert bucket.tokens >= 4  # At least 4 tokens refilled

    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)
        
        # Wait for potential over-refill
        await asyncio.sleep(0.5)
        
        bucket._refill()
        assert bucket.tokens <= 10


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_both_buckets(self):
        """Acquire checks both RPM and TPM."""
        limiter = RateLimiter(rpm_limit=60, tpm_limit=1000)
        
        wait_time = await limiter.acquire(estimated_tokens=100)
        
        assert wait_time == 0.0
        assert limiter._total_requests == 1
        assert limiter._total_tokens == 100

    @pytest.mark.asyncio
    async def test_rpm_limit_enforced(self):
        """RPM limit is enforced."""
        limiter = RateLimiter(rpm_limit=2, tpm_limit=10000)
        
        # Make 2 requests (exhausts RPM)
        await limiter.acquire(10)
        await limiter.acquire(10)
        
        # Third request should wait
        start = time.monotonic()
        await limiter.acquire(10)
        elapsed = time.monotonic() - start
        
        # Should have waited for RPM refill
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_try_acquire_respects_limits(self):
        """try_acquire fails when limits exceeded."""
        limiter = RateLimiter(rpm_limit=1, tpm_limit=10000)
        
        # First request succeeds
        assert await limiter.try_acquire(10) is True
        
        # Second request fails (RPM exhausted)
        assert await limiter.try_acquire(10) is False

    @pytest.mark.asyncio
    async def test_record_actual_tokens(self):
        """Actual tokens adjust the bucket."""
        limiter = RateLimiter(rpm_limit=60, tpm_limit=1000)
        
        # Acquire with estimate
        await limiter.acquire(estimated_tokens=500)
        tpm_before = limiter.tpm_bucket.tokens
        
        # Record actual (lower than estimate)
        limiter.record_actual_tokens(actual_tokens=200, estimated_tokens=500)
        
        # Tokens should be added back
        assert limiter.tpm_bucket.tokens > tpm_before

    def test_get_stats(self):
        """Stats are properly returned."""
        limiter = RateLimiter(rpm_limit=60, tpm_limit=1000)
        
        stats = limiter.get_stats()
        
        assert stats["rpm_limit"] == 60
        assert stats["tpm_limit"] == 1000
        assert "rpm_utilization_pct" in stats
        assert "tpm_utilization_pct" in stats


class TestMultiProviderRateLimiter:
    """Tests for MultiProviderRateLimiter class."""

    @pytest.mark.asyncio
    async def test_get_or_create(self):
        """get_or_create creates limiter on first call."""
        multi = MultiProviderRateLimiter()
        
        limiter1 = multi.get_or_create("provider1", rpm_limit=60, tpm_limit=1000)
        limiter2 = multi.get_or_create("provider1", rpm_limit=120, tpm_limit=2000)
        
        # Should return same limiter (not recreate)
        assert limiter1 is limiter2
        assert limiter1.rpm_limit == 60  # Original values kept

    @pytest.mark.asyncio
    async def test_separate_limiters_per_provider(self):
        """Each provider gets its own limiter."""
        multi = MultiProviderRateLimiter()
        
        limiter1 = multi.get_or_create("provider1", rpm_limit=60, tpm_limit=1000)
        limiter2 = multi.get_or_create("provider2", rpm_limit=120, tpm_limit=2000)
        
        assert limiter1 is not limiter2
        assert limiter1.rpm_limit == 60
        assert limiter2.rpm_limit == 120

    @pytest.mark.asyncio
    async def test_acquire_routes_to_provider(self):
        """acquire uses correct provider's limiter."""
        multi = MultiProviderRateLimiter()
        multi.get_or_create("provider1", rpm_limit=60, tpm_limit=1000)
        
        wait_time = await multi.acquire("provider1", estimated_tokens=100)
        
        assert wait_time == 0.0
        
        limiter = multi.get("provider1")
        assert limiter is not None
        assert limiter._total_requests == 1

    @pytest.mark.asyncio
    async def test_acquire_unknown_provider_raises(self):
        """acquire raises for unknown provider."""
        multi = MultiProviderRateLimiter()
        
        with pytest.raises(KeyError):
            await multi.acquire("unknown", estimated_tokens=100)

    def test_get_all_stats(self):
        """get_all_stats returns stats for all providers."""
        multi = MultiProviderRateLimiter()
        multi.get_or_create("p1", rpm_limit=60, tpm_limit=1000)
        multi.get_or_create("p2", rpm_limit=120, tpm_limit=2000)
        
        stats = multi.get_all_stats()
        
        assert "p1" in stats
        assert "p2" in stats
        assert stats["p1"]["rpm_limit"] == 60
        assert stats["p2"]["rpm_limit"] == 120
