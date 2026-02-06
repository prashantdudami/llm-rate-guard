"""Tests for standalone components (decorators and sync utilities)."""

import asyncio
import time
import pytest

from llm_rate_guard.standalone import (
    SyncRateLimiter,
    SyncCircuitBreaker,
    rate_limited,
    with_retry,
    circuit_protected,
)


class TestSyncRateLimiter:
    """Tests for SyncRateLimiter."""

    def test_basic_acquire(self):
        """Test basic acquire works."""
        limiter = SyncRateLimiter(rpm=600, tpm=100_000)
        wait = limiter.acquire(estimated_tokens=100)
        assert wait >= 0
        assert limiter.total_requests == 1
        assert limiter.total_tokens == 100

    def test_try_acquire_success(self):
        """Test try_acquire when capacity available."""
        limiter = SyncRateLimiter(rpm=600, tpm=100_000)
        assert limiter.try_acquire(estimated_tokens=100) is True
        assert limiter.total_requests == 1

    def test_try_acquire_failure_tpm(self):
        """Test try_acquire when TPM exhausted."""
        limiter = SyncRateLimiter(rpm=600, tpm=100)
        # First call consumes all tokens
        assert limiter.try_acquire(estimated_tokens=100) is True
        # Second call should fail
        assert limiter.try_acquire(estimated_tokens=100) is False
        assert limiter.total_requests == 1  # Only first succeeded

    def test_try_acquire_failure_rpm(self):
        """Test try_acquire when RPM exhausted."""
        limiter = SyncRateLimiter(rpm=1, tpm=100_000)
        assert limiter.try_acquire(estimated_tokens=100) is True
        assert limiter.try_acquire(estimated_tokens=100) is False

    def test_record_actual_tokens(self):
        """Test token adjustment after actual count known."""
        limiter = SyncRateLimiter(rpm=600, tpm=10_000)
        limiter.acquire(estimated_tokens=5000)
        # We overestimated, record actual
        limiter.record_actual_tokens(actual=2000, estimated=5000)
        # Should have more tokens now
        assert limiter.try_acquire(estimated_tokens=3000) is True

    def test_get_stats(self):
        """Test statistics reporting."""
        limiter = SyncRateLimiter(rpm=100, tpm=50_000)
        limiter.acquire(estimated_tokens=1000)
        limiter.acquire(estimated_tokens=2000)
        stats = limiter.get_stats()
        assert stats["rpm"] == 100
        assert stats["tpm"] == 50_000
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 3000

    def test_acquire_timeout(self):
        """Test acquire times out when no capacity."""
        limiter = SyncRateLimiter(rpm=1, tpm=100_000)
        limiter.acquire(estimated_tokens=100)  # Consume all RPM
        with pytest.raises(TimeoutError):
            limiter.acquire(estimated_tokens=100, timeout=0.1)

    def test_burst_multiplier(self):
        """Test burst multiplier increases capacity."""
        limiter = SyncRateLimiter(rpm=1, tpm=100_000, burst_multiplier=3.0)
        # Should allow 3 requests due to burst
        assert limiter.try_acquire(100) is True
        assert limiter.try_acquire(100) is True
        assert limiter.try_acquire(100) is True
        assert limiter.try_acquire(100) is False


class TestSyncCircuitBreaker:
    """Tests for SyncCircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit starts closed."""
        cb = SyncCircuitBreaker()
        assert cb.state == SyncCircuitBreaker.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = SyncCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == SyncCircuitBreaker.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        cb = SyncCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Still closed because success reset the count
        assert cb.state == SyncCircuitBreaker.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """Test circuit transitions to half-open after recovery."""
        cb = SyncCircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == SyncCircuitBreaker.OPEN

        time.sleep(0.15)
        assert cb.state == SyncCircuitBreaker.HALF_OPEN
        assert cb.can_execute() is True

    def test_half_open_to_closed_on_success(self):
        """Test half-open closes after enough successes."""
        cb = SyncCircuitBreaker(
            failure_threshold=2, success_threshold=2, recovery_timeout=0.1
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == SyncCircuitBreaker.HALF_OPEN

        cb.record_success()
        cb.record_success()
        assert cb.state == SyncCircuitBreaker.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test half-open reopens on failure."""
        cb = SyncCircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == SyncCircuitBreaker.HALF_OPEN

        cb.record_failure()
        assert cb.state == SyncCircuitBreaker.OPEN

    def test_reset(self):
        """Test manual reset."""
        cb = SyncCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == SyncCircuitBreaker.OPEN

        cb.reset()
        assert cb.state == SyncCircuitBreaker.CLOSED
        assert cb.can_execute() is True

    def test_get_stats(self):
        """Test stats reporting."""
        cb = SyncCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        cb.record_failure()
        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 5


class TestRateLimitedDecorator:
    """Tests for @rate_limited decorator."""

    def test_sync_function(self):
        """Test decorator on sync function."""
        call_count = 0

        @rate_limited(rpm=600, tpm=100_000)
        def my_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = my_func()
        assert result == "ok"
        assert call_count == 1
        assert my_func._rate_limiter.total_requests == 1

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test decorator on async function."""
        call_count = 0

        @rate_limited(rpm=600, tpm=100_000)
        async def my_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await my_func()
        assert result == "ok"
        assert call_count == 1

    def test_preserves_function_name(self):
        """Test decorator preserves function metadata."""

        @rate_limited(rpm=100)
        def my_special_func():
            """My docstring."""
            pass

        assert my_special_func.__name__ == "my_special_func"
        assert my_special_func.__doc__ == "My docstring."


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""

    def test_sync_no_retry_needed(self):
        """Test sync function that succeeds first try."""

        @with_retry(max_retries=3)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_sync_retries_on_failure(self):
        """Test sync function retries and eventually succeeds."""
        attempts = {"count": 0}

        @with_retry(max_retries=3, initial_delay=0.01)
        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ValueError("fail")
            return "ok"

        assert flaky() == "ok"
        assert attempts["count"] == 3

    def test_sync_exhausts_retries(self):
        """Test sync function raises after exhausting retries."""

        @with_retry(max_retries=2, initial_delay=0.01)
        def always_fail():
            raise ValueError("permanent failure")

        with pytest.raises(ValueError, match="permanent failure"):
            always_fail()

    def test_specific_exceptions(self):
        """Test retries only on specified exceptions."""

        @with_retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        def wrong_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            wrong_error()

    @pytest.mark.asyncio
    async def test_async_retries(self):
        """Test async function with retries."""
        attempts = {"count": 0}

        @with_retry(max_retries=3, initial_delay=0.01)
        async def async_flaky():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ConnectionError("fail")
            return "ok"

        result = await async_flaky()
        assert result == "ok"
        assert attempts["count"] == 2

    def test_preserves_function_name(self):
        """Test decorator preserves function metadata."""

        @with_retry(max_retries=2)
        def my_retrying_func():
            """Retry docstring."""
            pass

        assert my_retrying_func.__name__ == "my_retrying_func"


class TestCircuitProtectedDecorator:
    """Tests for @circuit_protected decorator."""

    def test_sync_normal_operation(self):
        """Test circuit passes through on success."""

        @circuit_protected(failure_threshold=3)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_sync_opens_after_failures(self):
        """Test circuit opens after threshold failures."""

        @circuit_protected(failure_threshold=2, recovery_timeout=10)
        def always_fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                always_fail()

        # Now circuit is open
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            always_fail()

    def test_sync_fallback(self):
        """Test fallback when circuit is open."""

        @circuit_protected(
            failure_threshold=2,
            recovery_timeout=10,
            fallback=lambda: "fallback_value",
        )
        def failing():
            raise ValueError("fail")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                failing()

        # Now should use fallback
        result = failing()
        assert result == "fallback_value"

    @pytest.mark.asyncio
    async def test_async_circuit(self):
        """Test circuit breaker on async function."""

        @circuit_protected(failure_threshold=2, recovery_timeout=10)
        async def async_fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await async_fail()

        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await async_fail()

    def test_circuit_breaker_accessible(self):
        """Test circuit breaker is accessible on decorated function."""

        @circuit_protected(failure_threshold=5)
        def my_func():
            return "ok"

        assert hasattr(my_func, "_circuit_breaker")
        assert my_func._circuit_breaker.failure_threshold == 5
