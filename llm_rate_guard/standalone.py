"""Standalone components for incremental adoption.

Use individual rate limiting, retry, circuit breaker, and caching
components without the full RateGuardClient. Perfect for adding
rate-guard capabilities to existing code with minimal changes.

Example:
    ```python
    from llm_rate_guard.standalone import rate_limited, with_retry

    @rate_limited(rpm=250, tpm=2_000_000)
    @with_retry(max_retries=3)
    async def call_bedrock(prompt):
        return await existing_client.invoke(prompt)
    ```
"""

import asyncio
import functools
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Synchronous Rate Limiter
# =============================================================================


class SyncRateLimiter:
    """Thread-safe synchronous rate limiter using the token bucket algorithm.

    Designed for Lambda, scripts, and sync code. No asyncio required.

    Example:
        ```python
        limiter = SyncRateLimiter(rpm=250, tpm=2_000_000)

        # Block until capacity available
        limiter.acquire(estimated_tokens=500)
        response = bedrock_client.invoke(...)

        # Or check without blocking
        if limiter.try_acquire(estimated_tokens=500):
            response = bedrock_client.invoke(...)
        ```
    """

    def __init__(
        self,
        rpm: int = 250,
        tpm: int = 2_000_000,
        burst_multiplier: float = 1.0,
    ):
        """Initialize sync rate limiter.

        Args:
            rpm: Requests per minute limit.
            tpm: Tokens per minute limit.
            burst_multiplier: Allow bursting up to this multiple.
        """
        self.rpm = rpm
        self.tpm = tpm

        # RPM bucket
        self._rpm_capacity = rpm * burst_multiplier
        self._rpm_tokens = self._rpm_capacity
        self._rpm_refill_rate = rpm / 60.0

        # TPM bucket
        self._tpm_capacity = tpm * burst_multiplier
        self._tpm_tokens = self._tpm_capacity
        self._tpm_refill_rate = tpm / 60.0

        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_seconds = 0.0
        self.rate_limit_hits = 0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._rpm_tokens = min(self._rpm_capacity, self._rpm_tokens + elapsed * self._rpm_refill_rate)
        self._tpm_tokens = min(self._tpm_capacity, self._tpm_tokens + elapsed * self._tpm_refill_rate)
        self._last_refill = now

    def acquire(self, estimated_tokens: int = 1000, timeout: float = 60.0) -> float:
        """Acquire capacity, blocking until available.

        Args:
            estimated_tokens: Estimated token usage.
            timeout: Maximum seconds to wait.

        Returns:
            Total wait time in seconds.

        Raises:
            TimeoutError: If timeout exceeded.
        """
        start = time.monotonic()
        total_wait = 0.0

        while True:
            with self._lock:
                self._refill()

                if self._rpm_tokens >= 1 and self._tpm_tokens >= estimated_tokens:
                    self._rpm_tokens -= 1
                    self._tpm_tokens -= estimated_tokens
                    self.total_requests += 1
                    self.total_tokens += estimated_tokens
                    self.total_wait_seconds += total_wait
                    if total_wait > 0:
                        self.rate_limit_hits += 1
                    return total_wait

                # Calculate wait time
                rpm_wait = max(0, (1 - self._rpm_tokens) / self._rpm_refill_rate)
                tpm_wait = max(0, (estimated_tokens - self._tpm_tokens) / self._tpm_refill_rate)
                wait = max(rpm_wait, tpm_wait)

            elapsed = time.monotonic() - start
            if elapsed + wait > timeout:
                raise TimeoutError(
                    f"Rate limit acquisition timed out after {elapsed:.1f}s "
                    f"(timeout={timeout}s)"
                )

            time.sleep(min(wait, 0.1))  # Sleep in small increments
            total_wait = time.monotonic() - start

    def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire capacity without blocking.

        Args:
            estimated_tokens: Estimated token usage.

        Returns:
            True if acquired, False if would need to wait.
        """
        with self._lock:
            self._refill()

            if self._rpm_tokens >= 1 and self._tpm_tokens >= estimated_tokens:
                self._rpm_tokens -= 1
                self._tpm_tokens -= estimated_tokens
                self.total_requests += 1
                self.total_tokens += estimated_tokens
                return True
            return False

    def record_actual_tokens(self, actual: int, estimated: int) -> None:
        """Adjust after learning actual token usage.

        Args:
            actual: Actual tokens used.
            estimated: Previously estimated tokens.
        """
        diff = estimated - actual
        if diff > 0:
            with self._lock:
                self._tpm_tokens = min(self._tpm_capacity, self._tpm_tokens + diff)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "rpm": self.rpm,
                "tpm": self.tpm,
                "rpm_available": self._rpm_tokens,
                "tpm_available": self._tpm_tokens,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_wait_seconds": self.total_wait_seconds,
                "rate_limit_hits": self.rate_limit_hits,
            }


# =============================================================================
# Synchronous Circuit Breaker
# =============================================================================


class SyncCircuitBreaker:
    """Thread-safe synchronous circuit breaker.

    Tracks failures and opens the circuit when threshold is reached.
    After a cooldown period, allows a test request through (half-open).

    Example:
        ```python
        cb = SyncCircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        if cb.can_execute():
            try:
                result = call_api()
                cb.record_success()
            except Exception as e:
                cb.record_failure()
                raise
        ```
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening.
            success_threshold: Successes in half-open to close.
            recovery_timeout: Seconds before trying half-open.
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current circuit state."""
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._success_count = 0
            return self._state

    def can_execute(self) -> bool:
        """Check if a request is allowed."""
        return self.state != self.OPEN

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == self.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._success_count = 0

            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


# =============================================================================
# Decorators
# =============================================================================


def rate_limited(
    rpm: int = 250,
    tpm: int = 2_000_000,
    estimated_tokens: int = 1000,
    burst_multiplier: float = 1.0,
) -> Callable[[F], F]:
    """Decorator to add rate limiting to any function (sync or async).

    Example:
        ```python
        @rate_limited(rpm=250, tpm=2_000_000)
        async def call_bedrock(prompt):
            return await client.invoke(prompt)

        @rate_limited(rpm=100, tpm=500_000)
        def call_openai(prompt):
            return client.completions.create(prompt=prompt)
        ```

    Args:
        rpm: Requests per minute limit.
        tpm: Tokens per minute limit.
        estimated_tokens: Default estimated tokens per request.
        burst_multiplier: Burst multiplier.
    """
    limiter = SyncRateLimiter(rpm=rpm, tpm=tpm, burst_multiplier=burst_multiplier)

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tokens = kwargs.pop("_estimated_tokens", estimated_tokens)
                # Use thread to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, limiter.acquire, tokens)
                return await func(*args, **kwargs)
            async_wrapper._rate_limiter = limiter  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tokens = kwargs.pop("_estimated_tokens", estimated_tokens)
                limiter.acquire(tokens)
                return func(*args, **kwargs)
            sync_wrapper._rate_limiter = limiter  # type: ignore[attr-defined]
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple[type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator to add retry with exponential backoff (sync or async).

    Example:
        ```python
        @with_retry(max_retries=3, backoff_base=2.0)
        async def call_api():
            return await client.invoke(...)

        @with_retry(max_retries=5, retryable_exceptions=(ConnectionError, TimeoutError))
        def call_service():
            return requests.get(...)
        ```

    Args:
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        backoff_base: Exponential backoff base.
        jitter: Add random jitter to delays.
        retryable_exceptions: Tuple of exception types to retry on.
            Defaults to (Exception,).
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def _calc_delay(attempt: int) -> float:
        delay = initial_delay * (backoff_base ** attempt)
        delay = min(delay, max_delay)
        if jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)
        return delay

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: Optional[Exception] = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_error = e
                        if attempt >= max_retries:
                            raise
                        delay = _calc_delay(attempt)
                        await asyncio.sleep(delay)
                raise last_error  # type: ignore[misc]
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: Optional[Exception] = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_error = e
                        if attempt >= max_retries:
                            raise
                        delay = _calc_delay(attempt)
                        time.sleep(delay)
                raise last_error  # type: ignore[misc]
            return sync_wrapper  # type: ignore[return-value]

    return decorator


def circuit_protected(
    failure_threshold: int = 5,
    success_threshold: int = 2,
    recovery_timeout: float = 30.0,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """Decorator to add circuit breaker protection (sync or async).

    Example:
        ```python
        @circuit_protected(failure_threshold=5, recovery_timeout=30)
        async def call_api():
            return await client.invoke(...)

        # With fallback
        @circuit_protected(failure_threshold=3, fallback=lambda: "default")
        def call_service():
            return requests.get(...)
        ```

    Args:
        failure_threshold: Consecutive failures before opening.
        success_threshold: Successes in half-open to close.
        recovery_timeout: Seconds before attempting recovery.
        fallback: Optional fallback function when circuit is open.
    """
    breaker = SyncCircuitBreaker(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        recovery_timeout=recovery_timeout,
    )

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not breaker.can_execute():
                    if fallback is not None:
                        result = fallback(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            return await result
                        return result
                    raise RuntimeError(
                        f"Circuit breaker is open (failures={breaker._failure_count})"
                    )
                try:
                    result = await func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception:
                    breaker.record_failure()
                    raise
            async_wrapper._circuit_breaker = breaker  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not breaker.can_execute():
                    if fallback is not None:
                        return fallback(*args, **kwargs)
                    raise RuntimeError(
                        f"Circuit breaker is open (failures={breaker._failure_count})"
                    )
                try:
                    result = func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception:
                    breaker.record_failure()
                    raise
            sync_wrapper._circuit_breaker = breaker  # type: ignore[attr-defined]
            return sync_wrapper  # type: ignore[return-value]

    return decorator
