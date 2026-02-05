"""Retry logic with exponential backoff and jitter."""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from llm_rate_guard.config import RetryConfig
from llm_rate_guard.exceptions import RateLimitExceeded, ProviderError

T = TypeVar("T")


@dataclass
class RetryStats:
    """Statistics from a retry operation."""

    attempts: int = 0
    """Total number of attempts made."""

    total_delay: float = 0.0
    """Total time spent waiting in delays (seconds)."""

    success: bool = False
    """Whether the operation ultimately succeeded."""

    last_error: Optional[Exception] = None
    """The last error encountered, if any."""


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter.

    Implements truncated exponential backoff with optional jitter
    for robust retry behavior.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize the retry handler.

        Args:
            config: Retry configuration. Uses defaults if not provided.
        """
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate the delay before the next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed).
            retry_after: Optional server-provided retry-after value.

        Returns:
            Delay in seconds before next attempt.
        """
        # If server provided retry-after, respect it (with a cap)
        if retry_after is not None:
            return min(retry_after, self.config.max_delay)

        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = self.config.initial_delay * (self.config.exponential_base ** attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled (±25% randomization)
        if self.config.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure positive delay

        return delay

    async def execute(
        self,
        operation: Callable[[], T],
        should_retry: Optional[Callable[[Exception], bool]] = None,
    ) -> tuple[T, RetryStats]:
        """Execute an operation with retry logic.

        Args:
            operation: Async callable to execute.
            should_retry: Optional function to determine if an exception is retryable.
                         Defaults to retrying on RateLimitExceeded.

        Returns:
            Tuple of (result, stats).

        Raises:
            The last exception if all retries are exhausted.
        """
        if should_retry is None:
            should_retry = self._default_should_retry

        stats = RetryStats()

        for attempt in range(self.config.max_retries + 1):
            stats.attempts = attempt + 1

            try:
                result = await operation()
                stats.success = True
                return result, stats

            except Exception as e:
                stats.last_error = e

                # Check if we should retry
                if attempt >= self.config.max_retries:
                    # No more retries
                    raise

                if not should_retry(e):
                    # Not a retryable error
                    raise

                # Calculate delay
                retry_after = getattr(e, "retry_after", None)
                delay = self.calculate_delay(attempt, retry_after)
                stats.total_delay += delay

                # Wait before retrying
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise stats.last_error  # type: ignore

    def _default_should_retry(self, error: Exception) -> bool:
        """Default retry predicate.

        Returns True for rate limit errors, False for others.
        """
        if isinstance(error, RateLimitExceeded):
            return True

        # Retry on certain provider errors (e.g., 429, 503, 504)
        if isinstance(error, ProviderError):
            if error.status_code in (429, 503, 504):
                return True

        return False


async def with_retry(
    operation: Callable[[], T],
    config: Optional[RetryConfig] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """Convenience function to execute an operation with retry logic.

    Args:
        operation: Async callable to execute.
        config: Retry configuration.
        should_retry: Optional retry predicate.

    Returns:
        Result of the operation.
    """
    handler = RetryHandler(config)
    result, _ = await handler.execute(operation, should_retry)
    return result
