"""Multi-region and multi-provider router with health tracking."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from llm_rate_guard.config import ProviderConfig, RateGuardConfig
from llm_rate_guard.exceptions import AllProvidersExhausted, ProviderError, RateLimitExceeded
from llm_rate_guard.logging import get_logger
from llm_rate_guard.providers import BaseProvider, CompletionResponse, Message, get_provider_class
from llm_rate_guard.rate_limiter import MultiProviderRateLimiter


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for a provider."""

    failure_threshold: int = 5
    success_threshold: int = 2
    half_open_timeout: float = 30.0

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        self.success_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if time.monotonic() - self.last_failure_time >= self.half_open_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False

        # HALF_OPEN - allow one request through
        return True


@dataclass
class ProviderHealth:
    """Health status for a provider."""

    provider_id: str
    """Unique identifier for the provider."""

    is_healthy: bool = True
    """Whether the provider is currently healthy."""

    consecutive_failures: int = 0
    """Number of consecutive failures."""

    last_failure_time: Optional[float] = None
    """Timestamp of last failure."""

    cooldown_until: Optional[float] = None
    """Timestamp when cooldown ends."""

    total_requests: int = 0
    """Total requests sent to this provider."""

    total_failures: int = 0
    """Total failures from this provider."""

    total_rate_limits: int = 0
    """Total rate limit hits from this provider."""

    total_timeouts: int = 0
    """Total timeout errors from this provider."""

    avg_latency_ms: float = 0.0
    """Average latency in milliseconds."""

    _latency_samples: list[float] = field(default_factory=list, repr=False)
    """Recent latency samples for calculating average."""

    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    """Circuit breaker for this provider."""

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.is_healthy = True
        self.consecutive_failures = 0
        self.total_requests += 1
        self.circuit_breaker.record_success()

        # Update latency (keep last 100 samples)
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 100:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self, is_rate_limit: bool = False, is_timeout: bool = False) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = time.monotonic()
        self.circuit_breaker.record_failure()

        if is_rate_limit:
            self.total_rate_limits += 1
        if is_timeout:
            self.total_timeouts += 1

    def set_cooldown(self, duration_seconds: float) -> None:
        """Put provider in cooldown."""
        self.is_healthy = False
        self.cooldown_until = time.monotonic() + duration_seconds

    def check_cooldown(self) -> bool:
        """Check if cooldown has expired.

        Returns:
            True if provider is now available, False if still in cooldown.
        """
        if self.cooldown_until is None:
            return True

        if time.monotonic() >= self.cooldown_until:
            self.cooldown_until = None
            self.is_healthy = True
            return True

        return False

    def can_accept_request(self) -> bool:
        """Check if provider can accept a request (health + circuit breaker)."""
        if not self.is_healthy:
            return False
        return self.circuit_breaker.can_execute()


@dataclass
class ManagedProvider:
    """A provider instance with its configuration and health status."""

    provider: BaseProvider
    """The provider instance."""

    config: ProviderConfig
    """Provider configuration."""

    health: ProviderHealth
    """Health status."""

    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        parts = [self.config.type.value, self.config.model]
        if self.config.region:
            parts.append(self.config.region)
        return ":".join(parts)


class MultiRouter:
    """Routes requests across multiple providers with health tracking and failover.

    Features:
    - Weighted load balancing
    - Health-based routing (avoids unhealthy providers)
    - Automatic cooldown on rate limits
    - Failover to next provider on failure
    """

    def __init__(
        self,
        config: RateGuardConfig,
        rate_limiters: Optional[MultiProviderRateLimiter] = None,
    ):
        """Initialize the router.

        Args:
            config: Rate guard configuration.
            rate_limiters: Optional shared rate limiter manager.
        """
        self.config = config
        self.rate_limiters = rate_limiters or MultiProviderRateLimiter()
        self._providers: list[ManagedProvider] = []
        self._lock = asyncio.Lock()

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        logger = get_logger()
        initialization_errors: list[tuple[str, str]] = []

        for provider_config in self.config.providers:
            try:
                # Get provider class and instantiate
                provider_class = get_provider_class(provider_config.type.value)
                provider = provider_class(provider_config)

                # Create circuit breaker with config
                cb_config = self.config.circuit_breaker
                circuit_breaker = CircuitBreaker(
                    failure_threshold=cb_config.failure_threshold,
                    success_threshold=cb_config.success_threshold,
                    half_open_timeout=cb_config.half_open_timeout,
                ) if cb_config.enabled else CircuitBreaker(failure_threshold=999999)

                # Create managed provider
                managed = ManagedProvider(
                    provider=provider,
                    config=provider_config,
                    health=ProviderHealth(
                        provider_id=f"{provider_config.type.value}:{provider_config.model}"
                        + (f":{provider_config.region}" if provider_config.region else ""),
                        circuit_breaker=circuit_breaker,
                    ),
                )

                # Register rate limiter for this provider
                self.rate_limiters.get_or_create(
                    provider_id=managed.provider_id,
                    rpm_limit=provider.rpm_limit,
                    tpm_limit=provider.tpm_limit,
                )

                self._providers.append(managed)
                logger.debug(f"Initialized provider: {managed.provider_id}")

            except Exception as e:
                error_msg = str(e)
                initialization_errors.append((provider_config.type.value, error_msg))
                logger.warning(
                    f"Failed to initialize provider {provider_config.type.value}: {error_msg}"
                )

        if not self._providers:
            error_details = "; ".join(f"{p}: {e}" for p, e in initialization_errors)
            raise ValueError(f"No providers could be initialized. Errors: {error_details}")

    def _get_available_providers(self) -> list[ManagedProvider]:
        """Get list of available (healthy) providers."""
        available = []
        for managed in self._providers:
            # Check if cooldown has expired
            managed.health.check_cooldown()

            # Check health and circuit breaker
            if managed.health.can_accept_request():
                available.append(managed)

        return available

    def _select_provider(
        self,
        available: list[ManagedProvider],
        exclude: Optional[set[str]] = None,
    ) -> Optional[ManagedProvider]:
        """Select a provider using weighted random selection.

        Args:
            available: List of available providers.
            exclude: Set of provider IDs to exclude.

        Returns:
            Selected provider or None if none available.
        """
        exclude = exclude or set()
        candidates = [p for p in available if p.provider_id not in exclude]

        if not candidates:
            return None

        # Calculate total weight
        total_weight = sum(p.config.weight for p in candidates)
        if total_weight <= 0:
            return random.choice(candidates)

        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0.0

        for provider in candidates:
            cumulative += provider.config.weight
            if r <= cumulative:
                return provider

        return candidates[-1]

    async def route(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        estimated_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Route a completion request to an available provider.

        Args:
            messages: List of messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            estimated_tokens: Estimated total tokens (input + output).
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse from the provider.

        Raises:
            AllProvidersExhausted: If all providers are unavailable.
        """
        attempted: set[str] = set()
        last_error: Optional[Exception] = None

        # Estimate tokens if not provided
        if estimated_tokens is None:
            total_chars = sum(len(m.content) for m in messages)
            estimated_tokens = (total_chars // 4) + max_tokens

        while len(attempted) < len(self._providers):
            # Get available providers
            available = self._get_available_providers()

            if not available:
                # All providers in cooldown, wait a bit and retry
                await asyncio.sleep(1.0)
                available = self._get_available_providers()

            if not available:
                break

            # Select a provider
            managed = self._select_provider(available, exclude=attempted)
            if managed is None:
                break

            attempted.add(managed.provider_id)

            try:
                # Acquire rate limit capacity
                wait_time = await self.rate_limiters.acquire(
                    managed.provider_id,
                    estimated_tokens,
                )

                if wait_time > 0:
                    # We had to wait for rate limit
                    pass

                # Make the request with timeout
                timeout_seconds = managed.config.timeout_seconds or 60.0
                try:
                    response = await asyncio.wait_for(
                        managed.provider.complete(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    managed.health.record_failure(is_timeout=True)
                    raise ProviderError(
                        message=f"Request timed out after {timeout_seconds}s",
                        provider=managed.provider_id,
                    )

                # Record success
                managed.health.record_success(response.latency_ms)

                # Adjust rate limiter based on actual tokens
                actual_tokens = response.usage.total_tokens
                limiter = self.rate_limiters.get(managed.provider_id)
                if limiter:
                    limiter.record_actual_tokens(actual_tokens, estimated_tokens)

                return response

            except RateLimitExceeded as e:
                # Record rate limit and put in cooldown
                managed.health.record_failure(is_rate_limit=True)
                managed.health.set_cooldown(self.config.cooldown_seconds)
                last_error = e

                if not self.config.failover_enabled:
                    raise

                # Continue to next provider
                continue

            except Exception as e:
                # Record failure
                managed.health.record_failure(is_rate_limit=False)
                last_error = e

                # After multiple consecutive failures, put in short cooldown
                if managed.health.consecutive_failures >= 3:
                    managed.health.set_cooldown(self.config.cooldown_seconds / 2)

                if not self.config.failover_enabled:
                    raise

                # Continue to next provider
                continue

        # All providers exhausted
        raise AllProvidersExhausted(
            message=f"All {len(attempted)} providers exhausted: {last_error}",
            attempted_providers=list(attempted),
        )

    def get_provider_stats(self) -> list[dict]:
        """Get stats for all providers."""
        stats = []
        for managed in self._providers:
            health = managed.health
            limiter_stats = self.rate_limiters.get(managed.provider_id)

            stats.append({
                "provider_id": managed.provider_id,
                "type": managed.config.type.value,
                "model": managed.config.model,
                "region": managed.config.region,
                "is_healthy": health.is_healthy,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "total_failures": health.total_failures,
                "total_rate_limits": health.total_rate_limits,
                "avg_latency_ms": health.avg_latency_ms,
                "in_cooldown": health.cooldown_until is not None,
                "rate_limiter": limiter_stats.get_stats() if limiter_stats else None,
            })

        return stats

    @property
    def healthy_provider_count(self) -> int:
        """Get count of healthy providers."""
        return len(self._get_available_providers())

    @property
    def total_provider_count(self) -> int:
        """Get total count of providers."""
        return len(self._providers)
