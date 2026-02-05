"""Metrics collection for LLM Rate Guard."""

import bisect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Try to import llm-cost-guard for advanced cost tracking
try:
    from llm_cost_guard import CostTracker as LLMCostGuardTracker
    from llm_cost_guard import UsageRecord

    LLM_COST_GUARD_AVAILABLE = True
except ImportError:
    LLM_COST_GUARD_AVAILABLE = False
    LLMCostGuardTracker = None  # type: ignore
    UsageRecord = None  # type: ignore


# Default pricing per 1M tokens (USD) - rough estimates
# Used as fallback when llm-cost-guard is not installed
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic.claude": {"input": 3.0, "output": 15.0},  # Bedrock Claude default
    "amazon.titan": {"input": 0.8, "output": 1.0},
    "default": {"input": 1.0, "output": 2.0},
}


@dataclass
class PercentileTracker:
    """Tracks latency percentiles using a sorted list of samples."""

    samples: list[float] = field(default_factory=list)
    """Sorted list of latency samples."""

    max_samples: int = 10000
    """Maximum samples to keep (older samples are dropped)."""

    def add(self, value: float) -> None:
        """Add a sample."""
        bisect.insort(self.samples, value)
        if len(self.samples) > self.max_samples:
            # Remove oldest sample (middle of list roughly)
            self.samples.pop(0)

    def percentile(self, p: float) -> float:
        """Get the p-th percentile (0-100)."""
        if not self.samples:
            return 0.0
        if p <= 0:
            return self.samples[0]
        if p >= 100:
            return self.samples[-1]

        index = (p / 100) * (len(self.samples) - 1)
        lower = int(index)
        upper = min(lower + 1, len(self.samples) - 1)
        weight = index - lower

        return self.samples[lower] * (1 - weight) + self.samples[upper] * weight

    @property
    def p50(self) -> float:
        """Median (50th percentile)."""
        return self.percentile(50)

    @property
    def p90(self) -> float:
        """90th percentile."""
        return self.percentile(90)

    @property
    def p95(self) -> float:
        """95th percentile."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile."""
        return self.percentile(99)

    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()


@runtime_checkable
class CostTrackerProtocol(Protocol):
    """Protocol for cost trackers."""

    total_cost_usd: float
    cost_by_provider: dict[str, float]
    cost_by_model: dict[str, float]

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float: ...
    def record(self, model: str, provider: str, input_tokens: int, output_tokens: int) -> float: ...
    def clear(self) -> None: ...


@dataclass
class SimpleCostTracker:
    """Simple cost tracker with hardcoded pricing.

    This is a fallback when llm-cost-guard is not installed.
    For production use, install llm-cost-guard for more accurate
    and up-to-date pricing:

        pip install llm-rate-guard[cost-tracking]
    """

    total_cost_usd: float = 0.0
    """Total estimated cost in USD."""

    cost_by_provider: dict[str, float] = field(default_factory=dict)
    """Cost breakdown by provider."""

    cost_by_model: dict[str, float] = field(default_factory=dict)
    """Cost breakdown by model."""

    pricing: dict[str, dict[str, float]] = field(default_factory=lambda: DEFAULT_PRICING.copy())
    """Pricing per 1M tokens."""

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        # Find matching pricing
        pricing = self.pricing.get("default", {"input": 1.0, "output": 2.0})
        for key in self.pricing:
            if key in model.lower():
                pricing = self.pricing[key]
                break

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage and calculate cost.

        Returns:
            Estimated cost for this request.
        """
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        self.total_cost_usd += cost

        self.cost_by_provider[provider] = self.cost_by_provider.get(provider, 0.0) + cost
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost

        return cost

    def clear(self) -> None:
        """Reset cost tracking."""
        self.total_cost_usd = 0.0
        self.cost_by_provider.clear()
        self.cost_by_model.clear()


class LLMCostGuardAdapter:
    """Adapter for llm-cost-guard library.

    Provides the same interface as SimpleCostTracker but uses
    llm-cost-guard for more accurate pricing and features.
    """

    def __init__(self, tracker: Optional[Any] = None):
        """Initialize the adapter.

        Args:
            tracker: Optional llm-cost-guard CostTracker instance.
                    If not provided, creates a new one.
        """
        if not LLM_COST_GUARD_AVAILABLE:
            raise ImportError(
                "llm-cost-guard is not installed. "
                "Install with: pip install llm-rate-guard[cost-tracking]"
            )

        self._tracker = tracker or LLMCostGuardTracker()
        self._cost_by_provider: dict[str, float] = {}
        self._cost_by_model: dict[str, float] = {}

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost in USD."""
        return self._tracker.total_cost

    @property
    def cost_by_provider(self) -> dict[str, float]:
        """Cost breakdown by provider."""
        return self._cost_by_provider

    @property
    def cost_by_model(self) -> dict[str, float]:
        """Cost breakdown by model."""
        return self._cost_by_model

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request using llm-cost-guard."""
        return self._tracker.estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage using llm-cost-guard."""
        # Create usage record
        record = UsageRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Record in llm-cost-guard
        cost = self._tracker.record(record)

        # Also track locally for breakdown
        self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + cost
        self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost

        return cost

    def clear(self) -> None:
        """Reset cost tracking."""
        self._tracker.reset()
        self._cost_by_provider.clear()
        self._cost_by_model.clear()

    @property
    def underlying_tracker(self) -> Any:
        """Get the underlying llm-cost-guard tracker for advanced features."""
        return self._tracker


def create_cost_tracker(use_llm_cost_guard: bool = True) -> CostTrackerProtocol:
    """Create a cost tracker.

    Args:
        use_llm_cost_guard: If True and llm-cost-guard is installed, use it.
                           Otherwise, fall back to SimpleCostTracker.

    Returns:
        A cost tracker instance.
    """
    if use_llm_cost_guard and LLM_COST_GUARD_AVAILABLE:
        return LLMCostGuardAdapter()  # type: ignore
    return SimpleCostTracker()


# Backwards compatibility alias
CostTracker = SimpleCostTracker


# Type for metrics hooks/callbacks
MetricsHook = Callable[["Metrics", dict[str, Any]], None]


@dataclass
class Metrics:
    """Aggregated metrics for the rate guard client."""

    # Request counts
    total_requests: int = 0
    """Total number of requests processed."""

    successful_requests: int = 0
    """Number of successful requests."""

    failed_requests: int = 0
    """Number of failed requests."""

    # Cache metrics
    cache_hits: int = 0
    """Number of cache hits."""

    cache_misses: int = 0
    """Number of cache misses."""

    # Rate limiting
    rate_limit_waits: int = 0
    """Number of times we waited for rate limits."""

    total_rate_limit_wait_ms: float = 0.0
    """Total time spent waiting for rate limits (ms)."""

    # Failover
    failovers: int = 0
    """Number of times we failed over to another provider."""

    # Token usage
    total_input_tokens: int = 0
    """Total input tokens consumed."""

    total_output_tokens: int = 0
    """Total output tokens generated."""

    # Latency
    total_latency_ms: float = 0.0
    """Total latency across all requests (ms)."""

    min_latency_ms: float = float("inf")
    """Minimum request latency (ms)."""

    max_latency_ms: float = 0.0
    """Maximum request latency (ms)."""

    # Provider breakdown
    requests_by_provider: dict[str, int] = field(default_factory=dict)
    """Request count by provider."""

    tokens_by_provider: dict[str, int] = field(default_factory=dict)
    """Token usage by provider."""

    # Timing
    start_time: float = field(default_factory=time.time)
    """When metrics collection started."""

    # Advanced tracking
    latency_percentiles: PercentileTracker = field(default_factory=PercentileTracker)
    """Latency percentile tracker (p50, p90, p95, p99)."""

    cost_tracker: Any = field(default=None)
    """Cost estimation tracker (SimpleCostTracker or LLMCostGuardAdapter)."""

    # Hooks for external systems (OpenTelemetry, etc.)
    _hooks: list[MetricsHook] = field(default_factory=list, repr=False)
    """Registered metric hooks."""

    # Whether to use llm-cost-guard if available
    _use_llm_cost_guard: bool = field(default=True, repr=False)
    """Whether to use llm-cost-guard for cost tracking."""

    def __post_init__(self) -> None:
        """Initialize cost tracker after dataclass init."""
        if self.cost_tracker is None:
            self.cost_tracker = create_cost_tracker(use_llm_cost_guard=self._use_llm_cost_guard)

    @property
    def using_llm_cost_guard(self) -> bool:
        """Check if using llm-cost-guard for cost tracking."""
        return isinstance(self.cost_tracker, LLMCostGuardAdapter) if LLM_COST_GUARD_AVAILABLE else False

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request (ms)."""
        return (self.total_latency_ms / self.successful_requests) if self.successful_requests > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def uptime_seconds(self) -> float:
        """Time since metrics collection started."""
        return time.time() - self.start_time

    @property
    def requests_per_minute(self) -> float:
        """Average requests per minute."""
        uptime_minutes = self.uptime_seconds / 60
        return (self.total_requests / uptime_minutes) if uptime_minutes > 0 else 0.0

    def add_hook(self, hook: MetricsHook) -> None:
        """Add a metrics hook for external integrations.

        Hooks are called after each request with metrics and event data.
        Use this for OpenTelemetry, Prometheus, or custom monitoring.

        Args:
            hook: Callback function(metrics, event_data).
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: MetricsHook) -> bool:
        """Remove a metrics hook.

        Returns:
            True if hook was found and removed.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def record_request(
        self,
        success: bool,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cached: bool = False,
        rate_limit_wait_ms: float = 0.0,
        failover: bool = False,
        model: str = "",
    ) -> None:
        """Record a completed request.

        Args:
            success: Whether the request succeeded.
            provider: Provider that handled the request.
            input_tokens: Input tokens consumed.
            output_tokens: Output tokens generated.
            latency_ms: Request latency in milliseconds.
            cached: Whether response was served from cache.
            rate_limit_wait_ms: Time spent waiting for rate limits.
            failover: Whether this was a failover request.
            model: Model identifier for cost tracking.
        """
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if rate_limit_wait_ms > 0:
            self.rate_limit_waits += 1
            self.total_rate_limit_wait_ms += rate_limit_wait_ms

        if failover:
            self.failovers += 1

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

        # Track latency percentiles
        if success and not cached:
            self.latency_percentiles.add(latency_ms)

        # Track costs
        cost = 0.0
        if model and input_tokens + output_tokens > 0:
            cost = self.cost_tracker.record(model, provider, input_tokens, output_tokens)

        # By provider
        self.requests_by_provider[provider] = self.requests_by_provider.get(provider, 0) + 1
        self.tokens_by_provider[provider] = (
            self.tokens_by_provider.get(provider, 0) + input_tokens + output_tokens
        )

        # Call hooks
        event_data = {
            "success": success,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cached": cached,
            "rate_limit_wait_ms": rate_limit_wait_ms,
            "failover": failover,
            "estimated_cost_usd": cost,
        }
        for hook in self._hooks:
            try:
                hook(self, event_data)
            except Exception:
                pass  # Don't let hooks break metrics

    def record_cache_check(self, hit: bool) -> None:
        """Record a cache check (before request).

        Args:
            hit: Whether it was a cache hit.
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limit_waits = 0
        self.total_rate_limit_wait_ms = 0.0
        self.failovers = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float("inf")
        self.max_latency_ms = 0.0
        self.requests_by_provider.clear()
        self.tokens_by_provider.clear()
        self.latency_percentiles.clear()
        self.cost_tracker.clear()
        self.start_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_pct": self.success_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_pct": self.cache_hit_rate,
            "rate_limit_waits": self.rate_limit_waits,
            "total_rate_limit_wait_ms": self.total_rate_limit_wait_ms,
            "failovers": self.failovers,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float("inf") else 0,
            "max_latency_ms": self.max_latency_ms,
            "latency_p50_ms": self.latency_percentiles.p50,
            "latency_p90_ms": self.latency_percentiles.p90,
            "latency_p95_ms": self.latency_percentiles.p95,
            "latency_p99_ms": self.latency_percentiles.p99,
            "requests_per_minute": self.requests_per_minute,
            "uptime_seconds": self.uptime_seconds,
            "requests_by_provider": self.requests_by_provider,
            "tokens_by_provider": self.tokens_by_provider,
            "estimated_cost_usd": self.cost_tracker.total_cost_usd,
            "cost_by_provider": self.cost_tracker.cost_by_provider,
            "cost_by_model": self.cost_tracker.cost_by_model,
        }

    def __repr__(self) -> str:
        return (
            f"Metrics(requests={self.total_requests}, "
            f"success_rate={self.success_rate:.1f}%, "
            f"cache_hit_rate={self.cache_hit_rate:.1f}%, "
            f"avg_latency={self.avg_latency_ms:.1f}ms)"
        )
