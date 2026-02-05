"""Tests for metrics module."""

import pytest

from llm_rate_guard.metrics import (
    Metrics,
    PercentileTracker,
    SimpleCostTracker,
    CostTracker,  # Alias for SimpleCostTracker
    create_cost_tracker,
    LLM_COST_GUARD_AVAILABLE,
    DEFAULT_PRICING,
)


class TestPercentileTracker:
    """Tests for PercentileTracker class."""

    def test_empty_tracker(self):
        """Empty tracker returns 0 for percentiles."""
        tracker = PercentileTracker()
        assert tracker.p50 == 0.0
        assert tracker.p90 == 0.0
        assert tracker.p99 == 0.0

    def test_single_value(self):
        """Single value is returned for all percentiles."""
        tracker = PercentileTracker()
        tracker.add(100.0)
        assert tracker.p50 == 100.0
        assert tracker.p99 == 100.0

    def test_percentile_calculation(self):
        """Percentiles are calculated correctly."""
        tracker = PercentileTracker()
        for i in range(1, 101):  # 1-100
            tracker.add(float(i))

        assert tracker.p50 == pytest.approx(50.0, abs=1.0)
        assert tracker.p90 == pytest.approx(90.0, abs=1.0)
        assert tracker.p95 == pytest.approx(95.0, abs=1.0)
        assert tracker.p99 == pytest.approx(99.0, abs=1.0)

    def test_percentile_boundaries(self):
        """Boundary percentiles work correctly."""
        tracker = PercentileTracker()
        tracker.add(1.0)
        tracker.add(10.0)
        tracker.add(100.0)

        assert tracker.percentile(0) == 1.0
        assert tracker.percentile(100) == 100.0

    def test_max_samples_limit(self):
        """Tracker respects max_samples limit."""
        tracker = PercentileTracker(max_samples=100)

        for i in range(200):
            tracker.add(float(i))

        assert len(tracker.samples) == 100

    def test_clear(self):
        """Clear removes all samples."""
        tracker = PercentileTracker()
        tracker.add(1.0)
        tracker.add(2.0)
        tracker.clear()

        assert len(tracker.samples) == 0
        assert tracker.p50 == 0.0


class TestSimpleCostTracker:
    """Tests for SimpleCostTracker class."""

    def test_estimate_cost_default(self):
        """Unknown models use default pricing."""
        tracker = SimpleCostTracker()

        cost = tracker.estimate_cost("unknown-model", 1_000_000, 500_000)

        # Default: $1/1M input, $2/1M output
        expected = 1.0 + 1.0  # 1M input + 0.5M output
        assert cost == pytest.approx(expected, rel=0.01)

    def test_estimate_cost_known_model(self):
        """Known models use specific pricing."""
        tracker = SimpleCostTracker()

        # GPT-4 pricing: $30/1M input, $60/1M output
        cost = tracker.estimate_cost("gpt-4", 1_000_000, 500_000)
        expected = 30.0 + 30.0
        assert cost == pytest.approx(expected, rel=0.01)

    def test_estimate_cost_partial_model_match(self):
        """Model names can be partial matches."""
        tracker = SimpleCostTracker()

        # Should match "claude-3-sonnet"
        cost = tracker.estimate_cost(
            "anthropic.claude-3-sonnet-20240229-v1:0",
            1_000_000,
            1_000_000,
        )
        # Claude 3 Sonnet: $3/1M input, $15/1M output
        expected = 3.0 + 15.0
        assert cost == pytest.approx(expected, rel=0.01)

    def test_record_accumulates(self):
        """Recording accumulates costs."""
        tracker = SimpleCostTracker()

        tracker.record("gpt-4", "openai", 100_000, 50_000)
        tracker.record("gpt-4", "openai", 100_000, 50_000)

        assert tracker.total_cost_usd > 0
        assert tracker.cost_by_provider["openai"] > 0
        assert tracker.cost_by_model["gpt-4"] > 0

    def test_clear_resets(self):
        """Clear resets all tracked costs."""
        tracker = SimpleCostTracker()
        tracker.record("gpt-4", "openai", 100_000, 50_000)
        tracker.clear()

        assert tracker.total_cost_usd == 0.0
        assert len(tracker.cost_by_provider) == 0
        assert len(tracker.cost_by_model) == 0

    def test_cost_tracker_alias(self):
        """CostTracker is an alias for SimpleCostTracker."""
        assert CostTracker is SimpleCostTracker


class TestCostTrackerFactory:
    """Tests for cost tracker factory function."""

    def test_create_cost_tracker_returns_simple_by_default(self):
        """create_cost_tracker returns SimpleCostTracker when llm-cost-guard not installed."""
        tracker = create_cost_tracker(use_llm_cost_guard=False)
        assert isinstance(tracker, SimpleCostTracker)

    def test_create_cost_tracker_respects_flag(self):
        """create_cost_tracker respects use_llm_cost_guard flag."""
        # When flag is False, always use SimpleCostTracker
        tracker = create_cost_tracker(use_llm_cost_guard=False)
        assert isinstance(tracker, SimpleCostTracker)

    def test_llm_cost_guard_available_flag_exists(self):
        """LLM_COST_GUARD_AVAILABLE flag is defined."""
        assert isinstance(LLM_COST_GUARD_AVAILABLE, bool)


class TestMetricsEnhancements:
    """Tests for enhanced Metrics features."""

    def test_latency_percentiles_tracked(self):
        """Latency percentiles are tracked on successful requests."""
        metrics = Metrics()

        for i in range(100):
            metrics.record_request(
                success=True,
                provider="test",
                input_tokens=10,
                output_tokens=5,
                latency_ms=float(i + 1),
                model="gpt-4",
            )

        assert metrics.latency_percentiles.p50 > 0
        assert metrics.latency_percentiles.p90 > 0

    def test_latency_not_tracked_for_cached(self):
        """Cached responses don't affect latency percentiles."""
        metrics = Metrics()

        # Record a fast cached response
        metrics.record_request(
            success=True,
            provider="cache",
            input_tokens=0,
            output_tokens=0,
            latency_ms=1.0,
            cached=True,
            model="",
        )

        # Percentiles should be empty (no non-cached requests)
        assert len(metrics.latency_percentiles.samples) == 0

    def test_cost_tracking(self):
        """Costs are tracked with model info."""
        metrics = Metrics()

        metrics.record_request(
            success=True,
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100.0,
            model="gpt-4",
        )

        assert metrics.cost_tracker.total_cost_usd > 0

    def test_to_dict_includes_percentiles_and_cost(self):
        """to_dict includes percentile and cost data."""
        metrics = Metrics()

        metrics.record_request(
            success=True,
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100.0,
            model="gpt-4",
        )

        d = metrics.to_dict()

        assert "latency_p50_ms" in d
        assert "latency_p90_ms" in d
        assert "latency_p95_ms" in d
        assert "latency_p99_ms" in d
        assert "estimated_cost_usd" in d
        assert "cost_by_provider" in d
        assert "cost_by_model" in d

    def test_hooks_called(self):
        """Metrics hooks are called on record_request."""
        metrics = Metrics()
        hook_data = []

        def my_hook(metrics, event):
            hook_data.append(event)

        metrics.add_hook(my_hook)

        metrics.record_request(
            success=True,
            provider="test",
            input_tokens=10,
            output_tokens=5,
            latency_ms=50.0,
            model="test-model",
        )

        assert len(hook_data) == 1
        assert hook_data[0]["provider"] == "test"
        assert hook_data[0]["model"] == "test-model"

    def test_hook_exception_doesnt_break_metrics(self):
        """Hook exceptions don't break metrics recording."""
        metrics = Metrics()

        def bad_hook(metrics, event):
            raise RuntimeError("Hook error")

        metrics.add_hook(bad_hook)

        # Should not raise
        metrics.record_request(
            success=True,
            provider="test",
            input_tokens=10,
            output_tokens=5,
            latency_ms=50.0,
        )

        assert metrics.total_requests == 1

    def test_remove_hook(self):
        """Hooks can be removed."""
        metrics = Metrics()
        call_count = [0]

        def my_hook(metrics, event):
            call_count[0] += 1

        metrics.add_hook(my_hook)
        metrics.record_request(
            success=True, provider="test",
            input_tokens=10, output_tokens=5, latency_ms=50.0
        )
        assert call_count[0] == 1

        metrics.remove_hook(my_hook)
        metrics.record_request(
            success=True, provider="test",
            input_tokens=10, output_tokens=5, latency_ms=50.0
        )
        assert call_count[0] == 1  # Should not increase

    def test_reset_clears_percentiles_and_cost(self):
        """Reset clears percentile and cost trackers."""
        metrics = Metrics()

        metrics.record_request(
            success=True,
            provider="test",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100.0,
            model="gpt-4",
        )

        metrics.reset()

        assert len(metrics.latency_percentiles.samples) == 0
        assert metrics.cost_tracker.total_cost_usd == 0.0
