"""Tests for circuit breaker functionality."""

import pytest
import time

from llm_rate_guard.router import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_failure_threshold(self):
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()

        # Should be open now
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Success resets the failure counter."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0

        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """Circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=1, half_open_timeout=0.1)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

        # Wait for timeout
        time.sleep(0.15)

        # Should allow one request (half-open)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self):
        """Circuit closes after success threshold in half-open state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            success_threshold=2,
            half_open_timeout=0.01,
        )

        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for half-open
        time.sleep(0.02)
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            half_open_timeout=0.01,
        )

        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for half-open
        time.sleep(0.02)
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_high_threshold_stays_closed(self):
        """Very high threshold effectively disables circuit breaker."""
        cb = CircuitBreaker(failure_threshold=999999)

        for _ in range(100):
            cb.record_failure()

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True
