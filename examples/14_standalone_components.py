#!/usr/bin/env python3
"""Example: Standalone Components

Demonstrates using individual components (rate limiter, circuit breaker,
retry) without the full RateGuardClient. Perfect for adding specific
capabilities to existing code with minimal changes.

Features shown:
- @rate_limited decorator
- @with_retry decorator
- @circuit_protected decorator
- SyncRateLimiter standalone usage
- SyncCircuitBreaker standalone usage
- Composing decorators
"""

import asyncio
import time
import random

from llm_rate_guard.standalone import (
    SyncRateLimiter,
    SyncCircuitBreaker,
    rate_limited,
    with_retry,
    circuit_protected,
)


def main():
    """Run standalone component examples."""

    print("=" * 60)
    print("LLM Rate Guard - Standalone Components")
    print("=" * 60)

    # =========================================================================
    # Example 1: SyncRateLimiter
    # =========================================================================
    print("\n1. SyncRateLimiter - Add to Existing Code")
    print("-" * 40)

    limiter = SyncRateLimiter(rpm=60, tpm=100_000)

    # Simulate API calls
    for i in range(5):
        wait = limiter.acquire(estimated_tokens=500)
        if wait > 0:
            print(f"  Request {i+1}: waited {wait:.3f}s for rate limit")
        else:
            print(f"  Request {i+1}: immediate (no wait)")

    stats = limiter.get_stats()
    print(f"\n  Stats: {stats['total_requests']} requests, "
          f"{stats['total_tokens']} tokens, "
          f"{stats['rate_limit_hits']} rate limit hits")

    # =========================================================================
    # Example 2: @rate_limited Decorator (Sync)
    # =========================================================================
    print("\n2. @rate_limited Decorator (Sync)")
    print("-" * 40)

    @rate_limited(rpm=120, tpm=500_000)
    def call_bedrock_api(prompt: str) -> str:
        """Simulated Bedrock API call."""
        time.sleep(0.01)  # Simulate latency
        return f"Response to: {prompt}"

    for i in range(3):
        result = call_bedrock_api(f"Question {i+1}")
        print(f"  {result}")

    # Access the underlying limiter
    print(f"  Limiter stats: {call_bedrock_api._rate_limiter.get_stats()['total_requests']} calls")

    # =========================================================================
    # Example 3: @rate_limited Decorator (Async)
    # =========================================================================
    print("\n3. @rate_limited Decorator (Async)")
    print("-" * 40)

    @rate_limited(rpm=100, tpm=200_000)
    async def async_call_api(prompt: str) -> str:
        """Simulated async API call."""
        await asyncio.sleep(0.01)
        return f"Async response to: {prompt}"

    async def run_async_demo():
        for i in range(3):
            result = await async_call_api(f"Async Q{i+1}")
            print(f"  {result}")

    asyncio.run(run_async_demo())

    # =========================================================================
    # Example 4: @with_retry Decorator
    # =========================================================================
    print("\n4. @with_retry Decorator")
    print("-" * 40)

    attempt_counter = {"count": 0}

    @with_retry(max_retries=3, initial_delay=0.1, backoff_base=2.0)
    def unreliable_api_call() -> str:
        """Simulated unreliable API that fails twice then succeeds."""
        attempt_counter["count"] += 1
        if attempt_counter["count"] <= 2:
            print(f"  Attempt {attempt_counter['count']}: Failed (simulated)")
            raise ConnectionError("Simulated failure")
        print(f"  Attempt {attempt_counter['count']}: Succeeded!")
        return "Success after retries"

    result = unreliable_api_call()
    print(f"  Final result: {result}")

    # =========================================================================
    # Example 5: SyncCircuitBreaker
    # =========================================================================
    print("\n5. SyncCircuitBreaker")
    print("-" * 40)

    cb = SyncCircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    # Simulate failures that trip the circuit
    for i in range(5):
        if cb.can_execute():
            # Simulate a failure
            cb.record_failure()
            print(f"  Call {i+1}: Failed (state={cb.state})")
        else:
            print(f"  Call {i+1}: Blocked by circuit breaker (state={cb.state})")

    # Wait for recovery
    print("  Waiting 2s for recovery timeout...")
    time.sleep(2.1)

    # Circuit should be half-open now
    if cb.can_execute():
        cb.record_success()
        print(f"  Recovery call: Succeeded (state={cb.state})")
    if cb.can_execute():
        cb.record_success()
        print(f"  Recovery call 2: Succeeded (state={cb.state})")

    # =========================================================================
    # Example 6: @circuit_protected Decorator
    # =========================================================================
    print("\n6. @circuit_protected Decorator")
    print("-" * 40)

    @circuit_protected(failure_threshold=2, fallback=lambda: "Fallback response")
    def protected_api_call() -> str:
        """API call protected by circuit breaker."""
        if random.random() < 0.8:
            raise ConnectionError("API down")
        return "API response"

    for i in range(5):
        try:
            result = protected_api_call()
            print(f"  Call {i+1}: {result}")
        except (ConnectionError, RuntimeError) as e:
            print(f"  Call {i+1}: {e}")

    # =========================================================================
    # Example 7: Composing Decorators
    # =========================================================================
    print("\n7. Composing Decorators (Rate Limit + Retry + Circuit Breaker)")
    print("-" * 40)

    print("""
    # Stack decorators for comprehensive protection
    @rate_limited(rpm=250, tpm=2_000_000)
    @with_retry(max_retries=3, retryable_exceptions=(ConnectionError,))
    @circuit_protected(failure_threshold=5)
    async def robust_bedrock_call(prompt: str):
        # Your existing Bedrock code - unchanged
        response = await bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet",
            body=json.dumps({"prompt": prompt})
        )
        return response

    # Now your function has:
    # 1. Rate limiting (250 RPM, 2M TPM)
    # 2. Automatic retry with backoff (3 attempts)
    # 3. Circuit breaker (opens after 5 failures)
    """)

    # =========================================================================
    # Example 8: Using with Existing Bedrock Client
    # =========================================================================
    print("\n8. Integration with Existing Code")
    print("-" * 40)

    print("""
    import boto3

    # Your existing setup - UNCHANGED
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    # Just add the decorator
    @rate_limited(rpm=250, tpm=2_000_000)
    @with_retry(max_retries=3, retryable_exceptions=(Exception,))
    def invoke_claude(prompt: str) -> str:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
            }),
        )
        return json.loads(response['body'].read())

    # Use exactly as before
    result = invoke_claude("What is Python?")
    """)

    print("\n" + "=" * 60)
    print("Standalone component examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
