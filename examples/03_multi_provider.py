#!/usr/bin/env python3
"""Multi-provider failover example for LLM Rate Guard.

This example shows how to configure multiple providers with automatic
failover when one is rate limited or unavailable.
"""

import asyncio

from llm_rate_guard import (
    ProviderConfig,
    RateGuardClient,
    RateGuardConfig,
    RetryConfig,
    CacheConfig,
    CircuitBreakerConfig,
)


async def main():
    # Configure multiple providers with different weights
    config = RateGuardConfig(
        providers=[
            # Primary provider - highest weight
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-east-1",
                weight=3.0,  # Gets 3x traffic
            ),
            # Secondary provider
            ProviderConfig(
                type="openai",
                model="gpt-4-turbo",
                weight=2.0,  # Gets 2x traffic
            ),
            # Tertiary provider - backup
            ProviderConfig(
                type="anthropic",
                model="claude-3-haiku-20240307",
                weight=1.0,  # Gets 1x traffic
            ),
        ],
        # Retry configuration
        retry=RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,
        ),
        # Cache for repeated queries
        cache=CacheConfig(
            enabled=True,
            ttl_seconds=3600,
        ),
        # Circuit breaker to avoid hammering failing providers
        circuit_breaker=CircuitBreakerConfig(
            enabled=True,
            failure_threshold=5,  # Open after 5 failures
            success_threshold=2,  # Close after 2 successes
            half_open_timeout=30.0,
        ),
        # Failover settings
        failover_enabled=True,
        cooldown_seconds=60.0,
    )

    client = RateGuardClient(config=config)

    print("Provider configuration:")
    for i, provider in enumerate(config.providers):
        print(f"  {i+1}. {provider.type.value} ({provider.model}) - weight {provider.weight}")

    # Make requests - they'll be routed based on weight
    print("\nMaking 10 requests...")
    for i in range(10):
        response = await client.complete(
            messages=[{"role": "user", "content": f"Say hello #{i}"}],
            max_tokens=20,
        )
        print(f"  Request {i}: via {response.provider} ({response.model})")

    # Show distribution
    print("\nTraffic distribution:")
    for stat in client.get_provider_stats():
        pct = (stat["total_requests"] / max(1, client.get_metrics().total_requests)) * 100
        print(f"  {stat['provider_id']}: {stat['total_requests']} requests ({pct:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
