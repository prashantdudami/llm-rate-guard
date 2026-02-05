#!/usr/bin/env python3
"""Basic usage example for LLM Rate Guard.

This example shows how to get started with the library using a single provider.
"""

import asyncio

from llm_rate_guard import ProviderConfig, RateGuardClient


async def main():
    # Create a client with a single provider
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="openai",  # or "bedrock", "azure_openai", "vertex", "anthropic"
                model="gpt-4-turbo",
                # api_key is read from OPENAI_API_KEY env var automatically
            ),
        ],
        cache_enabled=True,  # Enable response caching
    )

    # Make a simple completion request
    response = await client.complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Latency: {response.latency_ms:.1f}ms")
    print(f"Cached: {response.cached}")

    # Make the same request again - should be cached
    response2 = await client.complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    print(f"\nSecond request cached: {response2.cached}")

    # Check metrics
    metrics = client.get_metrics()
    print(f"\nMetrics:")
    print(f"  Total requests: {metrics.total_requests}")
    print(f"  Cache hit rate: {metrics.cache_hit_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
