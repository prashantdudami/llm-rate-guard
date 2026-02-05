#!/usr/bin/env python3
"""Environment variable configuration example for LLM Rate Guard.

This example shows how to configure the client entirely from
environment variables, which is useful for:
- Containerized deployments (Docker, Kubernetes)
- CI/CD pipelines
- Configuration management systems
"""

import asyncio
import os

from llm_rate_guard import create_client_from_env, load_config_from_env


def setup_example_env():
    """Set up example environment variables for demo."""
    # JSON format for providers
    os.environ["LLM_RATE_GUARD_PROVIDERS"] = """[
        {
            "type": "openai",
            "model": "gpt-4-turbo",
            "weight": 2.0
        },
        {
            "type": "anthropic",
            "model": "claude-3-haiku-20240307",
            "weight": 1.0
        }
    ]"""

    # Or use numbered format:
    # os.environ["LLM_RATE_GUARD_PROVIDER_1_TYPE"] = "openai"
    # os.environ["LLM_RATE_GUARD_PROVIDER_1_MODEL"] = "gpt-4-turbo"
    # os.environ["LLM_RATE_GUARD_PROVIDER_1_WEIGHT"] = "2.0"
    # os.environ["LLM_RATE_GUARD_PROVIDER_2_TYPE"] = "anthropic"
    # os.environ["LLM_RATE_GUARD_PROVIDER_2_MODEL"] = "claude-3-haiku-20240307"

    # Cache settings
    os.environ["LLM_RATE_GUARD_CACHE_ENABLED"] = "true"
    os.environ["LLM_RATE_GUARD_CACHE_TTL"] = "7200"  # 2 hours
    os.environ["LLM_RATE_GUARD_CACHE_MAX_ENTRIES"] = "5000"

    # Retry settings
    os.environ["LLM_RATE_GUARD_RETRY_MAX"] = "5"
    os.environ["LLM_RATE_GUARD_RETRY_INITIAL_DELAY"] = "0.5"

    # Failover settings
    os.environ["LLM_RATE_GUARD_FAILOVER_ENABLED"] = "true"
    os.environ["LLM_RATE_GUARD_COOLDOWN_SECONDS"] = "30.0"

    # Global limits
    os.environ["LLM_RATE_GUARD_GLOBAL_RPM_LIMIT"] = "500"
    os.environ["LLM_RATE_GUARD_GLOBAL_TPM_LIMIT"] = "5000000"

    # Queue settings
    os.environ["LLM_RATE_GUARD_QUEUE_ENABLED"] = "true"
    os.environ["LLM_RATE_GUARD_MAX_QUEUE_SIZE"] = "500"

    # Circuit breaker
    os.environ["LLM_RATE_GUARD_CIRCUIT_BREAKER_ENABLED"] = "true"
    os.environ["LLM_RATE_GUARD_CIRCUIT_BREAKER_FAILURE_THRESHOLD"] = "3"

    print("Environment variables set for demo.\n")


async def main():
    # Set up example environment
    setup_example_env()

    print("=" * 60)
    print("Environment Configuration Demo")
    print("=" * 60)

    # Method 1: Load config from environment (for inspection/modification)
    print("\n1. Loading configuration from environment...")
    config = load_config_from_env()

    if config:
        print(f"   Loaded {len(config.providers)} providers")
        print(f"   Cache enabled: {config.cache.enabled}")
        print(f"   Cache TTL: {config.cache.ttl_seconds}s")
        print(f"   Global RPM: {config.global_rpm_limit}")
        print(f"   Failover: {config.failover_enabled}")
    else:
        print("   No configuration found in environment")

    # Method 2: Create client directly from environment
    print("\n2. Creating client from environment...")
    client = create_client_from_env()

    if client:
        print(f"   Client created with {client.total_providers} providers")

        # Make a test request
        print("\n3. Making a test request...")
        try:
            response = await client.complete(
                messages=[{"role": "user", "content": "Say 'Hello from env config!'"}],
                max_tokens=30,
            )
            print(f"   Response: {response.content}")
            print(f"   Provider: {response.provider}")
        except Exception as e:
            print(f"   Error (expected if no API keys): {type(e).__name__}")

        # Show health
        health = await client.health_check()
        print(f"\n4. Health Status: {health['status']}")
    else:
        print("   Could not create client from environment")

    # Show all LLM_RATE_GUARD env vars
    print("\n" + "=" * 60)
    print("Current Environment Variables:")
    print("=" * 60)
    for key, value in sorted(os.environ.items()):
        if key.startswith("LLM_RATE_GUARD"):
            # Truncate long values
            display_value = value if len(value) < 50 else value[:47] + "..."
            print(f"  {key}={display_value}")


if __name__ == "__main__":
    asyncio.run(main())
