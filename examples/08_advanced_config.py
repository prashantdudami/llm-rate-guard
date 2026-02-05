#!/usr/bin/env python3
"""Advanced configuration example for LLM Rate Guard.

This example shows a complete production-ready configuration with
all options configured explicitly.
"""

import asyncio
import os

from llm_rate_guard import (
    CacheConfig,
    CircuitBreakerConfig,
    ProviderConfig,
    ProviderType,
    RateGuardClient,
    RateGuardConfig,
    RetryConfig,
)


async def main():
    # Full configuration with all options
    config = RateGuardConfig(
        # Multiple providers with explicit configuration
        providers=[
            # Primary: AWS Bedrock (Claude)
            ProviderConfig(
                type=ProviderType.BEDROCK,
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-east-1",
                rpm_limit=250,  # Override default
                tpm_limit=2_000_000,
                weight=3.0,  # Primary, gets most traffic
                timeout_seconds=120.0,  # 2 minute timeout
                extra={
                    # Provider-specific options
                    "inference_profile": "default",
                },
            ),
            # Secondary: AWS Bedrock (different region)
            ProviderConfig(
                type=ProviderType.BEDROCK,
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-west-2",
                rpm_limit=250,
                tpm_limit=2_000_000,
                weight=2.0,
                timeout_seconds=120.0,
            ),
            # Tertiary: OpenAI GPT-4 (fallback)
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4-turbo",
                # api_key loaded from OPENAI_API_KEY env var
                rpm_limit=500,
                tpm_limit=150_000,
                weight=1.0,  # Lower weight = fallback
                timeout_seconds=60.0,
            ),
            # Quaternary: Anthropic direct (final fallback)
            ProviderConfig(
                type=ProviderType.ANTHROPIC,
                model="claude-3-haiku-20240307",
                # api_key loaded from ANTHROPIC_API_KEY env var
                rpm_limit=50,
                tpm_limit=40_000,
                weight=0.5,  # Last resort
                timeout_seconds=60.0,
            ),
        ],
        # Retry behavior for transient failures
        retry=RetryConfig(
            max_retries=3,
            initial_delay=1.0,  # Start with 1 second
            max_delay=60.0,  # Cap at 1 minute
            exponential_base=2.0,  # Double each retry
            jitter=True,  # Add randomness to prevent thundering herd
        ),
        # Caching for cost reduction
        cache=CacheConfig(
            enabled=True,
            mode="exact",  # "exact" or "semantic"
            similarity_threshold=0.95,  # For semantic mode
            max_entries=10_000,
            max_size_bytes=100_000_000,  # 100MB total cache
            max_entry_size_bytes=1_000_000,  # 1MB per entry
            ttl_seconds=3600,  # 1 hour TTL
            embedding_model=None,  # Use provider default for semantic
        ),
        # Circuit breaker for failing providers
        circuit_breaker=CircuitBreakerConfig(
            enabled=True,
            failure_threshold=5,  # Open after 5 consecutive failures
            success_threshold=2,  # Close after 2 successes in half-open
            half_open_timeout=30.0,  # Try half-open after 30 seconds
        ),
        # Global rate limiting (across all providers)
        global_rpm_limit=1000,  # Overall RPM cap
        global_tpm_limit=10_000_000,  # Overall TPM cap
        # Failover behavior
        failover_enabled=True,
        cooldown_seconds=60.0,  # Cool down rate-limited providers for 1 min
        # Priority queue for request management
        queue_enabled=True,
        max_queue_size=1000,
        # Input validation
        max_message_length=100_000,  # Max chars per message
        max_messages_per_request=100,  # Max messages per request
    )

    # Create client with config
    async with RateGuardClient(config=config) as client:
        print("=" * 60)
        print("Advanced Configuration Demo")
        print("=" * 60)

        # Show configuration summary
        print(f"""
Configuration Summary:
  Providers: {len(config.providers)}
  Cache: {'enabled' if config.cache.enabled else 'disabled'}
  Circuit Breaker: {'enabled' if config.circuit_breaker.enabled else 'disabled'}
  Failover: {'enabled' if config.failover_enabled else 'disabled'}
  Queue: {'enabled' if config.queue_enabled else 'disabled'}
  Global RPM Limit: {config.global_rpm_limit}
  Global TPM Limit: {config.global_tpm_limit:,}
""")

        # Show provider configuration
        print("Provider Configuration:")
        for i, p in enumerate(config.providers):
            print(f"  {i+1}. {p.type.value}")
            print(f"     Model: {p.model}")
            print(f"     Region: {p.region or 'N/A'}")
            print(f"     Weight: {p.weight}")
            print(f"     RPM/TPM: {p.rpm_limit or 'default'} / {p.tpm_limit or 'default'}")
            print(f"     Timeout: {p.timeout_seconds}s")
            print()

        # Make a request
        print("Making a test request...")
        response = await client.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the benefits of multi-provider LLM setups in one sentence."},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print(f"\nResponse from {response.provider} ({response.model}):")
        print(f"  {response.content}")
        print(f"\nMetrics:")
        print(f"  Tokens: {response.usage.total_tokens}")
        print(f"  Latency: {response.latency_ms:.1f}ms")
        print(f"  Cached: {response.cached}")


if __name__ == "__main__":
    asyncio.run(main())
