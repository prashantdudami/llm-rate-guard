#!/usr/bin/env python3
"""Monitoring and observability example for LLM Rate Guard.

This example shows how to:
- Track metrics
- Set up health checks
- Integrate with monitoring systems (OpenTelemetry example)
- Monitor latency percentiles for SLAs
"""

import asyncio
import random

from llm_rate_guard import ProviderConfig, RateGuardClient


# Simulated OpenTelemetry tracer (in real code, use opentelemetry-api)
class MockTracer:
    """Mock tracer for demo purposes."""

    def start_span(self, name):
        return MockSpan(name)


class MockSpan:
    """Mock span for demo purposes."""

    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self):
        print(f"  [Trace] {self.name}: {self.attributes}")


tracer = MockTracer()


def opentelemetry_hook(metrics, event):
    """Send metrics to OpenTelemetry (or similar system)."""
    span = tracer.start_span("llm_request")
    span.set_attribute("provider", event["provider"])
    span.set_attribute("model", event["model"])
    span.set_attribute("latency_ms", round(event["latency_ms"], 2))
    span.set_attribute("tokens", event["input_tokens"] + event["output_tokens"])
    span.set_attribute("cached", event["cached"])
    span.set_attribute("success", event["success"])
    span.set_attribute("cost_usd", round(event["estimated_cost_usd"], 6))
    span.end()


async def main():
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="openai",
                model="gpt-4-turbo",
            ),
        ],
        cache_enabled=True,
    )

    # Register the OpenTelemetry hook
    client.get_metrics().add_hook(opentelemetry_hook)

    print("=" * 60)
    print("Monitoring Demo")
    print("=" * 60)

    # Make some requests
    print("\n1. Making requests with tracing...\n")

    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "What is 2+2?",  # Will be cached
        "What color is the sky?",
        "What is 2+2?",  # Will be cached
    ]

    for q in questions:
        await client.complete(
            [{"role": "user", "content": q}],
            max_tokens=20,
        )
        print()

    # Health check
    print("\n" + "=" * 60)
    print("2. Health Check")
    print("=" * 60)

    health = await client.health_check()
    print(f"""
Status: {health['status']}
Healthy: {health['healthy']}
Started: {health['started']}
Active Requests: {health['active_requests']}

Providers:
  Total: {health['providers']['total']}
  Healthy: {health['providers']['healthy']}
  Unhealthy: {health['providers']['unhealthy']}

Cache:
  Enabled: {health['cache']['enabled']}
  Entries: {health['cache']['entries']}
  Hit Rate: {health['cache']['hit_rate_pct']:.1f}%

Queue:
  Enabled: {health['queue']['enabled']}
  Size: {health['queue']['size']}
  Max Size: {health['queue']['max_size']}

Metrics:
  Total Requests: {health['metrics']['total_requests']}
  Success Rate: {health['metrics']['success_rate_pct']:.1f}%
  Avg Latency: {health['metrics']['avg_latency_ms']:.1f}ms
""")

    # Detailed metrics
    print("=" * 60)
    print("3. Detailed Metrics")
    print("=" * 60)

    metrics = client.get_metrics()
    metrics_dict = metrics.to_dict()

    print(f"""
Request Metrics:
  Total: {metrics_dict['total_requests']}
  Successful: {metrics_dict['successful_requests']}
  Failed: {metrics_dict['failed_requests']}
  Success Rate: {metrics_dict['success_rate_pct']:.1f}%

Cache Metrics:
  Hits: {metrics_dict['cache_hits']}
  Misses: {metrics_dict['cache_misses']}
  Hit Rate: {metrics_dict['cache_hit_rate_pct']:.1f}%

Latency (SLA Monitoring):
  Average: {metrics_dict['avg_latency_ms']:.1f}ms
  Min: {metrics_dict['min_latency_ms']:.1f}ms
  Max: {metrics_dict['max_latency_ms']:.1f}ms
  p50: {metrics_dict['latency_p50_ms']:.1f}ms
  p90: {metrics_dict['latency_p90_ms']:.1f}ms
  p95: {metrics_dict['latency_p95_ms']:.1f}ms
  p99: {metrics_dict['latency_p99_ms']:.1f}ms

Token Usage:
  Input: {metrics_dict['total_input_tokens']:,}
  Output: {metrics_dict['total_output_tokens']:,}
  Total: {metrics_dict['total_tokens']:,}

Cost Tracking:
  Estimated Cost: ${metrics_dict['estimated_cost_usd']:.4f}
""")

    # Provider stats
    print("=" * 60)
    print("4. Provider Stats")
    print("=" * 60)

    for stat in client.get_provider_stats():
        print(f"""
Provider: {stat['provider_id']}
  Healthy: {stat['is_healthy']}
  Total Requests: {stat['total_requests']}
  Total Failures: {stat['total_failures']}
  Rate Limits: {stat['total_rate_limits']}
  Avg Latency: {stat['avg_latency_ms']:.1f}ms
  In Cooldown: {stat['in_cooldown']}
""")

    # Cache stats
    print("=" * 60)
    print("5. Cache Stats")
    print("=" * 60)

    cache_stats = client.get_cache_stats()
    print(f"""
Cache Statistics:
  Entries: {cache_stats['entries']}
  Hit Rate: {cache_stats['hit_rate']:.1f}%
  Hits: {cache_stats['hits']}
  Misses: {cache_stats['misses']}
  Evictions: {cache_stats['evictions']}
""")


if __name__ == "__main__":
    asyncio.run(main())
