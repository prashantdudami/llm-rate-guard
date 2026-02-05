#!/usr/bin/env python3
"""Multi-region routing example for LLM Rate Guard.

This example shows how to use multiple AWS regions with Bedrock to
multiply your effective rate limits. Each region has independent limits.
"""

import asyncio

from llm_rate_guard import ProviderConfig, RateGuardClient


async def main():
    # Configure multiple regions for higher throughput
    # AWS Bedrock: 250 RPM per region, so 4 regions = ~1000 RPM
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-east-1",  # 250 RPM
                weight=1.0,
            ),
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-west-2",  # +250 RPM
                weight=1.0,
            ),
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="eu-west-1",  # +250 RPM
                weight=1.0,
            ),
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="ap-northeast-1",  # +250 RPM
                weight=1.0,
            ),
        ],
        cache_enabled=True,
    )

    print(f"Configured {client.total_providers} providers across regions")
    print(f"Healthy providers: {client.healthy_providers}")

    # Send multiple concurrent requests - they'll be distributed across regions
    async def make_request(i: int) -> str:
        response = await client.complete(
            messages=[{"role": "user", "content": f"Say 'Hello {i}' in French"}],
            max_tokens=50,
        )
        return f"Request {i}: {response.content} (via {response.provider})"

    # Send 20 concurrent requests
    print("\nSending 20 concurrent requests...")
    tasks = [make_request(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)

    # Check provider distribution
    print("\nProvider Stats:")
    for stat in client.get_provider_stats():
        print(f"  {stat['provider_id']}: {stat['total_requests']} requests")


if __name__ == "__main__":
    asyncio.run(main())
