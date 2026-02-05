#!/usr/bin/env python3
"""Multi-tenancy example for LLM Rate Guard.

This example shows how to track requests by tenant for:
- Cost attribution
- Usage tracking
- Per-tenant quotas
"""

import asyncio

from llm_rate_guard import (
    ProviderConfig,
    QuotaManager,
    RateGuardClient,
    RequestContext,
)


async def main():
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="openai",
                model="gpt-4-turbo",
            ),
        ],
        cache_enabled=False,  # Disable cache for demo
    )

    # Set up quota manager
    quota = QuotaManager()

    # Configure quotas for different tenants
    quota.set_limit(
        "acme-corp",
        tokens_per_day=100_000,
        requests_per_day=1000,
        cost_per_day_usd=50.0,
    )

    quota.set_limit(
        "startup-inc",
        tokens_per_day=10_000,
        requests_per_day=100,
        cost_per_day_usd=5.0,
    )

    # Very limited quota for demo
    quota.set_limit(
        "free-tier",
        tokens_per_day=1000,
        requests_per_day=10,
        cost_per_day_usd=1.0,
    )

    client.set_quota_manager(quota)

    # Middleware to enforce quotas and record usage
    async def enforce_quota(data, ctx):
        """Block requests that exceed quota."""
        if ctx and ctx.tenant_id:
            if not quota.check(ctx.tenant_id, requests=1):
                print(f"  [BLOCKED] Tenant {ctx.tenant_id} over quota!")
                return None  # Block request
        return data

    async def record_usage(data, ctx):
        """Record usage after successful request."""
        if ctx and ctx.tenant_id:
            tokens = data["usage"]["total_tokens"]
            cost = data.get("estimated_cost_usd", 0)
            quota.record(ctx.tenant_id, tokens=tokens, requests=1, cost_usd=cost)

    client.add_pre_middleware(enforce_quota)
    client.add_post_middleware(record_usage)

    # Simulate requests from different tenants
    tenants = [
        ("acme-corp", "user-123", {"project": "chatbot"}),
        ("acme-corp", "user-456", {"project": "support"}),
        ("startup-inc", "user-789", {"project": "mvp"}),
        ("free-tier", "user-000", {}),
    ]

    print("Making requests from different tenants...\n")

    for tenant_id, user_id, labels in tenants:
        ctx = RequestContext(
            tenant_id=tenant_id,
            user_id=user_id,
            labels=labels,
            cost_center=f"{tenant_id}-engineering",
        )

        try:
            response = await client.complete(
                messages=[{"role": "user", "content": "Say hello!"}],
                max_tokens=20,
                context=ctx,
            )
            print(f"  [{tenant_id}/{user_id}] Success: {response.content[:30]}...")
        except Exception as e:
            print(f"  [{tenant_id}/{user_id}] Error: {e}")

    # Show usage by tenant
    print("\n" + "=" * 50)
    print("Usage by Tenant:")
    print("=" * 50)

    for tenant_id, _, _ in set((t[0], t[1], "") for t in tenants):
        usage = quota.get_usage(tenant_id)
        print(f"\n{tenant_id}:")
        print(f"  Tokens: {usage['tokens_used']:,} / {usage['tokens_limit']:,}")
        print(f"  Requests: {usage['requests_used']} / {usage['requests_limit']}")
        if usage['cost_limit_usd']:
            print(f"  Cost: ${usage['cost_used_usd']:.4f} / ${usage['cost_limit_usd']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
