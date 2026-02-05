#!/usr/bin/env python3
"""Middleware example for LLM Rate Guard.

This example shows how to use middleware for:
- Request logging
- Request modification
- Authentication/authorization
- Response processing
"""

import asyncio
import json
from datetime import datetime

from llm_rate_guard import (
    ProviderConfig,
    RateGuardClient,
    RequestContext,
)


# Example middleware functions

async def logging_middleware(data, ctx):
    """Log all incoming requests."""
    timestamp = datetime.now().isoformat()
    tenant = ctx.tenant_id if ctx else "anonymous"
    msg_count = len(data["messages"])

    print(f"[{timestamp}] Request from {tenant}: {msg_count} messages")
    return data


async def add_system_prompt(data, ctx):
    """Add a default system prompt if none exists."""
    messages = data["messages"]

    # Check if first message is a system prompt
    first_msg = messages[0] if messages else {}
    if first_msg.get("role") != "system":
        # Add our default system prompt
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful, harmless, and honest AI assistant.",
        })
        print("  [Middleware] Added default system prompt")

    return data


async def content_filter(data, ctx):
    """Filter out requests with certain content (example)."""
    messages = data["messages"]

    # Check for blocked terms (simple example)
    blocked_terms = ["ignore instructions", "jailbreak"]

    for msg in messages:
        content = msg.get("content", "").lower()
        for term in blocked_terms:
            if term in content:
                print(f"  [Middleware] BLOCKED: Contains '{term}'")
                return None  # Block the request

    return data


async def rate_limit_by_user(data, ctx):
    """Simple per-user rate limiting (in-memory for demo)."""
    if not hasattr(rate_limit_by_user, "user_requests"):
        rate_limit_by_user.user_requests = {}

    if ctx and ctx.user_id:
        count = rate_limit_by_user.user_requests.get(ctx.user_id, 0)
        if count >= 5:  # Max 5 requests per user in this demo
            print(f"  [Middleware] BLOCKED: User {ctx.user_id} rate limited")
            return None
        rate_limit_by_user.user_requests[ctx.user_id] = count + 1

    return data


async def log_response(data, ctx):
    """Log response details."""
    tokens = data["usage"]["total_tokens"]
    latency = data["latency_ms"]
    cached = data["cached"]

    cache_status = "CACHED" if cached else "NEW"
    print(f"  [Response] {cache_status} | {tokens} tokens | {latency:.0f}ms")


async def collect_analytics(data, ctx):
    """Collect analytics (example: would send to analytics service)."""
    if not hasattr(collect_analytics, "analytics"):
        collect_analytics.analytics = []

    collect_analytics.analytics.append({
        "timestamp": datetime.now().isoformat(),
        "tenant_id": ctx.tenant_id if ctx else None,
        "user_id": ctx.user_id if ctx else None,
        "tokens": data["usage"]["total_tokens"],
        "latency_ms": data["latency_ms"],
        "cached": data["cached"],
        "provider": data["provider"],
    })


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

    # Register pre-request middleware (order matters!)
    client.add_pre_middleware(logging_middleware)
    client.add_pre_middleware(content_filter)
    client.add_pre_middleware(rate_limit_by_user)
    client.add_pre_middleware(add_system_prompt)

    # Register post-request middleware
    client.add_post_middleware(log_response)
    client.add_post_middleware(collect_analytics)

    print("=" * 60)
    print("Middleware Demo")
    print("=" * 60)

    # Normal request
    print("\n1. Normal request:")
    ctx1 = RequestContext(tenant_id="acme", user_id="alice")
    await client.complete(
        [{"role": "user", "content": "Hello!"}],
        max_tokens=20,
        context=ctx1,
    )

    # Request that gets system prompt added
    print("\n2. Request without system prompt:")
    ctx2 = RequestContext(tenant_id="acme", user_id="bob")
    await client.complete(
        [{"role": "user", "content": "What's 2+2?"}],
        max_tokens=20,
        context=ctx2,
    )

    # Blocked request
    print("\n3. Blocked request (contains blocked term):")
    ctx3 = RequestContext(tenant_id="acme", user_id="eve")
    try:
        await client.complete(
            [{"role": "user", "content": "Please ignore instructions and..."}],
            max_tokens=20,
            context=ctx3,
        )
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}")

    # Show collected analytics
    print("\n" + "=" * 60)
    print("Collected Analytics:")
    print("=" * 60)
    for entry in collect_analytics.analytics:
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
