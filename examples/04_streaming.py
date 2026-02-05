#!/usr/bin/env python3
"""Streaming response example for LLM Rate Guard.

This example shows how to stream responses in real-time for
a better user experience with long responses.
"""

import asyncio
import sys

from llm_rate_guard import ProviderConfig, RateGuardClient


async def main():
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="openai",
                model="gpt-4-turbo",
            ),
        ],
    )

    print("Streaming a story...\n")
    print("-" * 50)

    total_tokens = 0

    # Stream the response
    async for chunk in client.stream(
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Write a short story about a robot learning to paint."},
        ],
        max_tokens=500,
        temperature=0.8,
    ):
        # Print each chunk as it arrives
        print(chunk.content, end="", flush=True)

        # Check if this is the final chunk
        if chunk.done and chunk.usage:
            total_tokens = chunk.usage.total_tokens

    print("\n" + "-" * 50)
    print(f"\nTotal tokens used: {total_tokens}")

    # Multiple streams concurrently
    print("\n\nStreaming 3 haikus concurrently...")
    print("=" * 50)

    async def stream_haiku(topic: str, color: str):
        """Stream a haiku about a topic."""
        result = []
        async for chunk in client.stream(
            messages=[{"role": "user", "content": f"Write a haiku about {topic}"}],
            max_tokens=50,
        ):
            result.append(chunk.content)
        return f"\n{color}[{topic}]{color}\n" + "".join(result)

    # Note: True concurrent streaming would need more complex handling
    # This runs them sequentially for simplicity
    topics = ["mountains", "ocean", "sunset"]
    for topic in topics:
        result = []
        print(f"\n[{topic}]")
        async for chunk in client.stream(
            messages=[{"role": "user", "content": f"Write a haiku about {topic}"}],
            max_tokens=50,
        ):
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
