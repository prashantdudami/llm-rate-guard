#!/usr/bin/env python3
"""Example: Batch Processing

Demonstrates efficient batch processing of multiple prompts with controlled
concurrency, error handling, and progress tracking.

Features shown:
- complete_batch() for parallel processing
- Concurrency control with max_concurrency
- Error handling with return_exceptions
- Progress tracking
- Order preservation
- Performance comparison
"""

import asyncio
import time
from typing import Union

from llm_rate_guard import (
    RateGuardClient,
    ProviderConfig,
    ProviderType,
    RateGuardError,
)
from llm_rate_guard.providers.base import CompletionResponse


async def main():
    """Run batch processing examples."""
    
    # Create client with multiple providers for better throughput
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4o-mini",
                weight=1.0,
            ),
        ],
        cache_enabled=True,
    )
    
    print("=" * 60)
    print("LLM Rate Guard - Batch Processing Examples")
    print("=" * 60)
    
    # =========================================================================
    # Example 1: Basic Batch Processing
    # =========================================================================
    print("\n1. Basic Batch Processing")
    print("-" * 40)
    
    # Prepare a batch of prompts
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]
    
    prompts = [[{"role": "user", "content": q}] for q in questions]
    
    start = time.time()
    responses = await client.complete_batch(
        prompts,
        max_tokens=50,
        temperature=0.0,
    )
    elapsed = time.time() - start
    
    print(f"Processed {len(responses)} prompts in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(responses)*1000:.0f}ms per prompt")
    print("\nResults:")
    for q, r in zip(questions, responses):
        print(f"  Q: {q}")
        print(f"  A: {r.content.strip()[:60]}...")
        print()
    
    # =========================================================================
    # Example 2: Concurrency Control
    # =========================================================================
    print("\n2. Concurrency Control Comparison")
    print("-" * 40)
    
    # Generate 20 math problems
    math_prompts = [
        [{"role": "user", "content": f"What is {i} * {i+1}? Answer with just the number."}]
        for i in range(1, 21)
    ]
    
    # Process with low concurrency
    start = time.time()
    await client.complete_batch(math_prompts, max_concurrency=2, max_tokens=10)
    low_time = time.time() - start
    
    # Process with high concurrency
    start = time.time()
    await client.complete_batch(math_prompts, max_concurrency=10, max_tokens=10)
    high_time = time.time() - start
    
    print(f"With max_concurrency=2:  {low_time:.2f}s")
    print(f"With max_concurrency=10: {high_time:.2f}s")
    print(f"Speedup: {low_time/high_time:.1f}x faster")
    
    # =========================================================================
    # Example 3: Error Handling with return_exceptions
    # =========================================================================
    print("\n3. Error Handling with return_exceptions")
    print("-" * 40)
    
    # Mix of valid and potentially problematic prompts
    mixed_prompts = [
        [{"role": "user", "content": "Say hello"}],
        [{"role": "user", "content": "Say goodbye"}],
        [{"role": "user", "content": "Count to 3"}],
    ]
    
    # With return_exceptions=True, errors are returned instead of raised
    responses = await client.complete_batch(
        mixed_prompts,
        max_tokens=20,
        return_exceptions=True,
    )
    
    success_count = 0
    error_count = 0
    
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            print(f"  Prompt {i+1}: ERROR - {type(resp).__name__}: {resp}")
            error_count += 1
        else:
            print(f"  Prompt {i+1}: SUCCESS - {resp.content.strip()[:40]}")
            success_count += 1
    
    print(f"\nSummary: {success_count} succeeded, {error_count} failed")
    
    # =========================================================================
    # Example 4: Order Preservation
    # =========================================================================
    print("\n4. Order Preservation")
    print("-" * 40)
    
    # Prompts with identifiable outputs
    numbered_prompts = [
        [{"role": "user", "content": f"Reply with only the number {i}"}]
        for i in range(1, 11)
    ]
    
    responses = await client.complete_batch(
        numbered_prompts,
        max_concurrency=10,  # High concurrency might reorder internally
        max_tokens=5,
    )
    
    print("Checking order preservation:")
    all_correct = True
    for i, resp in enumerate(responses):
        expected = i + 1
        content = resp.content.strip()
        # Check if the expected number is in the response
        correct = str(expected) in content
        status = "OK" if correct else "MISMATCH"
        if not correct:
            all_correct = False
        print(f"  Position {i+1}: Expected '{expected}', Got '{content}' - {status}")
    
    print(f"\nOrder preserved: {'Yes' if all_correct else 'No'}")
    
    # =========================================================================
    # Example 5: Large Batch with Progress Tracking
    # =========================================================================
    print("\n5. Large Batch with Progress Tracking")
    print("-" * 40)
    
    # Create 50 prompts
    large_batch = [
        [{"role": "user", "content": f"What is {i}+{i}? Just the number."}]
        for i in range(1, 51)
    ]
    
    # Custom progress tracking using a wrapper
    completed = 0
    total = len(large_batch)
    
    async def process_with_progress():
        nonlocal completed
        
        # Process in smaller chunks to show progress
        chunk_size = 10
        all_results = []
        
        for i in range(0, total, chunk_size):
            chunk = large_batch[i:i+chunk_size]
            results = await client.complete_batch(
                chunk,
                max_concurrency=5,
                max_tokens=10,
            )
            all_results.extend(results)
            completed += len(results)
            progress = (completed / total) * 100
            print(f"  Progress: {completed}/{total} ({progress:.0f}%)")
        
        return all_results
    
    start = time.time()
    results = await process_with_progress()
    elapsed = time.time() - start
    
    print(f"\nCompleted {len(results)} prompts in {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} prompts/second")
    
    # =========================================================================
    # Example 6: Batch Processing with Caching Benefits
    # =========================================================================
    print("\n6. Batch Processing with Caching")
    print("-" * 40)
    
    # Same prompts as before (some should be cached)
    repeat_prompts = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is the capital of Germany?"}],
        [{"role": "user", "content": "What is the capital of France?"}],  # Repeat
        [{"role": "user", "content": "What is the capital of Italy?"}],
        [{"role": "user", "content": "What is the capital of France?"}],  # Repeat again
    ]
    
    start = time.time()
    responses = await client.complete_batch(repeat_prompts, max_tokens=50)
    elapsed = time.time() - start
    
    print(f"Processed {len(responses)} prompts in {elapsed:.2f}s")
    
    # Check cache stats
    health = await client.health_check()
    print(f"Cache hit rate: {health['cache']['hit_rate_pct']:.1f}%")
    
    # =========================================================================
    # Example 7: Different Parameters per Batch
    # =========================================================================
    print("\n7. Processing with Different Parameters")
    print("-" * 40)
    
    # Creative batch
    creative_prompts = [
        [{"role": "user", "content": "Write a haiku about coding"}],
        [{"role": "user", "content": "Write a haiku about coffee"}],
    ]
    
    creative_responses = await client.complete_batch(
        creative_prompts,
        max_tokens=100,
        temperature=0.9,  # High creativity
    )
    
    print("Creative outputs (temperature=0.9):")
    for resp in creative_responses:
        print(f"  {resp.content.strip()[:50]}...")
    
    # Factual batch
    factual_prompts = [
        [{"role": "user", "content": "What year was Python created?"}],
        [{"role": "user", "content": "Who created Python?"}],
    ]
    
    factual_responses = await client.complete_batch(
        factual_prompts,
        max_tokens=50,
        temperature=0.0,  # Deterministic
    )
    
    print("\nFactual outputs (temperature=0.0):")
    for resp in factual_responses:
        print(f"  {resp.content.strip()[:50]}...")
    
    print("\n" + "=" * 60)
    print("Batch processing examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
