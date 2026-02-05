#!/usr/bin/env python3
"""Example: Synchronous API Usage

Demonstrates using llm-rate-guard from synchronous code without async/await.
Perfect for scripts, notebooks, Django/Flask views, or quick prototypes.

Features shown:
- complete_sync() for synchronous completions
- embed_sync() for synchronous embeddings
- health_check_sync() for synchronous health checks
- complete_batch_sync() for synchronous batch processing
"""

from llm_rate_guard import (
    RateGuardClient,
    ProviderConfig,
    ProviderType,
    CacheConfig,
)


def main():
    """Run synchronous API examples."""
    
    # Create client (same as async usage)
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type=ProviderType.OPENAI,
                model="gpt-4o-mini",
                # api_key="sk-...",  # Or set OPENAI_API_KEY env var
            ),
        ],
        cache_enabled=True,
    )
    
    print("=" * 60)
    print("LLM Rate Guard - Synchronous API Examples")
    print("=" * 60)
    
    # =========================================================================
    # Example 1: Simple Synchronous Completion
    # =========================================================================
    print("\n1. Simple Synchronous Completion")
    print("-" * 40)
    
    response = client.complete_sync(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=50,
        temperature=0.0,
    )
    
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    
    # =========================================================================
    # Example 2: Synchronous Completion with Caching
    # =========================================================================
    print("\n2. Synchronous Completion with Caching")
    print("-" * 40)
    
    # First call - cache miss
    import time
    start = time.time()
    response1 = client.complete_sync(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=50,
    )
    first_time = time.time() - start
    print(f"First call: {response1.content[:50]}... ({first_time:.3f}s)")
    
    # Second call - cache hit (much faster)
    start = time.time()
    response2 = client.complete_sync(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=50,
    )
    second_time = time.time() - start
    print(f"Second call (cached): {response2.content[:50]}... ({second_time:.3f}s)")
    print(f"Speed improvement: {first_time/second_time:.1f}x faster")
    
    # =========================================================================
    # Example 3: Synchronous Health Check
    # =========================================================================
    print("\n3. Synchronous Health Check")
    print("-" * 40)
    
    health = client.health_check_sync()
    
    print(f"Status: {health['status']}")
    print(f"Healthy: {health['healthy']}")
    print(f"Providers: {health['providers']['healthy']}/{health['providers']['total']} healthy")
    print(f"Cache: {health['cache']['entries']} entries, {health['cache']['hit_rate_pct']:.1f}% hit rate")
    
    # =========================================================================
    # Example 4: Synchronous Embeddings
    # =========================================================================
    print("\n4. Synchronous Embeddings")
    print("-" * 40)
    
    embedding = client.embed_sync(
        text="Machine learning is fascinating.",
        # model="text-embedding-3-small",  # Optional specific model
    )
    
    print(f"Embedding dimensions: {len(embedding.embedding)}")
    print(f"Model: {embedding.model}")
    print(f"First 5 values: {embedding.embedding[:5]}")
    
    # =========================================================================
    # Example 5: Synchronous Batch Processing
    # =========================================================================
    print("\n5. Synchronous Batch Processing")
    print("-" * 40)
    
    prompts = [
        [{"role": "user", "content": "What is 1+1?"}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 3+3?"}],
        [{"role": "user", "content": "What is 4+4?"}],
        [{"role": "user", "content": "What is 5+5?"}],
    ]
    
    start = time.time()
    responses = client.complete_batch_sync(
        prompts,
        max_concurrency=5,  # Process up to 5 in parallel
        max_tokens=20,
    )
    elapsed = time.time() - start
    
    print(f"Processed {len(responses)} prompts in {elapsed:.2f}s")
    for prompt, resp in zip(prompts, responses):
        question = prompt[0]["content"]
        answer = resp.content.strip()
        print(f"  {question} -> {answer}")
    
    # =========================================================================
    # Example 6: Cost Estimation (No API Call)
    # =========================================================================
    print("\n6. Cost Estimation")
    print("-" * 40)
    
    estimate = client.estimate_cost(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a 500 word essay about AI."},
        ],
        max_tokens=1000,
    )
    
    print(f"Estimated input tokens: {estimate['estimated_input_tokens']}")
    print(f"Estimated output tokens: {estimate['estimated_output_tokens']}")
    print(f"Estimated cost: ${estimate['total_usd']:.6f}")
    
    # =========================================================================
    # Example 7: Using in a Flask-like Context (Simulation)
    # =========================================================================
    print("\n7. Flask-like Request Handler (Simulated)")
    print("-" * 40)
    
    def handle_chat_request(user_message: str) -> dict:
        """Simulates a Flask/Django view handler."""
        try:
            response = client.complete_sync(
                messages=[{"role": "user", "content": user_message}],
                max_tokens=100,
            )
            return {
                "success": True,
                "response": response.content,
                "tokens_used": response.usage.total_tokens,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    # Simulate handling a request
    result = handle_chat_request("Hello, how are you?")
    print(f"Result: {result}")
    
    print("\n" + "=" * 60)
    print("Sync API examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
