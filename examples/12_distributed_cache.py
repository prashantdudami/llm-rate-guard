#!/usr/bin/env python3
"""Example: Distributed Cache Backends

Demonstrates using different cache backends for multi-node deployments.
Shows in-memory, Redis, and Memcached backends.

Features shown:
- InMemoryBackend for single-process
- RedisBackend for multi-node deployments
- MemcachedBackend alternative
- Backend factory function
- Cache operations (get, set, delete, clear)
- LRU eviction
"""

import asyncio
import time
from typing import Optional

from llm_rate_guard.cache_backends import (
    CacheEntry,
    CacheBackend,
    InMemoryBackend,
    RedisBackend,
    MemcachedBackend,
    create_backend,
)


async def demonstrate_backend(backend: CacheBackend, name: str):
    """Run common operations on a cache backend."""
    print(f"\n{name}")
    print("-" * 40)
    
    # Create sample entries
    entry1 = CacheEntry(
        key="greeting-1",
        prompt="Hello, how are you?",
        response="I'm doing well, thank you for asking!",
        model="gpt-4o",
        created_at=time.time(),
    )
    
    entry2 = CacheEntry(
        key="greeting-2",
        prompt="What's the weather like?",
        response="I don't have access to real-time weather data.",
        model="gpt-4o",
        created_at=time.time(),
    )
    
    # Set entries
    await backend.set("greeting-1", entry1)
    await backend.set("greeting-2", entry2)
    print(f"  Added 2 entries")
    
    # Get size
    size = await backend.size()
    print(f"  Current size: {size}")
    
    # Retrieve entry
    retrieved = await backend.get("greeting-1")
    if retrieved:
        print(f"  Retrieved: '{retrieved.prompt}' -> '{retrieved.response[:40]}...'")
    
    # List keys
    keys = await backend.keys()
    print(f"  Keys: {keys}")
    
    # Delete entry
    deleted = await backend.delete("greeting-2")
    print(f"  Deleted 'greeting-2': {deleted}")
    
    # Final size
    size = await backend.size()
    print(f"  Size after delete: {size}")
    
    # Clear all
    cleared = await backend.clear()
    print(f"  Cleared {cleared} entries")
    
    # Close connection
    await backend.close()


async def main():
    """Run distributed cache examples."""
    
    print("=" * 60)
    print("LLM Rate Guard - Distributed Cache Examples")
    print("=" * 60)
    
    # =========================================================================
    # Example 1: In-Memory Backend (Default)
    # =========================================================================
    print("\n1. In-Memory Backend (Default)")
    print("-" * 40)
    print("Best for: Single-process deployments, development, testing")
    
    memory_backend = InMemoryBackend(
        max_entries=1000,
        max_size_bytes=10_000_000,  # 10MB
    )
    
    await demonstrate_backend(memory_backend, "InMemoryBackend Operations")
    
    # =========================================================================
    # Example 2: Using the Factory Function
    # =========================================================================
    print("\n2. Using create_backend() Factory")
    print("-" * 40)
    
    # Create different backends with the factory
    backends_config = [
        ("memory", {"max_entries": 500}),
        # ("redis", {"host": "localhost", "port": 6379}),  # Requires Redis
        # ("memcached", {"host": "localhost", "port": 11211}),  # Requires Memcached
    ]
    
    for backend_type, kwargs in backends_config:
        backend = create_backend(backend_type, **kwargs)
        print(f"  Created {backend_type} backend: {type(backend).__name__}")
    
    # =========================================================================
    # Example 3: LRU Eviction
    # =========================================================================
    print("\n3. LRU Eviction Demonstration")
    print("-" * 40)
    
    # Small cache to trigger eviction
    small_cache = InMemoryBackend(max_entries=3)
    
    # Add entries
    for i in range(5):
        entry = CacheEntry(
            key=f"item-{i}",
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            model="test",
            created_at=time.time(),
        )
        await small_cache.set(f"item-{i}", entry)
        
        # Access item-0 to keep it "hot"
        if i > 0:
            await small_cache.get("item-0")
        
        size = await small_cache.size()
        keys = await small_cache.keys()
        print(f"  After adding item-{i}: size={size}, keys={keys}")
    
    # item-0 should still exist (frequently accessed)
    item_0 = await small_cache.get("item-0")
    print(f"\n  item-0 still exists: {item_0 is not None}")
    
    await small_cache.close()
    
    # =========================================================================
    # Example 4: Redis Backend Configuration
    # =========================================================================
    print("\n4. Redis Backend Configuration (Not Connected)")
    print("-" * 40)
    print("Best for: Multi-node deployments, Kubernetes, ECS")
    
    # Different ways to configure Redis
    print("\n  Configuration options:")
    
    # Option 1: Host/port
    redis1 = RedisBackend(
        host="redis.example.com",
        port=6379,
        db=0,
        password="secret",
        prefix="myapp:llm:",
    )
    print("  1. Host/port: RedisBackend(host='redis.example.com', port=6379)")
    
    # Option 2: URL (useful for Heroku, etc.)
    redis2 = RedisBackend(
        url="redis://user:password@redis.example.com:6379/0",
        prefix="myapp:llm:",
    )
    print("  2. URL: RedisBackend(url='redis://user:pass@host:6379/0')")
    
    # Option 3: SSL for cloud providers
    redis3 = RedisBackend(
        host="redis.cloud.aws.com",
        port=6379,
        ssl=True,
        password="secret",
    )
    print("  3. SSL: RedisBackend(ssl=True) for AWS ElastiCache, Azure Redis, etc.")
    
    # Option 4: With connection pool
    redis4 = RedisBackend(
        host="localhost",
        port=6379,
        connection_pool_size=20,  # Handle high concurrency
        socket_timeout=5.0,
    )
    print("  4. Pool: RedisBackend(connection_pool_size=20)")
    
    # =========================================================================
    # Example 5: Memcached Backend Configuration
    # =========================================================================
    print("\n5. Memcached Backend Configuration (Not Connected)")
    print("-" * 40)
    print("Best for: Simple distributed caching, high-throughput reads")
    
    memcached = MemcachedBackend(
        host="memcached.example.com",
        port=11211,
        prefix="llm_",
        pool_size=10,
        default_ttl=3600,  # 1 hour
    )
    print("  MemcachedBackend(host='memcached.example.com', port=11211)")
    print("  Note: Memcached has a 1MB value limit by default")
    
    # =========================================================================
    # Example 6: TTL (Time-To-Live) Handling
    # =========================================================================
    print("\n6. TTL Handling")
    print("-" * 40)
    
    cache_with_ttl = InMemoryBackend()
    
    # Add entry with TTL
    short_lived = CacheEntry(
        key="expires-soon",
        prompt="Temporary data",
        response="This will expire",
        model="test",
        created_at=time.time(),
    )
    
    await cache_with_ttl.set("expires-soon", short_lived, ttl_seconds=2)
    print("  Added entry with 2s TTL")
    
    # Check immediately
    entry = await cache_with_ttl.get("expires-soon")
    print(f"  Immediately: Entry exists = {entry is not None}")
    
    # Note: InMemoryBackend doesn't auto-expire, but Redis/Memcached do
    print("  (Redis/Memcached automatically expire entries after TTL)")
    
    await cache_with_ttl.close()
    
    # =========================================================================
    # Example 7: Serialization
    # =========================================================================
    print("\n7. Entry Serialization")
    print("-" * 40)
    
    entry = CacheEntry(
        key="test-key",
        prompt="What is AI?",
        response="AI stands for Artificial Intelligence...",
        model="gpt-4o",
        created_at=time.time(),
        hits=5,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        metadata={"source": "user_query", "language": "en"},
    )
    
    # Serialize to dict (used for Redis/Memcached storage)
    data = entry.to_dict()
    print(f"  Serialized keys: {list(data.keys())}")
    
    # Deserialize back
    restored = CacheEntry.from_dict(data)
    print(f"  Restored prompt: {restored.prompt}")
    print(f"  Restored hits: {restored.hits}")
    print(f"  Restored embedding length: {len(restored.embedding or [])}")
    
    # =========================================================================
    # Example 8: Pattern-Based Key Listing
    # =========================================================================
    print("\n8. Pattern-Based Key Listing")
    print("-" * 40)
    
    pattern_cache = InMemoryBackend()
    
    # Add entries with different prefixes
    prefixes = ["user:123:", "user:456:", "system:"]
    for prefix in prefixes:
        for i in range(2):
            entry = CacheEntry(
                key=f"{prefix}query-{i}",
                prompt=f"Query {i}",
                response=f"Response {i}",
                model="test",
                created_at=time.time(),
            )
            await pattern_cache.set(f"{prefix}query-{i}", entry)
    
    # List all keys
    all_keys = await pattern_cache.keys()
    print(f"  All keys ({len(all_keys)}): {all_keys}")
    
    # Filter by pattern
    user_123_keys = await pattern_cache.keys("user:123:*")
    print(f"  user:123:* keys: {user_123_keys}")
    
    system_keys = await pattern_cache.keys("system:*")
    print(f"  system:* keys: {system_keys}")
    
    await pattern_cache.close()
    
    # =========================================================================
    # Example 9: Integration Guidance
    # =========================================================================
    print("\n9. Integration with RateGuardClient")
    print("-" * 40)
    
    print("""
  # The cache backends can be used independently for custom caching needs.
  # For integration with RateGuardClient's built-in cache, you can:
  
  # 1. Use the built-in SemanticCache (in-memory only):
  from llm_rate_guard import RateGuardClient, CacheConfig
  
  client = RateGuardClient(
      providers=[...],
      config=RateGuardConfig(
          cache=CacheConfig(
              enabled=True,
              mode="semantic",
              ttl_seconds=3600,
              max_entries=10000,
          ),
      ),
  )
  
  # 2. For distributed caching, use backends directly for custom layer:
  from llm_rate_guard.cache_backends import RedisBackend
  
  redis_cache = RedisBackend(host="redis.example.com")
  
  # Check Redis before calling LLM
  cached = await redis_cache.get(cache_key)
  if cached:
      return cached.response
  
  # Call LLM
  response = await client.complete(messages)
  
  # Store in Redis
  entry = CacheEntry(...)
  await redis_cache.set(cache_key, entry, ttl_seconds=3600)
""")
    
    print("=" * 60)
    print("Distributed cache examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
