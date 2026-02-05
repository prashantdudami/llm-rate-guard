"""Tests for cache module."""

import asyncio
import pytest
import time

from llm_rate_guard.cache import SemanticCache, CacheConfig, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_is_expired_with_ttl(self):
        """Entry expires after TTL."""
        entry = CacheEntry(
            key="test",
            prompt="hello",
            response="world",
            model="gpt-4",
            created_at=time.time() - 100,  # Created 100 seconds ago
        )
        
        assert entry.is_expired(ttl_seconds=50) is True
        assert entry.is_expired(ttl_seconds=200) is False

    def test_is_expired_no_ttl(self):
        """Entry never expires without TTL."""
        entry = CacheEntry(
            key="test",
            prompt="hello",
            response="world",
            model="gpt-4",
            created_at=time.time() - 10000,
        )
        
        assert entry.is_expired(ttl_seconds=None) is False


class TestSemanticCache:
    """Tests for SemanticCache class."""

    @pytest.mark.asyncio
    async def test_set_and_get_exact_match(self):
        """Basic set and get with exact matching."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("hello", "world", model="gpt-4")
        result = await cache.get("hello", model="gpt-4")
        
        assert result == "world"

    @pytest.mark.asyncio
    async def test_get_miss(self):
        """Get returns None for missing entry."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        result = await cache.get("nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_model_isolation(self):
        """Different models have separate cache entries."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("hello", "world1", model="gpt-4")
        await cache.set("hello", "world2", model="claude")
        
        assert await cache.get("hello", model="gpt-4") == "world1"
        assert await cache.get("hello", model="claude") == "world2"

    @pytest.mark.asyncio
    async def test_disabled_cache(self):
        """Disabled cache returns None and doesn't store."""
        cache = SemanticCache(CacheConfig(enabled=False))
        
        await cache.set("hello", "world")
        result = await cache.get("hello")
        
        assert result is None
        assert cache.stats.entries == 0

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Expired entries are not returned."""
        # Use minimum valid TTL and manually set entry creation time
        cache = SemanticCache(CacheConfig(ttl_seconds=60))
        
        await cache.set("hello", "world")
        
        # Entry should be fresh
        assert await cache.get("hello") == "world"
        
        # Manually expire the entry by modifying created_at
        key = cache._hash_prompt("hello", "")
        cache._cache[key].created_at = time.time() - 100  # Created 100s ago
        
        # Entry should be expired
        assert await cache.get("hello") is None

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Invalidate removes specific entry."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("hello", "world")
        assert await cache.get("hello") == "world"
        
        removed = await cache.invalidate("hello")
        
        assert removed is True
        assert await cache.get("hello") is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Clear removes all entries."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        count = await cache.clear()
        
        assert count == 3
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """LRU eviction when max entries reached."""
        cache = SemanticCache(CacheConfig(max_entries=100))
        
        # Fill the cache to capacity
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")
        
        # Access key0 to make it recently used
        await cache.get("key0")
        
        # Add more entries to trigger eviction
        for i in range(100, 120):
            await cache.set(f"key{i}", f"value{i}")
        
        # Should have evicted some entries
        assert cache.stats.evictions > 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Stats are properly tracked."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("hello", "world")
        await cache.get("hello")  # Hit
        await cache.get("hello")  # Hit
        await cache.get("missing")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["hit_rate_pct"] == pytest.approx(66.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """cleanup_expired removes expired entries."""
        cache = SemanticCache(CacheConfig(ttl_seconds=60))
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Manually expire entries
        for key in list(cache._cache.keys()):
            cache._cache[key].created_at = time.time() - 100
        
        # Add fresh entry
        await cache.set("key3", "value3")
        
        removed = await cache.cleanup_expired()
        
        assert removed == 2
        assert cache.stats.entries == 1

    @pytest.mark.asyncio
    async def test_hit_counter(self):
        """Hit counter increments on cache hits."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        await cache.set("hello", "world")
        
        # Multiple hits
        await cache.get("hello")
        await cache.get("hello")
        await cache.get("hello")
        
        # Check internal entry hit count
        key = cache._hash_prompt("hello", "")
        entry = cache._cache.get(key)
        
        assert entry is not None
        assert entry.hits == 3


class TestSemanticCacheWithEmbeddings:
    """Tests for semantic cache with embedding-based matching."""

    @pytest.mark.asyncio
    async def test_semantic_mode_fallback_without_fn(self):
        """Semantic mode falls back to exact without embedding fn."""
        cache = SemanticCache(
            CacheConfig(mode="semantic"),
            embedding_fn=None,  # No embedding function
        )
        
        # Should fall back to exact mode
        assert cache.config.mode == "exact"

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self):
        """Cosine similarity is calculated correctly."""
        cache = SemanticCache(CacheConfig(mode="exact"))
        
        # Identical vectors
        sim = cache._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert sim == pytest.approx(1.0)
        
        # Orthogonal vectors
        sim = cache._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert sim == pytest.approx(0.0)
        
        # Opposite vectors
        sim = cache._cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert sim == pytest.approx(-1.0)
