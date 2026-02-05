"""Advanced tests for cache module - size limits and byte tracking."""

import pytest

from llm_rate_guard.cache import SemanticCache, CacheEntry
from llm_rate_guard.config import CacheConfig


class TestCacheSizeLimits:
    """Tests for cache size limit features."""

    @pytest.mark.asyncio
    async def test_entry_size_tracking(self):
        """Cache tracks entry sizes in bytes."""
        cache = SemanticCache(CacheConfig())

        await cache.set("prompt1", "response1", model="test")

        stats = cache.get_stats()
        assert stats["current_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_large_entry_rejected(self):
        """Entries exceeding max_entry_size_bytes are rejected."""
        config = CacheConfig(max_entry_size_bytes=1000)  # 1KB limit
        cache = SemanticCache(config)

        large_response = "x" * 2000  # ~2KB
        result = await cache.set("prompt", large_response, model="test")

        assert result is False
        stats = cache.get_stats()
        assert stats["rejected_size"] == 1
        assert stats["entries"] == 0

    @pytest.mark.asyncio
    async def test_size_eviction(self):
        """Cache evicts entries when max_size_bytes is reached."""
        config = CacheConfig(
            max_entries=1000,  # High entry limit
            max_size_bytes=1_000_000,  # 1MB limit
            max_entry_size_bytes=100_000,  # 100KB per entry
        )
        cache = SemanticCache(config)

        # Add entries until we exceed size limit
        for i in range(20):
            response = "x" * 80_000  # ~80KB each
            await cache.set(f"prompt{i}", response, model="test")

        stats = cache.get_stats()

        # Should have evicted some entries
        assert stats["evictions"] > 0
        assert stats["current_size_bytes"] <= config.max_size_bytes

    @pytest.mark.asyncio
    async def test_size_utilization_stat(self):
        """Cache reports size utilization percentage."""
        config = CacheConfig(max_size_bytes=10_000_000)  # 10MB
        cache = SemanticCache(config)

        await cache.set("prompt", "response", model="test")

        stats = cache.get_stats()
        assert "size_utilization_pct" in stats
        assert stats["size_utilization_pct"] > 0
        assert stats["size_utilization_pct"] < 100

    @pytest.mark.asyncio
    async def test_clear_resets_size(self):
        """Clear resets size tracking."""
        cache = SemanticCache(CacheConfig())

        await cache.set("prompt", "response", model="test")
        await cache.clear()

        stats = cache.get_stats()
        assert stats["current_size_bytes"] == 0

    @pytest.mark.asyncio
    async def test_update_entry_updates_size(self):
        """Updating an entry updates size tracking correctly."""
        cache = SemanticCache(CacheConfig())

        await cache.set("prompt", "short", model="test")
        size1 = cache.get_stats()["current_size_bytes"]

        await cache.set("prompt", "this is a much longer response", model="test")
        size2 = cache.get_stats()["current_size_bytes"]

        assert size2 > size1  # Size should increase
        assert cache.get_stats()["entries"] == 1  # Still one entry


class TestCacheConfigValidation:
    """Tests for cache configuration validation."""

    def test_max_size_bytes_minimum(self):
        """max_size_bytes has minimum value."""
        with pytest.raises(Exception):
            CacheConfig(max_size_bytes=100)  # Too small

    def test_max_entry_size_bytes_minimum(self):
        """max_entry_size_bytes has minimum value."""
        with pytest.raises(Exception):
            CacheConfig(max_entry_size_bytes=100)  # Too small

    def test_default_values(self):
        """Default size limits are reasonable."""
        config = CacheConfig()

        assert config.max_size_bytes == 100_000_000  # 100MB
        assert config.max_entry_size_bytes == 1_000_000  # 1MB
