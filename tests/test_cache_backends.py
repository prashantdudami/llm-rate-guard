"""Tests for cache backends."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_rate_guard.cache_backends import (
    CacheEntry,
    CacheBackend,
    CacheBackendProtocol,
    InMemoryBackend,
    RedisBackend,
    MemcachedBackend,
    create_backend,
)


@pytest.fixture
def sample_entry():
    """Create a sample cache entry."""
    return CacheEntry(
        key="test-key",
        prompt="Hello, how are you?",
        response="I'm doing well, thank you!",
        model="test-model",
        created_at=1000000.0,
        hits=5,
        embedding=[0.1, 0.2, 0.3],
        metadata={"source": "test"},
    )


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self, sample_entry):
        """Test basic entry creation."""
        assert sample_entry.key == "test-key"
        assert sample_entry.prompt == "Hello, how are you?"
        assert sample_entry.response == "I'm doing well, thank you!"
        assert sample_entry.model == "test-model"
        assert sample_entry.hits == 5

    def test_is_expired_no_ttl(self, sample_entry):
        """Test is_expired with no TTL."""
        assert sample_entry.is_expired(None) is False

    def test_is_expired_with_ttl(self, sample_entry):
        """Test is_expired with TTL."""
        # Entry created at 1000000.0, check if expired after TTL
        import time
        current_time = time.time()
        # Entry was created way before current time
        assert sample_entry.is_expired(60) is True

    def test_to_dict(self, sample_entry):
        """Test serialization to dict."""
        data = sample_entry.to_dict()
        assert data["key"] == "test-key"
        assert data["prompt"] == "Hello, how are you?"
        assert data["response"] == "I'm doing well, thank you!"
        assert data["model"] == "test-model"
        assert data["hits"] == 5
        assert data["embedding"] == [0.1, 0.2, 0.3]

    def test_from_dict(self, sample_entry):
        """Test deserialization from dict."""
        data = sample_entry.to_dict()
        restored = CacheEntry.from_dict(data)
        assert restored.key == sample_entry.key
        assert restored.prompt == sample_entry.prompt
        assert restored.response == sample_entry.response
        assert restored.hits == sample_entry.hits


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, sample_entry):
        """Test basic set and get operations."""
        backend = InMemoryBackend()

        success = await backend.set("test-key", sample_entry)
        assert success is True

        retrieved = await backend.get("test-key")
        assert retrieved is not None
        assert retrieved.prompt == sample_entry.prompt
        assert retrieved.response == sample_entry.response

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting a nonexistent key."""
        backend = InMemoryBackend()

        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sample_entry):
        """Test delete operation."""
        backend = InMemoryBackend()

        await backend.set("test-key", sample_entry)
        assert await backend.get("test-key") is not None

        deleted = await backend.delete("test-key")
        assert deleted is True

        assert await backend.get("test-key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        backend = InMemoryBackend()

        deleted = await backend.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self, sample_entry):
        """Test clearing all entries."""
        backend = InMemoryBackend()

        for i in range(5):
            entry = CacheEntry(
                key=f"key-{i}",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"key-{i}", entry)

        assert await backend.size() == 5

        cleared = await backend.clear()
        assert cleared == 5
        assert await backend.size() == 0

    @pytest.mark.asyncio
    async def test_keys_all(self, sample_entry):
        """Test listing all keys."""
        backend = InMemoryBackend()

        for i in range(3):
            entry = CacheEntry(
                key=f"key-{i}",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"key-{i}", entry)

        keys = await backend.keys()
        assert len(keys) == 3
        assert set(keys) == {"key-0", "key-1", "key-2"}

    @pytest.mark.asyncio
    async def test_keys_pattern(self):
        """Test listing keys with pattern."""
        backend = InMemoryBackend()

        for prefix in ["user", "admin"]:
            for i in range(2):
                entry = CacheEntry(
                    key=f"{prefix}-{i}",
                    prompt=f"Prompt {prefix} {i}",
                    response=f"Response {prefix} {i}",
                    model="test-model",
                    created_at=1000000.0,
                )
                await backend.set(f"{prefix}-{i}", entry)

        user_keys = await backend.keys("user-*")
        assert len(user_keys) == 2

        admin_keys = await backend.keys("admin-*")
        assert len(admin_keys) == 2

    @pytest.mark.asyncio
    async def test_lru_eviction_by_entries(self):
        """Test LRU eviction when max entries exceeded."""
        backend = InMemoryBackend(max_entries=3)

        # Add 3 entries
        for i in range(3):
            entry = CacheEntry(
                key=f"key-{i}",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"key-{i}", entry)
            await asyncio.sleep(0.01)  # Ensure different access times

        assert await backend.size() == 3

        # Access key-2 to make it recently used
        await backend.get("key-2")

        # Add 4th entry - should evict key-0 (oldest)
        entry = CacheEntry(
            key="key-3",
            prompt="Prompt 3",
            response="Response 3",
            model="test-model",
            created_at=1000000.0,
        )
        await backend.set("key-3", entry)

        # Check eviction happened
        size = await backend.size()
        assert size <= 3

    @pytest.mark.asyncio
    async def test_size(self, sample_entry):
        """Test size reporting."""
        backend = InMemoryBackend()

        assert await backend.size() == 0

        await backend.set("key-1", sample_entry)
        assert await backend.size() == 1

        entry2 = CacheEntry(
            key="key-2",
            prompt="Another prompt",
            response="Another response",
            model="test-model",
            created_at=1000000.0,
        )
        await backend.set("key-2", entry2)
        assert await backend.size() == 2


class TestRedisBackend:
    """Tests for RedisBackend (mocked)."""

    @pytest.mark.asyncio
    async def test_redis_init(self):
        """Test Redis backend initialization."""
        backend = RedisBackend(
            host="localhost",
            port=6379,
            prefix="test:",
        )
        assert backend._host == "localhost"
        assert backend._port == 6379
        assert backend._prefix == "test:"

    @pytest.mark.asyncio
    async def test_redis_make_key(self):
        """Test key prefixing."""
        backend = RedisBackend(prefix="myapp:")
        key = backend._make_key("test-key")
        assert key == "myapp:test-key"

    @pytest.mark.asyncio
    async def test_redis_serialization(self, sample_entry):
        """Test entry serialization/deserialization."""
        backend = RedisBackend()

        serialized = backend._serialize(sample_entry)
        assert isinstance(serialized, bytes)

        deserialized = backend._deserialize(serialized)
        assert deserialized.key == sample_entry.key
        assert deserialized.prompt == sample_entry.prompt
        assert deserialized.response == sample_entry.response

    @pytest.mark.asyncio
    async def test_redis_set_get_mocked(self, sample_entry):
        """Test Redis set/get with mocked client."""
        backend = RedisBackend(prefix="test:")

        # Mock the Redis client
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=backend._serialize(sample_entry))
        mock_client.set = AsyncMock(return_value=True)
        mock_client.setex = AsyncMock(return_value=True)
        mock_client.ttl = AsyncMock(return_value=-1)

        backend._client = mock_client
        backend._initialized = True

        # Test set
        result = await backend.set("test-key", sample_entry)
        assert result is True
        mock_client.set.assert_called()

        # Test get
        entry = await backend.get("test-key")
        assert entry is not None
        assert entry.prompt == sample_entry.prompt

    @pytest.mark.asyncio
    async def test_redis_url_init(self):
        """Test Redis URL initialization."""
        backend = RedisBackend(url="redis://user:pass@localhost:6379/0")
        assert backend._url == "redis://user:pass@localhost:6379/0"


class TestMemcachedBackend:
    """Tests for MemcachedBackend (mocked)."""

    @pytest.mark.asyncio
    async def test_memcached_init(self):
        """Test Memcached backend initialization."""
        backend = MemcachedBackend(
            host="localhost",
            port=11211,
            prefix="mc_",
        )
        assert backend._host == "localhost"
        assert backend._port == 11211
        assert backend._prefix == "mc_"

    @pytest.mark.asyncio
    async def test_memcached_make_key(self):
        """Test key creation (should be hashed for safety)."""
        backend = MemcachedBackend(prefix="app_")
        key = backend._make_key("test-key")
        # Key should be bytes and contain prefix
        assert isinstance(key, bytes)
        assert key.startswith(b"app_")

    @pytest.mark.asyncio
    async def test_memcached_serialization(self, sample_entry):
        """Test entry serialization/deserialization."""
        backend = MemcachedBackend()

        serialized = backend._serialize(sample_entry)
        assert isinstance(serialized, bytes)

        deserialized = backend._deserialize(serialized)
        assert deserialized.key == sample_entry.key
        assert deserialized.prompt == sample_entry.prompt


class TestCreateBackend:
    """Tests for the create_backend factory function."""

    def test_create_memory_backend(self):
        """Test creating in-memory backend."""
        backend = create_backend("memory", max_entries=5000)
        assert isinstance(backend, InMemoryBackend)
        assert backend._max_entries == 5000

    def test_create_redis_backend(self):
        """Test creating Redis backend."""
        backend = create_backend("redis", host="redis.example.com", port=6380)
        assert isinstance(backend, RedisBackend)
        assert backend._host == "redis.example.com"
        assert backend._port == 6380

    def test_create_memcached_backend(self):
        """Test creating Memcached backend."""
        backend = create_backend("memcached", host="memcached.example.com")
        assert isinstance(backend, MemcachedBackend)
        assert backend._host == "memcached.example.com"

    def test_create_backend_case_insensitive(self):
        """Test that backend type is case-insensitive."""
        backend1 = create_backend("MEMORY")
        backend2 = create_backend("Memory")
        backend3 = create_backend("memory")

        assert isinstance(backend1, InMemoryBackend)
        assert isinstance(backend2, InMemoryBackend)
        assert isinstance(backend3, InMemoryBackend)

    def test_create_backend_unknown_type(self):
        """Test error on unknown backend type."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("unknown")


class TestCacheBackendProtocol:
    """Tests for CacheBackendProtocol."""

    def test_inmemory_implements_protocol(self):
        """Test InMemoryBackend implements protocol."""
        backend = InMemoryBackend()
        assert isinstance(backend, CacheBackendProtocol)

    def test_redis_implements_protocol(self):
        """Test RedisBackend implements protocol."""
        backend = RedisBackend()
        assert isinstance(backend, CacheBackendProtocol)

    def test_memcached_implements_protocol(self):
        """Test MemcachedBackend implements protocol."""
        backend = MemcachedBackend()
        assert isinstance(backend, CacheBackendProtocol)


class TestCacheBackendConcurrency:
    """Tests for concurrent access to cache backends."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """Test concurrent write operations."""
        backend = InMemoryBackend()

        async def write_entry(i: int):
            entry = CacheEntry(
                key=f"key-{i}",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"key-{i}", entry)

        # Concurrent writes
        await asyncio.gather(*[write_entry(i) for i in range(100)])

        assert await backend.size() == 100

    @pytest.mark.asyncio
    async def test_concurrent_reads_writes(self):
        """Test concurrent read and write operations."""
        backend = InMemoryBackend()

        # Pre-populate
        for i in range(50):
            entry = CacheEntry(
                key=f"key-{i}",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"key-{i}", entry)

        async def read_and_write(i: int):
            # Read existing
            await backend.get(f"key-{i % 50}")
            # Write new
            entry = CacheEntry(
                key=f"new-key-{i}",
                prompt=f"New Prompt {i}",
                response=f"New Response {i}",
                model="test-model",
                created_at=1000000.0,
            )
            await backend.set(f"new-key-{i}", entry)

        # Concurrent operations
        await asyncio.gather(*[read_and_write(i) for i in range(50)])

        assert await backend.size() == 100
