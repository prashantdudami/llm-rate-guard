"""Pluggable cache backends for distributed caching.

This module provides abstract and concrete cache backend implementations
for use with SemanticCache. Supports in-memory, Redis, and custom backends.
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class CacheEntry:
    """A single cache entry."""

    key: str
    """Cache key (hash of the prompt)."""

    prompt: str
    """Original prompt."""

    response: str
    """Cached response."""

    model: str
    """Model used for generation."""

    created_at: float
    """Timestamp when entry was created."""

    hits: int = 0
    """Number of cache hits for this entry."""

    embedding: Optional[list[float]] = None
    """Embedding vector for semantic matching."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def is_expired(self, ttl_seconds: Optional[int]) -> bool:
        """Check if entry has expired."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "key": self.key,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "created_at": self.created_at,
            "hits": self.hits,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Deserialize entry from dictionary."""
        return cls(
            key=data["key"],
            prompt=data["prompt"],
            response=data["response"],
            model=data["model"],
            created_at=data["created_at"],
            hits=data.get("hits", 0),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class CacheBackendProtocol(Protocol):
    """Protocol for cache backend implementations."""

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        ...

    async def set(self, key: str, entry: CacheEntry, ttl_seconds: Optional[int] = None) -> bool:
        """Set a cache entry."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        ...

    async def clear(self) -> int:
        """Clear all entries. Returns count of entries cleared."""
        ...

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        ...

    async def size(self) -> int:
        """Return current number of entries."""
        ...

    async def close(self) -> None:
        """Close backend connections."""
        ...


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key.

        Args:
            key: Cache key to look up.

        Returns:
            CacheEntry if found, None otherwise.
        """

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry, ttl_seconds: Optional[int] = None) -> bool:
        """Set a cache entry.

        Args:
            key: Cache key.
            entry: CacheEntry to store.
            ttl_seconds: Optional TTL in seconds.

        Returns:
            True if successful, False otherwise.
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key to delete.

        Returns:
            True if entry existed and was deleted, False otherwise.
        """

    @abstractmethod
    async def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared.
        """

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob-style pattern (default "*" for all).

        Returns:
            List of matching keys.
        """

    @abstractmethod
    async def size(self) -> int:
        """Return current number of entries."""

    async def close(self) -> None:
        """Close backend connections. Override if needed."""
        pass


class InMemoryBackend(CacheBackend):
    """In-memory cache backend (default).

    Simple dict-based implementation with optional LRU eviction.
    Suitable for single-process deployments.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_size_bytes: int = 100_000_000,  # 100MB
    ):
        """Initialize in-memory backend.

        Args:
            max_entries: Maximum number of entries.
            max_size_bytes: Maximum total size in bytes.
        """
        self._cache: dict[str, CacheEntry] = {}
        self._access_times: dict[str, float] = {}
        self._entry_sizes: dict[str, int] = {}
        self._total_size_bytes: int = 0
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._lock = asyncio.Lock()

    def _estimate_size(self, entry: CacheEntry) -> int:
        """Estimate entry size in bytes."""
        size = len(entry.prompt.encode("utf-8"))
        size += len(entry.response.encode("utf-8"))
        size += len(entry.model.encode("utf-8"))
        if entry.embedding:
            size += len(entry.embedding) * 8
        size += 100  # Overhead
        return size

    async def _evict_if_needed(self, needed_bytes: int = 0) -> None:
        """Evict LRU entries if at capacity."""
        needs_eviction = (
            len(self._cache) >= self._max_entries
            or (self._total_size_bytes + needed_bytes > self._max_size_bytes)
        )

        if not needs_eviction:
            return

        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])

        for key, _ in sorted_keys:
            if key in self._cache:
                entry_size = self._entry_sizes.get(key, 0)
                self._total_size_bytes -= entry_size
                del self._cache[key]
                del self._access_times[key]
                if key in self._entry_sizes:
                    del self._entry_sizes[key]

                if (
                    len(self._cache) < self._max_entries
                    and self._total_size_bytes + needed_bytes <= self._max_size_bytes
                ):
                    break

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                self._access_times[key] = time.time()
            return entry

    async def set(self, key: str, entry: CacheEntry, ttl_seconds: Optional[int] = None) -> bool:
        """Set entry with optional TTL."""
        entry_size = self._estimate_size(entry)

        async with self._lock:
            await self._evict_if_needed(entry_size)

            # Remove old entry size if updating
            if key in self._entry_sizes:
                self._total_size_bytes -= self._entry_sizes[key]

            self._cache[key] = entry
            self._access_times[key] = time.time()
            self._entry_sizes[key] = entry_size
            self._total_size_bytes += entry_size
            return True

    async def delete(self, key: str) -> bool:
        """Delete entry."""
        async with self._lock:
            if key in self._cache:
                entry_size = self._entry_sizes.get(key, 0)
                self._total_size_bytes -= entry_size
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                if key in self._entry_sizes:
                    del self._entry_sizes[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._entry_sizes.clear()
            self._total_size_bytes = 0
            return count

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        import fnmatch

        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]

    async def size(self) -> int:
        """Return number of entries."""
        return len(self._cache)


class RedisBackend(CacheBackend):
    """Redis-based distributed cache backend.

    Suitable for multi-process/multi-node deployments.
    Requires redis[async] or redis package.

    Example:
        ```python
        from llm_rate_guard.cache_backends import RedisBackend

        backend = RedisBackend(
            host="localhost",
            port=6379,
            db=0,
            prefix="llm_cache:",
        )

        # Use with SemanticCache
        from llm_rate_guard import RateGuardConfig, CacheConfig
        config = RateGuardConfig(
            cache=CacheConfig(enabled=True),
            ...
        )
        # Then pass backend to the cache wrapper
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "llm_rate_guard:",
        url: Optional[str] = None,
        ssl: bool = False,
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
    ):
        """Initialize Redis backend.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password.
            prefix: Key prefix for namespacing.
            url: Redis URL (overrides host/port/db/password).
            ssl: Enable SSL/TLS.
            connection_pool_size: Connection pool size.
            socket_timeout: Socket timeout in seconds.
        """
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._prefix = prefix
        self._url = url
        self._ssl = ssl
        self._pool_size = connection_pool_size
        self._socket_timeout = socket_timeout
        self._client: Any = None
        self._initialized = False

    async def _ensure_client(self) -> Any:
        """Lazy initialize Redis client."""
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as aioredis
        except ImportError:
            try:
                import aioredis
            except ImportError:
                raise ImportError(
                    "Redis backend requires 'redis' package. "
                    "Install with: pip install redis[async] or pip install aioredis"
                )

        if self._url:
            self._client = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=False,  # We handle encoding ourselves
                socket_timeout=self._socket_timeout,
            )
        else:
            self._client = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                ssl=self._ssl,
                socket_timeout=self._socket_timeout,
                max_connections=self._pool_size,
            )

        self._initialized = True
        return self._client

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._prefix}{key}"

    def _serialize(self, entry: CacheEntry) -> bytes:
        """Serialize entry to JSON bytes."""
        return json.dumps(entry.to_dict()).encode("utf-8")

    def _deserialize(self, data: bytes) -> CacheEntry:
        """Deserialize entry from JSON bytes."""
        return CacheEntry.from_dict(json.loads(data.decode("utf-8")))

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis."""
        client = await self._ensure_client()
        redis_key = self._make_key(key)

        data = await client.get(redis_key)
        if data is None:
            return None

        try:
            entry = self._deserialize(data)
            # Update hits counter
            entry.hits += 1
            # Re-save with updated hits (async, fire-and-forget)
            _task = asyncio.create_task(self._update_hits(redis_key, entry))  # noqa: F841
            return entry
        except (json.JSONDecodeError, KeyError):
            # Corrupted entry, delete it
            await client.delete(redis_key)
            return None

    async def _update_hits(self, redis_key: str, entry: CacheEntry) -> None:
        """Update hit counter in Redis."""
        try:
            client = await self._ensure_client()
            # Get remaining TTL
            ttl = await client.ttl(redis_key)
            if ttl > 0:
                await client.setex(redis_key, ttl, self._serialize(entry))
            elif ttl == -1:  # No TTL set
                await client.set(redis_key, self._serialize(entry))
        except Exception:
            pass  # Best effort

    async def set(self, key: str, entry: CacheEntry, ttl_seconds: Optional[int] = None) -> bool:
        """Set entry in Redis."""
        client = await self._ensure_client()
        redis_key = self._make_key(key)
        data = self._serialize(entry)

        try:
            if ttl_seconds:
                await client.setex(redis_key, ttl_seconds, data)
            else:
                await client.set(redis_key, data)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from Redis."""
        client = await self._ensure_client()
        redis_key = self._make_key(key)
        result = await client.delete(redis_key)
        return result > 0

    async def clear(self) -> int:
        """Clear all entries with prefix."""
        client = await self._ensure_client()
        pattern = self._make_key("*")

        # Use SCAN for safety
        count = 0
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break

        return count

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        client = await self._ensure_client()
        redis_pattern = self._make_key(pattern)

        result = []
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=redis_pattern, count=100)
            for key in keys:
                # Remove prefix
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                result.append(key[len(self._prefix) :])
            if cursor == 0:
                break

        return result

    async def size(self) -> int:
        """Return approximate number of entries."""
        client = await self._ensure_client()
        pattern = self._make_key("*")

        count = 0
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break

        return count

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            client = await self._ensure_client()
            return await client.ping()
        except Exception:
            return False


class MemcachedBackend(CacheBackend):
    """Memcached-based distributed cache backend.

    Suitable for simple distributed caching scenarios.
    Requires aiomcache package.

    Note: Memcached has a 1MB value limit by default.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11211,
        prefix: str = "llm_rg_",
        pool_size: int = 10,
        default_ttl: int = 3600,
    ):
        """Initialize Memcached backend.

        Args:
            host: Memcached host.
            port: Memcached port.
            prefix: Key prefix for namespacing.
            pool_size: Connection pool size.
            default_ttl: Default TTL in seconds.
        """
        self._host = host
        self._port = port
        self._prefix = prefix
        self._pool_size = pool_size
        self._default_ttl = default_ttl
        self._client: Any = None
        self._keys_set: set[str] = set()  # Track keys for clear()
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> Any:
        """Lazy initialize Memcached client."""
        if self._client is not None:
            return self._client

        try:
            import aiomcache
        except ImportError:
            raise ImportError(
                "Memcached backend requires 'aiomcache' package. "
                "Install with: pip install aiomcache"
            )

        self._client = aiomcache.Client(
            self._host,
            self._port,
            pool_size=self._pool_size,
        )
        return self._client

    def _make_key(self, key: str) -> bytes:
        """Create memcached-safe key."""
        # Memcached keys: max 250 bytes, no whitespace/control chars
        safe_key = f"{self._prefix}{hashlib.sha256(key.encode()).hexdigest()[:32]}"
        return safe_key.encode("utf-8")

    def _serialize(self, entry: CacheEntry) -> bytes:
        """Serialize entry to JSON bytes."""
        return json.dumps(entry.to_dict()).encode("utf-8")

    def _deserialize(self, data: bytes) -> CacheEntry:
        """Deserialize entry from JSON bytes."""
        return CacheEntry.from_dict(json.loads(data.decode("utf-8")))

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Memcached."""
        client = await self._ensure_client()
        mc_key = self._make_key(key)

        data = await client.get(mc_key)
        if data is None:
            return None

        try:
            return self._deserialize(data)
        except (json.JSONDecodeError, KeyError):
            await client.delete(mc_key)
            return None

    async def set(self, key: str, entry: CacheEntry, ttl_seconds: Optional[int] = None) -> bool:
        """Set entry in Memcached."""
        client = await self._ensure_client()
        mc_key = self._make_key(key)
        data = self._serialize(entry)
        ttl = ttl_seconds or self._default_ttl

        try:
            await client.set(mc_key, data, exptime=ttl)
            async with self._lock:
                self._keys_set.add(key)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from Memcached."""
        client = await self._ensure_client()
        mc_key = self._make_key(key)

        try:
            result = await client.delete(mc_key)
            async with self._lock:
                self._keys_set.discard(key)
            return result
        except Exception:
            return False

    async def clear(self) -> int:
        """Clear tracked entries."""
        client = await self._ensure_client()

        async with self._lock:
            count = len(self._keys_set)
            for key in list(self._keys_set):
                mc_key = self._make_key(key)
                try:
                    await client.delete(mc_key)
                except Exception:
                    pass
            self._keys_set.clear()
            return count

    async def keys(self, pattern: str = "*") -> list[str]:
        """List tracked keys matching pattern."""
        import fnmatch

        async with self._lock:
            if pattern == "*":
                return list(self._keys_set)
            return [k for k in self._keys_set if fnmatch.fnmatch(k, pattern)]

    async def size(self) -> int:
        """Return tracked key count."""
        return len(self._keys_set)

    async def close(self) -> None:
        """Close Memcached connection."""
        if self._client:
            await self._client.close()
            self._client = None


def create_backend(
    backend_type: str = "memory",
    **kwargs: Any,
) -> CacheBackend:
    """Factory function to create cache backends.

    Args:
        backend_type: One of "memory", "redis", "memcached".
        **kwargs: Backend-specific configuration.

    Returns:
        Configured CacheBackend instance.

    Example:
        ```python
        # In-memory (default)
        backend = create_backend("memory", max_entries=10000)

        # Redis
        backend = create_backend("redis", host="localhost", port=6379)

        # Redis with URL
        backend = create_backend("redis", url="redis://user:pass@host:6379/0")

        # Memcached
        backend = create_backend("memcached", host="localhost", port=11211)
        ```
    """
    backend_type = backend_type.lower()

    if backend_type == "memory":
        return InMemoryBackend(**kwargs)
    elif backend_type == "redis":
        return RedisBackend(**kwargs)
    elif backend_type == "memcached":
        return MemcachedBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported: memory, redis, memcached"
        )
