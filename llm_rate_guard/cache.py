"""In-memory semantic cache for LLM responses."""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from llm_rate_guard.config import CacheConfig


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


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    """Total cache hits."""

    misses: int = 0
    """Total cache misses."""

    entries: int = 0
    """Current number of entries."""

    evictions: int = 0
    """Total number of evicted entries."""

    current_size_bytes: int = 0
    """Current total cache size in bytes."""

    rejected_size: int = 0
    """Number of entries rejected due to size limits."""

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class SemanticCache:
    """In-memory cache with optional semantic similarity matching.

    Supports two modes:
    - 'exact': Hash-based exact matching (fast, no dependencies)
    - 'semantic': Embedding-based similarity matching (requires numpy)

    Uses LRU eviction when max_entries is reached.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        """Initialize the cache.

        Args:
            config: Cache configuration.
            embedding_fn: Optional async function to generate embeddings.
                         Required for semantic mode.
        """
        self.config = config or CacheConfig()
        self._embedding_fn = embedding_fn

        # Cache storage: key -> CacheEntry
        self._cache: dict[str, CacheEntry] = {}

        # LRU tracking: key -> last access time
        self._access_times: dict[str, float] = {}

        # Size tracking: key -> size in bytes
        self._entry_sizes: dict[str, int] = {}
        self._total_size_bytes: int = 0

        # Statistics
        self.stats = CacheStats()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Validate semantic mode requirements
        if self.config.mode == "semantic" and embedding_fn is None:
            # Fall back to exact mode if no embedding function provided
            self.config.mode = "exact"

    def _estimate_entry_size(self, entry: CacheEntry) -> int:
        """Estimate memory size of a cache entry in bytes."""
        size = len(entry.prompt.encode("utf-8"))
        size += len(entry.response.encode("utf-8"))
        size += len(entry.model.encode("utf-8"))
        size += len(entry.key.encode("utf-8"))
        if entry.embedding:
            size += len(entry.embedding) * 8  # float64
        size += 100  # Overhead estimate
        return size

    def _hash_prompt(self, prompt: str, model: str = "") -> str:
        """Generate hash key for a prompt."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np

            a_arr = np.array(a)
            b_arr = np.array(b)
            return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
        except ImportError:
            # Fallback to pure Python
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

    async def _find_semantic_match(
        self,
        prompt: str,
        embedding: list[float],
    ) -> Optional[CacheEntry]:
        """Find a semantically similar cached entry."""
        best_match: Optional[CacheEntry] = None
        best_similarity = 0.0

        for entry in self._cache.values():
            if entry.embedding is None:
                continue

            if entry.is_expired(self.config.ttl_seconds):
                continue

            similarity = self._cosine_similarity(embedding, entry.embedding)

            if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        return best_match

    async def _evict_lru(self, needed_bytes: int = 0) -> None:
        """Evict least recently used entries to make room.

        Args:
            needed_bytes: Additional bytes needed for new entry.
        """
        # Check if eviction is needed
        needs_eviction = (
            len(self._cache) >= self.config.max_entries or
            (self._total_size_bytes + needed_bytes > self.config.max_size_bytes)
        )

        if not needs_eviction:
            return

        # Sort by access time and remove oldest entries
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])

        bytes_freed = 0
        entries_removed = 0
        target_bytes = needed_bytes + (self.config.max_size_bytes // 10)  # Free extra 10%

        for key, _ in sorted_keys:
            if key in self._cache:
                # Track size before removal
                entry_size = self._entry_sizes.get(key, 0)
                bytes_freed += entry_size
                self._total_size_bytes -= entry_size

                del self._cache[key]
                del self._access_times[key]
                if key in self._entry_sizes:
                    del self._entry_sizes[key]

                self.stats.evictions += 1
                entries_removed += 1

                # Check if we've freed enough
                if (
                    len(self._cache) < self.config.max_entries and
                    (bytes_freed >= target_bytes or self._total_size_bytes + needed_bytes <= self.config.max_size_bytes)
                ):
                    break

        self.stats.current_size_bytes = self._total_size_bytes

    async def get(
        self,
        prompt: str,
        model: str = "",
        embedding: Optional[list[float]] = None,
    ) -> Optional[str]:
        """Get a cached response for the given prompt.

        Args:
            prompt: The prompt to look up.
            model: Model identifier (for exact matching).
            embedding: Pre-computed embedding (for semantic matching).

        Returns:
            Cached response if found, None otherwise.
        """
        if not self.config.enabled:
            self.stats.misses += 1
            return None

        async with self._lock:
            # Try exact match first (always, for performance)
            key = self._hash_prompt(prompt, model)

            if key in self._cache:
                entry = self._cache[key]

                if entry.is_expired(self.config.ttl_seconds):
                    # Entry expired, remove it
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
                else:
                    # Cache hit
                    entry.hits += 1
                    self._access_times[key] = time.time()
                    self.stats.hits += 1
                    return entry.response

            # Try semantic matching if enabled
            if self.config.mode == "semantic" and embedding is not None:
                match = await self._find_semantic_match(prompt, embedding)

                if match:
                    match.hits += 1
                    self._access_times[match.key] = time.time()
                    self.stats.hits += 1
                    return match.response

            self.stats.misses += 1
            return None

    async def set(
        self,
        prompt: str,
        response: str,
        model: str = "",
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Cache a response for the given prompt.

        Args:
            prompt: The prompt.
            response: The response to cache.
            model: Model identifier.
            embedding: Embedding vector (for semantic matching).
            metadata: Additional metadata to store.

        Returns:
            True if cached successfully, False if rejected due to size limits.
        """
        if not self.config.enabled:
            return False

        # Create entry to check size
        key = self._hash_prompt(prompt, model)
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            model=model,
            created_at=time.time(),
            embedding=embedding,
            metadata=metadata or {},
        )

        entry_size = self._estimate_entry_size(entry)

        # Reject if single entry exceeds limit
        if entry_size > self.config.max_entry_size_bytes:
            self.stats.rejected_size += 1
            return False

        async with self._lock:
            # Evict if at capacity
            await self._evict_lru(needed_bytes=entry_size)

            # Remove existing entry size if updating
            if key in self._entry_sizes:
                self._total_size_bytes -= self._entry_sizes[key]

            self._cache[key] = entry
            self._access_times[key] = time.time()
            self._entry_sizes[key] = entry_size
            self._total_size_bytes += entry_size

            self.stats.entries = len(self._cache)
            self.stats.current_size_bytes = self._total_size_bytes

            return True

    async def invalidate(self, prompt: str, model: str = "") -> bool:
        """Invalidate a cached entry.

        Args:
            prompt: The prompt to invalidate.
            model: Model identifier.

        Returns:
            True if entry was found and removed, False otherwise.
        """
        async with self._lock:
            key = self._hash_prompt(prompt, model)

            if key in self._cache:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self.stats.entries = len(self._cache)
                return True

            return False

    async def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._entry_sizes.clear()
            self._total_size_bytes = 0
            self.stats.entries = 0
            self.stats.current_size_bytes = 0
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        if self.config.ttl_seconds is None:
            return 0

        async with self._lock:
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired(self.config.ttl_seconds):
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]

            self.stats.entries = len(self._cache)
            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate_pct": self.stats.hit_rate,
            "entries": self.stats.entries,
            "max_entries": self.config.max_entries,
            "evictions": self.stats.evictions,
            "rejected_size": self.stats.rejected_size,
            "current_size_bytes": self.stats.current_size_bytes,
            "max_size_bytes": self.config.max_size_bytes,
            "size_utilization_pct": (self._total_size_bytes / self.config.max_size_bytes * 100)
            if self.config.max_size_bytes > 0 else 0.0,
            "ttl_seconds": self.config.ttl_seconds,
        }
