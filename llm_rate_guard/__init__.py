"""LLM Rate Guard - Cloud-agnostic rate limit mitigation for LLM APIs."""

from llm_rate_guard.config import (
    CacheConfig,
    CircuitBreakerConfig,
    ProviderConfig,
    ProviderType,
    RateGuardConfig,
    RetryConfig,
)
from llm_rate_guard.client import RateGuardClient
from llm_rate_guard.context import (
    MiddlewareChain,
    QuotaManager,
    RequestContext,
    get_current_context,
    set_current_context,
)
from llm_rate_guard.exceptions import (
    RateGuardError,
    RateLimitExceeded,
    AllProvidersExhausted,
    ProviderError,
    ConfigurationError,
    CacheError,
    ProviderNotAvailable,
)
from llm_rate_guard.logging import get_logger, set_log_level, mask_sensitive_value
from llm_rate_guard.queue import Priority
from llm_rate_guard.metrics import (
    Metrics,
    SimpleCostTracker,
    create_cost_tracker,
    LLM_COST_GUARD_AVAILABLE,
)
from llm_rate_guard.env_config import load_config_from_env, create_client_from_env
from llm_rate_guard.providers.base import Message, StreamChunk, CompletionResponse, EmbeddingResponse
from llm_rate_guard.cache_backends import (
    CacheBackend,
    CacheBackendProtocol,
    InMemoryBackend,
    RedisBackend,
    MemcachedBackend,
    create_backend,
)

from llm_rate_guard.standalone import (
    SyncRateLimiter,
    SyncCircuitBreaker,
    rate_limited,
    with_retry,
    circuit_protected,
)

__version__ = "0.2.0"

__all__ = [
    # Main client
    "RateGuardClient",
    # Configuration
    "RateGuardConfig",
    "ProviderConfig",
    "ProviderType",
    "CacheConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
    # Context and Middleware
    "RequestContext",
    "MiddlewareChain",
    "QuotaManager",
    "get_current_context",
    "set_current_context",
    # Cache backends
    "CacheBackend",
    "CacheBackendProtocol",
    "InMemoryBackend",
    "RedisBackend",
    "MemcachedBackend",
    "create_backend",
    # Exceptions
    "RateGuardError",
    "RateLimitExceeded",
    "AllProvidersExhausted",
    "ProviderError",
    "ConfigurationError",
    "CacheError",
    "ProviderNotAvailable",
    # Enums
    "Priority",
    # Metrics
    "Metrics",
    "SimpleCostTracker",
    "create_cost_tracker",
    "LLM_COST_GUARD_AVAILABLE",
    # Logging
    "get_logger",
    "set_log_level",
    "mask_sensitive_value",
    # Environment configuration
    "load_config_from_env",
    "create_client_from_env",
    # Data classes
    "Message",
    "StreamChunk",
    "CompletionResponse",
    "EmbeddingResponse",
    # Standalone components
    "SyncRateLimiter",
    "SyncCircuitBreaker",
    "rate_limited",
    "with_retry",
    "circuit_protected",
]
