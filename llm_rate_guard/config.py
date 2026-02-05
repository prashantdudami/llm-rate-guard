"""Configuration classes for LLM Rate Guard."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"
    VERTEX = "vertex"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider endpoint."""

    model_config = {"extra": "forbid"}  # Catch typos in config

    type: ProviderType
    """Provider type (bedrock, azure_openai, vertex, openai, anthropic)."""

    model: str
    """Model identifier (e.g., 'claude-3-sonnet-20240229', 'gpt-4-turbo')."""

    # Provider-specific settings
    region: Optional[str] = None
    """AWS region for Bedrock, Azure region for Azure OpenAI."""

    endpoint: Optional[str] = None
    """Custom endpoint URL (for Azure OpenAI, or custom deployments)."""

    api_key: Optional[SecretStr] = None
    """API key (for OpenAI, Anthropic, Azure). Can also use env vars."""

    deployment_name: Optional[str] = None
    """Azure OpenAI deployment name."""

    project_id: Optional[str] = None
    """GCP project ID for Vertex AI."""

    # Rate limits (provider defaults used if not specified)
    rpm_limit: Optional[int] = Field(default=None, ge=1, le=100000)
    """Requests per minute limit. Uses provider default if not set."""

    tpm_limit: Optional[int] = Field(default=None, ge=1, le=100_000_000)
    """Tokens per minute limit. Uses provider default if not set."""

    # Timeout configuration
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=600.0)
    """Request timeout in seconds."""

    # Routing weight
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    """Relative weight for load balancing (higher = more traffic)."""

    # Provider-specific extra config
    extra: dict[str, Any] = Field(default_factory=dict)
    """Additional provider-specific configuration."""

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model identifier cannot be empty")
        if len(v) > 256:
            raise ValueError("Model identifier too long (max 256 chars)")
        return v.strip()

    def get_api_key(self) -> Optional[str]:
        """Get the API key value (for internal use only)."""
        return self.api_key.get_secret_value() if self.api_key else None

    def __repr__(self) -> str:
        """Safe repr that doesn't expose API keys."""
        return (
            f"ProviderConfig(type={self.type.value!r}, model={self.model!r}, "
            f"region={self.region!r})"
        )


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10)
    """Maximum number of retry attempts."""

    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    """Initial delay between retries in seconds."""

    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    """Maximum delay between retries in seconds."""

    exponential_base: float = Field(default=2.0, ge=1.5, le=4.0)
    """Base for exponential backoff calculation."""

    jitter: bool = True
    """Add random jitter to retry delays."""


class CacheConfig(BaseModel):
    """Configuration for semantic caching."""

    enabled: bool = True
    """Enable/disable caching."""

    mode: Literal["exact", "semantic"] = "exact"
    """Cache mode: 'exact' for hash-based, 'semantic' for embedding similarity."""

    similarity_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    """Similarity threshold for semantic cache hits (0.5-1.0)."""

    max_entries: int = Field(default=10000, ge=100)
    """Maximum number of cached entries."""

    max_size_bytes: int = Field(default=100_000_000, ge=1_000_000)
    """Maximum total cache size in bytes (default 100MB)."""

    max_entry_size_bytes: int = Field(default=1_000_000, ge=1000)
    """Maximum size for a single cache entry in bytes (default 1MB)."""

    ttl_seconds: Optional[int] = Field(default=3600, ge=60)
    """Time-to-live for cache entries in seconds. None for no expiry."""

    embedding_model: Optional[str] = None
    """Model to use for generating embeddings (for semantic mode)."""


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern."""

    enabled: bool = True
    """Enable circuit breaker."""

    failure_threshold: int = Field(default=5, ge=1, le=100)
    """Number of failures before opening circuit."""

    success_threshold: int = Field(default=2, ge=1, le=20)
    """Number of successes needed to close circuit."""

    half_open_timeout: float = Field(default=30.0, ge=5.0, le=300.0)
    """Timeout before trying half-open state (seconds)."""


class RateGuardConfig(BaseModel):
    """Main configuration for RateGuardClient."""

    model_config = {"extra": "forbid"}  # Catch typos in config

    providers: list[ProviderConfig] = Field(min_length=1)
    """List of provider configurations. At least one required."""

    retry: RetryConfig = Field(default_factory=RetryConfig)
    """Retry configuration."""

    cache: CacheConfig = Field(default_factory=CacheConfig)
    """Cache configuration."""

    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    """Circuit breaker configuration."""

    # Global rate limiting
    global_rpm_limit: Optional[int] = Field(default=None, ge=1)
    """Global RPM limit across all providers. None for no global limit."""

    global_tpm_limit: Optional[int] = Field(default=None, ge=1)
    """Global TPM limit across all providers. None for no global limit."""

    # Failover settings
    failover_enabled: bool = True
    """Enable automatic failover to next provider on rate limit."""

    cooldown_seconds: float = Field(default=60.0, ge=5.0, le=600.0)
    """Cooldown period for rate-limited providers."""

    # Queue settings
    queue_enabled: bool = True
    """Enable priority queue for request management."""

    max_queue_size: int = Field(default=1000, ge=10, le=100000)
    """Maximum number of requests in queue."""

    # Validation settings
    max_message_length: int = Field(default=100000, ge=100, le=10_000_000)
    """Maximum length for a single message content."""

    max_messages_per_request: int = Field(default=100, ge=1, le=1000)
    """Maximum number of messages per request."""

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: list[ProviderConfig]) -> list[ProviderConfig]:
        if not v:
            raise ValueError("At least one provider must be configured")
        return v


# Default rate limits per provider (conservative estimates)
DEFAULT_RATE_LIMITS: dict[ProviderType, dict[str, int]] = {
    ProviderType.BEDROCK: {"rpm": 250, "tpm": 2_000_000},
    ProviderType.AZURE_OPENAI: {"rpm": 60, "tpm": 40_000},
    ProviderType.VERTEX: {"rpm": 60, "tpm": 1_000_000},
    ProviderType.OPENAI: {"rpm": 500, "tpm": 150_000},
    ProviderType.ANTHROPIC: {"rpm": 50, "tpm": 40_000},
}
