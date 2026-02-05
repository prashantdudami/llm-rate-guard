"""Environment-based configuration for LLM Rate Guard."""

import json
import os
from typing import Any, Optional

from pydantic import SecretStr

from llm_rate_guard.config import (
    CacheConfig,
    CircuitBreakerConfig,
    ProviderConfig,
    ProviderType,
    RateGuardConfig,
    RetryConfig,
)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer from environment variable."""
    value = get_env(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float from environment variable."""
    value = get_env(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_providers_from_env() -> list[ProviderConfig]:
    """Load provider configurations from environment variables.

    Supports two formats:
    1. JSON array: LLM_RATE_GUARD_PROVIDERS='[{"type": "openai", "model": "gpt-4"}]'
    2. Simple format: LLM_RATE_GUARD_PROVIDER_1_TYPE=openai
                      LLM_RATE_GUARD_PROVIDER_1_MODEL=gpt-4
                      LLM_RATE_GUARD_PROVIDER_1_API_KEY=sk-xxx

    Returns:
        List of provider configurations.
    """
    providers: list[ProviderConfig] = []

    # Try JSON format first
    json_config = get_env("LLM_RATE_GUARD_PROVIDERS")
    if json_config:
        try:
            provider_dicts = json.loads(json_config)
            for pdict in provider_dicts:
                # Convert type string to enum
                if "type" in pdict and isinstance(pdict["type"], str):
                    pdict["type"] = ProviderType(pdict["type"])
                providers.append(ProviderConfig(**pdict))
            return providers
        except (json.JSONDecodeError, Exception):
            pass  # Fall through to numbered format

    # Try numbered format
    for i in range(1, 11):  # Support up to 10 providers
        prefix = f"LLM_RATE_GUARD_PROVIDER_{i}"

        provider_type = get_env(f"{prefix}_TYPE")
        model = get_env(f"{prefix}_MODEL")

        if not provider_type or not model:
            continue

        try:
            config = ProviderConfig(
                type=ProviderType(provider_type),
                model=model,
                region=get_env(f"{prefix}_REGION"),
                endpoint=get_env(f"{prefix}_ENDPOINT"),
                api_key=SecretStr(key) if (key := get_env(f"{prefix}_API_KEY")) else None,
                deployment_name=get_env(f"{prefix}_DEPLOYMENT_NAME"),
                project_id=get_env(f"{prefix}_PROJECT_ID"),
                rpm_limit=get_env_int(f"{prefix}_RPM_LIMIT") or None,
                tpm_limit=get_env_int(f"{prefix}_TPM_LIMIT") or None,
                timeout_seconds=get_env_float(f"{prefix}_TIMEOUT", 120.0),
                weight=get_env_float(f"{prefix}_WEIGHT", 1.0),
            )
            providers.append(config)
        except Exception:
            continue

    return providers


def load_config_from_env() -> Optional[RateGuardConfig]:
    """Load full configuration from environment variables.

    Environment variables:
        LLM_RATE_GUARD_PROVIDERS: JSON array of provider configs
        LLM_RATE_GUARD_PROVIDER_N_*: Numbered provider configs (N=1-10)
        LLM_RATE_GUARD_CACHE_ENABLED: Enable caching (default: true)
        LLM_RATE_GUARD_CACHE_MODE: Cache mode ('exact' or 'semantic')
        LLM_RATE_GUARD_CACHE_TTL: Cache TTL in seconds
        LLM_RATE_GUARD_FAILOVER_ENABLED: Enable failover (default: true)
        LLM_RATE_GUARD_COOLDOWN_SECONDS: Cooldown duration
        LLM_RATE_GUARD_QUEUE_ENABLED: Enable queue (default: true)
        LLM_RATE_GUARD_MAX_QUEUE_SIZE: Maximum queue size
        LLM_RATE_GUARD_RETRY_MAX: Maximum retry attempts
        LLM_RATE_GUARD_GLOBAL_RPM: Global RPM limit
        LLM_RATE_GUARD_GLOBAL_TPM: Global TPM limit

    Returns:
        RateGuardConfig if providers are configured, None otherwise.
    """
    providers = load_providers_from_env()
    if not providers:
        return None

    # Build cache config
    cache_mode = get_env("LLM_RATE_GUARD_CACHE_MODE", "exact")
    cache_config = CacheConfig(
        enabled=get_env_bool("LLM_RATE_GUARD_CACHE_ENABLED", True),
        mode=cache_mode if cache_mode in ("exact", "semantic") else "exact",  # type: ignore
        ttl_seconds=get_env_int("LLM_RATE_GUARD_CACHE_TTL", 3600) or None,
        max_entries=get_env_int("LLM_RATE_GUARD_CACHE_MAX_ENTRIES", 10000),
    )

    # Build retry config
    retry_config = RetryConfig(
        max_retries=get_env_int("LLM_RATE_GUARD_RETRY_MAX", 3),
    )

    # Build circuit breaker config
    circuit_breaker_config = CircuitBreakerConfig(
        enabled=get_env_bool("LLM_RATE_GUARD_CIRCUIT_BREAKER_ENABLED", True),
        failure_threshold=get_env_int("LLM_RATE_GUARD_CIRCUIT_BREAKER_THRESHOLD", 5),
    )

    # Build main config
    config = RateGuardConfig(
        providers=providers,
        cache=cache_config,
        retry=retry_config,
        circuit_breaker=circuit_breaker_config,
        failover_enabled=get_env_bool("LLM_RATE_GUARD_FAILOVER_ENABLED", True),
        cooldown_seconds=get_env_float("LLM_RATE_GUARD_COOLDOWN_SECONDS", 60.0),
        queue_enabled=get_env_bool("LLM_RATE_GUARD_QUEUE_ENABLED", True),
        max_queue_size=get_env_int("LLM_RATE_GUARD_MAX_QUEUE_SIZE", 1000),
        global_rpm_limit=get_env_int("LLM_RATE_GUARD_GLOBAL_RPM") or None,
        global_tpm_limit=get_env_int("LLM_RATE_GUARD_GLOBAL_TPM") or None,
    )

    return config


def create_client_from_env():
    """Create a RateGuardClient from environment variables.

    Returns:
        RateGuardClient if configured, raises ValueError otherwise.

    Raises:
        ValueError: If no providers are configured.
    """
    from llm_rate_guard.client import RateGuardClient

    config = load_config_from_env()
    if config is None:
        raise ValueError(
            "No providers configured. Set LLM_RATE_GUARD_PROVIDERS or "
            "LLM_RATE_GUARD_PROVIDER_1_TYPE/MODEL environment variables."
        )

    return RateGuardClient(config=config)
