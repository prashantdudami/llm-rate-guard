"""Custom exceptions for LLM Rate Guard."""

from typing import Optional


class RateGuardError(Exception):
    """Base exception for all LLM Rate Guard errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RateLimitExceeded(RateGuardError):
    """Raised when rate limit is exceeded and cannot be mitigated."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: Optional[str] = None,
        region: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        details = {
            "provider": provider,
            "region": region,
            "retry_after": retry_after,
        }
        super().__init__(message, details)
        self.provider = provider
        self.region = region
        self.retry_after = retry_after


class AllProvidersExhausted(RateGuardError):
    """Raised when all configured providers are rate limited or unavailable."""

    def __init__(
        self,
        message: str = "All providers exhausted",
        attempted_providers: Optional[list[str]] = None,
    ):
        details = {"attempted_providers": attempted_providers or []}
        super().__init__(message, details)
        self.attempted_providers = attempted_providers or []


class ProviderError(RateGuardError):
    """Raised when a provider returns an error (non-rate-limit)."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {
            "provider": provider,
            "status_code": status_code,
        }
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code
        self.original_error = original_error


class ConfigurationError(RateGuardError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field}
        super().__init__(message, details)
        self.field = field


class CacheError(RateGuardError):
    """Raised when cache operations fail."""

    pass


class ProviderNotAvailable(RateGuardError):
    """Raised when a provider SDK is not installed."""

    def __init__(self, provider: str, install_extra: str):
        message = (
            f"Provider '{provider}' requires additional dependencies. "
            f"Install with: pip install llm-rate-guard[{install_extra}]"
        )
        super().__init__(message, {"provider": provider, "install_extra": install_extra})
        self.provider = provider
        self.install_extra = install_extra
