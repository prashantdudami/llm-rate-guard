"""Logging infrastructure for LLM Rate Guard."""

import logging
import sys
from typing import Any, Optional

# Module-level logger
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the LLM Rate Guard logger.

    Returns:
        Configured logger instance.
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("llm_rate_guard")
        # Don't add handlers if the logger already has them
        # (user may have configured their own)
        if not _logger.handlers and not _logger.parent.handlers:  # type: ignore
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            _logger.addHandler(handler)
            _logger.setLevel(logging.WARNING)  # Default to WARNING
    return _logger


def set_log_level(level: int) -> None:
    """Set the log level for LLM Rate Guard.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    get_logger().setLevel(level)


def mask_sensitive_value(value: Optional[str], visible_chars: int = 4) -> str:
    """Mask a sensitive value for logging.

    Args:
        value: The value to mask.
        visible_chars: Number of characters to show at the end.

    Returns:
        Masked value (e.g., "****abcd").
    """
    if value is None:
        return "<not set>"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


class LogContext:
    """Context for structured logging with request correlation."""

    def __init__(
        self,
        request_id: str = "",
        provider: str = "",
        region: str = "",
        extra: Optional[dict[str, Any]] = None,
    ):
        self.request_id = request_id
        self.provider = provider
        self.region = region
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "request_id": self.request_id,
            "provider": self.provider,
            "region": self.region,
        }
        result.update(self.extra)
        return {k: v for k, v in result.items() if v}  # Filter empty values

    def __str__(self) -> str:
        parts = []
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.region:
            parts.append(f"region={self.region}")
        for k, v in self.extra.items():
            parts.append(f"{k}={v}")
        return " ".join(parts) if parts else ""
