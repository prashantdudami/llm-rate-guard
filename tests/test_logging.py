"""Tests for logging module."""

import logging
import pytest

from llm_rate_guard.logging import (
    get_logger,
    set_log_level,
    mask_sensitive_value,
    LogContext,
)


class TestLogging:
    """Tests for logging utilities."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "llm_rate_guard"

    def test_get_logger_singleton(self):
        """get_logger returns the same logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_set_log_level(self):
        """set_log_level changes the logger level."""
        set_log_level(logging.DEBUG)
        assert get_logger().level == logging.DEBUG

        set_log_level(logging.WARNING)
        assert get_logger().level == logging.WARNING


class TestMaskSensitiveValue:
    """Tests for mask_sensitive_value function."""

    def test_mask_normal_value(self):
        """Normal values are masked with visible suffix."""
        value = "sk-1234567890abcdef"  # 18 chars
        result = mask_sensitive_value(value, visible_chars=4)
        assert result.endswith("cdef")
        assert len(result) == len(value)
        assert "1234567890" not in result

    def test_mask_short_value(self):
        """Short values are fully masked."""
        result = mask_sensitive_value("abc", visible_chars=4)
        assert result == "***"

    def test_mask_none_value(self):
        """None values return placeholder."""
        result = mask_sensitive_value(None)
        assert result == "<not set>"

    def test_mask_empty_value(self):
        """Empty string is fully masked."""
        result = mask_sensitive_value("", visible_chars=4)
        assert result == ""

    def test_mask_custom_visible_chars(self):
        """Custom visible_chars is respected."""
        result = mask_sensitive_value("sk-1234567890", visible_chars=6)
        assert result.endswith("567890")
        assert result.startswith("*")


class TestLogContext:
    """Tests for LogContext class."""

    def test_to_dict_includes_all_fields(self):
        """to_dict includes all non-empty fields."""
        ctx = LogContext(
            request_id="abc123",
            provider="openai",
            region="us-east-1",
            extra={"model": "gpt-4"},
        )

        d = ctx.to_dict()

        assert d["request_id"] == "abc123"
        assert d["provider"] == "openai"
        assert d["region"] == "us-east-1"
        assert d["model"] == "gpt-4"

    def test_to_dict_excludes_empty_fields(self):
        """to_dict excludes empty fields."""
        ctx = LogContext(request_id="abc123")

        d = ctx.to_dict()

        assert "request_id" in d
        assert "provider" not in d
        assert "region" not in d

    def test_str_representation(self):
        """String representation is formatted correctly."""
        ctx = LogContext(
            request_id="abc123",
            provider="openai",
        )

        s = str(ctx)

        assert "request_id=abc123" in s
        assert "provider=openai" in s

    def test_empty_context_str(self):
        """Empty context has empty string representation."""
        ctx = LogContext()
        assert str(ctx) == ""
