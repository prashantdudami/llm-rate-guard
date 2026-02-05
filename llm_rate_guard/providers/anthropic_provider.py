"""Anthropic provider implementation."""

import os
import time
from typing import Any

from llm_rate_guard.config import ProviderConfig
from llm_rate_guard.exceptions import ProviderError, ProviderNotAvailable, RateLimitExceeded
from llm_rate_guard.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    Usage,
)

try:
    import anthropic
    from anthropic import RateLimitError, APIError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """Anthropic API provider implementation.

    Supports Claude 3 models (Opus, Sonnet, Haiku).
    """

    def __init__(self, config: ProviderConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ProviderNotAvailable("anthropic", "anthropic")

        super().__init__(config)
        self._client: Any = None

    def _validate_config(self) -> None:
        # API key can come from config or environment
        if not self.config.get_api_key() and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "Anthropic provider requires 'api_key' in config or ANTHROPIC_API_KEY env var"
            )

    @property
    def client(self) -> Any:
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.get_api_key() or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=self.config.timeout_seconds,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_rpm_limit(self) -> int:
        return 50

    @property
    def default_tpm_limit(self) -> int:
        return 40_000

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Anthropic."""
        start_time = time.perf_counter()

        # Separate system messages
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]

        # Convert to Anthropic format
        anthropic_messages = [{"role": m.role, "content": m.content} for m in other_messages]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add system prompt if present
        if system_messages:
            request_kwargs["system"] = " ".join(m.content for m in system_messages)

        # Add optional parameters
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                request_kwargs[key] = kwargs[key]

        try:
            response = await self.client.messages.create(**request_kwargs)

            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text

            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                content=content,
                model=response.model,
                usage=usage,
                provider=self.provider_name,
                finish_reason=response.stop_reason,
                raw_response=response.model_dump(),
                latency_ms=latency_ms,
            )

        except RateLimitError as e:
            raise RateLimitExceeded(
                message=f"Anthropic rate limit exceeded: {str(e)}",
                provider=self.provider_name,
            )

        except APIError as e:
            raise ProviderError(
                message=f"Anthropic error: {str(e)}",
                provider=self.provider_name,
                status_code=getattr(e, "status_code", None),
                original_error=e,
            )
