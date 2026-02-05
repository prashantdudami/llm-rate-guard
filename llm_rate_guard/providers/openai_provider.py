"""OpenAI provider implementation."""

import os
import time
from typing import Any

from llm_rate_guard.config import ProviderConfig
from llm_rate_guard.exceptions import ProviderError, ProviderNotAvailable, RateLimitExceeded
from llm_rate_guard.providers.base import (
    BaseProvider,
    CompletionResponse,
    EmbeddingResponse,
    Message,
    Usage,
)

try:
    from openai import AsyncOpenAI, RateLimitError, APIError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """

    def __init__(self, config: ProviderConfig):
        if not OPENAI_AVAILABLE:
            raise ProviderNotAvailable("openai", "openai")

        super().__init__(config)
        self._client: Any = None

    def _validate_config(self) -> None:
        # API key can come from config or environment
        if not self.config.get_api_key() and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI provider requires 'api_key' in config or OPENAI_API_KEY env var"
            )

    @property
    def client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.get_api_key() or os.environ.get("OPENAI_API_KEY"),
                timeout=self.config.timeout_seconds,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_rpm_limit(self) -> int:
        return 500

    @property
    def default_tpm_limit(self) -> int:
        return 150_000

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using OpenAI."""
        start_time = time.perf_counter()

        # Convert messages to OpenAI format
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k in [
                    "top_p", "frequency_penalty", "presence_penalty", 
                    "stop", "logprobs", "n", "seed"
                ]},
            )

            choice = response.choices[0]
            usage = Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump(),
                latency_ms=latency_ms,
            )

        except RateLimitError as e:
            raise RateLimitExceeded(
                message=f"OpenAI rate limit exceeded: {e.message}",
                provider=self.provider_name,
                retry_after=float(e.response.headers.get("retry-after", 60))
                if e.response
                else None,
            )

        except APIError as e:
            raise ProviderError(
                message=f"OpenAI error: {e.message}",
                provider=self.provider_name,
                status_code=e.status_code,
                original_error=e,
            )

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings using OpenAI."""
        embedding_model = kwargs.get("model", "text-embedding-3-small")

        try:
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=text,
            )

            usage = Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
            )

            return EmbeddingResponse(
                embedding=response.data[0].embedding,
                model=response.model,
                usage=usage,
                provider=self.provider_name,
            )

        except RateLimitError as e:
            raise RateLimitExceeded(
                message=f"OpenAI rate limit exceeded: {e.message}",
                provider=self.provider_name,
            )

        except APIError as e:
            raise ProviderError(
                message=f"OpenAI embedding error: {e.message}",
                provider=self.provider_name,
                status_code=e.status_code,
                original_error=e,
            )
