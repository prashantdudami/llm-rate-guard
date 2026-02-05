"""Azure OpenAI provider implementation."""

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
    from openai import AsyncAzureOpenAI, RateLimitError, APIError

    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation.

    Supports GPT-4, GPT-3.5-turbo, and other models deployed on Azure.
    """

    def __init__(self, config: ProviderConfig):
        if not AZURE_OPENAI_AVAILABLE:
            raise ProviderNotAvailable("azure_openai", "azure")

        super().__init__(config)
        self._client: Any = None

    def _validate_config(self) -> None:
        if not self.config.endpoint and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            raise ValueError(
                "Azure OpenAI provider requires 'endpoint' in config or "
                "AZURE_OPENAI_ENDPOINT env var"
            )

        if not self.config.get_api_key() and not os.environ.get("AZURE_OPENAI_API_KEY"):
            raise ValueError(
                "Azure OpenAI provider requires 'api_key' in config or "
                "AZURE_OPENAI_API_KEY env var"
            )

    @property
    def client(self) -> Any:
        """Lazy-initialize the Azure OpenAI client."""
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=self.config.get_api_key() or os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=self.config.extra.get("api_version", "2024-02-15-preview"),
                timeout=self.config.timeout_seconds,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    @property
    def default_rpm_limit(self) -> int:
        return 60

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
        """Generate a completion using Azure OpenAI."""
        start_time = time.perf_counter()

        # Convert messages to OpenAI format
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Use deployment name if specified, otherwise use model
        deployment = self.config.deployment_name or self.config.model

        try:
            response = await self.client.chat.completions.create(
                model=deployment,
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
                region=self.config.region,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump(),
                latency_ms=latency_ms,
            )

        except RateLimitError as e:
            # Extract retry-after from headers if available
            retry_after = None
            if e.response and e.response.headers:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass

            raise RateLimitExceeded(
                message=f"Azure OpenAI rate limit exceeded: {e.message}",
                provider=self.provider_name,
                region=self.config.region,
                retry_after=retry_after,
            )

        except APIError as e:
            raise ProviderError(
                message=f"Azure OpenAI error: {e.message}",
                provider=self.provider_name,
                status_code=e.status_code,
                original_error=e,
            )

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings using Azure OpenAI."""
        # Use deployment name for embeddings
        deployment = kwargs.get("deployment", "text-embedding-ada-002")

        try:
            response = await self.client.embeddings.create(
                model=deployment,
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
                message=f"Azure OpenAI rate limit exceeded: {e.message}",
                provider=self.provider_name,
                region=self.config.region,
            )

        except APIError as e:
            raise ProviderError(
                message=f"Azure OpenAI embedding error: {e.message}",
                provider=self.provider_name,
                status_code=e.status_code,
                original_error=e,
            )
