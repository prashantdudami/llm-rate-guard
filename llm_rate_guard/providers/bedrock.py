"""AWS Bedrock provider implementation."""

import json
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
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider implementation.

    Supports Claude, Llama, Titan, and other Bedrock models.
    """

    def __init__(self, config: ProviderConfig):
        if not BOTO3_AVAILABLE:
            raise ProviderNotAvailable("bedrock", "bedrock")

        super().__init__(config)
        self._client = None

    def _validate_config(self) -> None:
        if not self.config.region:
            raise ValueError("Bedrock provider requires 'region' in config")

    @property
    def client(self) -> Any:
        """Lazy-initialize the Bedrock client."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.config.region,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def default_rpm_limit(self) -> int:
        return 250

    @property
    def default_tpm_limit(self) -> int:
        return 2_000_000

    def _is_claude_model(self) -> bool:
        """Check if the model is a Claude model."""
        return "anthropic.claude" in self.config.model.lower()

    def _is_llama_model(self) -> bool:
        """Check if the model is a Llama model."""
        return "llama" in self.config.model.lower()

    def _is_titan_model(self) -> bool:
        """Check if the model is a Titan model."""
        return "amazon.titan" in self.config.model.lower()

    def _format_messages_for_claude(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict:
        """Format request body for Claude models on Bedrock."""
        # Separate system messages
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in other_messages],
        }

        if system_messages:
            body["system"] = " ".join(m.content for m in system_messages)

        # Add any extra kwargs
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                body[key] = kwargs[key]

        return body

    def _format_messages_for_llama(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict:
        """Format request body for Llama models on Bedrock."""
        # Convert messages to Llama prompt format
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>")

        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)

        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
        }

    def _format_messages_for_titan(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict:
        """Format request body for Titan models on Bedrock."""
        # Combine all messages into a single prompt
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)

        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": kwargs.get("top_p", 0.9),
            },
        }

    def _parse_claude_response(self, response_body: dict) -> tuple[str, Usage, str]:
        """Parse Claude response."""
        content = ""
        if response_body.get("content"):
            content = response_body["content"][0].get("text", "")

        usage = Usage(
            input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
            output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
        )

        finish_reason = response_body.get("stop_reason", "")
        return content, usage, finish_reason

    def _parse_llama_response(self, response_body: dict) -> tuple[str, Usage, str]:
        """Parse Llama response."""
        content = response_body.get("generation", "")
        usage = Usage(
            input_tokens=response_body.get("prompt_token_count", 0),
            output_tokens=response_body.get("generation_token_count", 0),
        )
        finish_reason = response_body.get("stop_reason", "")
        return content, usage, finish_reason

    def _parse_titan_response(self, response_body: dict) -> tuple[str, Usage, str]:
        """Parse Titan response."""
        results = response_body.get("results", [{}])
        content = results[0].get("outputText", "") if results else ""
        usage = Usage(
            input_tokens=response_body.get("inputTextTokenCount", 0),
            output_tokens=results[0].get("tokenCount", 0) if results else 0,
        )
        finish_reason = results[0].get("completionReason", "") if results else ""
        return content, usage, finish_reason

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Bedrock."""
        start_time = time.perf_counter()

        # Format request based on model type
        if self._is_claude_model():
            body = self._format_messages_for_claude(messages, max_tokens, temperature, **kwargs)
        elif self._is_llama_model():
            body = self._format_messages_for_llama(messages, max_tokens, temperature, **kwargs)
        elif self._is_titan_model():
            body = self._format_messages_for_titan(messages, max_tokens, temperature, **kwargs)
        else:
            # Default to Claude format
            body = self._format_messages_for_claude(messages, max_tokens, temperature, **kwargs)

        try:
            response = self.client.invoke_model(
                modelId=self.config.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Parse response based on model type
            if self._is_claude_model():
                content, usage, finish_reason = self._parse_claude_response(response_body)
            elif self._is_llama_model():
                content, usage, finish_reason = self._parse_llama_response(response_body)
            elif self._is_titan_model():
                content, usage, finish_reason = self._parse_titan_response(response_body)
            else:
                content, usage, finish_reason = self._parse_claude_response(response_body)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                content=content,
                model=self.config.model,
                usage=usage,
                provider=self.provider_name,
                region=self.config.region,
                finish_reason=finish_reason,
                raw_response=response_body,
                latency_ms=latency_ms,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                raise RateLimitExceeded(
                    message=f"Bedrock rate limit exceeded: {error_message}",
                    provider=self.provider_name,
                    region=self.config.region,
                )

            raise ProviderError(
                message=f"Bedrock error: {error_message}",
                provider=self.provider_name,
                original_error=e,
            )

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings using Bedrock Titan Embeddings."""
        # Use Titan Embeddings model
        embedding_model = kwargs.get("model", "amazon.titan-embed-text-v1")

        body = {"inputText": text}

        try:
            response = self.client.invoke_model(
                modelId=embedding_model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding", [])

            return EmbeddingResponse(
                embedding=embedding,
                model=embedding_model,
                usage=Usage(input_tokens=response_body.get("inputTextTokenCount", 0)),
                provider=self.provider_name,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                raise RateLimitExceeded(
                    message=f"Bedrock rate limit exceeded: {error_message}",
                    provider=self.provider_name,
                    region=self.config.region,
                )

            raise ProviderError(
                message=f"Bedrock embedding error: {error_message}",
                provider=self.provider_name,
                original_error=e,
            )
