"""Google Vertex AI provider implementation."""

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
    from google.cloud import aiplatform
    from google.api_core import exceptions as google_exceptions
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False


class VertexAIProvider(BaseProvider):
    """Google Vertex AI provider implementation.

    Supports Gemini, PaLM, and Claude models on Vertex AI.
    """

    def __init__(self, config: ProviderConfig):
        if not VERTEX_AVAILABLE:
            raise ProviderNotAvailable("vertex", "vertex")

        super().__init__(config)
        self._model: Any = None
        self._initialized = False

    def _validate_config(self) -> None:
        if not self.config.project_id and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            raise ValueError(
                "Vertex AI provider requires 'project_id' in config or "
                "GOOGLE_CLOUD_PROJECT env var"
            )

        if not self.config.region:
            raise ValueError("Vertex AI provider requires 'region' in config")

    def _initialize(self) -> None:
        """Initialize Vertex AI SDK."""
        if not self._initialized:
            project_id = self.config.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            vertexai.init(project=project_id, location=self.config.region)
            self._initialized = True

    @property
    def model(self) -> Any:
        """Lazy-initialize the Vertex AI model."""
        if self._model is None:
            self._initialize()
            self._model = GenerativeModel(self.config.model)
        return self._model

    @property
    def provider_name(self) -> str:
        return "vertex"

    @property
    def default_rpm_limit(self) -> int:
        return 60

    @property
    def default_tpm_limit(self) -> int:
        return 1_000_000

    def _is_gemini_model(self) -> bool:
        """Check if the model is a Gemini model."""
        return "gemini" in self.config.model.lower()

    def _is_claude_model(self) -> bool:
        """Check if the model is a Claude model on Vertex."""
        return "claude" in self.config.model.lower()

    def _convert_messages_to_contents(self, messages: list[Message]) -> tuple[list[Any], str]:
        """Convert messages to Vertex AI format.

        Returns:
            Tuple of (contents list, system instruction)
        """
        system_instruction = ""
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                # Map 'assistant' to 'model' for Vertex AI
                role = "model" if msg.role == "assistant" else msg.role
                contents.append(Content(role=role, parts=[Part.from_text(msg.content)]))

        return contents, system_instruction

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Vertex AI."""
        start_time = time.perf_counter()

        contents, system_instruction = self._convert_messages_to_contents(messages)

        # Create generation config
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            generation_config["top_k"] = kwargs["top_k"]

        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = GenerativeModel(
                    self.config.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self.model

            # Generate content
            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
            )

            # Extract content
            content = response.text if response.text else ""

            # Extract usage (Vertex AI provides token counts)
            usage_metadata = getattr(response, "usage_metadata", None)
            usage = Usage(
                input_tokens=getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
                output_tokens=getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
            )

            # Get finish reason
            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason.name)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                content=content,
                model=self.config.model,
                usage=usage,
                provider=self.provider_name,
                region=self.config.region,
                finish_reason=finish_reason,
                raw_response=response,
                latency_ms=latency_ms,
            )

        except google_exceptions.ResourceExhausted as e:
            raise RateLimitExceeded(
                message=f"Vertex AI rate limit exceeded: {str(e)}",
                provider=self.provider_name,
                region=self.config.region,
            )

        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(
                message=f"Vertex AI error: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            )

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings using Vertex AI."""
        from vertexai.language_models import TextEmbeddingModel

        embedding_model_name = kwargs.get("model", "textembedding-gecko@003")

        try:
            self._initialize()
            embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)

            # Generate embedding
            embeddings = embedding_model.get_embeddings([text])

            return EmbeddingResponse(
                embedding=embeddings[0].values,
                model=embedding_model_name,
                usage=Usage(input_tokens=len(text.split())),  # Rough estimate
                provider=self.provider_name,
            )

        except google_exceptions.ResourceExhausted as e:
            raise RateLimitExceeded(
                message=f"Vertex AI rate limit exceeded: {str(e)}",
                provider=self.provider_name,
                region=self.config.region,
            )

        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(
                message=f"Vertex AI embedding error: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            )
