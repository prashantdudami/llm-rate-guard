"""Base provider interface for LLM Rate Guard."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

from llm_rate_guard.config import ProviderConfig


@dataclass(frozen=True)
class Message:
    """A message in the conversation.

    This is an immutable, hashable message object that can be used
    as a cache key or in sets.
    """

    __slots__ = ("role", "content")

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Message":
        """Create from dictionary."""
        return cls(role=data["role"], content=data["content"])  # type: ignore

    def __hash__(self) -> int:
        """Hash based on role and content."""
        return hash((self.role, self.content))

    def __eq__(self, other: object) -> bool:
        """Compare messages by role and content."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.role == other.role and self.content == other.content


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    content: str
    """The generated text content."""

    model: str
    """Model that generated the response."""

    usage: Usage = field(default_factory=Usage)
    """Token usage information."""

    provider: str = ""
    """Provider that handled the request."""

    region: Optional[str] = None
    """Region of the provider (if applicable)."""

    finish_reason: Optional[str] = None
    """Reason for completion (e.g., 'stop', 'length')."""

    raw_response: Optional[Any] = None
    """Raw response from the provider (for debugging)."""

    latency_ms: float = 0.0
    """Request latency in milliseconds."""

    cached: bool = False
    """Whether the response was served from cache."""


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""

    embedding: list[float]
    """The embedding vector."""

    model: str
    """Model that generated the embedding."""

    usage: Usage = field(default_factory=Usage)
    """Token usage information."""

    provider: str = ""
    """Provider that handled the request."""


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str
    """Content for this chunk."""

    done: bool = False
    """Whether this is the final chunk."""

    model: Optional[str] = None
    """Model identifier (usually in final chunk)."""

    usage: Optional[Usage] = None
    """Token usage (usually in final chunk)."""

    finish_reason: Optional[str] = None
    """Finish reason (usually in final chunk)."""


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider configuration. Override in subclasses for custom validation."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'bedrock', 'openai')."""
        ...

    @property
    @abstractmethod
    def default_rpm_limit(self) -> int:
        """Return the default requests per minute limit for this provider."""
        ...

    @property
    @abstractmethod
    def default_tpm_limit(self) -> int:
        """Return the default tokens per minute limit for this provider."""
        ...

    @property
    def rpm_limit(self) -> int:
        """Return the configured or default RPM limit."""
        return self.config.rpm_limit or self.default_rpm_limit

    @property
    def tpm_limit(self) -> int:
        """Return the configured or default TPM limit."""
        return self.config.tpm_limit or self.default_tpm_limit

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion for the given messages.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0-1.0).
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with generated content and metadata.

        Raises:
            RateLimitExceeded: If the provider's rate limit is hit.
            ProviderError: If the provider returns an error.
        """
        ...

    async def embed(self, text: str, **kwargs: Any) -> EmbeddingResponse:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed.
            **kwargs: Additional provider-specific parameters.

        Returns:
            EmbeddingResponse with embedding vector and metadata.

        Raises:
            NotImplementedError: If the provider doesn't support embeddings.
            RateLimitExceeded: If the provider's rate limit is hit.
            ProviderError: If the provider returns an error.
        """
        raise NotImplementedError(
            f"Provider {self.provider_name} does not support embeddings"
        )

    async def stream(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response.

        Yields chunks of the response as they are generated.
        Override in subclasses to provide streaming support.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Yields:
            StreamChunk objects with partial content.

        Raises:
            NotImplementedError: If the provider doesn't support streaming.
        """
        raise NotImplementedError(
            f"Provider {self.provider_name} does not support streaming"
        )
        # Make this an async generator (yield after raise to satisfy type checker)
        yield  # type: ignore  # pragma: no cover

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text.

        Uses improved heuristics for better accuracy across languages.
        Override in subclasses for model-specific counting (e.g., tiktoken).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        # Count words and characters
        words = len(text.split())
        chars = len(text)

        # Improved estimation:
        # - English: ~1.3 tokens per word, ~4 chars per token
        # - Code: More tokens due to symbols, ~3.5 chars per token
        # - CJK: Each character is roughly 1-2 tokens

        # Detect if text is primarily CJK (Chinese, Japanese, Korean)
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                        '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or
                        '\uac00' <= c <= '\ud7af')
        cjk_ratio = cjk_count / max(chars, 1)

        # Detect if text is code-like (high punctuation ratio)
        code_chars = sum(1 for c in text if c in '{}[]()<>;:=+-*/%&|!@#$^~`')
        code_ratio = code_chars / max(chars, 1)

        if cjk_ratio > 0.3:
            # CJK text: ~1.5 tokens per CJK character + word-based for rest
            cjk_tokens = int(cjk_count * 1.5)
            other_tokens = max(1, (chars - cjk_count) // 4)
            return cjk_tokens + other_tokens
        elif code_ratio > 0.1:
            # Code: ~3.5 chars per token
            return max(1, int(chars / 3.5))
        else:
            # Regular text: blend of word and char based estimation
            word_estimate = int(words * 1.3)
            char_estimate = chars // 4
            # Use weighted average, favoring word estimate for short text
            if words < 20:
                return max(1, word_estimate)
            return max(1, (word_estimate + char_estimate) // 2)

    def estimate_request_tokens(self, messages: list[Message], max_tokens: int) -> int:
        """Estimate total tokens for a completion request.

        Args:
            messages: Input messages.
            max_tokens: Maximum output tokens.

        Returns:
            Estimated total tokens (input + output).
        """
        input_tokens = sum(self.estimate_tokens(m.content) for m in messages)
        # Add overhead for message structure (~4 tokens per message)
        input_tokens += len(messages) * 4
        return input_tokens + max_tokens

    async def health_check(self) -> bool:
        """Check if the provider is healthy and reachable.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Simple health check with minimal tokens
            await self.complete(
                messages=[Message(role="user", content="hi")],
                max_tokens=1,
                temperature=0.0,
            )
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        region_str = f", region={self.config.region}" if self.config.region else ""
        return f"{self.__class__.__name__}(model={self.config.model}{region_str})"
