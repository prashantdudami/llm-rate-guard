"""LLM Provider implementations."""

from llm_rate_guard.providers.base import (
    BaseProvider,
    CompletionResponse,
    EmbeddingResponse,
    Message,
    Usage,
)

# Lazy imports for providers to avoid requiring all SDKs
def get_provider_class(provider_type: str) -> type[BaseProvider]:
    """Get provider class by type name.

    Args:
        provider_type: Provider type (bedrock, openai, azure_openai, anthropic, vertex).

    Returns:
        Provider class.

    Raises:
        ValueError: If provider type is unknown.
    """
    if provider_type == "bedrock":
        from llm_rate_guard.providers.bedrock import BedrockProvider
        return BedrockProvider
    elif provider_type == "openai":
        from llm_rate_guard.providers.openai_provider import OpenAIProvider
        return OpenAIProvider
    elif provider_type == "azure_openai":
        from llm_rate_guard.providers.azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider
    elif provider_type == "anthropic":
        from llm_rate_guard.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider
    elif provider_type == "vertex":
        from llm_rate_guard.providers.vertex import VertexAIProvider
        return VertexAIProvider
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


__all__ = [
    "BaseProvider",
    "CompletionResponse",
    "EmbeddingResponse",
    "Message",
    "Usage",
    "get_provider_class",
]
