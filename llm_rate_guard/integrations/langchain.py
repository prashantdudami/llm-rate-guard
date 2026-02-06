"""LangChain integration for llm-rate-guard.

Provides LangChain-compatible wrappers so users can add rate limiting,
caching, and failover to existing LangChain chains without rewriting code.

Example:
    ```python
    # Swap one line to add rate limiting to existing LangChain code
    from llm_rate_guard.integrations.langchain import RateGuardChatModel

    llm = RateGuardChatModel(client=rate_guard_client)
    chain = LLMChain(llm=llm, prompt=my_prompt)  # Unchanged
    ```

Requires: pip install llm-rate-guard[langchain]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.callbacks.base import BaseCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Stubs so the module can be imported without langchain
    class BaseChatModel:  # type: ignore[no-redef]
        pass

    class Embeddings:  # type: ignore[no-redef]
        pass

    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


def _require_langchain() -> None:
    """Raise ImportError if langchain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain integration requires 'langchain-core' package. "
            "Install with: pip install llm-rate-guard[langchain]"
        )


def _messages_to_dicts(messages: List[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages to dicts for RateGuardClient."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = getattr(msg, "role", "user")
        result.append({"role": role, "content": msg.content})
    return result


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import threading

        result = None
        exception = None

        def _run() -> None:
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        if exception:
            raise exception
        return result
    else:
        return asyncio.run(coro)


class RateGuardChatModel(BaseChatModel):  # type: ignore[misc]
    """LangChain ChatModel backed by RateGuardClient.

    Drop-in replacement for ChatBedrock, ChatOpenAI, etc. Provides
    rate limiting, caching, failover, and all other llm-rate-guard features.

    Example:
        ```python
        from llm_rate_guard import RateGuardClient, ProviderConfig
        from llm_rate_guard.integrations.langchain import RateGuardChatModel

        client = RateGuardClient(providers=[
            ProviderConfig(type="bedrock", model="anthropic.claude-3-sonnet", region="us-east-1"),
            ProviderConfig(type="bedrock", model="anthropic.claude-3-sonnet", region="us-west-2"),
        ])

        llm = RateGuardChatModel(client=client)

        # Use with any LangChain chain
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=my_prompt)
        result = chain.run("Hello!")
        ```
    """

    client: Any = None  # RateGuardClient
    max_tokens: int = 1024
    temperature: float = 0.7
    skip_cache: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: Any, **kwargs: Any):
        _require_langchain()
        super().__init__(client=client, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "rate-guard"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "providers": self.client.total_providers,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response synchronously."""
        msg_dicts = _messages_to_dicts(messages)

        response = self.client.complete_sync(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            skip_cache=kwargs.get("skip_cache", self.skip_cache),
        )

        message = AIMessage(content=response.content)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "model": response.model,
                "provider": response.provider,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "cached": response.cached,
                "latency_ms": response.latency_ms,
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": response.model,
                "provider": response.provider,
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response asynchronously."""
        msg_dicts = _messages_to_dicts(messages)

        response = await self.client.complete(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            skip_cache=kwargs.get("skip_cache", self.skip_cache),
        )

        message = AIMessage(content=response.content)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "model": response.model,
                "provider": response.provider,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "cached": response.cached,
                "latency_ms": response.latency_ms,
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": response.model,
                "provider": response.provider,
            },
        )


class RateGuardEmbeddings(Embeddings):  # type: ignore[misc]
    """LangChain Embeddings backed by RateGuardClient.

    Example:
        ```python
        from llm_rate_guard.integrations.langchain import RateGuardEmbeddings

        embeddings = RateGuardEmbeddings(client=rate_guard_client)

        # Use with any LangChain component that needs embeddings
        vectors = embeddings.embed_documents(["Hello", "World"])
        query_vector = embeddings.embed_query("Search query")
        ```
    """

    def __init__(self, client: Any, model: Optional[str] = None, **kwargs: Any):
        _require_langchain()
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        results = []
        for text in texts:
            response = self.client.embed_sync(text=text, model=self.model)
            results.append(response.embedding)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.embed_sync(text=text, model=self.model)
        return response.embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents asynchronously."""
        results = []
        for text in texts:
            response = await self.client.embed(text=text, model=self.model)
            results.append(response.embedding)
        return results

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query asynchronously."""
        response = await self.client.embed(text=text, model=self.model)
        return response.embedding


class RateGuardCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback handler for monitoring with llm-rate-guard.

    Adds rate-guard metrics to existing LangChain chains without
    requiring any code changes. Just add the callback.

    Example:
        ```python
        from llm_rate_guard.integrations.langchain import RateGuardCallbackHandler

        handler = RateGuardCallbackHandler(client=rate_guard_client)

        # Add to any existing LangChain chain
        result = chain.run("Hello!", callbacks=[handler])

        # Check metrics
        print(handler.get_stats())
        ```
    """

    def __init__(self, client: Any, **kwargs: Any):
        _require_langchain()
        super().__init__(**kwargs)
        self.client = client
        self._total_calls = 0
        self._total_tokens = 0
        self._errors = 0

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Called when LLM starts."""
        self._total_calls += 1
        logger.debug(
            f"LLM call #{self._total_calls} | "
            f"Rate Guard healthy providers: {self.client.healthy_providers}/{self.client.total_providers}"
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM completes."""
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens = usage.get("total_tokens", 0)
            self._total_tokens += tokens

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when LLM errors."""
        self._errors += 1
        logger.warning(f"LLM error #{self._errors}: {error}")

    def get_stats(self) -> dict[str, Any]:
        """Get callback statistics."""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "errors": self._errors,
            "client_metrics": self.client.get_metrics(),
        }
