"""Tests for LangChain integration (mocked, no langchain dependency required)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_rate_guard.providers.base import CompletionResponse, EmbeddingResponse, Usage


class TestRateGuardChatModelWithoutLangChain:
    """Tests for RateGuardChatModel when LangChain is NOT installed."""

    def test_import_without_langchain(self):
        """Test module imports without langchain."""
        # Should import without error
        from llm_rate_guard.integrations import langchain as lc_mod
        assert hasattr(lc_mod, "RateGuardChatModel")
        assert hasattr(lc_mod, "RateGuardEmbeddings")
        assert hasattr(lc_mod, "RateGuardCallbackHandler")

    def test_require_langchain_error(self):
        """Test _require_langchain raises when not available."""
        from llm_rate_guard.integrations.langchain import _require_langchain, LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            with pytest.raises(ImportError, match="langchain-core"):
                _require_langchain()


class TestMessagesConversion:
    """Tests for message format conversion."""

    def test_messages_to_dicts(self):
        """Test converting LangChain messages to dicts."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            from llm_rate_guard.integrations.langchain import _messages_to_dicts

            messages = [
                SystemMessage(content="You are helpful"),
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ]

            result = _messages_to_dicts(messages)
            assert len(result) == 3
            assert result[0] == {"role": "system", "content": "You are helpful"}
            assert result[1] == {"role": "user", "content": "Hello"}
            assert result[2] == {"role": "assistant", "content": "Hi there"}
        except ImportError:
            pytest.skip("langchain-core not installed")


class TestRateGuardChatModel:
    """Tests for RateGuardChatModel with LangChain installed."""

    def test_generate_sync(self):
        """Test synchronous generation."""
        try:
            from langchain_core.messages import HumanMessage
            from llm_rate_guard.integrations.langchain import RateGuardChatModel

            mock_client = MagicMock()
            mock_client.complete_sync.return_value = CompletionResponse(
                content="Hello!",
                model="test-model",
                usage=Usage(input_tokens=5, output_tokens=10),
                provider="test",
                cached=False,
                latency_ms=100.0,
            )

            llm = RateGuardChatModel(client=mock_client)
            result = llm.invoke([HumanMessage(content="Hi")])

            assert result.content == "Hello!"
            mock_client.complete_sync.assert_called_once()
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_llm_type(self):
        """Test _llm_type property."""
        try:
            from llm_rate_guard.integrations.langchain import RateGuardChatModel

            mock_client = MagicMock()
            mock_client.total_providers = 2
            llm = RateGuardChatModel(client=mock_client)
            assert llm._llm_type == "rate-guard"
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_identifying_params(self):
        """Test _identifying_params property."""
        try:
            from llm_rate_guard.integrations.langchain import RateGuardChatModel

            mock_client = MagicMock()
            mock_client.total_providers = 3
            llm = RateGuardChatModel(
                client=mock_client,
                max_tokens=512,
                temperature=0.5,
            )

            params = llm._identifying_params
            assert params["max_tokens"] == 512
            assert params["temperature"] == 0.5
            assert params["providers"] == 3
        except ImportError:
            pytest.skip("langchain-core not installed")


class TestRateGuardEmbeddings:
    """Tests for RateGuardEmbeddings."""

    def test_embed_documents(self):
        """Test embedding documents."""
        try:
            from llm_rate_guard.integrations.langchain import RateGuardEmbeddings

            mock_client = MagicMock()
            mock_client.embed_sync.side_effect = [
                EmbeddingResponse(
                    embedding=[0.1, 0.2],
                    model="test",
                    usage=Usage(input_tokens=3, output_tokens=0),
                ),
                EmbeddingResponse(
                    embedding=[0.3, 0.4],
                    model="test",
                    usage=Usage(input_tokens=4, output_tokens=0),
                ),
            ]

            embeddings = RateGuardEmbeddings(client=mock_client)
            result = embeddings.embed_documents(["Hello", "World"])

            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]
            assert mock_client.embed_sync.call_count == 2
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_embed_query(self):
        """Test embedding a single query."""
        try:
            from llm_rate_guard.integrations.langchain import RateGuardEmbeddings

            mock_client = MagicMock()
            mock_client.embed_sync.return_value = EmbeddingResponse(
                embedding=[0.5, 0.6, 0.7],
                model="test",
                usage=Usage(input_tokens=3, output_tokens=0),
            )

            embeddings = RateGuardEmbeddings(client=mock_client)
            result = embeddings.embed_query("Search query")

            assert result == [0.5, 0.6, 0.7]
        except ImportError:
            pytest.skip("langchain-core not installed")


class TestRateGuardCallbackHandler:
    """Tests for RateGuardCallbackHandler."""

    def test_callback_tracking(self):
        """Test callback tracks calls and errors."""
        try:
            from llm_rate_guard.integrations.langchain import RateGuardCallbackHandler

            mock_client = MagicMock()
            mock_client.healthy_providers = 2
            mock_client.total_providers = 3
            mock_client.get_metrics.return_value = {"total_requests": 10}

            handler = RateGuardCallbackHandler(client=mock_client)

            # Simulate LLM calls
            handler.on_llm_start({}, ["prompt 1"])
            handler.on_llm_start({}, ["prompt 2"])
            handler.on_llm_end(MagicMock(llm_output={}))
            handler.on_llm_error(ValueError("test error"))

            stats = handler.get_stats()
            assert stats["total_calls"] == 2
            assert stats["errors"] == 1
        except ImportError:
            pytest.skip("langchain-core not installed")
