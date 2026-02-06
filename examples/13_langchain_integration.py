#!/usr/bin/env python3
"""Example: LangChain Integration

Demonstrates using llm-rate-guard as a drop-in replacement in LangChain
chains and agents. No need to rewrite existing LangChain code.

Features shown:
- RateGuardChatModel as LangChain LLM
- RateGuardEmbeddings for vector operations
- RateGuardCallbackHandler for monitoring
- Existing chains work unchanged

Requires: pip install llm-rate-guard[langchain]
"""

import asyncio


def main():
    """Run LangChain integration examples."""

    print("=" * 60)
    print("LLM Rate Guard - LangChain Integration")
    print("=" * 60)

    # =========================================================================
    # Example 1: Drop-in Replacement for ChatBedrock
    # =========================================================================
    print("\n1. Drop-in Replacement for ChatBedrock/ChatOpenAI")
    print("-" * 40)

    print("""
    # BEFORE: Direct LangChain usage (no rate limiting)
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet")

    # AFTER: One-line swap adds rate limiting, caching, failover
    from llm_rate_guard import RateGuardClient, ProviderConfig
    from llm_rate_guard.integrations.langchain import RateGuardChatModel

    client = RateGuardClient(providers=[
        ProviderConfig(type="bedrock", model="anthropic.claude-3-sonnet", region="us-east-1"),
        ProviderConfig(type="bedrock", model="anthropic.claude-3-sonnet", region="us-west-2"),
    ])

    llm = RateGuardChatModel(client=client)

    # Everything else stays the same
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(template="Tell me about {topic}", input_variables=["topic"])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run("Python programming")
    """)

    # =========================================================================
    # Example 2: Multi-Region with LangChain
    # =========================================================================
    print("\n2. Multi-Region Failover with LangChain")
    print("-" * 40)

    print("""
    # Triple your rate limits with multi-region
    client = RateGuardClient(providers=[
        ProviderConfig(type="bedrock", model="claude-3-sonnet", region="us-east-1"),
        ProviderConfig(type="bedrock", model="claude-3-sonnet", region="us-west-2"),
        ProviderConfig(type="bedrock", model="claude-3-sonnet", region="eu-west-1"),
    ])

    llm = RateGuardChatModel(client=client)

    # Your existing agent works unchanged, but now with 3x rate limits
    from langchain.agents import initialize_agent, AgentType

    agent = initialize_agent(
        tools=my_tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )
    result = agent.run("Analyze this data...")
    """)

    # =========================================================================
    # Example 3: Embeddings with Rate Limiting
    # =========================================================================
    print("\n3. Embeddings with Rate Limiting")
    print("-" * 40)

    print("""
    from llm_rate_guard.integrations.langchain import RateGuardEmbeddings

    embeddings = RateGuardEmbeddings(client=client)

    # Use with FAISS, Chroma, or any vector store
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_texts(
        texts=["Document 1", "Document 2", "Document 3"],
        embedding=embeddings,
    )

    # Rate-limited similarity search
    results = vectorstore.similarity_search("query", k=3)
    """)

    # =========================================================================
    # Example 4: Callback Handler for Monitoring
    # =========================================================================
    print("\n4. Callback Handler (Zero Code Changes)")
    print("-" * 40)

    print("""
    from llm_rate_guard.integrations.langchain import RateGuardCallbackHandler

    handler = RateGuardCallbackHandler(client=client)

    # Add to ANY existing chain without changing code
    result = existing_chain.run(
        "Hello!",
        callbacks=[handler],  # Just add this
    )

    # Check metrics
    stats = handler.get_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Errors: {stats['errors']}")
    """)

    # =========================================================================
    # Example 5: Async LangChain Usage
    # =========================================================================
    print("\n5. Async LangChain Usage")
    print("-" * 40)

    print("""
    # RateGuardChatModel supports both sync and async
    llm = RateGuardChatModel(client=client, max_tokens=1024, temperature=0.7)

    # Sync
    result = llm.invoke("Hello!")

    # Async
    result = await llm.ainvoke("Hello!")

    # Batch (uses rate limiting automatically)
    results = llm.batch(["Q1", "Q2", "Q3"])
    """)

    print("\n" + "=" * 60)
    print("LangChain integration examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
