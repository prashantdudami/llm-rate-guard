# LLM Rate Guard

**Cloud-agnostic rate limit mitigation for LLM APIs**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM Rate Guard provides a unified interface for interacting with multiple LLM providers while automatically handling rate limits through:

- **Multi-region routing** - Distribute requests across regions (each with independent rate limits)
- **Multi-provider failover** - Automatically switch providers when rate limited
- **Token bucket rate limiting** - Client-side RPM/TPM enforcement
- **Circuit breaker pattern** - Automatic failure detection and recovery
- **Semantic caching** - Reduce API calls with intelligent response caching (with size limits)
- **Priority queuing** - Process critical requests first
- **Retry with backoff** - Graceful handling of transient failures
- **Streaming responses** - Real-time response streaming support
- **Multi-tenancy** - Request context with tenant/user/project tracking
- **Middleware** - Pre/post request interceptors for logging, quotas, modifications
- **Quota management** - Per-tenant token/request/cost limits
- **Cost estimation** - Estimate costs before sending requests
- **Latency percentiles** - SLA monitoring with p50/p90/p95/p99 tracking
- **Request timeouts** - Configurable per-provider request timeouts
- **Graceful shutdown** - Wait for in-flight requests before stopping
- **Environment configuration** - Configure via environment variables
- **Secure API key handling** - Keys never logged or exposed in repr
- **OpenTelemetry hooks** - Integrate with external monitoring systems
- **Sync wrappers** - Use from synchronous code without async/await
- **Batch processing** - Process multiple prompts with controlled concurrency
- **Distributed cache** - Pluggable cache backends (Redis, Memcached)
- **LangChain integration** - Drop-in replacement for ChatBedrock/ChatOpenAI
- **Standalone decorators** - `@rate_limited`, `@with_retry`, `@circuit_protected`
- **Serverless/Lambda support** - DynamoDB and Redis-backed rate limiting for stateless environments

## Supported Providers

| Provider | Completion | Embeddings | Status |
|----------|------------|------------|--------|
| AWS Bedrock | ✅ | ✅ | Full support |
| Azure OpenAI | ✅ | ✅ | Full support |
| Google Vertex AI | ✅ | ✅ | Full support |
| OpenAI | ✅ | ✅ | Full support |
| Anthropic | ✅ | ❌ | Completion only |

## Installation

```bash
# Core package (no provider dependencies)
pip install llm-rate-guard

# With specific providers
pip install llm-rate-guard[bedrock]      # AWS Bedrock
pip install llm-rate-guard[openai]       # OpenAI
pip install llm-rate-guard[azure]        # Azure OpenAI
pip install llm-rate-guard[vertex]       # Google Vertex AI
pip install llm-rate-guard[anthropic]    # Anthropic

# With advanced cost tracking
pip install llm-rate-guard[cost-tracking]  # llm-cost-guard integration

# With LangChain integration
pip install llm-rate-guard[langchain]

# With Redis (distributed rate limiting / caching)
pip install llm-rate-guard[redis]

# All providers + all integrations
pip install llm-rate-guard[all]
```

## Quick Start

### Basic Usage

```python
import asyncio
from llm_rate_guard import RateGuardClient, ProviderConfig

async def main():
    client = RateGuardClient(
        providers=[
            ProviderConfig(
                type="bedrock",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-east-1",
            ),
        ],
    )

    response = await client.complete([
        {"role": "user", "content": "What is the capital of France?"}
    ])

    print(response.content)
    # "The capital of France is Paris."

asyncio.run(main())
```

### Multi-Region for Higher Throughput

Each AWS region has independent rate limits. By configuring multiple regions, you multiply your effective capacity:

```python
from llm_rate_guard import RateGuardClient, ProviderConfig

client = RateGuardClient(
    providers=[
        ProviderConfig(
            type="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-east-1",  # 250 RPM
        ),
        ProviderConfig(
            type="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-west-2",  # +250 RPM
        ),
        ProviderConfig(
            type="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="eu-west-1",  # +250 RPM
        ),
    ],
    # Effective capacity: 750 RPM
)
```

### Multi-Provider Failover

Use multiple providers as fallbacks:

```python
from llm_rate_guard import RateGuardClient, ProviderConfig

client = RateGuardClient(
    providers=[
        # Primary: AWS Bedrock
        ProviderConfig(
            type="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-east-1",
        ),
        # Fallback 1: Azure OpenAI
        ProviderConfig(
            type="azure_openai",
            model="gpt-4",
            endpoint="https://myresource.openai.azure.com/",
            deployment_name="gpt-4-deployment",
        ),
        # Fallback 2: Direct Anthropic API
        ProviderConfig(
            type="anthropic",
            model="claude-3-sonnet-20240229",
        ),
    ],
    failover_enabled=True,
)
```

### Priority Queuing

Process critical requests first:

```python
from llm_rate_guard import RateGuardClient, Priority

async with RateGuardClient(providers=[...]) as client:
    # Critical request - processed first
    response = await client.complete(
        messages=[{"role": "user", "content": "Urgent!"}],
        priority=Priority.CRITICAL,
    )

    # Background request - processed when capacity available
    response = await client.complete(
        messages=[{"role": "user", "content": "Not urgent"}],
        priority=Priority.BACKGROUND,
    )
```

### Caching

Enable caching to reduce API calls for repeated queries:

```python
from llm_rate_guard import RateGuardClient, ProviderConfig

client = RateGuardClient(
    providers=[...],
    cache_enabled=True,
    cache_similarity_threshold=0.95,  # For semantic matching
)

# First call hits the API
response1 = await client.complete([{"role": "user", "content": "Hello"}])

# Second identical call served from cache
response2 = await client.complete([{"role": "user", "content": "Hello"}])
assert response2.cached == True
```

## Configuration

### Full Configuration Example

```python
from llm_rate_guard import (
    RateGuardClient,
    RateGuardConfig,
    ProviderConfig,
    CacheConfig,
    RetryConfig,
    CircuitBreakerConfig,
)

config = RateGuardConfig(
    providers=[
        ProviderConfig(
            type="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-east-1",
            rpm_limit=250,      # Override default
            tpm_limit=2000000,
            weight=2.0,         # Higher weight = more traffic
        ),
        ProviderConfig(
            type="openai",
            model="gpt-4-turbo",
            weight=1.0,
        ),
    ],
    
    # Retry configuration
    retry=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    ),
    
    # Cache configuration
    cache=CacheConfig(
        enabled=True,
        mode="exact",              # or "semantic" for embedding-based
        similarity_threshold=0.95,
        max_entries=10000,
        max_size_bytes=100_000_000,  # 100MB limit
        max_entry_size_bytes=1_000_000,  # 1MB per entry
        ttl_seconds=3600,
    ),
    
    # Circuit breaker configuration
    circuit_breaker=CircuitBreakerConfig(
        enabled=True,
        failure_threshold=5,
        success_threshold=2,
        half_open_timeout=30.0,
    ),
    
    # Global rate limiting
    global_rpm_limit=1000,
    global_tpm_limit=10000000,
    
    # Failover settings
    failover_enabled=True,
    cooldown_seconds=60.0,
    
    # Queue settings
    queue_enabled=True,
    max_queue_size=1000,
)

client = RateGuardClient(config=config)
```

### Environment Variables

Provider credentials can be set via environment variables:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...

# Google Vertex AI
export GOOGLE_CLOUD_PROJECT=my-project

# AWS Bedrock (uses standard AWS credential chain)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Configuration from Environment

You can configure the entire client from environment variables:

```bash
# JSON format
export LLM_RATE_GUARD_PROVIDERS='[{"type": "openai", "model": "gpt-4"}]'

# Or numbered format
export LLM_RATE_GUARD_PROVIDER_1_TYPE=openai
export LLM_RATE_GUARD_PROVIDER_1_MODEL=gpt-4
export LLM_RATE_GUARD_PROVIDER_1_API_KEY=sk-...

# Other settings
export LLM_RATE_GUARD_CACHE_ENABLED=true
export LLM_RATE_GUARD_CACHE_TTL=7200
export LLM_RATE_GUARD_FAILOVER_ENABLED=true
export LLM_RATE_GUARD_RETRY_MAX=5
```

```python
from llm_rate_guard import create_client_from_env

# Create client automatically from environment
client = create_client_from_env()
```

## Metrics & Monitoring

```python
client = RateGuardClient(providers=[...])

# After some requests...
metrics = client.get_metrics()

print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
print(f"Avg latency: {metrics.avg_latency_ms:.1f}ms")
print(f"Failovers: {metrics.failovers}")

# Latency percentiles (SLA monitoring)
metrics_dict = metrics.to_dict()
print(f"p50 latency: {metrics_dict['latency_p50_ms']:.1f}ms")
print(f"p95 latency: {metrics_dict['latency_p95_ms']:.1f}ms")
print(f"p99 latency: {metrics_dict['latency_p99_ms']:.1f}ms")

# Cost tracking
print(f"Total cost: ${metrics_dict['estimated_cost_usd']:.4f}")

# Provider-level stats
for stat in client.get_provider_stats():
    print(f"{stat['provider_id']}: {stat['total_requests']} requests")

# Cache stats
cache_stats = client.get_cache_stats()
print(f"Cache entries: {cache_stats['entries']}")
print(f"Cache size: {cache_stats['current_size_bytes'] / 1_000_000:.1f}MB")
```

### Cost Estimation

Estimate costs before sending requests:

```python
estimate = client.estimate_cost(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=1000,
)

print(f"Estimated cost: ${estimate['total_usd']:.4f}")
print(f"Input tokens: ~{estimate['estimated_input_tokens']}")
print(f"Output tokens: ~{estimate['estimated_output_tokens']}")
```

### Advanced Cost Tracking with llm-cost-guard

For production use, install [llm-cost-guard](https://github.com/prashantdeva/llm-cost-guard) for more accurate pricing, budget enforcement, and advanced features:

```bash
pip install llm-rate-guard[cost-tracking]
```

When installed, llm-rate-guard automatically uses llm-cost-guard for:
- **Up-to-date pricing** - Pricing data updated regularly
- **Budget enforcement** - Set limits with configurable actions
- **Hierarchical tracking** - Group related calls with spans
- **Multiple storage backends** - SQLite, Redis for persistence
- **Metrics export** - Prometheus, StatsD, OpenTelemetry

```python
from llm_rate_guard import LLM_COST_GUARD_AVAILABLE

# Check if llm-cost-guard is being used
metrics = client.get_metrics()
if metrics.using_llm_cost_guard:
    # Access advanced features
    tracker = metrics.cost_tracker.underlying_tracker
    # Use llm-cost-guard features like budget enforcement
```

### Health Check

Monitor client health for production deployments:

```python
health = await client.health_check()

if health["healthy"]:
    print("Client is healthy")
else:
    print(f"Client is degraded: {health['providers']['unhealthy']} unhealthy providers")

# Detailed health info
print(f"Active requests: {health['active_requests']}")
print(f"Cache hit rate: {health['cache']['hit_rate_pct']:.1f}%")
print(f"Queue size: {health['queue']['size']}")
```

### OpenTelemetry Integration

Add custom hooks for external monitoring:

```python
def opentelemetry_hook(metrics, event):
    # Send to your monitoring system
    span = tracer.start_span("llm_request")
    span.set_attribute("provider", event["provider"])
    span.set_attribute("latency_ms", event["latency_ms"])
    span.set_attribute("cached", event["cached"])
    span.set_attribute("cost_usd", event["estimated_cost_usd"])
    span.end()

client.get_metrics().add_hook(opentelemetry_hook)
```

## Streaming Responses

Stream responses in real-time for better UX:

```python
async for chunk in client.stream([
    {"role": "user", "content": "Write a poem about Python"}
]):
    print(chunk.content, end="", flush=True)
    if chunk.done:
        print(f"\n\nTokens used: {chunk.usage.total_tokens}")
```

## Sync API

Use the library from synchronous code (scripts, notebooks, Django views):

```python
from llm_rate_guard import RateGuardClient, ProviderConfig

client = RateGuardClient(
    providers=[ProviderConfig(type="openai", model="gpt-4o")]
)

# Synchronous completion - no async/await needed
response = client.complete_sync([
    {"role": "user", "content": "What is 2+2?"}
])
print(response.content)

# Synchronous embedding
embedding = client.embed_sync("Hello world")
print(f"Dimensions: {len(embedding.embedding)}")

# Synchronous health check
health = client.health_check_sync()
print(f"Healthy: {health['healthy']}")
```

## Batch Processing

Process multiple prompts efficiently with controlled concurrency:

```python
# Async batch processing
prompts = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is 3+3?"}],
    [{"role": "user", "content": "What is 4+4?"}],
]

# Process with max 5 concurrent requests
responses = await client.complete_batch(
    prompts,
    max_concurrency=5,  # Limit concurrent requests
    max_tokens=100,
    temperature=0.5,
)

for prompt, resp in zip(prompts, responses):
    print(f"{prompt[0]['content']} -> {resp.content}")

# Handle errors gracefully
responses = await client.complete_batch(
    prompts,
    return_exceptions=True,  # Don't raise, return Exception objects
)

for resp in responses:
    if isinstance(resp, Exception):
        print(f"Error: {resp}")
    else:
        print(resp.content)

# Sync version available too
responses = client.complete_batch_sync(prompts, max_concurrency=10)
```

## Distributed Cache

Use Redis or Memcached for multi-node deployments:

```python
from llm_rate_guard.cache_backends import create_backend, RedisBackend

# Create Redis backend
backend = create_backend(
    "redis",
    host="redis.example.com",
    port=6379,
    prefix="llm_cache:",
)

# Or with URL
backend = RedisBackend(url="redis://user:pass@host:6379/0")

# Or use Memcached
backend = create_backend("memcached", host="memcached.example.com")

# Backend operations
await backend.set("key", entry, ttl_seconds=3600)
entry = await backend.get("key")
await backend.delete("key")
count = await backend.size()
await backend.clear()
```

Available backends:
- `InMemoryBackend` - Default, single-process (no external dependencies)
- `RedisBackend` - Multi-node, requires `redis` package
- `MemcachedBackend` - Multi-node, requires `aiomcache` package

## LangChain Integration

Drop-in replacement for ChatBedrock, ChatOpenAI, or any LangChain chat model. No need to rewrite chains or agents:

```python
# Before: Direct LangChain (no rate limiting)
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id="anthropic.claude-3-sonnet")

# After: One-line swap adds rate limiting, caching, failover
from llm_rate_guard.integrations.langchain import RateGuardChatModel

llm = RateGuardChatModel(client=rate_guard_client)

# All existing chains, agents, and prompts work unchanged
chain = LLMChain(llm=llm, prompt=my_prompt)
result = chain.run("Hello!")
```

Also provides `RateGuardEmbeddings` for vector operations and `RateGuardCallbackHandler` for monitoring existing chains.

## Standalone Decorators

Use individual components without the full client. Add rate limiting, retry, or circuit breaker to any existing function:

```python
from llm_rate_guard import rate_limited, with_retry, circuit_protected

@rate_limited(rpm=250, tpm=2_000_000)
@with_retry(max_retries=3, retryable_exceptions=(ConnectionError,))
@circuit_protected(failure_threshold=5)
def call_bedrock(prompt):
    # Your existing code - unchanged
    return bedrock_client.invoke_model(...)
```

Or use the `SyncRateLimiter` directly:

```python
from llm_rate_guard import SyncRateLimiter

limiter = SyncRateLimiter(rpm=250, tpm=2_000_000)
limiter.acquire(estimated_tokens=500)  # Blocks until capacity available
response = bedrock_client.invoke(...)
```

## Serverless / Lambda Support

Rate limiting that survives cold starts using external state:

```python
from llm_rate_guard.serverless import DynamoDBRateLimiter, lambda_rate_limited

# State persists in DynamoDB across Lambda invocations
limiter = DynamoDBRateLimiter(table_name="rate-limits", rpm=250, tpm=2_000_000)

@lambda_rate_limited(limiter)
def handler(event, context):
    response = bedrock.invoke_model(...)
    return {"statusCode": 200, "body": response}
```

Also available: `RedisRateLimiter` for Redis-backed distributed rate limiting.

## Multi-Tenancy & Request Context

Track requests by tenant, user, or project for cost attribution:

```python
from llm_rate_guard import RequestContext

# Create context with tenant info
ctx = RequestContext(
    tenant_id="acme-corp",
    user_id="user-123",
    labels={"project": "chatbot", "environment": "production"},
    cost_center="engineering",
)

response = await client.complete(
    messages=[{"role": "user", "content": "Hello!"}],
    context=ctx,
)

# Access context in middleware or hooks for cost attribution
```

## Middleware

Add custom pre/post-processing to requests:

```python
# Log all requests
async def log_requests(data, ctx):
    print(f"Request from {ctx.tenant_id if ctx else 'unknown'}")
    return data  # Pass through

# Block requests from over-quota tenants
async def enforce_quota(data, ctx):
    if ctx and is_over_quota(ctx.tenant_id):
        return None  # Block request
    return data

# Modify requests
async def add_system_prompt(data, ctx):
    messages = data["messages"]
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": "Be helpful."})
    return data

client.add_pre_middleware(log_requests)
client.add_pre_middleware(enforce_quota)
client.add_pre_middleware(add_system_prompt)

# Post-request logging
async def log_usage(data, ctx):
    print(f"Used {data['usage']['total_tokens']} tokens")

client.add_post_middleware(log_usage)
```

## Quota Management

Built-in quota manager for per-tenant limits:

```python
from llm_rate_guard import QuotaManager

quota = QuotaManager()

# Set limits per tenant
quota.set_limit(
    "tenant-123",
    tokens_per_day=1_000_000,
    requests_per_day=10_000,
    cost_per_day_usd=100.0,
)

# Use with middleware to enforce
async def enforce_quota(data, ctx):
    if ctx and not quota.check(ctx.tenant_id or "", requests=1):
        return None  # Block - over quota
    return data

client.add_pre_middleware(enforce_quota)
client.set_quota_manager(quota)

# After each request, record usage
async def record_usage(data, ctx):
    if ctx and ctx.tenant_id:
        quota.record(
            ctx.tenant_id,
            tokens=data["usage"]["total_tokens"],
            cost_usd=data.get("estimated_cost_usd", 0),
        )

client.add_post_middleware(record_usage)

# Check usage
usage = quota.get_usage("tenant-123")
print(f"Tokens: {usage['tokens_used']}/{usage['tokens_limit']}")
```

## Rate Limits by Provider

Default rate limits (can be overridden in config):

| Provider | Default RPM | Default TPM |
|----------|-------------|-------------|
| AWS Bedrock | 250 | 2,000,000 |
| Azure OpenAI | 60 | 40,000 |
| Google Vertex AI | 60 | 1,000,000 |
| OpenAI | 500 | 150,000 |
| Anthropic | 50 | 40,000 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RateGuardClient                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     Semantic Cache        │
                    │   (Optional, in-memory)   │
                    └─────────────┬─────────────┘
                                  │ Cache Miss
                    ┌─────────────▼─────────────┐
                    │    Priority Queue         │
                    │  (Critical > Normal > Low)│
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Token Bucket Limiter    │
                    │    (RPM + TPM buckets)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Multi-Provider Router   │
                    │  (Weighted, health-aware) │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        │             │           │           │             │
        ▼             ▼           ▼           ▼             ▼
   ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌──────────┐
   │ Bedrock │  │ Azure   │ │ Vertex  │ │ OpenAI  │  │Anthropic │
   │us-east-1│  │ OpenAI  │ │   AI    │ │         │  │          │
   └─────────┘  └─────────┘ └─────────┘ └─────────┘  └──────────┘
```

## Graceful Shutdown

The client supports graceful shutdown, waiting for in-flight requests:

```python
async with RateGuardClient(providers=[...]) as client:
    # Make requests...
    response = await client.complete([...])

# On exit, waits for active requests to complete

# Or manual shutdown with timeout
await client.stop(graceful=True, timeout=30.0)
```

## Security

The library implements several security best practices:

- **API keys are never logged** - Uses `SecretStr` for API keys
- **Safe repr** - Provider configs don't expose secrets in string representation
- **Input validation** - Message length and count limits prevent abuse
- **Configuration validation** - Catches typos with `extra="forbid"`

## Examples

See the [examples/](examples/) directory for complete working examples:

| Example | Description |
|---------|-------------|
| [01_basic_usage.py](examples/01_basic_usage.py) | Simple getting started |
| [02_multi_region.py](examples/02_multi_region.py) | Multi-region routing |
| [03_multi_provider.py](examples/03_multi_provider.py) | Multi-provider failover |
| [04_streaming.py](examples/04_streaming.py) | Streaming responses |
| [05_multi_tenancy.py](examples/05_multi_tenancy.py) | Multi-tenant setup |
| [06_middleware.py](examples/06_middleware.py) | Custom middleware |
| [07_monitoring.py](examples/07_monitoring.py) | Metrics & monitoring |
| [08_advanced_config.py](examples/08_advanced_config.py) | Full configuration |
| [09_env_config.py](examples/09_env_config.py) | Environment config |
| [10_sync_api.py](examples/10_sync_api.py) | Synchronous API usage |
| [11_batch_processing.py](examples/11_batch_processing.py) | Batch processing |
| [12_distributed_cache.py](examples/12_distributed_cache.py) | Redis/Memcached cache |
| [13_langchain_integration.py](examples/13_langchain_integration.py) | LangChain drop-in |
| [14_standalone_components.py](examples/14_standalone_components.py) | Decorators & standalone |
| [15_serverless_lambda.py](examples/15_serverless_lambda.py) | Lambda/serverless |

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=llm_rate_guard

# Run with verbose output
pytest -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [llm-cost-guard](https://github.com/prashantdeva/llm-cost-guard) - Cost tracking and budget enforcement for LLM applications
