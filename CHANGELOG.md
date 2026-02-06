# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-06

### Added

- **LangChain Integration** (`llm_rate_guard.integrations.langchain`)
  - `RateGuardChatModel` - Drop-in replacement for ChatBedrock/ChatOpenAI
  - `RateGuardEmbeddings` - LangChain-compatible embeddings with rate limiting
  - `RateGuardCallbackHandler` - Add rate-guard metrics to existing chains

- **Standalone Components** (`llm_rate_guard.standalone`)
  - `@rate_limited` decorator for any sync/async function
  - `@with_retry` decorator with exponential backoff
  - `@circuit_protected` decorator with configurable thresholds
  - `SyncRateLimiter` - Thread-safe sync rate limiter
  - `SyncCircuitBreaker` - Thread-safe sync circuit breaker

- **Serverless Support** (`llm_rate_guard.serverless`)
  - `RedisRateLimiter` - Distributed rate limiting via Redis Lua scripts
  - `DynamoDBRateLimiter` - AWS-native distributed rate limiting
  - `@lambda_rate_limited` decorator for Lambda handlers
  - `ServerlessConfig` - Configuration preset for Lambda

- **CI/CD** - GitHub Actions workflow for tests on Python 3.10-3.12
- **CHANGELOG.md** - Versioning discipline

## [0.1.0] - 2026-02-05

### Added

- Initial release
- Multi-region/multi-provider routing with automatic failover
- Token bucket rate limiting (RPM + TPM)
- Semantic caching with LRU eviction and size limits
- Circuit breaker pattern
- Priority queuing
- Retry with exponential backoff and jitter
- Streaming responses
- Multi-tenancy with request context
- Middleware (pre/post request hooks)
- Quota management per tenant
- Cost tracking with llm-cost-guard integration
- Sync API wrappers (complete_sync, embed_sync)
- Batch processing with controlled concurrency
- Distributed cache backends (Redis, Memcached)
- Environment variable configuration
- Comprehensive test suite (231 tests)
- Support for AWS Bedrock, Azure OpenAI, Google Vertex AI, OpenAI, Anthropic
