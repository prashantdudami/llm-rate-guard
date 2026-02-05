# LLM Rate Guard Examples

This directory contains examples demonstrating the features of LLM Rate Guard.

## Prerequisites

```bash
# Install the library with all providers
pip install llm-rate-guard[all]

# Or install with specific providers
pip install llm-rate-guard[openai]
pip install llm-rate-guard[bedrock]
```

## Examples

### 1. Basic Usage (`01_basic_usage.py`)
Simple getting started example with a single provider.

```bash
export OPENAI_API_KEY=sk-...
python 01_basic_usage.py
```

### 2. Multi-Region Routing (`02_multi_region.py`)
Distribute requests across AWS regions to multiply rate limits.

```bash
# Uses AWS credential chain (env vars, ~/.aws/credentials, IAM role)
python 02_multi_region.py
```

### 3. Multi-Provider Failover (`03_multi_provider.py`)
Configure multiple providers with weighted routing and automatic failover.

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python 03_multi_provider.py
```

### 4. Streaming Responses (`04_streaming.py`)
Stream responses in real-time for better UX.

```bash
export OPENAI_API_KEY=sk-...
python 04_streaming.py
```

### 5. Multi-Tenancy (`05_multi_tenancy.py`)
Track requests by tenant with quotas and cost attribution.

```bash
export OPENAI_API_KEY=sk-...
python 05_multi_tenancy.py
```

### 6. Middleware (`06_middleware.py`)
Add custom pre/post-processing with middleware.

```bash
export OPENAI_API_KEY=sk-...
python 06_middleware.py
```

### 7. Monitoring (`07_monitoring.py`)
Health checks, metrics, and OpenTelemetry integration.

```bash
export OPENAI_API_KEY=sk-...
python 07_monitoring.py
```

### 8. Advanced Configuration (`08_advanced_config.py`)
Full production-ready configuration with all options.

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python 08_advanced_config.py
```

### 9. Environment Configuration (`09_env_config.py`)
Configure entirely from environment variables (for containers/K8s).

```bash
python 09_env_config.py
```

### 10. Sync API (`10_sync_api.py`)
Use the library from synchronous code - scripts, notebooks, Django/Flask.

```bash
export OPENAI_API_KEY=sk-...
python 10_sync_api.py
```

### 11. Batch Processing (`11_batch_processing.py`)
Process multiple prompts efficiently with controlled concurrency.

```bash
export OPENAI_API_KEY=sk-...
python 11_batch_processing.py
```

### 12. Distributed Cache (`12_distributed_cache.py`)
Use Redis or Memcached for multi-node deployments.

```bash
python 12_distributed_cache.py
```

## Quick Reference

### Provider Types
- `openai` - OpenAI API
- `anthropic` - Anthropic API
- `bedrock` - AWS Bedrock
- `azure_openai` - Azure OpenAI
- `vertex` - Google Vertex AI

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | `...` |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint | `https://...` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `my-project` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `...` |

### Configuration via Environment

```bash
# Providers (JSON format)
export LLM_RATE_GUARD_PROVIDERS='[{"type": "openai", "model": "gpt-4"}]'

# Cache
export LLM_RATE_GUARD_CACHE_ENABLED=true
export LLM_RATE_GUARD_CACHE_TTL=3600

# Rate limits
export LLM_RATE_GUARD_GLOBAL_RPM_LIMIT=1000

# Failover
export LLM_RATE_GUARD_FAILOVER_ENABLED=true
```

## Running Without API Keys

Examples will fail when making actual API calls without valid keys. To test the configuration and flow without real API calls, you can mock the providers in your own tests.
