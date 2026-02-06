#!/usr/bin/env python3
"""Example: Serverless / AWS Lambda Support

Demonstrates using llm-rate-guard in AWS Lambda where in-memory state
is lost on cold starts. Uses external backends (DynamoDB, Redis) for
persistent rate limiting.

Features shown:
- DynamoDBRateLimiter for AWS-native rate limiting
- RedisRateLimiter for Redis-backed rate limiting
- @lambda_rate_limited decorator
- ServerlessConfig for Lambda-optimized settings
"""


def main():
    """Run serverless examples."""

    print("=" * 60)
    print("LLM Rate Guard - Serverless / Lambda Support")
    print("=" * 60)

    # =========================================================================
    # Example 1: DynamoDB Rate Limiter
    # =========================================================================
    print("\n1. DynamoDB Rate Limiter (AWS-Native)")
    print("-" * 40)

    print("""
    from llm_rate_guard.serverless import DynamoDBRateLimiter

    # DynamoDB table with pk (String) partition key and ttl TTL attribute
    limiter = DynamoDBRateLimiter(
        table_name="llm-rate-limits",
        rpm=250,
        tpm=2_000_000,
        region="us-east-1",
    )

    # Use in Lambda handler
    def handler(event, context):
        # Rate limit persists across cold starts via DynamoDB
        limiter.acquire(estimated_tokens=500)
        response = bedrock.invoke_model(...)
        return {"statusCode": 200, "body": response}
    """)

    # =========================================================================
    # Example 2: Redis Rate Limiter
    # =========================================================================
    print("\n2. Redis Rate Limiter (Distributed)")
    print("-" * 40)

    print("""
    from llm_rate_guard.serverless import RedisRateLimiter

    limiter = RedisRateLimiter(
        host="redis.example.com",
        port=6379,
        rpm=250,
        tpm=2_000_000,
    )

    # Or with URL (ElastiCache, Redis Cloud, etc.)
    limiter = RedisRateLimiter(
        url="redis://user:pass@redis.example.com:6379/0",
        rpm=250,
        tpm=2_000_000,
    )

    # Uses Lua scripts for atomic token bucket operations
    if limiter.try_acquire(estimated_tokens=500):
        response = bedrock.invoke_model(...)
    else:
        return {"statusCode": 429, "body": "Rate limited"}
    """)

    # =========================================================================
    # Example 3: Lambda Decorator
    # =========================================================================
    print("\n3. @lambda_rate_limited Decorator")
    print("-" * 40)

    print("""
    from llm_rate_guard.serverless import DynamoDBRateLimiter, lambda_rate_limited

    # Initialize outside handler (reused across warm invocations)
    limiter = DynamoDBRateLimiter(
        table_name="llm-rate-limits",
        rpm=250,
        tpm=2_000_000,
    )

    @lambda_rate_limited(limiter, estimated_tokens=1000, timeout=10)
    def handler(event, context):
        # Rate limiting happens automatically
        # Returns 429 if timeout exceeded
        prompt = event.get("prompt", "Hello")
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({"messages": [{"role": "user", "content": prompt}]}),
        )
        return {
            "statusCode": 200,
            "body": json.loads(response["body"].read()),
        }
    """)

    # =========================================================================
    # Example 4: Combining with Standalone Decorators
    # =========================================================================
    print("\n4. Lambda + Retry + Circuit Breaker")
    print("-" * 40)

    print("""
    from llm_rate_guard.serverless import DynamoDBRateLimiter, lambda_rate_limited
    from llm_rate_guard.standalone import with_retry, circuit_protected

    limiter = DynamoDBRateLimiter(table_name="rate-limits", rpm=250)

    @lambda_rate_limited(limiter)
    def handler(event, context):
        return invoke_bedrock(event["prompt"])

    @with_retry(max_retries=2, retryable_exceptions=(Exception,))
    @circuit_protected(failure_threshold=5, recovery_timeout=60)
    def invoke_bedrock(prompt):
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet",
            body=json.dumps({"messages": [{"role": "user", "content": prompt}]}),
        )
        return json.loads(response["body"].read())
    """)

    # =========================================================================
    # Example 5: ServerlessConfig
    # =========================================================================
    print("\n5. ServerlessConfig - Lambda-Optimized Settings")
    print("-" * 40)

    from llm_rate_guard.serverless import ServerlessConfig

    config = ServerlessConfig.for_lambda(
        cache_enabled=False,    # Lost on cold start
        queue_enabled=False,    # No persistent workers
        failover_enabled=True,  # Still useful
        max_retries=2,          # Low for Lambda timeout
    )

    print("  Lambda-optimized config:")
    for key, value in config.items():
        if key == "notes":
            print("  Notes:")
            for note in value:
                print(f"    - {note}")
        else:
            print(f"    {key}: {value}")

    # =========================================================================
    # Example 6: DynamoDB Table Setup
    # =========================================================================
    print("\n6. DynamoDB Table Setup (CloudFormation)")
    print("-" * 40)

    print("""
    # CloudFormation template for the rate limiting table:
    Resources:
      RateLimitTable:
        Type: AWS::DynamoDB::Table
        Properties:
          TableName: llm-rate-limits
          BillingMode: PAY_PER_REQUEST
          AttributeDefinitions:
            - AttributeName: pk
              AttributeType: S
          KeySchema:
            - AttributeName: pk
              KeyType: HASH
          TimeToLiveSpecification:
            AttributeName: ttl
            Enabled: true

    # Or with AWS CDK:
    from aws_cdk import aws_dynamodb as dynamodb

    table = dynamodb.Table(self, "RateLimitTable",
        table_name="llm-rate-limits",
        partition_key=dynamodb.Attribute(
            name="pk",
            type=dynamodb.AttributeType.STRING,
        ),
        billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        time_to_live_attribute="ttl",
    )
    """)

    # =========================================================================
    # Example 7: Multi-Tenant Lambda
    # =========================================================================
    print("\n7. Multi-Tenant Lambda Rate Limiting")
    print("-" * 40)

    print("""
    from llm_rate_guard.serverless import DynamoDBRateLimiter

    def handler(event, context):
        tenant_id = event["requestContext"]["authorizer"]["tenantId"]

        # Per-tenant rate limiter (keyed by tenant)
        limiter = DynamoDBRateLimiter(
            table_name="rate-limits",
            rpm=100,            # Per-tenant limit
            tpm=500_000,
            key_prefix=f"tenant#{tenant_id}",
        )

        if not limiter.try_acquire(estimated_tokens=1000):
            return {
                "statusCode": 429,
                "body": json.dumps({
                    "error": f"Rate limit exceeded for tenant {tenant_id}"
                }),
            }

        # Process request...
    """)

    print("\n" + "=" * 60)
    print("Serverless examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
