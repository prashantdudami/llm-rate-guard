"""Unified RateGuardClient - main entry point for LLM Rate Guard."""

import asyncio
import hashlib
import time
import uuid
from typing import Any, AsyncIterator, Optional, Union

from llm_rate_guard.cache import SemanticCache
from llm_rate_guard.config import CacheConfig, ProviderConfig, RateGuardConfig, RetryConfig
from llm_rate_guard.context import (
    MiddlewareChain,
    PostRequestMiddleware,
    PreRequestMiddleware,
    QuotaManager,
    RequestContext,
    set_current_context,
)
from llm_rate_guard.exceptions import AllProvidersExhausted, ConfigurationError, RateGuardError
from llm_rate_guard.logging import get_logger, LogContext
from llm_rate_guard.metrics import Metrics
from llm_rate_guard.providers.base import (
    CompletionResponse,
    EmbeddingResponse,
    Message,
    StreamChunk,
)
from llm_rate_guard.queue import Priority, PriorityQueue
from llm_rate_guard.rate_limiter import MultiProviderRateLimiter, RateLimiter
from llm_rate_guard.retry import RetryHandler
from llm_rate_guard.router import MultiRouter

# Error messages
_ERR_SHUTTING_DOWN = "Client is shutting down"


class RateGuardClient:
    """Unified client for rate-limit-aware LLM interactions.

    Combines all rate limit mitigation strategies:
    - Multi-region/multi-provider routing with automatic failover
    - Token bucket rate limiting (RPM + TPM)
    - Semantic caching
    - Priority queuing
    - Retry with exponential backoff

    Example:
        ```python
        from llm_rate_guard import RateGuardClient, ProviderConfig

        client = RateGuardClient(
            providers=[
                ProviderConfig(
                    type="bedrock",
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                    region="us-east-1",
                ),
                ProviderConfig(
                    type="bedrock",
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                    region="us-west-2",
                ),
            ],
            cache_enabled=True,
        )

        response = await client.complete([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """

    def __init__(
        self,
        providers: Optional[list[ProviderConfig]] = None,
        config: Optional[RateGuardConfig] = None,
        *,
        cache_enabled: bool = True,
        cache_similarity_threshold: float = 0.95,
        failover_enabled: bool = True,
        retry_max_attempts: int = 3,
    ):
        """Initialize the RateGuardClient.

        You can either provide a full RateGuardConfig or use the convenience
        parameters to configure common options.

        Args:
            providers: List of provider configurations.
            config: Full configuration object (overrides other params if provided).
            cache_enabled: Enable response caching.
            cache_similarity_threshold: Threshold for semantic cache matching.
            failover_enabled: Enable automatic failover to next provider.
            retry_max_attempts: Maximum retry attempts on failure.
        """
        # Build config from params or use provided config
        if config is not None:
            self.config = config
        elif providers is not None:
            self.config = RateGuardConfig(
                providers=providers,
                cache=CacheConfig(
                    enabled=cache_enabled,
                    similarity_threshold=cache_similarity_threshold,
                ),
                failover_enabled=failover_enabled,
                retry=RetryConfig(max_retries=retry_max_attempts),
            )
        else:
            raise ConfigurationError(
                "Either 'providers' or 'config' must be provided",
                field="providers",
            )

        # Initialize components
        self._rate_limiters = MultiProviderRateLimiter()
        self._router = MultiRouter(self.config, self._rate_limiters)
        self._cache = SemanticCache(self.config.cache)
        self._retry_handler = RetryHandler(self.config.retry)
        self._metrics = Metrics()

        # Global rate limiter (if configured)
        self._global_limiter: Optional[RateLimiter] = None
        if self.config.global_rpm_limit or self.config.global_tpm_limit:
            self._global_limiter = RateLimiter(
                rpm_limit=self.config.global_rpm_limit or 10000,
                tpm_limit=self.config.global_tpm_limit or 100_000_000,
            )

        # Priority queue (if enabled)
        self._queue: Optional[PriorityQueue[dict[str, Any]]] = None
        if self.config.queue_enabled:
            self._queue = PriorityQueue(
                max_size=self.config.max_queue_size,
                process_fn=self._process_queued_request,
            )

        # State
        self._started = False
        self._shutting_down = False
        self._active_requests = 0
        self._active_requests_lock = asyncio.Lock()
        self._shutdown_event: Optional[asyncio.Event] = None
        self._requests_complete_event: Optional[asyncio.Event] = None

        # Middleware and quotas
        self._middleware = MiddlewareChain()
        self._quota_manager: Optional[QuotaManager] = None

    async def start(self) -> None:
        """Start background processing (queue processor, etc.)."""
        if self._started:
            return

        if self._queue:
            await self._queue.start_processor()

        self._shutdown_event = asyncio.Event()
        self._requests_complete_event = asyncio.Event()
        self._requests_complete_event.set()  # Initially no requests active
        self._started = True
        self._shutting_down = False

    async def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Stop background processing.

        Args:
            graceful: If True, wait for active requests to complete.
            timeout: Maximum time to wait for graceful shutdown (seconds).
        """
        if not self._started:
            return

        logger = get_logger()
        self._shutting_down = True

        if graceful and self._active_requests > 0:
            logger.info(
                f"Graceful shutdown: waiting for {self._active_requests} active requests"
            )
            # Wait for active requests with timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_requests_complete(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Shutdown timeout: {self._active_requests} requests still active"
                )

        if self._queue:
            await self._queue.stop_processor()

        self._started = False
        self._shutting_down = False

        if self._shutdown_event:
            self._shutdown_event.set()

    async def _wait_for_requests_complete(self) -> None:
        """Wait for all active requests to complete."""
        if self._requests_complete_event:
            await self._requests_complete_event.wait()
        else:
            # Fallback to polling if event not initialized
            while self._active_requests > 0:
                await asyncio.sleep(0.1)

    async def _increment_active_requests(self) -> None:
        """Thread-safe increment of active request counter."""
        async with self._active_requests_lock:
            if self._active_requests == 0 and self._requests_complete_event:
                self._requests_complete_event.clear()
            self._active_requests += 1

    async def _decrement_active_requests(self) -> None:
        """Thread-safe decrement of active request counter."""
        async with self._active_requests_lock:
            self._active_requests -= 1
            if self._active_requests == 0 and self._requests_complete_event:
                self._requests_complete_event.set()

    @property
    def is_shutting_down(self) -> bool:
        """Check if client is shutting down."""
        return self._shutting_down

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_requests

    async def __aenter__(self) -> "RateGuardClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def _validate_and_convert_messages(
        self, messages: Union[list[dict[str, str]], list[Message]]
    ) -> list[Message]:
        """Validate and convert messages to Message objects.

        Args:
            messages: Input messages.

        Returns:
            List of validated Message objects.

        Raises:
            ConfigurationError: If messages are invalid.
        """
        if not messages:
            raise ConfigurationError("Messages list cannot be empty", field="messages")

        if len(messages) > self.config.max_messages_per_request:
            raise ConfigurationError(
                f"Too many messages: {len(messages)} "
                f"(max: {self.config.max_messages_per_request})",
                field="messages",
            )

        result = []
        for i, msg in enumerate(messages):
            try:
                if isinstance(msg, Message):
                    content = msg.content
                    role = msg.role
                elif isinstance(msg, dict):
                    if "role" not in msg or "content" not in msg:
                        raise ConfigurationError(
                            f"Message {i} missing required keys 'role' and 'content'",
                            field="messages",
                        )
                    content = msg["content"]
                    role = msg["role"]
                else:
                    raise ConfigurationError(
                        f"Invalid message type at index {i}: {type(msg).__name__}",
                        field="messages",
                    )

                # Validate content length
                if len(content) > self.config.max_message_length:
                    raise ConfigurationError(
                        f"Message {i} content too long: {len(content)} chars "
                        f"(max: {self.config.max_message_length})",
                        field="messages",
                    )

                # Validate role
                if role not in ("system", "user", "assistant"):
                    raise ConfigurationError(
                        f"Invalid role '{role}' at index {i}. "
                        "Must be 'system', 'user', or 'assistant'",
                        field="messages",
                    )

                result.append(Message(role=role, content=content))  # type: ignore

            except ConfigurationError:
                raise
            except Exception as e:
                raise ConfigurationError(
                    f"Error processing message {i}: {e}",
                    field="messages",
                )

        return result

    def _get_cache_key(self, messages: list[Message], model: str = "") -> str:
        """Generate cache key from messages.

        Uses hashing for long content to avoid memory issues with large keys.
        """
        content = "|".join(f"{m.role}:{m.content}" for m in messages)
        full_key = f"{model}:{content}"

        # Hash if key is too long (> 1KB)
        if len(full_key) > 1024:
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            return f"hash:{key_hash}"

        return full_key

    async def complete(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        priority: Priority = Priority.NORMAL,
        skip_cache: bool = False,
        request_id: str = "",
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion with rate limit mitigation.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0-1.0).
            priority: Request priority (for queue ordering).
            skip_cache: Skip cache lookup/storage for this request.
            request_id: Optional request identifier for tracking.
            context: Optional request context for multi-tenancy and tracking.
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with generated content and metadata.

        Raises:
            AllProvidersExhausted: If all providers are rate limited.
            RateLimitExceeded: If rate limits cannot be satisfied.
            ConfigurationError: If messages are invalid or blocked by middleware.
        """
        logger = get_logger()

        # Check if shutting down
        if self._shutting_down:
            raise ConfigurationError(_ERR_SHUTTING_DOWN, field="client")

        # Create or use provided context
        if context is None:
            context = RequestContext(request_id=request_id or None)
        elif not request_id:
            request_id = context.request_id

        if not request_id:
            request_id = context.request_id

        # Set context variable for middleware access
        _token = set_current_context(context)  # noqa: F841

        log_ctx = LogContext(
            request_id=request_id,
            extra={
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
            } if context.tenant_id or context.user_id else None,
        )

        # Track active requests (thread-safe)
        await self._increment_active_requests()

        try:
            # Run pre-request middleware
            request_data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "priority": priority,
                "skip_cache": skip_cache,
                "kwargs": kwargs,
            }

            processed_data = await self._middleware.process_pre(request_data, context)
            if processed_data is None:
                raise ConfigurationError(
                    "Request blocked by middleware",
                    field="middleware",
                )

            # Use potentially modified data
            messages = processed_data.get("messages", messages)
            max_tokens = processed_data.get("max_tokens", max_tokens)
            temperature = processed_data.get("temperature", temperature)
            skip_cache = processed_data.get("skip_cache", skip_cache)

            response = await self._do_complete(
                messages, max_tokens, temperature, priority,
                skip_cache, request_id, log_ctx, logger, context=context, **kwargs
            )

            # Run post-request middleware
            response_data = {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": response.latency_ms,
                "cached": response.cached,
            }
            await self._middleware.process_post(response_data, context)

            return response

        finally:
            await self._decrement_active_requests()
            # Reset context
            set_current_context(None)

    async def _do_complete(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        max_tokens: int,
        temperature: float,
        priority: Priority,
        skip_cache: bool,
        request_id: str,
        log_ctx: LogContext,
        logger: Any,
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Internal completion implementation."""
        # Validate and convert messages
        converted_messages = self._validate_and_convert_messages(messages)
        start_time = time.perf_counter()

        # Check cache first (unless skipped)
        if not skip_cache and self.config.cache.enabled:
            cache_key = self._get_cache_key(converted_messages)
            cached_response = await self._cache.get(cache_key)

            if cached_response is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000

                self._metrics.record_request(
                    success=True,
                    provider="cache",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    cached=True,
                )

                logger.debug(f"Cache hit {log_ctx} latency={latency_ms:.1f}ms")

                return CompletionResponse(
                    content=cached_response,
                    model="cached",
                    provider="cache",
                    latency_ms=latency_ms,
                    cached=True,
                )

        # Apply global rate limit if configured
        rate_limit_wait_ms = 0.0
        if self._global_limiter:
            estimated_tokens = sum(len(m.content) // 4 for m in converted_messages) + max_tokens
            wait_time = await self._global_limiter.acquire(estimated_tokens)
            rate_limit_wait_ms = wait_time * 1000
            if wait_time > 0:
                logger.debug(f"Rate limit wait {log_ctx} wait_ms={rate_limit_wait_ms:.1f}")

        # Route request to available provider
        try:
            response = await self._router.route(
                messages=converted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Cache the response
            if not skip_cache and self.config.cache.enabled:
                cache_key = self._get_cache_key(converted_messages)
                await self._cache.set(
                    prompt=cache_key,
                    response=response.content,
                    model=response.model,
                )

            # Record metrics
            self._metrics.record_request(
                success=True,
                provider=response.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=response.latency_ms,
                cached=False,
                rate_limit_wait_ms=rate_limit_wait_ms,
                model=response.model,
            )

            logger.debug(
                f"Request complete {log_ctx} provider={response.provider} "
                f"tokens={response.usage.total_tokens} latency={response.latency_ms:.1f}ms"
            )

            return response

        except RateGuardError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._metrics.record_request(
                success=False,
                provider=getattr(e, "provider", "unknown"),
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cached=False,
                rate_limit_wait_ms=rate_limit_wait_ms,
            )

            logger.warning(f"Request failed {log_ctx} error={e.message}")
            raise

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._metrics.record_request(
                success=False,
                provider="unknown",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cached=False,
                rate_limit_wait_ms=rate_limit_wait_ms,
            )

            logger.error(f"Unexpected error {log_ctx} error={e}")
            raise

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed.
            model: Optional embedding model to use.
            request_id: Optional request identifier for tracking.
            **kwargs: Additional provider-specific parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        logger = get_logger()

        # Check if shutting down
        if self._shutting_down:
            raise ConfigurationError(_ERR_SHUTTING_DOWN, field="client")

        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        log_ctx = LogContext(request_id=request_id)

        # Track active requests
        await self._increment_active_requests()

        try:
            # Estimate tokens for rate limiting
            estimated_tokens = max(1, len(text) // 4)

            # Apply global rate limit if configured
            if self._global_limiter:
                await self._global_limiter.acquire(estimated_tokens)

            attempted: list[str] = []

            # Use available providers that support embeddings
            for managed in self._router._providers:
                # Check health and circuit breaker
                if not managed.health.can_accept_request():
                    continue

                attempted.append(managed.provider_id)

                try:
                    # Acquire provider-specific rate limit
                    await self._router.rate_limiters.acquire(
                        managed.provider_id,
                        estimated_tokens,
                    )

                    response = await managed.provider.embed(text, model=model, **kwargs)

                    # Record success
                    managed.health.record_success(0.0)  # No latency for embeddings currently

                    logger.debug(
                        f"Embedding complete {log_ctx} provider={managed.provider_id}"
                    )

                    return response

                except NotImplementedError:
                    continue
                except Exception as e:
                    managed.health.record_failure(is_rate_limit=False)
                    logger.warning(
                        f"Embedding failed {log_ctx} provider={managed.provider_id} error={e}"
                    )
                    continue

            raise AllProvidersExhausted(
                message="No provider available for embeddings",
                attempted_providers=attempted,
            )

        finally:
            await self._decrement_active_requests()

    async def stream(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        request_id: str = "",
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response.

        Yields chunks of the response as they are generated, enabling
        real-time display of responses.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            request_id: Optional request identifier.
            context: Optional request context.
            **kwargs: Additional provider-specific parameters.

        Yields:
            StreamChunk objects with partial content.

        Raises:
            AllProvidersExhausted: If no provider supports streaming.
            ConfigurationError: If client is shutting down.

        Example:
            ```python
            async for chunk in client.stream(messages):
                print(chunk.content, end="", flush=True)
            ```
        """
        logger = get_logger()

        # Check if shutting down
        if self._shutting_down:
            raise ConfigurationError(_ERR_SHUTTING_DOWN, field="client")

        if context is None:
            context = RequestContext(request_id=request_id or None)
        if not request_id:
            request_id = context.request_id

        log_ctx = LogContext(request_id=request_id)

        # Validate messages
        converted_messages = self._validate_and_convert_messages(messages)

        # Track active requests
        await self._increment_active_requests()
        start_time = time.perf_counter()

        try:
            attempted: list[str] = []
            total_content = ""
            final_usage = None

            # Try available providers
            for managed in self._router._providers:
                if not managed.health.can_accept_request():
                    continue

                attempted.append(managed.provider_id)

                try:
                    # Acquire rate limit
                    estimated_tokens = sum(len(m.content) // 4 for m in converted_messages) + max_tokens
                    await self._router.rate_limiters.acquire(
                        managed.provider_id,
                        estimated_tokens,
                    )

                    # Stream from provider
                    async for chunk in managed.provider.stream(
                        messages=converted_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    ):
                        total_content += chunk.content
                        if chunk.usage:
                            final_usage = chunk.usage
                        yield chunk

                    # Record success
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    managed.health.record_success(latency_ms)

                    self._metrics.record_request(
                        success=True,
                        provider=managed.provider_id,
                        input_tokens=final_usage.input_tokens if final_usage else 0,
                        output_tokens=final_usage.output_tokens if final_usage else len(total_content) // 4,
                        latency_ms=latency_ms,
                        cached=False,
                        model=managed.config.model,
                    )

                    logger.debug(
                        f"Stream complete {log_ctx} provider={managed.provider_id}"
                    )
                    return

                except NotImplementedError:
                    continue
                except Exception as e:
                    managed.health.record_failure(is_rate_limit=False)
                    logger.warning(
                        f"Stream failed {log_ctx} provider={managed.provider_id} error={e}"
                    )
                    continue

            raise AllProvidersExhausted(
                message="No provider available for streaming",
                attempted_providers=attempted,
            )

        finally:
            await self._decrement_active_requests()

    async def _process_queued_request(self, request: dict[str, Any]) -> CompletionResponse:
        """Process a request from the queue."""
        return await self.complete(
            messages=request["messages"],
            max_tokens=request.get("max_tokens", 1024),
            temperature=request.get("temperature", 0.7),
            skip_cache=request.get("skip_cache", False),
            **request.get("kwargs", {}),
        )

    async def submit(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        priority: Priority = Priority.NORMAL,
        request_id: str = "",
        **kwargs: Any,
    ) -> asyncio.Future[CompletionResponse]:
        """Submit a request to the priority queue.

        Use this for high-throughput scenarios where you want
        requests to be processed in priority order.

        Args:
            messages: Conversation messages.
            priority: Request priority.
            request_id: Optional request identifier.
            **kwargs: Additional parameters for complete().

        Returns:
            Future that resolves to CompletionResponse.
        """
        if self._queue is None:
            raise ConfigurationError("Queue is not enabled", field="queue_enabled")

        request = {
            "messages": messages,
            **kwargs,
        }

        return await self._queue.submit(request, priority, request_id)

    def get_metrics(self) -> Metrics:
        """Get current metrics."""
        return self._metrics

    def get_provider_stats(self) -> list[dict[str, Any]]:
        """Get detailed stats for all providers."""
        return self._router.get_provider_stats()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def get_queue_stats(self) -> Optional[dict[str, Any]]:
        """Get queue statistics (if queue is enabled)."""
        if self._queue:
            return self._queue.get_stats()
        return None

    async def clear_cache(self) -> int:
        """Clear the response cache.

        Returns:
            Number of entries cleared.
        """
        return await self._cache.clear()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._metrics.reset()

    # Middleware methods

    def add_pre_middleware(self, middleware: PreRequestMiddleware) -> None:
        """Add pre-request middleware.

        Pre-middleware can modify requests or block them by returning None.

        Example:
            ```python
            async def log_request(data, ctx):
                print(f"Request from tenant: {ctx.tenant_id if ctx else 'unknown'}")
                return data  # Pass through

            client.add_pre_middleware(log_request)
            ```
        """
        self._middleware.add_pre(middleware)

    def add_post_middleware(self, middleware: PostRequestMiddleware) -> None:
        """Add post-request middleware.

        Post-middleware can log or process responses.

        Example:
            ```python
            async def log_response(data, ctx):
                print(f"Response tokens: {data['usage']['total_tokens']}")

            client.add_post_middleware(log_response)
            ```
        """
        self._middleware.add_post(middleware)

    def remove_pre_middleware(self, middleware: PreRequestMiddleware) -> bool:
        """Remove pre-request middleware.

        Returns:
            True if middleware was found and removed.
        """
        return self._middleware.remove_pre(middleware)

    def remove_post_middleware(self, middleware: PostRequestMiddleware) -> bool:
        """Remove post-request middleware.

        Returns:
            True if middleware was found and removed.
        """
        return self._middleware.remove_post(middleware)

    def clear_middleware(self) -> None:
        """Clear all middleware."""
        self._middleware.clear()

    # Quota management methods

    def set_quota_manager(self, manager: QuotaManager) -> None:
        """Set a quota manager for usage limits.

        Args:
            manager: QuotaManager instance.
        """
        self._quota_manager = manager

    @property
    def quota_manager(self) -> Optional[QuotaManager]:
        """Get the current quota manager."""
        return self._quota_manager

    @property
    def healthy_providers(self) -> int:
        """Number of currently healthy providers."""
        return self._router.healthy_provider_count

    @property
    def total_providers(self) -> int:
        """Total number of configured providers."""
        return self._router.total_provider_count

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the client and its components.

        Returns:
            Health status dictionary with component states.
        """
        healthy_count = self.healthy_providers
        total_count = self.total_providers
        queue_size = self._queue.size if self._queue else 0
        cache_entries = self._cache.stats.entries

        is_healthy = (
            healthy_count > 0 and
            not self._shutting_down
        )

        return {
            "healthy": is_healthy,
            "status": "ok" if is_healthy else "degraded",
            "started": self._started,
            "shutting_down": self._shutting_down,
            "active_requests": self._active_requests,
            "providers": {
                "total": total_count,
                "healthy": healthy_count,
                "unhealthy": total_count - healthy_count,
            },
            "cache": {
                "enabled": self.config.cache.enabled,
                "entries": cache_entries,
                "hit_rate_pct": self._cache.stats.hit_rate,
            },
            "queue": {
                "enabled": self.config.queue_enabled,
                "size": queue_size,
                "max_size": self.config.max_queue_size,
            },
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "success_rate_pct": self._metrics.success_rate,
                "avg_latency_ms": self._metrics.avg_latency_ms,
            },
        }

    def estimate_cost(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        max_tokens: int = 1024,
        model: str = "",
    ) -> dict[str, Any]:
        """Estimate the cost for a request before sending it.

        Args:
            messages: Input messages.
            max_tokens: Maximum tokens to generate.
            model: Model to estimate for (uses first provider if not specified).

        Returns:
            Cost estimate dictionary with input/output/total costs.
        """
        # Estimate input tokens (rough: 4 chars = 1 token)
        if not messages:
            return {
                "estimated_input_tokens": 0,
                "estimated_output_tokens": max_tokens,
                "input_usd": 0.0,
                "output_usd": 0.0,
                "total_usd": 0.0,
                "model": model,
            }

        total_chars = sum(
            len(m.content if isinstance(m, Message) else m.get("content", ""))
            for m in messages
        )
        estimated_input_tokens = max(1, total_chars // 4)

        # Use provided model or first provider's model
        if not model and self.config.providers:
            model = self.config.providers[0].model

        # Get cost estimate from tracker
        cost_estimate = self._metrics.cost_tracker.estimate_cost(
            model=model,
            input_tokens=estimated_input_tokens,
            output_tokens=max_tokens,
        )

        # For breakdown, check if tracker has pricing attribute (SimpleCostTracker)
        # LLMCostGuardAdapter doesn't expose pricing directly
        if hasattr(self._metrics.cost_tracker, 'pricing'):
            pricing = self._metrics.cost_tracker.pricing.get(
                "default", {"input": 1.0, "output": 2.0}
            )
            for key in self._metrics.cost_tracker.pricing:
                if key in model.lower():
                    pricing = self._metrics.cost_tracker.pricing[key]
                    break
            input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (max_tokens / 1_000_000) * pricing["output"]
        else:
            # Use proportional split for LLMCostGuardAdapter
            # Assume 1:2 ratio input:output if we can't determine pricing
            input_cost = cost_estimate * 0.33
            output_cost = cost_estimate * 0.67

        return {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": max_tokens,
            "input_usd": input_cost,
            "output_usd": output_cost,
            "total_usd": cost_estimate,
            "model": model,
        }

    # =========================================================================
    # Sync Wrappers
    # =========================================================================

    def complete_sync(
        self,
        messages: Union[list[dict[str, str]], list[Message]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        priority: Priority = Priority.NORMAL,
        skip_cache: bool = False,
        request_id: str = "",
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronous version of complete().

        Convenient for scripts, notebooks, and non-async code.
        Creates a new event loop if none is running.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            priority: Request priority.
            skip_cache: Skip cache lookup.
            request_id: Optional request identifier.
            context: Optional request context.
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with generated content.

        Example:
            ```python
            response = client.complete_sync([
                {"role": "user", "content": "Hello!"}
            ])
            print(response.content)
            ```
        """
        return self._run_sync(
            self.complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
                skip_cache=skip_cache,
                request_id=request_id,
                context=context,
                **kwargs,
            )
        )

    def embed_sync(
        self,
        text: str,
        model: Optional[str] = None,
        request_id: str = "",
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Synchronous version of embed().

        Args:
            text: Text to embed.
            model: Optional embedding model.
            request_id: Optional request identifier.
            **kwargs: Additional parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        return self._run_sync(
            self.embed(text=text, model=model, request_id=request_id, **kwargs)
        )

    def health_check_sync(self) -> dict[str, Any]:
        """Synchronous version of health_check()."""
        return self._run_sync(self.health_check())

    def _run_sync(self, coro: Any) -> Any:
        """Run a coroutine synchronously.

        Handles the case where an event loop may or may not be running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Event loop is running - use nest_asyncio pattern or thread
            import concurrent.futures
            import threading

            result = None
            exception = None

            def run_in_thread():
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

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            # No event loop running - simple case
            return asyncio.run(coro)

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def complete_batch(
        self,
        prompts: list[Union[list[dict[str, str]], list[Message]]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        skip_cache: bool = False,
        max_concurrency: int = 10,
        return_exceptions: bool = False,
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> list[Union[CompletionResponse, Exception]]:
        """Process multiple prompts in parallel with controlled concurrency.

        Efficiently handles batch processing while respecting rate limits.
        Uses a semaphore to control maximum concurrent requests.

        Args:
            prompts: List of message lists to process.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            skip_cache: Skip cache for all requests.
            max_concurrency: Maximum concurrent requests (default 10).
            return_exceptions: If True, exceptions are returned in results
                              instead of raised.
            context: Optional request context (shared across batch).
            **kwargs: Additional provider-specific parameters.

        Returns:
            List of CompletionResponse objects (or exceptions if return_exceptions=True).
            Order matches input prompts.

        Example:
            ```python
            prompts = [
                [{"role": "user", "content": "What is 2+2?"}],
                [{"role": "user", "content": "What is 3+3?"}],
                [{"role": "user", "content": "What is 4+4?"}],
            ]

            responses = await client.complete_batch(prompts, max_concurrency=5)

            for prompt, response in zip(prompts, responses):
                print(f"{prompt[0]['content']} -> {response.content}")
            ```
        """
        if not prompts:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_one(
            idx: int,
            messages: Union[list[dict[str, str]], list[Message]],
        ) -> tuple[int, Union[CompletionResponse, Exception]]:
            async with semaphore:
                try:
                    response = await self.complete(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        skip_cache=skip_cache,
                        context=context,
                        **kwargs,
                    )
                    return (idx, response)
                except Exception as e:
                    if return_exceptions:
                        return (idx, e)
                    raise

        # Create tasks for all prompts
        tasks = [process_one(i, prompt) for i, prompt in enumerate(prompts)]

        # Execute with controlled concurrency
        if return_exceptions:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = await asyncio.gather(*tasks)

        # Sort results back to original order
        sorted_results: list[Union[CompletionResponse, Exception]] = [None] * len(prompts)  # type: ignore
        for result in results:
            if isinstance(result, Exception):
                # This shouldn't happen with return_exceptions=True in gather
                raise result
            idx, response = result
            sorted_results[idx] = response

        return sorted_results

    def complete_batch_sync(
        self,
        prompts: list[Union[list[dict[str, str]], list[Message]]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        skip_cache: bool = False,
        max_concurrency: int = 10,
        return_exceptions: bool = False,
        context: Optional[RequestContext] = None,
        **kwargs: Any,
    ) -> list[Union[CompletionResponse, Exception]]:
        """Synchronous version of complete_batch().

        Args:
            prompts: List of message lists to process.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            skip_cache: Skip cache for all requests.
            max_concurrency: Maximum concurrent requests.
            return_exceptions: Return exceptions instead of raising.
            context: Optional request context.
            **kwargs: Additional parameters.

        Returns:
            List of CompletionResponse objects (or exceptions).
        """
        return self._run_sync(
            self.complete_batch(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                skip_cache=skip_cache,
                max_concurrency=max_concurrency,
                return_exceptions=return_exceptions,
                context=context,
                **kwargs,
            )
        )

    def __repr__(self) -> str:
        return (
            f"RateGuardClient("
            f"providers={self.total_providers}, "
            f"healthy={self.healthy_providers}, "
            f"cache={self.config.cache.enabled})"
        )
