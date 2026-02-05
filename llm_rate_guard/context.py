"""Request context for multi-tenancy and request tracking."""

from typing import Any, Awaitable, Callable, Optional, Union
import contextvars
import uuid


class RequestContext:
    """Context for a single request.

    Use this to track tenant information, labels, and metadata
    for cost attribution and observability.

    Example:
        ```python
        ctx = RequestContext(
            tenant_id="tenant-123",
            user_id="user-456",
            labels={"project": "chatbot", "environment": "production"},
        )

        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            context=ctx,
        )
        ```
    """

    __slots__ = (
        "request_id", "tenant_id", "user_id", "labels",
        "metadata", "priority_override", "cost_center"
    )

    def __init__(
        self,
        request_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        priority_override: Optional[int] = None,
        cost_center: Optional[str] = None,
    ):
        """Initialize request context.

        Args:
            request_id: Unique request ID. Auto-generated if not provided.
            tenant_id: Tenant identifier for multi-tenant deployments.
            user_id: User identifier for per-user tracking.
            labels: Custom labels for categorization (e.g., project, team).
            metadata: Additional metadata to associate with the request.
            priority_override: Priority override (1-10, higher = more important).
            cost_center: Cost center for billing attribution.
        """
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.labels = labels or {}
        self.metadata = metadata or {}
        self.priority_override = priority_override
        self.cost_center = cost_center

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "labels": self.labels,
            "metadata": self.metadata,
            "priority_override": self.priority_override,
            "cost_center": self.cost_center,
        }

    def __repr__(self) -> str:
        parts = [f"request_id={self.request_id!r}"]
        if self.tenant_id:
            parts.append(f"tenant_id={self.tenant_id!r}")
        if self.user_id:
            parts.append(f"user_id={self.user_id!r}")
        if self.labels:
            parts.append(f"labels={self.labels!r}")
        return f"RequestContext({', '.join(parts)})"


# Context variable for storing current request context
_current_context: contextvars.ContextVar[Optional[RequestContext]] = (
    contextvars.ContextVar("rate_guard_context", default=None)
)


def get_current_context() -> Optional[RequestContext]:
    """Get the current request context (if set)."""
    return _current_context.get()


def set_current_context(ctx: Optional[RequestContext]) -> contextvars.Token:
    """Set the current request context.

    Returns a token that can be used to reset the context.
    """
    return _current_context.set(ctx)


# Type for request/response interceptors
RequestData = dict[str, Any]
ResponseData = dict[str, Any]

# Pre-request middleware: receives request data, returns modified data or None to block
PreRequestMiddleware = Callable[
    [RequestData, Optional[RequestContext]],
    Union[RequestData, Awaitable[RequestData], None, Awaitable[None]],
]

# Post-request middleware: receives response data and context
PostRequestMiddleware = Callable[
    [ResponseData, Optional[RequestContext]],
    Union[None, Awaitable[None]],
]


class MiddlewareChain:
    """Chain of middleware functions for request/response processing.

    Middleware allows you to:
    - Log all requests
    - Modify requests before processing
    - Block requests based on custom rules
    - Track responses for monitoring

    Example:
        ```python
        async def log_requests(data, ctx):
            print(f"Request from tenant: {ctx.tenant_id if ctx else 'unknown'}")
            return data  # Pass through

        async def enforce_quota(data, ctx):
            if ctx and is_over_quota(ctx.tenant_id):
                return None  # Block request
            return data

        client.add_pre_middleware(log_requests)
        client.add_pre_middleware(enforce_quota)
        ```
    """

    def __init__(self):
        """Initialize empty middleware chain."""
        self._pre_middleware: list[PreRequestMiddleware] = []
        self._post_middleware: list[PostRequestMiddleware] = []

    def add_pre(self, middleware: PreRequestMiddleware) -> None:
        """Add pre-request middleware."""
        self._pre_middleware.append(middleware)

    def add_post(self, middleware: PostRequestMiddleware) -> None:
        """Add post-request middleware."""
        self._post_middleware.append(middleware)

    def remove_pre(self, middleware: PreRequestMiddleware) -> bool:
        """Remove pre-request middleware.

        Returns:
            True if middleware was found and removed.
        """
        try:
            self._pre_middleware.remove(middleware)
            return True
        except ValueError:
            return False

    def remove_post(self, middleware: PostRequestMiddleware) -> bool:
        """Remove post-request middleware.

        Returns:
            True if middleware was found and removed.
        """
        try:
            self._post_middleware.remove(middleware)
            return True
        except ValueError:
            return False

    async def process_pre(
        self,
        data: RequestData,
        ctx: Optional[RequestContext] = None,
    ) -> Optional[RequestData]:
        """Process request through pre-middleware chain.

        Args:
            data: Request data.
            ctx: Optional request context.

        Returns:
            Modified request data, or None if request should be blocked.
        """
        import asyncio

        current_data = data
        for middleware in self._pre_middleware:
            result = middleware(current_data, ctx)

            # Handle async middleware
            if asyncio.iscoroutine(result):
                result = await result

            if result is None:
                return None  # Request blocked

            current_data = result

        return current_data

    async def process_post(
        self,
        data: ResponseData,
        ctx: Optional[RequestContext] = None,
    ) -> None:
        """Process response through post-middleware chain.

        Args:
            data: Response data.
            ctx: Optional request context.
        """
        import asyncio

        for middleware in self._post_middleware:
            result = middleware(data, ctx)

            # Handle async middleware
            if asyncio.iscoroutine(result):
                await result

    def clear(self) -> None:
        """Clear all middleware."""
        self._pre_middleware.clear()
        self._post_middleware.clear()

    @property
    def pre_count(self) -> int:
        """Number of pre-request middleware."""
        return len(self._pre_middleware)

    @property
    def post_count(self) -> int:
        """Number of post-request middleware."""
        return len(self._post_middleware)


class QuotaManager:
    """Simple in-memory quota manager for per-tenant usage limits.

    Example:
        ```python
        quota = QuotaManager()
        quota.set_limit("tenant-123", tokens_per_day=1_000_000)

        # Check before request
        if not quota.check("tenant-123", estimated_tokens=500):
            raise Exception("Quota exceeded")

        # Record after request
        quota.record("tenant-123", tokens=500)
        ```
    """

    def __init__(self):
        """Initialize quota manager."""
        self._limits: dict[str, dict[str, int]] = {}
        self._usage: dict[str, dict[str, int]] = {}

    def set_limit(
        self,
        tenant_id: str,
        tokens_per_day: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        cost_per_day_usd: Optional[float] = None,
    ) -> None:
        """Set quota limits for a tenant.

        Args:
            tenant_id: Tenant identifier.
            tokens_per_day: Maximum tokens per day.
            requests_per_day: Maximum requests per day.
            cost_per_day_usd: Maximum cost per day in USD (stored as micro-cents).
        """
        limits: dict[str, int] = {}
        if tokens_per_day is not None:
            limits["tokens"] = tokens_per_day
        if requests_per_day is not None:
            limits["requests"] = requests_per_day
        if cost_per_day_usd is not None:
            limits["cost_micro"] = int(cost_per_day_usd * 1_000_000)

        self._limits[tenant_id] = limits

        # Initialize usage if not exists
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {"tokens": 0, "requests": 0, "cost_micro": 0}

    def check(
        self,
        tenant_id: str,
        tokens: int = 0,
        requests: int = 1,
        cost_usd: float = 0.0,
    ) -> bool:
        """Check if usage would exceed quota.

        Args:
            tenant_id: Tenant identifier.
            tokens: Estimated tokens to use.
            requests: Number of requests (usually 1).
            cost_usd: Estimated cost in USD.

        Returns:
            True if within quota, False if would exceed.
        """
        limits = self._limits.get(tenant_id)
        if not limits:
            return True  # No limits set

        usage = self._usage.get(tenant_id, {"tokens": 0, "requests": 0, "cost_micro": 0})

        if "tokens" in limits:
            if usage["tokens"] + tokens > limits["tokens"]:
                return False

        if "requests" in limits:
            if usage["requests"] + requests > limits["requests"]:
                return False

        if "cost_micro" in limits:
            cost_micro = int(cost_usd * 1_000_000)
            if usage["cost_micro"] + cost_micro > limits["cost_micro"]:
                return False

        return True

    def record(
        self,
        tenant_id: str,
        tokens: int = 0,
        requests: int = 1,
        cost_usd: float = 0.0,
    ) -> None:
        """Record usage for a tenant.

        Args:
            tenant_id: Tenant identifier.
            tokens: Tokens used.
            requests: Requests made.
            cost_usd: Cost in USD.
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {"tokens": 0, "requests": 0, "cost_micro": 0}

        self._usage[tenant_id]["tokens"] += tokens
        self._usage[tenant_id]["requests"] += requests
        self._usage[tenant_id]["cost_micro"] += int(cost_usd * 1_000_000)

    def get_usage(self, tenant_id: str) -> dict[str, Any]:
        """Get current usage for a tenant."""
        usage = self._usage.get(tenant_id, {"tokens": 0, "requests": 0, "cost_micro": 0})
        limits = self._limits.get(tenant_id, {})

        return {
            "tokens_used": usage["tokens"],
            "tokens_limit": limits.get("tokens"),
            "requests_used": usage["requests"],
            "requests_limit": limits.get("requests"),
            "cost_used_usd": usage["cost_micro"] / 1_000_000,
            "cost_limit_usd": limits.get("cost_micro", 0) / 1_000_000 if "cost_micro" in limits else None,
        }

    def reset(self, tenant_id: Optional[str] = None) -> None:
        """Reset usage counters.

        Args:
            tenant_id: Specific tenant to reset. If None, resets all.
        """
        if tenant_id:
            if tenant_id in self._usage:
                self._usage[tenant_id] = {"tokens": 0, "requests": 0, "cost_micro": 0}
        else:
            for tid in self._usage:
                self._usage[tid] = {"tokens": 0, "requests": 0, "cost_micro": 0}
