"""Async priority queue for request management."""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class Priority(IntEnum):
    """Request priority levels.

    Lower values = higher priority.
    """

    CRITICAL = 0
    """Critical priority - processed immediately."""

    HIGH = 1
    """High priority - processed before normal requests."""

    NORMAL = 2
    """Normal priority - standard processing."""

    LOW = 3
    """Low priority - processed when no higher priority requests pending."""

    BACKGROUND = 4
    """Background priority - processed only when queue is otherwise empty."""


@dataclass(order=True)
class PrioritizedItem(Generic[T]):
    """A prioritized item in the queue."""

    priority: int
    """Priority level (lower = higher priority)."""

    timestamp: float = field(compare=True)
    """Submission timestamp (for FIFO within same priority)."""

    item: T = field(compare=False)
    """The actual item."""

    future: asyncio.Future[Any] = field(compare=False, repr=False)
    """Future to resolve when item is processed."""

    id: str = field(default="", compare=False)
    """Optional request identifier."""


@dataclass
class QueueStats:
    """Queue statistics."""

    total_submitted: int = 0
    """Total requests submitted."""

    total_processed: int = 0
    """Total requests processed."""

    total_cancelled: int = 0
    """Total requests cancelled."""

    total_errors: int = 0
    """Total requests that errored."""

    current_size: int = 0
    """Current queue size."""

    by_priority: dict[str, int] = field(default_factory=dict)
    """Requests processed by priority level."""

    avg_wait_time_ms: float = 0.0
    """Average wait time in queue (ms)."""

    _wait_times: list[float] = field(default_factory=list, repr=False)


class PriorityQueue(Generic[T]):
    """Async priority queue for managing LLM requests.

    Features:
    - Priority-based processing (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
    - FIFO ordering within same priority
    - Configurable max size with rejection
    - Cancellation support
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        process_fn: Optional[Callable[[T], Any]] = None,
    ):
        """Initialize the priority queue.

        Args:
            max_size: Maximum queue size. New requests rejected when full.
            process_fn: Optional processing function for automatic processing.
        """
        self.max_size = max_size
        self._process_fn = process_fn
        self._queue: list[PrioritizedItem[T]] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._processing = False
        self._processor_task: Optional[asyncio.Task[None]] = None
        self.stats = QueueStats()

    async def submit(
        self,
        item: T,
        priority: Priority = Priority.NORMAL,
        request_id: str = "",
    ) -> asyncio.Future[Any]:
        """Submit an item to the queue.

        Args:
            item: The item to queue.
            priority: Priority level.
            request_id: Optional request identifier.

        Returns:
            Future that resolves when the item is processed.

        Raises:
            asyncio.QueueFull: If queue is at max capacity.
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                raise asyncio.QueueFull(f"Queue full (max size: {self.max_size})")

            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

            prioritized = PrioritizedItem(
                priority=priority.value,
                timestamp=time.time(),
                item=item,
                future=future,
                id=request_id,
            )

            heapq.heappush(self._queue, prioritized)
            self.stats.total_submitted += 1
            self.stats.current_size = len(self._queue)

            # Notify processor
            self._not_empty.notify()

        return future

    async def get(self) -> PrioritizedItem[T]:
        """Get the next item from the queue.

        Blocks until an item is available.

        Returns:
            The next prioritized item.
        """
        async with self._not_empty:
            while not self._queue:
                await self._not_empty.wait()

            item = heapq.heappop(self._queue)
            self.stats.current_size = len(self._queue)

            # Track wait time
            wait_time = (time.time() - item.timestamp) * 1000
            self.stats._wait_times.append(wait_time)
            if len(self.stats._wait_times) > 1000:
                self.stats._wait_times.pop(0)
            self.stats.avg_wait_time_ms = (
                sum(self.stats._wait_times) / len(self.stats._wait_times)
            )

            return item

    async def get_nowait(self) -> Optional[PrioritizedItem[T]]:
        """Get the next item without waiting.

        Returns:
            The next item, or None if queue is empty.
        """
        async with self._lock:
            if not self._queue:
                return None

            item = heapq.heappop(self._queue)
            self.stats.current_size = len(self._queue)
            return item

    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request by ID.

        Args:
            request_id: The request ID to cancel.

        Returns:
            True if request was found and cancelled.
        """
        async with self._lock:
            for i, item in enumerate(self._queue):
                if item.id == request_id:
                    # Cancel the future
                    if not item.future.done():
                        item.future.cancel()

                    # Remove from queue (need to rebuild heap)
                    self._queue.pop(i)
                    heapq.heapify(self._queue)

                    self.stats.total_cancelled += 1
                    self.stats.current_size = len(self._queue)
                    return True

        return False

    async def start_processor(self) -> None:
        """Start the background processor task."""
        if self._process_fn is None:
            raise ValueError("No process_fn provided")

        if self._processing:
            return

        self._processing = True
        self._processor_task = asyncio.create_task(self._process_loop())

    async def stop_processor(self) -> None:
        """Stop the background processor task."""
        self._processing = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._processing:
            try:
                item = await self.get()

                if item.future.cancelled():
                    continue

                try:
                    result = await self._process_fn(item.item)  # type: ignore
                    item.future.set_result(result)
                    self.stats.total_processed += 1

                    # Track by priority
                    priority_name = Priority(item.priority).name
                    self.stats.by_priority[priority_name] = (
                        self.stats.by_priority.get(priority_name, 0) + 1
                    )

                except Exception as e:
                    item.future.set_exception(e)
                    self.stats.total_errors += 1

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue processing
                pass

    def get_queue_snapshot(self) -> list[dict[str, Any]]:
        """Get a snapshot of current queue contents.

        Returns:
            List of pending items with metadata.
        """
        return [
            {
                "id": item.id,
                "priority": Priority(item.priority).name,
                "wait_time_ms": (time.time() - item.timestamp) * 1000,
            }
            for item in sorted(self._queue)
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_submitted": self.stats.total_submitted,
            "total_processed": self.stats.total_processed,
            "total_cancelled": self.stats.total_cancelled,
            "total_errors": self.stats.total_errors,
            "current_size": self.stats.current_size,
            "max_size": self.max_size,
            "by_priority": self.stats.by_priority,
            "avg_wait_time_ms": self.stats.avg_wait_time_ms,
            "is_processing": self._processing,
        }

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self._queue) >= self.max_size
