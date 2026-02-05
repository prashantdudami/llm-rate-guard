"""Tests for priority queue module."""

import asyncio
import pytest

from llm_rate_guard.queue import Priority, PriorityQueue, PrioritizedItem


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_ordering(self):
        """Priority values are ordered correctly."""
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW
        assert Priority.LOW < Priority.BACKGROUND


class TestPriorityQueue:
    """Tests for PriorityQueue class."""

    @pytest.mark.asyncio
    async def test_submit_and_get(self):
        """Basic submit and get."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        future = await queue.submit("item1", Priority.NORMAL)
        
        assert queue.size == 1
        assert not queue.is_empty

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Items are retrieved in priority order."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        # Submit in reverse priority order
        await queue.submit("low", Priority.LOW)
        await queue.submit("normal", Priority.NORMAL)
        await queue.submit("critical", Priority.CRITICAL)
        await queue.submit("high", Priority.HIGH)
        
        # Get should return in priority order
        item1 = await queue.get_nowait()
        item2 = await queue.get_nowait()
        item3 = await queue.get_nowait()
        item4 = await queue.get_nowait()
        
        assert item1 is not None and item1.item == "critical"
        assert item2 is not None and item2.item == "high"
        assert item3 is not None and item3.item == "normal"
        assert item4 is not None and item4.item == "low"

    @pytest.mark.asyncio
    async def test_fifo_within_priority(self):
        """Items with same priority are FIFO."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        await queue.submit("first", Priority.NORMAL)
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await queue.submit("second", Priority.NORMAL)
        await asyncio.sleep(0.01)
        await queue.submit("third", Priority.NORMAL)
        
        item1 = await queue.get_nowait()
        item2 = await queue.get_nowait()
        item3 = await queue.get_nowait()
        
        assert item1 is not None and item1.item == "first"
        assert item2 is not None and item2.item == "second"
        assert item3 is not None and item3.item == "third"

    @pytest.mark.asyncio
    async def test_max_size_rejection(self):
        """Queue rejects when full."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=2)
        
        await queue.submit("item1")
        await queue.submit("item2")
        
        with pytest.raises(asyncio.QueueFull):
            await queue.submit("item3")

    @pytest.mark.asyncio
    async def test_get_nowait_empty(self):
        """get_nowait returns None when empty."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        item = await queue.get_nowait()
        
        assert item is None

    @pytest.mark.asyncio
    async def test_cancel_request(self):
        """Cancel removes item from queue."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        await queue.submit("item1", request_id="req1")
        await queue.submit("item2", request_id="req2")
        await queue.submit("item3", request_id="req3")
        
        cancelled = await queue.cancel("req2")
        
        assert cancelled is True
        assert queue.size == 2
        assert queue.stats.total_cancelled == 1

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        """Cancel returns False for nonexistent request."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        await queue.submit("item1", request_id="req1")
        
        cancelled = await queue.cancel("nonexistent")
        
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_processor_lifecycle(self):
        """Processor starts and stops correctly."""
        results = []
        
        async def process(item: str) -> str:
            results.append(item)
            return f"processed:{item}"
        
        queue: PriorityQueue[str] = PriorityQueue(max_size=10, process_fn=process)
        
        await queue.start_processor()
        
        # Submit items
        future1 = await queue.submit("item1")
        future2 = await queue.submit("item2")
        
        # Wait for processing
        result1 = await asyncio.wait_for(future1, timeout=1.0)
        result2 = await asyncio.wait_for(future2, timeout=1.0)
        
        await queue.stop_processor()
        
        assert result1 == "processed:item1"
        assert result2 == "processed:item2"
        assert "item1" in results
        assert "item2" in results

    @pytest.mark.asyncio
    async def test_processor_error_handling(self):
        """Processor handles errors without crashing."""
        async def process(item: str) -> str:
            if item == "error":
                raise ValueError("Test error")
            return f"processed:{item}"
        
        queue: PriorityQueue[str] = PriorityQueue(max_size=10, process_fn=process)
        
        await queue.start_processor()
        
        future_ok = await queue.submit("ok")
        future_error = await queue.submit("error")
        future_ok2 = await queue.submit("ok2")
        
        # First should succeed
        result = await asyncio.wait_for(future_ok, timeout=1.0)
        assert result == "processed:ok"
        
        # Error should raise
        with pytest.raises(ValueError):
            await asyncio.wait_for(future_error, timeout=1.0)
        
        # Third should still succeed
        result2 = await asyncio.wait_for(future_ok2, timeout=1.0)
        assert result2 == "processed:ok2"
        
        await queue.stop_processor()
        
        assert queue.stats.total_errors == 1

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Stats are properly tracked."""
        async def process(item: str) -> str:
            return item
        
        queue: PriorityQueue[str] = PriorityQueue(max_size=10, process_fn=process)
        
        await queue.start_processor()
        
        await queue.submit("item1", Priority.CRITICAL)
        await queue.submit("item2", Priority.NORMAL)
        await queue.submit("item3", Priority.LOW)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        await queue.stop_processor()
        
        stats = queue.get_stats()
        
        assert stats["total_submitted"] == 3
        assert stats["total_processed"] == 3
        assert "CRITICAL" in stats["by_priority"]
        assert "NORMAL" in stats["by_priority"]
        assert "LOW" in stats["by_priority"]

    @pytest.mark.asyncio
    async def test_queue_snapshot(self):
        """Queue snapshot returns pending items."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=10)
        
        await queue.submit("item1", Priority.HIGH, request_id="req1")
        await queue.submit("item2", Priority.LOW, request_id="req2")
        
        snapshot = queue.get_queue_snapshot()
        
        assert len(snapshot) == 2
        # Should be sorted by priority
        assert snapshot[0]["priority"] == "HIGH"
        assert snapshot[1]["priority"] == "LOW"

    @pytest.mark.asyncio
    async def test_is_full_property(self):
        """is_full property works correctly."""
        queue: PriorityQueue[str] = PriorityQueue(max_size=2)
        
        assert queue.is_full is False
        
        await queue.submit("item1")
        assert queue.is_full is False
        
        await queue.submit("item2")
        assert queue.is_full is True
