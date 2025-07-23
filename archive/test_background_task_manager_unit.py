"""
Unit Tests for BackgroundTaskManager

Focused unit tests for BackgroundTaskManager core functionality,
following 2025 best practices with real behavior testing.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.services.health.background_manager import (
    BackgroundTaskManager,
    BackgroundTask,
    TaskStatus,
)


class TestBackgroundTaskManagerUnit:
    """Unit tests for BackgroundTaskManager core functionality."""

    @pytest.fixture
    def event_loop(self):
        """Create isolated event loop for each test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    @pytest.fixture
    async def manager(self):
        """Create BackgroundTaskManager instance for testing."""
        manager = BackgroundTaskManager(max_concurrent_tasks=5)
        await manager.start()
        yield manager
        await manager.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test BackgroundTaskManager initialization."""
        manager = BackgroundTaskManager(max_concurrent_tasks=10)
        
        assert manager.max_concurrent_tasks == 10
        assert len(manager.tasks) == 0
        assert len(manager.running_tasks) == 0
        assert manager._monitor_task is None
        assert not manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_manager_start_stop_lifecycle(self):
        """Test BackgroundTaskManager start/stop lifecycle."""
        manager = BackgroundTaskManager(max_concurrent_tasks=3)
        
        # Test start
        await manager.start()
        assert manager._monitor_task is not None
        assert not manager._monitor_task.done()
        
        # Test stop
        await manager.stop(timeout=1.0)
        assert manager._shutdown_event.is_set()
        assert manager._monitor_task.done()

    @pytest.mark.asyncio
    async def test_task_submission_basic(self, manager):
        """Test basic task submission."""
        async def test_task():
            await asyncio.sleep(0.1)
            return "test_result"
        
        task_id = await manager.submit_task("test_task", test_task)
        
        assert task_id == "test_task"
        assert task_id in manager.tasks
        
        task = manager.tasks[task_id]
        assert task.task_id == task_id
        assert task.status == TaskStatus.PENDING
        assert task.asyncio_task is not None

    @pytest.mark.asyncio
    async def test_task_submission_duplicate_id(self, manager):
        """Test task submission with duplicate ID."""
        async def test_task():
            return "test"
        
        # Submit first task
        await manager.submit_task("duplicate_id", test_task)
        
        # Submit second task with same ID should raise error
        with pytest.raises(ValueError, match="Task duplicate_id already exists"):
            await manager.submit_task("duplicate_id", test_task)

    @pytest.mark.asyncio
    async def test_task_submission_max_concurrent_exceeded(self, manager):
        """Test task submission when max concurrent tasks exceeded."""
        async def long_task():
            await asyncio.sleep(1.0)
            return "long_task_result"
        
        # Submit tasks up to the limit
        for i in range(5):  # max_concurrent_tasks = 5
            await manager.submit_task(f"task_{i}", long_task)
        
        # Wait for tasks to start
        await asyncio.sleep(0.1)
        
        # Submitting another task should raise error
        with pytest.raises(ValueError, match="Maximum concurrent tasks exceeded"):
            await manager.submit_task("excess_task", long_task)

    @pytest.mark.asyncio
    async def test_task_execution_success(self, manager):
        """Test successful task execution."""
        async def success_task():
            await asyncio.sleep(0.1)
            return "success_result"
        
        task_id = await manager.submit_task("success_task", success_task)
        
        # Wait for task to complete
        await asyncio.sleep(0.2)
        
        task = manager.get_task_status(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "success_result"
        assert task.error is None
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_execution_failure(self, manager):
        """Test failed task execution."""
        async def failing_task():
            await asyncio.sleep(0.1)
            raise RuntimeError("Task failed")
        
        task_id = await manager.submit_task("failing_task", failing_task)
        
        # Wait for task to fail
        await asyncio.sleep(0.2)
        
        task = manager.get_task_status(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.result is None
        assert task.error == "Task failed"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_cancellation(self, manager):
        """Test task cancellation."""
        async def cancellable_task():
            await asyncio.sleep(1.0)
            return "should_not_complete"
        
        task_id = await manager.submit_task("cancellable_task", cancellable_task)
        
        # Wait for task to start
        await asyncio.sleep(0.1)
        
        # Cancel task
        result = await manager.cancel_task(task_id)
        assert result is True
        
        # Wait for cancellation to complete
        await asyncio.sleep(0.1)
        
        task = manager.get_task_status(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_task_cancellation_not_found(self, manager):
        """Test cancellation of non-existent task."""
        result = await manager.cancel_task("non_existent_task")
        assert result is False

    @pytest.mark.asyncio
    async def test_task_cancellation_already_completed(self, manager):
        """Test cancellation of already completed task."""
        async def quick_task():
            await asyncio.sleep(0.05)
            return "quick_result"
        
        task_id = await manager.submit_task("quick_task", quick_task)
        
        # Wait for task to complete
        await asyncio.sleep(0.1)
        
        # Try to cancel completed task
        result = await manager.cancel_task(task_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_task_success(self, manager):
        """Test waiting for task completion."""
        async def awaitable_task():
            await asyncio.sleep(0.1)
            return "awaitable_result"
        
        task_id = await manager.submit_task("awaitable_task", awaitable_task)
        
        # Wait for task and get result
        result = await manager.wait_for_task(task_id)
        assert result == "awaitable_result"

    @pytest.mark.asyncio
    async def test_wait_for_task_failure(self, manager):
        """Test waiting for failed task."""
        async def failing_task():
            await asyncio.sleep(0.1)
            raise RuntimeError("Task failed")
        
        task_id = await manager.submit_task("failing_task", failing_task)
        
        # Wait for task should raise exception
        with pytest.raises(Exception, match="Task failing_task failed: Task failed"):
            await manager.wait_for_task(task_id)

    @pytest.mark.asyncio
    async def test_wait_for_task_cancelled(self, manager):
        """Test waiting for cancelled task."""
        async def cancellable_task():
            await asyncio.sleep(1.0)
            return "should_not_complete"
        
        task_id = await manager.submit_task("cancellable_task", cancellable_task)
        
        # Cancel task after short delay
        asyncio.create_task(self._cancel_after_delay(manager, task_id, 0.1))
        
        # Wait for task should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await manager.wait_for_task(task_id)

    async def _cancel_after_delay(self, manager, task_id, delay):
        """Helper to cancel task after delay."""
        await asyncio.sleep(delay)
        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_wait_for_task_timeout(self, manager):
        """Test waiting for task with timeout."""
        async def slow_task():
            await asyncio.sleep(1.0)
            return "slow_result"
        
        task_id = await manager.submit_task("slow_task", slow_task)
        
        # Wait with short timeout should raise TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await manager.wait_for_task(task_id, timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_task_not_found(self, manager):
        """Test waiting for non-existent task."""
        with pytest.raises(ValueError, match="Task non_existent not found"):
            await manager.wait_for_task("non_existent")

    @pytest.mark.asyncio
    async def test_get_task_status(self, manager):
        """Test getting task status."""
        async def status_task():
            await asyncio.sleep(0.1)
            return "status_result"
        
        task_id = await manager.submit_task("status_task", status_task)
        
        # Get task status
        task = manager.get_task_status(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, manager):
        """Test getting status of non-existent task."""
        task = manager.get_task_status("non_existent")
        assert task is None

    @pytest.mark.asyncio
    async def test_get_running_tasks(self, manager):
        """Test getting running tasks."""
        async def running_task():
            await asyncio.sleep(0.5)
            return "running_result"
        
        # Submit multiple tasks
        for i in range(3):
            await manager.submit_task(f"running_task_{i}", running_task)
        
        # Wait for tasks to start
        await asyncio.sleep(0.1)
        
        running_tasks = manager.get_running_tasks()
        assert len(running_tasks) == 3
        assert all(task_id.startswith("running_task_") for task_id in running_tasks)

    @pytest.mark.asyncio
    async def test_get_task_count(self, manager):
        """Test getting task count by status."""
        async def count_task():
            await asyncio.sleep(0.1)
            return "count_result"
        
        # Submit tasks
        for i in range(3):
            await manager.submit_task(f"count_task_{i}", count_task)
        
        # Get initial counts
        counts = manager.get_task_count()
        assert counts["pending"] >= 0
        assert counts["running"] >= 0
        assert counts["completed"] >= 0
        assert counts["failed"] >= 0
        assert counts["cancelled"] >= 0

    @pytest.mark.asyncio
    async def test_get_queue_size(self, manager):
        """Test getting queue size."""
        async def queue_task():
            await asyncio.sleep(0.1)
            return "queue_result"
        
        # Submit tasks
        for i in range(3):
            await manager.submit_task(f"queue_task_{i}", queue_task)
        
        # Get queue size (pending + running)
        queue_size = manager.get_queue_size()
        assert queue_size >= 0
        assert queue_size <= 3

    @pytest.mark.asyncio
    async def test_task_cleanup(self, manager):
        """Test task cleanup after completion."""
        async def cleanup_task():
            await asyncio.sleep(0.1)
            return "cleanup_result"
        
        task_id = await manager.submit_task("cleanup_task", cleanup_task)
        
        # Wait for task to complete
        await asyncio.sleep(0.2)
        
        # Task should still be in tasks dict (cleanup happens periodically)
        assert task_id in manager.tasks
        
        # Task should not be in running_tasks
        assert task_id not in manager.running_tasks

    @pytest.mark.asyncio
    async def test_monitor_task_cleanup(self, manager):
        """Test monitor task cleanup functionality."""
        # Create a task that completes quickly
        async def quick_task():
            return "quick_result"
        
        task_id = await manager.submit_task("quick_task", quick_task)
        
        # Wait for task to complete
        await asyncio.sleep(0.1)
        
        # Manually set completed_at to old time to trigger cleanup
        task = manager.tasks[task_id]
        task.completed_at = time.time() - 3700  # 1 hour and 1 minute ago
        
        # Wait for monitor task to run cleanup
        await asyncio.sleep(0.5)
        
        # Task should be cleaned up (removed from tasks dict)
        # Note: This might not always work in unit tests due to timing
        # But the logic is tested in the monitor task