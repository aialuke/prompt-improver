"""Background Task Manager for the Adaptive Prompt Enhancement System (APES).

This module provides a reusable BackgroundTaskManager class that handles
background task lifecycle management, monitoring, and graceful shutdown.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TaskStatus(Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Represents a background task with metadata."""

    task_id: str
    coroutine: Callable
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Any | None = None
    asyncio_task: asyncio.Task | None = None


class BackgroundTaskManager:
    """Manages background tasks with monitoring and graceful shutdown capabilities."""

    def __init__(self, max_concurrent_tasks: int = 10):
        """Initialize the background task manager.

        Args:
            max_concurrent_tasks: Maximum number of tasks that can run concurrently.
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: dict[str, BackgroundTask] = {}
        self.running_tasks: set[str] = set()
        self.logger = logging.getLogger(__name__)
        self._shutdown_event = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background task manager."""
        self.logger.info("Starting BackgroundTaskManager")
        self._monitor_task = asyncio.create_task(self._monitor_tasks())

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the background task manager gracefully.

        Args:
            timeout: Maximum time to wait for tasks to complete.
        """
        self.logger.info("Stopping BackgroundTaskManager")
        self._shutdown_event.set()

        # Cancel all running tasks
        running_tasks = [
            task.asyncio_task
            for task in self.tasks.values()
            if task.asyncio_task and not task.asyncio_task.done()
        ]

        if running_tasks:
            self.logger.info(f"Cancelling {len(running_tasks)} running tasks")
            for task in running_tasks:
                task.cancel()

            # Wait for tasks to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*running_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                self.logger.warning("Some tasks did not complete within timeout")

        # Stop the monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("BackgroundTaskManager stopped")

    async def submit_task(self, task_id: str, coroutine: Callable, **kwargs) -> str:
        """Submit a task for background execution.

        Args:
            task_id: Unique identifier for the task.
            coroutine: The coroutine to execute.
            **kwargs: Additional arguments to pass to the coroutine.

        Returns:
            The task ID.

        Raises:
            ValueError: If task_id already exists or max concurrent tasks exceeded.
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        if len(self.running_tasks) >= self.max_concurrent_tasks:
            raise ValueError("Maximum concurrent tasks exceeded")

        # Create background task
        bg_task = BackgroundTask(task_id=task_id, coroutine=coroutine)

        self.tasks[task_id] = bg_task

        # Start the task
        asyncio_task = asyncio.create_task(self._execute_task(bg_task, **kwargs))
        bg_task.asyncio_task = asyncio_task

        self.logger.info(f"Submitted task {task_id}")
        return task_id

    async def _execute_task(self, bg_task: BackgroundTask, **kwargs) -> None:
        """Execute a background task."""
        bg_task.status = TaskStatus.RUNNING
        bg_task.started_at = time.time()
        self.running_tasks.add(bg_task.task_id)

        try:
            self.logger.debug(f"Executing task {bg_task.task_id}")
            result = await bg_task.coroutine(**kwargs)
            bg_task.result = result
            bg_task.status = TaskStatus.COMPLETED
            self.logger.info(f"Task {bg_task.task_id} completed successfully")
        except asyncio.CancelledError:
            bg_task.status = TaskStatus.CANCELLED
            self.logger.info(f"Task {bg_task.task_id} was cancelled")
            raise
        except Exception as e:
            bg_task.error = str(e)
            bg_task.status = TaskStatus.FAILED
            self.logger.error(f"Task {bg_task.task_id} failed: {e}")
        finally:
            bg_task.completed_at = time.time()
            self.running_tasks.discard(bg_task.task_id)

    async def _monitor_tasks(self) -> None:
        """Monitor background tasks and clean up completed ones."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up completed tasks older than 1 hour
                current_time = time.time()
                tasks_to_remove = []

                for task_id, task in self.tasks.items():
                    if (
                        task.status
                        in [
                            TaskStatus.COMPLETED,
                            TaskStatus.FAILED,
                            TaskStatus.CANCELLED,
                        ]
                        and task.completed_at
                        and current_time - task.completed_at > 3600
                    ):  # 1 hour
                        tasks_to_remove.append(task_id)

                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    self.logger.debug(f"Cleaned up task {task_id}")

                # Log status periodically
                if len(self.tasks) > 0:
                    self.logger.debug(
                        f"Task status: {len(self.running_tasks)} running, "
                        f"{len(self.tasks)} total"
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task monitor: {e}")
                await asyncio.sleep(30)

    def get_task_status(self, task_id: str) -> BackgroundTask | None:
        """Get the status of a specific task.

        Args:
            task_id: The task ID to check.

        Returns:
            The BackgroundTask object or None if not found.
        """
        return self.tasks.get(task_id)

    def get_running_tasks(self) -> list[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks)

    def get_task_count(self) -> dict[str, int]:
        """Get count of tasks by status."""
        counts = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            counts[task.status.value] += 1
        return counts

    def get_queue_size(self) -> int:
        """Get the current queue size (pending + running tasks)."""
        return len([
            task
            for task in self.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ])

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if task was cancelled, False if not found or already completed.
        """
        task = self.tasks.get(task_id)
        if not task or task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            return False

        if task.asyncio_task and not task.asyncio_task.done():
            task.asyncio_task.cancel()
            self.logger.info(f"Cancelled task {task_id}")
            return True

        return False

    async def wait_for_task(self, task_id: str, timeout: float | None = None) -> Any:
        """Wait for a specific task to complete.

        Args:
            task_id: The task ID to wait for.
            timeout: Maximum time to wait.

        Returns:
            The task result.

        Raises:
            ValueError: If task not found.
            asyncio.TimeoutError: If timeout exceeded.
            Exception: If task failed.
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task.asyncio_task:
            if timeout:
                await asyncio.wait_for(task.asyncio_task, timeout=timeout)
            else:
                await task.asyncio_task

        if task.status == TaskStatus.FAILED:
            raise Exception(f"Task {task_id} failed: {task.error}")
        if task.status == TaskStatus.CANCELLED:
            raise asyncio.CancelledError(f"Task {task_id} was cancelled")

        return task.result


# Global instance for easy access
_background_task_manager: BackgroundTaskManager | None = None


def get_background_task_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance."""
    global _background_task_manager
    if _background_task_manager is None:
        _background_task_manager = BackgroundTaskManager()
    return _background_task_manager


async def init_background_task_manager(
    max_concurrent_tasks: int = 10,
) -> BackgroundTaskManager:
    """Initialize and start the global background task manager.

    Args:
        max_concurrent_tasks: Maximum number of concurrent tasks.

    Returns:
        The initialized BackgroundTaskManager instance.
    """
    global _background_task_manager
    _background_task_manager = BackgroundTaskManager(max_concurrent_tasks)
    await _background_task_manager.start()
    return _background_task_manager


async def shutdown_background_task_manager(timeout: float = 30.0) -> None:
    """Shutdown the global background task manager.

    Args:
        timeout: Maximum time to wait for tasks to complete.
    """
    global _background_task_manager
    if _background_task_manager:
        await _background_task_manager.stop(timeout)
        _background_task_manager = None
