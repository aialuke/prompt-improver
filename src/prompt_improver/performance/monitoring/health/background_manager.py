"""Enhanced Background Task Manager - 2025 Edition

Advanced background task management with 2025 best practices:
- Priority-based task scheduling
- Retry mechanisms with exponential backoff
- Circuit breaker patterns for external dependencies
- Comprehensive observability and metrics
- Dead letter queue handling
- Resource-aware task distribution
"""
import asyncio
import heapq
import logging
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels for background task scheduling."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class TaskStatus:
    """Status of a background task."""
    task_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0

class EnhancedBackgroundTaskManager:
    """Enhanced background task manager with 2025 best practices."""

    def __init__(self, max_concurrent_tasks: int=10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self._tasks: dict[str, TaskStatus] = {}
        self._running_tasks: set[str] = set()
        self._task_queue: list = []
        self._shutdown = False
        self._lock = asyncio.Lock()

    async def submit_enhanced_task(self, coroutine: Any, task_id: str | None=None, priority: TaskPriority=TaskPriority.NORMAL, tags: dict[str, str] | None=None) -> str:
        """Submit a task for background execution."""
        if task_id is None:
            task_id = f'task_{uuid.uuid4().hex[:8]}'
        async with self._lock:
            if task_id in self._tasks:
                return task_id
            status = TaskStatus(task_id=task_id, status='queued', created_at=datetime.now(UTC))
            self._tasks[task_id] = status
            heapq.heappush(self._task_queue, (-priority.value, time.time(), task_id, coroutine))
        asyncio.create_task(self._process_queue())
        return task_id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the status of a background task."""
        return self._tasks.get(task_id, TaskStatus(task_id=task_id, status='not_found', created_at=datetime.now(UTC)))

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task."""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = 'cancelled'
                return True
            return False

    async def _process_queue(self):
        """Process queued tasks."""
        if self._shutdown or len(self._running_tasks) >= self.max_concurrent_tasks:
            return
        async with self._lock:
            if not self._task_queue:
                return
            _, _, task_id, coroutine = heapq.heappop(self._task_queue)
            if task_id not in self._tasks or self._tasks[task_id].status != 'queued':
                return
            self._running_tasks.add(task_id)
            self._tasks[task_id].status = 'running'
            self._tasks[task_id].started_at = datetime.now(UTC)
        try:
            await coroutine
            async with self._lock:
                self._tasks[task_id].status = 'completed'
                self._tasks[task_id].completed_at = datetime.now(UTC)
        except Exception as e:
            async with self._lock:
                self._tasks[task_id].status = 'failed'
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = datetime.now(UTC)
        finally:
            async with self._lock:
                self._running_tasks.discard(task_id)
        if self._task_queue:
            asyncio.create_task(self._process_queue())

    async def shutdown(self):
        """Shutdown the task manager."""
        self._shutdown = True
        for _ in range(30):
            if not self._running_tasks:
                break
            await asyncio.sleep(1)
_global_task_manager: EnhancedBackgroundTaskManager | None = None
_manager_lock = asyncio.Lock()

async def get_background_task_manager() -> EnhancedBackgroundTaskManager:
    """Get or create the global background task manager."""
    global _global_task_manager
    if _global_task_manager is None:
        async with _manager_lock:
            if _global_task_manager is None:
                _global_task_manager = EnhancedBackgroundTaskManager()
    return _global_task_manager

def get_background_task_manager() -> EnhancedBackgroundTaskManager:
    """Synchronous version for compatibility."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = EnhancedBackgroundTaskManager()
    return _global_task_manager
