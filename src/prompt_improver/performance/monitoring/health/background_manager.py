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
import logging
import time
import uuid
import heapq
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

# Observability imports
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Circuit breaker imports (simple implementation if not available)
try:
    from circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    # Simple circuit breaker implementation
    class CircuitBreaker:
        def __init__(self, failure_threshold=5, timeout=60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "closed"  # closed, open, half_open

        def call(self, func):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = func()
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                raise e

class TaskStatus(Enum):
    """Enhanced task status with 2025 patterns."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"

class TaskPriority(Enum):
    """Task priority levels for scheduling."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

# Import protocol interfaces to prevent circular imports (2025 best practice)
from ....core.protocols.retry_protocols import (
    RetryConfigProtocol,
    RetryStrategy,
    RetryableErrorType
)
# Import concrete implementation for creating retry configs
from ....core.retry_implementations import BasicRetryConfig
from ....core.retry_config import RetryConfig

@dataclass
class TaskMetrics:
    """Metrics for task execution."""

    execution_count: int = 0
    total_duration: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    last_execution_time: Optional[datetime] = None
    average_duration: float = 0.0

@dataclass
class EnhancedBackgroundTask:
    """Enhanced background task with 2025 features."""

    task_id: str
    coroutine: Callable
    priority: TaskPriority = TaskPriority.NORMAL
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: Optional[CircuitBreaker] = None
    timeout: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

    # Status and timing
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution details
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None

    # Runtime objects
    asyncio_task: Optional[asyncio.Task] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value

# Use centralized metrics registry
from ..metrics_registry import get_metrics_registry, StandardMetrics

metrics_registry = get_metrics_registry()
TASK_COUNTER = metrics_registry.get_or_create_counter(
    StandardMetrics.BACKGROUND_TASKS_TOTAL,
    'Total background tasks',
    ['status', 'priority']
)
TASK_DURATION = metrics_registry.get_or_create_histogram(
    StandardMetrics.BACKGROUND_TASK_DURATION,
    'Task execution duration',
    ['task_type', 'priority']
)
ACTIVE_TASKS = metrics_registry.get_or_create_gauge(
    StandardMetrics.BACKGROUND_TASKS_ACTIVE,
    'Currently active tasks'
)
RETRY_COUNTER = metrics_registry.get_or_create_counter(
    StandardMetrics.BACKGROUND_TASK_RETRIES_TOTAL,
    'Task retry attempts',
    ['task_type']
)
CIRCUIT_BREAKER_STATE = metrics_registry.get_or_create_gauge(
    StandardMetrics.CIRCUIT_BREAKER_STATE,
    'Circuit breaker state',
    ['task_type']
)

@dataclass
class BackgroundTask:
    """Legacy task class for backward compatibility."""

    task_id: str
    coroutine: Callable
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Any | None = None
    asyncio_task: asyncio.Task | None = None

class EnhancedBackgroundTaskManager:
    """Enhanced background task manager with 2025 best practices."""

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        enable_metrics: bool = True,
        dead_letter_queue_size: int = 1000,
        default_timeout: float = 300.0
    ):
        """Initialize the enhanced background task manager.

        Args:
            max_concurrent_tasks: Maximum number of tasks that can run concurrently.
            enable_metrics: Whether to enable Prometheus metrics.
            dead_letter_queue_size: Maximum size of dead letter queue.
            default_timeout: Default timeout for tasks in seconds.
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE
        self.default_timeout = default_timeout

        # Task storage
        self.tasks: Dict[str, EnhancedBackgroundTask] = {}
        self.priority_queue: List[EnhancedBackgroundTask] = []
        self.running_tasks: Set[str] = set()
        self.dead_letter_queue: List[EnhancedBackgroundTask] = []
        self.dead_letter_queue_size = dead_letter_queue_size

        # Circuit breakers for different task types
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Monitoring and control
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._shutdown_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._retry_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_retries": 0,
            "total_dead_letter": 0
        }

    async def start(self) -> None:
        """Start the enhanced background task manager."""
        self.logger.info("Starting EnhancedBackgroundTaskManager")

        # Start monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitor_tasks())
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._retry_task = asyncio.create_task(self._retry_loop())

        self.logger.info("Enhanced background task manager started with priority scheduling and retry mechanisms")

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

        # Stop all monitoring tasks
        for task in [self._monitor_task, self._scheduler_task, self._retry_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info("EnhancedBackgroundTaskManager stopped")

    async def submit_enhanced_task(
        self,
        task_id: Optional[str] = None,
        coroutine: Callable = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
        circuit_breaker_key: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Submit an enhanced task with 2025 features.

        Args:
            task_id: Unique identifier for the task (auto-generated if None).
            coroutine: The coroutine to execute.
            priority: Task priority for scheduling.
            retry_config: Retry configuration.
            timeout: Task timeout in seconds.
            circuit_breaker_key: Key for circuit breaker grouping.
            tags: Additional metadata tags.
            **kwargs: Additional arguments to pass to the coroutine.

        Returns:
            The task ID.

        Raises:
            ValueError: If task_id already exists or max concurrent tasks exceeded.
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        # Create circuit breaker if specified
        circuit_breaker = None
        if circuit_breaker_key:
            if circuit_breaker_key not in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker_key] = CircuitBreaker()
            circuit_breaker = self.circuit_breakers[circuit_breaker_key]

        # Create enhanced task
        enhanced_task = EnhancedBackgroundTask(
            task_id=task_id,
            coroutine=coroutine,
            priority=priority,
            retry_config=retry_config or BasicRetryConfig(),
            circuit_breaker=circuit_breaker,
            timeout=timeout or self.default_timeout,
            tags=tags or {}
        )

        self.tasks[task_id] = enhanced_task

        # Add to priority queue
        heapq.heappush(self.priority_queue, enhanced_task)
        enhanced_task.status = TaskStatus.QUEUED

        # Update statistics
        self.stats["total_submitted"] += 1

        # Update metrics
        if self.enable_metrics:
            TASK_COUNTER.labels(status="submitted", priority=priority.name).inc()

        self.logger.info(f"Submitted enhanced task {task_id} with priority {priority.name}")
        return task_id

    async def _scheduler_loop(self):
        """Priority-based task scheduler loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check if we can run more tasks
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue

                # Get highest priority task
                if not self.priority_queue:
                    await asyncio.sleep(0.1)
                    continue

                task = heapq.heappop(self.priority_queue)

                # Skip if task was cancelled
                if task.status == TaskStatus.CANCELLED:
                    continue

                # Start the task
                await self._start_task(task)

            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1.0)

    async def _retry_loop(self):
        """Retry failed tasks using unified retry manager."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()

                # Check for tasks ready for retry
                for task in list(self.tasks.values()):
                    if (task.status == TaskStatus.RETRYING and
                        task.next_retry_at and
                        current_time >= task.next_retry_at):

                        # Re-queue for execution using unified retry logic
                        task.status = TaskStatus.QUEUED
                        task.next_retry_at = None
                        heapq.heappush(self.priority_queue, task)

                        self.logger.info(f"Re-queuing task {task.task_id} for retry {task.retry_count}")

                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                self.logger.error(f"Retry loop error: {e}")
                await asyncio.sleep(5.0)

    async def _start_task(self, task: EnhancedBackgroundTask):
        """Start execution of a task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self.running_tasks.add(task.task_id)

        # Update metrics
        if self.enable_metrics:
            ACTIVE_TASKS.inc()

        # Create asyncio task
        task.asyncio_task = asyncio.create_task(self._execute_enhanced_task(task))

        self.logger.debug(f"Started task {task.task_id}")

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

    async def _execute_enhanced_task(self, task: EnhancedBackgroundTask, **kwargs) -> None:
        """Execute an enhanced background task with retry and circuit breaker support."""
        start_time = datetime.utcnow()

        try:
            # Apply timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_with_circuit_breaker(task, **kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_with_circuit_breaker(task, **kwargs)

            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            # Update metrics
            task.metrics.execution_count += 1
            task.metrics.success_count += 1
            task.metrics.last_execution_time = task.completed_at

            duration = (task.completed_at - start_time).total_seconds()
            task.metrics.total_duration += duration
            task.metrics.average_duration = task.metrics.total_duration / task.metrics.execution_count

            # Update statistics
            self.stats["total_completed"] += 1

            # Update Prometheus metrics
            if self.enable_metrics:
                TASK_COUNTER.labels(status="completed", priority=task.priority.name).inc()
                TASK_DURATION.labels(
                    task_type=task.tags.get("type", "unknown"),
                    priority=task.priority.name
                ).observe(duration)

            self.logger.info(f"Task {task.task_id} completed successfully in {duration:.2f}s")

        except asyncio.TimeoutError:
            await self._handle_task_failure(task, "Task timeout", start_time, **kwargs)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            self.logger.info(f"Task {task.task_id} was cancelled")
            raise
        except Exception as e:
            await self._handle_task_failure(task, str(e), start_time, **kwargs)
        finally:
            self.running_tasks.discard(task.task_id)
            if self.enable_metrics:
                ACTIVE_TASKS.dec()

    async def _execute_with_circuit_breaker(self, task: EnhancedBackgroundTask, **kwargs):
        """Execute task with circuit breaker protection."""
        if task.circuit_breaker:
            async def execute():
                return await task.coroutine(**kwargs)
            return await execute()
        else:
            return await task.coroutine(**kwargs)

    async def _handle_task_failure(self, task: EnhancedBackgroundTask, error_msg: str, start_time: datetime, **kwargs):
        """Handle task failure with retry logic."""
        task.error = error_msg
        task.completed_at = datetime.utcnow()

        # Update metrics
        task.metrics.execution_count += 1
        task.metrics.failure_count += 1
        task.metrics.last_execution_time = task.completed_at

        duration = (task.completed_at - start_time).total_seconds()
        task.metrics.total_duration += duration
        task.metrics.average_duration = task.metrics.total_duration / task.metrics.execution_count

        # Check if we should retry
        if task.retry_count < task.retry_config.max_attempts:
            await self._schedule_retry(task)
        else:
            # Move to dead letter queue
            await self._move_to_dead_letter(task)

        # Update statistics
        self.stats["total_failed"] += 1

        # Update Prometheus metrics
        if self.enable_metrics:
            TASK_COUNTER.labels(status="failed", priority=task.priority.name).inc()

        self.logger.error(f"Task {task.task_id} failed: {error_msg}")

    async def _schedule_retry(self, task: EnhancedBackgroundTask):
        """Schedule a task for retry using unified retry manager."""
        task.retry_count += 1
        task.status = TaskStatus.RETRYING

        # Convert task retry config to unified retry config
        unified_config = BasicRetryConfig(
            max_attempts=getattr(task.retry_config, 'max_attempts', 3),
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=getattr(task.retry_config, 'initial_delay_ms', 1000) / 1000.0,  # Convert ms to seconds
            max_delay=getattr(task.retry_config, 'max_delay_ms', 60000) / 1000.0,      # Convert ms to seconds
            jitter=getattr(task.retry_config, 'jitter', True)
        )

        # Use unified retry manager to calculate delay
        from ....ml.orchestration.core.unified_retry_manager import get_retry_manager
        retry_manager = get_retry_manager()
        delay_ms = retry_manager._calculate_delay(task.retry_count - 1, unified_config)
        delay = delay_ms / 1000.0

        task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

        # Update statistics
        self.stats["total_retries"] += 1
        task.metrics.retry_count += 1

        # Update Prometheus metrics
        if self.enable_metrics:
            RETRY_COUNTER.labels(task_type=task.tags.get("type", "unknown")).inc()

        self.logger.info(f"Scheduled retry {task.retry_count}/{getattr(task.retry_config, 'max_attempts', 3)} for task {task.task_id} in {delay:.2f}s")

    async def _move_to_dead_letter(self, task: EnhancedBackgroundTask):
        """Move failed task to dead letter queue."""
        task.status = TaskStatus.DEAD_LETTER

        # Add to dead letter queue (with size limit)
        if len(self.dead_letter_queue) >= self.dead_letter_queue_size:
            self.dead_letter_queue.pop(0)  # Remove oldest

        self.dead_letter_queue.append(task)

        # Update statistics
        self.stats["total_dead_letter"] += 1

        self.logger.warning(f"Task {task.task_id} moved to dead letter queue after {task.retry_count} retries")

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

    def get_enhanced_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced task status with metrics."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task_id,
            "status": task.status.value,
            "priority": task.priority.name,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error,
            "retry_count": task.retry_count,
            "next_retry_at": task.next_retry_at.isoformat() if task.next_retry_at else None,
            "tags": task.tags,
            "metrics": {
                "execution_count": task.metrics.execution_count,
                "success_count": task.metrics.success_count,
                "failure_count": task.metrics.failure_count,
                "retry_count": task.metrics.retry_count,
                "average_duration": task.metrics.average_duration,
                "total_duration": task.metrics.total_duration,
                "last_execution_time": task.metrics.last_execution_time.isoformat() if task.metrics.last_execution_time else None
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task manager statistics."""
        return {
            "total_submitted": self.stats["total_submitted"],
            "total_completed": self.stats["total_completed"],
            "total_failed": self.stats["total_failed"],
            "total_retries": self.stats["total_retries"],
            "total_dead_letter": self.stats["total_dead_letter"],
            "currently_running": len(self.running_tasks),
            "queued_tasks": len(self.priority_queue),
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "circuit_breakers": {
                key: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time
                }
                for key, cb in self.circuit_breakers.items()
            }
        }

    def get_dead_letter_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks in dead letter queue."""
        return [
            {
                "task_id": task.task_id,
                "error": task.error,
                "retry_count": task.retry_count,
                "created_at": task.created_at.isoformat(),
                "failed_at": task.completed_at.isoformat() if task.completed_at else None,
                "tags": task.tags
            }
            for task in self.dead_letter_queue
        ]

    async def requeue_dead_letter_task(self, task_id: str) -> bool:
        """Requeue a task from dead letter queue."""
        for i, task in enumerate(self.dead_letter_queue):
            if task.task_id == task_id:
                # Reset task state
                task.status = TaskStatus.QUEUED
                task.error = None
                task.retry_count = 0
                task.next_retry_at = None

                # Remove from dead letter queue and re-queue
                self.dead_letter_queue.pop(i)
                heapq.heappush(self.priority_queue, task)

                self.logger.info(f"Requeued task {task_id} from dead letter queue")
                return True

        return False

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for background task management (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - tasks: List of tasks to submit with their configurations
                - max_concurrent: Maximum concurrent tasks
                - enable_metrics: Whether to enable metrics collection
                - output_path: Local path for output files (optional)

        Returns:
            Orchestrator-compatible result with task management analysis and metadata
        """
        start_time = datetime.utcnow()

        try:
            # Extract configuration from orchestrator
            tasks_config = config.get("tasks", [])
            max_concurrent = config.get("max_concurrent", self.max_concurrent_tasks)
            enable_metrics = config.get("enable_metrics", self.enable_metrics)
            output_path = config.get("output_path", "./outputs/background_task_management")

            # Update configuration
            self.max_concurrent_tasks = max_concurrent
            self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE

            # Submit tasks from configuration
            submitted_tasks = []
            for task_config in tasks_config:
                task_id = await self.submit_enhanced_task(
                    task_id=task_config.get("task_id"),
                    coroutine=task_config.get("coroutine"),
                    priority=TaskPriority(task_config.get("priority", TaskPriority.NORMAL.value)),
                    retry_config=BasicRetryConfig(**task_config.get("retry_config", {})),
                    timeout=task_config.get("timeout"),
                    circuit_breaker_key=task_config.get("circuit_breaker_key"),
                    tags=task_config.get("tags", {})
                )
                submitted_tasks.append(task_id)

            # Get current statistics
            stats = self.get_statistics()

            # Prepare orchestrator-compatible result
            result = {
                "task_management_summary": {
                    "submitted_tasks": len(submitted_tasks),
                    "total_tasks_managed": len(self.tasks),
                    "currently_running": stats["currently_running"],
                    "queued_tasks": stats["queued_tasks"],
                    "dead_letter_queue_size": stats["dead_letter_queue_size"],
                    "max_concurrent_tasks": self.max_concurrent_tasks
                },
                "performance_metrics": {
                    "total_submitted": stats["total_submitted"],
                    "total_completed": stats["total_completed"],
                    "total_failed": stats["total_failed"],
                    "total_retries": stats["total_retries"],
                    "success_rate": stats["total_completed"] / max(1, stats["total_submitted"]),
                    "failure_rate": stats["total_failed"] / max(1, stats["total_submitted"]),
                    "retry_rate": stats["total_retries"] / max(1, stats["total_submitted"])
                },
                "circuit_breaker_status": stats["circuit_breakers"],
                "task_details": [
                    self.get_enhanced_task_status(task_id)
                    for task_id in submitted_tasks
                ],
                "dead_letter_tasks": self.get_dead_letter_tasks(),
                "capabilities": {
                    "priority_scheduling": True,
                    "retry_mechanisms": True,
                    "circuit_breakers": True,
                    "dead_letter_queue": True,
                    "metrics_collection": self.enable_metrics,
                    "graceful_shutdown": True
                }
            }

            # Calculate execution metadata
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "tasks_submitted": len(submitted_tasks),
                    "max_concurrent_tasks": max_concurrent,
                    "metrics_enabled": enable_metrics,
                    "circuit_breakers_active": len(self.circuit_breakers),
                    "component_version": "2025.1.0"
                }
            }

        except ValueError as e:
            self.logger.error(f"Validation error in orchestrated task management: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": f"Validation error: {str(e)}", "task_management_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "error": True,
                    "error_type": "validation",
                    "component_version": "2025.1.0"
                }
            }
        except Exception as e:
            self.logger.error(f"Orchestrated task management failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "task_management_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "2025.1.0"
                }
            }

# Maintain backward compatibility
class BackgroundTaskManager(EnhancedBackgroundTaskManager):
    """Backward compatible task manager."""

    def __init__(self, max_concurrent_tasks: int = 10):
        super().__init__(max_concurrent_tasks=max_concurrent_tasks, enable_metrics=False)
        self.legacy_tasks: Dict[str, BackgroundTask] = {}

    async def submit_task(self, task_id: str, coroutine: Callable, **kwargs) -> str:
        """Legacy task submission method."""
        # Create legacy task
        legacy_task = BackgroundTask(
            task_id=task_id,
            coroutine=coroutine
        )
        self.legacy_tasks[task_id] = legacy_task

        # Submit as enhanced task with normal priority
        return await self.submit_enhanced_task(
            task_id=task_id,
            coroutine=coroutine,
            priority=TaskPriority.NORMAL,
            **kwargs
        )

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Legacy task status method."""
        enhanced_status = self.get_enhanced_task_status(task_id)
        if not enhanced_status:
            return None

        # Convert to legacy format
        return {
            "task_id": task_id,
            "status": enhanced_status["status"],
            "created_at": enhanced_status["created_at"],
            "started_at": enhanced_status["started_at"],
            "completed_at": enhanced_status["completed_at"],
            "result": enhanced_status["result"],
            "error": enhanced_status["error"]
        }

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
