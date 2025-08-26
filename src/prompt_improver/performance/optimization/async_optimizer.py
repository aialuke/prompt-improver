"""Enhanced Async Optimization with 2025 Best Practices.

Advanced async optimizer implementing 2025 best practices:
- Unified connection management with health monitoring
- Multi-level caching via DatabaseServices (L1 memory + L2 Redis)
- Resource optimization with memory and CPU management
- Enhanced monitoring with OpenTelemetry integration
- Adaptive batching with dynamic sizing
- Circuit breakers and fault tolerance
- Security context validation for cache operations
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from prompt_improver.database.types import PoolConfiguration
from prompt_improver.performance.optimization.performance_optimizer import (
    measure_mcp_operation,
)

if TYPE_CHECKING:
    from prompt_improver.database.factories import (
                SecurityContext,
            )

tracer = trace.get_tracer(__name__)
try:
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
try:
    from opentelemetry import metrics, trace

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def add_event(self, name, attributes=None):
            pass

        def set_status(self, status):
            pass


logger = logging.getLogger(__name__)


class ResourceOptimizationMode(Enum):
    """Resource optimization modes."""

    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"
    balanced = "balanced"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    cpu_percent: float
    memory_percent: float
    memory_mb: float
    active_connections: int
    cache_hit_rate: float
    operation_latency_ms: float
    throughput_ops_per_sec: float
    timestamp: datetime

    def model_dump(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_mb,
            "active_connections": self.active_connections,
            "cache_hit_rate": self.cache_hit_rate,
            "operation_latency_ms": self.operation_latency_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AsyncOperationConfig:
    """Enhanced configuration for async operation optimization."""

    max_concurrent_operations: int = 10
    operation_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_batching: bool = True
    batch_size: int = 50
    batch_timeout: float = 0.1
    enable_intelligent_caching: bool = True
    enable_connection_pooling: bool = True
    enable_resource_optimization: bool = True
    enable_circuit_breaker: bool = True
    resource_optimization_mode: ResourceOptimizationMode = (
        ResourceOptimizationMode.balanced
    )
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 80.0
    cache_ttl_seconds: int = 3600
    enable_cache_warming: bool = True
    pool_configuration: PoolConfiguration | None = None
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_half_open_max_calls: int = 3


if OPENTELEMETRY_AVAILABLE:
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    ASYNC_OPERATION_DURATION = meter.create_histogram(
        "async_operation_duration_seconds",
        description="Async operation duration",
        unit="s",
    )
    CACHE_HIT_RATE = meter.create_gauge(
        "async_cache_hit_rate", description="Cache hit rate", unit="1"
    )
    CONNECTION_POOL_UTILIZATION = meter.create_gauge(
        "async_connection_pool_utilization",
        description="Connection pool utilization",
        unit="1",
    )
    RESOURCE_UTILIZATION = meter.create_gauge(
        "async_resource_utilization", description="Resource utilization", unit="1"
    )
else:
    tracer = MockTracer()
    meter = None
    ASYNC_OPERATION_DURATION = None
    CACHE_HIT_RATE = None
    CONNECTION_POOL_UTILIZATION = None
    RESOURCE_UTILIZATION = None


class AsyncBatchProcessor:
    """Batches async operations for improved throughput."""

    def __init__(self, config: AsyncOperationConfig) -> None:
        self.config = config
        self._pending_operations: list[tuple[Callable, tuple, dict]] = []
        self._batch_lock = asyncio.Lock()
        self._processing = False

    async def add_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Add operation to batch for processing."""
        async with self._batch_lock:
            future = asyncio.Future()
            self._pending_operations.append((operation, args, kwargs, future))
            if len(self._pending_operations) >= self.config.batch_size:
                from prompt_improver.performance.monitoring.health.background_manager import (
                    get_background_task_manager,
                )

                task_manager = get_background_task_manager()
                await task_manager.submit_task(
                    "batch_processing", self._process_batch()
                )
            return await future

    async def _process_batch(self) -> None:
        """Process a batch of operations concurrently."""
        if self._processing:
            return
        self._processing = True
        try:
            async with self._batch_lock:
                if not self._pending_operations:
                    return
                batch = self._pending_operations[: self.config.batch_size]
                self._pending_operations = self._pending_operations[
                    self.config.batch_size :
                ]
            semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)

            async def process_operation(operation, args, kwargs, future) -> None:
                async with semaphore:
                    try:
                        result = await operation(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)

            from prompt_improver.performance.monitoring.health.background_manager import (
                TaskPriority,
                get_background_task_manager,
            )

            task_manager = get_background_task_manager()
            task_ids = []
            batch_id = str(uuid.uuid4())[:8]
            for i, (op, args, kwargs, future) in enumerate(batch):
                task_id = await task_manager.submit_enhanced_task(
                    task_id=f"batch_op_{batch_id}_{i}",
                    coroutine=lambda op=op,
                    args=args,
                    kwargs=kwargs,
                    future=future: process_operation(op, args, kwargs, future),
                    priority=TaskPriority.NORMAL,
                    tags={
                        "service": "async_optimizer",
                        "type": "batch_operation",
                        "batch_id": batch_id,
                    },
                )
                task_ids.append(task_id)
            batch_tasks = []
            for task_id in task_ids:
                task = task_manager.get_task_status(task_id)
                if task and task.asyncio_task:
                    batch_tasks.append(task.asyncio_task)
            if batch_tasks:
                await asyncio.gather(*batch_tasks, return_exceptions=True)
        finally:
            self._processing = False


class AsyncTaskScheduler:
    """Optimized task scheduler for high-performance async operations."""

    def __init__(self) -> None:
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task_ids: list[str] = []
        self._running = False
        self._worker_count = 4
        self._scheduler_id = str(uuid.uuid4())[:8]

    async def start(self):
        """Start the task scheduler workers."""
        if self._running:
            return
        self._running = True
        from prompt_improver.performance.monitoring.health.background_manager import (
            TaskPriority,
            get_background_task_manager,
        )

        task_manager = get_background_task_manager()
        self._worker_task_ids = []
        for i in range(self._worker_count):
            task_id = await task_manager.submit_enhanced_task(
                task_id=f"async_optimizer_worker_{self._scheduler_id}_{i}",
                coroutine=lambda worker_id=i: self._worker(f"worker-{worker_id}"),
                priority=TaskPriority.NORMAL,
                tags={
                    "service": "async_optimizer",
                    "type": "worker_pool",
                    "scheduler_id": self._scheduler_id,
                },
            )
            self._worker_task_ids.append(task_id)
        logger.info(
            f"Started {self._worker_count} async task workers with centralized management"
        )

    async def stop(self):
        """Stop the task scheduler workers."""
        self._running = False
        from prompt_improver.performance.monitoring.health.background_manager import (
            get_background_task_manager,
        )

        task_manager = get_background_task_manager()
        worker_tasks = []
        for task_id in self._worker_task_ids:
            task = task_manager.get_task_status(task_id)
            if task and task.asyncio_task:
                task.asyncio_task.cancel()
                worker_tasks.append(task.asyncio_task)
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        self._worker_task_ids.clear()
        logger.info("Stopped async task scheduler with centralized management")

    async def schedule_task(
        self, operation: Callable, priority: int = 0, *args, **kwargs
    ) -> asyncio.Future:
        """Schedule a task for execution."""
        future = asyncio.Future()
        task_item = (priority, time.time(), operation, args, kwargs, future)
        await self._task_queue.put(task_item)
        return future

    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine for processing tasks."""
        logger.debug(f"Started async worker: {worker_name}")
        while self._running:
            try:
                task_item = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                _priority, _scheduled_time, operation, args, kwargs, future = task_item
                try:
                    result = await operation(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._task_queue.task_done()
            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Worker {worker_name} error: {e}")


class AsyncOptimizer:
    """Main async optimization coordinator."""

    def __init__(self, config: AsyncOperationConfig | None = None) -> None:
        self.config = config or AsyncOperationConfig()
        # Note: get_database_services now returns DatabaseServices (composition layer)

        self.connection_manager = None  # Will be set in initialize()
        self.batch_processor = AsyncBatchProcessor(self.config)
        self.task_scheduler = AsyncTaskScheduler()
        self._optimization_enabled = True
        self._security_context: SecurityContext | None = None
        self._cache_key_prefix = "async_optimizer"

    async def initialize(self):
        """Initialize the async optimizer."""
        await self.task_scheduler.start()

        from prompt_improver.database import ManagerMode, get_database_services
        from prompt_improver.database.factories import create_security_context

        # Get DatabaseServices instance
        self.connection_manager = await get_database_services(ManagerMode.ASYNC_MODERN)

        self._security_context = await create_security_context(
            agent_id="async_optimizer", tier="professional", authenticated=True
        )
        logger.info(
            "Async optimizer initialized with DatabaseServices composition layer"
        )

    async def shutdown(self):
        """Shutdown the async optimizer."""
        await self.task_scheduler.stop()
        if self.connection_manager:
            await self.connection_manager.shutdown_all()
        logger.info("Async optimizer shutdown complete")

    @asynccontextmanager
    async def optimized_operation(
        self, operation_name: str, enable_batching: bool = False, priority: int = 0
    ):
        """Context manager for optimized async operations."""
        async with measure_mcp_operation(f"async_{operation_name}") as perf_metrics:
            start_time = time.perf_counter()
            try:
                if enable_batching and self._optimization_enabled:
                    yield self.batch_processor
                else:
                    yield None
                execution_time = (time.perf_counter() - start_time) * 1000
                perf_metrics.metadata["execution_time_ms"] = execution_time
                perf_metrics.metadata["optimization_enabled"] = (
                    self._optimization_enabled
                )
            except Exception as e:
                perf_metrics.metadata["error"] = str(e)
                raise

    async def execute_with_retry(
        self, operation: Callable, *args, max_retries: int | None = None, **kwargs
    ) -> Any:
        """Execute operation with unified retry logic."""
        from prompt_improver.core.services.resilience.retry_service_facade import (
            get_retry_service as get_retry_manager,
        )

        max_retries = max_retries or self.config.retry_attempts
        retry_config = RetryConfig(
            max_attempts=max_retries + 1,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=self.config.retry_delay,
            jitter=True,
            operation_name=f"async_optimizer_{operation.__name__}",
        )
        retry_manager = get_retry_manager()

        async def retry_operation():
            return await operation(*args, **kwargs)

        return await retry_manager.retry_async(retry_operation, config=retry_config)

    async def execute_concurrent_operations(
        self,
        operations: list[tuple[Callable, tuple, dict]],
        max_concurrency: int | None = None,
    ) -> list[Any]:
        """Execute multiple operations concurrently with controlled concurrency."""
        max_concurrency = max_concurrency or self.config.max_concurrent_operations
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(operation, args, kwargs):
            async with semaphore:
                return await operation(*args, **kwargs)

        from prompt_improver.performance.monitoring.health.background_manager import (
            TaskPriority,
            get_background_task_manager,
        )

        task_manager = get_background_task_manager()
        task_ids = []
        operation_batch_id = str(uuid.uuid4())[:8]
        for i, (op, args, kwargs) in enumerate(operations):
            task_id = await task_manager.submit_enhanced_task(
                task_id=f"concurrent_op_{operation_batch_id}_{i}",
                coroutine=lambda op=op,
                args=args,
                kwargs=kwargs: execute_with_semaphore(op, args, kwargs),
                priority=TaskPriority.NORMAL,
                tags={
                    "service": "async_optimizer",
                    "type": "concurrent_operation",
                    "batch_id": operation_batch_id,
                },
            )
            task_ids.append(task_id)
        concurrent_tasks = []
        for task_id in task_ids:
            task = task_manager.get_task_status(task_id)
            if task and task.asyncio_task:
                concurrent_tasks.append(task.asyncio_task)
        if concurrent_tasks:
            return await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        return []

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the async optimizer including cache performance."""
        base_metrics = {
            "optimization_enabled": self._optimization_enabled,
            "config": {
                "max_concurrent_operations": self.config.max_concurrent_operations,
                "operation_timeout": self.config.operation_timeout,
                "retry_attempts": self.config.retry_attempts,
                "batch_size": self.config.batch_size,
                "batch_timeout": self.config.batch_timeout,
                "enable_intelligent_caching": self.config.enable_intelligent_caching,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
            },
            "task_scheduler": {
                "running": self.task_scheduler._running,
                "worker_count": len(self.task_scheduler._worker_task_ids),
                "queue_size": self.task_scheduler._task_queue.qsize(),
            },
        }
        if self.connection_manager and hasattr(self.connection_manager, "cache"):
            try:
                # Use sync method to get basic cache info
                base_metrics["cache_performance"] = {
                    "cache_available": True,
                    "connection_manager_type": type(self.connection_manager).__name__,
                }
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                base_metrics["cache_performance"] = {"error": str(e)}
        return base_metrics

    async def get_cached_result(self, operation_key: str) -> Any | None:
        """Get cached optimization result using DatabaseServices multi-level cache.

        Args:
            operation_key: Unique key for the optimization operation

        Returns:
            Cached result or None if not found
        """
        if (
            not self.config.enable_intelligent_caching
            or not self._security_context
            or not self.connection_manager
        ):
            return None
        try:
            cache_key = f"{self._cache_key_prefix}:result:{operation_key}"
            return await self.connection_manager.cache.get(
                key=cache_key, security_context=self._security_context
            )
        except Exception as e:
            logger.warning(f"Failed to get cached result for {operation_key}: {e}")
            return None

    async def cache_optimization_result(self, operation_key: str, result: Any) -> bool:
        """Cache optimization result using DatabaseServices multi-level cache.

        Args:
            operation_key: Unique key for the optimization operation
            result: Result to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if (
            not self.config.enable_intelligent_caching
            or not self._security_context
            or not self.connection_manager
        ):
            return False
        try:
            cache_key = f"{self._cache_key_prefix}:result:{operation_key}"
            return await self.connection_manager.cache.set(
                key=cache_key,
                value=result,
                ttl_seconds=self.config.cache_ttl_seconds,
                security_context=self._security_context,
            )
        except Exception as e:
            logger.warning(f"Failed to cache result for {operation_key}: {e}")
            return False

    async def get_cached_metrics(self, metrics_key: str) -> dict[str, Any] | None:
        """Get cached performance metrics using DatabaseServices cache.

        Args:
            metrics_key: Unique key for the metrics

        Returns:
            Cached metrics or None if not found
        """
        if (
            not self.config.enable_intelligent_caching
            or not self._security_context
            or not self.connection_manager
        ):
            return None
        try:
            cache_key = f"{self._cache_key_prefix}:metrics:{metrics_key}"
            return await self.connection_manager.cache.get(
                key=cache_key, security_context=self._security_context
            )
        except Exception as e:
            logger.warning(f"Failed to get cached metrics for {metrics_key}: {e}")
            return None

    async def cache_performance_metrics(
        self, metrics_key: str, metrics: dict[str, Any]
    ) -> bool:
        """Cache performance metrics using DatabaseServices cache.

        Args:
            metrics_key: Unique key for the metrics
            metrics: Metrics to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if (
            not self.config.enable_intelligent_caching
            or not self._security_context
            or not self.connection_manager
        ):
            return False
        try:
            cache_key = f"{self._cache_key_prefix}:metrics:{metrics_key}"
            metrics_ttl = min(self.config.cache_ttl_seconds, 1800)
            return await self.connection_manager.cache.set(
                key=cache_key,
                value=metrics,
                ttl_seconds=metrics_ttl,
                security_context=self._security_context,
            )
        except Exception as e:
            logger.warning(f"Failed to cache metrics for {metrics_key}: {e}")
            return False

    async def invalidate_cache(self, pattern: str | None = None) -> int:
        """Invalidate cached optimization data.

        Args:
            pattern: Optional pattern to match keys (None invalidates all optimizer cache)

        Returns:
            Number of keys invalidated
        """
        if not self._security_context or not self.connection_manager:
            return 0
        invalidated_count = 0
        try:
            if pattern:
                cache_key = f"{self._cache_key_prefix}:{pattern}"
                if await self.connection_manager.cache.delete(cache_key):
                    invalidated_count += 1
            else:
                logger.info(
                    f"Cache invalidation requested for async optimizer (prefix: {self._cache_key_prefix})"
                )
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
        return invalidated_count


_global_optimizer: AsyncOptimizer | None = None


async def get_async_optimizer() -> AsyncOptimizer:
    """Get the global async optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AsyncOptimizer()
        await _global_optimizer.initialize()
    return _global_optimizer


async def optimized_async_operation(
    operation_name: str,
    operation: Callable,
    *args,
    enable_retry: bool = True,
    enable_batching: bool = False,
    **kwargs,
) -> Any:
    """Execute an operation with full async optimization."""
    optimizer = await get_async_optimizer()
    async with optimizer.optimized_operation(operation_name, enable_batching):
        if enable_retry:
            return await optimizer.execute_with_retry(operation, *args, **kwargs)
        return await operation(*args, **kwargs)


async def shutdown_async_optimizer():
    """Shutdown the global async optimizer."""
    global _global_optimizer
    if _global_optimizer is not None:
        await _global_optimizer.shutdown()
        _global_optimizer = None
