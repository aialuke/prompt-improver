import asyncio
import heapq
import logging
import random
import time
from asyncio import create_task, gather, sleep
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union

from pydantic import BaseModel, ConfigDict, Field

from prompt_improver.database import get_session
# get_ml_service imported lazily to break circular imports
from prompt_improver.utils.datetime_utils import aware_utc_now

# Enhanced 2025 imports with graceful fallbacks
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Enhanced 2025 enums and data structures
class ProcessingMode(Enum):
    """Processing modes for batch processor"""
    local = "local"
    DISTRIBUTED_RAY = "distributed_ray"
    DISTRIBUTED_DASK = "distributed_dask"
    STREAM_PROCESSING = "stream_processing"

class PartitionStrategy(Enum):
    """Data partitioning strategies"""
    ROUND_ROBIN = "round_robin"
    HASH_BASED = "hash_based"
    SIZE_BASED = "size_based"
    CONTENT_AWARE = "content_aware"
    LOAD_BALANCED = "load_balanced"

class WorkerScalingMode(Enum):
    """Worker auto-scaling modes"""
    fixed = "fixed"
    QUEUE_DEPTH = "queue_depth"
    LATENCY_BASED = "latency_based"
    ADAPTIVE = "adaptive"

@dataclass
class BatchMetrics:
    """Enhanced metrics for batch processing"""
    batch_id: str
    processing_mode: ProcessingMode
    partition_strategy: PartitionStrategy
    worker_count: int
    items_processed: int
    processing_time_ms: float
    throughput_items_per_sec: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    queue_depth: int
    error_count: int
    retry_count: int
    timestamp: datetime

@dataclass
class WorkerStats:
    """Worker statistics for auto-scaling"""
    current_workers: int
    target_workers: int
    queue_depth: int
    avg_processing_time_ms: float
    cpu_utilization: float
    memory_utilization: float

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for batch processing resilience"""
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - (self.last_failure_time or 0) > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN

async def periodic_batch_processor_coroutine(batch_processor: "BatchProcessor") -> None:
    """Periodic coroutine that triggers batch processing at configured intervals.

    This coroutine runs continuously and calls BatchProcessor.process_training_batch
    every BATCH_TIMEOUT seconds. It handles queue checking and batch formation.

    Args:
        batch_processor: The BatchProcessor instance to trigger periodic processing on
    """
    logger = logging.getLogger(f"{__name__}.PeriodicBatchProcessor")
    logger.info(
        f"Starting periodic batch processor with {batch_processor.config.batch_timeout}s interval"
    )

    while True:
        try:
            # Wait for the configured batch timeout
            await asyncio.sleep(batch_processor.config.batch_timeout)

            # Check if there are items in the queue to process
            if not batch_processor._has_pending_items():
                logger.debug("No pending items for batch processing")
                continue

            # Fetch training batch from the queue
            training_batch = await batch_processor._get_training_batch()

            if training_batch:
                logger.info(
                    f"Triggering periodic batch processing for {len(training_batch)} items"
                )
                result = await batch_processor.process_training_batch(training_batch)

                if result.get("status") == "success":
                    logger.info(
                        f"Periodic batch processing completed: {result.get('processed_records', 0)} records"
                    )
                else:
                    logger.warning(
                        f"Periodic batch processing failed: {result.get('error', 'unknown error')}"
                    )
            else:
                logger.debug("No training batch formed during periodic trigger")

        except asyncio.CancelledError:
            logger.info("Periodic batch processor cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in periodic batch processor: {e}")
            # Continue running despite errors
            await asyncio.sleep(1)

class BatchProcessorConfig(BaseModel):
    """Configuration for BatchProcessor with validation and best practices."""

    # Core processing settings
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of records to process in each batch",
    )
    concurrency: int = Field(
        default=3, ge=1, le=20, description="Number of concurrent processing workers"
    )
    timeout: int = Field(
        default=30000, ge=1000, description="Timeout per batch in milliseconds"
    )
    batch_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout between periodic batch processing triggers in seconds",
    )

    # Retry and exponential backoff settings
    max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts per record"
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay for exponential backoff in seconds",
    )
    max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay for exponential backoff in seconds",
    )
    jitter: bool = Field(
        default=True, description="Add jitter to prevent thundering herd"
    )

    # Queue management
    max_queue_size: int = Field(
        default=1000, ge=10, le=10000, description="Maximum size of the internal queue"
    )
    rate_limit_per_second: float | None = Field(
        default=None, ge=0.1, description="Rate limit in requests per second"
    )

    # Processing behavior
    dry_run: bool = Field(
        default=False, description="Run in dry-run mode without actual processing"
    )
    enable_priority_queue: bool = Field(
        default=True, description="Enable priority-based processing"
    )

    # Monitoring and logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")

    # Enhanced 2025 features
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.local, description="Processing mode (local, distributed, stream)"
    )
    partition_strategy: PartitionStrategy = Field(
        default=PartitionStrategy.ROUND_ROBIN, description="Data partitioning strategy"
    )
    worker_scaling_mode: WorkerScalingMode = Field(
        default=WorkerScalingMode.fixed, description="Worker auto-scaling mode"
    )

    # Distributed processing settings
    enable_distributed_processing: bool = Field(
        default=False, description="Enable distributed processing with Ray/Dask"
    )
    enable_stream_processing: bool = Field(
        default=False, description="Enable stream processing with Kafka"
    )
    enable_intelligent_partitioning: bool = Field(
        default=False, description="Enable intelligent data partitioning"
    )

    # Worker scaling settings
    min_workers: int = Field(default=1, ge=1, le=100, description="Minimum number of workers")
    max_workers: int = Field(default=10, ge=1, le=100, description="Maximum number of workers")
    scale_up_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Queue utilization threshold for scaling up")
    scale_down_threshold: float = Field(default=0.3, ge=0.1, le=1.0, description="Queue utilization threshold for scaling down")

    # Circuit breaker settings
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker pattern")
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1, le=20, description="Circuit breaker failure threshold")
    circuit_breaker_recovery_timeout: float = Field(default=60.0, ge=10.0, le=300.0, description="Circuit breaker recovery timeout in seconds")
    circuit_breaker_success_threshold: int = Field(default=3, ge=1, le=10, description="Circuit breaker success threshold for recovery")

    # Dead letter queue settings
    enable_dead_letter_queue: bool = Field(default=True, description="Enable dead letter queue for failed items")
    max_dead_letter_items: int = Field(default=1000, ge=10, le=10000, description="Maximum items in dead letter queue")

    # OpenTelemetry settings
    enable_opentelemetry: bool = Field(default=True, description="Enable OpenTelemetry tracing and metrics")
    trace_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="OpenTelemetry trace sampling rate")

    # Stream processing settings
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    kafka_topic_prefix: str = Field(default="batch_processor", description="Kafka topic prefix")

    # Distributed processing settings
    ray_address: Optional[str] = Field(default=None, description="Ray cluster address (None for local)")
    dask_scheduler_address: Optional[str] = Field(default=None, description="Dask scheduler address")

    model_config = ConfigDict(
        extra="forbid",  # Prevent extra fields
        validate_assignment=True
    )

    @property
    def rate_limit_delay(self) -> float | None:
        """Calculate delay between requests based on rate limit."""
        if self.rate_limit_per_second is None:
            return None
        return 1.0 / self.rate_limit_per_second

class AsyncQueueWrapper:
    """Asyncio Queue wrapper with exponential backoff and rate limiting."""

    def __init__(self, config: BatchProcessorConfig):
        self.config = config
        self.queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.failure_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.logger = logging.getLogger(f"{__name__}.AsyncQueueWrapper")
        self.processing = False
        self.last_request_time = 0.0

        # Rate limiting semaphore
        self.rate_semaphore = asyncio.Semaphore(config.concurrency)

    async def put(self, item: dict[str, Any], priority: int = 50) -> None:
        """Put an item into the queue with optional priority."""
        queue_item = {
            "item": item,
            "priority": priority,
            "timestamp": time.time(),
            "attempts": 0,
            "next_retry": None,
        }

        try:
            await self.queue.put(queue_item)
            self.logger.debug(f"Enqueued item with priority {priority}")
        except asyncio.QueueFull:
            self.logger.warning("Queue is full, dropping item")
            raise

    async def get(self) -> dict[str, Any] | None:
        """Get an item from the queue with rate limiting."""
        try:
            # Apply rate limiting
            if self.config.rate_limit_delay:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.config.rate_limit_delay:
                    await asyncio.sleep(self.config.rate_limit_delay - elapsed)
                self.last_request_time = time.time()

            # Get item from queue
            queue_item = await self.queue.get()

            # Check if it's time to retry
            if queue_item["next_retry"] and time.time() < queue_item["next_retry"]:
                # Re-enqueue for later processing
                await self.queue.put(queue_item)
                return None

            return queue_item

        except asyncio.QueueEmpty:
            return None

    async def put_failed(self, item: dict[str, Any]) -> None:
        """Put a failed item into the failure queue for retry using unified retry manager."""
        item["attempts"] += 1

        if item["attempts"] >= self.config.max_attempts:
            self.logger.error(
                f"Max attempts ({self.config.max_attempts}) reached for item, giving up"
            )
            return

        # Use unified retry manager for delay calculation
        from ...orchestration.core.unified_retry_manager import get_retry_manager, RetryConfig, RetryStrategy

        retry_config = RetryConfig(
            max_attempts=self.config.max_attempts,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay_ms=int(self.config.base_delay * 1000),
            max_delay_ms=int(self.config.max_delay * 1000),
            jitter=self.config.jitter,
            operation_name=f"batch_item_{item.get('id', 'unknown')}"
        )

        retry_manager = get_retry_manager()
        delay_ms = retry_manager._calculate_delay(item["attempts"] - 1, retry_config)
        delay = delay_ms / 1000.0

        item["next_retry"] = time.time() + delay

        # Re-enqueue with calculated delay
        await self.queue.put(item)

        self.logger.warning(
            f"Retrying item (attempt {item['attempts']}/{self.config.max_attempts}) "
            f"after {delay:.2f} seconds"
        )

    def qsize(self) -> int:
        """Return the current size of the queue."""
        return self.queue.qsize()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.queue.empty()

    def full(self) -> bool:
        """Check if the queue is full."""
        return self.queue.full()

    async def wait_for_completion(self, timeout: float | None = None) -> None:
        """Wait for all items in the queue to be processed."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except TimeoutError:
            self.logger.warning(f"Queue processing timeout after {timeout} seconds")

@dataclass(order=True)
class PriorityRecord:
    """Wrapper for priority queue records with priority ordering."""

    priority: int
    record: dict[str, Any] = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    attempts: int = field(default=0, compare=False)
    next_retry: float | None = field(default=None, compare=False)

class PriorityQueue:
    """Heap-based priority queue with support for priority:int on records."""

    def __init__(self):
        self._queue = []
        self._index = 0
        self._removed = set()

    def enqueue(self, record: dict[str, Any], priority: int) -> None:
        """Add a record with priority to the queue."""
        priority_record = PriorityRecord(
            priority=priority, record=record, timestamp=time.time()
        )
        heapq.heappush(self._queue, priority_record)

    def dequeue(self) -> PriorityRecord | None:
        """Remove and return the highest priority record (lowest priority number)."""
        while self._queue:
            priority_record = heapq.heappop(self._queue)
            if id(priority_record) not in self._removed:
                return priority_record
        return None

    def peek(self) -> PriorityRecord | None:
        """Return the highest priority record without removing it."""
        while self._queue:
            if id(self._queue[0]) not in self._removed:
                return self._queue[0]
            heapq.heappop(self._queue)
        return None

    def remove(self, priority_record: PriorityRecord) -> None:
        """Mark a record as removed (lazy deletion)."""
        self._removed.add(id(priority_record))

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0 or all(
            id(item) in self._removed for item in self._queue
        )

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._queue) - len(self._removed)

class BatchProcessor:
    def __init__(self, config: BatchProcessorConfig | None = None):
        # Initialize with enhanced configuration system
        self.config = config or BatchProcessorConfig()

        # Set up logging
        self.logger = logging.getLogger("BatchProcessor")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Initialize queue systems
        if self.config.enable_priority_queue:
            self.priority_queue = PriorityQueue()
        else:
            self.priority_queue = None

        # Initialize async queue wrapper
        self.async_queue = AsyncQueueWrapper(self.config)

        # Processing state
        self.processing = False
        self.metrics = (
            {"processed": 0, "failed": 0, "retries": 0, "start_time": time.time()}
            if self.config.metrics_enabled
            else None
        )

        # Enhanced 2025 features
        self.distributed_client = None
        self.stream_producer = None
        self.stream_consumer = None
        self.worker_stats = WorkerStats(
            current_workers=self.config.min_workers,
            target_workers=self.config.min_workers,
            queue_depth=0,
            avg_processing_time_ms=0.0,
            cpu_utilization=0.0,
            memory_utilization=0.0
        )

        # Enhanced metrics and monitoring
        self.batch_metrics: deque = deque(maxlen=1000)
        self.partition_cache: Dict[str, List[Any]] = {}
        self.processing_stats = defaultdict(list)
        self.processing_tasks: Dict[str, asyncio.Task] = {}

        # Dead letter queue
        if self.config.enable_dead_letter_queue:
            self.dead_letter_queue = asyncio.Queue(maxsize=self.config.max_dead_letter_items)
        else:
            self.dead_letter_queue = None

        # Circuit breaker
        if self.config.enable_circuit_breaker:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout,
                success_threshold=self.config.circuit_breaker_success_threshold
            )
            self.circuit_breaker = CircuitBreaker(circuit_config)
        else:
            self.circuit_breaker = None

        # OpenTelemetry setup
        if self.config.enable_opentelemetry and OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self._setup_opentelemetry_metrics()
        else:
            self.tracer = None
            self.meter = None

        # Initialize distributed processing
        if self.config.enable_distributed_processing:
            asyncio.create_task(self._initialize_distributed_processing())

        # Initialize stream processing
        if self.config.enable_stream_processing:
            asyncio.create_task(self._initialize_stream_processing())

        # Start auto-scaling if enabled
        if self.config.worker_scaling_mode != WorkerScalingMode.fixed:
            asyncio.create_task(self._auto_scaling_loop())

        self.logger.info(f"Enhanced batch processor initialized with {self.config.processing_mode.value} mode")

    async def enqueue(self, record: dict[str, Any], priority: int = 50) -> None:
        """Public async API to enqueue a record with priority.

        Supports both priority queue and async queue based on configuration.
        """
        if self.config.enable_priority_queue and self.priority_queue:
            # Use priority queue for immediate processing
            self.priority_queue.enqueue(record, priority)
            self.logger.debug(
                f"Enqueued record to priority queue with priority {priority}"
            )

            # Start processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_queue())
        else:
            # Use async queue wrapper with exponential backoff
            await self.async_queue.put(record, priority)
            self.logger.debug(
                f"Enqueued record to async queue with priority {priority}"
            )

            # Start async processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_async_queue())

    async def _process_queue(self) -> None:
        """Process items in the priority queue with retry logic."""
        if self.processing:
            return

        self.processing = True
        try:
            while not self.priority_queue.empty():
                priority_record = self.priority_queue.dequeue()
                if priority_record is None:
                    break

                # Check if it's time to retry
                if (
                    priority_record.next_retry
                    and time.time() < priority_record.next_retry
                ):
                    # Re-enqueue for later processing
                    self.priority_queue.enqueue(
                        priority_record.record, priority_record.priority
                    )
                    continue

                success = await self._process_single_record(priority_record)
                if not success:
                    await self._handle_retry(priority_record)

                # Small delay between processing items
                await asyncio.sleep(0.01)

        finally:
            self.processing = False

    async def _process_async_queue(self) -> None:
        """Process items from the async queue wrapper with exponential backoff."""
        if self.processing:
            return

        self.processing = True
        try:
            while not self.async_queue.empty():
                queue_item = await self.async_queue.get()
                if queue_item is None:
                    await asyncio.sleep(0.01)  # Brief pause before checking again
                    continue

                success = await self._process_queue_item(queue_item)
                if not success:
                    await self.async_queue.put_failed(queue_item)
                # Update metrics
                elif self.metrics:
                    self.metrics["processed"] += 1

                # Mark task as done for queue.join()
                self.async_queue.queue.task_done()

        finally:
            self.processing = False

    async def _process_queue_item(self, queue_item: dict[str, Any]) -> bool:
        """Process a single queue item with transaction safety."""
        try:
            item = queue_item["item"]
            priority = queue_item["priority"]

            if self.config.dry_run:
                self.logger.info(f"[DRY RUN] Would process item: {item}")
                return True

            await self._persist_to_database(item)
            self.logger.debug(f"Successfully processed item with priority {priority}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing queue item: {e}")
            if self.metrics:
                self.metrics["failed"] += 1
            return False

    async def _process_single_record(self, priority_record: PriorityRecord) -> bool:
        """Process a single record with transaction safety."""
        try:
            if self.config.dry_run:
                self.logger.info(
                    f"[DRY RUN] Would process record: {priority_record.record}"
                )
                return True

            await self._persist_to_database(priority_record.record)
            self.logger.debug(
                f"Successfully processed record with priority {priority_record.priority}"
            )

            # Update metrics
            if self.metrics:
                self.metrics["processed"] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error processing record: {e}")
            if self.metrics:
                self.metrics["failed"] += 1
            return False

    async def _persist_to_database(self, record: dict[str, Any]) -> None:
        """Persist batch data to training_prompts table with transaction safety."""
        try:
            async with get_session() as db_session:
                await db_session.execute(
                    """
                    INSERT INTO training_prompts (
                        prompt_text, enhancement_result, 
                        data_source, training_priority, created_at
                    ) VALUES (
                        :prompt_text, :enhancement_result::jsonb,
                        :data_source, :training_priority, NOW()
                    )
                    """,
                    {
                        "prompt_text": record.get("original", ""),
                        "enhancement_result": {
                            "enhanced_prompt": record.get("enhanced", ""),
                            "metrics": record.get("metrics", {}),
                            "session_id": record.get("session_id"),
                            "batch_processed": True,
                            "processed_at": aware_utc_now().isoformat(),
                        },
                        "data_source": record.get("data_source", "batch"),
                        "training_priority": record.get("priority", 50),
                    },
                )
                await db_session.commit()

        except Exception as e:
            self.logger.error(f"Database persistence error: {e}")
            raise

    async def _handle_retry(self, priority_record: PriorityRecord) -> None:
        """Handle retry logic using unified retry manager."""
        priority_record.attempts += 1
        max_attempts = self.config.get("maxAttempts", 3)

        if priority_record.attempts >= max_attempts:
            self.logger.error(
                f"Max attempts ({max_attempts}) reached for record, giving up"
            )
            return

        # Use unified retry manager for delay calculation
        from ...orchestration.core.unified_retry_manager import get_retry_manager, RetryConfig, RetryStrategy

        retry_config = RetryConfig(
            max_attempts=max_attempts,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay_ms=int(self.config.get("baseDelay", 1.0) * 1000),
            max_delay_ms=int(self.config.get("maxDelay", 60.0) * 1000),
            jitter=self.config.get("jitter", True),
            operation_name=f"priority_record_{priority_record.record.get('id', 'unknown')}"
        )

        retry_manager = get_retry_manager()
        delay_ms = retry_manager._calculate_delay(priority_record.attempts - 1, retry_config)
        delay = delay_ms / 1000.0

        priority_record.next_retry = time.time() + delay

        # Re-enqueue with same priority but for later processing
        self.priority_queue.enqueue(priority_record.record, priority_record.priority)

        self.logger.warning(
            f"Retrying record (attempt {priority_record.attempts}/{max_attempts}) "
            f"after {delay:.2f} seconds"
        )

    def get_queue_size(self) -> int:
        """Return the current size of the priority queue."""
        return (
            self.priority_queue.size()
            if self.priority_queue
            else self.async_queue.qsize()
        )

    def _has_pending_items(self) -> bool:
        """Check if there are pending items in the queue for batch processing."""
        if self.config.enable_priority_queue and self.priority_queue:
            return not self.priority_queue.empty()
        return not self.async_queue.empty()

    async def _get_training_batch(self) -> list[dict[str, Any]]:
        """Extract a training batch from the queue for periodic processing.

        Returns:
            List of training records up to batch_size limit
        """
        batch = []
        batch_size = self.config.batch_size

        if self.config.enable_priority_queue and self.priority_queue:
            # Extract from priority queue
            while len(batch) < batch_size and not self.priority_queue.empty():
                priority_record = self.priority_queue.dequeue()
                if priority_record is not None:
                    # Check if it's time to process (not in retry backoff)
                    if (
                        priority_record.next_retry is None
                        or time.time() >= priority_record.next_retry
                    ):
                        batch.append(priority_record.record)
                    else:
                        # Re-enqueue for later processing
                        self.priority_queue.enqueue(
                            priority_record.record, priority_record.priority
                        )
        else:
            # Extract from async queue
            while len(batch) < batch_size and not self.async_queue.empty():
                try:
                    queue_item = await asyncio.wait_for(
                        self.async_queue.get(), timeout=0.1
                    )
                    if queue_item is not None:
                        # Check if it's time to process (not in retry backoff)
                        if (
                            queue_item.get("next_retry") is None
                            or time.time() >= queue_item["next_retry"]
                        ):
                            batch.append(queue_item["item"])
                        else:
                            # Re-enqueue for later processing
                            await self.async_queue.put(
                                queue_item["item"], queue_item["priority"]
                            )
                except TimeoutError:
                    break

        return batch

    async def process_batch(
        self, test_cases: list[dict[str, Any]], options: dict[str, Any] = None
    ) -> dict[str, Any]:
        if not test_cases:
            raise ValueError("Test cases must be a non-empty list")

        # Use new config object instead of dict access
        batch_size = (options.get("batchSize") if options else None) or self.config.batch_size
        batches = [
            test_cases[i : i + batch_size] for i in range(0, len(test_cases), batch_size)
        ]

        results = {"processed": 0, "failed": 0, "results": [], "errors": []}

        for index, batch in enumerate(batches):
            self.logger.debug(f"Processing batch {index + 1}/{len(batches)}")
            batch_results = await self.process_single_batch(batch)
            results["processed"] += batch_results["processed"]
            results["failed"] += batch_results["failed"]

            results["results"].extend(batch_results["results"])
            results["errors"].extend(batch_results["errors"])

            # Memory optimization: explicit cleanup for large batches
            if index % 10 == 0 and index > 0:  # Every 10 batches
                import gc
                gc.collect()
                self.logger.debug(f"Memory cleanup after batch {index + 1}")

            # Rate limiting using new config
            await sleep(self.config.base_delay)

        results["executionTime"] = (
            len(batches) * (self.config.timeout / 1000)
        )  # Convert timeout from ms to seconds
        return results

    async def process_single_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_results = {"processed": 0, "failed": 0, "results": [], "errors": []}

        async def process_test_case(test_case: dict[str, Any]) -> dict[str, Any]:
            try:
                # Simulate test case processing
                await sleep(0.1)
                return {"success": True, "test": test_case}
            except Exception as e:
                return {"success": False, "error": str(e)}

        tasks = [create_task(process_test_case(tc)) for tc in batch]
        results = await gather(*tasks)

        for result in results:
            if result["success"]:
                batch_results["processed"] += 1
                batch_results["results"].append(result)
            else:
                batch_results["failed"] += 1
                batch_results["errors"].append(result["error"])

        return batch_results

    async def process_training_batch(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Process training batch by sending to ML integration service.

        Args:
            batch: List of training records to process

        Returns:
            Result of batch processing operation
        """
        try:
            start_time = time.time()

            # Get ML service instance (lazy import to break circular dependency)
            from prompt_improver.ml.core.ml_integration import get_ml_service
            ml_service = await get_ml_service()

            # Send batch to ML integration stub
            result = await ml_service.send_training_batch(batch)

            processing_time = (time.time() - start_time) * 1000

            self.logger.info(
                f"Processed training batch: {result.get('status', 'unknown')} "
                f"({len(batch)} records in {processing_time:.2f}ms)"
            )

            return {
                "status": "success",
                "processed_records": len(batch),
                "processing_time_ms": processing_time,
                "ml_result": result,
            }

        except Exception as e:
            self.logger.error(f"Failed to process training batch: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    # Enhanced 2025 Methods

    def _setup_opentelemetry_metrics(self):
        """Setup OpenTelemetry metrics for batch processing"""
        if not self.meter:
            return

        self.batch_counter = self.meter.create_counter(
            "batch_processor_batches_total",
            description="Total number of batches processed"
        )
        self.item_counter = self.meter.create_counter(
            "batch_processor_items_total",
            description="Total number of items processed"
        )
        self.error_counter = self.meter.create_counter(
            "batch_processor_errors_total",
            description="Total number of processing errors"
        )
        self.processing_time_histogram = self.meter.create_histogram(
            "batch_processor_processing_time_seconds",
            description="Batch processing time in seconds"
        )
        self.queue_depth_gauge = self.meter.create_up_down_counter(
            "batch_processor_queue_depth",
            description="Current queue depth"
        )

    async def _initialize_distributed_processing(self):
        """Initialize distributed processing with Ray or Dask"""
        try:
            if self.config.processing_mode == ProcessingMode.DISTRIBUTED_RAY and RAY_AVAILABLE:
                if self.config.ray_address:
                    ray.init(address=self.config.ray_address)
                else:
                    ray.init()
                self.logger.info("Ray distributed processing initialized")
            elif self.config.processing_mode == ProcessingMode.DISTRIBUTED_DASK and DASK_AVAILABLE:
                if self.config.dask_scheduler_address:
                    self.distributed_client = Client(self.config.dask_scheduler_address)
                else:
                    self.distributed_client = Client()
                self.logger.info("Dask distributed processing initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize distributed processing: {e}")
            self.config.processing_mode = ProcessingMode.local

    async def _initialize_stream_processing(self):
        """Initialize stream processing with Kafka"""
        if not KAFKA_AVAILABLE:
            self.logger.warning("Kafka not available, disabling stream processing")
            self.config.enable_stream_processing = False
            return

        try:
            self.stream_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers.split(','),
                value_serializer=lambda v: str(v).encode('utf-8')
            )
            self.logger.info("Kafka stream processing initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize stream processing: {e}")
            self.config.enable_stream_processing = False

    async def _auto_scaling_loop(self):
        """Auto-scaling loop for dynamic worker management"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_queue_depth = self.get_queue_size()
                queue_utilization = current_queue_depth / self.config.max_queue_size

                # Update worker stats
                self.worker_stats.queue_depth = current_queue_depth

                # Scale up if queue utilization is high
                if (queue_utilization > self.config.scale_up_threshold and
                    self.worker_stats.current_workers < self.config.max_workers):
                    self.worker_stats.target_workers = min(
                        self.worker_stats.current_workers + 1,
                        self.config.max_workers
                    )
                    self.logger.info(f"Scaling up workers to {self.worker_stats.target_workers}")

                # Scale down if queue utilization is low
                elif (queue_utilization < self.config.scale_down_threshold and
                      self.worker_stats.current_workers > self.config.min_workers):
                    self.worker_stats.target_workers = max(
                        self.worker_stats.current_workers - 1,
                        self.config.min_workers
                    )
                    self.logger.info(f"Scaling down workers to {self.worker_stats.target_workers}")

                # Apply scaling changes
                await self._apply_worker_scaling()

            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")

    async def _apply_worker_scaling(self):
        """Apply worker scaling changes"""
        if self.worker_stats.current_workers != self.worker_stats.target_workers:
            self.worker_stats.current_workers = self.worker_stats.target_workers
            # In a real implementation, this would start/stop worker processes
            self.logger.debug(f"Worker count adjusted to {self.worker_stats.current_workers}")

    async def process_with_circuit_breaker(self, batch_items: List[Any]) -> Dict[str, Any]:
        """Process batch with circuit breaker protection"""
        if self.circuit_breaker and not self.circuit_breaker.should_allow_request():
            return {
                "status": "circuit_breaker_open",
                "error": "Circuit breaker is open",
                "processed": 0,
                "failed": len(batch_items)
            }

        try:
            result = await self._process_batch_internal(batch_items)

            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            return result

        except Exception as e:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            # Move failed items to dead letter queue
            if self.dead_letter_queue:
                for item in batch_items:
                    try:
                        await self.dead_letter_queue.put(item)
                    except asyncio.QueueFull:
                        self.logger.warning("Dead letter queue is full, dropping item")

            raise e

    async def _process_batch_internal(self, batch_items: List[Any]) -> Dict[str, Any]:
        """Internal batch processing with enhanced features"""
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()

        # OpenTelemetry tracing
        if self.tracer:
            with self.tracer.start_as_current_span("batch_processing") as span:
                span.set_attribute("batch_id", batch_id)
                span.set_attribute("batch_size", len(batch_items))
                span.set_attribute("processing_mode", self.config.processing_mode.value)

                result = await self._process_with_mode(batch_items, batch_id)

                span.set_attribute("processed_items", result.get("processed", 0))
                span.set_attribute("failed_items", result.get("failed", 0))

                return result
        else:
            return await self._process_with_mode(batch_items, batch_id)

    async def _process_with_mode(self, batch_items: List[Any], batch_id: str) -> Dict[str, Any]:
        """Process batch items based on configured processing mode"""
        try:
            # Apply intelligent partitioning
            if self.config.enable_intelligent_partitioning:
                partitions = await self._partition_data(batch_items)
            else:
                partitions = [batch_items]  # Single partition

            # Process partitions based on mode
            if self.config.processing_mode == ProcessingMode.DISTRIBUTED_RAY:
                results = await self._process_partitions_ray(partitions, batch_id)
            elif self.config.processing_mode == ProcessingMode.DISTRIBUTED_DASK:
                results = await self._process_partitions_dask(partitions, batch_id)
            elif self.config.processing_mode == ProcessingMode.STREAM_PROCESSING:
                results = await self._process_partitions_stream(partitions, batch_id)
            else:
                results = await self._process_partitions_local(partitions, batch_id)

            # Aggregate results
            total_processed = sum(r.get("processed", 0) for r in results)
            total_failed = sum(r.get("failed", 0) for r in results)

            # Record metrics
            processing_time = time.time() - time.time()
            if self.meter:
                self.batch_counter.add(1)
                self.item_counter.add(total_processed)
                self.error_counter.add(total_failed)
                self.processing_time_histogram.record(processing_time)

            return {
                "status": "success",
                "batch_id": batch_id,
                "processed": total_processed,
                "failed": total_failed,
                "processing_time_ms": processing_time * 1000,
                "partitions": len(partitions),
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            if self.meter:
                self.error_counter.add(len(batch_items))

            return {
                "status": "error",
                "batch_id": batch_id,
                "error": str(e),
                "processed": 0,
                "failed": len(batch_items)
            }

    async def _partition_data(self, batch_items: List[Any]) -> List[List[Any]]:
        """Apply intelligent data partitioning"""
        if self.config.partition_strategy == PartitionStrategy.ROUND_ROBIN:
            return self._partition_round_robin(batch_items)
        elif self.config.partition_strategy == PartitionStrategy.HASH_BASED:
            return self._partition_hash_based(batch_items)
        elif self.config.partition_strategy == PartitionStrategy.SIZE_BASED:
            return self._partition_size_based(batch_items)
        elif self.config.partition_strategy == PartitionStrategy.CONTENT_AWARE:
            return await self._partition_content_aware(batch_items)
        else:
            return [batch_items]  # No partitioning

    def _partition_round_robin(self, batch_items: List[Any]) -> List[List[Any]]:
        """Partition data using round-robin strategy"""
        num_partitions = min(self.worker_stats.current_workers, len(batch_items))
        partitions = [[] for _ in range(num_partitions)]

        for i, item in enumerate(batch_items):
            partitions[i % num_partitions].append(item)

        return [p for p in partitions if p]  # Remove empty partitions

    def _partition_hash_based(self, batch_items: List[Any]) -> List[List[Any]]:
        """Partition data using hash-based strategy"""
        num_partitions = min(self.worker_stats.current_workers, len(batch_items))
        partitions = [[] for _ in range(num_partitions)]

        for item in batch_items:
            # Use hash of item string representation for partitioning
            item_hash = hash(str(item))
            partition_idx = item_hash % num_partitions
            partitions[partition_idx].append(item)

        return [p for p in partitions if p]

    def _partition_size_based(self, batch_items: List[Any]) -> List[List[Any]]:
        """Partition data based on estimated processing size"""
        target_partition_size = max(1, len(batch_items) // self.worker_stats.current_workers)
        partitions = []

        current_partition = []
        for item in batch_items:
            current_partition.append(item)
            if len(current_partition) >= target_partition_size:
                partitions.append(current_partition)
                current_partition = []

        if current_partition:
            partitions.append(current_partition)

        return partitions

    async def _partition_content_aware(self, batch_items: List[Any]) -> List[List[Any]]:
        """Partition data based on content analysis"""
        # Simple content-aware partitioning based on item type/structure
        partitions = defaultdict(list)

        for item in batch_items:
            # Group by item type or key structure
            if isinstance(item, dict):
                key_signature = tuple(sorted(item.keys()))
                partitions[key_signature].append(item)
            else:
                partitions[type(item).__name__].append(item)

        return list(partitions.values())

    async def _process_partitions_local(self, partitions: List[List[Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Process partitions locally with concurrency"""
        tasks = []
        for i, partition in enumerate(partitions):
            task = asyncio.create_task(
                self._process_partition_local(partition, f"{batch_id}_partition_{i}")
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_partition_local(self, partition: List[Any], partition_id: str) -> Dict[str, Any]:
        """Process a single partition locally"""
        processed = 0
        failed = 0
        errors = []

        for item in partition:
            try:
                # Process individual item (placeholder - implement actual processing logic)
                await asyncio.sleep(0.01)  # Simulate processing time
                processed += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))

        return {
            "partition_id": partition_id,
            "processed": processed,
            "failed": failed,
            "errors": errors
        }

    async def _process_partitions_ray(self, partitions: List[List[Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Process partitions using Ray (fallback to local if Ray not available)"""
        if not RAY_AVAILABLE:
            return await self._process_partitions_local(partitions, batch_id)

        # Ray processing implementation would go here
        # For now, fallback to local processing
        return await self._process_partitions_local(partitions, batch_id)

    async def _process_partitions_dask(self, partitions: List[List[Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Process partitions using Dask (fallback to local if Dask not available)"""
        if not DASK_AVAILABLE or not self.distributed_client:
            return await self._process_partitions_local(partitions, batch_id)

        # Dask processing implementation would go here
        # For now, fallback to local processing
        return await self._process_partitions_local(partitions, batch_id)

    async def _process_partitions_stream(self, partitions: List[List[Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Process partitions using stream processing (fallback to local if Kafka not available)"""
        if not KAFKA_AVAILABLE or not self.stream_producer:
            return await self._process_partitions_local(partitions, batch_id)

        # Stream processing implementation would go here
        # For now, fallback to local processing
        return await self._process_partitions_local(partitions, batch_id)

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including 2025 features"""
        base_metrics = {
            "queue_size": self.get_queue_size(),
            "processing": self.processing,
            "worker_stats": {
                "current_workers": self.worker_stats.current_workers,
                "target_workers": self.worker_stats.target_workers,
                "queue_depth": self.worker_stats.queue_depth,
                "avg_processing_time_ms": self.worker_stats.avg_processing_time_ms,
                "cpu_utilization": self.worker_stats.cpu_utilization,
                "memory_utilization": self.worker_stats.memory_utilization
            },
            "configuration": {
                "processing_mode": self.config.processing_mode.value,
                "partition_strategy": self.config.partition_strategy.value,
                "worker_scaling_mode": self.config.worker_scaling_mode.value,
                "circuit_breaker_enabled": self.config.enable_circuit_breaker,
                "opentelemetry_enabled": self.config.enable_opentelemetry
            }
        }

        # Add circuit breaker status
        if self.circuit_breaker:
            base_metrics["circuit_breaker"] = {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count
            }

        # Add dead letter queue status
        if self.dead_letter_queue:
            base_metrics["dead_letter_queue"] = {
                "size": self.dead_letter_queue.qsize(),
                "max_size": self.config.max_dead_letter_items
            }

        # Add traditional metrics if available
        if self.metrics:
            base_metrics.update(self.metrics)

        return base_metrics

    async def get_dead_letter_items(self, max_items: int = 100) -> List[Any]:
        """Retrieve items from dead letter queue for inspection"""
        if not self.dead_letter_queue:
            return []

        items = []
        count = 0
        while count < max_items and not self.dead_letter_queue.empty():
            try:
                item = self.dead_letter_queue.get_nowait()
                items.append(item)
                count += 1
            except asyncio.QueueEmpty:
                break

        return items

    async def reprocess_dead_letter_items(self, max_items: int = 10) -> Dict[str, Any]:
        """Reprocess items from dead letter queue"""
        if not self.dead_letter_queue:
            return {"status": "no_dead_letter_queue", "reprocessed": 0}

        items = await self.get_dead_letter_items(max_items)
        if not items:
            return {"status": "no_items", "reprocessed": 0}

        try:
            result = await self.process_with_circuit_breaker(items)
            return {
                "status": "success",
                "reprocessed": result.get("processed", 0),
                "failed": result.get("failed", 0)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "reprocessed": 0
            }
