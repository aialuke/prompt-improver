import logging
import asyncio
import heapq
import random
import time
from asyncio import sleep, create_task, gather
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field

from prompt_improver.database import get_session
from prompt_improver.services.ml_integration import get_ml_service

async def periodic_batch_processor_coroutine(batch_processor: "BatchProcessor") -> None:
    """Periodic coroutine that triggers batch processing at configured intervals.
    
    This coroutine runs continuously and calls BatchProcessor.process_training_batch
    every BATCH_TIMEOUT seconds. It handles queue checking and batch formation.
    
    Args:
        batch_processor: The BatchProcessor instance to trigger periodic processing on
    """
    logger = logging.getLogger(f'{__name__}.PeriodicBatchProcessor')
    logger.info(f"Starting periodic batch processor with {batch_processor.config.batch_timeout}s interval")
    
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
                logger.info(f"Triggering periodic batch processing for {len(training_batch)} items")
                result = await batch_processor.process_training_batch(training_batch)
                
                if result.get("status") == "success":
                    logger.info(f"Periodic batch processing completed: {result.get('processed_records', 0)} records")
                else:
                    logger.warning(f"Periodic batch processing failed: {result.get('error', 'unknown error')}")
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
    batch_size: int = Field(default=10, ge=1, le=100, description="Number of records to process in each batch")
    concurrency: int = Field(default=3, ge=1, le=20, description="Number of concurrent processing workers")
    timeout: int = Field(default=30000, ge=1000, description="Timeout per batch in milliseconds")
    batch_timeout: int = Field(default=30, ge=1, le=300, description="Timeout between periodic batch processing triggers in seconds")
    
    # Retry and exponential backoff settings
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts per record")
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Base delay for exponential backoff in seconds")
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0, description="Maximum delay for exponential backoff in seconds")
    jitter: bool = Field(default=True, description="Add jitter to prevent thundering herd")
    
    # Queue management
    max_queue_size: int = Field(default=1000, ge=10, le=10000, description="Maximum size of the internal queue")
    rate_limit_per_second: Optional[float] = Field(default=None, ge=0.1, description="Rate limit in requests per second")
    
    # Processing behavior
    dry_run: bool = Field(default=False, description="Run in dry-run mode without actual processing")
    enable_priority_queue: bool = Field(default=True, description="Enable priority-based processing")
    
    # Monitoring and logging
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True
        
    @property
    def rate_limit_delay(self) -> Optional[float]:
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
        self.logger = logging.getLogger(f'{__name__}.AsyncQueueWrapper')
        self.processing = False
        self.last_request_time = 0.0
        
        # Rate limiting semaphore
        self.rate_semaphore = asyncio.Semaphore(config.concurrency)
        
    async def put(self, item: Dict[str, Any], priority: int = 50) -> None:
        """Put an item into the queue with optional priority."""
        queue_item = {
            "item": item,
            "priority": priority,
            "timestamp": time.time(),
            "attempts": 0,
            "next_retry": None
        }
        
        try:
            await self.queue.put(queue_item)
            self.logger.debug(f"Enqueued item with priority {priority}")
        except asyncio.QueueFull:
            self.logger.warning("Queue is full, dropping item")
            raise
    
    async def get(self) -> Optional[Dict[str, Any]]:
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
    
    async def put_failed(self, item: Dict[str, Any]) -> None:
        """Put a failed item into the failure queue for retry."""
        item["attempts"] += 1
        
        if item["attempts"] >= self.config.max_attempts:
            self.logger.error(f"Max attempts ({self.config.max_attempts}) reached for item, giving up")
            return
            
        # Calculate exponential backoff delay
        delay = min(
            self.config.base_delay * (2 ** (item["attempts"] - 1)),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.5)
            delay *= (1 + jitter)
            
        item["next_retry"] = time.time() + delay
        
        # Re-enqueue with exponential backoff
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
        
    async def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """Wait for all items in the queue to be processed."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Queue processing timeout after {timeout} seconds")

@dataclass(order=True)
class PriorityRecord:
    """Wrapper for priority queue records with priority ordering."""
    priority: int
    record: Dict[str, Any] = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    attempts: int = field(default=0, compare=False)
    next_retry: Optional[float] = field(default=None, compare=False)


class PriorityQueue:
    """Heap-based priority queue with support for priority:int on records."""
    
    def __init__(self):
        self._queue = []
        self._index = 0
        self._removed = set()
        
    def enqueue(self, record: Dict[str, Any], priority: int) -> None:
        """Add a record with priority to the queue."""
        priority_record = PriorityRecord(
            priority=priority,
            record=record,
            timestamp=time.time()
        )
        heapq.heappush(self._queue, priority_record)
        
    def dequeue(self) -> Optional[PriorityRecord]:
        """Remove and return the highest priority record (lowest priority number)."""
        while self._queue:
            priority_record = heapq.heappop(self._queue)
            if id(priority_record) not in self._removed:
                return priority_record
        return None
        
    def peek(self) -> Optional[PriorityRecord]:
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
        return len(self._queue) == 0 or all(id(item) in self._removed for item in self._queue)
        
    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._queue) - len(self._removed)


class BatchProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize with new configuration system
        if config is None:
            self.config = BatchProcessorConfig()
        elif isinstance(config, dict):
            # Convert legacy dict config to new config system
            new_config = {}
            if "batchSize" in config:
                new_config["batch_size"] = config["batchSize"]
            if "maxAttempts" in config:
                new_config["max_attempts"] = config["maxAttempts"]
            if "baseDelay" in config:
                new_config["base_delay"] = config["baseDelay"]
            if "maxDelay" in config:
                new_config["max_delay"] = config["maxDelay"]
            if "dryRun" in config:
                new_config["dry_run"] = config["dryRun"]
            if "jitter" in config:
                new_config["jitter"] = config["jitter"]
            if "concurrency" in config:
                new_config["concurrency"] = config["concurrency"]
            if "timeout" in config:
                new_config["timeout"] = config["timeout"]
            
            self.config = BatchProcessorConfig(**new_config)
        else:
            self.config = config
            
        # Set up logging
        self.logger = logging.getLogger('BatchProcessor')
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
        self.metrics = {
            "processed": 0,
            "failed": 0,
            "retries": 0,
            "start_time": time.time()
        } if self.config.metrics_enabled else None
        
    async def enqueue(self, record: Dict[str, Any], priority: int = 50) -> None:
        """Public async API to enqueue a record with priority.
        
        Supports both priority queue and async queue based on configuration.
        """
        if self.config.enable_priority_queue and self.priority_queue:
            # Use priority queue for immediate processing
            self.priority_queue.enqueue(record, priority)
            self.logger.debug(f"Enqueued record to priority queue with priority {priority}")
            
            # Start processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_queue())
        else:
            # Use async queue wrapper with exponential backoff
            await self.async_queue.put(record, priority)
            self.logger.debug(f"Enqueued record to async queue with priority {priority}")
            
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
                if priority_record.next_retry and time.time() < priority_record.next_retry:
                    # Re-enqueue for later processing
                    self.priority_queue.enqueue(priority_record.record, priority_record.priority)
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
                else:
                    # Update metrics
                    if self.metrics:
                        self.metrics["processed"] += 1
                    
                # Mark task as done for queue.join()
                self.async_queue.queue.task_done()
                
        finally:
            self.processing = False
            
    async def _process_queue_item(self, queue_item: Dict[str, Any]) -> bool:
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
                self.logger.info(f"[DRY RUN] Would process record: {priority_record.record}")
                return True
                
            await self._persist_to_database(priority_record.record)
            self.logger.debug(f"Successfully processed record with priority {priority_record.priority}")
            
            # Update metrics
            if self.metrics:
                self.metrics["processed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing record: {e}")
            if self.metrics:
                self.metrics["failed"] += 1
            return False
            
    async def _persist_to_database(self, record: Dict[str, Any]) -> None:
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
                            "processed_at": datetime.utcnow().isoformat(),
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
        """Handle retry logic with exponential backoff and jitter."""
        priority_record.attempts += 1
        max_attempts = self.config.get("maxAttempts", 3)
        
        if priority_record.attempts >= max_attempts:
            self.logger.error(f"Max attempts ({max_attempts}) reached for record, giving up")
            return
            
        # Calculate exponential backoff delay
        base_delay = self.config.get("baseDelay", 1.0)
        max_delay = self.config.get("maxDelay", 60.0)
        
        # Exponential backoff: delay = base_delay * (2 ^ (attempts - 1))
        delay = min(base_delay * (2 ** (priority_record.attempts - 1)), max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.get("jitter", True):
            jitter = random.uniform(0.1, 0.5)
            delay *= (1 + jitter)
            
        priority_record.next_retry = time.time() + delay
        
        # Re-enqueue with same priority but for later processing
        self.priority_queue.enqueue(priority_record.record, priority_record.priority)
        
        self.logger.warning(
            f"Retrying record (attempt {priority_record.attempts}/{max_attempts}) "
            f"after {delay:.2f} seconds"
        )
        
    def get_queue_size(self) -> int:
        """Return the current size of the priority queue."""
        return self.priority_queue.size() if self.priority_queue else self.async_queue.qsize()
    
    def _has_pending_items(self) -> bool:
        """Check if there are pending items in the queue for batch processing."""
        if self.config.enable_priority_queue and self.priority_queue:
            return not self.priority_queue.empty()
        else:
            return not self.async_queue.empty()
    
    async def _get_training_batch(self) -> List[Dict[str, Any]]:
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
                    if priority_record.next_retry is None or time.time() >= priority_record.next_retry:
                        batch.append(priority_record.record)
                    else:
                        # Re-enqueue for later processing
                        self.priority_queue.enqueue(priority_record.record, priority_record.priority)
        else:
            # Extract from async queue
            while len(batch) < batch_size and not self.async_queue.empty():
                try:
                    queue_item = await asyncio.wait_for(self.async_queue.get(), timeout=0.1)
                    if queue_item is not None:
                        # Check if it's time to process (not in retry backoff)
                        if queue_item.get("next_retry") is None or time.time() >= queue_item["next_retry"]:
                            batch.append(queue_item["item"])
                        else:
                            # Re-enqueue for later processing
                            await self.async_queue.put(queue_item["item"], queue_item["priority"])
                except asyncio.TimeoutError:
                    break
                    
        return batch

    async def process_batch(self, test_cases: List[Dict[str, Any]], options: Dict[str, Any] = None) -> Dict[str, Any]:
        if not test_cases:
            raise ValueError('Test cases must be a non-empty list')

        batchSize = options.get("batchSize") or self.config["batchSize"]
        batches = [test_cases[i:i + batchSize] for i in range(0, len(test_cases), batchSize)]

        results = {
            "processed": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }

        for index, batch in enumerate(batches):
            self.logger.debug(f"Processing batch {index + 1}/{len(batches)}")
            batch_results = await self.process_single_batch(batch)
            results["processed"] += batch_results["processed"]
            results["failed"] += batch_results["failed"]

            results["results"].extend(batch_results["results"])
            results["errors"].extend(batch_results["errors"])

            await sleep(self.config["retryDelay"] / 1000)  # Rate limiting

        results["executionTime"] = len(batches) * self.config["timeout"]  # Placeholder for real timing logic
        return results

    async def process_single_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_results = {
            "processed": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }

        async def process_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def process_training_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process training batch by sending to ML integration service.
        
        Args:
            batch: List of training records to process
            
        Returns:
            Result of batch processing operation
        """
        try:
            start_time = time.time()
            
            # Get ML service instance
            ml_service = await get_ml_service()
            
            # Send batch to ML integration stub
            result = await ml_service.send_training_batch(batch)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"Processed training batch: {result.get('status', 'unknown')} "
                           f"({len(batch)} records in {processing_time:.2f}ms)")
            
            return {
                "status": "success",
                "processed_records": len(batch),
                "processing_time_ms": processing_time,
                "ml_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process training batch: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }

