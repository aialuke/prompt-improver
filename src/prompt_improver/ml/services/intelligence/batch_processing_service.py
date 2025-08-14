"""ML Batch Processing Service.

Provides parallel batch processing orchestration for ML operations.
Extracted from intelligence_processor.py god object to follow single responsibility principle.

Performance Target: <500ms for batch operations
Memory Target: <100MB for batch processing
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
import math

from prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    BatchProcessingServiceProtocol,
    IntelligenceResult,
    MLCircuitBreakerServiceProtocol,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    default_batch_size: int = 25
    max_batch_size: int = 100
    min_batch_size: int = 5
    max_workers: int = 5
    timeout_seconds: float = 300.0
    memory_limit_mb: int = 100
    retry_failed_batches: bool = True


@dataclass
class BatchMetrics:
    """Metrics for batch processing operations."""
    total_items: int
    batch_count: int
    successful_batches: int
    failed_batches: int
    processing_time_ms: float
    items_per_second: float
    memory_usage_mb: float


class BatchProcessingService:
    """ML Batch Processing Service.
    
    Handles parallel batch processing with memory management, performance optimization,
    and fault tolerance for large-scale ML operations.
    """
    
    def __init__(
        self,
        ml_repository: MLRepositoryProtocol,
        circuit_breaker_service: MLCircuitBreakerServiceProtocol,
        config: Optional[BatchProcessingConfig] = None
    ):
        """Initialize batch processing service.
        
        Args:
            ml_repository: ML repository for data access
            circuit_breaker_service: Circuit breaker protection service
            config: Batch processing configuration
        """
        self._ml_repository = ml_repository
        self._circuit_breaker_service = circuit_breaker_service
        self._config = config or BatchProcessingConfig()
        self._metrics_registry = get_metrics_registry()
        self._active_batches: Dict[str, asyncio.Task] = {}
        self._batch_results: Dict[str, List[Any]] = {}
        
        logger.info(f"BatchProcessingService initialized with config: {self._config}")
    
    async def process_parallel_batches(
        self, 
        data: List[Dict[str, Any]], 
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None
    ) -> IntelligenceResult:
        """Process data in parallel batches.
        
        Args:
            data: Data to process in batches
            batch_size: Size of each batch (default from config)
            max_workers: Maximum parallel workers (default from config)
            
        Returns:
            Intelligence result with batch processing results
        """
        start_time = datetime.now(timezone.utc)
        
        if not data:
            return IntelligenceResult(
                success=True,
                data={"results": [], "batch_metrics": None},
                confidence=1.0,
                processing_time_ms=0.0,
                cache_hit=False
            )
        
        # Use provided values or defaults
        effective_batch_size = batch_size or self._config.default_batch_size
        effective_max_workers = max_workers or self._config.max_workers
        
        try:
            # Optimize batch size based on data size and memory constraints
            optimized_batch_size = await self.optimize_batch_size(
                len(data), 
                self._config.memory_limit_mb
            )
            
            if optimized_batch_size != effective_batch_size:
                logger.info(f"Optimized batch size from {effective_batch_size} to {optimized_batch_size}")
                effective_batch_size = optimized_batch_size
            
            # Calculate batch ranges
            batch_ranges = await self.calculate_batch_ranges(len(data), effective_batch_size)
            
            # Process batches with circuit breaker protection
            batch_results = await self._circuit_breaker_service.call_with_breaker(
                "batch_processing",
                self._process_batches_internal,
                data,
                batch_ranges,
                effective_max_workers
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Calculate metrics
            batch_metrics = BatchMetrics(
                total_items=len(data),
                batch_count=len(batch_ranges),
                successful_batches=batch_results["successful_batches"],
                failed_batches=batch_results["failed_batches"],
                processing_time_ms=processing_time,
                items_per_second=len(data) / (processing_time / 1000) if processing_time > 0 else 0,
                memory_usage_mb=batch_results.get("peak_memory_mb", 0)
            )
            
            # Record metrics
            self._record_batch_metrics(batch_metrics)
            
            return IntelligenceResult(
                success=batch_results["success"],
                data={
                    "results": batch_results["results"],
                    "batch_metrics": batch_metrics.__dict__,
                    "processing_summary": {
                        "total_items": len(data),
                        "successful_items": batch_results["successful_items"],
                        "failed_items": batch_results["failed_items"],
                        "throughput_items_per_second": batch_metrics.items_per_second
                    }
                },
                confidence=batch_results["overall_confidence"],
                processing_time_ms=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._metrics_registry.increment(
                "ml_batch_processing_operations_total",
                tags={"service": "batch_processing", "result": "error"}
            )
            
            logger.error(f"Batch processing failed: {e}")
            
            return IntelligenceResult(
                success=False,
                data={"results": [], "error_details": str(e)},
                confidence=0.0,
                processing_time_ms=processing_time,
                cache_hit=False,
                error_message=str(e)
            )
    
    async def _process_batches_internal(
        self,
        data: List[Dict[str, Any]],
        batch_ranges: List[Tuple[int, int]],
        max_workers: int
    ) -> Dict[str, Any]:
        """Internal batch processing implementation.
        
        Args:
            data: Data to process
            batch_ranges: List of (start, end) ranges for batches
            max_workers: Maximum parallel workers
            
        Returns:
            Batch processing results
        """
        semaphore = asyncio.Semaphore(max_workers)
        results = []
        successful_batches = 0
        failed_batches = 0
        successful_items = 0
        failed_items = 0
        peak_memory_mb = 0
        
        async def process_single_batch(batch_id: int, start: int, end: int) -> Dict[str, Any]:
            """Process a single batch of data."""
            nonlocal successful_batches, failed_batches, successful_items, failed_items, peak_memory_mb
            
            async with semaphore:
                try:
                    batch_data = data[start:end]
                    batch_start_time = datetime.now(timezone.utc)
                    
                    logger.debug(f"Processing batch {batch_id}: items {start}-{end}")
                    
                    # Simulate batch processing (in real implementation, this would call ML models)
                    batch_result = await self._process_batch_data(batch_data, batch_id)
                    
                    processing_time = (datetime.now(timezone.utc) - batch_start_time).total_seconds() * 1000
                    
                    # Update metrics
                    if batch_result["success"]:
                        successful_batches += 1
                        successful_items += len(batch_data)
                    else:
                        failed_batches += 1
                        failed_items += len(batch_data)
                    
                    # Track memory usage (simulated)
                    estimated_memory = len(batch_data) * 0.1  # Rough estimate
                    peak_memory_mb = max(peak_memory_mb, estimated_memory)
                    
                    return {
                        "batch_id": batch_id,
                        "start_index": start,
                        "end_index": end,
                        "items_processed": len(batch_data),
                        "processing_time_ms": processing_time,
                        "success": batch_result["success"],
                        "results": batch_result["results"],
                        "confidence": batch_result.get("confidence", 0.0)
                    }
                    
                except Exception as e:
                    failed_batches += 1
                    failed_items += len(data[start:end])
                    
                    logger.error(f"Batch {batch_id} failed: {e}")
                    
                    return {
                        "batch_id": batch_id,
                        "start_index": start,
                        "end_index": end,
                        "items_processed": 0,
                        "processing_time_ms": 0,
                        "success": False,
                        "error": str(e),
                        "confidence": 0.0
                    }
        
        # Create tasks for all batches
        tasks = []
        for batch_id, (start, end) in enumerate(batch_ranges):
            task = asyncio.create_task(
                process_single_batch(batch_id, start, end)
            )
            tasks.append(task)
            self._active_batches[f"batch_{batch_id}"] = task
        
        try:
            # Wait for all batches to complete with timeout
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.timeout_seconds
            )
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_batches += 1
                    logger.error(f"Batch task failed with exception: {result}")
                else:
                    results.append(result)
            
        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {self._config.timeout_seconds} seconds")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            failed_batches = len(batch_ranges)
            
        finally:
            # Clean up active batch tracking
            self._active_batches.clear()
        
        # Calculate overall confidence
        successful_results = [r for r in results if r.get("success", False)]
        overall_confidence = (
            sum(r.get("confidence", 0.0) for r in successful_results) / len(successful_results)
            if successful_results else 0.0
        )
        
        return {
            "success": successful_batches > 0,
            "results": results,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "successful_items": successful_items,
            "failed_items": failed_items,
            "overall_confidence": overall_confidence,
            "peak_memory_mb": peak_memory_mb
        }
    
    async def _process_batch_data(self, batch_data: List[Dict[str, Any]], batch_id: int) -> Dict[str, Any]:
        """Process a single batch of data.
        
        Args:
            batch_data: Data for this batch
            batch_id: Unique batch identifier
            
        Returns:
            Batch processing results
        """
        try:
            # Simulate ML processing for each item in the batch
            processed_items = []
            
            for item in batch_data:
                # In real implementation, this would call actual ML models
                processed_item = {
                    "item_id": item.get("id", f"item_{len(processed_items)}"),
                    "processed": True,
                    "confidence": 0.8 + (len(processed_items) % 3) * 0.05,  # Simulated confidence
                    "processing_result": {
                        "effectiveness": 0.7 + (hash(str(item)) % 100) / 100 * 0.3,
                        "category": "processed",
                        "metadata": {
                            "batch_id": batch_id,
                            "processing_timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    }
                }
                processed_items.append(processed_item)
            
            # Calculate batch-level metrics
            confidences = [item["confidence"] for item in processed_items]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "success": True,
                "results": processed_items,
                "confidence": avg_confidence,
                "batch_summary": {
                    "items_processed": len(processed_items),
                    "avg_confidence": avg_confidence,
                    "min_confidence": min(confidences) if confidences else 0.0,
                    "max_confidence": max(confidences) if confidences else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Batch data processing failed for batch {batch_id}: {e}")
            return {
                "success": False,
                "results": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def calculate_batch_ranges(self, total_items: int, batch_size: int) -> List[Tuple[int, int]]:
        """Calculate optimal batch ranges for processing.
        
        Args:
            total_items: Total number of items to process
            batch_size: Size of each batch
            
        Returns:
            List of (start, end) tuples for batch ranges
        """
        if total_items <= 0:
            return []
        
        # Ensure batch size is within bounds
        effective_batch_size = max(
            self._config.min_batch_size,
            min(batch_size, self._config.max_batch_size)
        )
        
        ranges = []
        for start in range(0, total_items, effective_batch_size):
            end = min(start + effective_batch_size, total_items)
            ranges.append((start, end))
        
        logger.debug(f"Calculated {len(ranges)} batch ranges for {total_items} items")
        return ranges
    
    async def manage_parallel_workers(self, tasks: List[Callable], max_workers: int) -> List[Any]:
        """Manage parallel worker execution.
        
        Args:
            tasks: List of tasks to execute
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of task results
        """
        if not tasks:
            return []
        
        semaphore = asyncio.Semaphore(max_workers)
        
        async def execute_with_semaphore(task: Callable) -> Any:
            async with semaphore:
                return await task()
        
        # Create coroutines with semaphore control
        controlled_tasks = [execute_with_semaphore(task) for task in tasks]
        
        # Execute all tasks and return results
        return await asyncio.gather(*controlled_tasks, return_exceptions=True)
    
    async def optimize_batch_size(self, data_size: int, memory_limit_mb: int) -> int:
        """Calculate optimal batch size based on data size and memory constraints.
        
        Args:
            data_size: Total number of items to process
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Optimized batch size
        """
        # Estimate memory per item (rough heuristic)
        estimated_memory_per_item_mb = 0.1  # 100KB per item
        
        # Calculate maximum batch size based on memory
        max_batch_by_memory = int(memory_limit_mb / estimated_memory_per_item_mb)
        
        # Consider data size for optimal parallelization
        optimal_batch_count = min(self._config.max_workers * 2, math.ceil(data_size / 10))
        optimal_batch_by_parallelization = max(1, data_size // optimal_batch_count)
        
        # Choose the most restrictive constraint
        optimized_size = min(
            max_batch_by_memory,
            optimal_batch_by_parallelization,
            self._config.max_batch_size
        )
        
        # Ensure minimum batch size
        optimized_size = max(optimized_size, self._config.min_batch_size)
        
        logger.debug(
            f"Batch size optimization: data_size={data_size}, "
            f"memory_limit={memory_limit_mb}MB, "
            f"max_by_memory={max_batch_by_memory}, "
            f"optimal_by_parallelization={optimal_batch_by_parallelization}, "
            f"final_size={optimized_size}"
        )
        
        return optimized_size
    
    def _record_batch_metrics(self, metrics: BatchMetrics) -> None:
        """Record batch processing metrics.
        
        Args:
            metrics: Batch metrics to record
        """
        self._metrics_registry.increment(
            "ml_batch_processing_operations_total",
            tags={"service": "batch_processing", "result": "success"}
        )
        
        self._metrics_registry.record_value(
            "ml_batch_processing_duration_ms",
            metrics.processing_time_ms,
            tags={"service": "batch_processing"}
        )
        
        self._metrics_registry.record_value(
            "ml_batch_processing_items_per_second",
            metrics.items_per_second,
            tags={"service": "batch_processing"}
        )
        
        self._metrics_registry.record_value(
            "ml_batch_processing_memory_usage_mb",
            metrics.memory_usage_mb,
            tags={"service": "batch_processing"}
        )
        
        self._metrics_registry.set_gauge(
            "ml_batch_processing_success_rate",
            metrics.successful_batches / (metrics.successful_batches + metrics.failed_batches) 
            if (metrics.successful_batches + metrics.failed_batches) > 0 else 0,
            tags={"service": "batch_processing"}
        )
        
        logger.info(
            f"Batch processing completed: {metrics.total_items} items, "
            f"{metrics.batch_count} batches, "
            f"{metrics.processing_time_ms:.1f}ms, "
            f"{metrics.items_per_second:.1f} items/sec"
        )
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status.
        
        Returns:
            Processing status information
        """
        active_count = len(self._active_batches)
        
        return {
            "active_batches": active_count,
            "batch_details": {
                batch_id: {
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "exception": str(task.exception()) if task.done() and task.exception() else None
                }
                for batch_id, task in self._active_batches.items()
            },
            "config": self._config.__dict__,
            "is_processing": active_count > 0
        }
    
    async def cancel_all_batches(self) -> int:
        """Cancel all active batch operations.
        
        Returns:
            Number of batches cancelled
        """
        cancelled_count = 0
        
        for batch_id, task in list(self._active_batches.items()):
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        self._active_batches.clear()
        
        logger.info(f"Cancelled {cancelled_count} active batch operations")
        return cancelled_count