"""Enhanced Memory Safety Guard for ML Pipeline Orchestration.

Implements 2025 memory management best practices including:
- Async memory monitoring patterns
- Resource lifecycle management
- Event-driven memory alerts
- ML-specific memory protection
- Comprehensive resource monitoring
"""

import asyncio
import gc
import logging
import resource
import time
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryThreatLevel(Enum):
    """Memory threat levels for resource monitoring."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class MemoryEvent(BaseModel):
    """Memory event for monitoring and alerting."""

    event_type: str
    threat_level: MemoryThreatLevel
    memory_usage_mb: float
    operation_name: str | None = Field(default=None)
    component_name: str | None = Field(default=None)
    memory_delta_mb: float | None = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResourceStats(BaseModel):
    """Comprehensive resource statistics."""

    current_memory_mb: float
    peak_memory_mb: float
    initial_memory_mb: float
    allocated_memory_mb: float
    memory_limit_mb: float
    usage_percent: float
    threat_level: MemoryThreatLevel
    gc_collections: int
    active_operations: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryGuard:
    """Enhanced memory safety and monitoring for ML pipeline orchestration.

    features:
    - Async memory monitoring patterns
    - Resource lifecycle management
    - Event-driven memory alerts
    - ML-specific memory protection
    - Comprehensive resource monitoring
    """

    def __init__(
        self,
        max_memory_mb: int = 500,
        max_buffer_size: int = 100 * 1024 * 1024,
        event_bus=None,
    ):
        self.max_memory_mb = max_memory_mb
        self.max_buffer_size = max_buffer_size  # 100MB default
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Memory usage tracking
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory

        # Enhanced monitoring
        self.memory_events: list[MemoryEvent] = []
        self.active_operations: dict[str, datetime] = {}
        self.gc_collections = 0
        self._monitoring_stats = {
            "total_operations": 0,
            "memory_warnings": 0,
            "memory_errors": 0,
            "gc_forced": 0,
        }

    def set_event_bus(self, event_bus):
        """Set event bus for memory event emission."""
        self.event_bus = event_bus
        logger.info("Event bus integrated with MemoryGuard for resource monitoring")

    async def _emit_memory_event(self, event: MemoryEvent):
        """Emit memory event for monitoring and alerting."""
        self.memory_events.append(event)

        if self.event_bus:
            try:
                # Import here to avoid circular imports
                from prompt_improver.ml.orchestration.events.event_types import (
                    EventType,
                    MLEvent,
                )

                # Choose specific event type based on threat level
                if event.threat_level == MemoryThreatLevel.CRITICAL:
                    event_type = EventType.RESOURCE_EXHAUSTED
                elif event.threat_level == MemoryThreatLevel.HIGH:
                    event_type = EventType.RESOURCE_WARNING
                else:
                    event_type = EventType.RESOURCE_ALLOCATED

                ml_event = MLEvent(
                    event_type=event_type,
                    source="memory_guard",
                    data={
                        "event_type": event.event_type,
                        "threat_level": event.threat_level.name.lower(),
                        "memory_usage_mb": event.memory_usage_mb,
                        "memory_delta_mb": event.memory_delta_mb,
                        "operation_name": event.operation_name,
                        "component_name": event.component_name,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )
                await self.event_bus.emit(ml_event)
            except Exception as e:
                logger.error(f"Failed to emit memory event: {e}")

        # Log memory events
        if event.threat_level in [MemoryThreatLevel.HIGH, MemoryThreatLevel.CRITICAL]:
            logger.error(
                f"MEMORY ALERT: {event.event_type} - {event.threat_level.name.lower()} - Usage: {event.memory_usage_mb:.1f}MB"
            )
        else:
            logger.info(
                f"Memory event: {event.event_type} - {event.threat_level.name.lower()} - Usage: {event.memory_usage_mb:.1f}MB"
            )

    def get_resource_stats(self) -> ResourceStats:
        """Get comprehensive resource statistics."""
        current_memory = self._get_memory_usage()
        allocated_memory = current_memory - self.initial_memory
        usage_percent = (current_memory / self.max_memory_mb) * 100

        # Determine threat level
        if usage_percent >= 95:
            threat_level = MemoryThreatLevel.CRITICAL
        elif usage_percent >= 85:
            threat_level = MemoryThreatLevel.HIGH
        elif usage_percent >= 70:
            threat_level = MemoryThreatLevel.MEDIUM
        else:
            threat_level = MemoryThreatLevel.LOW

        return ResourceStats(
            current_memory_mb=current_memory,
            peak_memory_mb=self.peak_memory,
            initial_memory_mb=self.initial_memory,
            allocated_memory_mb=allocated_memory,
            memory_limit_mb=self.max_memory_mb,
            usage_percent=usage_percent,
            threat_level=threat_level,
            gc_collections=self.gc_collections,
            active_operations=len(self.active_operations),
        )

    def check_memory_usage(self) -> dict[str, float]:
        """Check current memory usage and return statistics."""
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)

        memory_stats = {
            "current_mb": current_memory,
            "peak_mb": self.peak_memory,
            "initial_mb": self.initial_memory,
            "allocated_mb": current_memory - self.initial_memory,
            "limit_mb": self.max_memory_mb,
            "usage_percent": (current_memory / self.max_memory_mb) * 100,
        }

        # Log warning if approaching limit
        if memory_stats["usage_percent"] > 80:
            self.logger.warning(
                f"High memory usage: {memory_stats['usage_percent']:.1f}%"
            )

        return memory_stats

    async def check_memory_usage_async(
        self, operation_name: str = "unknown", component_name: str = None
    ) -> ResourceStats:
        """Async memory usage check with event emission."""
        stats = self.get_resource_stats()
        self._monitoring_stats["total_operations"] += 1

        # Emit memory event if significant
        if stats.threat_level >= MemoryThreatLevel.MEDIUM:
            if stats.threat_level >= MemoryThreatLevel.HIGH:
                self._monitoring_stats["memory_warnings"] += 1

            await self._emit_memory_event(
                MemoryEvent(
                    event_type="memory_check",
                    threat_level=stats.threat_level,
                    memory_usage_mb=stats.current_memory_mb,
                    operation_name=operation_name,
                    component_name=component_name,
                )
            )

        return stats

    def monitor_operation_async(self, operation_name: str, component_name: str = None):
        """Async context manager for monitoring memory during operations."""
        return AsyncMemoryMonitor(self, operation_name, component_name)

    def validate_buffer_size(
        self, data: bytes | np.ndarray | Any, operation: str = "unknown"
    ) -> bool:
        """Validate buffer size before operations to prevent overflow.

        Args:
            data: Data to validate
            operation: Description of operation for logging

        Returns:
            True if safe, False if too large

        Raises:
            MemoryError: If data exceeds safety limits
        """
        size = self._get_data_size(data)

        if size > self.max_buffer_size:
            error_msg = f"Buffer size {size:,} bytes exceeds limit {self.max_buffer_size:,} for {operation}"
            self.logger.error(error_msg)
            raise MemoryError(error_msg)

        # Check if operation would exceed memory limit
        projected_memory = self._get_memory_usage() + (size / 1024 / 1024)
        if projected_memory > self.max_memory_mb:
            error_msg = f"Operation {operation} would exceed memory limit: {projected_memory:.1f}MB > {self.max_memory_mb}MB"
            self.logger.error(error_msg)
            raise MemoryError(error_msg)

        return True

    def safe_frombuffer(
        self, buffer: bytes, dtype: np.dtype, count: int = -1
    ) -> np.ndarray:
        """Safely create numpy array from buffer with validation.

        Args:
            buffer: Bytes buffer
            dtype: Target numpy dtype
            count: Number of items (default: all)

        Returns:
            Numpy array

        Raises:
            MemoryError: If buffer validation fails
            ValueError: If buffer size is invalid
        """
        # Validate buffer size
        self.validate_buffer_size(buffer, f"frombuffer with dtype {dtype}")

        # Validate buffer size matches dtype requirements
        dtype_size = np.dtype(dtype).itemsize
        if len(buffer) % dtype_size != 0:
            raise ValueError(
                f"Buffer size {len(buffer)} not divisible by dtype size {dtype_size}"
            )

        # Calculate expected element count
        expected_count = len(buffer) // dtype_size
        if count != -1 and count > expected_count:
            raise ValueError(
                f"Requested count {count} exceeds available elements {expected_count}"
            )

        # Additional safety: limit array size
        max_elements = min(
            1_000_000, self.max_buffer_size // dtype_size
        )  # 1M elements or buffer limit
        actual_count = (
            min(expected_count, max_elements)
            if count == -1
            else min(count, max_elements)
        )

        if actual_count < expected_count:
            self.logger.warning(
                f"Truncating array from {expected_count} to {actual_count} elements for safety"
            )

        # Create array safely
        try:
            return np.frombuffer(
                buffer[: actual_count * dtype_size], dtype=dtype, count=actual_count
            )
        except Exception as e:
            self.logger.error(f"Failed to create array from buffer: {e}")
            raise

    def safe_tobytes(self, array: np.ndarray) -> bytes:
        """Safely convert numpy array to bytes with validation.

        Args:
            array: Numpy array to convert

        Returns:
            Bytes representation

        Raises:
            MemoryError: If array is too large
        """
        self.validate_buffer_size(array, "tobytes conversion")

        try:
            return array.tobytes()
        except Exception as e:
            self.logger.error(f"Failed to convert array to bytes: {e}")
            raise

    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring memory during operations."""
        return MemoryMonitor(self, operation_name)

    async def force_garbage_collection_async(self, operation_name: str = "gc_cleanup"):
        """Async force garbage collection with event emission."""
        memory_before = self._get_memory_usage()

        # Run garbage collection in executor to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, gc.collect)

        memory_after = self._get_memory_usage()
        freed_mb = memory_before - memory_after
        self.gc_collections += 1
        self._monitoring_stats["gc_forced"] += 1

        if freed_mb > 0:
            self.logger.info("Async garbage collection freed %.1f MB", freed_mb)

            # Emit memory event for significant cleanup
            await self._emit_memory_event(
                MemoryEvent(
                    event_type="garbage_collection",
                    threat_level=MemoryThreatLevel.LOW,
                    memory_usage_mb=memory_after,
                    memory_delta_mb=-freed_mb,
                    operation_name=operation_name,
                )
            )

        return freed_mb

    def force_garbage_collection(self):
        """Force garbage collection and return memory freed."""
        memory_before = self._get_memory_usage()
        gc.collect()
        memory_after = self._get_memory_usage()
        freed_mb = memory_before - memory_after
        self.gc_collections += 1

        if freed_mb > 0:
            self.logger.info("Garbage collection freed %.1f MB", freed_mb)

        return freed_mb

    async def validate_ml_operation_memory(
        self, data: Any, operation_name: str, component_name: str = None
    ) -> bool:
        """Validate memory requirements for ML operations with 2025 best practices.

        Args:
            data: ML data to validate (arrays, tensors, etc.)
            operation_name: Name of the ML operation
            component_name: Name of the component performing the operation

        Returns:
            True if memory validation passes

        Raises:
            MemoryError: If memory requirements exceed limits
        """
        # Check current memory state
        stats = await self.check_memory_usage_async(operation_name, component_name)

        # Validate buffer size
        try:
            self.validate_buffer_size(data, operation_name)
        except MemoryError as e:
            self._monitoring_stats["memory_errors"] += 1
            await self._emit_memory_event(
                MemoryEvent(
                    event_type="memory_validation_failed",
                    threat_level=MemoryThreatLevel.CRITICAL,
                    memory_usage_mb=stats.current_memory_mb,
                    operation_name=operation_name,
                    component_name=component_name,
                )
            )
            raise

        # Check for ML-specific memory patterns
        if isinstance(data, np.ndarray):
            # Check for memory-intensive operations
            data_size_mb = data.nbytes / 1024 / 1024
            if data_size_mb > 100:  # Large arrays
                if stats.usage_percent > 70:
                    error_msg = f"Large array operation {operation_name} would risk memory exhaustion: {data_size_mb:.1f}MB array with {stats.usage_percent:.1f}% memory usage"
                    self.logger.error(error_msg)
                    self._monitoring_stats["memory_errors"] += 1

                    await self._emit_memory_event(
                        MemoryEvent(
                            event_type="large_array_risk",
                            threat_level=MemoryThreatLevel.HIGH,
                            memory_usage_mb=stats.current_memory_mb,
                            operation_name=operation_name,
                            component_name=component_name,
                        )
                    )

                    raise MemoryError(error_msg)

        return True

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.process()
                return process.memory_info().rss / 1024 / 1024  # Convert to MB
            except Exception:
                pass

        # Fallback to resource module
        try:
            return (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )  # Linux: KB to MB
        except Exception:
            return 0.0

    def _get_data_size(self, data: Any) -> int:
        """Get size of data in bytes."""
        if isinstance(data, bytes):
            return len(data)
        if isinstance(data, np.ndarray):
            return data.nbytes
        if hasattr(data, "__sizeof__"):
            return data.__sizeof__()
        # Rough estimate for other types
        return len(str(data).encode("utf-8"))


class AsyncMemoryMonitor:
    """Async context manager for monitoring memory usage during operations."""

    def __init__(
        self, memory_guard: MemoryGuard, operation_name: str, component_name: str = None
    ):
        self.guard = memory_guard
        self.operation_name = operation_name
        self.component_name = component_name
        self.start_memory = None
        self.start_time = None
        self.operation_id = f"{operation_name}_{int(time.time() * 1000)}"

    async def __aenter__(self):
        self.start_memory = self.guard._get_memory_usage()
        self.start_time = time.time()

        # Register active operation
        self.guard.active_operations[self.operation_id] = datetime.now(UTC)

        self.guard.logger.debug(
            f"Starting async {self.operation_name} - Memory: {self.start_memory:.1f}MB"
        )

        # Emit start event
        await self.guard._emit_memory_event(
            MemoryEvent(
                event_type="operation_started",
                threat_level=MemoryThreatLevel.LOW,
                memory_usage_mb=self.start_memory,
                operation_name=self.operation_name,
                component_name=self.component_name,
            )
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end_memory = self.guard._get_memory_usage()
        end_time = time.time()

        # Remove from active operations
        self.guard.active_operations.pop(self.operation_id, None)

        memory_delta = end_memory - self.start_memory
        time_delta = end_time - self.start_time

        # Determine threat level based on memory delta
        if memory_delta > 100:  # 100MB increase
            threat_level = MemoryThreatLevel.HIGH
        elif memory_delta > 50:  # 50MB increase
            threat_level = MemoryThreatLevel.MEDIUM
        else:
            threat_level = MemoryThreatLevel.LOW

        level = (
            logging.WARNING
            if memory_delta > 50
            else logging.INFO
            if memory_delta > 10
            else logging.DEBUG
        )
        self.guard.logger.log(
            level,
            f"Completed async {self.operation_name} - Memory: {end_memory:.1f}MB "
            f"(Δ{memory_delta:+.1f}MB) in {time_delta:.2f}s",
        )

        # Update peak memory
        self.guard.peak_memory = max(self.guard.peak_memory, end_memory)

        # Emit completion event
        await self.guard._emit_memory_event(
            MemoryEvent(
                event_type="operation_completed",
                threat_level=threat_level,
                memory_usage_mb=end_memory,
                memory_delta_mb=memory_delta,
                operation_name=self.operation_name,
                component_name=self.component_name,
            )
        )

        # Force cleanup if memory delta is significant
        if memory_delta > 50:  # 50MB threshold
            await self.guard.force_garbage_collection_async(
                f"cleanup_{self.operation_name}"
            )


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""

    def __init__(self, memory_guard: MemoryGuard, operation_name: str):
        self.guard = memory_guard
        self.operation_name = operation_name
        self.start_memory = None
        self.start_time = None

    def __enter__(self):
        self.start_memory = self.guard._get_memory_usage()
        self.start_time = time.time()
        self.guard.logger.debug(
            f"Starting {self.operation_name} - Memory: {self.start_memory:.1f}MB"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        end_memory = self.guard._get_memory_usage()
        end_time = time.time()

        memory_delta = end_memory - self.start_memory
        time_delta = end_time - self.start_time

        level = logging.INFO if memory_delta > 10 else logging.DEBUG
        self.guard.logger.log(
            level,
            f"Completed {self.operation_name} - Memory: {end_memory:.1f}MB "
            f"(Δ{memory_delta:+.1f}MB) in {time_delta:.2f}s",
        )

        # Update peak memory
        self.guard.peak_memory = max(self.guard.peak_memory, end_memory)

        # Force cleanup if memory delta is significant
        if memory_delta > 50:  # 50MB threshold
            self.guard.force_garbage_collection()


# Global memory guard instance
_default_memory_guard = None


def get_memory_guard() -> MemoryGuard:
    """Get global memory guard instance."""
    global _default_memory_guard
    if _default_memory_guard is None:
        _default_memory_guard = MemoryGuard()
    return _default_memory_guard
