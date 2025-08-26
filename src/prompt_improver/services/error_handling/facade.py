"""Error Handling Facade - Unified Interface for All Error Services.

Provides a unified interface for comprehensive error handling across the application:
- Centralized error handling coordination and routing
- Context-aware error routing to specialized services
- Cross-cutting concerns (correlation tracking, metrics aggregation)
- Protocol-based interfaces for dependency injection
- Performance optimization with caching and intelligent routing

Core Components:
- ErrorHandlingFacade: Main facade coordinating all error services
- Intelligent error categorization and service routing
- Unified metrics aggregation across all error types
- Correlation tracking across service boundaries
- Circuit breaker coordination and state management

Performance Features:
- Error classification caching for repeated patterns
- Batch processing for multiple related errors
- Async processing for non-blocking error handling
- Memory-efficient error context management

Performance Target: <1ms error routing, <5ms end-to-end error processing
Memory Target: <20MB for unified error state management
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from prompt_improver.core.domain.enums import SecurityLevel
from prompt_improver.performance.monitoring.metrics_registry import (
    get_metrics_registry,
)
from prompt_improver.services.error_handling.database_error_service import (
    DatabaseErrorContext,
    DatabaseErrorService,
)
from prompt_improver.services.error_handling.network_error_service import (
    NetworkErrorContext,
    NetworkErrorService,
)
from prompt_improver.services.error_handling.validation_error_service import (
    ValidationErrorContext,
    ValidationErrorService,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class ErrorServiceType(Enum):
    """Types of specialized error services."""

    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorProcessingMode(Enum):
    """Error processing modes for different scenarios."""

    SYNCHRONOUS = "synchronous"      # Immediate processing, blocking
    ASYNCHRONOUS = "asynchronous"    # Background processing, non-blocking
    BATCH = "batch"                  # Batch processing for multiple errors
    FIRE_AND_FORGET = "fire_and_forget"  # Log and continue, minimal processing


@dataclass
class UnifiedErrorContext:
    """Unified error context aggregating all error service contexts."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: str = ""
    operation_name: str = ""
    service_type: ErrorServiceType = ErrorServiceType.UNKNOWN
    processing_mode: ErrorProcessingMode = ErrorProcessingMode.SYNCHRONOUS

    # Timing and performance
    timestamp: datetime = field(default_factory=aware_utc_now)
    processing_start_time: float = field(default_factory=time.time)
    processing_duration_ms: float | None = None

    # Service-specific contexts
    database_context: DatabaseErrorContext | None = None
    network_context: NetworkErrorContext | None = None
    validation_context: ValidationErrorContext | None = None

    # Cross-cutting concerns
    security_level: SecurityLevel | None = None
    user_context: dict[str, Any] = field(default_factory=dict)
    system_context: dict[str, Any] = field(default_factory=dict)

    # Aggregated results
    is_critical: bool = False
    is_retryable: bool = False
    recommended_action: str | None = None
    escalation_required: bool = False


@runtime_checkable
class ErrorHandlingFacadeProtocol(Protocol):
    """Protocol for the error handling facade."""

    async def handle_unified_error(
        self,
        error: Exception,
        operation_name: str,
        service_type: ErrorServiceType | None = None,
        processing_mode: ErrorProcessingMode = ErrorProcessingMode.SYNCHRONOUS,
        **context_kwargs: Any
    ) -> UnifiedErrorContext:
        """Handle error through unified interface."""
        ...

    async def batch_handle_errors(
        self,
        errors: list[tuple[Exception, str, dict[str, Any]]],
        processing_mode: ErrorProcessingMode = ErrorProcessingMode.BATCH
    ) -> list[UnifiedErrorContext]:
        """Handle multiple errors in batch."""
        ...

    def get_unified_error_statistics(self) -> dict[str, Any]:
        """Get unified error statistics."""
        ...


class ErrorHandlingFacade:
    """Unified error handling facade coordinating all error services.

    Provides centralized error handling with:
    - Intelligent error routing to appropriate specialized services
    - Context-aware processing mode selection
    - Cross-service correlation tracking and metrics aggregation
    - Performance optimization through caching and batching
    - Circuit breaker coordination across all error services
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        enable_caching: bool = True,
        enable_batch_processing: bool = True,
        default_security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    ) -> None:
        """Initialize error handling facade.

        Args:
            correlation_id: Optional correlation ID for request tracking
            enable_caching: Enable error classification caching
            enable_batch_processing: Enable batch error processing
            default_security_level: Default security level for error handling
        """
        super().__init__()
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.enable_caching = enable_caching
        self.enable_batch_processing = enable_batch_processing
        self.default_security_level = default_security_level

        # Initialize specialized error services
        self._database_service = DatabaseErrorService(correlation_id=self.correlation_id)
        self._network_service = NetworkErrorService(correlation_id=self.correlation_id)
        self._validation_service = ValidationErrorService(correlation_id=self.correlation_id)

        # Service registry - using Any type for flexibility since each service has different interfaces
        self._error_services: dict[ErrorServiceType, Any] = {
            ErrorServiceType.DATABASE: self._database_service,
            ErrorServiceType.NETWORK: self._network_service,
            ErrorServiceType.VALIDATION: self._validation_service,
        }

        # Performance optimization
        self._metrics_registry = get_metrics_registry()
        self._error_classification_cache: dict[str, ErrorServiceType] = {}
        self._batch_processing_queue: list[tuple[Exception, str, dict[str, Any]]] = []
        self._batch_processing_task: asyncio.Task[None] | None = None

        # Statistics tracking
        self._error_counts: dict[ErrorServiceType, int] = dict.fromkeys(ErrorServiceType, 0)
        self._processing_times: dict[ErrorServiceType, list[float]] = {
            service_type: [] for service_type in ErrorServiceType
        }

        logger.info(f"ErrorHandlingFacade initialized with correlation_id: {self.correlation_id}")

    async def handle_unified_error(
        self,
        error: Exception,
        operation_name: str,
        service_type: ErrorServiceType | None = None,
        processing_mode: ErrorProcessingMode = ErrorProcessingMode.SYNCHRONOUS,
        **context_kwargs: Any
    ) -> UnifiedErrorContext:
        """Handle error through unified interface with intelligent routing.

        Args:
            error: Exception that occurred
            operation_name: Name of the operation that failed
            service_type: Specific service type (auto-detected if None)
            processing_mode: How to process the error
            **context_kwargs: Additional context for error handling

        Returns:
            UnifiedErrorContext with processing results
        """
        start_time = time.time()

        # Create unified error context
        unified_context = UnifiedErrorContext(
            correlation_id=self.correlation_id,
            operation_name=operation_name,
            processing_mode=processing_mode,
            security_level=context_kwargs.get('security_level', self.default_security_level),
            user_context=context_kwargs.get('user_context', {}),
            system_context=context_kwargs.get('system_context', {}),
        )

        # Determine service type if not specified
        if service_type is None:
            service_type = self._classify_error_service_type(error, operation_name)

        unified_context.service_type = service_type

        # Handle based on processing mode
        if processing_mode == ErrorProcessingMode.FIRE_AND_FORGET:
            # Minimal processing, just log and continue
            await self._handle_fire_and_forget(error, unified_context)
        elif processing_mode == ErrorProcessingMode.BATCH:
            # Add to batch processing queue
            await self._add_to_batch_queue(error, operation_name, context_kwargs)
            # Return immediately with placeholder context
            unified_context.recommended_action = "queued_for_batch_processing"
        elif processing_mode == ErrorProcessingMode.ASYNCHRONOUS:
            # Process asynchronously
            asyncio.create_task(self._process_error_async(error, unified_context, **context_kwargs))
            unified_context.recommended_action = "processing_asynchronously"
        else:
            # Synchronous processing
            await self._process_error_sync(error, unified_context, **context_kwargs)

        # Calculate processing time
        unified_context.processing_duration_ms = (time.time() - start_time) * 1000

        # Record unified metrics
        await self._record_unified_metrics(unified_context)

        # Update statistics
        self._update_statistics(unified_context)

        return unified_context

    async def batch_handle_errors(
        self,
        errors: list[tuple[Exception, str, dict[str, Any]]]
    ) -> list[UnifiedErrorContext]:
        """Handle multiple errors in batch for improved performance.

        Args:
            errors: List of (exception, operation_name, context_kwargs) tuples

        Returns:
            List of UnifiedErrorContext results
        """
        if not self.enable_batch_processing:
            # Fall back to individual processing
            results: list[UnifiedErrorContext] = []
            for error, operation_name, context_kwargs in errors:
                result = await self.handle_unified_error(
                    error, operation_name, processing_mode=ErrorProcessingMode.SYNCHRONOUS, **context_kwargs
                )
                results.append(result)
            return results

        start_time = time.time()
        batch_id = str(uuid.uuid4())[:8]

        logger.info(f"Processing batch of {len(errors)} errors with batch_id: {batch_id}")

        # Group errors by service type for efficient processing
        grouped_errors: dict[ErrorServiceType, list[tuple[Exception, str, dict[str, Any]]]] = {}

        for error, operation_name, context_kwargs in errors:
            service_type = self._classify_error_service_type(error, operation_name)
            if service_type not in grouped_errors:
                grouped_errors[service_type] = []
            grouped_errors[service_type].append((error, operation_name, context_kwargs))

        # Process groups concurrently
        batch_tasks: list[asyncio.Task[list[UnifiedErrorContext]]] = []
        for service_type, group_errors in grouped_errors.items():
            task = asyncio.create_task(
                self._process_error_group(service_type, group_errors, batch_id)
            )
            batch_tasks.append(task)

        # Wait for all groups to complete
        group_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results
        unified_results: list[UnifiedErrorContext] = []
        for group_result in group_results:
            if isinstance(group_result, Exception):
                # Handle batch processing error
                logger.error(f"Batch processing error for batch_id {batch_id}: {group_result}")
                continue
            # group_result should be a list of UnifiedErrorContext
            if isinstance(group_result, list):
                unified_results.extend(group_result)
            else:
                logger.warning(f"Unexpected group result type: {type(group_result)}")

        # Record batch metrics
        batch_duration = (time.time() - start_time) * 1000
        histogram = self._metrics_registry.get_or_create_histogram(
            "error_batch_processing_duration_ms",
            "Error batch processing duration in milliseconds",
            ["batch_id", "batch_size", "groups_processed"]
        )
        histogram.observe(
            batch_duration,
            {
                "batch_id": batch_id,
                "batch_size": str(len(errors)),
                "groups_processed": str(len(grouped_errors))
            }
        )

        logger.info(f"Completed batch processing for batch_id {batch_id} in {batch_duration:.2f}ms")

        return unified_results

    async def handle_database_error(
        self,
        error: Exception,
        operation_name: str,
        **context_kwargs: Any
    ) -> DatabaseErrorContext:
        """Handle database error through specialized service.

        Args:
            error: Database exception
            operation_name: Database operation name
            **context_kwargs: Additional context

        Returns:
            DatabaseErrorContext with processing results
        """
        return await self._database_service.handle_database_error(
            error, operation_name, **context_kwargs
        )

    async def handle_network_error(
        self,
        error: Exception,
        operation_name: str,
        **context_kwargs: Any
    ) -> NetworkErrorContext:
        """Handle network error through specialized service.

        Args:
            error: Network exception
            operation_name: Network operation name
            **context_kwargs: Additional context

        Returns:
            NetworkErrorContext with processing results
        """
        return await self._network_service.handle_network_error(
            error, operation_name, **context_kwargs
        )

    async def handle_validation_error(
        self,
        error: Exception,
        operation_name: str,
        **context_kwargs: Any
    ) -> ValidationErrorContext:
        """Handle validation error through specialized service.

        Args:
            error: Validation exception
            operation_name: Validation operation name
            **context_kwargs: Additional context

        Returns:
            ValidationErrorContext with processing results
        """
        return await self._validation_service.handle_validation_error(
            error, operation_name, **context_kwargs
        )

    def _classify_error_service_type(self, error: Exception, operation_name: str) -> ErrorServiceType:
        """Classify error to determine appropriate service type.

        Args:
            error: Exception to classify
            operation_name: Operation name for context

        Returns:
            Appropriate ErrorServiceType
        """
        # Check cache first if enabled
        error_key = f"{type(error).__name__}:{operation_name}"
        if self.enable_caching and error_key in self._error_classification_cache:
            return self._error_classification_cache[error_key]

        error_type = type(error)
        error_msg = str(error).lower()
        operation_lower = operation_name.lower()

        # Database error patterns
        if any(pattern in operation_lower for pattern in ["db", "database", "sql", "query", "transaction"]) or any(pattern in error_msg for pattern in ["connection", "database", "sql", "transaction", "rollback"]):
            service_type = ErrorServiceType.DATABASE

        # Network error patterns
        elif any(pattern in operation_lower for pattern in ["http", "api", "request", "network", "client"]) or any(pattern in error_msg for pattern in ["connection", "timeout", "http", "ssl", "dns"]):
            service_type = ErrorServiceType.NETWORK

        # Validation error patterns
        elif any(pattern in operation_lower for pattern in ["validate", "validation", "input", "schema"]) or any(pattern in error_msg for pattern in ["validation", "invalid", "format", "required", "constraint"]):
            service_type = ErrorServiceType.VALIDATION

        # Exception type-based classification
        elif "sql" in error_type.__name__.lower() or "database" in error_type.__name__.lower():
            service_type = ErrorServiceType.DATABASE
        elif "http" in error_type.__name__.lower() or "connection" in error_type.__name__.lower():
            service_type = ErrorServiceType.NETWORK
        elif "validation" in error_type.__name__.lower() or "value" in error_type.__name__.lower():
            service_type = ErrorServiceType.VALIDATION

        else:
            service_type = ErrorServiceType.SYSTEM  # Default for unknown errors

        # Cache the result if caching is enabled
        if self.enable_caching:
            self._error_classification_cache[error_key] = service_type

        return service_type

    async def _process_error_sync(
        self,
        error: Exception,
        unified_context: UnifiedErrorContext,
        **context_kwargs: Any
    ) -> None:
        """Process error synchronously through appropriate service.

        Args:
            error: Exception to process
            unified_context: Unified error context to update
            **context_kwargs: Additional context
        """
        service_type = unified_context.service_type

        try:
            if service_type == ErrorServiceType.DATABASE:
                db_context = await self._database_service.handle_database_error(
                    error, unified_context.operation_name, **context_kwargs
                )
                unified_context.database_context = db_context
                unified_context.is_retryable = db_context.is_retryable

            elif service_type == ErrorServiceType.NETWORK:
                net_context = await self._network_service.handle_network_error(
                    error, unified_context.operation_name, **context_kwargs
                )
                unified_context.network_context = net_context
                unified_context.is_retryable = net_context.is_retryable

            elif service_type == ErrorServiceType.VALIDATION:
                val_context = await self._validation_service.handle_validation_error(
                    error, unified_context.operation_name, **context_kwargs
                )
                unified_context.validation_context = val_context
                unified_context.is_retryable = False  # Validation errors generally not retryable

                # Check for critical security issues
                if val_context.threat_detected:
                    unified_context.is_critical = True
                    unified_context.escalation_required = True
                    unified_context.recommended_action = "immediate_security_review"

            else:
                # Handle unknown/system errors with basic processing
                unified_context.recommended_action = "generic_error_handling"
                logger.warning(f"Unhandled error type {service_type} for error: {error}")

        except Exception as processing_error:
            logger.exception(
                f"Error occurred while processing {service_type.value} error: {processing_error}",
                extra={"correlation_id": self.correlation_id, "original_error": str(error)}
            )
            unified_context.recommended_action = "error_processing_failed"

    async def _process_error_async(
        self,
        error: Exception,
        unified_context: UnifiedErrorContext,
        **context_kwargs: Any
    ) -> None:
        """Process error asynchronously.

        Args:
            error: Exception to process
            unified_context: Unified error context
            **context_kwargs: Additional context
        """
        try:
            await self._process_error_sync(error, unified_context, **context_kwargs)
            logger.info(f"Async error processing completed for {unified_context.error_id}")
        except Exception as async_error:
            logger.exception(f"Async error processing failed for {unified_context.error_id}: {async_error}")

    async def _handle_fire_and_forget(
        self,
        error: Exception,
        unified_context: UnifiedErrorContext
    ) -> None:
        """Handle error with minimal processing for fire-and-forget mode.

        Args:
            error: Exception to handle
            unified_context: Unified error context
        """
        # Just log the error and continue
        logger.info(
            f"Fire-and-forget error handling for {unified_context.operation_name}: {str(error)[:200]}",
            extra={
                "correlation_id": self.correlation_id,
                "error_id": unified_context.error_id,
                "service_type": unified_context.service_type.value
            }
        )

        unified_context.recommended_action = "logged_and_ignored"

    async def _add_to_batch_queue(
        self,
        error: Exception,
        operation_name: str,
        context_kwargs: dict[str, Any]
    ) -> None:
        """Add error to batch processing queue.

        Args:
            error: Exception to queue
            operation_name: Operation name
            context_kwargs: Context arguments
        """
        self._batch_processing_queue.append((error, operation_name, context_kwargs))

        # Start batch processing task if not already running
        if self._batch_processing_task is None or self._batch_processing_task.done():
            self._batch_processing_task = asyncio.create_task(self._process_batch_queue())

    async def _process_batch_queue(self) -> None:
        """Process queued errors in batch."""
        while self._batch_processing_queue:
            # Wait a bit to collect more errors
            await asyncio.sleep(0.1)

            # Process current queue
            current_batch = self._batch_processing_queue.copy()
            self._batch_processing_queue.clear()

            if current_batch:
                try:
                    await self.batch_handle_errors(current_batch)
                except Exception as batch_error:
                    logger.exception(f"Batch processing error: {batch_error}")

    async def _process_error_group(
        self,
        service_type: ErrorServiceType,
        errors: list[tuple[Exception, str, dict[str, Any]]],
        batch_id: str
    ) -> list[UnifiedErrorContext]:
        """Process a group of errors of the same service type.

        Args:
            service_type: Type of service to handle errors
            errors: List of errors to process
            batch_id: Batch identifier

        Returns:
            List of UnifiedErrorContext results
        """
        results: list[UnifiedErrorContext] = []

        for error, operation_name, context_kwargs in errors:
            unified_context = await self.handle_unified_error(
                error,
                operation_name,
                service_type=service_type,
                processing_mode=ErrorProcessingMode.SYNCHRONOUS,
                batch_id=batch_id,
                **context_kwargs
            )
            results.append(unified_context)

        return results

    async def _record_unified_metrics(self, unified_context: UnifiedErrorContext) -> None:
        """Record unified metrics for error handling.

        Args:
            unified_context: Unified error context with metrics
        """
        try:
            counter = self._metrics_registry.get_or_create_counter(
                "unified_errors_total",
                "Total unified errors processed",
                ["service_type", "processing_mode", "is_critical", "is_retryable", "escalation_required"]
            )
            counter.inc(1, {
                "service_type": unified_context.service_type.value,
                "processing_mode": unified_context.processing_mode.value,
                "is_critical": str(unified_context.is_critical).lower(),
                "is_retryable": str(unified_context.is_retryable).lower(),
                "escalation_required": str(unified_context.escalation_required).lower()
            })

            if unified_context.processing_duration_ms:
                histogram = self._metrics_registry.get_or_create_histogram(
                    "unified_error_processing_duration_ms",
                    "Unified error processing duration in milliseconds",
                    ["service_type", "processing_mode"]
                )
                histogram.observe(
                    unified_context.processing_duration_ms,
                    {
                        "service_type": unified_context.service_type.value,
                        "processing_mode": unified_context.processing_mode.value
                    }
                )

        except Exception as e:
            logger.exception(f"Failed to record unified error metrics: {e}")

    def _update_statistics(self, unified_context: UnifiedErrorContext) -> None:
        """Update internal statistics tracking.

        Args:
            unified_context: Unified error context
        """
        service_type = unified_context.service_type

        # Update error counts
        self._error_counts[service_type] += 1

        # Update processing times
        if unified_context.processing_duration_ms:
            self._processing_times[service_type].append(unified_context.processing_duration_ms)

            # Keep only recent processing times (last 1000)
            if len(self._processing_times[service_type]) > 1000:
                self._processing_times[service_type] = self._processing_times[service_type][-1000:]

    def get_unified_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics across all services.

        Returns:
            Dictionary with unified error statistics
        """
        # Calculate average processing times
        avg_processing_times = {}
        for service_type, times in self._processing_times.items():
            if times:
                avg_processing_times[service_type.value] = sum(times) / len(times)
            else:
                avg_processing_times[service_type.value] = 0.0

        # Get individual service statistics
        service_stats = {}
        for service_type, service in self._error_services.items():
            try:
                service_stats[service_type.value] = service.get_error_statistics()
            except Exception as e:
                logger.exception(f"Failed to get statistics from {service_type.value} service: {e}")
                service_stats[service_type.value] = {"error": str(e)}

        return {
            "correlation_id": self.correlation_id,
            "facade_stats": {
                "total_errors_processed": sum(self._error_counts.values()),
                "error_counts_by_service": {st.value: count for st, count in self._error_counts.items()},
                "average_processing_times_ms": avg_processing_times,
                "cache_enabled": self.enable_caching,
                "batch_processing_enabled": self.enable_batch_processing,
                "classification_cache_size": len(self._error_classification_cache),
                "batch_queue_size": len(self._batch_processing_queue),
            },
            "service_statistics": service_stats,
            "service_health": "operational",
        }


# Global facade instance for decorator functions
_global_facade: ErrorHandlingFacade | None = None


def _get_global_facade() -> ErrorHandlingFacade:
    """Get or create the global error handling facade."""
    global _global_facade
    if _global_facade is None:
        _global_facade = ErrorHandlingFacade()
    return _global_facade


# Decorator functions for backward compatibility
def handle_common_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for common error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.SYSTEM
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.SYSTEM
                ))
            except RuntimeError:
                # No event loop, just log the error
                import logging
                logging.exception(f"Error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_database_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for database error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.DATABASE
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.DATABASE
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Database error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_network_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for network error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.NETWORK
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.NETWORK
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Network error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_validation_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for validation error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.VALIDATION
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.VALIDATION
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Validation error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_filesystem_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for filesystem error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.SYSTEM
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.SYSTEM
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Filesystem error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_repository_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for repository error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.DATABASE
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.DATABASE
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Repository error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_service_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for service error handling."""
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            await facade.handle_unified_error(
                e, func.__name__, ErrorServiceType.SYSTEM
            )
            raise

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            facade = _get_global_facade()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(facade.handle_unified_error(
                    e, func.__name__, ErrorServiceType.SYSTEM
                ))
            except RuntimeError:
                import logging
                logging.exception(f"Service error in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
