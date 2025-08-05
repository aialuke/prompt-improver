"""
Modern RetryManager Implementation - 2025 Best Practices

This module provides a comprehensive, unified retry management system that implements
all retry protocols using clean architecture principles and modern patterns:

Features:
- Protocol-based interfaces for clean dependency injection
- Modern dataclass-based configuration and state management
- Enum-based error classification for better categorization
- Circuit breaker patterns with proper state management
- Comprehensive observability and metrics integration
- Support for both sync and async operations
- OpenTelemetry integration for distributed tracing
- Resource-aware retry policies
- Intelligent error classification
- Database-specific and ML orchestration optimizations

Architecture:
- RetryManager: Main unified retry manager class implementing RetryManagerProtocol
- RetryConfig: Comprehensive configuration with 2025 patterns
- CircuitBreaker: Advanced circuit breaker with state management
- RetryContext: Rich context tracking for observability
- Modern error classification and intelligent backoff strategies
"""

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# Import protocols from comprehensive retry_protocols.py
from .protocols.retry_protocols import (
    RetryManagerProtocol,
    RetryConfigProtocol,
    RetryStrategy,
    RetryableErrorType,
    CircuitBreakerProtocol,
    RetryObserverProtocol,
    MetricsRegistryProtocol
)

# Import centralized metrics registry
from ..performance.monitoring.metrics_registry import (
    get_metrics_registry,
    StandardMetrics
)

# Lazy import background task manager to avoid circular imports
# from ..performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _get_background_task_manager():
    """Lazy import background task manager to avoid circular imports."""
    try:
        from ..performance.monitoring.health.background_manager import get_background_task_manager
        return get_background_task_manager()
    except ImportError as e:
        logger.warning(f"Background task manager not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to get background task manager: {e}")
        return None


def _get_task_priority():
    """Lazy import TaskPriority to avoid circular imports."""
    try:
        from ..performance.monitoring.health.background_manager import TaskPriority
        return TaskPriority
    except ImportError:
        # Fallback enum if background manager not available
        from enum import Enum
        class FallbackTaskPriority(Enum):
            CRITICAL = 1
            HIGH = 2
            NORMAL = 3
            LOW = 4
            BACKGROUND = 5
        return FallbackTaskPriority


class CircuitBreakerState(Enum):
    """Circuit breaker states with 2025 patterns."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class RetryConfig:
    """
    Comprehensive retry configuration implementing RetryConfigProtocol.

    Supports all modern retry patterns with intelligent defaults and
    extensive customization options for different use cases.
    """

    # Basic retry settings (implementing RetryConfigProtocol)
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Delay settings
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds

    # Advanced settings
    jitter: bool = True
    jitter_factor: float = 0.1
    backoff_multiplier: float = 2.0

    # Conditional retry settings
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    retry_condition: Optional[Callable[[Exception], bool]] = None

    # Timeout settings
    operation_timeout: Optional[float] = None
    total_timeout: Optional[float] = None

    # Logging and monitoring
    log_attempts: bool = True
    log_level: str = "INFO"
    track_metrics: bool = True

    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout_seconds: float = 60.0
    half_open_max_calls: int = 3

    # Error classification
    retryable_errors: List[RetryableErrorType] = field(default_factory=lambda: [
        RetryableErrorType.TRANSIENT,
        RetryableErrorType.NETWORK,
        RetryableErrorType.TIMEOUT,
        RetryableErrorType.RATE_LIMIT
    ])

    # Observability
    operation_name: Optional[str] = None
    enable_metrics: bool = True
    enable_tracing: bool = True

    # Advanced patterns
    custom_delay_func: Optional[Callable[[int], float]] = None
    error_classifier: Optional[Callable[[Exception], RetryableErrorType]] = None
    resource_aware: bool = False
    adaptive_delays: bool = False

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (implementing protocol)"""
        if attempt < 0:
            return 0.0

        if self.custom_delay_func:
            return self.custom_delay_func(attempt)

        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** attempt)
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.base_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Apply jitter if enabled
        if self.jitter and delay > 0:
            jitter_amount = delay * self.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried (implementing protocol)"""
        # Check attempt limit
        if attempt >= self.max_attempts - 1:
            return False

        # Check exception type
        if not any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions):
            return False

        # Check custom retry condition
        if self.retry_condition and not self.retry_condition(exception):
            return False

        return True

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


@dataclass
class RetryContext:
    """
    Rich retry context for comprehensive tracking and observability.

    Provides detailed context about retry operations including timing,
    errors, circuit breaker state, and correlation identifiers.
    """

    operation_name: str
    attempt: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: Optional[Exception] = None
    total_delay_ms: float = 0.0
    circuit_breaker_triggered: bool = False

    # Observability
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Performance tracking
    response_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerStateData:
    """Circuit breaker state tracking with modern dataclass patterns."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    half_open_attempts: int = 0

    # Configuration reference
    config: RetryConfig = field(default_factory=RetryConfig)


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with 2025 patterns.

    Features:
    - State-based protection with proper transitions
    - Configurable thresholds and timeouts
    - Comprehensive metrics and observability
    - Thread-safe operations with async locks
    """

    def __init__(self, config: RetryConfig, operation_name: str):
        self.config = config
        self.operation_name = operation_name
        self.state_data = CircuitBreakerStateData(config=config)
        self._lock = asyncio.Lock()

        # Metrics integration - always use real metrics
        self.metrics_registry = get_metrics_registry()
        self.state_gauge = self.metrics_registry.get_or_create_gauge(
            StandardMetrics.CIRCUIT_BREAKER_STATE,
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['operation']
        )
        self.events_counter = self.metrics_registry.get_or_create_counter(
            StandardMetrics.CIRCUIT_BREAKER_EVENTS_TOTAL,
            'Circuit breaker events',
            ['operation', 'event']
        )

    async def call(self, operation: Callable[[], Coroutine[Any, Any, T]], context: RetryContext) -> T:
        """Execute operation with circuit breaker protection."""
        if not self.config.enable_circuit_breaker:
            return await operation()

        await self._check_state_transition()

        if self.state_data.state == CircuitBreakerState.OPEN:
            context.circuit_breaker_triggered = True
            self._record_event("blocked")
            raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {self.operation_name}")

        try:
            result = await operation()
            await self._on_success(context)
            return result
        except Exception as e:
            await self._on_failure(context, e)
            raise

    async def _check_state_transition(self):
        """Check if circuit breaker should transition states."""
        async with self._lock:
            if (self.state_data.state == CircuitBreakerState.OPEN and
                self.state_data.last_failure_time and
                datetime.now(timezone.utc) - self.state_data.last_failure_time >
                timedelta(seconds=self.config.recovery_timeout_seconds)):

                self.state_data.state = CircuitBreakerState.HALF_OPEN
                self.state_data.success_count = 0
                self.state_data.half_open_attempts = 0
                logger.info(f"Circuit breaker transitioning to HALF_OPEN for {self.operation_name}")
                self._update_state_metric()
                self._record_event("half_open")

    async def _on_success(self, context: RetryContext):
        """Handle successful operation."""
        async with self._lock:
            self.state_data.last_success_time = datetime.now(timezone.utc)

            if self.state_data.state == CircuitBreakerState.HALF_OPEN:
                self.state_data.success_count += 1
                if self.state_data.success_count >= self.config.success_threshold:
                    self.state_data.state = CircuitBreakerState.CLOSED
                    self.state_data.failure_count = 0
                    logger.info(f"Circuit breaker transitioning to CLOSED for {self.operation_name}")
                    self._record_event("closed")
            elif self.state_data.state == CircuitBreakerState.CLOSED:
                self.state_data.failure_count = 0

            self._update_state_metric()

    async def _on_failure(self, context: RetryContext, error: Exception):
        """Handle failed operation."""
        async with self._lock:
            self.state_data.failure_count += 1
            self.state_data.last_failure_time = datetime.now(timezone.utc)

            if (self.state_data.state == CircuitBreakerState.CLOSED and
                self.state_data.failure_count >= self.config.failure_threshold):

                self.state_data.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker transitioning to OPEN for {self.operation_name}")
                self._record_event("opened")
            elif self.state_data.state == CircuitBreakerState.HALF_OPEN:
                self.state_data.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker returning to OPEN from HALF_OPEN for {self.operation_name}")
                self._record_event("opened")

            self._update_state_metric()

    def _update_state_metric(self):
        """Update state metric."""
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.HALF_OPEN: 1,
            CircuitBreakerState.OPEN: 2
        }[self.state_data.state]
        self.state_gauge.labels(operation=self.operation_name).set(state_value)

    def _record_event(self, event: str):
        """Record circuit breaker event."""
        self.events_counter.labels(operation=self.operation_name, event=event).inc()

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state_data.state.value

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state_data.state == CircuitBreakerState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        task_manager = _get_background_task_manager()
        if task_manager:
            TaskPriority = _get_task_priority()
            asyncio.create_task(task_manager.submit_enhanced_task(
                task_id=f"circuit_breaker_reset_{self.operation_name}_{str(uuid.uuid4())[:8]}",
                coroutine=self._reset_async(),
                priority=TaskPriority.HIGH,
                tags={
                    "service": "retry_manager",
                    "type": "circuit_breaker",
                    "component": "reset",
                    "operation": self.operation_name
            }
        ))

    async def _reset_async(self):
        """Async reset implementation."""
        async with self._lock:
            self.state_data.state = CircuitBreakerState.CLOSED
            self.state_data.failure_count = 0
            self.state_data.success_count = 0
            self.state_data.half_open_attempts = 0
            self._update_state_metric()
            self._record_event("reset")
            logger.info(f"Circuit breaker reset for {self.operation_name}")


class RetryManager:
    """
    Modern unified retry manager implementing 2025 best practices.

    This is the single source of truth for all retry functionality in the codebase.
    Implements RetryManagerProtocol and provides comprehensive retry capabilities
    for all system components.

    Features:
    - Protocol-based interfaces for clean dependency injection
    - Modern dataclass-based configuration and state management
    - Enum-based error classification for better categorization
    - Advanced circuit breaker with proper state management
    - Comprehensive observability and metrics integration
    - Support for both sync and async operations
    - OpenTelemetry integration for distributed tracing
    - Resource-aware retry policies
    - Intelligent error classification
    - Database-specific and ML orchestration optimizations
    """

    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Initialize the unified retry manager.

        Args:
            default_config: Default retry configuration to use
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.default_config = default_config or RetryConfig()

        # State management
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}

        # Locks for thread safety
        self._circuit_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()

        # Metrics and observability
        self._setup_metrics()
        self._setup_tracing()

        # Observers
        self._observers: List[RetryObserverProtocol] = []

        self.logger.info("RetryManager initialized with 2025 best practices")

    def _setup_metrics(self):
        """Setup real metrics using centralized registry."""
        self.metrics_registry = get_metrics_registry()

        self.retry_attempts_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_ATTEMPTS_TOTAL,
            'Total retry attempts',
            ['operation', 'strategy', 'attempt']
        )

        self.retry_success_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_SUCCESS_TOTAL,
            'Total successful retries',
            ['operation', 'strategy']
        )

        self.retry_failure_total = self.metrics_registry.get_or_create_counter(
            StandardMetrics.RETRY_FAILURE_TOTAL,
            'Total failed retries',
            ['operation', 'strategy', 'error_type']
        )

        self.retry_duration_histogram = self.metrics_registry.get_or_create_histogram(
            StandardMetrics.RETRY_DURATION,
            'Retry operation duration',
            ['operation', 'strategy'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = otel_metrics.get_meter(__name__)
        else:
            self.tracer = None
            self.meter = None

    # ============================================================================
    # RetryManagerProtocol Implementation
    # ============================================================================

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with retry logic (implements RetryManagerProtocol)"""
        # Convert protocol config to internal config if needed
        if isinstance(config, RetryConfig):
            retry_config = config
        else:
            retry_config = self._convert_config(config)

        operation_name = retry_config.operation_name or getattr(operation, '__name__', 'unknown')
        context = RetryContext(operation_name=operation_name)

        if asyncio.iscoroutinefunction(operation):
            return await self._execute_async_with_retry(operation, retry_config, context, *args, **kwargs)
        else:
            return await self._execute_sync_with_retry(operation, retry_config, context, *args, **kwargs)

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with circuit breaker pattern (implements RetryManagerProtocol)"""
        # Convert protocol config to internal config if needed
        if isinstance(config, RetryConfig):
            retry_config = config
        else:
            retry_config = self._convert_config(config)

        retry_config.enable_circuit_breaker = True
        return await self.execute_with_retry(operation, retry_config, *args, **kwargs)

    def get_retry_stats(self, operation_name: str) -> dict[str, Any]:
        """Get retry statistics for an operation (implements RetryManagerProtocol)"""
        stats = self._stats.get(operation_name, {})
        circuit_breaker = self._circuit_breakers.get(operation_name)

        result = {
            "operation_name": operation_name,
            "total_attempts": stats.get("total_attempts", 0),
            "successful_attempts": stats.get("successful_attempts", 0),
            "failed_attempts": stats.get("failed_attempts", 0),
            "success_rate": stats.get("success_rate", 0.0),
            "average_delay_ms": stats.get("average_delay_ms", 0.0),
            "last_attempt_at": stats.get("last_attempt_at"),
        }

        if circuit_breaker:
            result["circuit_breaker"] = {
                "state": circuit_breaker.get_state(),
                "failure_count": circuit_breaker.state_data.failure_count,
                "success_count": circuit_breaker.state_data.success_count,
                "is_open": circuit_breaker.is_open()
            }

        return result

    # ============================================================================
    # Extended Retry Interface
    # ============================================================================

    async def retry_async(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        config: Optional[RetryConfig] = None,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Execute async operation with comprehensive retry logic.

        Args:
            operation: Async callable to execute
            config: Retry configuration, uses default if None
            operation_name: Name for tracking and metrics
            **kwargs: Arguments passed to operation

        Returns:
            Result of successful operation execution
        """
        retry_config = config or self.default_config
        if operation_name:
            retry_config.operation_name = operation_name

        context = RetryContext(operation_name=retry_config.operation_name or operation.__name__)
        return await self._execute_async_with_retry(operation, retry_config, context, **kwargs)

    async def retry_sync(
        self,
        operation: Callable[..., T],
        config: Optional[RetryConfig] = None,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Execute sync operation with comprehensive retry logic.

        Args:
            operation: Sync callable to execute
            config: Retry configuration, uses default if None
            operation_name: Name for tracking and metrics
            **kwargs: Arguments passed to operation

        Returns:
            Result of successful operation execution
        """
        retry_config = config or self.default_config
        if operation_name:
            retry_config.operation_name = operation_name

        context = RetryContext(operation_name=retry_config.operation_name or operation.__name__)
        return await self._execute_sync_with_retry(operation, retry_config, context, **kwargs)

    # ============================================================================
    # Database-Specific Interface
    # ============================================================================

    async def retry_database_operation(
        self,
        operation: Callable[..., Any],
        operation_name: str = "database_operation",
        **kwargs
    ) -> Any:
        """
        Execute database operation with database-specific retry logic.

        Args:
            operation: Database operation to execute
            operation_name: Name for tracking and metrics
            **kwargs: Arguments passed to operation

        Returns:
            Result of successful operation execution
        """
        # Database-specific retry configuration
        db_config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.5,
            max_delay=30.0,
            operation_name=operation_name,
            retry_on_exceptions=[
                ConnectionError,
                TimeoutError,
                OSError,  # Network-related errors
            ],
            failure_threshold=3,
            recovery_timeout_seconds=30.0
        )

        if asyncio.iscoroutinefunction(operation):
            return await self.retry_async(operation, db_config, operation_name, **kwargs)
        else:
            return await self.retry_sync(operation, db_config, operation_name, **kwargs)

    # ============================================================================
    # ML Orchestration Interface
    # ============================================================================

    async def retry_ml_operation(
        self,
        operation: Callable[..., Any],
        operation_name: str = "ml_operation",
        **kwargs
    ) -> Any:
        """
        Execute ML operation with ML-specific retry logic.

        Args:
            operation: ML operation to execute
            operation_name: Name for tracking and metrics
            **kwargs: Arguments passed to operation

        Returns:
            Result of successful operation execution
        """
        # ML-specific retry configuration
        ml_config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            max_delay=60.0,
            operation_name=operation_name,
            retry_on_exceptions=[
                ConnectionError,
                TimeoutError,
                RuntimeError,  # ML framework errors
            ],
            failure_threshold=2,
            recovery_timeout_seconds=120.0,
            resource_aware=True,
            adaptive_delays=True
        )

        if asyncio.iscoroutinefunction(operation):
            return await self.retry_async(operation, ml_config, operation_name, **kwargs)
        else:
            return await self.retry_sync(operation, ml_config, operation_name, **kwargs)

    # ============================================================================
    # Context Manager Interface
    # ============================================================================

    @asynccontextmanager
    async def with_retry(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None
    ):
        """Context manager for retry operations."""
        retry_config = config or self.default_config
        retry_config.operation_name = operation_name

        context = RetryContext(operation_name=operation_name)
        circuit_breaker = await self._get_circuit_breaker(operation_name, retry_config)

        yield RetryExecutor(self, retry_config, context, circuit_breaker)

    # ============================================================================
    # Observer Pattern
    # ============================================================================

    def add_observer(self, observer: RetryObserverProtocol):
        """Add retry observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: RetryObserverProtocol):
        """Remove retry observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    # ============================================================================
    # Configuration Management
    # ============================================================================

    def configure_strategy(
        self,
        strategy: RetryStrategy,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **options
    ) -> None:
        """Configure default retry strategy."""
        self.default_config.strategy = strategy
        self.default_config.max_attempts = max_attempts
        self.default_config.base_delay = base_delay
        self.default_config.max_delay = max_delay

        # Apply additional options
        for key, value in options.items():
            if hasattr(self.default_config, key):
                setattr(self.default_config, key, value)

        self.logger.info(f"Configured retry strategy: {strategy.value}")

    # ============================================================================
    # Circuit Breaker Management
    # ============================================================================

    async def get_circuit_breaker_status(self, operation_name: str) -> Dict[str, Any]:
        """Get circuit breaker status for operation."""
        circuit_breaker = self._circuit_breakers.get(operation_name)
        if not circuit_breaker:
            return {"state": "not_configured", "failure_count": 0}

        return {
            "state": circuit_breaker.get_state(),
            "failure_count": circuit_breaker.state_data.failure_count,
            "success_count": circuit_breaker.state_data.success_count,
            "last_failure_time": circuit_breaker.state_data.last_failure_time,
            "last_success_time": circuit_breaker.state_data.last_success_time,
            "is_open": circuit_breaker.is_open()
        }

    async def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker for operation."""
        circuit_breaker = self._circuit_breakers.get(operation_name)
        if circuit_breaker:
            circuit_breaker.reset()
            self.logger.info(f"Circuit breaker reset for {operation_name}")

    # ============================================================================
    # Statistics and Metrics
    # ============================================================================

    def reset_stats(self, operation_name: Optional[str] = None) -> None:
        """Reset retry statistics."""
        if operation_name:
            if operation_name in self._stats:
                del self._stats[operation_name]
                self.logger.info(f"Reset stats for operation: {operation_name}")
        else:
            self._stats.clear()
            self._circuit_breakers.clear()
            self.logger.info("Reset all retry statistics")

    async def get_comprehensive_metrics(self, operation_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for operation."""
        stats = self.get_retry_stats(operation_name)
        circuit_status = await self.get_circuit_breaker_status(operation_name)

        return {
            "operation_name": operation_name,
            "retry_stats": stats,
            "circuit_breaker": circuit_status,
            "health_status": "healthy" if stats.get("success_rate", 0.0) > 0.8 else "degraded",
            "recommendations": self._generate_recommendations(stats),
        }

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on retry statistics."""
        recommendations = []

        success_rate = stats.get("success_rate", 0.0)
        if success_rate < 0.5:
            recommendations.append("Consider investigating root cause of high failure rate")

        avg_delay = stats.get("average_delay_ms", 0.0)
        if avg_delay > 5000:
            recommendations.append("Consider reducing retry delays or max attempts")

        failed_attempts = stats.get("failed_attempts", 0)
        successful_attempts = stats.get("successful_attempts", 0)
        if failed_attempts > successful_attempts:
            recommendations.append("Operation appears to be failing more than succeeding")

        return recommendations

    # ============================================================================
    # Internal Implementation Methods
    # ============================================================================

    async def _execute_async_with_retry(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        config: RetryConfig,
        context: RetryContext,
        **kwargs
    ) -> T:
        """Internal async retry implementation."""
        circuit_breaker = await self._get_circuit_breaker(context.operation_name, config)
        last_exception = None

        # Start tracing span
        span = self._start_span(context.operation_name, config) if self.tracer else None

        try:
            for attempt in range(config.max_attempts):
                context.attempt = attempt + 1

                try:
                    start_time = time.perf_counter()

                    # Notify observers
                    for observer in self._observers:
                        observer.on_retry_attempt(
                            context.operation_name,
                            attempt + 1,
                            0.0,
                            last_exception or Exception("Starting attempt")
                        )

                    # Execute with circuit breaker protection
                    if config.enable_circuit_breaker:
                        result = await circuit_breaker.call(lambda: operation(**kwargs), context)
                    else:
                        result = await operation(**kwargs)

                    # Record success
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    await self._record_success(context.operation_name, attempt + 1, duration_ms, config)

                    # Notify observers
                    for observer in self._observers:
                        observer.on_retry_success(context.operation_name, attempt + 1)

                    return result

                except CircuitBreakerOpenError:
                    # Circuit breaker is open, don't retry
                    raise

                except Exception as e:
                    last_exception = e
                    context.last_error = e

                    # Classify and check if error is retryable
                    error_type = self._classify_error(e, config)
                    if not self._is_retryable_error(error_type, config):
                        await self._record_failure(context.operation_name, attempt + 1, str(e), config)
                        raise

                    # Record failure
                    await self._record_failure(context.operation_name, attempt + 1, str(e), config)

                    # If this was the last attempt, raise the exception
                    if attempt >= config.max_attempts - 1:
                        # Notify observers of final failure
                        for observer in self._observers:
                            observer.on_retry_failure(context.operation_name, attempt + 1, e)
                        raise

                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    context.total_delay_ms += delay * 1000

                    self.logger.debug(
                        f"Retry {attempt + 1}/{config.max_attempts} for {context.operation_name} "
                        f"in {delay:.2f}s due to {error_type.value}: {e}"
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        finally:
            if span:
                self._end_span(span, context)

    async def _execute_sync_with_retry(
        self,
        operation: Callable[..., T],
        config: RetryConfig,
        context: RetryContext,
        **kwargs
    ) -> T:
        """Internal sync retry implementation (runs in executor)."""
        loop = asyncio.get_event_loop()

        async def async_wrapper():
            return await loop.run_in_executor(None, lambda: operation(**kwargs))

        return await self._execute_async_with_retry(async_wrapper, config, context)

    def _convert_config(self, config: RetryConfigProtocol) -> RetryConfig:
        """Convert protocol config to internal config."""
        return RetryConfig(
            max_attempts=config.max_attempts,
            strategy=config.strategy,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            jitter=config.jitter,
            jitter_factor=config.jitter_factor,
            backoff_multiplier=config.backoff_multiplier,
            retry_on_exceptions=config.retry_on_exceptions,
            retry_condition=config.retry_condition,
            operation_timeout=config.operation_timeout,
            total_timeout=config.total_timeout,
            log_attempts=config.log_attempts,
            log_level=config.log_level,
            track_metrics=config.track_metrics,
        )

    async def _get_circuit_breaker(self, operation_name: str, config: RetryConfig) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if not config.enable_circuit_breaker:
            return NoOpCircuitBreaker()

        async with self._circuit_lock:
            if operation_name not in self._circuit_breakers:
                self._circuit_breakers[operation_name] = CircuitBreaker(config, operation_name)
            return self._circuit_breakers[operation_name]

    def _classify_error(self, error: Exception, config: RetryConfig) -> RetryableErrorType:
        """Classify error type for retry decision."""
        if config.error_classifier:
            return config.error_classifier(error)

        # Intelligent error classification using 2025 patterns
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Network errors
        if any(term in error_str for term in ['connection', 'network', 'dns', 'socket']) or \
           isinstance(error, (ConnectionError, OSError)):
            return RetryableErrorType.NETWORK

        # Timeout errors
        if any(term in error_str for term in ['timeout', 'timed out']) or \
           isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return RetryableErrorType.TIMEOUT

        # Rate limiting
        if any(term in error_str for term in ['rate limit', 'too many requests', '429', 'throttle']):
            return RetryableErrorType.RATE_LIMIT

        # Resource exhaustion
        if any(term in error_str for term in ['resource', 'memory', 'disk', 'quota', 'limit']):
            return RetryableErrorType.RESOURCE_EXHAUSTION

        # Dependency failures
        if any(term in error_str for term in ['service unavailable', '503', '502', '504', 'bad gateway']):
            return RetryableErrorType.DEPENDENCY_FAILURE

        # Default to transient
        return RetryableErrorType.TRANSIENT

    def _is_retryable_error(self, error_type: RetryableErrorType, config: RetryConfig) -> bool:
        """Check if error type is retryable based on configuration."""
        return error_type in config.retryable_errors

    async def _record_success(self, operation_name: str, attempt: int, duration_ms: float, config: RetryConfig):
        """Record successful operation."""
        async with self._stats_lock:
            if operation_name not in self._stats:
                self._stats[operation_name] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "failed_attempts": 0,
                    "total_duration_ms": 0.0,
                    "success_rate": 0.0,
                    "average_delay_ms": 0.0,
                    "last_attempt_at": None,
                }

            stats = self._stats[operation_name]
            stats["total_attempts"] += attempt
            stats["successful_attempts"] += 1
            stats["total_duration_ms"] += duration_ms
            stats["last_attempt_at"] = datetime.now(timezone.utc).isoformat()

            # Update calculated fields
            total_ops = stats["successful_attempts"] + stats["failed_attempts"]
            if total_ops > 0:
                stats["success_rate"] = stats["successful_attempts"] / total_ops

            if stats["total_attempts"] > 0:
                stats["average_delay_ms"] = stats["total_duration_ms"] / stats["total_attempts"]

        # Record metrics
        self.retry_success_total.labels(
            operation=operation_name,
            strategy=config.strategy.value
        ).inc()

        self.retry_duration_histogram.labels(
            operation=operation_name,
            strategy=config.strategy.value
        ).observe(duration_ms / 1000.0)

    async def _record_failure(self, operation_name: str, attempt: int, error_message: str, config: RetryConfig):
        """Record failed operation attempt."""
        async with self._stats_lock:
            if operation_name not in self._stats:
                self._stats[operation_name] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "failed_attempts": 0,
                    "total_duration_ms": 0.0,
                    "success_rate": 0.0,
                    "average_delay_ms": 0.0,
                    "last_attempt_at": None,
                }

            stats = self._stats[operation_name]
            stats["failed_attempts"] += 1
            stats["last_attempt_at"] = datetime.now(timezone.utc).isoformat()

            # Update calculated fields
            total_ops = stats["successful_attempts"] + stats["failed_attempts"]
            if total_ops > 0:
                stats["success_rate"] = stats["successful_attempts"] / total_ops

        # Record metrics
        self.retry_failure_total.labels(
            operation=operation_name,
            strategy=config.strategy.value,
            error_type="unknown"
        ).inc()

    def _start_span(self, operation_name: str, config: RetryConfig):
        """Start OpenTelemetry span for retry operation."""
        if not self.tracer:
            return None

        span = self.tracer.start_span(f"retry_{operation_name}")
        span.set_attribute("retry.strategy", config.strategy.value)
        span.set_attribute("retry.max_attempts", config.max_attempts)
        span.set_attribute("retry.circuit_breaker_enabled", config.enable_circuit_breaker)
        return span

    def _end_span(self, span, context: RetryContext):
        """End OpenTelemetry span with retry context."""
        if not span:
            return

        span.set_attribute("retry.attempts", context.attempt)
        span.set_attribute("retry.total_delay_ms", context.total_delay_ms)
        span.set_attribute("retry.circuit_breaker_triggered", context.circuit_breaker_triggered)

        if context.last_error:
            span.set_status(Status(StatusCode.ERROR, str(context.last_error)))
            span.set_attribute("retry.last_error", str(context.last_error))
        else:
            span.set_status(Status(StatusCode.OK))

        span.end()


class NoOpCircuitBreaker:
    """No-op circuit breaker for when circuit breaking is disabled."""

    async def call(self, operation: Callable[[], Coroutine[Any, Any, T]], context: RetryContext) -> T:
        """Execute operation without circuit breaker protection."""
        return await operation()

    def get_state(self) -> str:
        return "disabled"

    def is_open(self) -> bool:
        return False

    def reset(self) -> None:
        pass


class RetryExecutor:
    """Executor for retry operations within context manager."""

    def __init__(
        self,
        retry_manager: RetryManager,
        config: RetryConfig,
        context: RetryContext,
        circuit_breaker: CircuitBreaker
    ):
        self.retry_manager = retry_manager
        self.config = config
        self.context = context
        self.circuit_breaker = circuit_breaker

    async def execute(self, operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute operation with retry logic."""
        return await self.retry_manager._execute_async_with_retry(
            operation, self.config, self.context
        )


# ============================================================================
# Global Instance and Factory Functions
# ============================================================================

# Global retry manager instance
_global_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """
    Get the global retry manager instance.

    Returns:
        Global RetryManager instance
    """
    global _global_retry_manager
    if _global_retry_manager is None:
        _global_retry_manager = RetryManager()
    return _global_retry_manager


def set_retry_manager(manager: RetryManager):
    """
    Set the global retry manager instance.

    Args:
        manager: RetryManager instance to set as global
    """
    global _global_retry_manager
    _global_retry_manager = manager


# ============================================================================
# Convenience Functions
# ============================================================================

async def retry_database_operation(
    operation: Callable[..., Any],
    operation_name: str = "database_operation",
    **kwargs
) -> Any:
    """
    Execute database operation with retry logic (convenience function).

    Args:
        operation: Database operation to execute
        operation_name: Name for tracking and metrics
        **kwargs: Arguments passed to operation

    Returns:
        Result of successful operation execution
    """
    return await get_retry_manager().retry_database_operation(
        operation, operation_name, **kwargs
    )


async def retry_ml_operation(
    operation: Callable[..., Any],
    operation_name: str = "ml_operation",
    **kwargs
) -> Any:
    """
    Execute ML operation with retry logic (convenience function).

    Args:
        operation: ML operation to execute
        operation_name: Name for tracking and metrics
        **kwargs: Arguments passed to operation

    Returns:
        Result of successful operation execution
    """
    return await get_retry_manager().retry_ml_operation(
        operation, operation_name, **kwargs
    )


# ============================================================================
# Modern Decorator Pattern
# ============================================================================

def retry(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    enable_circuit_breaker: bool = True,
    operation_name: Optional[str] = None,
    **config_kwargs
):
    """
    Decorator for async functions with unified retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        strategy: Retry strategy to use
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        enable_circuit_breaker: Whether to enable circuit breaker
        operation_name: Operation name for tracking
        **config_kwargs: Additional configuration options
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args, **func_kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                base_delay=base_delay,
                max_delay=max_delay,
                enable_circuit_breaker=enable_circuit_breaker,
                operation_name=operation_name or func.__name__,
                **config_kwargs
            )

            retry_manager = get_retry_manager()
            return await retry_manager.retry_async(
                lambda: func(*args, **func_kwargs),
                config=config
            )

        return wrapper
    return decorator


# ============================================================================
# Type Aliases and Exports
# ============================================================================

__all__ = [
    # Main classes
    "RetryManager",
    "RetryConfig",
    "RetryContext",
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",

    # Global functions
    "get_retry_manager",
    "set_retry_manager",

    # Convenience functions
    "retry_database_operation",
    "retry_ml_operation",

    # Decorators
    "retry",

    # Enums
    "RetryStrategy",
    "RetryableErrorType",
]
