"""
Unified Retry Manager for ML Pipeline Orchestration.

Consolidates all retry logic across the system with 2025 best practices:
- Circuit breaker integration
- Comprehensive observability
- Async-first design
- Configurable strategies
- Prometheus metrics
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# Import protocol interfaces to prevent circular imports (2025 best practice)
from ....core.protocols.retry_protocols import (
    RetryStrategy,
    RetryableErrorType,
    RetryConfigProtocol,
    MetricsRegistryProtocol
)
from ....core.retry_config import RetryConfig as BaseRetryConfig

# Use centralized metrics registry to prevent duplicates
from ....performance.monitoring.metrics_registry import (
    get_metrics_registry,
    StandardMetrics,
    PROMETHEUS_AVAILABLE
)

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

T = TypeVar("T")
logger = logging.getLogger(__name__)

# RetryStrategy is now imported from protocols

class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

# RetryableErrorType is now imported from protocols

@dataclass
class RetryConfig:
    """Extended retry configuration with ML orchestration specific features."""

    # Basic retry settings (implementing RetryConfigProtocol)
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Delay settings
    base_delay: float = 0.1  # seconds (converted from initial_delay_ms)
    max_delay: float = 30.0  # seconds (converted from max_delay_ms)

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

    # ML-specific settings (additional to protocol)
    initial_delay_ms: int = field(init=False)  # Computed from base_delay
    max_delay_ms: int = field(init=False)      # Computed from max_delay

    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    success_threshold: int = 3
    recovery_timeout_ms: int = 60000  # 1 minute

    # Error classification
    retryable_errors: List[RetryableErrorType] = field(default_factory=lambda: [
        RetryableErrorType.TRANSIENT,
        RetryableErrorType.NETWORK,
        RetryableErrorType.TIMEOUT,
        RetryableErrorType.RATE_LIMIT
    ])

    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = True
    operation_name: Optional[str] = None

    # Advanced ML settings
    timeout_ms: Optional[int] = None
    custom_delay_func: Optional[Callable[[int], float]] = None
    error_classifier: Optional[Callable[[Exception], RetryableErrorType]] = None

    def __post_init__(self):
        """Initialize computed fields"""
        # Convert seconds to milliseconds for backward compatibility
        self.initial_delay_ms = int(self.base_delay * 1000)
        self.max_delay_ms = int(self.max_delay * 1000)

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
        else:
            delay = self.base_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Apply jitter if enabled
        if self.jitter and delay > 0:
            import random
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

@dataclass
class RetryContext:
    """Context for retry operations with comprehensive tracking."""

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

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker implementation with 2025 patterns."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

        # Metrics using centralized registry
        self.metrics_registry = get_metrics_registry()
        self.circuit_breaker_state_gauge = self.metrics_registry.get_or_create_gauge(
            StandardMetrics.CIRCUIT_BREAKER_STATE,
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['operation']
        )
        self.circuit_breaker_events = self.metrics_registry.get_or_create_counter(
            StandardMetrics.CIRCUIT_BREAKER_EVENTS_TOTAL,
            'Circuit breaker events',
            ['operation', 'event']
        )

    async def call(self, operation: Callable[[], Coroutine[Any, Any, T]], context: RetryContext) -> T:
        """Execute operation with circuit breaker protection."""
        if not self.config.enable_circuit_breaker:
            return await operation()

        await self._check_state_transition()

        if self.state == CircuitBreakerState.OPEN:
            context.circuit_breaker_triggered = True
            self._record_event(context.operation_name, "blocked")
            raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {context.operation_name}")

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
            if (self.state == CircuitBreakerState.OPEN and
                self.last_failure_time and
                datetime.now(timezone.utc) - self.last_failure_time >
                timedelta(milliseconds=self.config.recovery_timeout_ms)):

                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._update_state_metric()

    async def _on_success(self, context: RetryContext):
        """Handle successful operation."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self._record_event(context.operation_name, "closed")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

            self._update_state_metric()

    async def _on_failure(self, context: RetryContext, error: Exception):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if (self.state == CircuitBreakerState.CLOSED and
                self.failure_count >= self.config.failure_threshold):

                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker transitioning to OPEN for {context.operation_name}")
                self._record_event(context.operation_name, "opened")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker returning to OPEN from HALF_OPEN for {context.operation_name}")
                self._record_event(context.operation_name, "opened")

            self._update_state_metric()

    def _update_state_metric(self):
        """Update Prometheus state metric."""
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.HALF_OPEN: 1,
            CircuitBreakerState.OPEN: 2
        }[self.state]
        self.circuit_breaker_state_gauge.labels(operation="unified").set(state_value)

    def _record_event(self, operation_name: str, event: str):
        """Record circuit breaker event."""
        self.circuit_breaker_events.labels(operation=operation_name, event=event).inc()

        # Record in observability manager if available
        try:
            from .retry_observability import get_observability_manager
            observability_manager = get_observability_manager()

            state_value = {
                CircuitBreakerState.CLOSED: 0,
                CircuitBreakerState.HALF_OPEN: 1,
                CircuitBreakerState.OPEN: 2
            }[self.state]

            observability_manager.record_circuit_breaker_event(
                operation_name, event, state_value
            )
        except Exception as e:
            # Don't fail circuit breaker operation if observability fails
            logger.debug(f"Failed to record circuit breaker event in observability manager: {e}")

class UnifiedRetryManager:
    """
    Unified Retry Manager implementing 2025 best practices.

    features:
    - Multiple retry strategies
    - Circuit breaker integration
    - Comprehensive observability
    - Async-first design
    - Configurable error classification
    """

    def __init__(self, default_config: Optional[RetryConfig] = None, observability_manager=None):
        self.default_config = default_config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

        # Initialize observability
        if observability_manager:
            self.observability_manager = observability_manager
        else:
            from .retry_observability import get_observability_manager
            self.observability_manager = get_observability_manager()

        self._setup_metrics()
        self._setup_tracing()

        logger.info("UnifiedRetryManager initialized with 2025 best practices")

    def _setup_metrics(self):
        """Setup Prometheus metrics using centralized registry."""
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

        self.retry_delay_histogram = self.metrics_registry.get_or_create_histogram(
            StandardMetrics.RETRY_DELAY,
            'Retry delay duration',
            ['operation', 'strategy'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
        else:
            self.tracer = None
            self.meter = None

    async def retry_async(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        config: Optional[RetryConfig] = None,
        context: Optional[RetryContext] = None
    ) -> T:
        """
        Execute operation with unified retry logic and 2025 best practices.

        Args:
            operation: Async operation to retry
            config: Retry configuration (uses default if None)
            context: Retry context (creates new if None)

        Returns:
            Result of successful operation

        Raises:
            Exception: Last exception if all retries failed
            CircuitBreakerOpenError: If circuit breaker is open
        """
        retry_config = config or self.default_config
        operation_name = retry_config.operation_name or operation.__name__

        if context is None:
            context = RetryContext(operation_name=operation_name)

        # Get or create circuit breaker
        circuit_breaker = await self._get_circuit_breaker(operation_name, retry_config)

        # Start tracing span
        span_context = self._start_span(operation_name, retry_config) if self.tracer else None

        try:
            return await self._execute_with_retry(operation, retry_config, context, circuit_breaker)
        finally:
            if span_context:
                self._end_span(span_context, context)

    async def _execute_with_retry(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        config: RetryConfig,
        context: RetryContext,
        circuit_breaker: CircuitBreaker
    ) -> T:
        """Execute operation with retry logic."""
        last_exception = None

        for attempt in range(config.max_attempts):
            context.attempt = attempt + 1

            # Record attempt metric
            if PROMETHEUS_AVAILABLE:
                self.retry_attempts_total.labels(
                    operation=context.operation_name,
                    strategy=config.strategy.value,
                    attempt=str(attempt + 1)
                ).inc()

            try:
                # Execute with circuit breaker protection
                start_time = time.time()
                result = await circuit_breaker.call(operation, context)

                # Record success metrics
                duration = time.time() - start_time
                duration_ms = duration * 1000.0

                # Record in observability manager
                self.observability_manager.record_retry_attempt(
                    operation_name=context.operation_name,
                    strategy=config.strategy.value,
                    attempt_number=attempt + 1,
                    success=True,
                    duration_ms=duration_ms
                )

                if PROMETHEUS_AVAILABLE:
                    self.retry_success_total.labels(
                        operation=context.operation_name,
                        strategy=config.strategy.value
                    ).inc()

                    self.retry_duration_histogram.labels(
                        operation=context.operation_name,
                        strategy=config.strategy.value
                    ).observe(duration)

                logger.info(f"Operation {context.operation_name} succeeded on attempt {attempt + 1}")
                return result

            except CircuitBreakerOpenError:
                # Circuit breaker is open, don't retry
                raise

            except Exception as e:
                last_exception = e
                context.last_error = e

                # Classify error
                error_type = self._classify_error(e, config)

                # Check if error is retryable
                if not self._is_retryable_error(error_type, config):
                    logger.warning(f"Non-retryable error in {context.operation_name}: {e}")
                    break

                # Check if we should retry
                if attempt < config.max_attempts - 1:
                    delay_ms = self._calculate_delay(attempt, config)
                    context.total_delay_ms += delay_ms

                    # Record failure attempt in observability manager
                    self.observability_manager.record_retry_attempt(
                        operation_name=context.operation_name,
                        strategy=config.strategy.value,
                        attempt_number=attempt + 1,
                        success=False,
                        duration_ms=0.0,  # Failed attempt
                        error_type=error_type.value,
                        delay_ms=delay_ms
                    )

                    logger.warning(
                        f"Operation {context.operation_name} failed on attempt {attempt + 1}/{config.max_attempts}: {e}. "
                        f"Retrying in {delay_ms}ms"
                    )

                    # Record delay metric
                    if PROMETHEUS_AVAILABLE:
                        self.retry_delay_histogram.labels(
                            operation=context.operation_name,
                            strategy=config.strategy.value
                        ).observe(delay_ms / 1000.0)

                    # Wait before retry
                    await asyncio.sleep(delay_ms / 1000.0)
                else:
                    # Final failure - record in observability manager
                    self.observability_manager.record_retry_attempt(
                        operation_name=context.operation_name,
                        strategy=config.strategy.value,
                        attempt_number=attempt + 1,
                        success=False,
                        duration_ms=0.0,
                        error_type=error_type.value
                    )
                    logger.error(f"Operation {context.operation_name} failed after {config.max_attempts} attempts: {e}")

        # Record final failure
        if PROMETHEUS_AVAILABLE and last_exception:
            error_type = self._classify_error(last_exception, config)
            self.retry_failure_total.labels(
                operation=context.operation_name,
                strategy=config.strategy.value,
                error_type=error_type.value
            ).inc()

        # Raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation {context.operation_name} failed without exception")

    async def _get_circuit_breaker(self, operation_name: str, config: RetryConfig) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if not config.enable_circuit_breaker:
            # Return a no-op circuit breaker
            return NoOpCircuitBreaker()

        async with self._lock:
            if operation_name not in self.circuit_breakers:
                self.circuit_breakers[operation_name] = CircuitBreaker(config)
            return self.circuit_breakers[operation_name]

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt using configured strategy."""
        if config.custom_delay_func:
            base_delay = config.custom_delay_func(attempt)
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = config.initial_delay_ms * (config.multiplier ** attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = config.initial_delay_ms * (1 + attempt)
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            base_delay = config.initial_delay_ms
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            base_delay = config.initial_delay_ms * self._fibonacci(attempt + 1)
        else:
            base_delay = config.initial_delay_ms * (config.multiplier ** attempt)

        # Apply max delay cap
        delay = min(base_delay, config.max_delay_ms)

        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)

        return delay

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def _classify_error(self, error: Exception, config: RetryConfig) -> RetryableErrorType:
        """Classify error type for retry decision."""
        if config.error_classifier:
            return config.error_classifier(error)

        # Default error classification
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Network errors
        if any(term in error_str for term in ['connection', 'network', 'dns', 'socket']):
            return RetryableErrorType.NETWORK

        # Timeout errors
        if any(term in error_str for term in ['timeout', 'timed out']):
            return RetryableErrorType.TIMEOUT

        # Rate limiting
        if any(term in error_str for term in ['rate limit', 'too many requests', '429']):
            return RetryableErrorType.RATE_LIMIT

        # Resource exhaustion
        if any(term in error_str for term in ['resource', 'memory', 'disk', 'quota']):
            return RetryableErrorType.RESOURCE_EXHAUSTION

        # Dependency failures
        if any(term in error_str for term in ['service unavailable', '503', '502', '504']):
            return RetryableErrorType.DEPENDENCY_FAILURE

        # Default to transient
        return RetryableErrorType.TRANSIENT

    def _is_retryable_error(self, error_type: RetryableErrorType, config: RetryConfig) -> bool:
        """Check if error type is retryable based on configuration."""
        return error_type in config.retryable_errors

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

    async def get_circuit_breaker_status(self, operation_name: str) -> Dict[str, Any]:
        """Get circuit breaker status for operation."""
        if operation_name not in self.circuit_breakers:
            return {"state": "not_configured", "failure_count": 0}

        cb = self.circuit_breakers[operation_name]
        return {
            "state": cb.state.value,
            "failure_count": cb.failure_count,
            "success_count": cb.success_count,
            "last_failure_time": cb.last_failure_time.isoformat() if cb.last_failure_time else None
        }

    async def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker for operation."""
        if operation_name in self.circuit_breakers:
            cb = self.circuit_breakers[operation_name]
            async with cb._lock:
                cb.state = CircuitBreakerState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                cb.last_failure_time = None
                cb._update_state_metric()
            logger.info(f"Circuit breaker reset for {operation_name}")

    def get_retry_metrics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive retry metrics for operation."""
        metrics = self.observability_manager.get_operation_metrics(operation_name)
        if not metrics:
            return None

        return {
            "operation_name": metrics.operation_name,
            "total_attempts": metrics.total_attempts,
            "successful_attempts": metrics.successful_attempts,
            "failed_attempts": metrics.failed_attempts,
            "success_rate": metrics.success_rate,
            "failure_rate": metrics.failure_rate,
            "circuit_breaker_trips": metrics.circuit_breaker_trips,
            "avg_delay_ms": metrics.avg_delay_ms,
            "total_delay_ms": metrics.total_delay_ms,
            "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
            "last_failure_time": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
            "error_types": metrics.error_types,
            "p50_response_time_ms": metrics.p50_response_time_ms,
            "p95_response_time_ms": metrics.p95_response_time_ms,
            "p99_response_time_ms": metrics.p99_response_time_ms
        }

    def get_all_retry_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all operations."""
        all_metrics = self.observability_manager.get_all_metrics()
        return {
            operation_name: self.get_retry_metrics(operation_name)
            for operation_name in all_metrics.keys()
        }

    def add_alert_rule(self, rule_name: str, condition: str, severity: str, threshold: float):
        """Add custom alert rule."""
        from .retry_observability import AlertRule, AlertSeverity

        rule = AlertRule(
            name=rule_name,
            condition=condition,
            severity=AlertSeverity(severity),
            threshold=threshold
        )
        self.observability_manager.add_alert_rule(rule)

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        self.observability_manager.remove_alert_rule(rule_name)

class NoOpCircuitBreaker:
    """No-op circuit breaker for when circuit breaking is disabled."""

    async def call(self, operation: Callable[[], Coroutine[Any, Any, T]], context: RetryContext) -> T:
        """Execute operation without circuit breaker protection."""
        return await operation()

class RetryExecutor:
    """Executor for retry operations within context manager."""

    def __init__(
        self,
        retry_manager: UnifiedRetryManager,
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
        return await self.retry_manager._execute_with_retry(
            operation, self.config, self.context, self.circuit_breaker
        )

# Global instance for easy access
_global_retry_manager: Optional[UnifiedRetryManager] = None

def get_retry_manager() -> UnifiedRetryManager:
    """Get global retry manager instance."""
    global _global_retry_manager
    if _global_retry_manager is None:
        _global_retry_manager = UnifiedRetryManager()
    return _global_retry_manager

def set_retry_manager(manager: UnifiedRetryManager):
    """Set global retry manager instance."""
    global _global_retry_manager
    _global_retry_manager = manager

# Convenience decorators for 2025 patterns
def retry(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay_ms: int = 100,
    max_delay_ms: int = 30000,
    enable_circuit_breaker: bool = True,
    **kwargs
):
    """Decorator for async functions with unified retry logic."""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args, **func_kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                initial_delay_ms=initial_delay_ms,
                max_delay_ms=max_delay_ms,
                enable_circuit_breaker=enable_circuit_breaker,
                operation_name=func.__name__,
                **kwargs
            )

            retry_manager = get_retry_manager()
            return await retry_manager.retry_async(
                lambda: func(*args, **func_kwargs),
                config=config
            )

        return wrapper
    return decorator
