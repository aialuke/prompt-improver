"""RetryOrchestratorService - Coordination Hub for Retry System

High-performance orchestration service that coordinates retry configuration,
backoff strategies, and circuit breaker services with <10ms decision performance.

Key Features:
- Service coordination with dependency injection
- Observer pattern and event broadcasting
- OpenTelemetry tracing and metrics coordination
- Context management and error classification
- Support for both sync and async operations
- Background task integration for async operations
- Real behavior testable methods

Architecture:
- Protocol-based interface (RetryOrchestratorProtocol)
- Dependency injection for configuration, backoff, and circuit breaker services
- Comprehensive observability with tracing spans
- Event-driven observer pattern for extensibility
- Performance optimization for sub-10ms retry decisions
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, runtime_checkable

from prompt_improver.core.config.retry import RetryConfig
from prompt_improver.core.protocols.retry_protocols import (
    MetricsRegistryProtocol,
    RetryConfigProtocol,
    RetryObserverProtocol,
    RetryableErrorType,
)
from prompt_improver.core.services.resilience.backoff_strategy_service import (
    BackoffStrategyProtocol,
    get_backoff_strategy_service,
)
from prompt_improver.core.services.resilience.circuit_breaker_service import (
    CircuitBreakerService,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)
from prompt_improver.core.services.resilience.retry_configuration_service import (
    RetryConfigurationProtocol,
    get_retry_configuration_service,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# OpenTelemetry conditional imports
OPENTELEMETRY_AVAILABLE = False
try:
    from opentelemetry import metrics as otel_metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    trace = None
    otel_metrics = None
    Status = None
    StatusCode = None


@dataclass
class RetryExecutionContext:
    """Rich context for retry execution tracking and observability."""
    
    operation_name: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attempt: int = 0
    max_attempts: int = 3
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_error: Exception | None = None
    total_delay_ms: float = 0.0
    circuit_breaker_triggered: bool = False
    trace_id: str | None = None
    span_id: str | None = None
    error_types: dict[str, int] = field(default_factory=dict)
    response_times: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def record_attempt(self, error: Exception | None = None, duration_ms: float = 0.0) -> None:
        """Record attempt details for observability."""
        self.attempt += 1
        if error:
            self.last_error = error
            error_type = type(error).__name__
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        if duration_ms > 0:
            self.response_times.append(duration_ms)
    
    def is_final_attempt(self) -> bool:
        """Check if this is the final retry attempt."""
        return self.attempt >= self.max_attempts


@runtime_checkable
class RetryOrchestratorProtocol(Protocol):
    """Protocol for retry orchestrator service implementations."""
    
    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol | None = None,
        operation_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with coordinated retry logic."""
        ...
    
    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        circuit_name: str,
        config: RetryConfigProtocol | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        ...
    
    def add_observer(self, observer: RetryObserverProtocol) -> None:
        """Add retry event observer."""
        ...
    
    def remove_observer(self, observer: RetryObserverProtocol) -> None:
        """Remove retry event observer."""
        ...
    
    def get_execution_metrics(self, operation_name: str) -> dict[str, Any]:
        """Get comprehensive execution metrics."""
        ...


class RetryOrchestratorService:
    """High-performance retry orchestration service.
    
    Coordinates retry configuration, backoff strategies, and circuit breaker services
    with sub-10ms decision performance and comprehensive observability.
    """
    
    def __init__(
        self,
        config_service: RetryConfigurationProtocol | None = None,
        backoff_service: BackoffStrategyProtocol | None = None,
        circuit_breaker_service: CircuitBreakerService | None = None,
        metrics_registry: MetricsRegistryProtocol | None = None,
    ):
        """Initialize orchestrator with dependency injection.
        
        Args:
            config_service: Retry configuration service
            backoff_service: Backoff strategy service
            circuit_breaker_service: Circuit breaker service
            metrics_registry: Metrics registry for observability
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Dependency injection with lazy loading fallbacks
        self._config_service = config_service or get_retry_configuration_service()
        self._backoff_service = backoff_service or get_backoff_strategy_service()
        self._circuit_breaker_service = circuit_breaker_service or CircuitBreakerService(metrics_registry)
        self._metrics_registry = metrics_registry
        
        # Observer pattern support
        self._observers: list[RetryObserverProtocol] = []
        
        # Execution tracking
        self._execution_stats: dict[str, dict[str, Any]] = {}
        self._stats_lock = asyncio.Lock()
        
        # OpenTelemetry setup
        self._setup_tracing()
        self._setup_metrics()
        
        self.logger.info("RetryOrchestratorService initialized with service coordination")
    
    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if OPENTELEMETRY_AVAILABLE and trace:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None
    
    def _setup_metrics(self) -> None:
        """Setup metrics collection."""
        if self._metrics_registry:
            # Use injected metrics registry
            self.orchestration_duration = self._metrics_registry.get_metric("retry_orchestration_duration")
            self.decision_time = self._metrics_registry.get_metric("retry_decision_time")
        else:
            # Fallback to None for metrics
            self.orchestration_duration = None
            self.decision_time = None
    
    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        config: RetryConfigProtocol | None = None,
        operation_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with coordinated retry logic.
        
        Args:
            operation: Callable to execute with retry logic
            config: Retry configuration (uses template if None)
            operation_name: Operation name for tracking and metrics
            **kwargs: Arguments passed to operation
            
        Returns:
            Result of successful operation execution
        """
        start_time = time.perf_counter()
        
        # Resolve operation name
        op_name = operation_name or getattr(operation, "__name__", "unknown_operation")
        
        # Get or create configuration
        if config:
            retry_config = config if isinstance(config, RetryConfig) else self._convert_config(config)
        else:
            # Use configuration service to get appropriate template
            retry_config = self._config_service.create_config(
                domain="general", operation="default", operation_name=op_name
            )
        
        # Create execution context
        context = RetryExecutionContext(
            operation_name=op_name,
            max_attempts=retry_config.max_attempts,
        )
        
        # Start tracing span
        span = self._start_span(f"retry_orchestration_{op_name}", context) if self.tracer else None
        
        try:
            # Execute with coordination
            if asyncio.iscoroutinefunction(operation):
                result = await self._execute_async_with_coordination(
                    operation, retry_config, context, **kwargs
                )
            else:
                result = await self._execute_sync_with_coordination(
                    operation, retry_config, context, **kwargs
                )
            
            # Record success metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_execution_success(context, duration_ms)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_execution_failure(context, e, duration_ms)
            raise
            
        finally:
            if span:
                self._end_span(span, context)
    
    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        circuit_name: str,
        config: RetryConfigProtocol | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with circuit breaker protection.
        
        Args:
            operation: Callable to execute
            circuit_name: Circuit breaker name
            config: Configuration for circuit breaker
            **kwargs: Arguments passed to operation
            
        Returns:
            Result of successful operation execution
        """
        # Setup circuit breaker with configuration
        circuit_config = CircuitBreakerConfig()
        if config and hasattr(config, 'failure_threshold'):
            circuit_config.failure_threshold = getattr(config, 'failure_threshold', 5)
        
        await self._circuit_breaker_service.setup_circuit_breaker(circuit_name, circuit_config)
        
        # Execute through circuit breaker
        return await self._circuit_breaker_service.call(circuit_name, operation, **kwargs)
    
    async def _execute_async_with_coordination(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        config: RetryConfig,
        context: RetryExecutionContext,
        **kwargs: Any,
    ) -> T:
        """Execute async operation with full service coordination."""
        last_exception = None
        
        for attempt in range(config.max_attempts):
            context.attempt = attempt + 1
            attempt_start = time.perf_counter()
            
            try:
                # Notify observers of attempt
                self._notify_observers_attempt(context, last_exception)
                
                # Execute operation
                result = await operation(**kwargs)
                
                # Record successful attempt
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                context.record_attempt(duration_ms=duration_ms)
                
                # Notify observers of success
                self._notify_observers_success(context)
                
                return result
                
            except CircuitBreakerOpenError:
                context.circuit_breaker_triggered = True
                raise
                
            except Exception as e:
                last_exception = e
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                context.record_attempt(error=e, duration_ms=duration_ms)
                
                # Check if should retry
                if not config.should_retry(e, attempt) or context.is_final_attempt():
                    self._notify_observers_failure(context, e)
                    raise
                
                # Calculate delay using backoff service
                delay = self._backoff_service.calculate_delay(
                    strategy=config.strategy,
                    attempt=attempt,
                    base_delay=config.base_delay,
                    max_delay=config.max_delay,
                    jitter=config.jitter,
                    jitter_factor=config.jitter_factor,
                    backoff_multiplier=config.backoff_multiplier,
                )
                
                context.total_delay_ms += delay * 1000
                
                self.logger.debug(
                    f"Retry {attempt + 1}/{config.max_attempts} for {context.operation_name} "
                    f"in {delay:.2f}s due to {type(e).__name__}: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # Should not reach here due to logic above, but safety fallback
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry orchestration logic error")
    
    async def _execute_sync_with_coordination(
        self,
        operation: Callable[..., T],
        config: RetryConfig,
        context: RetryExecutionContext,
        **kwargs: Any,
    ) -> T:
        """Execute sync operation with coordination (runs in executor)."""
        loop = asyncio.get_event_loop()
        
        async def async_wrapper() -> T:
            return await loop.run_in_executor(None, lambda: operation(**kwargs))
        
        return await self._execute_async_with_coordination(async_wrapper, config, context)
    
    def _convert_config(self, config: RetryConfigProtocol) -> RetryConfig:
        """Convert protocol config to concrete RetryConfig."""
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
    
    def _notify_observers_attempt(self, context: RetryExecutionContext, error: Exception | None) -> None:
        """Notify observers of retry attempt."""
        for observer in self._observers:
            try:
                observer.on_retry_attempt(
                    context.operation_name,
                    context.attempt,
                    context.total_delay_ms / 1000,
                    error or Exception("Starting attempt"),
                )
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def _notify_observers_success(self, context: RetryExecutionContext) -> None:
        """Notify observers of retry success."""
        for observer in self._observers:
            try:
                observer.on_retry_success(context.operation_name, context.attempt)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def _notify_observers_failure(self, context: RetryExecutionContext, error: Exception) -> None:
        """Notify observers of retry failure."""
        for observer in self._observers:
            try:
                observer.on_retry_failure(context.operation_name, context.attempt, error)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    async def _record_execution_success(self, context: RetryExecutionContext, duration_ms: float) -> None:
        """Record successful execution statistics."""
        async with self._stats_lock:
            stats = self._execution_stats.setdefault(context.operation_name, {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_attempts": 0,
                "average_duration_ms": 0.0,
                "success_rate": 0.0,
                "last_execution": None,
            })
            
            stats["total_executions"] += 1
            stats["successful_executions"] += 1
            stats["total_attempts"] += context.attempt
            stats["last_execution"] = datetime.now(UTC).isoformat()
            
            # Update averages
            total_ops = stats["total_executions"]
            stats["success_rate"] = stats["successful_executions"] / total_ops
            stats["average_duration_ms"] = (
                (stats["average_duration_ms"] * (total_ops - 1) + duration_ms) / total_ops
            )
    
    async def _record_execution_failure(
        self, context: RetryExecutionContext, error: Exception, duration_ms: float
    ) -> None:
        """Record failed execution statistics."""
        async with self._stats_lock:
            stats = self._execution_stats.setdefault(context.operation_name, {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_attempts": 0,
                "average_duration_ms": 0.0,
                "success_rate": 0.0,
                "last_execution": None,
                "last_error": None,
            })
            
            stats["total_executions"] += 1
            stats["failed_executions"] += 1
            stats["total_attempts"] += context.attempt
            stats["last_execution"] = datetime.now(UTC).isoformat()
            stats["last_error"] = str(error)
            
            # Update averages
            total_ops = stats["total_executions"]
            stats["success_rate"] = stats["successful_executions"] / total_ops
            stats["average_duration_ms"] = (
                (stats["average_duration_ms"] * (total_ops - 1) + duration_ms) / total_ops
            )
    
    def _start_span(self, operation_name: str, context: RetryExecutionContext):
        """Start OpenTelemetry span for retry orchestration."""
        if not self.tracer:
            return None
        
        span = self.tracer.start_span(operation_name)
        span.set_attribute("retry.correlation_id", context.correlation_id)
        span.set_attribute("retry.operation_name", context.operation_name)
        span.set_attribute("retry.max_attempts", context.max_attempts)
        
        # Store trace context
        if hasattr(span, "get_span_context"):
            span_context = span.get_span_context()
            context.trace_id = f"{span_context.trace_id:032x}" if span_context.trace_id else None
            context.span_id = f"{span_context.span_id:016x}" if span_context.span_id else None
        
        return span
    
    def _end_span(self, span: Any, context: RetryExecutionContext) -> None:
        """End OpenTelemetry span with execution context."""
        if not span:
            return
        
        span.set_attribute("retry.total_attempts", context.attempt)
        span.set_attribute("retry.total_delay_ms", context.total_delay_ms)
        span.set_attribute("retry.circuit_breaker_triggered", context.circuit_breaker_triggered)
        
        if context.last_error and Status and StatusCode:
            span.set_status(Status(StatusCode.ERROR, str(context.last_error)))
            span.set_attribute("retry.last_error", str(context.last_error))
        elif Status and StatusCode:
            span.set_status(Status(StatusCode.OK))
        
        span.end()
    
    def add_observer(self, observer: RetryObserverProtocol) -> None:
        """Add retry event observer."""
        if observer not in self._observers:
            self._observers.append(observer)
            self.logger.debug(f"Added retry observer: {type(observer).__name__}")
    
    def remove_observer(self, observer: RetryObserverProtocol) -> None:
        """Remove retry event observer."""
        if observer in self._observers:
            self._observers.remove(observer)
            self.logger.debug(f"Removed retry observer: {type(observer).__name__}")
    
    def get_execution_metrics(self, operation_name: str) -> dict[str, Any]:
        """Get comprehensive execution metrics for operation."""
        stats = self._execution_stats.get(operation_name, {})
        
        return {
            "operation_name": operation_name,
            "execution_stats": stats,
            "configuration_cache_stats": self._config_service.get_cache_stats(),
            "backoff_performance_metrics": self._backoff_service.get_performance_metrics(),
            "observer_count": len(self._observers),
            "health_status": "healthy" if stats.get("success_rate", 0.0) > 0.8 else "degraded",
        }
    
    @asynccontextmanager
    async def with_retry_context(self, operation_name: str, config: RetryConfigProtocol | None = None):
        """Context manager for retry operations with cleanup."""
        context = RetryExecutionContext(operation_name=operation_name)
        span = self._start_span(f"retry_context_{operation_name}", context) if self.tracer else None
        
        try:
            yield context
        finally:
            if span:
                self._end_span(span, context)


# Global service instance
_global_orchestrator: RetryOrchestratorService | None = None


def get_retry_orchestrator_service() -> RetryOrchestratorService:
    """Get the global retry orchestrator service instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = RetryOrchestratorService()
    return _global_orchestrator


def set_retry_orchestrator_service(service: RetryOrchestratorService) -> None:
    """Set the global retry orchestrator service instance."""
    global _global_orchestrator
    _global_orchestrator = service


__all__ = [
    "RetryExecutionContext",
    "RetryOrchestratorProtocol",
    "RetryOrchestratorService",
    "get_retry_orchestrator_service",
    "set_retry_orchestrator_service",
]