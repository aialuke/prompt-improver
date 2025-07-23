"""
Database Error Handling with Unified Retry Manager Integration.

Provides error classification and circuit breaker functionality
integrated with the unified retry manager.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import psycopg
from psycopg import errors as psycopg_errors

from prompt_improver.utils.datetime_utils import aware_utc_now

# Use centralized metrics registry
from ..performance.monitoring.metrics_registry import (
    get_metrics_registry,
    StandardMetrics,
    PROMETHEUS_AVAILABLE
)

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Database error categories for classification."""

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    CONSTRAINT = "constraint"
    SYNTAX = "syntax"
    PERMISSION = "permission"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DatabaseErrorClassifier:
    """Classifies database errors for appropriate handling."""

    # Error classification mappings
    ERROR_MAPPINGS = {
        # Connection errors
        psycopg_errors.OperationalError: (ErrorCategory.CONNECTION, ErrorSeverity.HIGH),
        psycopg_errors.InterfaceError: (ErrorCategory.CONNECTION, ErrorSeverity.HIGH),

        # Timeout errors
        psycopg_errors.QueryCanceled: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),

        # Transient errors
        psycopg_errors.DeadlockDetected: (ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM),
        psycopg_errors.SerializationFailure: (ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM),

        # Constraint violations
        psycopg_errors.IntegrityError: (ErrorCategory.CONSTRAINT, ErrorSeverity.LOW),
        psycopg_errors.UniqueViolation: (ErrorCategory.CONSTRAINT, ErrorSeverity.LOW),
        psycopg_errors.ForeignKeyViolation: (ErrorCategory.CONSTRAINT, ErrorSeverity.LOW),

        # Syntax errors
        psycopg_errors.SyntaxError: (ErrorCategory.SYNTAX, ErrorSeverity.HIGH),
        psycopg_errors.UndefinedTable: (ErrorCategory.SYNTAX, ErrorSeverity.HIGH),
        psycopg_errors.UndefinedColumn: (ErrorCategory.SYNTAX, ErrorSeverity.HIGH),

        # Permission errors
        psycopg_errors.InsufficientPrivilege: (ErrorCategory.PERMISSION, ErrorSeverity.HIGH),

        # Resource errors
        psycopg_errors.DiskFull: (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
        psycopg_errors.OutOfMemory: (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
    }

    @classmethod
    def classify_error(cls, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify database error into category and severity."""
        error_type = type(error)

        # Direct mapping
        if error_type in cls.ERROR_MAPPINGS:
            return cls.ERROR_MAPPINGS[error_type]

        # Check parent classes
        for mapped_type, (category, severity) in cls.ERROR_MAPPINGS.items():
            if isinstance(error, mapped_type):
                return category, severity

        # Fallback to string analysis for unknown errors
        error_str = str(error).lower()

        if any(term in error_str for term in ['connection', 'connect', 'network']):
            return ErrorCategory.CONNECTION, ErrorSeverity.HIGH
        elif any(term in error_str for term in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        elif any(term in error_str for term in ['deadlock', 'serialization']):
            return ErrorCategory.TRANSIENT, ErrorSeverity.MEDIUM
        elif any(term in error_str for term in ['permission', 'privilege']):
            return ErrorCategory.PERMISSION, ErrorSeverity.HIGH
        elif any(term in error_str for term in ['syntax', 'undefined']):
            return ErrorCategory.SYNTAX, ErrorSeverity.HIGH

        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """Determine if error is retryable."""
        category, _ = cls.classify_error(error)

        # Retryable categories
        retryable_categories = {
            ErrorCategory.CONNECTION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.TRANSIENT,
            ErrorCategory.RESOURCE
        }

        return category in retryable_categories


class ErrorMetrics:
    """Metrics collection for database errors."""

    def __init__(self) -> None:
        self.metrics_registry = get_metrics_registry()

        self.error_counter = self.metrics_registry.get_or_create_counter(
            StandardMetrics.DATABASE_ERRORS_TOTAL,
            'Total database errors',
            ['category', 'severity', 'operation']
        )

        self.retry_counter = self.metrics_registry.get_or_create_counter(
            'database_retries_total',
            'Total database retry attempts',
            ['operation', 'attempt']
        )

        self.circuit_breaker_gauge = self.metrics_registry.get_or_create_gauge(
            'database_circuit_breaker_state',
            'Database circuit breaker state',
            ['operation']
        )

    def record_error(self, category: ErrorCategory, severity: ErrorSeverity, operation: str) -> None:
        """Record database error metrics."""
        self.error_counter.labels(
            category=category.value,
            severity=severity.value,
            operation=operation
        ).inc()

    def record_retry(self, operation: str, attempt: int) -> None:
        """Record retry attempt metrics."""
        self.retry_counter.labels(
            operation=operation,
            attempt=str(attempt)
        ).inc()

    def update_circuit_breaker_state(self, operation: str, state: int):
        """Update circuit breaker state metrics."""
        self.circuit_breaker_gauge.labels(operation=operation).set(state)


# Global error metrics instance
global_error_metrics = ErrorMetrics()


def enhance_error_context(error: Exception, operation: str, **kwargs) -> Dict[str, Any]:
    """Enhance error with context information."""
    category, severity = DatabaseErrorClassifier.classify_error(error)

    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "category": category.value,
        "severity": severity.value,
        "operation": operation,
        "timestamp": aware_utc_now().isoformat(),
        "retryable": DatabaseErrorClassifier.is_retryable(error),
        **kwargs
    }

    # Record metrics
    global_error_metrics.record_error(category, severity, operation)

    return context


def create_database_error_classifier():
    """Create error classifier function for unified retry manager."""
    def classify_for_retry(error: Exception):
        """Classify database error for retry manager."""
        from ..ml.orchestration.core.unified_retry_manager import RetryableErrorType

        category, _ = DatabaseErrorClassifier.classify_error(error)

        # Map database categories to retry error types
        category_mapping = {
            ErrorCategory.CONNECTION: RetryableErrorType.NETWORK,
            ErrorCategory.TIMEOUT: RetryableErrorType.TIMEOUT,
            ErrorCategory.TRANSIENT: RetryableErrorType.TRANSIENT,
            ErrorCategory.RESOURCE: RetryableErrorType.RESOURCE_EXHAUSTION,
        }

        return category_mapping.get(category, RetryableErrorType.TRANSIENT)

    return classify_for_retry


# Default configurations for database operations
def get_default_database_retry_config():
    """Get default retry configuration for database operations."""
    from ..ml.orchestration.core.unified_retry_manager import RetryConfig, RetryStrategy, RetryableErrorType

    return RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay_ms=100,
        max_delay_ms=10000,
        multiplier=2.0,
        jitter=True,
        enable_circuit_breaker=True,
        failure_threshold=5,
        recovery_timeout_ms=60000,
        retryable_errors=[
            RetryableErrorType.NETWORK,
            RetryableErrorType.TIMEOUT,
            RetryableErrorType.TRANSIENT,
            RetryableErrorType.RESOURCE_EXHAUSTION
        ],
        error_classifier=create_database_error_classifier(),
        enable_metrics=True,
        enable_tracing=True
    )


# Convenience function for database operations with retry
async def execute_with_database_retry(operation, operation_name: str, **kwargs):
    """Execute database operation with unified retry logic."""
    from ..ml.orchestration.core.unified_retry_manager import get_retry_manager

    retry_manager = get_retry_manager()
    config = get_default_database_retry_config()
    config.operation_name = operation_name

    return await retry_manager.retry_async(operation, config=config)


class RetryManager:
    """
    Database-specific Retry Manager with 2025 best practices.

    Integrates with UnifiedRetryManager for ML Pipeline Orchestrator while providing
    database-specific error handling, circuit breaker patterns, and observability.

    Features:
    - Database-specific error classification and retry logic
    - Circuit breaker pattern with adaptive thresholds
    - Comprehensive observability with OpenTelemetry-ready metrics
    - Async/await support for modern Python patterns
    - Integration with ML Pipeline Orchestrator
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RetryManager with 2025 best practices."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize metrics
        self.error_metrics = ErrorMetrics()

        # Circuit breaker state tracking
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Default retry configuration
        self.default_config = get_default_database_retry_config()

        # Integration with unified retry manager
        self._unified_retry_manager = None

        self.logger.info("RetryManager initialized with 2025 best practices")

    async def get_unified_retry_manager(self):
        """Get or create unified retry manager instance."""
        if self._unified_retry_manager is None:
            from ..ml.orchestration.core.unified_retry_manager import get_retry_manager
            self._unified_retry_manager = get_retry_manager()
        return self._unified_retry_manager

    async def retry_database_operation(
        self,
        operation: Callable,
        operation_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Retry database operation with enhanced error handling.

        Args:
            operation: Database operation to retry
            operation_name: Name for monitoring and logging
            config: Optional retry configuration override
            **kwargs: Additional arguments for operation

        Returns:
            Result of successful operation

        Raises:
            Exception: Last exception if all retries failed
        """
        retry_config = self._create_retry_config(operation_name, config)
        unified_manager = await self.get_unified_retry_manager()

        # Wrap operation with database-specific error handling
        async def wrapped_operation():
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(**kwargs)
                else:
                    result = operation(**kwargs)
                return result
            except Exception as e:
                # Enhance error with database context
                error_context = enhance_error_context(e, operation_name, **kwargs)
                self.logger.error(f"Database operation {operation_name} failed: {error_context}")
                raise

        return await unified_manager.retry_async(wrapped_operation, config=retry_config)

    def _create_retry_config(self, operation_name: str, config_override: Optional[Dict[str, Any]] = None):
        """Create retry configuration with database-specific settings."""
        from ..ml.orchestration.core.unified_retry_manager import RetryConfig

        # Start with default database config
        config = get_default_database_retry_config()
        config.operation_name = operation_name

        # Apply any overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    async def execute_with_circuit_breaker(
        self,
        operation: Callable,
        operation_name: str,
        **kwargs
    ) -> Any:
        """Execute operation with circuit breaker pattern."""
        circuit_breaker_key = f"db_{operation_name}"

        # Check circuit breaker state
        if await self._is_circuit_breaker_open(circuit_breaker_key):
            raise Exception(f"Circuit breaker open for {operation_name}")

        try:
            result = await self.retry_database_operation(operation, operation_name, **kwargs)
            await self._record_success(circuit_breaker_key)
            return result
        except Exception as e:
            await self._record_failure(circuit_breaker_key)
            raise

    async def _is_circuit_breaker_open(self, key: str) -> bool:
        """Check if circuit breaker is open."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
            return False

        breaker = self.circuit_breakers[key]

        if breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if breaker['last_failure']:
                recovery_time = breaker['last_failure'] + timedelta(
                    milliseconds=self.default_config.recovery_timeout_ms
                )
                if aware_utc_now() > recovery_time:
                    breaker['state'] = 'half-open'
                    self.logger.info(f"Circuit breaker {key} moved to half-open state")
                    return False
            return True

        return False

    async def _record_success(self, key: str):
        """Record successful operation for circuit breaker."""
        if key in self.circuit_breakers:
            breaker = self.circuit_breakers[key]
            if breaker['state'] == 'half-open':
                breaker['state'] = 'closed'
                breaker['failures'] = 0
                self.logger.info(f"Circuit breaker {key} closed after successful operation")

    async def _record_failure(self, key: str):
        """Record failed operation for circuit breaker."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }

        breaker = self.circuit_breakers[key]
        breaker['failures'] += 1
        breaker['last_failure'] = aware_utc_now()

        if breaker['failures'] >= self.default_config.failure_threshold:
            breaker['state'] = 'open'
            self.logger.warning(f"Circuit breaker {key} opened after {breaker['failures']} failures")

            # Update metrics
            self.error_metrics.update_circuit_breaker_state(key, 1)  # 1 = open

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring."""
        return {
            'component': 'RetryManager',
            'status': 'healthy',
            'circuit_breakers': {
                key: {
                    'state': breaker['state'],
                    'failures': breaker['failures'],
                    'last_failure': breaker['last_failure'].isoformat() if breaker['last_failure'] else None
                }
                for key, breaker in self.circuit_breakers.items()
            },
            'timestamp': aware_utc_now().isoformat()
        }

    async def reset_circuit_breaker(self, operation_name: str) -> bool:
        """Reset circuit breaker for specific operation."""
        key = f"db_{operation_name}"
        if key in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }
            self.logger.info(f"Circuit breaker {key} manually reset")
            return True
        return False
