"""
Database Error Handling with Unified Retry Manager Integration.

Provides error classification and circuit breaker functionality
integrated with the unified retry manager.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Tuple

import asyncpg

from prompt_improver.utils.datetime_utils import aware_utc_now

# Use centralized metrics registry
from ..performance.monitoring.metrics_registry import (
    get_metrics_registry,
    StandardMetrics,
)

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Database error categories for classification."""

    connection = "connection"
    timeout = "timeout"
    transient = "transient"
    constraint = "constraint"
    syntax = "syntax"
    permission = "permission"
    resource = "resource"
    unknown = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class DatabaseErrorClassifier:
    """Classifies database errors for appropriate handling."""

    # Error classification mappings for asyncpg
    ERROR_MAPPINGS = {
        # Connection errors
        asyncpg.ConnectionDoesNotExistError: (ErrorCategory.connection, ErrorSeverity.high),
        asyncpg.ConnectionFailureError: (ErrorCategory.connection, ErrorSeverity.high),
        asyncpg.InterfaceError: (ErrorCategory.connection, ErrorSeverity.high),

        # Timeout errors
        asyncio.CancelledError: (ErrorCategory.timeout, ErrorSeverity.medium),

        # Transient errors - asyncpg uses PostgresError with specific codes
        asyncpg.PostgresError: (ErrorCategory.transient, ErrorSeverity.medium),

        # Constraint violations
        asyncpg.IntegrityConstraintViolationError: (ErrorCategory.constraint, ErrorSeverity.low),
        asyncpg.UniqueViolationError: (ErrorCategory.constraint, ErrorSeverity.low),
        asyncpg.ForeignKeyViolationError: (ErrorCategory.constraint, ErrorSeverity.low),

        # Syntax errors
        asyncpg.SyntaxOrAccessError: (ErrorCategory.syntax, ErrorSeverity.high),
        asyncpg.UndefinedTableError: (ErrorCategory.syntax, ErrorSeverity.high),
        asyncpg.UndefinedColumnError: (ErrorCategory.syntax, ErrorSeverity.high),

        # Permission errors
        asyncpg.InsufficientPrivilegeError: (ErrorCategory.permission, ErrorSeverity.high),

        # Resource errors - asyncpg uses generic PostgresError for these
        asyncpg.PostgresError: (ErrorCategory.resource, ErrorSeverity.critical),
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
            return ErrorCategory.connection, ErrorSeverity.high
        elif any(term in error_str for term in ['timeout', 'timed out']):
            return ErrorCategory.timeout, ErrorSeverity.medium
        elif any(term in error_str for term in ['deadlock', 'serialization']):
            return ErrorCategory.transient, ErrorSeverity.medium
        elif any(term in error_str for term in ['permission', 'privilege']):
            return ErrorCategory.permission, ErrorSeverity.high
        elif any(term in error_str for term in ['syntax', 'undefined']):
            return ErrorCategory.syntax, ErrorSeverity.high

        return ErrorCategory.unknown, ErrorSeverity.medium

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """Determine if error is retryable."""
        category, _ = cls.classify_error(error)

        # Retryable categories
        retryable_categories = {
            ErrorCategory.connection,
            ErrorCategory.timeout,
            ErrorCategory.transient,
            ErrorCategory.resource
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
        from ..core.retry_manager import RetryableErrorType

        category, _ = DatabaseErrorClassifier.classify_error(error)

        # Map database categories to retry error types
        category_mapping = {
            ErrorCategory.connection: RetryableErrorType.NETWORK,
            ErrorCategory.timeout: RetryableErrorType.TIMEOUT,
            ErrorCategory.transient: RetryableErrorType.TRANSIENT,
            ErrorCategory.resource: RetryableErrorType.RESOURCE_EXHAUSTION,
        }

        return category_mapping.get(category, RetryableErrorType.TRANSIENT)

    return classify_for_retry

# Default configurations for database operations
def get_default_database_retry_config():
    """Get default retry configuration for database operations."""
    from ..core.retry_manager import RetryConfig, RetryStrategy, RetryableErrorType

    return RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.1,
        max_delay=10.0,
        backoff_multiplier=2.0,
        jitter=True,
        enable_circuit_breaker=True,
        failure_threshold=5,
        recovery_timeout_seconds=60.0,
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
async def execute_with_database_retry(operation, operation_name: str):
    """Execute database operation with unified retry logic."""
    from ..core.retry_manager import get_retry_manager

    retry_manager = get_retry_manager()
    config = get_default_database_retry_config()
    config.operation_name = operation_name

    return await retry_manager.retry_async(operation, config=config)

