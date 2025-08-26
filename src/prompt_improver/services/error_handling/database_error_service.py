"""Database Error Service - Specialized Database Error Handling.

Provides comprehensive database error handling with:
- Circuit breaker patterns for database operations
- Transaction rollback management with session cleanup
- Database-specific retry logic with backoff strategies
- SQL error classification and sensitive data redaction
- Integration with database health monitoring

Security Features:
- SQL injection attempt detection in error messages
- Database schema information redaction
- Connection string sanitization
- Sensitive error message filtering

Performance Target: <2ms error classification, <5ms circuit breaker operations
Memory Target: <15MB for error tracking and circuit breaker state
"""

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import asyncpg
import sqlalchemy.exc as sa_exc
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.core.config.retry import RetryConfig, RetryStrategy
from prompt_improver.core.services.resilience.retry_service_facade import (
    get_retry_service as get_retry_manager,
)
from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState as CBState,
)
from prompt_improver.performance.monitoring.metrics_registry import (
    get_metrics_registry,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


class DatabaseErrorCategory(Enum):
    """Database-specific error categories for precise handling."""

    CONNECTION_FAILURE = "connection_failure"
    TRANSACTION_ROLLBACK = "transaction_rollback"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    DEADLOCK = "deadlock"
    SERIALIZATION_FAILURE = "serialization_failure"
    DISK_FULL = "disk_full"
    CONNECTION_POOL_EXHAUSTED = "connection_pool_exhausted"
    QUERY_CANCELLED = "query_cancelled"
    UNKNOWN_ERROR = "unknown_error"


class DatabaseErrorSeverity(Enum):
    """Error severity levels for database operations."""

    CRITICAL = "critical"  # System-wide database failure
    HIGH = "high"         # Service degradation expected
    MEDIUM = "medium"     # Recoverable with retry
    LOW = "low"          # Expected operational errors
    INFO = "info"        # Informational only


@dataclass
class DatabaseErrorContext:
    """Comprehensive error context for database operations."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = ""
    category: DatabaseErrorCategory = DatabaseErrorCategory.UNKNOWN_ERROR
    severity: DatabaseErrorSeverity = DatabaseErrorSeverity.MEDIUM
    original_exception: Exception | None = None
    sanitized_message: str = ""
    is_retryable: bool = False
    retry_after_seconds: float | None = None
    connection_info: dict[str, Any] = field(default_factory=dict)
    session_info: dict[str, Any] = field(default_factory=dict)
    sql_statement: str | None = None
    timestamp: datetime = field(default_factory=aware_utc_now)
    correlation_id: str | None = None
    circuit_breaker_triggered: bool = False


@dataclass
class DatabaseCircuitBreakerConfig:
    """Circuit breaker configuration for database operations."""

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    success_threshold: int = 2
    timeout_ms: float = 5000.0
    slow_call_threshold_ms: float = 2000.0
    enable_half_open_max_calls: int = 3


class DatabaseErrorService:
    """Specialized database error handling service.

    Provides comprehensive error handling for database operations including:
    - Circuit breaker protection
    - Intelligent error classification
    - Security-aware error sanitization
    - Transaction rollback management
    - Integration with retry mechanisms
    """

    # SQL injection patterns for security detection
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(?i)\b(union|select|insert|update|delete|drop|alter|create|exec|execute)\b.*\b(from|where|into)\b"),
        re.compile(r"(?i)['\"];?\s*(union|select|insert|update|delete|drop)", re.IGNORECASE),
        re.compile(r"(?i)\b(or|and)\b\s*['\"]?\s*\d+\s*['\"]?\s*=\s*['\"]?\s*\d+"),
        re.compile(r"(?i)(script|javascript|vbscript|onload|onerror)", re.IGNORECASE),
    ]

    # Sensitive information patterns for redaction
    SENSITIVE_PATTERNS = [
        (re.compile(r"password\s*[=:]\s*['\"][^'\"]*['\"]", re.IGNORECASE), "password=[REDACTED]"),
        (re.compile(r"(host|server)\s*[=:]\s*['\"][^'\"]*['\"]", re.IGNORECASE), r"\1=[HOST_REDACTED]"),
        (re.compile(r"(database|db)\s*[=:]\s*['\"][^'\"]*['\"]", re.IGNORECASE), r"\1=[DB_REDACTED]"),
        (re.compile(r"(user|username|uid)\s*[=:]\s*['\"][^'\"]*['\"]", re.IGNORECASE), r"\1=[USER_REDACTED]"),
        (re.compile(r"postgresql://[^@]*@", re.IGNORECASE), "postgresql://[CREDENTIALS_REDACTED]@"),
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
    ]

    # Error mapping for precise classification
    ERROR_MAPPINGS = {
        # AsyncPG errors
        asyncpg.ConnectionDoesNotExistError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.CRITICAL),
        asyncpg.ConnectionFailureError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        asyncpg.InterfaceError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        asyncpg.PostgresConnectionError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        asyncpg.InvalidAuthorizationSpecificationError: (DatabaseErrorCategory.PERMISSION_DENIED, DatabaseErrorSeverity.HIGH),
        asyncpg.InsufficientPrivilegeError: (DatabaseErrorCategory.PERMISSION_DENIED, DatabaseErrorSeverity.HIGH),
        asyncpg.SyntaxOrAccessError: (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM),
        asyncpg.UndefinedTableError: (DatabaseErrorCategory.SCHEMA_ERROR, DatabaseErrorSeverity.MEDIUM),
        asyncpg.UndefinedColumnError: (DatabaseErrorCategory.SCHEMA_ERROR, DatabaseErrorSeverity.MEDIUM),
        asyncpg.IntegrityConstraintViolationError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),
        asyncpg.UniqueViolationError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),
        asyncpg.ForeignKeyViolationError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),
        asyncpg.NotNullViolationError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),
        asyncpg.CheckViolationError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),

        # SQLAlchemy errors
        sa_exc.DisconnectionError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        sa_exc.TimeoutError: (DatabaseErrorCategory.TIMEOUT_ERROR, DatabaseErrorSeverity.MEDIUM),
        sa_exc.InvalidRequestError: (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM),
        sa_exc.OperationalError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        sa_exc.IntegrityError: (DatabaseErrorCategory.CONSTRAINT_VIOLATION, DatabaseErrorSeverity.LOW),
        sa_exc.DataError: (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM),
        sa_exc.InternalError: (DatabaseErrorCategory.UNKNOWN_ERROR, DatabaseErrorSeverity.HIGH),
        sa_exc.ProgrammingError: (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM),
        sa_exc.NotSupportedError: (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM),

        # Generic errors
        asyncio.TimeoutError: (DatabaseErrorCategory.TIMEOUT_ERROR, DatabaseErrorSeverity.MEDIUM),
        asyncio.CancelledError: (DatabaseErrorCategory.QUERY_CANCELLED, DatabaseErrorSeverity.LOW),
        ConnectionRefusedError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
        ConnectionResetError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.MEDIUM),
        BrokenPipeError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.MEDIUM),
        OSError: (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH),
    }

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize database error service.

        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self._metrics_registry = get_metrics_registry()
        self._retry_manager = get_retry_manager()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._circuit_breaker_configs: dict[str, DatabaseCircuitBreakerConfig] = {}
        self._error_count_cache: dict[str, int] = {}

        # Initialize default circuit breaker configurations
        self._default_cb_config = DatabaseCircuitBreakerConfig()

        logger.info(f"DatabaseErrorService initialized with correlation_id: {self.correlation_id}")

    async def handle_database_error(
        self,
        error: Exception,
        operation_name: str,
        session: AsyncSession | None = None,
        sql_statement: str | None = None,
        rollback_on_error: bool = True,
        **context_kwargs: Any
    ) -> DatabaseErrorContext:
        """Handle database error with comprehensive processing.

        Args:
            error: The database exception that occurred
            operation_name: Name of the database operation
            session: Optional database session for rollback
            sql_statement: Optional SQL statement that caused the error
            rollback_on_error: Whether to rollback transaction on error
            **context_kwargs: Additional context information

        Returns:
            DatabaseErrorContext with processed error information
        """
        start_time = time.time()

        # Classify the error
        category, severity = self._classify_error(error)

        # Create error context
        error_context = DatabaseErrorContext(
            operation_name=operation_name,
            category=category,
            severity=severity,
            original_exception=error,
            sanitized_message=self._sanitize_error_message(str(error)),
            is_retryable=self._is_retryable_error(category, severity),
            sql_statement=self._sanitize_sql_statement(sql_statement),
            correlation_id=self.correlation_id,
            **context_kwargs
        )

        # Handle transaction rollback
        if rollback_on_error and session:
            await self._handle_transaction_rollback(session, error_context)

        # Check for security threats
        await self._check_security_threats(error_context)

        # Update circuit breaker
        circuit_breaker = await self._get_or_create_circuit_breaker(operation_name)
        if circuit_breaker:
            error_context.circuit_breaker_triggered = await self._update_circuit_breaker(
                circuit_breaker, False, error_context
            )

        # Record metrics
        await self._record_error_metrics(error_context)

        # Log the error
        await self._log_error(error_context)

        # Calculate processing time
        processing_time = time.time() - start_time
        histogram = self._metrics_registry.get_or_create_histogram(
            "database_error_processing_duration_seconds",
            "Database error processing duration",
            ["operation", "category"]
        )
        if hasattr(histogram, 'observe'):
            histogram.observe(processing_time, {"operation": operation_name, "category": category.value})

        return error_context

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute database operation with circuit breaker protection.

        Args:
            operation: Database operation to execute
            operation_name: Name of the operation for monitoring
            *args, **kwargs: Operation arguments

        Returns:
            Operation result

        Raises:
            Exception: If operation fails or circuit breaker is open
        """
        circuit_breaker = await self._get_or_create_circuit_breaker(operation_name)

        if not circuit_breaker:
            # Fallback to direct execution if circuit breaker unavailable
            return await operation(*args, **kwargs)

        # Check circuit breaker state
        if circuit_breaker.state == CBState.OPEN:
            error_msg = f"Circuit breaker OPEN for database operation: {operation_name}"
            logger.warning(error_msg)
            counter = self._metrics_registry.get_or_create_counter(
                "circuit_breaker_calls_total",
                "Total circuit breaker calls",
                ["operation", "state", "result"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "state": "open", "result": "rejected"})
            raise RuntimeError(error_msg)

        start_time = time.time()
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            # Record success
            execution_time = time.time() - start_time
            await self._update_circuit_breaker(circuit_breaker, True, None, execution_time)

            counter = self._metrics_registry.get_or_create_counter(
                "circuit_breaker_calls_total",
                "Total circuit breaker calls",
                ["operation", "state", "result"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "state": circuit_breaker.state.value, "result": "success"})

            return result

        except Exception as error:
            # Handle failure
            execution_time = time.time() - start_time
            error_context = await self.handle_database_error(
                error,
                operation_name,
                execution_time=execution_time
            )

            await self._update_circuit_breaker(circuit_breaker, False, error_context, execution_time)

            counter = self._metrics_registry.get_or_create_counter(
                "circuit_breaker_calls_total",
                "Total circuit breaker calls",
                ["operation", "state", "result"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "state": circuit_breaker.state.value, "result": "failure"})

            raise

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        session: AsyncSession | None = None,
        custom_retry_config: RetryConfig | None = None,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Execute database operation with intelligent retry logic.

        Args:
            operation: Database operation to execute
            operation_name: Name of the operation
            session: Optional database session
            custom_retry_config: Optional custom retry configuration
            *args, **kwargs: Operation arguments

        Returns:
            Operation result
        """
        retry_config = custom_retry_config or self._get_default_retry_config(operation_name)
        logger.debug(f"Using retry config with {retry_config.max_attempts} max attempts for {operation_name}")

        async def wrapped_operation():
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except Exception as error:
                # Handle and classify error
                await self.handle_database_error(
                    error,
                    operation_name,
                    session,
                    rollback_on_error=True
                )

                # Re-raise for retry manager to handle
                raise

        result = await self._retry_manager.execute_with_retry(
            operation=wrapped_operation,
            domain="database",
            operation_type=operation_name
        )
        return result.result

    @asynccontextmanager
    async def database_transaction_context(
        self,
        session: AsyncSession,
        operation_name: str,
        auto_rollback: bool = True
    ) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database transactions with error handling.

        Args:
            session: Database session
            operation_name: Name of the operation
            auto_rollback: Whether to auto-rollback on errors

        Yields:
            Database session
        """
        start_time = time.time()
        transaction_id = str(uuid.uuid4())[:8]

        try:
            # Begin transaction tracking
            logger.debug(f"Starting transaction {transaction_id} for {operation_name}")

            counter = self._metrics_registry.get_or_create_counter(
                "database_transactions_total",
                "Total database transactions",
                ["operation", "type"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "type": "started"})

            yield session

            # Successful completion
            await session.commit()
            execution_time = time.time() - start_time

            logger.debug(f"Transaction {transaction_id} committed successfully in {execution_time:.3f}s")

            counter = self._metrics_registry.get_or_create_counter(
                "database_transactions_total",
                "Total database transactions",
                ["operation", "type"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "type": "committed"})

            histogram = self._metrics_registry.get_or_create_histogram(
                "database_transaction_duration_seconds",
                "Database transaction duration",
                ["operation", "result"]
            )
            if hasattr(histogram, 'observe'):
                histogram.observe(execution_time, {"operation": operation_name, "result": "success"})

        except Exception as error:
            execution_time = time.time() - start_time

            # Handle error and rollback
            error_context = await self.handle_database_error(
                error,
                operation_name,
                session,
                rollback_on_error=auto_rollback
            )

            logger.exception(f"Transaction {transaction_id} failed after {execution_time:.3f}s: {error_context.sanitized_message}")

            counter = self._metrics_registry.get_or_create_counter(
                "database_transactions_total",
                "Total database transactions",
                ["operation", "type"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {"operation": operation_name, "type": "rolled_back"})

            histogram = self._metrics_registry.get_or_create_histogram(
                "database_transaction_duration_seconds",
                "Database transaction duration",
                ["operation", "result"]
            )
            if hasattr(histogram, 'observe'):
                histogram.observe(execution_time, {"operation": operation_name, "result": "failure"})

            raise

    def _classify_error(self, error: Exception) -> tuple[DatabaseErrorCategory, DatabaseErrorSeverity]:
        """Classify database error into category and severity.

        Args:
            error: Database exception

        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error)

        # Direct type mapping
        if error_type in self.ERROR_MAPPINGS:
            return self.ERROR_MAPPINGS[error_type]

        # Check inheritance hierarchy
        for mapped_type, (category, severity) in self.ERROR_MAPPINGS.items():
            if isinstance(error, mapped_type):
                return (category, severity)

        # Pattern-based classification for unknown errors
        error_msg = str(error).lower()

        if any(pattern in error_msg for pattern in ["connection", "connect", "network", "host"]):
            return (DatabaseErrorCategory.CONNECTION_FAILURE, DatabaseErrorSeverity.HIGH)

        if any(pattern in error_msg for pattern in ["timeout", "timed out", "deadline"]):
            return (DatabaseErrorCategory.TIMEOUT_ERROR, DatabaseErrorSeverity.MEDIUM)

        if any(pattern in error_msg for pattern in ["deadlock", "lock"]):
            return (DatabaseErrorCategory.DEADLOCK, DatabaseErrorSeverity.MEDIUM)

        if any(pattern in error_msg for pattern in ["serialization", "concurrent"]):
            return (DatabaseErrorCategory.SERIALIZATION_FAILURE, DatabaseErrorSeverity.MEDIUM)

        if any(pattern in error_msg for pattern in ["disk", "space", "full"]):
            return (DatabaseErrorCategory.DISK_FULL, DatabaseErrorSeverity.CRITICAL)

        if any(pattern in error_msg for pattern in ["pool", "connection limit", "max connections"]):
            return (DatabaseErrorCategory.CONNECTION_POOL_EXHAUSTED, DatabaseErrorSeverity.HIGH)

        if any(pattern in error_msg for pattern in ["permission", "privilege", "access denied"]):
            return (DatabaseErrorCategory.PERMISSION_DENIED, DatabaseErrorSeverity.HIGH)

        if any(pattern in error_msg for pattern in ["syntax", "invalid", "undefined", "does not exist"]):
            return (DatabaseErrorCategory.SYNTAX_ERROR, DatabaseErrorSeverity.MEDIUM)

        return (DatabaseErrorCategory.UNKNOWN_ERROR, DatabaseErrorSeverity.MEDIUM)

    def _is_retryable_error(self, category: DatabaseErrorCategory, severity: DatabaseErrorSeverity) -> bool:
        """Determine if error is retryable based on category and severity.

        Args:
            category: Error category
            severity: Error severity

        Returns:
            True if error should be retried
        """
        # Non-retryable categories
        non_retryable_categories = {
            DatabaseErrorCategory.PERMISSION_DENIED,
            DatabaseErrorCategory.SYNTAX_ERROR,
            DatabaseErrorCategory.SCHEMA_ERROR,
            DatabaseErrorCategory.CONSTRAINT_VIOLATION,
        }

        if category in non_retryable_categories:
            return False

        # Retryable categories
        retryable_categories = {
            DatabaseErrorCategory.CONNECTION_FAILURE,
            DatabaseErrorCategory.TIMEOUT_ERROR,
            DatabaseErrorCategory.DEADLOCK,
            DatabaseErrorCategory.SERIALIZATION_FAILURE,
            DatabaseErrorCategory.CONNECTION_POOL_EXHAUSTED,
        }

        return category in retryable_categories

    def _sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error message to remove sensitive information.

        Args:
            error_message: Raw error message

        Returns:
            Sanitized error message
        """
        sanitized = error_message

        # Apply all sensitive patterns
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def _sanitize_sql_statement(self, sql: str | None) -> str | None:
        """Sanitize SQL statement to remove sensitive data.

        Args:
            sql: Raw SQL statement

        Returns:
            Sanitized SQL statement or None
        """
        if not sql:
            return None

        sanitized = sql

        # Remove potential sensitive values in WHERE clauses
        sanitized = re.sub(
            r"(?i)(where\s+\w+\s*[=<>!]+\s*)['\"][^'\"]*['\"]",
            r"\1'[REDACTED]'",
            sanitized
        )

        # Remove values in INSERT statements
        sanitized = re.sub(
            r"(?i)(values\s*\()[^)]*(\))",
            r"\1[VALUES_REDACTED]\2",
            sanitized
        )

        # Limit length to prevent log flooding
        if len(sanitized) > 1000:
            sanitized = sanitized[:997] + "..."

        return sanitized

    async def _handle_transaction_rollback(
        self,
        session: AsyncSession,
        error_context: DatabaseErrorContext
    ) -> None:
        """Handle transaction rollback with proper cleanup.

        Args:
            session: Database session to rollback
            error_context: Error context for logging
        """
        try:
            await session.rollback()

            error_context.session_info["rollback_successful"] = True

            logger.info(
                f"Transaction rolled back successfully for operation: {error_context.operation_name}",
                extra={"correlation_id": self.correlation_id}
            )

            counter = self._metrics_registry.get_or_create_counter(
                "database_rollbacks_total",
                "Total database rollbacks",
                ["operation", "category", "result"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {
                    "operation": error_context.operation_name,
                    "category": error_context.category.value,
                    "result": "success"
                })

        except Exception as rollback_error:
            error_context.session_info["rollback_successful"] = False
            error_context.session_info["rollback_error"] = str(rollback_error)

            logger.exception(
                f"Failed to rollback transaction for operation {error_context.operation_name}: {rollback_error}",
                extra={"correlation_id": self.correlation_id}
            )

            counter = self._metrics_registry.get_or_create_counter(
                "database_rollbacks_total",
                "Total database rollbacks",
                ["operation", "category", "result"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {
                    "operation": error_context.operation_name,
                    "category": error_context.category.value,
                    "result": "failure"
                })

    async def _check_security_threats(self, error_context: DatabaseErrorContext) -> None:
        """Check for potential security threats in database errors.

        Args:
            error_context: Error context to analyze
        """
        threat_detected = False
        threat_types = []

        # Check error message for SQL injection patterns
        error_msg = str(error_context.original_exception)
        for pattern in self.SQL_INJECTION_PATTERNS:
            if pattern.search(error_msg):
                threat_detected = True
                threat_types.append("sql_injection_pattern")
                break

        # Check SQL statement for suspicious patterns
        if error_context.sql_statement:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if pattern.search(error_context.sql_statement):
                    threat_detected = True
                    threat_types.append("suspicious_sql")
                    break

        if threat_detected:
            logger.warning(
                f"SECURITY ALERT: Potential threats detected in database error for operation {error_context.operation_name}: {threat_types}",
                extra={
                    "correlation_id": self.correlation_id,
                    "threat_types": threat_types,
                    "error_id": error_context.error_id
                }
            )

            counter = self._metrics_registry.get_or_create_counter(
                "database_security_threats_total",
                "Total database security threats",
                ["operation", "threat_type"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {
                    "operation": error_context.operation_name,
                    "threat_type": ",".join(threat_types)
                })

    async def _get_or_create_circuit_breaker(self, operation_name: str) -> CircuitBreaker | None:
        """Get or create circuit breaker for operation.

        Args:
            operation_name: Name of the database operation

        Returns:
            Circuit breaker instance or None if creation fails
        """
        if operation_name not in self._circuit_breakers:
            try:
                config = self._circuit_breaker_configs.get(operation_name, self._default_cb_config)

                cb_config = CircuitBreakerConfig(
                    failure_threshold=config.failure_threshold,
                    recovery_timeout_seconds=config.recovery_timeout_seconds,
                    success_threshold=config.success_threshold,
                    timeout_ms=config.timeout_ms
                )

                self._circuit_breakers[operation_name] = CircuitBreaker(
                    service_name=f"db_{operation_name}",
                    config=cb_config
                )

                logger.info(f"Created circuit breaker for database operation: {operation_name}")

            except Exception as e:
                logger.exception(f"Failed to create circuit breaker for {operation_name}: {e}")
                return None

        return self._circuit_breakers.get(operation_name)

    async def _update_circuit_breaker(
        self,
        circuit_breaker: CircuitBreaker,
        success: bool,
        error_context: DatabaseErrorContext | None,
        execution_time: float | None = None
    ) -> bool:
        """Update circuit breaker state based on operation result.

        Args:
            circuit_breaker: Circuit breaker to update
            success: Whether operation was successful
            error_context: Error context if operation failed
            execution_time: Operation execution time in seconds

        Returns:
            True if circuit breaker was triggered (state changed)
        """
        previous_state = circuit_breaker.state

        try:
            if success:
                circuit_breaker.record_success(execution_time or 0.0)
            else:
                circuit_breaker.record_failure()

            current_state = circuit_breaker.state
            state_changed = previous_state != current_state

            if state_changed:
                logger.info(
                    f"Circuit breaker state changed for {circuit_breaker.service_name}: "
                    f"{previous_state.value} -> {current_state.value}"
                )

                counter = self._metrics_registry.get_or_create_counter(
                    "circuit_breaker_state_transitions",
                    "Circuit breaker state transitions",
                    ["name", "from_state", "to_state"]
                )
                if hasattr(counter, 'inc'):
                    counter.inc(1, {
                        "name": circuit_breaker.service_name,
                        "from_state": previous_state.value,
                        "to_state": current_state.value
                    })

            return state_changed

        except Exception as e:
            logger.exception(f"Failed to update circuit breaker {circuit_breaker.service_name}: {e}")
            return False

    def _get_default_retry_config(self, operation_name: str) -> RetryConfig:
        """Get default retry configuration for database operations.

        Args:
            operation_name: Name of the database operation

        Returns:
            Retry configuration
        """
        return RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True
        )

    async def _record_error_metrics(self, error_context: DatabaseErrorContext) -> None:
        """Record error metrics for monitoring.

        Args:
            error_context: Error context with metrics data
        """
        try:
            counter = self._metrics_registry.get_or_create_counter(
                "database_errors_total",
                "Total database errors",
                ["operation", "category", "severity", "retryable"]
            )
            if hasattr(counter, 'inc'):
                counter.inc(1, {
                    "operation": error_context.operation_name,
                    "category": error_context.category.value,
                    "severity": error_context.severity.value,
                    "retryable": str(error_context.is_retryable).lower()
                })

            # Track error counts for rate limiting
            operation_key = f"{error_context.operation_name}:{error_context.category.value}"
            self._error_count_cache[operation_key] = self._error_count_cache.get(operation_key, 0) + 1

        except Exception as e:
            logger.exception(f"Failed to record error metrics: {e}")

    async def _log_error(self, error_context: DatabaseErrorContext) -> None:
        """Log error with appropriate level and context.

        Args:
            error_context: Error context to log
        """
        log_data = {
            "correlation_id": self.correlation_id,
            "error_id": error_context.error_id,
            "operation": error_context.operation_name,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "retryable": error_context.is_retryable,
            "circuit_breaker_triggered": error_context.circuit_breaker_triggered,
        }

        log_message = (
            f"Database error in {error_context.operation_name}: "
            f"{error_context.sanitized_message}"
        )

        if error_context.severity == DatabaseErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_data)
        elif error_context.severity == DatabaseErrorSeverity.HIGH:
            logger.error(log_message, extra=log_data)
        elif error_context.severity == DatabaseErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics for monitoring dashboard.

        Returns:
            Dictionary with error statistics
        """
        circuit_breaker_states = {}
        for name, cb in self._circuit_breakers.items():
            circuit_breaker_states[name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": getattr(cb, 'success_count', 0),
                "last_failure_time": getattr(cb, 'last_failure_time', None),
            }

        return {
            "correlation_id": self.correlation_id,
            "active_circuit_breakers": len(self._circuit_breakers),
            "circuit_breaker_states": circuit_breaker_states,
            "error_counts": dict(self._error_count_cache),
            "service_health": "operational",
        }
