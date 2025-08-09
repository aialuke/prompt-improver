"""Centralized error handling utilities to eliminate duplicate error handling patterns.

Consolidates common error handling patterns:
- Configuration loading errors
- Service initialization errors
- Database connection errors
- Metrics collection errors
"""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional
from prompt_improver.core.common.logging_utils import get_logger
logger = get_logger(__name__)

class ErrorCategory(str, Enum):
    """Categories of errors for consistent handling."""
    CONFIGURATION = 'configuration'
    DATABASE = 'database'
    NETWORK = 'network'
    AUTHENTICATION = 'authentication'
    VALIDATION = 'validation'
    TIMEOUT = 'timeout'
    RESOURCE = 'resource'
    EXTERNAL_SERVICE = 'external_service'
    UNKNOWN = 'unknown'

@dataclass
class ErrorContext:
    """Context information for error handling."""
    category: ErrorCategory
    component: str
    operation: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    fallback_available: bool = False
    critical: bool = False

class InitializationError(Exception):
    """Custom exception for initialization failures."""

    def __init__(self, message: str, category: ErrorCategory, component: str, cause: Exception | None=None):
        super().__init__(message)
        self.category = category
        self.component = component
        self.cause = cause
        self.timestamp = datetime.now(UTC)

def safe_config_load(loader_func: Callable, fallback_value: Any=None, component_name: str='unknown', required: bool=False) -> tuple[Any, bool, str | None]:
    """Safely load configuration with consistent error handling.

    Consolidates the pattern:
    try:
        config = load_config()
    except Exception as e:
        logger.warning("Config load failed: %s", e)
        config = fallback

    Args:
        loader_func: Function to load configuration
        fallback_value: Value to use if loading fails
        component_name: Name of component for logging
        required: Whether this config is required (raise if fail and no fallback)

    Returns:
        Tuple of (config_value, success, error_message)

    Raises:
        InitializationError: If required=True and loading fails with no fallback
    """
    try:
        config = loader_func()
        return (config, True, None)
    except Exception as e:
        error_msg = f'Failed to load configuration for {component_name}: {e}'
        logger.warning(error_msg)
        if required and fallback_value is None:
            raise InitializationError(error_msg, ErrorCategory.CONFIGURATION, component_name, e)
        if fallback_value is not None:
            logger.info('Using fallback configuration for %s', component_name)
            return (fallback_value, False, error_msg)
        return (None, False, error_msg)

def handle_initialization_error(error: Exception, context: ErrorContext, fallback_action: Callable | None=None) -> Any | None:
    """Handle initialization errors with consistent logging and fallback.

    Args:
        error: The exception that occurred
        context: Error context information
        fallback_action: Optional fallback action to take

    Returns:
        Result of fallback action if available, None otherwise

    Raises:
        InitializationError: If error is critical and no fallback available
    """
    error_msg = f'Initialization failed for {context.component}.{context.operation}: {error}'
    if context.critical:
        logger.error(error_msg)
    else:
        logger.warning(error_msg)
    logger.debug('Stack trace for %s error:', context.component, exc_info=True)
    if fallback_action and context.fallback_available:
        try:
            logger.info('Attempting fallback for %s', context.component)
            result = fallback_action()
            logger.info('Fallback successful for %s', context.component)
            return result
        except Exception as fallback_error:
            logger.error('Fallback failed for {context.component}: %s', fallback_error)
    if context.critical:
        raise InitializationError(error_msg, context.category, context.component, error)
    return None

def safe_operation(operation: Callable, operation_name: str, component_name: str, category: ErrorCategory=ErrorCategory.UNKNOWN, fallback_value: Any=None, log_errors: bool=True, reraise: bool=False) -> tuple[Any, bool, str | None]:
    """Execute an operation safely with consistent error handling.

    Args:
        operation: Function to execute
        operation_name: Name of operation for logging
        component_name: Name of component for logging
        category: Category of error for classification
        fallback_value: Value to return on failure
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions

    Returns:
        Tuple of (result, success, error_message)
    """
    try:
        result = operation()
        return (result, True, None)
    except Exception as e:
        error_msg = f'{operation_name} failed in {component_name} [{category.value}]: {e}'
        if log_errors:
            logger.error(error_msg, extra={'error_category': category.value, 'component': component_name, 'operation': operation_name, 'exception_type': type(e).__name__})
            logger.debug('Stack trace for %s [%s]:', operation_name, category.value, exc_info=True)
        if reraise:
            raise
        return (fallback_value, False, error_msg)

def with_retry(operation: Callable, max_retries: int=3, delay: float=1.0, backoff_factor: float=2.0, operation_name: str='operation', component_name: str='unknown') -> Any:
    """Execute operation with retry logic and consistent error handling.

    Args:
        operation: Function to execute
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff_factor: Factor to multiply delay by each retry
        operation_name: Name of operation for logging
        component_name: Name of component for logging

    Returns:
        Result of successful operation

    Raises:
        Exception: Last exception if all retries fail
        ValueError: If max_retries is negative
    """
    import time
    if max_retries < 0:
        raise ValueError(f'max_retries must be non-negative, got {max_retries}')
    last_exception = None
    current_delay = delay
    for attempt in range(max_retries + 1):
        try:
            result = operation()
            if attempt > 0:
                logger.info('{operation_name} succeeded after %s retries', attempt)
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning('%s failed in %s (attempt %s/%s): %s', operation_name, component_name, attempt + 1, max_retries + 1, e)
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error('%s failed in %s after %s retries: %s', operation_name, component_name, max_retries, e)
    if last_exception is None:
        raise RuntimeError(f'Operation {operation_name} failed with no exception recorded')
    raise last_exception

class ErrorHandler:
    """Class for managing error handling context and patterns.

    Provides a consistent interface for error handling across components.
    """

    def __init__(self, component_name: str, default_category: ErrorCategory=ErrorCategory.UNKNOWN):
        self.component_name = component_name
        self.default_category = default_category
        self.error_counts: dict[ErrorCategory, int] = {}
        self.last_errors: dict[ErrorCategory, str] = {}

    def handle_error(self, error: Exception, operation: str, category: ErrorCategory | None=None, critical: bool=False, fallback_value: Any=None) -> Any:
        """Handle an error with consistent logging and tracking.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            category: Category of error (uses default if None)
            critical: Whether this is a critical error
            fallback_value: Value to return instead of raising

        Returns:
            fallback_value if provided, otherwise raises
        """
        error_category = category or self.default_category
        self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1
        self.last_errors[error_category] = str(error)
        context = ErrorContext(category=error_category, component=self.component_name, operation=operation, timestamp=datetime.now(UTC), critical=critical, fallback_available=fallback_value is not None)
        if fallback_value is not None:
            return handle_initialization_error(error, context, lambda: fallback_value)
        handle_initialization_error(error, context)

    def safe_execute(self, operation: Callable, operation_name: str, category: ErrorCategory | None=None, fallback_value: Any=None, critical: bool=False) -> tuple[Any, bool]:
        """Execute operation safely with error handling.

        Args:
            operation: Function to execute
            operation_name: Name of operation
            category: Error category
            fallback_value: Fallback value on error
            critical: Whether error is critical

        Returns:
            Tuple of (result, success)
        """
        try:
            result = operation()
            return (result, True)
        except Exception as e:
            handled_result = self.handle_error(e, operation_name, category, critical, fallback_value)
            return (handled_result, False)

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of errors handled by this handler."""
        return {'component': self.component_name, 'error_counts': dict(self.error_counts), 'last_errors': dict(self.last_errors), 'total_errors': sum(self.error_counts.values())}
