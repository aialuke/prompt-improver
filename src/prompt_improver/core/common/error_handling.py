"""
Centralized error handling utilities to eliminate duplicate error handling patterns.

Consolidates common error handling patterns:
- Configuration loading errors
- Service initialization errors  
- Database connection errors
- Metrics collection errors
"""

from typing import Any, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, UTC
from .logging_utils import get_logger

logger = get_logger(__name__)

class ErrorCategory(str, Enum):
    """Categories of errors for consistent handling."""
    CONFIGURATION = "configuration"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"

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
    
    def __init__(self, message: str, category: ErrorCategory, component: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.category = category
        self.component = component
        self.cause = cause
        self.timestamp = datetime.now(UTC)

def safe_config_load(
    loader_func: Callable,
    fallback_value: Any = None,
    component_name: str = "unknown",
    required: bool = False
) -> tuple[Any, bool, Optional[str]]:
    """
    Safely load configuration with consistent error handling.
    
    Consolidates the pattern:
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Config load failed: {e}")
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
        return config, True, None
    except Exception as e:
        error_msg = f"Failed to load configuration for {component_name}: {e}"
        logger.warning(error_msg)
        
        if required and fallback_value is None:
            raise InitializationError(
                error_msg,
                ErrorCategory.CONFIGURATION,
                component_name,
                e
            )
        
        if fallback_value is not None:
            logger.info(f"Using fallback configuration for {component_name}")
            return fallback_value, False, error_msg
        
        return None, False, error_msg

def handle_initialization_error(
    error: Exception,
    context: ErrorContext,
    fallback_action: Optional[Callable] = None
) -> Optional[Any]:
    """
    Handle initialization errors with consistent logging and fallback.
    
    Args:
        error: The exception that occurred
        context: Error context information
        fallback_action: Optional fallback action to take
        
    Returns:
        Result of fallback action if available, None otherwise
        
    Raises:
        InitializationError: If error is critical and no fallback available
    """
    error_msg = f"Initialization failed for {context.component}.{context.operation}: {error}"
    
    # Log error with appropriate level
    if context.critical:
        logger.error(error_msg)
    else:
        logger.warning(error_msg)
    
    # Log stack trace for debugging
    logger.debug(f"Stack trace for {context.component} error:", exc_info=True)
    
    # Try fallback if available
    if fallback_action and context.fallback_available:
        try:
            logger.info(f"Attempting fallback for {context.component}")
            result = fallback_action()
            logger.info(f"Fallback successful for {context.component}")
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback failed for {context.component}: {fallback_error}")
    
    # Raise critical errors
    if context.critical:
        raise InitializationError(
            error_msg,
            context.category,
            context.component,
            error
        )
    
    return None

def safe_operation(
    operation: Callable,
    operation_name: str,
    component_name: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    fallback_value: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> tuple[Any, bool, Optional[str]]:
    """
    Execute an operation safely with consistent error handling.
    
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
        return result, True, None
    except Exception as e:
        error_msg = f"{operation_name} failed in {component_name}: {e}"
        
        if log_errors:
            logger.error(error_msg)
            logger.debug(f"Stack trace for {operation_name}:", exc_info=True)
        
        if reraise:
            raise
        
        return fallback_value, False, error_msg

def with_retry(
    operation: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: str = "operation",
    component_name: str = "unknown"
) -> Any:
    """
    Execute operation with retry logic and consistent error handling.
    
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
    """
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            result = operation()
            if attempt > 0:
                logger.info(f"{operation_name} succeeded after {attempt} retries")
            return result
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(
                    f"{operation_name} failed in {component_name} "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(
                    f"{operation_name} failed in {component_name} "
                    f"after {max_retries} retries: {e}"
                )
    
    raise last_exception

class ErrorHandler:
    """
    Class for managing error handling context and patterns.
    
    Provides a consistent interface for error handling across components.
    """
    
    def __init__(self, component_name: str, default_category: ErrorCategory = ErrorCategory.UNKNOWN):
        self.component_name = component_name
        self.default_category = default_category
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.last_errors: Dict[ErrorCategory, str] = {}
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        category: Optional[ErrorCategory] = None,
        critical: bool = False,
        fallback_value: Any = None
    ) -> Any:
        """
        Handle an error with consistent logging and tracking.
        
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
        
        # Track error counts
        self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1
        self.last_errors[error_category] = str(error)
        
        # Create error context
        context = ErrorContext(
            category=error_category,
            component=self.component_name,
            operation=operation,
            timestamp=datetime.now(UTC),
            critical=critical,
            fallback_available=fallback_value is not None
        )
        
        # Handle the error
        if fallback_value is not None:
            return handle_initialization_error(error, context, lambda: fallback_value)
        else:
            handle_initialization_error(error, context)
    
    def safe_execute(
        self,
        operation: Callable,
        operation_name: str,
        category: Optional[ErrorCategory] = None,
        fallback_value: Any = None,
        critical: bool = False
    ) -> tuple[Any, bool]:
        """
        Execute operation safely with error handling.
        
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
            return result, True
        except Exception as e:
            handled_result = self.handle_error(
                e, operation_name, category, critical, fallback_value
            )
            return handled_result, False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors handled by this handler."""
        return {
            "component": self.component_name,
            "error_counts": dict(self.error_counts),
            "last_errors": dict(self.last_errors),
            "total_errors": sum(self.error_counts.values())
        }