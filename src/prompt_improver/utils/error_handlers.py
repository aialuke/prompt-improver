"""Error handling decorators for consistent exception management.

This module provides standardized error handling decorators that implement the
sophisticated error categorization and rollback patterns identified in
ab_testing.py as the gold standard for error handling in the codebase.

Features:
- Categorized exception handling (Database I/O, Validation, User interruption, etc.)
- Automatic rollback logic for database operations
- Consistent return value structures
- Context-aware logging with operation details
- Retry logic for transient errors
- Graceful degradation patterns

Usage:
    @handle_database_errors(rollback_session=True)
    async def database_operation(db_session: AsyncSession):
        # Database operations here
        pass

    @handle_validation_errors(return_format="dict")
    def validate_data(data: dict):
        # Validation logic here
        pass
"""

import logging
import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional, Union, Literal
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def handle_database_errors(
    rollback_session: bool = True,
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: Optional[str] = None,
    retry_count: int = 0,
    retry_delay: float = 1.0,
):
    """Decorator for database operations with comprehensive error handling.
    
    Implements sophisticated error categorization patterns from ab_testing.py:
    - Database I/O errors (OSError, IOError)
    - Data validation errors (ValueError, TypeError)
    - Data structure errors (AttributeError, KeyError)
    - User interruption (KeyboardInterrupt)
    - Unexpected errors (Exception)
    
    Args:
        rollback_session: Whether to automatically rollback on database errors
        return_format: How to handle errors ("dict", "raise", "none")
        operation_name: Human-readable operation name for logging
        retry_count: Number of retry attempts for transient errors
        retry_delay: Delay between retry attempts in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            db_session = None
            
            # Find database session in arguments
            if rollback_session:
                for arg in args:
                    if isinstance(arg, AsyncSession):
                        db_session = arg
                        break
                if not db_session and 'db_session' in kwargs:
                    db_session = kwargs['db_session']
            
            attempts = 0
            max_attempts = retry_count + 1
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                    
                except (OSError, IOError) as e:
                    logger.error(f"Database I/O error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    
                    # Retry for transient I/O errors
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "database_io"}
                    elif return_format == "raise":
                        raise
                    return []
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Data validation error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "validation"}
                    elif return_format == "raise":
                        raise
                    return []
                    
                except (AttributeError, KeyError) as e:
                    logger.error(f"Data structure error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "data_structure"}
                    elif return_format == "raise":
                        raise
                    return []
                    
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    if db_session:
                        await db_session.rollback()
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected error in {operation}: {e}")
                    logging.exception(f"Unexpected error in {operation}")
                    if db_session:
                        await db_session.rollback()
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "unexpected"}
                    elif return_format == "raise":
                        raise
                    return []
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            attempts = 0
            max_attempts = retry_count + 1
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                    
                except (OSError, IOError) as e:
                    logger.error(f"I/O error in {operation}: {e}")
                    
                    # Retry for transient I/O errors
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})")
                        time.sleep(retry_delay)
                        continue
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "io"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Data validation error in {operation}: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "validation"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except (AttributeError, KeyError) as e:
                    logger.error(f"Data structure error in {operation}: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "data_structure"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected error in {operation}: {e}")
                    logging.exception(f"Unexpected error in {operation}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "unexpected"}
                    elif return_format == "raise":
                        raise
                    return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def handle_filesystem_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: Optional[str] = None,
    retry_count: int = 0,
    retry_delay: float = 0.5,
):
    """Decorator for filesystem operations with error handling.
    
    Handles common filesystem errors:
    - File not found (FileNotFoundError)
    - Permission denied (PermissionError)
    - I/O errors (OSError, IOError)
    - Path validation errors (ValueError)
    
    Args:
        return_format: How to handle errors ("dict", "raise", "none")
        operation_name: Human-readable operation name for logging
        retry_count: Number of retry attempts for transient errors
        retry_delay: Delay between retry attempts in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            attempts = 0
            max_attempts = retry_count + 1
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                    
                except FileNotFoundError as e:
                    logger.error(f"File not found in {operation}: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "file_not_found"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except PermissionError as e:
                    logger.error(f"Permission denied in {operation}: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "permission_denied"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except (OSError, IOError) as e:
                    logger.error(f"I/O error in {operation}: {e}")
                    
                    # Retry for transient I/O errors
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})")
                        time.sleep(retry_delay)
                        continue
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "io"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except ValueError as e:
                    logger.error(f"Path validation error in {operation}: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "path_validation"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected filesystem error in {operation}: {e}")
                    logging.exception(f"Unexpected error in {operation}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "unexpected"}
                    elif return_format == "raise":
                        raise
                    return None
                    
        return wrapper
    return decorator


def handle_validation_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: Optional[str] = None,
    log_validation_details: bool = True,
):
    """Decorator for data validation operations with error handling.
    
    Handles validation-specific errors:
    - Value errors (ValueError)
    - Type errors (TypeError)
    - Assertion errors (AssertionError)
    - Key errors (KeyError)
    - Attribute errors (AttributeError)
    
    Args:
        return_format: How to handle errors ("dict", "raise", "none")
        operation_name: Human-readable operation name for logging
        log_validation_details: Whether to log detailed validation information
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                if log_validation_details and result is not None:
                    logger.debug(f"Validation successful in {operation}")
                    
                return result
                
            except ValueError as e:
                logger.error(f"Value validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "value_validation"}
                elif return_format == "raise":
                    raise
                return None
                
            except TypeError as e:
                logger.error(f"Type validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "type_validation"}
                elif return_format == "raise":
                    raise
                return None
                
            except AssertionError as e:
                logger.error(f"Assertion validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "assertion_validation"}
                elif return_format == "raise":
                    raise
                return None
                
            except (KeyError, AttributeError) as e:
                logger.error(f"Data structure validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "structure_validation"}
                elif return_format == "raise":
                    raise
                return None
                
            except KeyboardInterrupt:
                logger.warning(f"{operation} cancelled by user")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected validation error in {operation}: {e}")
                logging.exception(f"Unexpected error in {operation}")
                
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "unexpected"}
                elif return_format == "raise":
                    raise
                return None
                
        return wrapper
    return decorator


def handle_network_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: Optional[str] = None,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
):
    """Decorator for network operations with retry logic.
    
    Handles network-specific errors with exponential backoff:
    - Connection errors
    - Timeout errors
    - HTTP errors
    - DNS resolution errors
    
    Args:
        return_format: How to handle errors ("dict", "raise", "none")
        operation_name: Human-readable operation name for logging
        retry_count: Number of retry attempts
        retry_delay: Initial delay between retry attempts in seconds
        backoff_multiplier: Multiplier for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            
            for attempt in range(retry_count + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except (ConnectionError, TimeoutError) as e:
                    if attempt < retry_count:
                        delay = retry_delay * (backoff_multiplier ** attempt)
                        logger.warning(
                            f"Network error in {operation} (attempt {attempt + 1}/{retry_count + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    
                    logger.error(f"Network error in {operation} after {retry_count + 1} attempts: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "network"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected network error in {operation}: {e}")
                    logging.exception(f"Unexpected error in {operation}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "unexpected"}
                    elif return_format == "raise":
                        raise
                    return None
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                    
                except (ConnectionError, TimeoutError) as e:
                    if attempt < retry_count:
                        delay = retry_delay * (backoff_multiplier ** attempt)
                        logger.warning(
                            f"Network error in {operation} (attempt {attempt + 1}/{retry_count + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    
                    logger.error(f"Network error in {operation} after {retry_count + 1} attempts: {e}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "network"}
                    elif return_format == "raise":
                        raise
                    return None
                    
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected network error in {operation}: {e}")
                    logging.exception(f"Unexpected error in {operation}")
                    
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "unexpected"}
                    elif return_format == "raise":
                        raise
                    return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Convenience function for common error handling patterns
def handle_common_errors(
    rollback_session: bool = False,
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: Optional[str] = None,
    retry_count: int = 0,
    retry_delay: float = 1.0,
):
    """Convenience decorator that combines database, filesystem, and validation error handling.
    
    This is useful for operations that might encounter any combination of these error types.
    """
    def decorator(func: Callable) -> Callable:
        # Apply multiple decorators in sequence
        decorated_func = handle_validation_errors(
            return_format=return_format,
            operation_name=operation_name
        )(func)
        
        decorated_func = handle_filesystem_errors(
            return_format=return_format,
            operation_name=operation_name,
            retry_count=retry_count,
            retry_delay=retry_delay
        )(decorated_func)
        
        decorated_func = handle_database_errors(
            rollback_session=rollback_session,
            return_format=return_format,
            operation_name=operation_name,
            retry_count=retry_count,
            retry_delay=retry_delay
        )(decorated_func)
        
        return decorated_func
    
    return decorator
