"""Comprehensive error handling decorators and FastAPI integration.

This consolidated module provides standardized error handling decorators that implement
sophisticated error categorization and rollback patterns, plus FastAPI-specific exception
handlers and correlation tracking for web applications.

Core Error Handling Decorators:
- handle_database_errors(): Database operations with rollback support
- handle_filesystem_errors(): File operations with retry logic  
- handle_validation_errors(): Data validation with detailed logging
- handle_network_errors(): Network operations with exponential backoff
- handle_common_errors(): Combined error handling for complex operations

Advanced Features:
- AsyncErrorBoundary: Context manager for async error boundaries
- PIIRedactionFilter: Automatic PII redaction in logs
- AsyncContextLogger: Structured logging with correlation tracking
- Retry logic with exponential backoff and jitter

FastAPI Integration (consolidated from common.error_handling):
- Authentication/authorization exception handlers
- Validation and rate limit error handlers  
- Correlation middleware for request tracking
- Standardized JSON error responses

Repository/Service Layer Integration:
- handle_repository_errors(): Repository layer decorator
- handle_service_errors(): Service layer decorator
- Context-aware error propagation with correlation IDs

Usage:
    @handle_database_errors(rollback_session=True)
    async def database_operation(db_session: AsyncSession):
        # Database operations here
        pass

    @handle_validation_errors(return_format="dict")
    def validate_data(data: dict):
        # Validation logic here
        pass

This module consolidates error handling patterns from both generic application
layers and FastAPI-specific web framework requirements for comprehensive coverage.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Generic type parameters for decorators and wrappers
P = ParamSpec("P")
T = TypeVar("T")


class PIIRedactionFilter(logging.Filter):
    """Filter to redact PII from log records."""

    def __init__(self):
        super().__init__()
        self.patterns = [
            (
                re.compile("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"),
                "[EMAIL_REDACTED]",
            ),
            (
                re.compile("\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b"),
                "[PHONE_REDACTED]",
            ),
            (
                re.compile("\\b\\d{4}[-.\\s]?\\d{4}[-.\\s]?\\d{4}[-.\\s]?\\d{4}\\b"),
                "[CARD_REDACTED]",
            ),
            (re.compile("\\b\\d{3}[-.\\s]?\\d{2}[-.\\s]?\\d{4}\\b"), "[SSN_REDACTED]"),
            (
                re.compile("\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b"),
                "[IP_REDACTED]",
            ),
            (re.compile("\\b[A-Za-z0-9]{32,}\\b"), "[TOKEN_REDACTED]"),
            (
                re.compile(
                    "(password|pwd|pass|secret|key|token)\\s*[=:]\\s*\\S+",
                    re.IGNORECASE,
                ),
                lambda m: f"{m.group(1)}=[REDACTED]",
            ),
        ]
        self.pii_fields = {
            "password",
            "pwd",
            "pass",
            "secret",
            "key",
            "token",
            "api_key",
            "email",
            "phone",
            "ssn",
            "credit_card",
            "address",
            "full_name",
            "first_name",
            "last_name",
            "birth_date",
            "dob",
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact PII from log records."""
        if hasattr(record, "msg") and record.msg:
            record.msg = self._redact_message(record.msg)
        if hasattr(record, "args") and record.args:
            record.args = tuple(self._redact_message(str(arg)) for arg in record.args)
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key.lower() in self.pii_fields:
                    setattr(record, key, "[REDACTED]")
                elif isinstance(value, (str, dict)):
                    setattr(record, key, self._redact_message(value))
        return True

    def _redact_message(self, message: Any) -> Any:
        """Redact PII from a message."""
        if isinstance(message, dict):
            return self._redact_dict(message)
        if isinstance(message, str):
            return self._redact_string(message)
        return message

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact PII from dictionary data."""
        if not isinstance(data, dict):
            return data
        redacted = {}
        for key, value in data.items():
            if key.lower() in self.pii_fields:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, str):
                redacted[key] = self._redact_string(value)
            else:
                redacted[key] = value
        return redacted

    def _redact_string(self, text: str) -> str:
        """Redact PII from string text."""
        if not isinstance(text, str):
            return text
        for pattern, replacement in self.patterns:
            if callable(replacement):
                text = pattern.sub(replacement, text)
            else:
                text = pattern.sub(replacement, text)
        return text


class AsyncContextLogger:
    """Async-safe structured logger with correlation ID support."""

    def __init__(self, base_logger: logging.Logger):
        self.logger = base_logger
        self._context_vars = {}
        self._ensure_json_serializable = True

    def set_context(self, **kwargs):
        """Set context variables for this logger instance."""
        self._context_vars.update(kwargs)

    def _add_correlation_id(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Add correlation ID for request tracing."""
        if "correlation_id" not in extra:
            extra["correlation_id"] = str(uuid.uuid4())[:8]
        return extra

    def _ensure_serializable(self, data: Any) -> Any:
        """Ensure data is JSON serializable."""
        if not self._ensure_json_serializable:
            return data
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)

    def _prepare_extra(self, **kwargs) -> dict[str, Any]:
        """Prepare extra fields for logging."""
        extra = {**self._context_vars, **kwargs}
        extra = self._add_correlation_id(extra)
        if self._ensure_json_serializable:
            extra = {k: self._ensure_serializable(v) for k, v in extra.items()}
        return extra

    def info(self, msg: Any, **kwargs):
        extra = self._prepare_extra(**kwargs)
        self.logger.info(msg, extra=extra)

    def error(self, msg: Any, **kwargs):
        extra = self._prepare_extra(**kwargs)
        self.logger.error(msg, extra=extra)

    def exception(self, msg: Any, **kwargs):
        extra = self._prepare_extra(**kwargs)
        self.logger.exception(msg, extra=extra)

    def warning(self, msg: Any, **kwargs):
        extra = self._prepare_extra(**kwargs)
        self.logger.warning(msg, extra=extra)

    def debug(self, msg: Any, **kwargs):
        extra = self._prepare_extra(**kwargs)
        self.logger.debug(msg, extra=extra)


async_logger = AsyncContextLogger(logger)


class AsyncErrorBoundary:
    """Async context manager for wrapping background coroutines with centralized error handling."""

    def __init__(
        self,
        operation_name: str,
        logger: AsyncContextLogger | None = None,
        reraise: bool = True,
        fallback_result: Any = None,
        timeout: float | None = None,
    ):
        self.operation_name = operation_name
        self.logger = logger or async_logger
        self.reraise = reraise
        self.fallback_result = fallback_result
        self.timeout = timeout
        self.correlation_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.exception_info = None

    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.set_context(
            operation=self.operation_name, correlation_id=self.correlation_id
        )
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        if exc_type is None:
            self.logger.info(f"Operation completed successfully: {self.operation_name}")
            return False
        self.exception_info = {
            "type": exc_type.__name__,
            "message": str(exc_val),
            "operation": self.operation_name,
            "duration_seconds": duration,
            "correlation_id": self.correlation_id,
        }
        if exc_type == asyncio.TimeoutError:
            self.logger.error(f"Operation timed out: {self.operation_name}")
        elif exc_type == asyncio.CancelledError:
            self.logger.warning(f"Operation cancelled: {self.operation_name}")
        else:
            self.logger.exception(f"Operation failed: {self.operation_name}")
        return not self.reraise

    def get_exception_info(self) -> dict[str, Any] | None:
        """Get information about any exception that occurred."""
        return self.exception_info


@asynccontextmanager
async def async_error_boundary(
    operation_name: str,
    logger: AsyncContextLogger | None = None,
    reraise: bool = True,
    fallback_result: Any = None,
    timeout: float | None = None,
):
    """Async context manager factory for wrapping background coroutines.

    Args:
        operation_name: Name of the operation for logging
        logger: Optional logger instance (defaults to global async_logger)
        reraise: Whether to reraise exceptions (default: True)
        fallback_result: Result to return if exception occurs and reraise=False
        timeout: Optional timeout in seconds

    Example:
        async with async_error_boundary("api_call", reraise=False, fallback_result={}) as boundary:
            result = await some_api_call()
            # Note: Cannot return values from within context manager
    """
    boundary = AsyncErrorBoundary(
        operation_name=operation_name,
        logger=logger,
        reraise=reraise,
        fallback_result=fallback_result,
        timeout=timeout,
    )
    async with boundary:
        yield boundary


def configure_structured_logging(
    logger_name: str = __name__,
    level: int = logging.INFO,
    enable_pii_redaction: bool = True,
    json_format: bool = True,
) -> AsyncContextLogger:
    """Configure structured logging with PII redaction.

    Args:
        logger_name: Name of the logger
        level: Logging level
        enable_pii_redaction: Whether to enable PII redaction filter
        json_format: Whether to use JSON formatting

    Returns:
        Configured AsyncContextLogger instance
    """
    base_logger = logging.getLogger(logger_name)
    base_logger.setLevel(level)
    if json_format:

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "correlation_id": getattr(record, "correlation_id", "unknown"),
                }
                for key, value in record.__dict__.items():
                    if key not in (
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "getMessage",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                        "correlation_id",
                    ):
                        log_record[key] = value
                try:
                    import json

                    return json.dumps(log_record)
                except (TypeError, ValueError):
                    return str(log_record)

        formatter = JsonFormatter()
    else:

        class StandardFormatter(logging.Formatter):
            def format(self, record):
                if not hasattr(record, "correlation_id"):
                    record.correlation_id = "unknown"
                return super().format(record)

        formatter = StandardFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(correlation_id)s - %(message)s"
        )
    if not base_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        base_logger.addHandler(handler)
    if enable_pii_redaction:
        pii_filter = PIIRedactionFilter()
        base_logger.addFilter(pii_filter)
    return AsyncContextLogger(base_logger)


def handle_database_errors(
    rollback_session: bool = True,
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: str | None = None,
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
    P = ParamSpec("P")
    T = TypeVar("T")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation = operation_name or func.__name__
            db_session = None
            if rollback_session:
                for arg in args:
                    if isinstance(arg, AsyncSession):
                        db_session = arg
                        break
                if not db_session and "db_session" in kwargs:
                    db_session = kwargs["db_session"]
            attempts = 0
            max_attempts = retry_count + 1
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except OSError as e:
                    logger.error(f"Database I/O error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(
                            f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "database_io",
                        }
                    if return_format == "raise":
                        raise
                    return []
                except (ValueError, TypeError) as e:
                    logger.error(f"Data validation error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "validation",
                        }
                    if return_format == "raise":
                        raise
                    return []
                except (AttributeError, KeyError) as e:
                    logger.error(f"Data structure error in {operation}: {e}")
                    if db_session:
                        await db_session.rollback()
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "data_structure",
                        }
                    if return_format == "raise":
                        raise
                    return []
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    if db_session:
                        await db_session.rollback()
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in {operation}: {e}")
                    logging.exception("Unexpected error in %s", operation)
                    if db_session:
                        await db_session.rollback()
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return []

        @wraps(func)
        async def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation = operation_name or func.__name__
            attempts = 0
            max_attempts = retry_count + 1
            while attempts < max_attempts:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except OSError as e:
                    logger.error(f"I/O error in {operation}: {e}")
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(
                            f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "io"}
                    if return_format == "raise":
                        raise
                    return None
                except (ValueError, TypeError) as e:
                    logger.error(f"Data validation error in {operation}: {e}")
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "validation",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except (AttributeError, KeyError) as e:
                    logger.error(f"Data structure error in {operation}: {e}")
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "data_structure",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in {operation}: {e}")
                    logging.exception("Unexpected error in %s", operation)
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def handle_filesystem_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: str | None = None,
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

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation = operation_name or func.__name__
            attempts = 0
            max_attempts = retry_count + 1
            while attempts < max_attempts:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except FileNotFoundError as e:
                    logger.error(f"File not found in {operation}: {e}")
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "file_not_found",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except PermissionError as e:
                    logger.error(f"Permission denied in {operation}: {e}")
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "permission_denied",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except OSError as e:
                    logger.error(f"I/O error in {operation}: {e}")
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(
                            f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    if return_format == "dict":
                        return {"status": "error", "error": str(e), "error_type": "io"}
                    if return_format == "raise":
                        raise
                    return None
                except ValueError as e:
                    logger.error(f"Path validation error in {operation}: {e}")
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "path_validation",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected filesystem error in {operation}: {e}")
                    logging.exception("Unexpected error in %s", operation)
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return None

        return wrapper

    return decorator


def handle_validation_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: str | None = None,
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
                    return {
                        "status": "error",
                        "error": str(e),
                        "error_type": "value_validation",
                    }
                if return_format == "raise":
                    raise
                return None
            except TypeError as e:
                logger.error(f"Type validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                if return_format == "dict":
                    return {
                        "status": "error",
                        "error": str(e),
                        "error_type": "type_validation",
                    }
                if return_format == "raise":
                    raise
                return None
            except AssertionError as e:
                logger.error(f"Assertion validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                if return_format == "dict":
                    return {
                        "status": "error",
                        "error": str(e),
                        "error_type": "assertion_validation",
                    }
                if return_format == "raise":
                    raise
                return None
            except (KeyError, AttributeError) as e:
                logger.error(f"Data structure validation error in {operation}: {e}")
                if log_validation_details:
                    logger.debug(f"Validation args: {args}, kwargs: {kwargs}")
                if return_format == "dict":
                    return {
                        "status": "error",
                        "error": str(e),
                        "error_type": "structure_validation",
                    }
                if return_format == "raise":
                    raise
                return None
            except KeyboardInterrupt:
                logger.warning(f"{operation} cancelled by user")
                raise
            except Exception as e:
                logger.error(f"Unexpected validation error in {operation}: {e}")
                logging.exception("Unexpected error in %s", operation)
                if return_format == "dict":
                    return {
                        "status": "error",
                        "error": str(e),
                        "error_type": "unexpected",
                    }
                if return_format == "raise":
                    raise
                return None

        return wrapper

    return decorator


def handle_network_errors(
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: str | None = None,
    retry_count: int = 3,
    retry_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
):
    """Decorator for network operations with unified retry logic.

    Handles network-specific errors using the unified retry manager:
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
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation = operation_name or func.__name__
            from prompt_improver.core.retry_manager import (
                RetryableErrorType,
                RetryConfig,
                RetryStrategy,
                get_retry_manager,
            )

            retry_config = RetryConfig(
                max_attempts=retry_count + 1,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=retry_delay,
                backoff_multiplier=backoff_multiplier,
                jitter=True,
                retryable_errors=[
                    RetryableErrorType.NETWORK,
                    RetryableErrorType.TIMEOUT,
                ],
                operation_name=operation,
            )
            retry_manager = get_retry_manager()
            try:

                async def network_operation():
                    try:
                        return await func(*args, **kwargs)
                    except (ConnectionError, TimeoutError) as e:
                        raise
                    except KeyboardInterrupt:
                        logger.warning(f"{operation} cancelled by user")
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected network error in {operation}: {e}")
                        logging.exception("Unexpected error in %s", operation)
                        if return_format == "raise":
                            raise
                        return (
                            {
                                "status": "error",
                                "error": str(e),
                                "error_type": "unexpected",
                            }
                            if return_format == "dict"
                            else None
                        )

                return await retry_manager.retry_async(
                    network_operation, config=retry_config
                )
            except Exception as e:
                if return_format == "dict":
                    return {"status": "error", "error": str(e), "error_type": "network"}
                if return_format == "raise":
                    raise
                return None

        @wraps(func)
        async def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation = operation_name or func.__name__
            for attempt in range(retry_count + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    if attempt < retry_count:
                        delay = retry_delay * backoff_multiplier**attempt
                        logger.warning(
                            f"Network error in {operation} (attempt {attempt + 1}/{retry_count + 1}): {e}. Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(
                        f"Network error in {operation} after {retry_count + 1} attempts: {e}"
                    )
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "network",
                        }
                    if return_format == "raise":
                        raise
                    return None
                except KeyboardInterrupt:
                    logger.warning(f"{operation} cancelled by user")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected network error in {operation}: {e}")
                    logging.exception("Unexpected error in %s", operation)
                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def handle_common_errors(
    rollback_session: bool = False,
    return_format: Literal["dict", "raise", "none"] = "dict",
    operation_name: str | None = None,
    retry_count: int = 0,
    retry_delay: float = 1.0,
):
    """Convenience decorator that combines database, filesystem, and validation error handling.

    This is useful for operations that might encounter any combination of these error types.
    """

    def decorator(func: Callable) -> Callable:
        decorated_func = handle_validation_errors(
            return_format=return_format, operation_name=operation_name
        )(func)
        decorated_func = handle_filesystem_errors(
            return_format=return_format,
            operation_name=operation_name,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )(decorated_func)
        decorated_func = handle_database_errors(
            rollback_session=rollback_session,
            return_format=return_format,
            operation_name=operation_name,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )(decorated_func)
        return decorated_func

    return decorator


# ============================================================================
# FASTAPI INTEGRATION (from common.error_handling)
# ============================================================================

# Context variable for correlation ID tracking across async boundaries
correlation_context: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def handle_repository_errors(correlation_id: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for repository layer error handling.
    
    Catches external exceptions and wraps them in appropriate domain exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                # Get correlation ID from context or parameter
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "repository",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                logger.error(f"Repository error in {func.__name__}: {exc}", extra={"correlation_id": corr_id})
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Get correlation ID from context or parameter
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "repository",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                logger.error(f"Repository error in {func.__name__}: {exc}", extra={"correlation_id": corr_id})
                raise
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_service_errors(correlation_id: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for service layer error handling.
    
    Provides error context and ensures proper error propagation.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                # Wrap unexpected exceptions
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "service",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                logger.error(f"Service error in {func.__name__}: {exc}", extra={"correlation_id": corr_id})
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Wrap unexpected exceptions
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "service",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                logger.error(f"Service error in {func.__name__}: {exc}", extra={"correlation_id": corr_id})
                raise
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# FASTAPI ERROR HANDLERS
# ============================================================================

async def prompt_improver_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle PromptImproverError exceptions in FastAPI."""
    correlation_id = str(uuid.uuid4())
    correlation_context.set(correlation_id)
    
    # Log the error
    logger.error(f"API error: {exc}", extra={"correlation_id": correlation_id})
    
    # Create response (exclude debug info in production)
    include_debug = request.headers.get("X-Debug") == "true"
    response_data = {
        "error": str(exc),
        "correlation_id": correlation_id,
        "timestamp": time.time()
    }
    
    if include_debug:
        response_data["debug"] = {
            "type": type(exc).__name__,
            "module": exc.__class__.__module__
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
        headers={"X-Correlation-ID": correlation_id}
    )


async def validation_exception_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors specifically."""
    correlation_id = str(uuid.uuid4())
    correlation_context.set(correlation_id)
    logger.error(f"Validation error: {exc}", extra={"correlation_id": correlation_id})
    
    response_data = {
        "error": str(exc),
        "error_type": "validation",
        "correlation_id": correlation_id,
        "timestamp": time.time()
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
        headers={"X-Correlation-ID": correlation_id}
    )


async def authentication_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle authentication errors specifically."""
    correlation_id = str(uuid.uuid4())
    correlation_context.set(correlation_id)
    logger.error(f"Authentication error: {exc}", extra={"correlation_id": correlation_id})
    
    response_data = {
        "error": "Authentication failed",
        "correlation_id": correlation_id,
        "timestamp": time.time()
    }
    
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=response_data,
        headers={
            "X-Correlation-ID": correlation_id,
            "WWW-Authenticate": "Bearer"
        }
    )


async def authorization_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle authorization errors specifically."""
    correlation_id = str(uuid.uuid4())
    correlation_context.set(correlation_id)
    logger.error(f"Authorization error: {exc}", extra={"correlation_id": correlation_id})
    
    response_data = {
        "error": "Access denied",
        "correlation_id": correlation_id,
        "timestamp": time.time()
    }
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=response_data,
        headers={"X-Correlation-ID": correlation_id}
    )


async def rate_limit_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle rate limit errors specifically."""
    correlation_id = str(uuid.uuid4())
    correlation_context.set(correlation_id)
    logger.error(f"Rate limit error: {exc}", extra={"correlation_id": correlation_id})
    
    response_data = {
        "error": "Rate limit exceeded",
        "correlation_id": correlation_id,
        "timestamp": time.time()
    }
    headers = {"X-Correlation-ID": correlation_id}
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response_data,
        headers=headers
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_context.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context."""
    correlation_context.set(correlation_id)


def create_correlation_middleware():
    """Create middleware to handle correlation ID extraction and setting."""
    
    async def correlation_middleware(request: Request, call_next):
        # Extract correlation ID from header or generate new one
        correlation_id = (
            request.headers.get("X-Correlation-ID") or
            request.headers.get("X-Request-ID") or
            None
        )
        
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set in context
        correlation_context.set(correlation_id)
        
        # Add to request state for easy access
        request.state.correlation_id = correlation_id
        
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    return correlation_middleware


# ============================================================================
# ERROR RECOVERY UTILITIES
# ============================================================================

def create_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Create a retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Exception types that should trigger retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            import random
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        # Final attempt failed
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    await asyncio.sleep(delay + jitter)
                    
                    logger.warning(
                        f"Retrying {func.__name__} after {delay:.2f}s (attempt {attempt + 1}/{max_retries})",
                        extra={"correlation_id": get_correlation_id()}
                    )
            
            # All retries failed
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            import random
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        # Final attempt failed
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    time.sleep(delay + jitter)
                    
                    logger.warning(
                        f"Retrying {func.__name__} after {delay:.2f}s (attempt {attempt + 1}/{max_retries})",
                        extra={"correlation_id": get_correlation_id()}
                    )
            
            # All retries failed
            raise last_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
