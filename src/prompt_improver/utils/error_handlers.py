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

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from re import Pattern
from typing import Any, Dict, Literal, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PIIRedactionFilter(logging.Filter):
    """Filter to redact PII from log records."""

    def __init__(self):
        super().__init__()
        # Common PII patterns
        self.patterns = [
            # Email addresses
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                "[EMAIL_REDACTED]",
            ),
            # Phone numbers (various formats)
            (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
            # Credit card numbers
            (
                re.compile(r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"),
                "[CARD_REDACTED]",
            ),
            # Social Security Numbers
            (re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"), "[SSN_REDACTED]"),
            # IP addresses
            (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
            # API keys and tokens (common patterns)
            (re.compile(r"\b[A-Za-z0-9]{32,}\b"), "[TOKEN_REDACTED]"),
            # Passwords in URLs or key-value pairs
            (
                re.compile(
                    r"(password|pwd|pass|secret|key|token)\s*[=:]\s*\S+", re.IGNORECASE
                ),
                lambda m: f"{m.group(1)}=[REDACTED]",
            ),
        ]

        # PII field names to redact from structured logs
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
        # Redact PII from the message
        if hasattr(record, "msg") and record.msg:
            record.msg = self._redact_message(record.msg)

        # Redact PII from arguments
        if hasattr(record, "args") and record.args:
            record.args = tuple(self._redact_message(str(arg)) for arg in record.args)

        # Redact PII from extra fields
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
            # Try to serialize to check if it's JSON serializable
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # If not serializable, convert to string
            return str(data)

    def _prepare_extra(self, **kwargs) -> dict[str, Any]:
        """Prepare extra fields for logging."""
        extra = {**self._context_vars, **kwargs}
        extra = self._add_correlation_id(extra)

        # Ensure all extra fields are JSON serializable
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


# Global async context logger instance
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

        self.logger.info(
            f"Starting operation: {self.operation_name}",
            operation_start=True,
            timeout=self.timeout,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            # Success case
            self.logger.info(
                f"Operation completed successfully: {self.operation_name}",
                operation_complete=True,
                duration_seconds=duration,
                status="success",
            )
            return False

        # Error case
        self.exception_info = {
            "type": exc_type.__name__,
            "message": str(exc_val),
            "operation": self.operation_name,
            "duration_seconds": duration,
            "correlation_id": self.correlation_id,
        }

        if exc_type == asyncio.TimeoutError:
            self.logger.error(
                f"Operation timed out: {self.operation_name}",
                operation_timeout=True,
                **self.exception_info,
            )
        elif exc_type == asyncio.CancelledError:
            self.logger.warning(
                f"Operation cancelled: {self.operation_name}",
                operation_cancelled=True,
                **self.exception_info,
            )
        else:
            self.logger.exception(
                f"Operation failed: {self.operation_name}",
                operation_failed=True,
                **self.exception_info,
            )

        # Return True to suppress the exception if reraise is False
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

    # Create formatter
    if json_format:
        # Use a custom JSON formatter that handles extra fields properly
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "correlation_id": getattr(record, "correlation_id", "unknown"),
                }

                # Add extra fields
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
                    # Fallback to string representation if JSON serialization fails
                    return str(log_record)

        formatter = JsonFormatter()
    else:
        # For non-JSON format, provide a default correlation_id if missing
        class StandardFormatter(logging.Formatter):
            def format(self, record):
                if not hasattr(record, "correlation_id"):
                    record.correlation_id = "unknown"
                return super().format(record)

        formatter = StandardFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(correlation_id)s - %(message)s"
        )

    # Create handler if none exists
    if not base_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        base_logger.addHandler(handler)

    # Add PII redaction filter
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

                    # Retry for transient I/O errors
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
                    logging.exception(f"Unexpected error in {operation}")
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
        def sync_wrapper(*args, **kwargs) -> Any:
            operation = operation_name or func.__name__
            attempts = 0
            max_attempts = retry_count + 1

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)

                except OSError as e:
                    logger.error(f"I/O error in {operation}: {e}")

                    # Retry for transient I/O errors
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(
                            f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})"
                        )
                        time.sleep(retry_delay)
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
                    logging.exception(f"Unexpected error in {operation}")

                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return None

        # Return appropriate wrapper based on function type
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

                    # Retry for transient I/O errors
                    attempts += 1
                    if attempts < max_attempts:
                        logger.info(
                            f"Retrying {operation} (attempt {attempts + 1}/{max_attempts})"
                        )
                        time.sleep(retry_delay)
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
                    logging.exception(f"Unexpected error in {operation}")

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
                logging.exception(f"Unexpected error in {operation}")

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
                        delay = retry_delay * (backoff_multiplier**attempt)
                        logger.warning(
                            f"Network error in {operation} (attempt {attempt + 1}/{retry_count + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
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
                    logging.exception(f"Unexpected error in {operation}")

                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
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
                        delay = retry_delay * (backoff_multiplier**attempt)
                        logger.warning(
                            f"Network error in {operation} (attempt {attempt + 1}/{retry_count + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
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
                    logging.exception(f"Unexpected error in {operation}")

                    if return_format == "dict":
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": "unexpected",
                        }
                    if return_format == "raise":
                        raise
                    return None

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Convenience function for common error handling patterns
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
        # Apply multiple decorators in sequence
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
