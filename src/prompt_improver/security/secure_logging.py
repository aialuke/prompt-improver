"""Secure logging utilities to prevent information leakage.

Provides sanitized error logging that prevents exposure of sensitive information.
"""

import logging
import re
from typing import Any, Optional


class SecureLogger:
    """Secure logging wrapper that prevents information leakage."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # Patterns that might contain sensitive information
        self.sensitive_patterns = [
            r'password[^\s]*[\s=:]+[^\s]+',
            r'token[^\s]*[\s=:]+[^\s]+',
            r'key[^\s]*[\s=:]+[^\s]+',
            r'secret[^\s]*[\s=:]+[^\s]+',
            r'api[_-]?key[^\s]*[\s=:]+[^\s]+',
            r'auth[^\s]*[\s=:]+[^\s]+',
            r'credential[^\s]*[\s=:]+[^\s]+',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            r'/[a-zA-Z]:/.*',  # File paths (Windows)
            r'/[a-zA-Z0-9_/.-]*[a-zA-Z0-9_-]',  # File paths (Unix)
        ]

        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sensitive_patterns]

    def sanitize_message(self, message: str) -> str:
        """Sanitize log message to remove sensitive information.

        Args:
            message: Original log message

        Returns:
            Sanitized message
        """
        sanitized = message

        # Replace sensitive patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)

        return sanitized

    def log_error_safely(self, level: str, operation: str, error: Exception,
                        include_type: bool = True, extra_context: Optional[str] = None) -> None:
        """Log error safely without exposing sensitive information.

        Args:
            level: Log level (error, warning, info, debug)
            operation: Description of operation that failed
            error: The exception that occurred
            include_type: Whether to include exception type
            extra_context: Additional safe context to include
        """
        error_type = type(error).__name__ if include_type else "Exception"

        # Base message
        message = f"{operation} failed: {error_type}"

        # Add safe context if provided
        if extra_context:
            message += f" - {extra_context}"

        # Log at appropriate level
        log_method = getattr(self.logger, level.lower())
        log_method(message)

        # Log detailed error at debug level only
        if self.logger.isEnabledFor(logging.DEBUG):
            sanitized_error = self.sanitize_message(str(error))
            self.logger.debug(f"Detailed error for {operation}: {sanitized_error}")

    def error(self, operation: str, error: Exception, **kwargs: Any) -> None:
        """Log error at ERROR level."""
        self.log_error_safely("error", operation, error, **kwargs)

    def warning(self, operation: str, error: Exception, **kwargs: Any) -> None:
        """Log error at WARNING level."""
        self.log_error_safely("warning", operation, error, **kwargs)

    def info(self, operation: str, error: Exception, **kwargs: Any) -> None:
        """Log error at INFO level."""
        self.log_error_safely("info", operation, error, **kwargs)

    def debug(self, message: str) -> None:
        """Log debug message with sanitization."""
        sanitized = self.sanitize_message(message)
        self.logger.debug(sanitized)

    def log_security_event(self, event_type: str, details: dict[str, Any], level: str = "info") -> None:
        """Log security-related events with careful sanitization.

        Args:
            event_type: Type of security event
            details: Dictionary of event details
            level: Log level
        """
        # Sanitize details
        safe_details = {}
        for key, value in details.items():
            if key.lower() in ['password', 'token', 'key', 'secret', 'credential']:
                safe_details[key] = '[REDACTED]'
            elif isinstance(value, str):
                safe_details[key] = self.sanitize_message(value)
            else:
                safe_details[key] = value

        message = f"Security Event: {event_type}"
        if safe_details:
            message += f" - {safe_details}"

        log_method = getattr(self.logger, level.lower())
        log_method(message)

    def log_performance_metric(self, operation: str, metrics: dict[str, Any]) -> None:
        """Log performance metrics safely.

        Args:
            operation: Operation being measured
            metrics: Performance metrics dictionary
        """
        # Only log safe numeric metrics
        safe_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key.lower() not in ['id', 'hash']:
                safe_metrics[key] = value

        if safe_metrics:
            self.logger.info(f"Performance - {operation}: {safe_metrics}")

    # Delegate other methods to underlying logger
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying logger."""
        return getattr(self.logger, name)