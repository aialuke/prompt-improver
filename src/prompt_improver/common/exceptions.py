"""Shared exception classes for the prompt-improver system.

This module contains all custom exception classes used across
the system to provide consistent error handling and error propagation.

The exception hierarchy follows this structure:
- PromptImproverError (base)
  ├── DomainError (business rule violations)
  │   ├── ValidationError (input validation)
  │   ├── BusinessRuleViolationError (domain rules)
  │   └── AuthorizationError (permission denied)
  ├── InfrastructureError (external system failures)
  │   ├── DatabaseError (data access issues)
  │   ├── CacheError (caching failures)
  │   └── ExternalServiceError (API failures)
  └── SystemError (technical failures)
      ├── ConfigurationError (setup issues)
      ├── ResourceExhaustedError (limits exceeded)
      └── InternalError (unexpected failures)
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class PromptImproverError(Exception):
    """Base exception class for all prompt-improver errors.
    
    Provides structured error handling with correlation tracking,
    context preservation, and consistent logging.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
        severity: str = "ERROR",
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.context = context or {}
        self.cause = cause
        self.severity = severity
        self.timestamp = datetime.now(UTC)
        
        # Add error to context for chaining
        if cause:
            self.context["caused_by"] = {
                "type": type(cause).__name__,
                "message": str(cause),
                "correlation_id": getattr(cause, "correlation_id", None),
            }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to structured dictionary for logging and API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "details": self.details,
            "context": self.context,
        }
    
    def log_error(self, logger_instance: logging.Logger | None = None) -> None:
        """Log the error with structured information."""
        log_instance = logger_instance or logger
        error_data = self.to_dict()
        
        log_message = f"{self.error_code}: {self.message}"
        extra = {
            "correlation_id": self.correlation_id,
            "error_details": self.details,
            "error_context": self.context,
            "error_severity": self.severity,
        }
        
        if self.severity == "CRITICAL":
            log_instance.critical(log_message, extra=extra, exc_info=True)
        elif self.severity == "ERROR":
            log_instance.error(log_message, extra=extra, exc_info=True)
        elif self.severity == "WARNING":
            log_instance.warning(log_message, extra=extra)
        else:
            log_instance.info(log_message, extra=extra)


# ============================================================================
# DOMAIN ERRORS - Business rule violations and validation failures
# ============================================================================

class DomainError(PromptImproverError):
    """Base class for domain-level errors (business rule violations)."""
    
    def __init__(self, message: str, **kwargs):
        # Allow subclasses to override severity
        if "severity" not in kwargs:
            kwargs["severity"] = "ERROR"
        super().__init__(message, **kwargs)


class BusinessRuleViolationError(DomainError):
    """Raised when business rules are violated."""
    
    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        rule_details: dict[str, Any] | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if rule_name:
            details["rule_name"] = rule_name
        if rule_details:
            details["rule_details"] = rule_details
        super().__init__(message, details=details, **kwargs)
        self.rule_name = rule_name


# ============================================================================
# INFRASTRUCTURE ERRORS - External system and service failures
# ============================================================================

class InfrastructureError(PromptImproverError):
    """Base class for infrastructure-level errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="ERROR", **kwargs)


class ExternalServiceError(InfrastructureError):
    """Raised when external services fail or are unavailable."""
    
    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        service_response: dict[str, Any] | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if service_name:
            details["service_name"] = service_name
        if service_response:
            details["service_response"] = service_response
        super().__init__(message, details=details, **kwargs)
        self.service_name = service_name


# ============================================================================
# SYSTEM ERRORS - Technical failures and resource issues
# ============================================================================

class SystemError(PromptImproverError):
    """Base class for system-level technical failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="CRITICAL", **kwargs)


class ResourceExhaustedError(SystemError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        current_usage: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if resource_type:
            details["resource_type"] = resource_type
        if current_usage:
            details["current_usage"] = current_usage
        if limits:
            details["limits"] = limits
        super().__init__(message, details=details, **kwargs)
        self.resource_type = resource_type


class InternalError(SystemError):
    """Raised for unexpected internal failures that shouldn't occur."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class ConfigurationError(SystemError):
    """Raised when there are configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_section: str | None = None,
        expected_type: str | None = None,
        actual_value: Any = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if config_key:
            details["config_key"] = config_key
        if config_section:
            details["config_section"] = config_section
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
        super().__init__(message, error_code="CONFIG_ERROR", details=details, **kwargs)
        self.config_key = config_key


class ConnectionError(InfrastructureError):
    """Raised when connection-related errors occur."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if service:
            details["service"] = service
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, error_code="CONNECTION_ERROR", details=details, **kwargs)
        self.service = service


class ValidationError(DomainError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected_type: str | None = None,
        validation_rule: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if expected_type:
            details["expected_type"] = expected_type
        if validation_rule:
            details["validation_rule"] = validation_rule
        super().__init__(message, error_code="VALIDATION_ERROR", details=details, **kwargs)
        self.field = field
        self.value = value


class AuthenticationError(DomainError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        user_id: str | None = None,
        auth_method: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if user_id:
            details["user_id"] = user_id
        if auth_method:
            details["auth_method"] = auth_method
        super().__init__(message, error_code="AUTH_ERROR", details=details, severity="WARNING", **kwargs)


class AuthorizationError(DomainError):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str = "Authorization failed",
        user_id: str | None = None,
        required_permission: str | None = None,
        resource: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if user_id:
            details["user_id"] = user_id
        if required_permission:
            details["required_permission"] = required_permission
        if resource:
            details["resource"] = resource
        super().__init__(message, error_code="AUTHZ_ERROR", details=details, severity="WARNING", **kwargs)


class DatabaseError(InfrastructureError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        query: str | None = None,
        table: str | None = None,
        operation: str | None = None,
        database_name: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if query:
            # Sanitize query for logging (remove sensitive data)
            details["query"] = query[:200] + "..." if len(query) > 200 else query
        if table:
            details["table"] = table
        if operation:
            details["operation"] = operation
        if database_name:
            details["database_name"] = database_name
        super().__init__(message, error_code="DATABASE_ERROR", details=details, **kwargs)
        self.query = query


class CacheError(InfrastructureError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        cache_key: str | None = None,
        cache_type: str | None = None,
        operation: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if cache_key:
            details["cache_key"] = cache_key
        if cache_type:
            details["cache_type"] = cache_type
        if operation:
            details["operation"] = operation
        super().__init__(message, error_code="CACHE_ERROR", details=details, **kwargs)
        self.cache_key = cache_key


class MLError(InfrastructureError):
    """Raised when ML operations fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        operation: str | None = None,
        model_version: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if model_name:
            details["model_name"] = model_name
        if operation:
            details["operation"] = operation
        if model_version:
            details["model_version"] = model_version
        super().__init__(message, error_code="ML_ERROR", details=details, **kwargs)
        self.model_name = model_name


class TimeoutError(SystemError):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        elapsed_seconds: float | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        if elapsed_seconds:
            details["elapsed_seconds"] = elapsed_seconds
        super().__init__(message, error_code="TIMEOUT_ERROR", details=details, **kwargs)
        self.timeout_seconds = timeout_seconds


class RateLimitError(SystemError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        limit_type: str | None = None,
        current_usage: int | None = None,
        limit_value: int | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit_type:
            details["limit_type"] = limit_type
        if current_usage:
            details["current_usage"] = current_usage
        if limit_value:
            details["limit_value"] = limit_value
        super().__init__(message, error_code="RATE_LIMIT_ERROR", details=details, severity="WARNING", **kwargs)
        self.retry_after = retry_after


class ServiceUnavailableError(InfrastructureError):
    """Raised when a service is unavailable."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        service_version: str | None = None,
        last_known_status: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if service_name:
            details["service_name"] = service_name
        if service_version:
            details["service_version"] = service_version
        if last_known_status:
            details["last_known_status"] = last_known_status
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", details=details, **kwargs)
        self.service_name = service_name


class DataError(DomainError):
    """Raised when data-related errors occur."""

    def __init__(
        self,
        message: str,
        data_type: str | None = None,
        data_source: str | None = None,
        validation_failures: list[str] | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if data_type:
            details["data_type"] = data_type
        if data_source:
            details["data_source"] = data_source
        if validation_failures:
            details["validation_failures"] = validation_failures
        super().__init__(message, error_code="DATA_ERROR", details=details, **kwargs)
        self.data_type = data_type


class SecurityError(DomainError):
    """Raised when security-related errors occur."""

    def __init__(
        self,
        message: str,
        security_check: str | None = None,
        threat_level: str = "MEDIUM",
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if security_check:
            details["security_check"] = security_check
        details["threat_level"] = threat_level
        
        # Security errors are always at least WARNING severity
        severity = "CRITICAL" if threat_level == "HIGH" else "WARNING"
        super().__init__(message, error_code="SECURITY_ERROR", details=details, severity=severity, **kwargs)


class ResourceError(SystemError):
    """Raised when resource-related errors occur."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        operation: str | None = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        if operation:
            details["operation"] = operation
        super().__init__(message, error_code="RESOURCE_ERROR", details=details, **kwargs)
        self.resource_type = resource_type


# ============================================================================
# ERROR PROPAGATION UTILITIES
# ============================================================================

def wrap_external_exception(
    exc: Exception,
    message: str | None = None,
    correlation_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> PromptImproverError:
    """Wrap external exceptions in our exception hierarchy.
    
    Args:
        exc: The original exception
        message: Custom error message (defaults to str(exc))
        correlation_id: Optional correlation ID for tracking
        context: Additional context information
        
    Returns:
        Appropriate PromptImproverError subclass
    """
    error_message = message or str(exc)
    
    # Map common external exceptions to our hierarchy
    if isinstance(exc, (ConnectionError, OSError)) and "connection" in str(exc).lower():
        return ConnectionError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    elif isinstance(exc, TimeoutError):
        return TimeoutError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    elif isinstance(exc, ValueError):
        return ValidationError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    elif isinstance(exc, PermissionError):
        return AuthorizationError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    elif "database" in str(exc).lower() or "sql" in str(exc).lower():
        return DatabaseError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    elif "redis" in str(exc).lower() or "cache" in str(exc).lower():
        return CacheError(error_message, cause=exc, correlation_id=correlation_id, context=context)
    else:
        return InternalError(
            message=error_message,
            cause=exc,
            correlation_id=correlation_id,
            context=context
        )


def create_error_response(exc: PromptImproverError, include_debug: bool = False) -> dict[str, Any]:
    """Create a structured error response for API endpoints.
    
    Args:
        exc: The exception to convert
        include_debug: Whether to include debug information
        
    Returns:
        Dictionary suitable for JSON API responses
    """
    response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "correlation_id": exc.correlation_id,
            "timestamp": exc.timestamp.isoformat(),
        }
    }
    
    if include_debug and exc.details:
        response["error"]["details"] = exc.details
        
    if include_debug and exc.context:
        response["error"]["context"] = exc.context
    
    return response
