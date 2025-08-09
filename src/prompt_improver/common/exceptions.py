"""Shared exception classes for the prompt-improver system.

This module contains all custom exception classes used across
the system to provide consistent error handling.
"""
from typing import Any

class PromptImproverError(Exception):
    """Base exception class for all prompt-improver errors."""

    def __init__(self, message: str, error_code: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

class ConfigurationError(PromptImproverError):
    """Raised when there are configuration-related errors."""

    def __init__(self, message: str, config_key: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='CONFIG_ERROR', details=details)
        self.config_key = config_key

class ConnectionError(PromptImproverError):
    """Raised when connection-related errors occur."""

    def __init__(self, message: str, service: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='CONNECTION_ERROR', details=details)
        self.service = service

class ValidationError(PromptImproverError):
    """Raised when validation errors occur."""

    def __init__(self, message: str, field: str | None=None, value: Any=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='VALIDATION_ERROR', details=details)
        self.field = field
        self.value = value

class AuthenticationError(PromptImproverError):
    """Raised when authentication fails."""

    def __init__(self, message: str='Authentication failed', details: dict[str, Any] | None=None):
        super().__init__(message, error_code='AUTH_ERROR', details=details)

class AuthorizationError(PromptImproverError):
    """Raised when authorization fails."""

    def __init__(self, message: str='Authorization failed', details: dict[str, Any] | None=None):
        super().__init__(message, error_code='AUTHZ_ERROR', details=details)

class DatabaseError(PromptImproverError):
    """Raised when database operations fail."""

    def __init__(self, message: str, query: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='DATABASE_ERROR', details=details)
        self.query = query

class CacheError(PromptImproverError):
    """Raised when cache operations fail."""

    def __init__(self, message: str, cache_key: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='CACHE_ERROR', details=details)
        self.cache_key = cache_key

class MLError(PromptImproverError):
    """Raised when ML operations fail."""

    def __init__(self, message: str, model_name: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='ML_ERROR', details=details)
        self.model_name = model_name

class TimeoutError(PromptImproverError):
    """Raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: float | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='TIMEOUT_ERROR', details=details)
        self.timeout_seconds = timeout_seconds

class RateLimitError(PromptImproverError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str='Rate limit exceeded', retry_after: int | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='RATE_LIMIT_ERROR', details=details)
        self.retry_after = retry_after

class ServiceUnavailableError(PromptImproverError):
    """Raised when a service is unavailable."""

    def __init__(self, message: str, service_name: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='SERVICE_UNAVAILABLE', details=details)
        self.service_name = service_name

class DataError(PromptImproverError):
    """Raised when data-related errors occur."""

    def __init__(self, message: str, data_type: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='DATA_ERROR', details=details)
        self.data_type = data_type

class SecurityError(PromptImproverError):
    """Raised when security-related errors occur."""

    def __init__(self, message: str, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='SECURITY_ERROR', details=details)

class ResourceError(PromptImproverError):
    """Raised when resource-related errors occur."""

    def __init__(self, message: str, resource_type: str | None=None, details: dict[str, Any] | None=None):
        super().__init__(message, error_code='RESOURCE_ERROR', details=details)
        self.resource_type = resource_type
