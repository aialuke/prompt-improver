"""Error handling middleware and utilities.

This module provides consistent error handling across all application layers,
including error propagation, logging, and response formatting.
"""

import logging
import traceback
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from prompt_improver.common.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    DomainError,
    InfrastructureError,
    PromptImproverError,
    RateLimitError,
    ResourceExhaustedError,
    SecurityError,
    ServiceUnavailableError,
    SystemError,
    TimeoutError,
    ValidationError,
    create_error_response,
    wrap_external_exception,
)

logger = logging.getLogger(__name__)

# Context variable for correlation ID tracking across async boundaries
correlation_context: ContextVar[str | None] = ContextVar("correlation_id", default=None)

T = TypeVar("T")


# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_repository_errors(correlation_id: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for repository layer error handling.
    
    Catches external exceptions and wraps them in appropriate domain exceptions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except PromptImproverError:
                # Re-raise our own exceptions
                raise
            except Exception as exc:
                # Get correlation ID from context or parameter
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "repository",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                wrapped_exc = wrap_external_exception(exc, correlation_id=corr_id, context=context)
                wrapped_exc.log_error(logger)
                raise wrapped_exc
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except PromptImproverError:
                # Re-raise our own exceptions
                raise
            except Exception as exc:
                # Get correlation ID from context or parameter
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "repository",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                wrapped_exc = wrap_external_exception(exc, correlation_id=corr_id, context=context)
                wrapped_exc.log_error(logger)
                raise wrapped_exc
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
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
            except PromptImproverError as exc:
                # Add service layer context to existing exception
                if "service_layer" not in exc.context:
                    exc.context.update({
                        "service_layer": {
                            "function": func.__name__,
                            "module": func.__module__,
                        }
                    })
                exc.log_error(logger)
                raise
            except Exception as exc:
                # Wrap unexpected exceptions
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "service",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                wrapped_exc = wrap_external_exception(exc, correlation_id=corr_id, context=context)
                wrapped_exc.log_error(logger)
                raise wrapped_exc
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except PromptImproverError as exc:
                # Add service layer context to existing exception
                if "service_layer" not in exc.context:
                    exc.context.update({
                        "service_layer": {
                            "function": func.__name__,
                            "module": func.__module__,
                        }
                    })
                exc.log_error(logger)
                raise
            except Exception as exc:
                # Wrap unexpected exceptions
                corr_id = correlation_id or correlation_context.get()
                context = {
                    "layer": "service",
                    "function": func.__name__,
                    "module": func.__module__,
                }
                wrapped_exc = wrap_external_exception(exc, correlation_id=corr_id, context=context)
                wrapped_exc.log_error(logger)
                raise wrapped_exc
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# FASTAPI ERROR HANDLERS
# ============================================================================

async def prompt_improver_exception_handler(request: Request, exc: PromptImproverError) -> JSONResponse:
    """Handle PromptImproverError exceptions in FastAPI."""
    # Set correlation context for downstream logging
    correlation_context.set(exc.correlation_id)
    
    # Log the error
    exc.log_error(logger)
    
    # Determine HTTP status code based on exception type
    status_code = _get_http_status_code(exc)
    
    # Create response (exclude debug info in production)
    include_debug = request.headers.get("X-Debug") == "true"
    response_data = create_error_response(exc, include_debug=include_debug)
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
        headers={"X-Correlation-ID": exc.correlation_id}
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors specifically."""
    correlation_context.set(exc.correlation_id)
    exc.log_error(logger)
    
    response_data = create_error_response(exc, include_debug=True)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
        headers={"X-Correlation-ID": exc.correlation_id}
    )


async def authentication_exception_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """Handle authentication errors specifically."""
    correlation_context.set(exc.correlation_id)
    exc.log_error(logger)
    
    response_data = create_error_response(exc, include_debug=False)
    
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=response_data,
        headers={
            "X-Correlation-ID": exc.correlation_id,
            "WWW-Authenticate": "Bearer"
        }
    )


async def authorization_exception_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
    """Handle authorization errors specifically."""
    correlation_context.set(exc.correlation_id)
    exc.log_error(logger)
    
    response_data = create_error_response(exc, include_debug=False)
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=response_data,
        headers={"X-Correlation-ID": exc.correlation_id}
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Handle rate limit errors specifically."""
    correlation_context.set(exc.correlation_id)
    exc.log_error(logger)
    
    response_data = create_error_response(exc, include_debug=False)
    headers = {"X-Correlation-ID": exc.correlation_id}
    
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response_data,
        headers=headers
    )


def _get_http_status_code(exc: PromptImproverError) -> int:
    """Map exception types to HTTP status codes."""
    if isinstance(exc, ValidationError):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, AuthenticationError):
        return status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, AuthorizationError):
        return status.HTTP_403_FORBIDDEN
    elif isinstance(exc, RateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, ResourceExhaustedError):
        return status.HTTP_507_INSUFFICIENT_STORAGE
    elif isinstance(exc, ServiceUnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, TimeoutError):
        return status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(exc, DomainError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, InfrastructureError):
        return status.HTTP_502_BAD_GATEWAY
    elif isinstance(exc, SystemError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        return status.HTTP_500_INTERNAL_SERVER_ERROR


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
            import uuid
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
    exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, ServiceUnavailableError)
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
            import asyncio
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
            if isinstance(last_exception, PromptImproverError):
                raise last_exception
            else:
                raise wrap_external_exception(
                    last_exception,
                    f"Operation failed after {max_retries} retries",
                    correlation_id=get_correlation_id()
                )
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            import time
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
            if isinstance(last_exception, PromptImproverError):
                raise last_exception
            else:
                raise wrap_external_exception(
                    last_exception,
                    f"Operation failed after {max_retries} retries",
                    correlation_id=get_correlation_id()
                )
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator