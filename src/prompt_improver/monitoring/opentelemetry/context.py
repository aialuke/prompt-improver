"""
OpenTelemetry Context Propagation for Distributed Tracing
========================================================

Provides context propagation utilities for maintaining trace context
across async boundaries, service calls, and distributed systems.
"""

import asyncio
import functools
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable

try:
    from opentelemetry import trace, baggage, context as otel_context
    from opentelemetry.context import Context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.context import attach, detach, get_current
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create stub types for when OpenTelemetry is not available
    trace = None  # type: ignore
    baggage = None  # type: ignore
    otel_context = None  # type: ignore
    Context = None  # type: ignore
    TraceContextTextMapPropagator = None  # type: ignore
    W3CBaggagePropagator = None  # type: ignore
    CompositePropagator = None  # type: ignore
    attach = None  # type: ignore
    detach = None  # type: ignore
    get_current = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global composite propagator for context injection/extraction
_propagator = None

if OTEL_AVAILABLE and CompositePropagator and TraceContextTextMapPropagator and W3CBaggagePropagator:
    _propagator = CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator()
    ])

class ContextKeys:
    """Standard context keys for correlation and tracing."""
    CORRELATION_ID = "correlation_id"
    REQUEST_ID = "request_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    TENANT_ID = "tenant_id"
    SERVICE_NAME = "service_name"
    OPERATION_NAME = "operation_name"

class CorrelationContext:
    """Manages correlation context across async operations."""
    
    def __init__(self):
        self._context_vars: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self._context_vars[key] = value
        
        # Also set in OpenTelemetry baggage if available
        if OTEL_AVAILABLE and baggage:
            baggage.set_baggage(key, str(value))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        # Try local context first
        if key in self._context_vars:
            return self._context_vars[key]
        
        # Try OpenTelemetry baggage
        if OTEL_AVAILABLE and baggage:
            baggage_value = baggage.get_baggage(key)
            if baggage_value is not None:
                return baggage_value
        
        return default
    
    def clear(self) -> None:
        """Clear all context variables."""
        self._context_vars.clear()
    
    def copy(self) -> "CorrelationContext":
        """Create a copy of the context."""
        new_context = CorrelationContext()
        new_context._context_vars = self._context_vars.copy()
        return new_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        result = self._context_vars.copy()
        
        # Add OpenTelemetry baggage if available
        if OTEL_AVAILABLE and baggage:
            baggage_dict = baggage.get_all()
            result.update(baggage_dict)
        
        return result
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load context from dictionary."""
        self._context_vars.update(data)
        
        # Set in OpenTelemetry baggage if available
        if OTEL_AVAILABLE and baggage:
            for key, value in data.items():
                baggage.set_baggage(key, str(value))

# Global correlation context
_correlation_context = CorrelationContext()

def get_correlation_id() -> str:
    """Get or generate a correlation ID."""
    correlation_id = _correlation_context.get(ContextKeys.CORRELATION_ID)
    
    if not correlation_id:
        # Try to extract from current trace
        if OTEL_AVAILABLE and trace:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                if span_context.is_valid:
                    correlation_id = f"{span_context.trace_id:032x}"
        
        # Generate new ID if none found
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        set_correlation_id(correlation_id)
    
    return correlation_id

def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_context.set(ContextKeys.CORRELATION_ID, correlation_id)

def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _correlation_context.get(ContextKeys.REQUEST_ID)

def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context."""
    _correlation_context.set(ContextKeys.REQUEST_ID, request_id)

def get_user_id() -> Optional[str]:
    """Get the current user ID."""
    return _correlation_context.get(ContextKeys.USER_ID)

def set_user_id(user_id: str) -> None:
    """Set the user ID for the current context."""
    _correlation_context.set(ContextKeys.USER_ID, user_id)

def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _correlation_context.get(ContextKeys.SESSION_ID)

def set_session_id(session_id: str) -> None:
    """Set the session ID for the current context."""
    _correlation_context.set(ContextKeys.SESSION_ID, session_id)

def propagate_context(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to propagate context across async operations."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        # Capture current context
        current_context = _correlation_context.copy()
        otel_ctx = get_current() if OTEL_AVAILABLE and get_current else None
        
        # Create task with context
        if asyncio.iscoroutinefunction(func):
            async def context_aware_coro():
                # Restore context in new task
                global _correlation_context
                old_context = _correlation_context
                _correlation_context = current_context
                
                # Attach OpenTelemetry context if available
                token = None
                if OTEL_AVAILABLE and otel_ctx and attach:
                    token = attach(otel_ctx)
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    # Restore previous context
                    _correlation_context = old_context
                    if token and detach:
                        detach(token)
            
            return await context_aware_coro()
        else:
            # For sync functions, just run directly
            return func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> T:
        return func(*args, **kwargs)
    
    # Return appropriate wrapper
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def inject_context(carrier: Dict[str, str]) -> None:
    """Inject current trace and correlation context into a carrier."""
    if not carrier:
        carrier = {}
    
    # Inject OpenTelemetry context
    if OTEL_AVAILABLE and _propagator:
        _propagator.inject(carrier)
    
    # Inject correlation context
    context_data = _correlation_context.to_dict()
    for key, value in context_data.items():
        carrier[f"x-correlation-{key}"] = str(value)
    
    return carrier

def extract_context(carrier: Dict[str, str]) -> Optional[Any]:
    """Extract trace and correlation context from a carrier."""
    if not carrier:
        return None
    
    # Extract OpenTelemetry context
    otel_ctx = None
    if OTEL_AVAILABLE and _propagator:
        otel_ctx = _propagator.extract(carrier)
    
    # Extract correlation context
    correlation_data = {}
    for key, value in carrier.items():
        if key.startswith("x-correlation-"):
            context_key = key[len("x-correlation-"):]
            correlation_data[context_key] = value
    
    # Update global correlation context
    if correlation_data:
        _correlation_context.from_dict(correlation_data)
    
    return otel_ctx

@contextmanager
def context_scope(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **extra_context
):
    """Context manager for scoped context variables."""
    global _correlation_context
    # Save current context
    old_context = _correlation_context.copy()
    
    try:
        # Set new context values
        if correlation_id:
            set_correlation_id(correlation_id)
        if request_id:
            set_request_id(request_id)
        if user_id:
            set_user_id(user_id)
        if session_id:
            set_session_id(session_id)
        
        # Set extra context
        for key, value in extra_context.items():
            _correlation_context.set(key, value)
        
        yield _correlation_context
        
    finally:
        # Restore previous context
        _correlation_context = old_context

def with_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **extra_context
):
    """Decorator to run function with specific context."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with context_scope(
                correlation_id=correlation_id,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                **extra_context
            ):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            with context_scope(
                correlation_id=correlation_id,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                **extra_context
            ):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def create_child_context(
    operation_name: str,
    **context_updates
) -> Any:
    """Create a child context for spawned operations."""
    # Get current OpenTelemetry context
    parent_ctx = get_current() if OTEL_AVAILABLE and get_current else None
    
    # Create new correlation context
    child_correlation = _correlation_context.copy()
    child_correlation.set(ContextKeys.OPERATION_NAME, operation_name)
    
    # Apply updates
    for key, value in context_updates.items():
        child_correlation.set(key, value)
    
    return parent_ctx

async def run_with_context(
    coro: Awaitable[T],
    context: Optional[Context] = None,
    correlation_context: Optional[CorrelationContext] = None
) -> T:
    """Run a coroutine with specific context."""
    # Set up context
    token = None
    old_correlation = None
    
    if OTEL_AVAILABLE and context and attach:
        token = attach(context)
    
    if correlation_context:
        old_correlation = _correlation_context
        _correlation_context = correlation_context
    
    try:
        return await coro
    finally:
        # Restore context
        if token and detach:
            detach(token)
        if old_correlation:
            _correlation_context = old_correlation

def trace_context_middleware():
    """Middleware to extract and propagate trace context in web requests."""
    async def middleware(request, call_next):
        # Extract context from request headers
        headers = dict(request.headers)
        extracted_context = extract_context(headers)
        
        # Generate request ID if not present
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        
        # Run request with extracted context
        with context_scope(request_id=request_id):
            token = None
            if OTEL_AVAILABLE and extracted_context and attach:
                token = attach(extracted_context)
            
            try:
                response = await call_next(request)
                
                # Inject context into response headers
                response.headers["x-correlation-id"] = get_correlation_id()
                response.headers["x-request-id"] = request_id
                
                return response
            finally:
                if token and detach:
                    detach(token)
    
    return middleware

def get_current_context_info() -> Dict[str, Any]:
    """Get current context information for debugging."""
    info = {
        "correlation_context": _correlation_context.to_dict(),
        "otel_available": OTEL_AVAILABLE
    }
    
    if OTEL_AVAILABLE and trace:
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            info["trace_id"] = f"{span_context.trace_id:032x}"
            info["span_id"] = f"{span_context.span_id:016x}"
            info["trace_flags"] = span_context.trace_flags
    
    return info

def clear_context() -> None:
    """Clear all context (useful for testing)."""
    _correlation_context.clear()

# Convenience function for HTTP clients
def get_http_headers_with_context() -> Dict[str, str]:
    """Get HTTP headers with current context for outgoing requests."""
    headers = {}
    inject_context(headers)
    
    # Add correlation headers
    correlation_id = get_correlation_id()
    request_id = get_request_id()
    
    if correlation_id:
        headers["x-correlation-id"] = correlation_id
    if request_id:
        headers["x-request-id"] = request_id
    
    return headers
