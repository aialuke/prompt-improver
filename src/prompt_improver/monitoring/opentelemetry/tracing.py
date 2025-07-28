"""
OpenTelemetry Tracing Utilities for Async Operations
===================================================

Provides decorators and utilities for instrumenting async operations with
distributed tracing, following 2025 best practices for Python applications.
"""

import asyncio
import functools
import inspect
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any, Awaitable, Callable, Dict, Optional, TypeVar, Union, 
    ParamSpec, Concatenate, overload
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.context import attach, detach, get_current
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = Span = None

from .setup import get_tracer

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Global propagator for context injection/extraction
_propagator = CompositePropagator([
    TraceContextTextMapPropagator(),
    W3CBaggagePropagator()
]) if OTEL_AVAILABLE else None

class SpanAttributes:
    """Standard span attribute keys following OpenTelemetry semantic conventions."""
    
    # General
    COMPONENT = "component"
    OPERATION = "operation.name"
    DURATION_MS = "operation.duration_ms"
    
    # Database
    DB_SYSTEM = "db.system"
    DB_NAME = "db.name"
    DB_OPERATION = "db.operation"
    DB_STATEMENT = "db.statement"
    DB_ROWS_AFFECTED = "db.rows_affected"
    
    # HTTP
    HTTP_METHOD = "http.request.method"
    HTTP_URL = "http.url"
    HTTP_STATUS_CODE = "http.response.status_code"
    HTTP_USER_AGENT = "http.user_agent"
    
    # ML/AI Operations
    ML_MODEL_NAME = "ml.model.name"
    ML_MODEL_VERSION = "ml.model.version"
    ML_OPERATION = "ml.operation"
    ML_INPUT_SIZE = "ml.input.size"
    ML_OUTPUT_SIZE = "ml.output.size"
    ML_PROMPT_TOKENS = "ml.prompt.tokens"
    ML_COMPLETION_TOKENS = "ml.completion.tokens"
    
    # Business Logic
    USER_ID = "user.id"
    SESSION_ID = "session.id"
    REQUEST_ID = "request.id"
    CORRELATION_ID = "correlation.id"
    
    # Performance
    CACHE_HIT = "cache.hit"
    CACHE_KEY = "cache.key"
    RETRY_ATTEMPT = "retry.attempt"
    CIRCUIT_BREAKER_STATE = "circuit_breaker.state"

def trace_async(
    operation_name: Optional[str] = None,
    *,
    component: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    record_exception: bool = True,
    span_kind: Optional[trace.SpanKind] = None
):
    """Decorator to trace async functions with OpenTelemetry.
    
    Args:
        operation_name: Custom span name (defaults to function name)
        component: Component/service name for the operation
        capture_args: Whether to capture function arguments as span attributes
        capture_result: Whether to capture function result size/type
        record_exception: Whether to record exceptions in spans
        span_kind: OpenTelemetry span kind
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        if not OTEL_AVAILABLE:
            return func
            
        span_name = operation_name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(__name__)
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not tracer:
                return await func(*args, **kwargs)
                
            with tracer.start_as_current_span(
                span_name,
                kind=span_kind or trace.SpanKind.INTERNAL
            ) as span:
                # Add component attribute
                if component:
                    span.set_attribute(SpanAttributes.COMPONENT, component)
                
                # Capture function arguments if requested
                if capture_args:
                    _capture_function_args(span, func, args, kwargs)
                
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record operation duration
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)
                    
                    # Capture result metadata if requested
                    if capture_result:
                        _capture_result_metadata(span, result)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_sync(
    operation_name: Optional[str] = None,
    *,
    component: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    record_exception: bool = True,
    span_kind: Optional[trace.SpanKind] = None
):
    """Decorator to trace synchronous functions with OpenTelemetry."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if not OTEL_AVAILABLE:
            return func
            
        span_name = operation_name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(__name__)
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not tracer:
                return func(*args, **kwargs)
                
            with tracer.start_as_current_span(
                span_name,
                kind=span_kind or trace.SpanKind.INTERNAL
            ) as span:
                # Add component attribute
                if component:
                    span.set_attribute(SpanAttributes.COMPONENT, component)
                
                # Capture function arguments if requested
                if capture_args:
                    _capture_function_args(span, func, args, kwargs)
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record operation duration
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)
                    
                    # Capture result metadata if requested
                    if capture_result:
                        _capture_result_metadata(span, result)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

@contextmanager
def span_context(
    operation_name: str,
    *,
    component: Optional[str] = None,
    span_kind: Optional[trace.SpanKind] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Context manager for creating spans."""
    if not OTEL_AVAILABLE:
        yield None
        return
        
    tracer = get_tracer(__name__)
    if not tracer:
        yield None
        return
        
    with tracer.start_as_current_span(
        operation_name,
        kind=span_kind or trace.SpanKind.INTERNAL
    ) as span:
        # Add component
        if component:
            span.set_attribute(SpanAttributes.COMPONENT, component)
        
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))

@asynccontextmanager
async def async_span_context(
    operation_name: str,
    *,
    component: Optional[str] = None,
    span_kind: Optional[trace.SpanKind] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Async context manager for creating spans."""
    if not OTEL_AVAILABLE:
        yield None
        return
        
    tracer = get_tracer(__name__)
    if not tracer:
        yield None
        return
        
    with tracer.start_as_current_span(
        operation_name,
        kind=span_kind or trace.SpanKind.INTERNAL
    ) as span:
        # Add component
        if component:
            span.set_attribute(SpanAttributes.COMPONENT, component)
        
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))

def add_span_attributes(**attributes: Any) -> None:
    """Add attributes to the current active span."""
    if not OTEL_AVAILABLE:
        return
        
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)

def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record an exception in the current active span."""
    if not OTEL_AVAILABLE:
        return
        
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.record_exception(exception, attributes=attributes)
        current_span.set_status(Status(StatusCode.ERROR, str(exception)))

def create_span_link(
    span_context: trace.SpanContext,
    attributes: Optional[Dict[str, Any]] = None
) -> trace.Link:
    """Create a span link for connecting related spans."""
    if not OTEL_AVAILABLE:
        return None
        
    return trace.Link(span_context, attributes=attributes)

def get_correlation_id() -> Optional[str]:
    """Extract correlation ID from current trace context."""
    if not OTEL_AVAILABLE:
        return None
        
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        span_context = current_span.get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}"
    return None

def inject_context(carrier: Dict[str, str]) -> None:
    """Inject current trace context into a carrier (e.g., HTTP headers)."""
    if not OTEL_AVAILABLE or not _propagator:
        return
        
    _propagator.inject(carrier)

def extract_context(carrier: Dict[str, str]) -> Optional[Any]:
    """Extract trace context from a carrier."""
    if not OTEL_AVAILABLE or not _propagator:
        return None
        
    return _propagator.extract(carrier)

def with_extracted_context(carrier: Dict[str, str]):
    """Decorator to run function with extracted trace context."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if not OTEL_AVAILABLE:
            return func
            
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = extract_context(carrier)
            if context:
                token = attach(context)
                try:
                    return func(*args, **kwargs)
                finally:
                    detach(token)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

async def propagate_context_async(
    coro: Awaitable[T],
    context: Optional[Any] = None
) -> T:
    """Run a coroutine with propagated trace context."""
    if not OTEL_AVAILABLE:
        return await coro
        
    if context is None:
        context = get_current()
    
    # Create a new task with the context
    task = asyncio.create_task(coro)
    
    # Copy context to the task
    if hasattr(task, 'set_context'):
        task.set_context(context)
    
    return await task

def _capture_function_args(
    span: Span,
    func: Callable,
    args: tuple,
    kwargs: dict
) -> None:
    """Capture function arguments as span attributes."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Add arguments as attributes (excluding sensitive ones)
        for name, value in bound_args.arguments.items():
            if _is_safe_to_capture(name, value):
                span.set_attribute(f"function.arg.{name}", str(value)[:500])  # Limit length
                
    except Exception as e:
        logger.debug(f"Failed to capture function arguments: {e}")

def _capture_result_metadata(span: Span, result: Any) -> None:
    """Capture metadata about the function result."""
    try:
        span.set_attribute("function.result.type", type(result).__name__)
        
        # Capture size for collections
        if hasattr(result, '__len__'):
            span.set_attribute("function.result.size", len(result))
        
        # Capture specific metadata for common types
        if isinstance(result, dict):
            span.set_attribute("function.result.keys", list(result.keys())[:10])  # First 10 keys
        elif isinstance(result, (list, tuple)):
            span.set_attribute("function.result.length", len(result))
            
    except Exception as e:
        logger.debug(f"Failed to capture result metadata: {e}")

def _is_safe_to_capture(name: str, value: Any) -> bool:
    """Check if an argument is safe to capture (no sensitive data)."""
    # Skip sensitive parameter names
    sensitive_names = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key',
        'auth', 'credential', 'session', 'cookie', 'api_key'
    }
    
    if any(sensitive in name.lower() for sensitive in sensitive_names):
        return False
    
    # Skip large objects
    if hasattr(value, '__len__'):
        try:
            if len(value) > 1000:  # Skip large collections
                return False
        except (TypeError, AttributeError):
            pass
    
    # Skip non-serializable types
    if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
        return False
    
    return True

# Convenience aliases for common span kinds
class SpanKind:
    """Common span kinds for easy access."""
    INTERNAL = trace.SpanKind.INTERNAL if OTEL_AVAILABLE else None
    SERVER = trace.SpanKind.SERVER if OTEL_AVAILABLE else None
    CLIENT = trace.SpanKind.CLIENT if OTEL_AVAILABLE else None
    PRODUCER = trace.SpanKind.PRODUCER if OTEL_AVAILABLE else None
    CONSUMER = trace.SpanKind.CONSUMER if OTEL_AVAILABLE else None
