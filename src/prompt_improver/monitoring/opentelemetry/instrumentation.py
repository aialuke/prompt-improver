"""
OpenTelemetry Auto-Instrumentation for Production Applications
=============================================================

Provides automatic instrumentation for HTTP, database, Redis, and ML operations
with minimal performance overhead and comprehensive observability.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = SpanKind = Status = StatusCode = None

from .setup import get_tracer
from .metrics import get_database_metrics, get_ml_metrics
from .tracing import SpanAttributes

logger = logging.getLogger(__name__)

T = TypeVar("T")

class InstrumentationManager:
    """Manages automatic instrumentation for various components."""
    
    def __init__(self):
        self._instrumented: Dict[str, bool] = {}
        self._tracer = get_tracer(__name__)
    
    def instrument_all(self) -> None:
        """Enable all available auto-instrumentation."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, skipping instrumentation")
            return
        
        self.instrument_http()
        self.instrument_database()
        self.instrument_redis()
        logger.info("OpenTelemetry auto-instrumentation enabled for all components")
    
    def instrument_http(self) -> None:
        """Instrument HTTP client and server operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("http", False):
            return
        
        try:
            # Instrument HTTP clients
            AioHttpClientInstrumentor().instrument()
            RequestsInstrumentor().instrument()
            
            # FastAPI will be instrumented when the app is created
            # FastAPIInstrumentor().instrument_app(app)
            
            self._instrumented["http"] = True
            logger.info("HTTP instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to instrument HTTP: {e}")
    
    def instrument_database(self) -> None:
        """Instrument database operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("database", False):
            return
        
        try:
            # Instrument AsyncPG
            AsyncPGInstrumentor().instrument()
            
            self._instrumented["database"] = True
            logger.info("Database instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to instrument database: {e}")
    
    def instrument_redis(self) -> None:
        """Instrument Redis operations."""
        if not OTEL_AVAILABLE or self._instrumented.get("redis", False):
            return
        
        try:
            # Instrument Redis
            RedisInstrumentor().instrument()
            
            self._instrumented["redis"] = True
            logger.info("Redis instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to instrument Redis: {e}")
    
    def uninstrument_all(self) -> None:
        """Disable all instrumentation."""
        if not OTEL_AVAILABLE:
            return
        
        try:
            AioHttpClientInstrumentor().uninstrument()
            RequestsInstrumentor().uninstrument()
            AsyncPGInstrumentor().uninstrument()
            RedisInstrumentor().uninstrument()
            
            self._instrumented.clear()
            logger.info("All instrumentation disabled")
            
        except Exception as e:
            logger.error(f"Failed to uninstrument: {e}")

# Global instrumentation manager
_instrumentation_manager: Optional[InstrumentationManager] = None

def get_instrumentation_manager() -> InstrumentationManager:
    """Get global instrumentation manager."""
    global _instrumentation_manager
    if _instrumentation_manager is None:
        _instrumentation_manager = InstrumentationManager()
    return _instrumentation_manager

def instrument_http() -> None:
    """Enable HTTP instrumentation."""
    get_instrumentation_manager().instrument_http()

def instrument_database() -> None:
    """Enable database instrumentation."""
    get_instrumentation_manager().instrument_database()

def instrument_redis() -> None:
    """Enable Redis instrumentation."""
    get_instrumentation_manager().instrument_redis()

def instrument_ml_pipeline() -> None:
    """Enable ML pipeline instrumentation (custom implementation)."""
    # This is handled by our custom decorators rather than auto-instrumentation
    logger.info("ML pipeline instrumentation available via @trace_ml_operation decorator")

def instrument_external_apis() -> None:
    """Enable external API instrumentation."""
    # Covered by HTTP client instrumentation
    instrument_http()

# Custom instrumentation decorators for business logic
def trace_ml_operation(
    operation_type: str,
    model_name: Optional[str] = None,
    capture_io: bool = False
):
    """Decorator to trace ML operations with custom metrics."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not OTEL_AVAILABLE:
            return func
        
        tracer = get_tracer(__name__)
        ml_metrics = get_ml_metrics()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            span_name = f"ml.{operation_type}"
            if model_name:
                span_name += f".{model_name}"
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL
            ) as span:
                # Add ML-specific attributes
                span.set_attribute(SpanAttributes.ML_OPERATION, operation_type)
                if model_name:
                    span.set_attribute(SpanAttributes.ML_MODEL_NAME, model_name)
                
                # Capture input metadata if requested
                if capture_io and args:
                    _capture_ml_input_metadata(span, args, kwargs)
                
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    ml_metrics.record_inference(
                        model_name or "unknown",
                        "unknown",  # version
                        duration_ms,
                        success=True
                    )
                    
                    # Capture output metadata if requested
                    if capture_io:
                        _capture_ml_output_metadata(span, result)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    duration_ms = (time.time() - start_time) * 1000
                    ml_metrics.record_inference(
                        model_name or "unknown",
                        "unknown",
                        duration_ms,
                        success=False
                    )
                    
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            span_name = f"ml.{operation_type}"
            if model_name:
                span_name += f".{model_name}"
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL
            ) as span:
                # Add ML-specific attributes
                span.set_attribute(SpanAttributes.ML_OPERATION, operation_type)
                if model_name:
                    span.set_attribute(SpanAttributes.ML_MODEL_NAME, model_name)
                
                # Capture input metadata if requested
                if capture_io and args:
                    _capture_ml_input_metadata(span, args, kwargs)
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    ml_metrics.record_inference(
                        model_name or "unknown",
                        "unknown",  # version
                        duration_ms,
                        success=True
                    )
                    
                    # Capture output metadata if requested
                    if capture_io:
                        _capture_ml_output_metadata(span, result)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    duration_ms = (time.time() - start_time) * 1000
                    ml_metrics.record_inference(
                        model_name or "unknown",
                        "unknown",
                        duration_ms,
                        success=False
                    )
                    
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def trace_database_operation(
    operation: str,
    table: Optional[str] = None
):
    """Decorator to trace database operations with enhanced metrics."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not OTEL_AVAILABLE:
            return func
        
        tracer = get_tracer(__name__)
        db_metrics = get_database_metrics()
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            span_name = f"db.{operation}"
            if table:
                span_name += f".{table}"
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT
            ) as span:
                # Add database attributes
                span.set_attribute(SpanAttributes.DB_OPERATION, operation)
                if table:
                    span.set_attribute("db.table", table)
                
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    db_metrics.record_query(
                        operation,
                        table or "unknown",
                        duration_ms,
                        success=True
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    duration_ms = (time.time() - start_time) * 1000
                    db_metrics.record_query(
                        operation,
                        table or "unknown",
                        duration_ms,
                        success=False
                    )
                    
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            span_name = f"db.{operation}"
            if table:
                span_name += f".{table}"
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT
            ) as span:
                # Add database attributes
                span.set_attribute(SpanAttributes.DB_OPERATION, operation)
                if table:
                    span.set_attribute("db.table", table)
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record metrics
                    duration_ms = (time.time() - start_time) * 1000
                    db_metrics.record_query(
                        operation,
                        table or "unknown",
                        duration_ms,
                        success=True
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    duration_ms = (time.time() - start_time) * 1000
                    db_metrics.record_query(
                        operation,
                        table or "unknown",
                        duration_ms,
                        success=False
                    )
                    
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def trace_cache_operation(
    operation: str,
    cache_name: str = "default"
):
    """Decorator to trace cache operations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not OTEL_AVAILABLE:
            return func
        
        tracer = get_tracer(__name__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with tracer.start_as_current_span(
                f"cache.{operation}",
                kind=SpanKind.CLIENT
            ) as span:
                span.set_attribute("cache.operation", operation)
                span.set_attribute("cache.name", cache_name)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Detect cache hit/miss from result
                    if operation == "get":
                        hit = result is not None
                        span.set_attribute(SpanAttributes.CACHE_HIT, hit)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            with tracer.start_as_current_span(
                f"cache.{operation}",
                kind=SpanKind.CLIENT
            ) as span:
                span.set_attribute("cache.operation", operation)
                span.set_attribute("cache.name", cache_name)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Detect cache hit/miss from result
                    if operation == "get":
                        hit = result is not None
                        span.set_attribute(SpanAttributes.CACHE_HIT, hit)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def trace_business_operation(
    operation_name: str,
    component: str = "business"
):
    """Decorator to trace business logic operations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not OTEL_AVAILABLE:
            return func
        
        tracer = get_tracer(__name__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with tracer.start_as_current_span(
                f"{component}.{operation_name}",
                kind=SpanKind.INTERNAL
            ) as span:
                span.set_attribute(SpanAttributes.COMPONENT, component)
                span.set_attribute(SpanAttributes.OPERATION, operation_name)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            with tracer.start_as_current_span(
                f"{component}.{operation_name}",
                kind=SpanKind.INTERNAL
            ) as span:
                span.set_attribute(SpanAttributes.COMPONENT, component)
                span.set_attribute(SpanAttributes.OPERATION, operation_name)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _capture_ml_input_metadata(span, args: tuple, kwargs: dict) -> None:
    """Capture ML operation input metadata."""
    try:
        # Look for common ML input patterns
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str):  # Prompt/text input
                span.set_attribute(SpanAttributes.ML_INPUT_SIZE, len(first_arg))
                # Estimate token count (rough approximation)
                estimated_tokens = len(first_arg.split())
                span.set_attribute(SpanAttributes.ML_PROMPT_TOKENS, estimated_tokens)
            elif hasattr(first_arg, '__len__'):  # List, array, etc.
                span.set_attribute(SpanAttributes.ML_INPUT_SIZE, len(first_arg))
        
        # Look for specific parameters
        if 'max_tokens' in kwargs:
            span.set_attribute("ml.max_tokens", kwargs['max_tokens'])
        if 'temperature' in kwargs:
            span.set_attribute("ml.temperature", kwargs['temperature'])
        if 'model' in kwargs:
            span.set_attribute(SpanAttributes.ML_MODEL_NAME, kwargs['model'])
            
    except Exception as e:
        logger.debug(f"Failed to capture ML input metadata: {e}")

def _capture_ml_output_metadata(span, result: Any) -> None:
    """Capture ML operation output metadata."""
    try:
        if isinstance(result, str):  # Text output
            span.set_attribute(SpanAttributes.ML_OUTPUT_SIZE, len(result))
            # Estimate token count
            estimated_tokens = len(result.split())
            span.set_attribute(SpanAttributes.ML_COMPLETION_TOKENS, estimated_tokens)
        elif hasattr(result, '__len__'):  # List, array, etc.
            span.set_attribute(SpanAttributes.ML_OUTPUT_SIZE, len(result))
        elif hasattr(result, 'choices'):  # OpenAI-style response
            if result.choices:
                choice = result.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    span.set_attribute(SpanAttributes.ML_OUTPUT_SIZE, len(content))
            
            # Extract usage information if available
            if hasattr(result, 'usage'):
                usage = result.usage
                if hasattr(usage, 'prompt_tokens'):
                    span.set_attribute(SpanAttributes.ML_PROMPT_TOKENS, usage.prompt_tokens)
                if hasattr(usage, 'completion_tokens'):
                    span.set_attribute(SpanAttributes.ML_COMPLETION_TOKENS, usage.completion_tokens)
                    
    except Exception as e:
        logger.debug(f"Failed to capture ML output metadata: {e}")

# FastAPI instrumentation helper
def instrument_fastapi_app(app):
    """Instrument a FastAPI application."""
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available, skipping FastAPI instrumentation")
        return app
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI application instrumented")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI app: {e}")
    
    return app
