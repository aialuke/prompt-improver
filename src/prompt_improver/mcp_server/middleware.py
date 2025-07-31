"""Custom middleware implementation for APES MCP Server.

Since the MCP SDK doesn't have native middleware support, this module provides
a custom implementation following 2025 FastMCP patterns with clean architecture.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeVar

from mcp import McpError
from mcp.types import ErrorData

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

CallNext = Callable[[T], Coroutine[Any, Any, R]]


@dataclass
class MiddlewareContext:
    """Context object passed through middleware chain."""
    method: str
    source: str = "server"
    type: str = "tool"
    message: Any = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def copy(self, **kwargs: Any) -> 'MiddlewareContext':
        """Create a copy with updated fields."""
        data = {
            'method': self.method,
            'source': self.source,
            'type': self.type,
            'message': self.message,
            'timestamp': self.timestamp,
            'metadata': self.metadata.copy()
        }
        data.update(kwargs)
        return MiddlewareContext(**data)


class Middleware(ABC):
    """Base class for middleware components."""
    
    @abstractmethod
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        """Process the request and call the next middleware."""
        pass


class TimingMiddleware(Middleware):
    """Middleware for measuring request duration."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        start_time = time.perf_counter()
        
        try:
            result = await call_next(context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Store metrics
            self.metrics[context.method].append(duration_ms)
            
            # Log if exceeds threshold
            if duration_ms > 200:
                logger.warning(f"Request {context.method} took {duration_ms:.2f}ms (exceeds 200ms target)")
            else:
                logger.debug(f"Request {context.method} completed in {duration_ms:.2f}ms")
                
            # Add timing to result metadata if possible
            if isinstance(result, dict) and '_metadata' not in result:
                result['_metadata'] = {'duration_ms': duration_ms}
            
            return result
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Request {context.method} failed after {duration_ms:.2f}ms: {e}")
            raise
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of timing metrics."""
        summary = {}
        for method, durations in self.metrics.items():
            if durations:
                summary[method] = {
                    'count': len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'p95_ms': sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
                }
        return summary


class DetailedTimingMiddleware(TimingMiddleware):
    """Enhanced timing middleware with operation breakdown."""
    
    def __init__(self):
        super().__init__()
        self.operation_metrics = defaultdict(lambda: defaultdict(list))
    
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        # Add operation timing hooks to context
        operation_timings = {}
        
        def record_operation(name: str, duration_ms: float):
            operation_timings[name] = duration_ms
            self.operation_metrics[context.method][name].append(duration_ms)
        
        context.metadata['record_operation'] = record_operation
        
        # Call parent timing
        result = await super().__call__(context, call_next)
        
        # Add operation breakdown to metadata
        if isinstance(result, dict) and '_metadata' in result:
            result['_metadata']['operations'] = operation_timings
        
        return result


class StructuredLoggingMiddleware(Middleware):
    """Middleware for structured JSON logging."""
    
    def __init__(self, include_payloads: bool = True, max_payload_length: int = 1000):
        self.include_payloads = include_payloads
        self.max_payload_length = max_payload_length
    
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        # Log request
        log_data = {
            'event': 'mcp_request',
            'method': context.method,
            'source': context.source,
            'type': context.type,
            'timestamp': context.timestamp
        }
        
        if self.include_payloads and context.message:
            payload_str = str(context.message)
            if len(payload_str) > self.max_payload_length:
                payload_str = payload_str[:self.max_payload_length] + '...'
            log_data['payload'] = payload_str
        
        logger.info(log_data)
        
        try:
            result = await call_next(context)
            
            # Log response
            log_data = {
                'event': 'mcp_response',
                'method': context.method,
                'status': 'success',
                'duration_ms': (time.time() - context.timestamp) * 1000
            }
            
            if self.include_payloads and result:
                result_str = str(result)
                if len(result_str) > self.max_payload_length:
                    result_str = result_str[:self.max_payload_length] + '...'
                log_data['result'] = result_str
            
            logger.info(log_data)
            return result
            
        except Exception as e:
            # Log error
            log_data = {
                'event': 'mcp_error',
                'method': context.method,
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_ms': (time.time() - context.timestamp) * 1000
            }
            logger.error(log_data, exc_info=True)
            raise


class RateLimitingMiddleware(Middleware):
    """Token bucket rate limiting middleware."""
    
    def __init__(self, max_requests_per_second: float = 10.0, burst_capacity: int = 20):
        self.max_requests_per_second = max_requests_per_second
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_update
            
            # Replenish tokens
            self.tokens = min(
                self.burst_capacity,
                self.tokens + (time_passed * self.max_requests_per_second)
            )
            self.last_update = current_time
            
            # Check if request can proceed
            if self.tokens < 1:
                raise McpError(ErrorData(
                    code=-32000,
                    message="Rate limit exceeded",
                    data={"retry_after": 1.0 / self.max_requests_per_second}
                ))
            
            # Consume a token
            self.tokens -= 1
        
        # Process request
        return await call_next(context)


class ErrorHandlingMiddleware(Middleware):
    """Middleware for consistent error handling and transformation."""
    
    def __init__(self, include_traceback: bool = False, transform_errors: bool = True):
        self.include_traceback = include_traceback
        self.transform_errors = transform_errors
        self.error_counts = defaultdict(int)
    
    async def __call__(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        try:
            return await call_next(context)
            
        except McpError:
            # MCP errors pass through unchanged
            raise
            
        except Exception as e:
            # Track error statistics
            error_key = f"{type(e).__name__}:{context.method}"
            self.error_counts[error_key] += 1
            
            # Log the error
            logger.exception(f"Error in {context.method}: {type(e).__name__}: {e}")
            
            if self.transform_errors:
                # Transform to MCP error
                error_data = ErrorData(
                    code=-32603,  # Internal error
                    message=f"Internal error in {context.method}: {str(e)}"
                )
                
                if self.include_traceback:
                    import traceback
                    error_data.data = {"traceback": traceback.format_exc()}
                
                raise McpError(error_data) from e
            else:
                raise


class MiddlewareStack:
    """Manages a stack of middleware components."""
    
    def __init__(self):
        self.middleware: list[Middleware] = []
    
    def add(self, middleware: Middleware):
        """Add middleware to the stack."""
        self.middleware.append(middleware)
    
    def wrap(self, handler: Callable) -> Callable:
        """Wrap a handler with the middleware stack."""
        async def wrapped(*args, **kwargs):
            # Extract method name from args
            method_name = kwargs.get('__method__', 'unknown')
            
            # Create initial context
            context = MiddlewareContext(
                method=method_name,
                message={'args': args, 'kwargs': kwargs}
            )
            
            # Build the middleware chain
            async def call_handler(ctx: MiddlewareContext):
                return await handler(*args, **kwargs)
            
            # Apply middleware in reverse order
            chain = call_handler
            for mw in reversed(self.middleware):
                # Capture current middleware and chain
                current_mw = mw
                current_chain = chain
                
                async def next_chain(ctx: MiddlewareContext):
                    return await current_mw(ctx, current_chain)
                
                chain = next_chain
            
            # Execute the chain
            return await chain(context)
        
        return wrapped


def create_default_middleware_stack() -> MiddlewareStack:
    """Create a middleware stack with sensible defaults."""
    stack = MiddlewareStack()
    
    # Order matters - outermost first
    stack.add(ErrorHandlingMiddleware(include_traceback=True))
    stack.add(RateLimitingMiddleware(max_requests_per_second=50))
    stack.add(TimingMiddleware())
    stack.add(StructuredLoggingMiddleware())
    
    return stack