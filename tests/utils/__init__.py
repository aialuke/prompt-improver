"""
Test utilities package for Phase 4 consolidation.

Provides unified async testing infrastructure eliminating duplicate patterns
across 242+ test files.
"""

from .async_helpers import (
    # Event loop management
    get_or_create_event_loop,
    ensure_event_loop,
    
    # Async test execution  
    run_async_test,
    async_test_wrapper,
    async_test_context,
    
    # Performance testing
    UnifiedPerformanceTimer,
    measure_async_performance,
    
    # Validation framework
    UnifiedValidationResult,
    UnifiedAsyncValidator,
    
    # Connection testing
    test_async_database_connection,
    test_async_redis_connection,
)

__all__ = [
    "get_or_create_event_loop",
    "ensure_event_loop", 
    "run_async_test",
    "async_test_wrapper",
    "async_test_context",
    "UnifiedPerformanceTimer",
    "measure_async_performance",
    "UnifiedValidationResult", 
    "UnifiedAsyncValidator",
    "test_async_database_connection",
    "test_async_redis_connection",
]