"""Performance Optimization Components

Tools for optimizing async operations, response times, and batch processing.
Note: OptimizationValidator has been moved to ml.optimization.validation for enhanced 2025 features.
"""

from .async_optimizer import AsyncBatchProcessor, ConnectionPoolManager
from .performance_optimizer import get_performance_optimizer, PerformanceOptimizer
from .response_optimizer import ResponseOptimizer, FastJSONSerializer, CompressionResult

# Lazy import to avoid circular dependency with ml module
def get_batch_processor():
    """Lazy import of batch processor to avoid circular imports."""
    from ...ml.optimization.batch import UnifiedBatchProcessor as BatchProcessor
    return BatchProcessor

def get_batch_config():
    """Lazy import of batch config to avoid circular imports."""
    from ...ml.optimization.batch import UnifiedBatchConfig as BatchProcessorConfig
    return BatchProcessorConfig

__all__ = [
    "AsyncBatchProcessor",
    "ConnectionPoolManager",
    "get_performance_optimizer",
    "PerformanceOptimizer",
    "ResponseOptimizer",
    "FastJSONSerializer",
    "CompressionResult",
    "get_batch_processor",
    "get_batch_config",
]
