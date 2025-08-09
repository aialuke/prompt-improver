"""Performance Optimization Components

Tools for optimizing async operations, response times, and batch processing.
Note: OptimizationValidator has been moved to ml.optimization.validation for enhanced 2025 features.
"""
from prompt_improver.performance.optimization.async_optimizer import AsyncBatchProcessor
from prompt_improver.performance.optimization.performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from prompt_improver.performance.optimization.response_optimizer import CompressionResult, FastJSONSerializer, ResponseOptimizer

def get_batch_processor():
    """Lazy import of batch processor to avoid circular imports."""
    from prompt_improver.ml.optimization.batch import UnifiedBatchProcessor as BatchProcessor
    return BatchProcessor

def get_batch_config():
    """Lazy import of batch config to avoid circular imports."""
    from prompt_improver.ml.optimization.batch import UnifiedBatchConfig as BatchProcessorConfig
    return BatchProcessorConfig
__all__ = ['AsyncBatchProcessor', 'CompressionResult', 'FastJSONSerializer', 'PerformanceOptimizer', 'ResponseOptimizer', 'get_batch_config', 'get_batch_processor', 'get_performance_optimizer']
