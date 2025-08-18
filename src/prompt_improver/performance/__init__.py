"""Performance Optimization and Monitoring

Tools for optimizing system performance, monitoring metrics,
and validating performance targets.
"""

from prompt_improver.performance.monitoring import (
    UnifiedHealthMonitor,
    get_unified_health_monitor,
)
# Import enhanced performance benchmark factory
from prompt_improver.performance.monitoring.performance_benchmark_factory import (
    create_performance_benchmark,
)
# Legacy benchmark import removed to break circular dependency (cycles 4 & 5)
# Use factory pattern instead: create_performance_benchmark()
# from prompt_improver.performance.monitoring.performance_benchmark import (
#     MCPPerformanceBenchmark,
# )
# Temporarily removed AsyncBatchProcessor to break remaining circular dependency
# from prompt_improver.performance.optimization.async_optimizer import AsyncBatchProcessor
from prompt_improver.performance.optimization.performance_optimizer import (
    get_performance_optimizer,
)
from prompt_improver.performance.optimization.response_optimizer import (
    FastJSONSerializer,
    ResponseOptimizer,
)
from prompt_improver.performance.validation.performance_validation import (
    PerformanceValidator,
)

__all__ = [
    # "AsyncBatchProcessor",  # Temporarily removed to break circular dependency
    "FastJSONSerializer",
    "create_performance_benchmark",  # Modern factory pattern (2025)
    "PerformanceValidator",
    "ResponseOptimizer",
    "UnifiedHealthMonitor",
    "get_performance_optimizer",
    "get_unified_health_monitor",
    
    # NOTE: MCPPerformanceBenchmark and AsyncBatchProcessor removed to break circular dependencies
    # Use create_performance_benchmark() factory instead
]
