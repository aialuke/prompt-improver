"""Performance Monitoring Components - 2025 Service Locator Pattern

Tools for monitoring system performance, benchmarking, and tracking metrics.
Enhanced with service locator pattern to eliminate circular dependencies.
"""

from prompt_improver.performance.monitoring.health.unified_health_system import (
    UnifiedHealthMonitor,
    get_unified_health_monitor,
)

# Legacy benchmark import removed to break circular dependency (cycles 4 & 5)
# Use factory pattern instead: create_performance_benchmark()
# from prompt_improver.performance.monitoring.performance_benchmark import (
#     MCPPerformanceBenchmark,
# )

# Import enhanced benchmark with service locator pattern
from prompt_improver.performance.monitoring.performance_benchmark_enhanced import (
    MCPPerformanceBenchmarkEnhanced,
    run_enhanced_performance_benchmark,
)

# Import factory functions for modern service creation
from prompt_improver.performance.monitoring.performance_benchmark_factory import (
    create_performance_benchmark,
    create_performance_benchmark_with_dependencies,
    create_performance_benchmark_from_container,
)

# Import service locator and protocols for dependency injection
from prompt_improver.performance.monitoring.performance_service_locator import (
    PerformanceServiceLocator,
    create_performance_service_locator,
)

__all__ = [
    # Enhanced components with service locator pattern (2025)
    "MCPPerformanceBenchmarkEnhanced",
    "run_enhanced_performance_benchmark",
    
    # Factory functions for modern architecture
    "create_performance_benchmark",
    "create_performance_benchmark_with_dependencies", 
    "create_performance_benchmark_from_container",
    
    # Service locator pattern components
    "PerformanceServiceLocator",
    "create_performance_service_locator",
    
    # Health monitoring (unchanged)
    "UnifiedHealthMonitor",
    "get_unified_health_monitor",
    
    # NOTE: MCPPerformanceBenchmark removed to break circular dependencies
    # Use create_performance_benchmark() factory instead
]
