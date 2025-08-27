"""Analytics Module.

This module provides unified analytics capabilities for the prompt improver system.
It consolidates various analytics services into cohesive, high-performance
architecture.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Prompt Improver Team"

# Main exports - only expose what's needed

from prompt_improver.analytics.unified.ab_testing_component import ABTestingComponent
from prompt_improver.analytics.unified.analytics_service_facade import (
    AnalyticsServiceFacade,
    create_analytics_service,
)

# Component exports for advanced usage
from prompt_improver.analytics.unified.data_collection_component import (
    DataCollectionComponent,
)
from prompt_improver.analytics.unified.ml_analytics_component import (
    MLAnalyticsComponent,
)
from prompt_improver.analytics.unified.performance_analytics_component import (
    PerformanceAnalyticsComponent,
)
from prompt_improver.analytics.unified.protocols import (
    AnalyticsEvent,
    AnalyticsMetrics,
    AnalyticsServiceProtocol,
    ExperimentResult,
    MLModelMetrics,
    PerformanceMetrics,
    SessionMetrics,
)
from prompt_improver.analytics.unified.session_analytics_component import (
    SessionAnalyticsComponent,
)

# Module-level constants
DEFAULT_CONFIG = {
    "cache_enabled": True,
    "cache_ttl": 300,
    "max_memory_mb": 500,
    "performance_targets": {
        "max_response_time_ms": 200,
        "min_throughput_rps": 100,
        "max_error_rate": 0.01,
    },
}

# Public API
__all__ = [
    # Configuration
    "DEFAULT_CONFIG",
    "ABTestingComponent",
    "AnalyticsEvent",
    "AnalyticsMetrics",
    # Main facade
    "AnalyticsServiceFacade",
    # Protocol interfaces
    "AnalyticsServiceProtocol",
    # Individual components
    "DataCollectionComponent",
    "ExperimentResult",
    "MLAnalyticsComponent",
    "MLModelMetrics",
    "PerformanceAnalyticsComponent",
    "PerformanceMetrics",
    "SessionAnalyticsComponent",
    "SessionMetrics",
    "create_analytics_service",
]
