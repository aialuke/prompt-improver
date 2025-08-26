"""Analytics Module.

This module provides unified analytics capabilities for the prompt improver system.
It consolidates various analytics services into a cohesive, high-performance architecture.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Prompt Improver Team"

# Main exports - only expose what's needed
# Legacy compatibility - deprecated in v2.0
import warnings

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


def _deprecation_warning(old_name: str, new_name: str) -> None:
    """Show deprecation warning for legacy imports."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v3.0. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Legacy imports with deprecation warnings
def __getattr__(name: str) -> None:
    """Handle legacy attribute access with deprecation warnings."""
    # Legacy service mappings
    legacy_mappings = {
        "AnalyticsService": "AnalyticsServiceFacade",
        "EventBasedMLAnalysisService": "MLAnalyticsComponent",
        "ModernABTestingService": "ABTestingComponent",
        "MemoryOptimizedAnalyticsService": "DataCollectionComponent",
        "SessionSummaryReporter": "SessionAnalyticsComponent",
        "SessionComparisonAnalyzer": "SessionAnalyticsComponent",
        "PerformanceImprovementCalculator": "PerformanceAnalyticsComponent",
    }

    if name in legacy_mappings:
        _deprecation_warning(name, legacy_mappings[name])

        # Legacy services are no longer available - suggest new services
        warnings.warn(
            f"Legacy service {name} has been removed. Please migrate to {legacy_mappings[name]}.",
            ImportWarning,
            stacklevel=2
        )
        return

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Module-level constants
DEFAULT_CONFIG = {
    "cache_enabled": True,
    "cache_ttl": 300,
    "max_memory_mb": 500,
    "performance_targets": {
        "max_response_time_ms": 200,
        "min_throughput_rps": 100,
        "max_error_rate": 0.01
    }
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
