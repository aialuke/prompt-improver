"""Analytics Module

This module provides unified analytics capabilities for the prompt improver system.
It consolidates various analytics services into a cohesive, high-performance architecture.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Prompt Improver Team"

# Main exports - only expose what's needed
from .unified.analytics_service_facade import (
    AnalyticsServiceFacade,
    create_analytics_service,
)
from .unified.protocols import (
    AnalyticsEvent,
    AnalyticsMetrics,
    AnalyticsServiceProtocol,
    ExperimentResult,
    PerformanceMetrics,
    SessionMetrics,
    MLModelMetrics,
)

# Component exports for advanced usage
from .unified.data_collection_component import DataCollectionComponent
from .unified.performance_analytics_component import PerformanceAnalyticsComponent
from .unified.ab_testing_component import ABTestingComponent
from .unified.session_analytics_component import SessionAnalyticsComponent
from .unified.ml_analytics_component import MLAnalyticsComponent

# Legacy compatibility - deprecated in v2.0
import warnings

def _deprecation_warning(old_name: str, new_name: str):
    """Show deprecation warning for legacy imports"""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v3.0. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy imports with deprecation warnings
def __getattr__(name: str):
    """Handle legacy attribute access with deprecation warnings"""
    
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
        
        # Try to import from legacy location
        try:
            if name == "AnalyticsService":
                from .legacy.analytics_service import AnalyticsService
                return AnalyticsService
            elif name == "EventBasedMLAnalysisService":
                from .legacy.ml_analysis_service import EventBasedMLAnalysisService
                return EventBasedMLAnalysisService
            elif name == "ModernABTestingService":
                from .legacy.ab_testing_service import ModernABTestingService
                return ModernABTestingService
            elif name == "MemoryOptimizedAnalyticsService":
                from .legacy.memory_optimizer import MemoryOptimizedAnalyticsService
                return MemoryOptimizedAnalyticsService
            elif name == "SessionSummaryReporter":
                from .legacy.session_summary_reporter import SessionSummaryReporter
                return SessionSummaryReporter
            elif name == "SessionComparisonAnalyzer":
                from .legacy.session_comparison_analyzer import SessionComparisonAnalyzer
                return SessionComparisonAnalyzer
            elif name == "PerformanceImprovementCalculator":
                from .legacy.performance_improvement_calculator import PerformanceImprovementCalculator
                return PerformanceImprovementCalculator
        except ImportError:
            # Legacy service not available, suggest new service
            warnings.warn(
                f"Legacy service {name} not found. Please migrate to {legacy_mappings[name]}.",
                ImportWarning,
                stacklevel=2
            )
            return None
    
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
    # Main facade
    "AnalyticsServiceFacade",
    "create_analytics_service",
    
    # Protocol interfaces
    "AnalyticsServiceProtocol",
    "AnalyticsEvent",
    "AnalyticsMetrics",
    "PerformanceMetrics",
    "SessionMetrics",
    "MLModelMetrics",
    "ExperimentResult",
    
    # Individual components
    "DataCollectionComponent",
    "PerformanceAnalyticsComponent",
    "ABTestingComponent",
    "SessionAnalyticsComponent",
    "MLAnalyticsComponent",
    
    # Configuration
    "DEFAULT_CONFIG",
]