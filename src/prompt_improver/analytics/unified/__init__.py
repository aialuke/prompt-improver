"""Unified Analytics Module.

This module contains the consolidated analytics components that replace
the previous fragmented analytics services.
"""

from prompt_improver.analytics.unified.ab_testing_component import ABTestingComponent
from prompt_improver.analytics.unified.analytics_service_facade import (
    AnalyticsServiceFacade,
    create_analytics_service,
)

# Component exports
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

__all__ = [
    "ABTestingComponent",
    "AnalyticsEvent",
    "AnalyticsMetrics",
    "AnalyticsServiceFacade",
    "AnalyticsServiceProtocol",
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
