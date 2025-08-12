"""Unified Analytics Module

This module contains the consolidated analytics components that replace
the previous fragmented analytics services.
"""

from .analytics_service_facade import AnalyticsServiceFacade, create_analytics_service
from .protocols import (
    AnalyticsEvent,
    AnalyticsMetrics,
    PerformanceMetrics,
    SessionMetrics,
    MLModelMetrics,
    ExperimentResult,
    AnalyticsServiceProtocol,
)

# Component exports
from .data_collection_component import DataCollectionComponent
from .performance_analytics_component import PerformanceAnalyticsComponent
from .ab_testing_component import ABTestingComponent
from .session_analytics_component import SessionAnalyticsComponent
from .ml_analytics_component import MLAnalyticsComponent

__all__ = [
    "AnalyticsServiceFacade",
    "create_analytics_service",
    "AnalyticsEvent",
    "AnalyticsMetrics",
    "PerformanceMetrics",
    "SessionMetrics", 
    "MLModelMetrics",
    "ExperimentResult",
    "AnalyticsServiceProtocol",
    "DataCollectionComponent",
    "PerformanceAnalyticsComponent",
    "ABTestingComponent",
    "SessionAnalyticsComponent",
    "MLAnalyticsComponent",
]