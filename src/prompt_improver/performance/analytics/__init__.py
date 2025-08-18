"""Performance Analytics Components - 2025 Clean Architecture

Modern analytics components using clean service registry pattern.
All legacy compatibility layers have been removed for clean architecture.

Use the analytics factory for service access:
- Analytics: get_analytics_interface() from analytics_factory
- Real-time: get_analytics_router() from api.factories.analytics_api_factory
- Reporting: get_session_reporter() from analytics_factory
"""

from prompt_improver.core.services.analytics_factory import (
    get_analytics_interface,
    get_session_reporter,
)
from prompt_improver.api.factories.analytics_api_factory import get_analytics_router
from prompt_improver.database.analytics_query_interface import AnalyticsQueryInterface
from prompt_improver.ml.analytics.session_summary_reporter import SessionSummaryReporter

__all__ = [
    "AnalyticsQueryInterface",
    "SessionSummaryReporter",
    "get_analytics_interface",
    "get_analytics_router",
    "get_session_reporter",
]
