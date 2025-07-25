"""Performance Analytics Components - 2025 Clean Architecture

Modern analytics components using clean service registry pattern.
All legacy compatibility layers have been removed for clean architecture.

Use the analytics factory for service access:
- Analytics: get_analytics_interface() from analytics_factory
- Real-time: get_analytics_router() from analytics_factory
- Reporting: get_session_reporter() from analytics_factory
"""

# Direct imports to modern components
from ...ml.analytics.session_summary_reporter import SessionSummaryReporter
from ...database.analytics_query_interface import AnalyticsQueryInterface
from ...core.services.analytics_factory import (
    get_analytics_interface,
    get_analytics_router,
    get_session_reporter
)

__all__ = [
    # Modern service factory functions
    "get_analytics_interface",
    "get_analytics_router",
    "get_session_reporter",

    # Modern analytics components
    "SessionSummaryReporter",
    "AnalyticsQueryInterface",
]
