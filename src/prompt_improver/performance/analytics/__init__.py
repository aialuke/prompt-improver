"""Performance Analytics Components - 2025 Clean Architecture.

Modern analytics components using clean service registry pattern.
All legacy compatibility layers have been removed for clean architecture.

Use the analytics factory for service access:
- Analytics: create_analytics_service() from analytics_factory
- Real-time: get_analytics_router() from api.factories.analytics_api_factory
"""

from prompt_improver.api.factories.analytics_api_factory import get_analytics_router
from prompt_improver.core.services.analytics_factory import create_analytics_service
from prompt_improver.database.analytics_query_interface import AnalyticsQueryInterface

__all__ = [
    "AnalyticsQueryInterface",
    "create_analytics_service",
    "get_analytics_router",
]
