"""Analytics Service Access - 2025 Direct Service Pattern

This module provides direct access functions for analytics services.
Eliminates circular imports through direct imports and on-demand instantiation.
No backward compatibility layers - clean modern service access patterns.
"""
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)

def get_analytics_interface() -> Any | None:
    """Get the AnalyticsQueryInterface class for direct instantiation.

    Modern pattern: Returns the class itself, caller handles database session.
    Usage: interface_class = get_analytics_interface(); instance = interface_class(session)
    """
    try:
        from prompt_improver.database.analytics_query_interface import AnalyticsQueryInterface
        return AnalyticsQueryInterface
    except ImportError as e:
        logger.error('Failed to import AnalyticsQueryInterface: %s', e)
        return None

def get_session_reporter() -> Any | None:
    """Get the SessionSummaryReporter class for direct instantiation.

    Modern pattern: Returns the class itself, caller handles database session.
    Usage: reporter_class = get_session_reporter(); instance = reporter_class(session)
    """
    try:
        from prompt_improver.ml.analytics.session_summary_reporter import SessionSummaryReporter
        return SessionSummaryReporter
    except ImportError as e:
        logger.error('Failed to import SessionSummaryReporter: %s', e)
        return None

def get_analytics_router() -> Any | None:
    """Get the analytics router for real-time functionality.

    Modern pattern: Returns the FastAPI router directly for endpoint registration.
    """
    try:
        from prompt_improver.api.analytics_endpoints import analytics_router
        return analytics_router
    except ImportError as e:
        logger.error('Failed to import analytics_router: %s', e)
        return None
