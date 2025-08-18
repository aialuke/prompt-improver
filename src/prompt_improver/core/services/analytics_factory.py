"""Analytics Service Factory - 2025 Unified Service Pattern

This module provides factory functions for the unified analytics service.
Modern pattern with single entry point and comprehensive functionality.
"""

import logging
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def create_analytics_service(
    db_session: AsyncSession, config: Optional[dict] = None
) -> Any:
    """Create unified analytics service facade.
    
    Modern pattern: Single entry point for all analytics operations.
    Usage: analytics = await create_analytics_service(session, config)
    """
    try:
        from prompt_improver.analytics import create_analytics_service
        
        return await create_analytics_service(db_session, config)
    except ImportError as e:
        logger.error(f"Failed to import unified analytics service: {e}")
        return None


def get_analytics_interface() -> Any | None:
    """Get the AnalyticsServiceFacade class for direct instantiation.
    
    DEPRECATED: Use create_analytics_service() for the unified service instead.
    """
    logger.warning("get_analytics_interface() is deprecated. Use create_analytics_service() instead.")
    try:
        from prompt_improver.analytics.unified.analytics_service_facade import AnalyticsServiceFacade

        return AnalyticsServiceFacade
    except ImportError as e:
        logger.error(f"Failed to import AnalyticsServiceFacade: {e}")
        return None


def get_session_reporter() -> Any | None:
    """Get the SessionSummaryReporter class for direct instantiation.
    
    DEPRECATED: Use create_analytics_service() for the unified service instead.
    """
    logger.warning("get_session_reporter() is deprecated. Use create_analytics_service() instead.")
    try:
        from prompt_improver.ml.analytics.session_summary_reporter import SessionSummaryReporter

        return SessionSummaryReporter
    except ImportError as e:
        logger.error(f"Failed to import SessionSummaryReporter: {e}")
        return None


# Note: get_analytics_router() moved to api.factories.analytics_api_factory
# to respect Clean Architecture - API components belong in presentation layer
