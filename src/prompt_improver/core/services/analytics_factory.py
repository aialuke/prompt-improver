"""Analytics Service Factory - 2025 Unified Service Pattern.

This module provides factory functions for the unified analytics service.
Modern pattern with single entry point and comprehensive functionality.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def create_analytics_service(
    db_session: AsyncSession, config: dict | None = None
) -> Any:
    """Create unified analytics service facade.

    Modern pattern: Single entry point for all analytics operations.
    Usage: analytics = await create_analytics_service(session, config)
    """
    try:
        from prompt_improver.analytics import create_analytics_service

        return await create_analytics_service(db_session, config)
    except ImportError as e:
        logger.exception(f"Failed to import unified analytics service: {e}")
        return None


# Note: get_analytics_router() moved to api.factories.analytics_api_factory
# to respect Clean Architecture - API components belong in presentation layer
