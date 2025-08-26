"""Analytics API Factory - Factory functions for analytics API components.

This factory provides API-layer specific functions for analytics routing,
moved from core services to respect Clean Architecture layer boundaries.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_analytics_router() -> Any | None:
    """Get the analytics router for real-time functionality.

    Moved from core.services.analytics_factory to respect Clean Architecture.
    API layer components should be accessed from the API layer, not core services.

    Returns:
        FastAPI APIRouter instance for analytics endpoints, or None if import fails
    """
    try:
        from prompt_improver.api.analytics_endpoints import analytics_router
        return analytics_router
    except ImportError as e:
        logger.exception(f"Failed to import analytics_router: {e}")
        return None
