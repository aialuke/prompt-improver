"""Analytics Service Access - 2025 Direct Service Pattern

This module provides direct access functions for analytics services.
Eliminates circular imports through direct imports and on-demand instantiation.
No backward compatibility layers - clean modern service access patterns.
"""

import logging
from typing import Any, Optional

# Direct imports - no service registry dependencies needed

logger = logging.getLogger(__name__)

# Analytics Service Access Functions

# Use get_analytics_interface(), get_session_reporter(), get_analytics_router() instead

# Modern Direct Service Access
# Clean service access without factory complexity or backward compatibility layers


# Direct Service Access Functions
# Modern pattern: Direct imports and instantiation, no service registry needed

# Convenience Functions for Common Usage Patterns

def get_analytics_interface() -> Optional[Any]:
    """Get the AnalyticsQueryInterface class for direct instantiation.
    
    Modern pattern: Returns the class itself, caller handles database session.
    Usage: interface_class = get_analytics_interface(); instance = interface_class(session)
    """
    try:
        from ...database.analytics_query_interface import AnalyticsQueryInterface
        return AnalyticsQueryInterface
    except ImportError as e:
        logger.error(f"Failed to import AnalyticsQueryInterface: {e}")
        return None

def get_session_reporter() -> Optional[Any]:
    """Get the SessionSummaryReporter class for direct instantiation.
    
    Modern pattern: Returns the class itself, caller handles database session.
    Usage: reporter_class = get_session_reporter(); instance = reporter_class(session)
    """
    try:
        from ...ml.analytics.session_summary_reporter import SessionSummaryReporter
        return SessionSummaryReporter
    except ImportError as e:
        logger.error(f"Failed to import SessionSummaryReporter: {e}")
        return None

def get_analytics_router() -> Optional[Any]:
    """Get the analytics router for real-time functionality.
    
    Modern pattern: Returns the FastAPI router directly for endpoint registration.
    """
    try:
        from ...api.analytics_endpoints import analytics_router
        return analytics_router
    except ImportError as e:
        logger.error(f"Failed to import analytics_router: {e}")
        return None

# Modern Analytics Services - Direct Service Access
# Use get_analytics_interface(), get_analytics_router(), get_session_reporter() for clean service access

# Modern Analytics Implementation
# Direct service access eliminates circular imports and service registry complexity
# Services are imported and created on-demand by calling code
