"""Analytics Service Factory - 2025 Best Practice Implementation

This module provides factory functions for analytics services using the service registry pattern.
Eliminates circular imports by centralizing service creation and dependency management.
"""

import logging
from typing import Any, Optional

from .service_registry import (
    register_analytics_service,
    register_real_time_analytics_service,
    get_analytics_service,
    get_real_time_analytics_service,
    service_provider,
    ServiceScope
)

logger = logging.getLogger(__name__)

# Analytics Service Factory Functions

def create_analytics_query_interface():
    """
    Create AnalyticsQueryInterface factory function.

    Returns a factory function that creates AnalyticsQueryInterface instances
    with proper database session handling.
    """
    try:
        from ...database.analytics_query_interface import AnalyticsQueryInterface

        def analytics_factory(db_session=None):
            """Factory function for AnalyticsQueryInterface"""
            if db_session is None:
                # For backward compatibility, return the class itself
                # The calling code should handle session creation
                logger.info("Returning AnalyticsQueryInterface class - session should be provided by caller")
                return AnalyticsQueryInterface
            else:
                return AnalyticsQueryInterface(db_session)

        return analytics_factory
    except ImportError as e:
        logger.warning(f"Failed to import AnalyticsQueryInterface: {e}")
        return None

@service_provider("session_summary_reporter", ServiceScope.SINGLETON)
def create_session_summary_reporter():
    """Create SessionSummaryReporter instance"""
    try:
        from ...ml.analytics.session_summary_reporter import SessionSummaryReporter
        return SessionSummaryReporter()
    except ImportError as e:
        logger.warning(f"Failed to import SessionSummaryReporter: {e}")
        return None

@service_provider("analytics_endpoints", ServiceScope.SINGLETON)
def create_analytics_endpoints():
    """Create analytics endpoints router"""
    try:
        from ...api.analytics_endpoints import analytics_router
        return analytics_router
    except ImportError as e:
        logger.warning(f"Failed to import analytics_endpoints: {e}")
        return None

# Backward Compatibility Service Factories

def create_legacy_analytics_service():
    """
    Create legacy AnalyticsService replacement using new Week 10 components.

    This provides backward compatibility for existing code that expects AnalyticsService.
    """
    try:
        # Use the new AnalyticsQueryInterface as replacement
        return create_analytics_query_interface()
    except Exception as e:
        logger.error(f"Failed to create legacy analytics service: {e}")
        return None

def create_legacy_real_time_analytics_service():
    """
    Create legacy RealTimeAnalyticsService replacement.

    Returns None as real-time analytics should use the new WebSocket endpoints.
    """
    logger.warning(
        "RealTimeAnalyticsService is deprecated. "
        "Use analytics_endpoints WebSocket from prompt_improver.api.analytics_endpoints"
    )
    return None

# Service Registration Functions

def register_all_analytics_services():
    """Register all analytics services with the service registry"""
    logger.info("Registering analytics services...")

    # Register new Week 10 services
    from .service_registry import register_service, ServiceScope
    register_service("analytics_query_interface", create_analytics_query_interface, ServiceScope.SINGLETON)
    register_service("session_summary_reporter", create_session_summary_reporter, ServiceScope.SINGLETON)
    register_service("analytics_endpoints", create_analytics_endpoints, ServiceScope.SINGLETON)

    # Register backward compatibility services
    register_service("legacy_analytics", create_legacy_analytics_service, ServiceScope.SINGLETON)
    register_service("legacy_real_time_analytics", create_legacy_real_time_analytics_service, ServiceScope.SINGLETON)

    logger.info("Analytics services registered successfully")

# Convenience Functions for Common Usage Patterns

def get_analytics_interface() -> Optional[Any]:
    """Get the analytics query interface"""
    try:
        ensure_services_registered()
        from .service_registry import get_service
        return get_service("analytics_query_interface")
    except Exception as e:
        logger.error(f"Failed to get analytics interface: {e}")
        return None

def get_session_reporter() -> Optional[Any]:
    """Get the session summary reporter"""
    try:
        from .service_registry import get_service
        return get_service("session_summary_reporter")
    except Exception as e:
        logger.error(f"Failed to get session reporter: {e}")
        return None

def get_analytics_router() -> Optional[Any]:
    """Get the analytics router for real-time functionality"""
    try:
        from .service_registry import get_service
        return get_service("analytics_endpoints")
    except Exception as e:
        logger.error(f"Failed to get analytics router: {e}")
        return None

# Migration Helper Functions

def migrate_analytics_service_usage(legacy_service_name: str) -> Optional[Any]:
    """
    Helper function to migrate from legacy analytics services to new ones.

    Args:
        legacy_service_name: Name of the legacy service being migrated

    Returns:
        Appropriate new service instance
    """
    migration_map = {
        "AnalyticsService": get_analytics_interface,
        "RealTimeAnalyticsService": get_analytics_router,
        "analytics": get_analytics_interface,
        "real_time_analytics": get_analytics_router,
    }

    if legacy_service_name in migration_map:
        logger.info(f"Migrating {legacy_service_name} to new Week 10 analytics")
        return migration_map[legacy_service_name]()

    logger.warning(f"No migration path found for {legacy_service_name}")
    return None

# Auto-registration disabled to prevent circular imports during module loading
# Services will be registered on first access
_services_registered = False

def ensure_services_registered():
    """Ensure analytics services are registered (lazy registration)"""
    global _services_registered
    if not _services_registered:
        try:
            register_all_analytics_services()
            _services_registered = True
            logger.debug("Analytics services registered on first access")
        except Exception as e:
            logger.error(f"Failed to register analytics services: {e}")
            raise
