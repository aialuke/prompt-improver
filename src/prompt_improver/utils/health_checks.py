"""Health Check Consolidation Module for APES
PHASE 3: Extract monitoring.py patterns into standardized health checking utilities

This module provides a simplified interface to the comprehensive health checking system,
consolidating patterns from cli.py, monitoring.py, and manager.py as specified in
duplicationtesting.md Phase 3 requirements.
"""

import time
from datetime import datetime
from typing import Any

# Import the comprehensive health system
from ..services.health import (
    AggregatedHealthResult,
    HealthResult,
    HealthStatus,
    get_health_service,
)

# For backwards compatibility and simplified access


class HealthChecker:
    """Simplified HealthChecker interface for consolidated patterns

    This class provides the standardized interface extracted from monitoring.py
    as specified in Phase 3 requirements, while delegating to the comprehensive
    health system implementation.
    """

    def __init__(self, components: list[str] | None = None):
        """Initialize health checker with optional component selection

        Args:
            components: List of component names to check. If None, checks all available.
        """
        self.health_service = get_health_service()
        self.components = components or self.health_service.get_available_checks()

    async def check_all(self, parallel: bool = True) -> dict[str, Any]:
        """Run all health checks and return standardized results

        Consolidates the patterns from monitoring.py:591-626 into standardized format.

        Args:
            parallel: Whether to run checks in parallel (default: True)

        Returns:
            Dictionary with standardized health check results matching monitoring.py format
        """
        result = await self.health_service.run_health_check(parallel=parallel)

        # Convert to the standardized format from monitoring.py patterns
        return {
            "status": result.overall_status.value,
            "timestamp": result.timestamp.isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "response_time_ms": check.response_time_ms,
                    "message": check.message,
                    **({"error": check.error} if check.error else {}),
                    **({"details": check.details} if check.details else {}),
                }
                for name, check in result.checks.items()
                if name in self.components
            },
            **({"failed_checks": result.failed_checks} if result.failed_checks else {}),
            **(
                {"warning_checks": result.warning_checks}
                if result.warning_checks
                else {}
            ),
        }

    async def check_component(self, component_name: str) -> dict[str, Any]:
        """Check specific component health

        Args:
            component_name: Name of component to check

        Returns:
            Health check result for the specific component
        """
        result = await self.health_service.run_specific_check(component_name)

        return {
            "status": result.status.value,
            "component": result.component,
            "response_time_ms": result.response_time_ms,
            "message": result.message,
            "timestamp": result.timestamp.isoformat(),
            **({"error": result.error} if result.error else {}),
            **({"details": result.details} if result.details else {}),
        }

    async def check_database_health(self) -> dict[str, Any]:
        """Database connectivity and performance check

        Extracted from monitoring.py:591-626 pattern
        """
        return await self.check_component("database")

    async def check_mcp_performance(self) -> dict[str, Any]:
        """MCP server performance check

        Extracted from monitoring.py:628-658 pattern
        """
        return await self.check_component("mcp_server")

    async def check_analytics_service(self) -> dict[str, Any]:
        """Analytics service functionality check

        Extracted from monitoring.py:660-679 pattern
        """
        return await self.check_component("analytics")

    async def check_ml_service(self) -> dict[str, Any]:
        """ML service availability check

        Extracted from monitoring.py:681-701 pattern
        """
        return await self.check_component("ml_service")

    async def check_system_resources(self) -> dict[str, Any]:
        """System resource usage check

        Extracted from monitoring.py:703-751 pattern
        """
        return await self.check_component("system_resources")

    def get_available_components(self) -> list[str]:
        """Get list of available health check components"""
        return self.health_service.get_available_checks()


# Decorator for health check components (as specified in Phase 3)
def health_check_component(component_name: str):
    """Decorator for standardized health checking

    Usage:
        @health_check_component("custom_service")
        async def check_custom_service():
            # Health check logic
            return {"status": "healthy", "message": "Service OK"}
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()

                # Ensure standardized format
                if isinstance(result, dict):
                    result.setdefault("component", component_name)
                    result.setdefault("timestamp", datetime.now().isoformat())
                    if "response_time_ms" not in result:
                        result["response_time_ms"] = (end_time - start_time) * 1000

                return result
            except Exception as e:
                end_time = time.time()
                return {
                    "status": "failed",
                    "component": component_name,
                    "error": str(e),
                    "message": f"Health check failed: {e!s}",
                    "response_time_ms": (end_time - start_time) * 1000,
                    "timestamp": datetime.now().isoformat(),
                }

        return wrapper

    return decorator


# Convenience functions for backwards compatibility
async def run_health_check() -> dict[str, Any]:
    """Run comprehensive health check

    Convenience function matching the pattern from monitoring.py HealthMonitor.run_health_check()
    """
    checker = HealthChecker()
    return await checker.check_all()


async def check_database_health() -> dict[str, Any]:
    """Check database health - extracted pattern from monitoring.py:591-626"""
    checker = HealthChecker()
    return await checker.check_database_health()


async def check_mcp_performance() -> dict[str, Any]:
    """Check MCP performance - extracted pattern from monitoring.py:628-658"""
    checker = HealthChecker()
    return await checker.check_mcp_performance()


async def check_analytics_service() -> dict[str, Any]:
    """Check analytics service - extracted pattern from monitoring.py:660-679"""
    checker = HealthChecker()
    return await checker.check_analytics_service()


async def check_ml_service() -> dict[str, Any]:
    """Check ML service - extracted pattern from monitoring.py:681-701"""
    checker = HealthChecker()
    return await checker.check_ml_service()


async def check_system_resources() -> dict[str, Any]:
    """Check system resources - extracted pattern from monitoring.py:703-751"""
    checker = HealthChecker()
    return await checker.check_system_resources()


# Export standardized interface
__all__ = [
    # Main class
    "HealthChecker",
    # Decorator
    "health_check_component",
    # Convenience functions
    "run_health_check",
    "check_database_health",
    "check_mcp_performance",
    "check_analytics_service",
    "check_ml_service",
    "check_system_resources",
    # Re-exports from health system
    "HealthStatus",
    "HealthResult",
    "AggregatedHealthResult",
    "get_health_service",
]
