"""Health Monitor Protocol - Unified Interface for Health Monitoring

Provides a simplified, unified protocol for health monitoring that
focuses on plugin registration and core health checking functionality
needed for component integration.
"""

from typing import Protocol, Dict, Any, Optional, Callable, List
from enum import Enum


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult:
    """Standardized health check result"""
    
    def __init__(
        self,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        check_name: str = "",
        duration_ms: float = 0.0
    ):
        self.status = status
        self.message = message
        self.details = details or {}
        self.check_name = check_name
        self.duration_ms = duration_ms


class HealthMonitorProtocol(Protocol):
    """
    Unified protocol for health monitoring operations.
    
    This protocol provides a plugin-based health monitoring system
    that allows registration of custom health checkers and provides
    unified health reporting across system components.
    """

    async def check_health(
        self,
        component_name: Optional[str] = None,
        include_details: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """
        Perform health checks on registered components.
        
        Args:
            component_name: Specific component to check, None for all
            include_details: Whether to include detailed health information
            
        Returns:
            Dictionary mapping component names to health results
        """
        ...

    def register_checker(
        self,
        name: str,
        checker: Callable[[], Any],
        timeout: float = 30.0,
        critical: bool = False
    ) -> None:
        """
        Register a health checker function.
        
        Args:
            name: Unique name for the health checker
            checker: Callable that performs the health check
            timeout: Timeout for the health check in seconds
            critical: Whether this check is critical for overall health
        """
        ...

    def unregister_checker(self, name: str) -> bool:
        """
        Unregister a health checker.
        
        Args:
            name: Name of the checker to remove
            
        Returns:
            True if checker was removed, False if not found
        """
        ...

    def get_registered_checkers(self) -> List[str]:
        """
        Get list of registered health checker names.
        
        Returns:
            List of registered checker names
        """
        ...

    async def get_overall_health(self) -> HealthCheckResult:
        """
        Get overall system health status.
        
        Returns:
            Combined health result for the entire system
        """
        ...

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health monitoring summary and statistics.
        
        Returns:
            Dictionary containing health monitoring metrics
        """
        ...