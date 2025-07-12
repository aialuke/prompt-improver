"""Unified Health Service implementing Composite pattern for APES health monitoring.
PHASE 3: Health Check Consolidation - Main Service Aggregator
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

from .base import HealthChecker, HealthResult, HealthStatus, AggregatedHealthResult
from .checkers import (
    DatabaseHealthChecker,
    MCPServerHealthChecker, 
    AnalyticsServiceHealthChecker,
    MLServiceHealthChecker,
    SystemResourcesHealthChecker
)

# Lazy import to avoid circular dependency
try:
    from .checkers import QueueHealthChecker
except ImportError:
    # Use lazy import to avoid circular dependency issues
    QueueHealthChecker = None
from .metrics import instrument_health_check


class HealthService:
    """Unified health service implementing Composite pattern"""
    
    def __init__(self, checkers: Optional[List[HealthChecker]] = None):
        """Initialize with default or custom health checkers"""
        if checkers is None:
            self.checkers = [
                DatabaseHealthChecker(),
                MCPServerHealthChecker(),
                AnalyticsServiceHealthChecker(),
                MLServiceHealthChecker(),
                SystemResourcesHealthChecker()
            ]
            
            # Add QueueHealthChecker if available (avoid circular import)
            if QueueHealthChecker is not None:
                try:
                    self.checkers.append(QueueHealthChecker())
                except Exception as e:
                    # Log the error but continue without queue health checker
                    print(f"Warning: Could not initialize QueueHealthChecker: {e}")
        else:
            self.checkers = checkers
        
        # Create checker mapping for easy access
        self.checker_map = {checker.name: checker for checker in self.checkers}
    
    @instrument_health_check("aggregated")
    async def run_health_check(self, parallel: bool = True) -> AggregatedHealthResult:
        """Run all health checks and return aggregated result"""
        start_time = time.time()
        
        if parallel:
            # Run all checks in parallel for better performance
            tasks = [checker.check() for checker in self.checkers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run checks sequentially
            results = []
            for checker in self.checkers:
                try:
                    result = await checker.check()
                    results.append(result)
                except Exception as e:
                    # Create error result if checker fails
                    error_result = HealthResult(
                        status=HealthStatus.FAILED,
                        component=checker.name,
                        error=str(e),
                        message=f"Health check failed: {str(e)}"
                    )
                    results.append(error_result)
        
        # Process results and handle any exceptions
        check_results = {}
        for i, result in enumerate(results):
            checker = self.checkers[i]
            
            if isinstance(result, Exception):
                # Convert exception to failed health result
                check_results[checker.name] = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error=str(result),
                    message=f"Health check failed: {str(result)}"
                )
            elif isinstance(result, HealthResult):
                check_results[checker.name] = result
            else:
                # Unexpected result type
                check_results[checker.name] = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error="Invalid result type",
                    message="Health check returned invalid result"
                )
        
        # Calculate overall status using hierarchical logic
        overall_status = self._calculate_overall_status(check_results)
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        return AggregatedHealthResult(
            overall_status=overall_status,
            checks=check_results,
            timestamp=datetime.now()
        )
    
    async def run_specific_check(self, component_name: str) -> HealthResult:
        """Run health check for a specific component"""
        if component_name not in self.checker_map:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=component_name,
                error="Unknown component",
                message=f"No health checker found for component: {component_name}"
            )
        
        checker = self.checker_map[component_name]
        try:
            return await checker.check()
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=component_name,
                error=str(e),
                message=f"Health check failed: {str(e)}"
            )
    
    def _calculate_overall_status(self, results: Dict[str, HealthResult]) -> HealthStatus:
        """Calculate overall health status from individual results"""
        if not results:
            return HealthStatus.FAILED
        
        statuses = [result.status for result in results.values()]
        
        # Hierarchical status calculation: failed > warning > healthy
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_available_checks(self) -> List[str]:
        """Get list of available health check components"""
        return list(self.checker_map.keys())
    
    async def get_health_summary(self, include_details: bool = False) -> Dict:
        """Get health summary in dictionary format for API responses"""
        result = await self.run_health_check()
        
        summary = {
            "overall_status": result.overall_status.value,
            "timestamp": result.timestamp.isoformat(),
            "checks": {}
        }
        
        for component, check_result in result.checks.items():
            check_summary = {
                "status": check_result.status.value,
                "message": check_result.message
            }
            
            if check_result.response_time_ms is not None:
                check_summary["response_time_ms"] = check_result.response_time_ms
            
            if check_result.error:
                check_summary["error"] = check_result.error
            
            if include_details and check_result.details:
                check_summary["details"] = check_result.details
            
            summary["checks"][component] = check_summary
        
        if result.failed_checks:
            summary["failed_checks"] = result.failed_checks
        
        if result.warning_checks:
            summary["warning_checks"] = result.warning_checks
        
        return summary
    
    def add_checker(self, checker: HealthChecker) -> None:
        """Add a new health checker to the service"""
        self.checkers.append(checker)
        self.checker_map[checker.name] = checker
    
    def remove_checker(self, component_name: str) -> bool:
        """Remove a health checker from the service"""
        if component_name not in self.checker_map:
            return False
        
        checker = self.checker_map[component_name]
        self.checkers.remove(checker)
        del self.checker_map[component_name]
        return True
    
    def ensure_queue_checker(self) -> bool:
        """Ensure queue health checker is available, add it if not present.
        
        Returns:
            True if queue checker is available, False otherwise
        """
        # Check if queue checker already exists
        if "queue" in self.checker_map:
            return True
            
        # Try to dynamically import and add queue checker
        try:
            from .checkers import QueueHealthChecker
            queue_checker = QueueHealthChecker()
            self.add_checker(queue_checker)
            return True
        except Exception as e:
            print(f"Warning: Could not add QueueHealthChecker: {e}")
            return False


# Global health service instance
_health_service_instance: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Get or create the global health service instance"""
    global _health_service_instance
    if _health_service_instance is None:
        _health_service_instance = HealthService()
    return _health_service_instance


def reset_health_service() -> None:
    """Reset the global health service instance (useful for testing)"""
    global _health_service_instance
    _health_service_instance = None
