"""Unified Monitoring Facade.

The main facade that consolidates all monitoring functionality into a single,
clean interface following the facade pattern and clean architecture principles.

Integrates all decomposed monitoring services:
- HealthCheckService: Component health monitoring
- MetricsCollectorService: Metrics collection and processing  
- AlertingService: Alert management and notifications
- HealthReporterService: Health reporting and dashboards
- CacheMonitoringService: Cache performance monitoring
- MonitoringOrchestratorService: Cross-service coordination
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from prompt_improver.core.protocols.health_protocol import (
    HealthMonitorProtocol,
    HealthCheckResult as ProtocolHealthCheckResult,
    HealthStatus as ProtocolHealthStatus,
)
from prompt_improver.database import ManagerMode

from .alerting_service import AlertingService
from .cache_monitoring_service import CacheMonitoringService
from .health_reporter_service import HealthReporterService
from .metrics_collector_service import MetricsCollectorService
from .monitoring_orchestrator_service import MonitoringOrchestratorService
from .protocols import (
    HealthCheckComponentProtocol,
    UnifiedMonitoringFacadeProtocol,
)
from .repository import MonitoringRepository
from .services import HealthCheckService
from .types import (
    HealthCheckResult,
    HealthStatus,
    MetricPoint,
    MetricType,
    MonitoringConfig,
    SystemHealthSummary,
)

logger = logging.getLogger(__name__)


class UnifiedMonitoringFacade:
    """Unified facade for all monitoring operations.
    
    Consolidates all monitoring functionality into a single, clean interface:
    - Health checking for all system components
    - Metrics collection and processing
    - Alert management and notifications
    - Health reporting and dashboards
    - Cache performance monitoring
    - Cross-service coordination
    
    Now uses decomposed services (all <500 lines) with orchestrated coordination
    while maintaining the same external interface for backwards compatibility.
    """
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        manager_mode: ManagerMode = ManagerMode.HIGH_AVAILABILITY,
    ):
        self.config = config or MonitoringConfig()
        
        # Initialize repository
        self.repository = MonitoringRepository(manager_mode=manager_mode)
        
        # Initialize orchestrator service (coordinates all other services)
        self.orchestrator = MonitoringOrchestratorService(
            config=self.config,
            repository=self.repository,
        )
        
        # Get references to individual services for direct access
        self.health_service = self.orchestrator.get_health_service()
        self.metrics_service = self.orchestrator.get_metrics_collector()
        self.alerting_service = self.orchestrator.get_alerting_service()
        self.health_reporter = self.orchestrator.get_health_reporter()
        self.cache_monitor = self.orchestrator.get_cache_monitor()
        
        # Cache for performance (maintained for backwards compatibility)
        self._last_health_summary: Optional[SystemHealthSummary] = None
        self._last_health_check_time: Optional[float] = None
        
        # Facade state
        self._is_started = False
        
        logger.info("UnifiedMonitoringFacade initialized with decomposed services")
    
    async def start_monitoring(self) -> None:
        """Start all monitoring services and orchestration."""
        if self._is_started:
            logger.warning("Monitoring is already started")
            return
        
        try:
            await self.orchestrator.start_orchestration()
            self._is_started = True
            logger.info("Unified monitoring started successfully")
        except Exception as e:
            logger.error(f"Failed to start unified monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring services and orchestration."""
        if not self._is_started:
            return
        
        try:
            await self.orchestrator.stop_orchestration()
            self._is_started = False
            logger.info("Unified monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping unified monitoring: {e}")
    
    def record_cache_operation(
        self,
        operation: str,
        cache_level: str,
        hit: bool,
        duration_ms: float,
        key: Optional[str] = None,
        value_size: int = 0,
    ) -> None:
        """Record cache operation for monitoring (migrated from cache/ components)."""
        try:
            from prompt_improver.core.types import CacheLevel
            
            # Convert string to CacheLevel enum
            level_map = {
                "l1": CacheLevel.L1,
                "l2": CacheLevel.L2,
                "l3": CacheLevel.L3,
                "all": CacheLevel.ALL,
            }
            cache_level_enum = level_map.get(cache_level.lower(), CacheLevel.L1)
            
            self.cache_monitor.record_cache_operation(
                operation=operation,
                cache_level=cache_level_enum,
                hit=hit,
                duration_ms=duration_ms,
                key=key,
                value_size=value_size,
            )
        except Exception as e:
            logger.error(f"Failed to record cache operation: {e}")
    
    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report (migrated from cache/ components)."""
        return await self.cache_monitor.get_cache_performance_report()
    
    async def get_comprehensive_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all monitoring services."""
        return await self.orchestrator.get_comprehensive_status()
    
    async def get_system_health(self) -> SystemHealthSummary:
        """Get overall system health status."""
        start_time = time.time()
        
        try:
            # Check if we can use cached results
            if self._should_use_cached_health():
                logger.debug("Using cached health results")
                return self._last_health_summary
            
            # Run comprehensive health checks
            summary = await self.health_service.run_all_checks()
            
            # Update cache
            self._last_health_summary = summary
            self._last_health_check_time = time.time()
            
            # Record performance metric
            duration_ms = (time.time() - start_time) * 1000
            self.record_custom_metric(
                "monitoring.health_check.duration_ms",
                duration_ms,
                tags={"operation": "system_health"}
            )
            
            logger.info(
                f"System health check completed: {summary.overall_status.value} "
                f"({summary.healthy_components}/{summary.total_components} healthy) "
                f"in {duration_ms:.2f}ms"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            
            # Record error metric
            self.record_custom_metric(
                "monitoring.health_check.errors",
                1.0,
                tags={"operation": "system_health", "error": str(e)[:100]}
            )
            
            # Return degraded status
            return SystemHealthSummary(
                overall_status=HealthStatus.UNKNOWN,
                total_components=len(self.health_service.get_registered_components()),
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0,
                unknown_components=len(self.health_service.get_registered_components()),
                check_duration_ms=(time.time() - start_time) * 1000,
            )
    
    async def check_component_health(self, component_name: str) -> HealthCheckResult:
        """Check health of specific component."""
        start_time = time.time()
        
        try:
            result = await self.health_service.run_component_check(component_name)
            
            # Record performance metric
            duration_ms = (time.time() - start_time) * 1000
            self.record_custom_metric(
                "monitoring.component_health_check.duration_ms",
                duration_ms,
                tags={"component": component_name, "status": result.status.value}
            )
            
            logger.debug(f"Component health check for {component_name}: {result.status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Component health check failed for {component_name}: {e}")
            
            # Record error metric
            self.record_custom_metric(
                "monitoring.component_health_check.errors",
                1.0,
                tags={"component": component_name, "error": str(e)[:100]}
            )
            
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                component_name=component_name,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    async def collect_all_metrics(self) -> List[MetricPoint]:
        """Collect all available metrics."""
        start_time = time.time()
        
        try:
            metrics = await self.metrics_service.get_all_metrics()
            
            # Record collection metric
            duration_ms = (time.time() - start_time) * 1000
            self.record_custom_metric(
                "monitoring.metrics_collection.duration_ms",
                duration_ms,
                tags={"metrics_count": str(len(metrics))}
            )
            
            logger.debug(f"Collected {len(metrics)} metrics in {duration_ms:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            
            # Record error metric
            self.record_custom_metric(
                "monitoring.metrics_collection.errors",
                1.0,
                tags={"error": str(e)[:100]}
            )
            
            return []
    
    def record_custom_metric(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record custom application metric."""
        try:
            metric = MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                tags=tags or {},
                description=f"Custom metric: {name}",
            )
            
            self.metrics_service.record_metric(metric)
            
        except Exception as e:
            logger.warning(f"Failed to record custom metric {name}: {e}")
    
    def register_health_checker(self, checker: HealthCheckComponentProtocol) -> None:
        """Register custom health checker component."""
        try:
            self.health_service.register_component(checker)
            
            # Clear cached results since we have a new component
            self._invalidate_health_cache()
            
            logger.info(f"Registered custom health checker: {checker.get_component_name()}")
            
        except Exception as e:
            logger.error(f"Failed to register health checker: {e}")
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            # Get health summary
            health_summary = await self.get_system_health()
            
            # Get metrics
            metrics = await self.collect_all_metrics()
            
            # Get registered components
            components = self.health_service.get_registered_components()
            
            return {
                "health": {
                    "overall_status": health_summary.overall_status.value,
                    "total_components": health_summary.total_components,
                    "healthy_components": health_summary.healthy_components,
                    "degraded_components": health_summary.degraded_components,
                    "unhealthy_components": health_summary.unhealthy_components,
                    "health_percentage": health_summary.health_percentage,
                    "check_duration_ms": health_summary.check_duration_ms,
                    "critical_issues": health_summary.get_critical_issues(),
                },
                "metrics": {
                    "total_metrics": len(metrics),
                    "collection_enabled": self.config.metrics_collection_enabled,
                    "retention_hours": self.config.metrics_retention_hours,
                },
                "components": {
                    "registered_count": len(components),
                    "component_names": components,
                },
                "configuration": {
                    "health_check_timeout": self.config.health_check_timeout_seconds,
                    "parallel_execution": self.config.health_check_parallel_enabled,
                    "max_concurrent_checks": self.config.max_concurrent_checks,
                    "cache_results_seconds": self.config.cache_health_results_seconds,
                },
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def cleanup_old_monitoring_data(self) -> int:
        """Clean up old monitoring data."""
        try:
            cleaned_count = await self.repository.cleanup_old_data(
                self.config.metrics_retention_hours
            )
            
            logger.info(f"Cleaned up {cleaned_count} old monitoring records")
            
            # Record cleanup metric
            self.record_custom_metric(
                "monitoring.cleanup.records_removed",
                float(cleaned_count),
                tags={"retention_hours": str(self.config.metrics_retention_hours)}
            )
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old monitoring data: {e}")
            return 0
    
    def _should_use_cached_health(self) -> bool:
        """Check if we should use cached health results."""
        if not self._last_health_summary or not self._last_health_check_time:
            return False
        
        cache_age = time.time() - self._last_health_check_time
        return cache_age < self.config.cache_health_results_seconds
    
    def _invalidate_health_cache(self) -> None:
        """Invalidate cached health results."""
        self._last_health_summary = None
        self._last_health_check_time = None
    
    # Compatibility methods for existing HealthMonitorProtocol
    async def check_health(
        self, component_name: Optional[str] = None, include_details: bool = True
    ) -> Dict[str, ProtocolHealthCheckResult]:
        """Health check method compatible with HealthMonitorProtocol."""
        try:
            if component_name:
                # Check specific component
                result = await self.check_component_health(component_name)
                return {component_name: self._convert_to_protocol_result(result)}
            else:
                # Check all components
                summary = await self.get_system_health()
                protocol_results = {}
                
                for name, result in summary.component_results.items():
                    protocol_results[name] = self._convert_to_protocol_result(result)
                
                return protocol_results
                
        except Exception as e:
            logger.error(f"Health check compatibility method failed: {e}")
            return {}
    
    def register_checker(
        self,
        name: str,
        checker: Any,  # Callable that performs health check
        timeout: float = 30.0,
        critical: bool = False,
    ) -> None:
        """Register checker method compatible with HealthMonitorProtocol."""
        # This would need a wrapper to convert callable to HealthCheckComponentProtocol
        # For now, log the registration attempt
        logger.info(f"Legacy checker registration for {name} (use register_health_checker instead)")
    
    def unregister_checker(self, name: str) -> bool:
        """Unregister checker method compatible with HealthMonitorProtocol."""
        return self.health_service.unregister_component(name)
    
    def get_registered_checkers(self) -> List[str]:
        """Get registered checkers method compatible with HealthMonitorProtocol."""
        return self.health_service.get_registered_components()
    
    async def get_overall_health(self) -> ProtocolHealthCheckResult:
        """Get overall health method compatible with HealthMonitorProtocol."""
        try:
            summary = await self.get_system_health()
            
            # Convert to protocol result
            status_mapping = {
                HealthStatus.HEALTHY: ProtocolHealthStatus.HEALTHY,
                HealthStatus.DEGRADED: ProtocolHealthStatus.DEGRADED,
                HealthStatus.UNHEALTHY: ProtocolHealthStatus.UNHEALTHY,
                HealthStatus.UNKNOWN: ProtocolHealthStatus.UNKNOWN,
            }
            
            return ProtocolHealthCheckResult(
                status=status_mapping[summary.overall_status],
                message=f"{summary.healthy_components}/{summary.total_components} components healthy",
                details={
                    "total_components": summary.total_components,
                    "healthy_components": summary.healthy_components,
                    "degraded_components": summary.degraded_components,
                    "unhealthy_components": summary.unhealthy_components,
                    "health_percentage": summary.health_percentage,
                    "critical_issues": summary.get_critical_issues(),
                },
                check_name="system_overall",
                duration_ms=summary.check_duration_ms,
            )
            
        except Exception as e:
            logger.error(f"Overall health compatibility method failed: {e}")
            return ProtocolHealthCheckResult(
                status=ProtocolHealthStatus.UNKNOWN,
                message=f"Health check failed: {str(e)}",
                check_name="system_overall",
                duration_ms=0.0,
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary method compatible with HealthMonitorProtocol."""
        # Return synchronous summary - would need async context for full data
        return {
            "registered_components": len(self.health_service.get_registered_components()),
            "components": self.health_service.get_registered_components(),
            "config": {
                "timeout_seconds": self.config.health_check_timeout_seconds,
                "parallel_enabled": self.config.health_check_parallel_enabled,
                "max_concurrent": self.config.max_concurrent_checks,
            },
        }
    
    def _convert_to_protocol_result(self, result: HealthCheckResult) -> ProtocolHealthCheckResult:
        """Convert internal HealthCheckResult to protocol HealthCheckResult."""
        status_mapping = {
            HealthStatus.HEALTHY: ProtocolHealthStatus.HEALTHY,
            HealthStatus.DEGRADED: ProtocolHealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY: ProtocolHealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN: ProtocolHealthStatus.UNKNOWN,
        }
        
        return ProtocolHealthCheckResult(
            status=status_mapping[result.status],
            message=result.message,
            details=result.details,
            check_name=result.component_name,
            duration_ms=result.response_time_ms,
        )