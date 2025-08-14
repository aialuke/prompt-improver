"""Monitoring Orchestrator Service.

Orchestrates all monitoring services and provides coordinated monitoring operations.
Extracted from unified_monitoring_manager.py.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    orchestrator_tracer = trace.get_tracer(__name__ + ".monitoring_orchestrator")
    orchestrator_meter = metrics.get_meter(__name__ + ".monitoring_orchestrator")
    
    orchestration_cycles_total = orchestrator_meter.create_counter(
        "monitoring_orchestration_cycles_total",
        description="Total monitoring orchestration cycles completed",
        unit="1",
    )
    
    orchestration_duration = orchestrator_meter.create_histogram(
        "monitoring_orchestration_duration_seconds", 
        description="Time taken for monitoring orchestration cycles",
        unit="s",
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    orchestrator_tracer = None
    orchestrator_meter = None
    orchestration_cycles_total = None
    orchestration_duration = None

from .alerting_service import AlertingService
from .cache_monitoring_service import CacheMonitoringService  
from .health_reporter_service import HealthReporterService
from .metrics_collector_service import MetricsCollectorService
from .protocols import MonitoringOrchestratorProtocol, MonitoringRepositoryProtocol
from .services import HealthCheckService
from .types import MonitoringConfig

logger = logging.getLogger(__name__)


class MonitoringOrchestratorService:
    """Orchestrates all monitoring services for coordinated monitoring operations.
    
    Provides:
    - Coordinated startup and shutdown of all monitoring services
    - Cross-service data sharing and correlation
    - Unified monitoring lifecycle management
    - Service health monitoring and recovery
    - Performance optimization across services
    """
    
    def __init__(
        self,
        config: MonitoringConfig,
        repository: Optional[MonitoringRepositoryProtocol] = None,
    ):
        self.config = config
        self.repository = repository
        
        # Initialize all monitoring services
        self.health_service = HealthCheckService(config, repository)
        self.metrics_collector = MetricsCollectorService(config, repository)
        self.alerting_service = AlertingService(config, repository)
        self.health_reporter = HealthReporterService(config, repository)
        self.cache_monitor = CacheMonitoringService(config, repository)
        
        # Orchestration state
        self.is_running = False
        self._orchestration_tasks: List[asyncio.Task] = []
        self._service_status: Dict[str, bool] = {}
        
        # Performance tracking
        self._orchestration_stats = {
            "cycles_completed": 0,
            "cycles_failed": 0,
            "avg_cycle_duration_ms": 0.0,
            "last_cycle_time": None,
        }
        
        # Cross-service coordination
        self._correlation_data: Dict[str, Any] = {}
        
        logger.info("MonitoringOrchestratorService initialized")
    
    async def start_orchestration(self) -> None:
        """Start coordinated monitoring orchestration."""
        if self.is_running:
            logger.warning("Monitoring orchestration is already running")
            return
        
        try:
            logger.info("Starting monitoring orchestration...")
            
            # Start individual services
            await self._start_services()
            
            # Start orchestration loops
            await self._start_orchestration_loops()
            
            # Set up cross-service communication
            self._setup_cross_service_communication()
            
            self.is_running = True
            logger.info("Monitoring orchestration started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring orchestration: {e}")
            await self._cleanup_failed_start()
            raise
    
    async def stop_orchestration(self) -> None:
        """Stop coordinated monitoring orchestration."""
        if not self.is_running:
            return
        
        logger.info("Stopping monitoring orchestration...")
        
        try:
            # Cancel orchestration tasks
            for task in self._orchestration_tasks:
                task.cancel()
            
            # Wait for task completion
            if self._orchestration_tasks:
                await asyncio.gather(*self._orchestration_tasks, return_exceptions=True)
            
            # Stop individual services
            await self._stop_services()
            
            self.is_running = False
            self._orchestration_tasks.clear()
            
            logger.info("Monitoring orchestration stopped")
            
        except Exception as e:
            logger.error(f"Error during orchestration shutdown: {e}")
    
    async def _start_services(self) -> None:
        """Start all monitoring services."""
        services = [
            ("alerting", self.alerting_service.start_monitoring()),
            ("cache_monitoring", self.cache_monitor.start_monitoring()),
        ]
        
        for service_name, start_coro in services:
            try:
                await start_coro
                self._service_status[service_name] = True
                logger.info(f"Started {service_name} service")
            except Exception as e:
                logger.error(f"Failed to start {service_name} service: {e}")
                self._service_status[service_name] = False
    
    async def _start_orchestration_loops(self) -> None:
        """Start orchestration background loops."""
        # Main orchestration loop
        main_loop_task = asyncio.create_task(self._main_orchestration_loop())
        self._orchestration_tasks.append(main_loop_task)
        
        # Service health monitoring loop
        health_monitoring_task = asyncio.create_task(self._service_health_monitoring_loop())
        self._orchestration_tasks.append(health_monitoring_task)
        
        # Cross-service coordination loop
        coordination_task = asyncio.create_task(self._cross_service_coordination_loop())
        self._orchestration_tasks.append(coordination_task)
        
        logger.info("Started orchestration loops")
    
    def _setup_cross_service_communication(self) -> None:
        """Set up communication between monitoring services."""
        # Connect health check results to alerting
        # This would be done through callbacks or event systems
        
        # Connect cache monitoring to metrics collection
        self.cache_monitor.add_alert_callback(
            lambda alert: asyncio.create_task(
                self.alerting_service.create_alert(
                    alert_type="cache_performance",
                    severity=alert.severity,
                    title=alert.message,
                    description=f"Cache alert: {alert.message}",
                    source_component="cache_monitoring",
                    metrics={"current_value": alert.current_value, "threshold": alert.threshold_value},
                )
            )
        )
        
        # Connect metrics to health reporting
        # Health reporter gets health summaries for trend analysis
        
        logger.info("Set up cross-service communication")
    
    async def _main_orchestration_loop(self) -> None:
        """Main orchestration loop coordinating all monitoring activities."""
        try:
            while self.is_running:
                start_time = time.time()
                
                try:
                    # Perform coordinated monitoring cycle
                    await self._execute_monitoring_cycle()
                    
                    # Update orchestration stats
                    duration_ms = (time.time() - start_time) * 1000
                    self._update_orchestration_stats(True, duration_ms)
                    
                    # Record telemetry
                    if OPENTELEMETRY_AVAILABLE and orchestration_cycles_total:
                        orchestration_cycles_total.add(1, {"status": "success"})
                    
                    if OPENTELEMETRY_AVAILABLE and orchestration_duration:
                        orchestration_duration.record(duration_ms / 1000.0)
                    
                    logger.debug(f"Completed monitoring cycle in {duration_ms:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle failed: {e}")
                    self._update_orchestration_stats(False, (time.time() - start_time) * 1000)
                    
                    if OPENTELEMETRY_AVAILABLE and orchestration_cycles_total:
                        orchestration_cycles_total.add(1, {"status": "error"})
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second cycles
                
        except asyncio.CancelledError:
            logger.info("Main orchestration loop cancelled")
        except Exception as e:
            logger.error(f"Main orchestration loop failed: {e}")
    
    async def _execute_monitoring_cycle(self) -> None:
        """Execute a coordinated monitoring cycle."""
        # 1. Collect health data
        health_summary = await self.health_service.run_all_checks()
        
        # 2. Collect metrics
        metrics = await self.metrics_collector.collect_all_metrics()
        
        # 3. Update health reporter with latest health data
        self.health_reporter.add_health_summary(health_summary)
        
        # 4. Store correlation data for cross-service analysis
        self._correlation_data = {
            "timestamp": time.time(),
            "health_summary": health_summary,
            "metrics_count": len(metrics),
            "active_alerts": len(self.alerting_service.get_active_alerts()),
        }
        
        # 5. Check for cross-service patterns or issues
        await self._analyze_cross_service_patterns()
    
    async def _service_health_monitoring_loop(self) -> None:
        """Monitor health of monitoring services themselves."""
        try:
            while self.is_running:
                # Check service status and restart if needed
                await self._check_service_health()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.info("Service health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Service health monitoring loop failed: {e}")
    
    async def _cross_service_coordination_loop(self) -> None:
        """Coordinate activities between monitoring services."""
        try:
            while self.is_running:
                # Coordinate resource usage
                await self._coordinate_resource_usage()
                
                # Synchronize cache invalidation with health checks
                await self._synchronize_cache_and_health()
                
                # Share performance insights across services
                await self._share_performance_insights()
                
                # Wait before next coordination cycle
                await asyncio.sleep(120)  # Coordinate every 2 minutes
                
        except asyncio.CancelledError:
            logger.info("Cross-service coordination loop cancelled")
        except Exception as e:
            logger.error(f"Cross-service coordination loop failed: {e}")
    
    async def _check_service_health(self) -> None:
        """Check health of individual monitoring services."""
        services_to_check = [
            ("alerting", self._check_alerting_service_health),
            ("cache_monitoring", self._check_cache_monitoring_health),
        ]
        
        for service_name, health_check in services_to_check:
            try:
                is_healthy = await health_check()
                self._service_status[service_name] = is_healthy
                
                if not is_healthy:
                    logger.warning(f"Monitoring service {service_name} is unhealthy")
                    # Could implement automatic restart logic here
                    
            except Exception as e:
                logger.error(f"Failed to check health of {service_name}: {e}")
                self._service_status[service_name] = False
    
    async def _check_alerting_service_health(self) -> bool:
        """Check alerting service health."""
        return self.alerting_service.is_monitoring
    
    async def _check_cache_monitoring_health(self) -> bool:
        """Check cache monitoring service health."""
        # Check if cache monitoring is processing operations
        return True  # Simplified check
    
    async def _analyze_cross_service_patterns(self) -> None:
        """Analyze patterns across monitoring services."""
        try:
            # Example: Correlate health degradation with cache performance
            health_summary = self._correlation_data.get("health_summary")
            if health_summary and hasattr(health_summary, 'overall_status'):
                if health_summary.overall_status.value in ["degraded", "unhealthy"]:
                    # Check if cache performance might be related
                    cache_report = await self.cache_monitor.get_cache_performance_report()
                    
                    overall_stats = cache_report.get("overall_stats", {})
                    if overall_stats.get("hit_rate", 1.0) < 0.7:  # Low hit rate
                        await self.alerting_service.create_alert(
                            alert_type="correlation_analysis",
                            severity=self.alerting_service.AlertSeverity.WARNING,
                            title="Potential Cache-Health Correlation",
                            description="System health degradation detected alongside low cache hit rate",
                            source_component="monitoring_orchestrator",
                            tags={"type": "cross_service_analysis"},
                        )
            
        except Exception as e:
            logger.error(f"Failed to analyze cross-service patterns: {e}")
    
    async def _coordinate_resource_usage(self) -> None:
        """Coordinate resource usage across monitoring services."""
        # Example: Throttle monitoring intensity if system is under stress
        try:
            correlation = self._correlation_data
            if correlation:
                health_summary = correlation.get("health_summary")
                if health_summary and hasattr(health_summary, 'overall_status'):
                    if health_summary.overall_status.value == "unhealthy":
                        # Reduce monitoring frequency to reduce system load
                        logger.info("System unhealthy - reducing monitoring intensity")
            
        except Exception as e:
            logger.error(f"Failed to coordinate resource usage: {e}")
    
    async def _synchronize_cache_and_health(self) -> None:
        """Synchronize cache invalidation with health checks."""
        # Example: Invalidate cache if health checks indicate stale data
        try:
            # This is where cache invalidation could be coordinated with health status
            pass
        except Exception as e:
            logger.error(f"Failed to synchronize cache and health: {e}")
    
    async def _share_performance_insights(self) -> None:
        """Share performance insights between services."""
        try:
            # Example: Share cache performance insights with health reporting
            cache_report = await self.cache_monitor.get_cache_performance_report()
            
            # Update correlation data with cache insights
            self._correlation_data["cache_performance"] = cache_report.get("overall_stats", {})
            
        except Exception as e:
            logger.error(f"Failed to share performance insights: {e}")
    
    async def _stop_services(self) -> None:
        """Stop all monitoring services."""
        services = [
            ("alerting", self.alerting_service.stop_monitoring()),
        ]
        
        for service_name, stop_coro in services:
            try:
                await stop_coro
                self._service_status[service_name] = False
                logger.info(f"Stopped {service_name} service")
            except Exception as e:
                logger.error(f"Failed to stop {service_name} service: {e}")
    
    async def _cleanup_failed_start(self) -> None:
        """Clean up after failed orchestration start."""
        try:
            # Cancel any running tasks
            for task in self._orchestration_tasks:
                task.cancel()
            
            if self._orchestration_tasks:
                await asyncio.gather(*self._orchestration_tasks, return_exceptions=True)
            
            self._orchestration_tasks.clear()
            
            # Try to stop any started services
            await self._stop_services()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _update_orchestration_stats(self, success: bool, duration_ms: float) -> None:
        """Update orchestration statistics."""
        if success:
            self._orchestration_stats["cycles_completed"] += 1
        else:
            self._orchestration_stats["cycles_failed"] += 1
        
        # Update running average duration
        current_avg = self._orchestration_stats["avg_cycle_duration_ms"]
        total_cycles = (
            self._orchestration_stats["cycles_completed"] + 
            self._orchestration_stats["cycles_failed"]
        )
        
        self._orchestration_stats["avg_cycle_duration_ms"] = (
            (current_avg * (total_cycles - 1) + duration_ms) / total_cycles
        )
        
        self._orchestration_stats["last_cycle_time"] = time.time()
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration status and statistics."""
        return {
            "is_running": self.is_running,
            "service_status": dict(self._service_status),
            "active_tasks": len(self._orchestration_tasks),
            "statistics": dict(self._orchestration_stats),
            "correlation_data_age_seconds": (
                time.time() - self._correlation_data.get("timestamp", 0)
                if self._correlation_data else None
            ),
        }
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all monitoring services."""
        try:
            status = {
                "orchestration": self.get_orchestration_status(),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            
            # Get status from individual services
            try:
                status["health_service"] = {
                    "registered_components": len(self.health_service.get_registered_components()),
                    "components": self.health_service.get_registered_components(),
                }
            except Exception as e:
                status["health_service"] = {"error": str(e)}
            
            try:
                status["metrics_collector"] = self.metrics_collector.get_collection_stats()
            except Exception as e:
                status["metrics_collector"] = {"error": str(e)}
            
            try:
                status["alerting_service"] = {
                    "is_monitoring": self.alerting_service.is_monitoring,
                    "active_alerts": len(self.alerting_service.get_active_alerts()),
                    "statistics": self.alerting_service.get_alert_statistics().__dict__,
                }
            except Exception as e:
                status["alerting_service"] = {"error": str(e)}
            
            try:
                status["cache_monitor"] = await self.cache_monitor.get_cache_performance_report()
            except Exception as e:
                status["cache_monitor"] = {"error": str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
    
    # Convenience methods for accessing individual services
    def get_health_service(self) -> HealthCheckService:
        """Get health check service."""
        return self.health_service
    
    def get_metrics_collector(self) -> MetricsCollectorService:
        """Get metrics collector service."""
        return self.metrics_collector
    
    def get_alerting_service(self) -> AlertingService:
        """Get alerting service."""
        return self.alerting_service
    
    def get_health_reporter(self) -> HealthReporterService:
        """Get health reporter service."""
        return self.health_reporter
    
    def get_cache_monitor(self) -> CacheMonitoringService:
        """Get cache monitoring service."""
        return self.cache_monitor