"""Redis Health Manager

Unified facade for coordinating all Redis health monitoring services.
Provides centralized health management following SRE best practices with <25ms operations.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import coredis

from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry

from .alerting_service import RedisAlertingService
from .connection_monitor import RedisConnectionMonitor
from .health_checker import RedisHealthChecker
from .metrics_collector import RedisMetricsCollector
from .protocols import RedisClientProviderProtocol, RedisHealthManagerProtocol
from .recovery_service import RedisRecoveryService
from .types import AlertLevel, HealthMetrics, RecoveryAction

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Health manager metrics
HEALTH_CHECK_DURATION = _metrics_registry.get_or_create_histogram(
    "redis_health_manager_check_duration_ms",
    "Redis health manager check duration in milliseconds"
)

OVERALL_HEALTH_STATUS = _metrics_registry.get_or_create_gauge(
    "redis_overall_health_status",
    "Redis overall health status (1=healthy, 0=unhealthy)"
)

MONITORING_SERVICES_STATUS = _metrics_registry.get_or_create_gauge(
    "redis_monitoring_services_status",
    "Redis monitoring services status",
    ["service"]
)


class DefaultRedisClientProvider:
    """Default Redis client provider for health monitoring."""
    
    def __init__(self):
        """Initialize default client provider."""
        self._client: Optional[coredis.Redis] = None
        self._backup_client: Optional[coredis.Redis] = None
    
    async def get_client(self) -> Optional[coredis.Redis]:
        """Get Redis client for health monitoring."""
        if self._client is None:
            try:
                from prompt_improver.core.config import get_config
                
                config = get_config()
                redis_config = config.redis
                connection_params = redis_config.get_connection_params()
                self._client = coredis.Redis(**connection_params)
                
                # Test connection
                await self._client.ping()
                
            except Exception as e:
                logger.error(f"Failed to create Redis client: {e}")
                return None
        
        return self._client
    
    async def create_backup_client(self) -> Optional[coredis.Redis]:
        """Create backup Redis client for failover."""
        if self._backup_client is None:
            try:
                # In production, this would connect to a backup Redis instance
                # For now, we'll use the same configuration
                from prompt_improver.core.config import get_config
                
                config = get_config()
                redis_config = config.redis
                connection_params = redis_config.get_connection_params()
                
                # You could modify connection params for backup instance here
                # connection_params['host'] = 'backup-redis-host'
                
                self._backup_client = coredis.Redis(**connection_params)
                
                # Test backup connection
                await self._backup_client.ping()
                
            except Exception as e:
                logger.warning(f"Failed to create backup Redis client: {e}")
                return None
        
        return self._backup_client
    
    async def validate_client(self, client: coredis.Redis) -> bool:
        """Validate that Redis client is working."""
        try:
            await client.ping()
            return True
        except Exception:
            return False


class RedisHealthManager:
    """Redis health manager facade for coordinating all health monitoring services.
    
    Provides unified health management with automatic incident response,
    comprehensive monitoring, and SRE-focused operational capabilities.
    """
    
    def __init__(
        self,
        client_provider: Optional[RedisClientProviderProtocol] = None,
        monitoring_interval: float = 30.0,
        enable_auto_recovery: bool = True,
        enable_alerting: bool = True
    ):
        """Initialize Redis health manager.
        
        Args:
            client_provider: Redis client provider for connections
            monitoring_interval: Interval for health monitoring in seconds
            enable_auto_recovery: Whether to enable automatic recovery
            enable_alerting: Whether to enable alerting
        """
        self.client_provider = client_provider or DefaultRedisClientProvider()
        self.monitoring_interval = monitoring_interval
        self.enable_auto_recovery = enable_auto_recovery
        self.enable_alerting = enable_alerting
        
        # Initialize health services
        self.health_checker = RedisHealthChecker(self.client_provider)
        self.connection_monitor = RedisConnectionMonitor(self.client_provider)
        self.metrics_collector = RedisMetricsCollector(self.client_provider)
        self.alerting_service = RedisAlertingService() if enable_alerting else None
        self.recovery_service = RedisRecoveryService(self.client_provider) if enable_auto_recovery else None
        
        # Manager state
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
        self._last_comprehensive_health: Optional[Dict[str, Any]] = None
        
        # Statistics
        self._total_health_checks = 0
        self._failed_health_checks = 0
        self._incidents_handled = 0
        
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status from all services.
        
        Returns:
            Complete health status dictionary
        """
        start_time = time.time()
        
        try:
            with HEALTH_CHECK_DURATION.time():
                # Collect health data from all services
                health_data = {}
                
                # Basic health check
                health_metrics = await self.health_checker.check_health()
                health_data["health_check"] = {
                    "status": health_metrics.status.value,
                    "is_available": health_metrics.is_available,
                    "ping_latency_ms": health_metrics.ping_latency_ms,
                    "connection_utilization": health_metrics.connection_utilization,
                    "hit_rate": health_metrics.hit_rate,
                    "memory_usage_mb": health_metrics.memory_usage_mb,
                    "consecutive_failures": health_metrics.consecutive_failures,
                    "last_check_time": health_metrics.last_check_time.isoformat(),
                    "check_duration_ms": health_metrics.check_duration_ms,
                }
                
                # Connection monitoring
                try:
                    connection_metrics = await self.connection_monitor.monitor_connections()
                    connection_issues = await self.connection_monitor.detect_connection_issues()
                    health_data["connections"] = {
                        "status": connection_metrics.status.value,
                        "active_connections": connection_metrics.active_connections,
                        "pool_utilization": connection_metrics.pool_utilization,
                        "connection_failures": connection_metrics.connection_failures,
                        "detected_issues": connection_issues,
                    }
                except Exception as e:
                    logger.error(f"Connection monitoring failed: {e}")
                    health_data["connections"] = {"error": str(e)}
                
                # Performance metrics
                try:
                    performance_metrics = await self.metrics_collector.collect_performance_metrics()
                    health_data["performance"] = {
                        "instantaneous_ops_per_sec": performance_metrics.instantaneous_ops_per_sec,
                        "avg_ops_per_sec": performance_metrics.avg_ops_per_sec,
                        "hit_rate_percentage": performance_metrics.hit_rate_percentage,
                        "avg_latency_ms": performance_metrics.avg_latency_ms,
                        "p95_latency_ms": performance_metrics.p95_latency_ms,
                        "memory_usage_mb": round(performance_metrics.used_memory_bytes / 1024 / 1024, 2),
                        "fragmentation_ratio": performance_metrics.fragmentation_ratio,
                    }
                except Exception as e:
                    logger.error(f"Performance metrics collection failed: {e}")
                    health_data["performance"] = {"error": str(e)}
                
                # Alerting status
                if self.alerting_service:
                    try:
                        active_alerts = self.alerting_service.get_active_alerts()
                        alert_summary = self.alerting_service.get_alert_summary()
                        health_data["alerting"] = {
                            "active_alerts_count": len(active_alerts),
                            "active_alerts": [
                                {
                                    "id": alert.id,
                                    "level": alert.level.value,
                                    "message": alert.message,
                                    "timestamp": alert.timestamp.isoformat(),
                                }
                                for alert in active_alerts[:5]  # Show top 5
                            ],
                            "alert_summary": alert_summary,
                        }
                    except Exception as e:
                        logger.error(f"Alerting status collection failed: {e}")
                        health_data["alerting"] = {"error": str(e)}
                
                # Recovery status
                if self.recovery_service:
                    try:
                        recovery_stats = self.recovery_service.get_recovery_statistics()
                        circuit_breaker_info = self.recovery_service.get_circuit_breaker_info()
                        health_data["recovery"] = {
                            "statistics": recovery_stats,
                            "circuit_breaker": circuit_breaker_info,
                        }
                    except Exception as e:
                        logger.error(f"Recovery status collection failed: {e}")
                        health_data["recovery"] = {"error": str(e)}
                
                # Overall status calculation
                overall_healthy = self._calculate_overall_health(health_data)
                
                # Manager metadata
                check_duration_ms = (time.time() - start_time) * 1000
                health_data["manager"] = {
                    "overall_healthy": overall_healthy,
                    "check_duration_ms": round(check_duration_ms, 2),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "monitoring_enabled": self._is_monitoring,
                    "services_enabled": {
                        "health_checker": True,
                        "connection_monitor": True,
                        "metrics_collector": True,
                        "alerting": self.enable_alerting,
                        "recovery": self.enable_auto_recovery,
                    },
                    "statistics": {
                        "total_health_checks": self._total_health_checks,
                        "failed_health_checks": self._failed_health_checks,
                        "incidents_handled": self._incidents_handled,
                        "uptime_percentage": self._calculate_uptime_percentage(),
                    },
                }
                
                # Update metrics
                OVERALL_HEALTH_STATUS.set(1 if overall_healthy else 0)
                self._update_service_status_metrics()
                
                # Cache result
                self._last_comprehensive_health = health_data
                self._last_health_check = datetime.now(UTC)
                self._total_health_checks += 1
                
                return health_data
                
        except Exception as e:
            logger.error(f"Comprehensive health check failed: {e}")
            self._failed_health_checks += 1
            
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "manager": {
                    "overall_healthy": False,
                    "check_duration_ms": (time.time() - start_time) * 1000,
                    "error": "Health check failed",
                }
            }
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self._is_monitoring = True
        
        # Start individual services
        try:
            if self.alerting_service:
                await self.alerting_service.start_alerting()
            
            if self.recovery_service:
                await self.recovery_service.start_monitoring()
            
            await self.connection_monitor.start_monitoring()
            await self.metrics_collector.start_collection()
            
            # Start main monitoring loop
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Redis health monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            self._is_monitoring = False
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Stop main monitoring loop
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        # Stop individual services
        try:
            await self.metrics_collector.stop_collection()
            await self.connection_monitor.stop_monitoring()
            
            if self.recovery_service:
                await self.recovery_service.stop_monitoring()
            
            if self.alerting_service:
                await self.alerting_service.stop_alerting()
            
            logger.info("Redis health monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {e}")
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get quick health summary for integration.
        
        Returns:
            Health summary dictionary optimized for load balancer checks
        """
        try:
            # Use cached result if recent (within 30 seconds)
            if (self._last_comprehensive_health and 
                self._last_health_check and
                (datetime.now(UTC) - self._last_health_check).total_seconds() < 30):
                
                cached_data = self._last_comprehensive_health
                return {
                    "healthy": cached_data.get("manager", {}).get("overall_healthy", False),
                    "status": "healthy" if cached_data.get("manager", {}).get("overall_healthy") else "unhealthy",
                    "timestamp": cached_data.get("manager", {}).get("timestamp"),
                    "cached": True,
                }
            
            # Perform quick health check
            health_metrics = await self.health_checker.check_health()
            
            return {
                "healthy": health_metrics.is_available,
                "status": health_metrics.status.value,
                "response_time_ms": health_metrics.ping_latency_ms,
                "timestamp": health_metrics.last_check_time.isoformat(),
                "cached": False,
            }
            
        except Exception as e:
            logger.error(f"Health summary failed: {e}")
            return {
                "healthy": False,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
    
    def is_healthy(self) -> bool:
        """Quick health check for load balancer integration.
        
        Returns:
            True if Redis is healthy overall
        """
        try:
            # Use cached result if available and recent
            if (self._last_comprehensive_health and 
                self._last_health_check and
                (datetime.now(UTC) - self._last_health_check).total_seconds() < 60):
                
                return self._last_comprehensive_health.get("manager", {}).get("overall_healthy", False)
            
            # Fall back to basic availability check
            last_metrics = self.health_checker.get_last_metrics()
            if last_metrics:
                return last_metrics.is_available
            
            return False
            
        except Exception as e:
            logger.error(f"Quick health check failed: {e}")
            return False
    
    async def handle_incident(self, severity: AlertLevel) -> List[RecoveryAction]:
        """Handle incident with appropriate recovery actions.
        
        Args:
            severity: Incident severity level
            
        Returns:
            List of recovery actions taken
        """
        actions_taken = []
        
        try:
            self._incidents_handled += 1
            logger.warning(f"Handling Redis incident with severity: {severity.value}")
            
            # Get current health status
            health_data = await self.get_comprehensive_health()
            
            # Determine recovery strategy based on severity
            if severity in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                # Critical incidents - immediate recovery action
                if self.recovery_service:
                    recovery_event = await self.recovery_service.attempt_recovery(
                        f"Critical incident: {severity.value}"
                    )
                    actions_taken.append(recovery_event.action)
                    
                    # If recovery failed, try failover
                    if not recovery_event.success:
                        failover_success = await self.recovery_service.execute_failover()
                        if failover_success:
                            actions_taken.append(RecoveryAction.FAILOVER)
                        else:
                            actions_taken.append(RecoveryAction.ESCALATE)
            
            elif severity == AlertLevel.WARNING:
                # Warning incidents - monitoring and potential recovery
                if self.recovery_service:
                    # Check if circuit breaker allows recovery attempts
                    circuit_state = self.recovery_service.get_circuit_breaker_state()
                    if circuit_state.can_attempt():
                        recovery_event = await self.recovery_service.attempt_recovery(
                            f"Warning incident: {severity.value}"
                        )
                        actions_taken.append(recovery_event.action)
            
            # Send alerts if enabled
            if self.alerting_service:
                # This would typically be triggered by the monitoring loop
                # but we can also send immediate incident alerts
                pass
            
            logger.info(f"Incident handling completed. Actions taken: {actions_taken}")
            return actions_taken
            
        except Exception as e:
            logger.error(f"Failed to handle incident: {e}")
            return [RecoveryAction.ESCALATE]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of monitoring services.
        
        Returns:
            Monitoring services status
        """
        return {
            "manager": {
                "is_monitoring": self._is_monitoring,
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
                "total_health_checks": self._total_health_checks,
                "failed_health_checks": self._failed_health_checks,
                "incidents_handled": self._incidents_handled,
            },
            "services": {
                "health_checker": {
                    "enabled": True,
                    "last_metrics_available": self.health_checker.get_last_metrics() is not None,
                },
                "connection_monitor": {
                    "enabled": True,
                    "is_monitoring": hasattr(self.connection_monitor, "_is_monitoring") and self.connection_monitor._is_monitoring,
                },
                "metrics_collector": {
                    "enabled": True,
                    "is_collecting": hasattr(self.metrics_collector, "_is_collecting") and self.metrics_collector._is_collecting,
                },
                "alerting_service": {
                    "enabled": self.enable_alerting,
                    "is_running": self.alerting_service._is_running if self.alerting_service else False,
                    "active_alerts": len(self.alerting_service.get_active_alerts()) if self.alerting_service else 0,
                },
                "recovery_service": {
                    "enabled": self.enable_auto_recovery,
                    "is_monitoring": self.recovery_service._is_monitoring if self.recovery_service else False,
                    "circuit_breaker_state": self.recovery_service.get_circuit_breaker_state().state if self.recovery_service else "disabled",
                },
            },
        }
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for health management."""
        logger.info("Starting Redis health manager monitoring loop")
        
        while self._is_monitoring:
            try:
                # Perform comprehensive health check
                health_data = await self.get_comprehensive_health()
                
                # Check for alerts if alerting is enabled
                if self.alerting_service and "health_check" in health_data:
                    health_metrics = HealthMetrics(
                        ping_latency_ms=health_data["health_check"]["ping_latency_ms"],
                        connection_utilization=health_data["health_check"]["connection_utilization"],
                        hit_rate=health_data["health_check"]["hit_rate"],
                        memory_usage_mb=health_data["health_check"]["memory_usage_mb"],
                        is_available=health_data["health_check"]["is_available"],
                        consecutive_failures=health_data["health_check"]["consecutive_failures"],
                    )
                    
                    # Check for alerts
                    triggered_alerts = await self.alerting_service.check_thresholds(health_metrics)
                    
                    # Send any triggered alerts
                    for alert in triggered_alerts:
                        await self.alerting_service.send_alert(alert)
                        
                        # Handle critical alerts automatically
                        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                            await self.handle_incident(alert.level)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health manager monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief delay on error
    
    def _calculate_overall_health(self, health_data: Dict[str, Any]) -> bool:
        """Calculate overall health status from all monitoring data.
        
        Args:
            health_data: Complete health monitoring data
            
        Returns:
            True if overall system is healthy
        """
        try:
            # Check basic availability
            if not health_data.get("health_check", {}).get("is_available", False):
                return False
            
            # Check for critical alerts
            if self.alerting_service:
                active_alerts = health_data.get("alerting", {}).get("active_alerts", [])
                critical_alerts = [
                    alert for alert in active_alerts
                    if alert.get("level") in ["critical", "emergency"]
                ]
                if critical_alerts:
                    return False
            
            # Check connection health
            connections = health_data.get("connections", {})
            if connections.get("status") == "disconnected":
                return False
            
            # Check performance thresholds
            performance = health_data.get("performance", {})
            if performance.get("p95_latency_ms", 0) > 200:  # High latency threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate overall health: {e}")
            return False
    
    def _calculate_uptime_percentage(self) -> float:
        """Calculate uptime percentage based on health check history.
        
        Returns:
            Uptime percentage
        """
        if self._total_health_checks == 0:
            return 100.0
        
        successful_checks = self._total_health_checks - self._failed_health_checks
        return (successful_checks / self._total_health_checks) * 100
    
    def _update_service_status_metrics(self) -> None:
        """Update Prometheus metrics for service status."""
        try:
            services = {
                "health_checker": 1,
                "connection_monitor": 1,
                "metrics_collector": 1,
                "alerting_service": 1 if self.enable_alerting else 0,
                "recovery_service": 1 if self.enable_auto_recovery else 0,
            }
            
            for service, status in services.items():
                MONITORING_SERVICES_STATUS.labels(service=service).set(status)
                
        except Exception as e:
            logger.debug(f"Failed to update service status metrics: {e}")