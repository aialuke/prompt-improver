"""Pool health monitoring and metrics collection for PostgreSQL connections.

Handles connection monitoring, health checks, and performance tracking.
Extracted from PostgreSQLPoolManager following single responsibility principle.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from sqlalchemy import text

from prompt_improver.database.services.connection.pool_protocols import (
    PoolMonitoringServiceProtocol,
    PoolScalingManagerProtocol,
)
from prompt_improver.database.services.connection.pool_shared_context import (
    HealthStatus,
    PoolSharedContext,
    PoolState,
)

logger = logging.getLogger(__name__)


class PoolMonitoringService:
    """Pool health monitoring and metrics collection.
    
    Responsible for:
    - Comprehensive health checks across all pool components
    - Metrics collection and reporting
    - Circuit breaker monitoring and management
    - Background monitoring task coordination
    - Connection registry management
    """
    
    def __init__(
        self,
        shared_context: PoolSharedContext,
        scaling_manager: Optional[PoolScalingManagerProtocol] = None,
    ):
        self.context = shared_context
        self.scaling_manager = scaling_manager
        
        # Background monitoring configuration
        self._health_monitor_interval = 30  # seconds
        self._metrics_update_interval = 60  # seconds
        
        logger.info(
            f"PoolMonitoringService initialized for service: {self.context.service_name}"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the pool manager."""
        start_time = time.time()
        
        health_info = {
            "status": "unknown",
            "timestamp": start_time,
            "service": self.context.service_name,
            "components": {},
            "metrics": await self.get_metrics(),
            "response_time_ms": 0,
        }
        
        try:
            # Test SQLAlchemy session if available
            if (
                self.context.async_session_factory
                and not self.context.is_circuit_breaker_open()
            ):
                try:
                    session = self.context.async_session_factory()
                    async with session:
                        await session.execute(text("SELECT 1"))
                    health_info["components"]["sqlalchemy_session"] = "healthy"
                except Exception as e:
                    health_info["components"]["sqlalchemy_session"] = f"error: {e}"
                    logger.warning(f"SQLAlchemy session health check failed: {e}")
            else:
                health_info["components"]["sqlalchemy_session"] = "unavailable"
            
            # Test HA pools
            for pool_name, pool in self.context.pg_pools.items():
                try:
                    async with pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    health_info["components"][f"ha_pool_{pool_name}"] = "healthy"
                except Exception as e:
                    health_info["components"][f"ha_pool_{pool_name}"] = f"error: {e}"
                    logger.warning(f"HA pool {pool_name} health check failed: {e}")
            
            # Overall status determination
            failed_components = [
                k for k, v in health_info["components"].items() 
                if not v.startswith("healthy")
            ]
            
            total_components = len(health_info["components"])
            if total_components == 0:
                health_info["status"] = "unknown"
                self.context.health_status = HealthStatus.UNKNOWN
            elif not failed_components:
                health_info["status"] = "healthy"
                self.context.health_status = HealthStatus.HEALTHY
            elif len(failed_components) < total_components / 2:
                health_info["status"] = "degraded"
                self.context.health_status = HealthStatus.DEGRADED
            else:
                health_info["status"] = "unhealthy"
                self.context.health_status = HealthStatus.UNHEALTHY
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            self.context.health_status = HealthStatus.UNHEALTHY
            logger.error(f"Health check failed: {e}")
        
        health_info["response_time_ms"] = (time.time() - start_time) * 1000
        return health_info
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pool metrics."""
        # Collect pool metrics if scaling manager is available
        pool_metrics = {}
        if self.scaling_manager:
            pool_metrics = await self.scaling_manager.collect_pool_metrics()
        
        # Calculate efficiency metrics
        efficiency = self.context.metrics.get_efficiency_metrics()
        
        return {
            "service": self.context.service_name,
            "pool_state": self.context.pool_state.value,
            "health_status": self.context.health_status.value,
            "current_pool_size": self.context.current_pool_size,
            "min_pool_size": self.context.min_pool_size,
            "max_pool_size": self.context.max_pool_size,
            "pool_metrics": pool_metrics,
            "connection_metrics": self.context.metrics.to_dict(),
            "efficiency_metrics": efficiency,
            "circuit_breaker": {
                "enabled": self.context.pool_config.enable_circuit_breaker,
                "state": "open" if self.context.is_circuit_breaker_open() else "closed",
                "failures": self.context.circuit_breaker_failures,
                "threshold": self.context.circuit_breaker_threshold,
            },
            "ha_pools": list(self.context.pg_pools.keys()),
            "connection_registry_size": len(self.context.connection_registry),
            "performance_window_size": len(self.context.performance_window),
            "background_tasks_active": len([
                task for task in self.context.background_tasks.values() 
                if task is not None and not task.done()
            ]),
        }
    
    def is_healthy(self) -> bool:
        """Quick health status check."""
        # Check circuit breaker state
        if self.context.is_circuit_breaker_open():
            return False
        
        # Check basic health indicators
        if self.context.health_status == HealthStatus.UNHEALTHY:
            return False
        
        # Check pool utilization
        if self.context.metrics.pool_utilization > 95.0:
            return False
        
        # Check error rate
        if self.context.metrics.error_rate > 0.1:  # 10% error rate threshold
            return False
        
        return True
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start health monitor task
            if "health_monitor" not in self.context.background_tasks:
                self.context.background_tasks["health_monitor"] = asyncio.create_task(
                    self._health_monitor_loop()
                )
                logger.info("Health monitor task started")
            
            # Start metrics update task
            if "metrics_update" not in self.context.background_tasks:
                self.context.background_tasks["metrics_update"] = asyncio.create_task(
                    self._metrics_update_loop()
                )
                logger.info("Metrics update task started")
            
            logger.info(
                f"Background monitoring started for service: {self.context.service_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start background monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        logger.info(f"Stopping monitoring for service: {self.context.service_name}")
        
        tasks_to_stop = ["health_monitor", "metrics_update"]
        
        for task_name in tasks_to_stop:
            task = self.context.background_tasks.get(task_name)
            if task and not task.done():
                try:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    logger.info(f"Stopped {task_name} task")
                except Exception as e:
                    logger.warning(f"Error stopping {task_name} task: {e}")
                finally:
                    self.context.background_tasks[task_name] = None
        
        logger.info("Background monitoring stopped")
    
    def get_connection_registry(self) -> Dict[str, Any]:
        """Get connection registry information."""
        registry_info = {
            "total_connections": len(self.context.connection_registry),
            "connections": {},
        }
        
        for conn_id, conn_info in self.context.connection_registry.items():
            registry_info["connections"][conn_id] = {
                "connection_id": conn_info.connection_id,
                "created_at": conn_info.created_at.isoformat(),
                "last_used": conn_info.last_used.isoformat(),
                "query_count": conn_info.query_count,
                "error_count": conn_info.error_count,
                "pool_name": conn_info.pool_name,
            }
        
        return registry_info
    
    def record_performance_event(
        self, event_type: str, duration_ms: float, success: bool
    ) -> None:
        """Record a performance event in the monitoring system."""
        self.context.record_performance_event(event_type, duration_ms, success)
        
        # Update circuit breaker state based on success
        if success:
            self.context.reset_circuit_breaker()
        
        logger.debug(
            f"Performance event recorded: {event_type}, "
            f"duration={duration_ms:.2f}ms, success={success}"
        )
    
    def handle_connection_failure(self, error: Exception) -> None:
        """Handle connection failure and update circuit breaker."""
        self.context.record_circuit_breaker_failure(error)
        
        # Check if circuit breaker should be opened
        if self.context.is_circuit_breaker_open():
            logger.error(
                f"Circuit breaker opened due to {self.context.circuit_breaker_failures} failures"
            )
        
        logger.warning(f"Connection failure handled: {error}")
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self.context.is_initialized:
            try:
                await asyncio.sleep(self._health_monitor_interval)
                await self.health_check()
                logger.debug("Health check completed")
                
            except asyncio.CancelledError:
                logger.debug("Health monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self._health_monitor_interval)
    
    async def _metrics_update_loop(self) -> None:
        """Background metrics update loop."""
        while self.context.is_initialized:
            try:
                await asyncio.sleep(self._metrics_update_interval)
                
                # Update pool utilization
                self.context.update_pool_utilization()
                
                # Trigger scaling evaluation if scaling manager is available
                if self.scaling_manager:
                    await self.scaling_manager.perform_background_scaling_evaluation()
                
                logger.debug("Metrics update completed")
                
            except asyncio.CancelledError:
                logger.debug("Metrics update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(self._metrics_update_interval)
    
    @property
    def circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.context.is_circuit_breaker_open()