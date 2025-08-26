"""Monitoring Coordinator Service.

Focused service responsible for health monitoring, metrics collection, and performance tracking.
Handles comprehensive health checks, resource monitoring, and component observability.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ....shared.interfaces.protocols.ml import (
    CacheServiceProtocol,
    ComponentInvokerProtocol,
    DatabaseServiceProtocol,
    EventBusProtocol,
    HealthMonitorProtocol,
    MLflowServiceProtocol,
    ResourceManagerProtocol,
)
from ..core.orchestrator_service_protocols import MonitoringCoordinatorProtocol
from ..events.event_types import EventType, MLEvent


class MonitoringCoordinator:
    """
    Monitoring Coordinator Service.
    
    Responsible for:
    - Comprehensive health monitoring of all components
    - Resource usage tracking and performance metrics
    - Component observability and health status coordination
    - System-wide health reporting and alerting
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        resource_manager: ResourceManagerProtocol,
        mlflow_service: MLflowServiceProtocol,
        cache_service: CacheServiceProtocol,
        database_service: DatabaseServiceProtocol,
        health_monitor: HealthMonitorProtocol | None = None,
        component_invoker: ComponentInvokerProtocol | None = None,
    ):
        """Initialize MonitoringCoordinator with required dependencies.
        
        Args:
            event_bus: Event bus for monitoring event communication
            resource_manager: Resource management for usage statistics
            mlflow_service: MLflow service for experiment tracking health
            cache_service: Cache service for performance health
            database_service: Database service for persistence health
            health_monitor: Optional dedicated health monitoring service
            component_invoker: Optional component invoker for history tracking
        """
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        self.mlflow_service = mlflow_service
        self.cache_service = cache_service
        self.database_service = database_service
        self.health_monitor = health_monitor
        self.component_invoker = component_invoker
        self.logger = logging.getLogger(__name__)
        
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the monitoring coordinator."""
        if self._is_initialized:
            return
        
        self.logger.info("Initializing Monitoring Coordinator")
        
        try:
            # Setup monitoring infrastructure
            await self.setup_monitoring()
            
            self._is_initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="monitoring_coordinator",
                data={"component": "monitoring_coordinator", "timestamp": datetime.now(timezone.utc).isoformat()}
            ))
            
            self.logger.info("Monitoring Coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring coordinator: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the monitoring coordinator."""
        self.logger.info("Shutting down Monitoring Coordinator")
        
        try:
            self._is_initialized = False
            self.logger.info("Monitoring Coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during monitoring coordinator shutdown: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """
        Comprehensive health check of the orchestrator and all components.
        
        Returns:
            Dictionary with overall health status and component details
        """
        try:
            # Get resource usage
            resource_usage = await self.get_resource_usage()
            
            # Check injected services health
            external_services_health = await self._check_external_services_health()
            
            # Determine overall health
            resource_ok = resource_usage.get("memory_usage_percent", 0) < 90  # Less than 90% memory usage
            external_services_ok = all(
                'error' not in str(status) and status != 'unhealthy' 
                for status in external_services_health.values()
            ) if external_services_health else True
            
            overall_healthy = resource_ok and external_services_ok
            
            health_result = {
                "healthy": overall_healthy,
                "status": "healthy" if overall_healthy else "degraded",
                "resource_usage": resource_usage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monitoring_coordinator_version": "2025.1"
            }
            
            if external_services_health:
                health_result["external_services"] = external_services_health
            
            # Add dedicated health monitor results if available
            if self.health_monitor and hasattr(self.health_monitor, 'comprehensive_health_check'):
                try:
                    dedicated_health = await self.health_monitor.comprehensive_health_check()
                    health_result["dedicated_health_monitor"] = dedicated_health
                except Exception as e:
                    health_result["dedicated_health_monitor"] = f"error: {e}"
            
            return health_result
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        try:
            return await self.resource_manager.get_usage_stats()
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def setup_monitoring(self) -> None:
        """Setup monitoring for the orchestrator."""
        self.logger.info("Setting up monitoring infrastructure")
        
        try:
            # Setup event handlers for monitoring events
            self._setup_monitoring_event_handlers()
            
            # Initialize health monitor if available
            if self.health_monitor:
                # This would setup periodic health checks, metrics collection, etc.
                self.logger.info("Dedicated health monitor integration enabled")
            
            self.logger.info("Monitoring setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            raise

    def get_invocation_history(self, component_name: str | None = None) -> list[dict[str, Any]]:
        """Get component invocation history."""
        if not self.component_invoker:
            self.logger.warning("Component invoker not available for history tracking")
            return []
        
        try:
            history = self.component_invoker.get_invocation_history(component_name)
            return [
                {
                    "component_name": result.component_name,
                    "method_name": result.method_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat(),
                    "error": result.error
                }
                for result in history
            ]
        except Exception as e:
            self.logger.error(f"Failed to get invocation history: {e}")
            return []

    def _setup_monitoring_event_handlers(self) -> None:
        """Setup event handlers for monitoring-related events."""
        # Resource monitoring events
        self.event_bus.subscribe(EventType.RESOURCE_EXHAUSTED, self._handle_resource_exhausted)
        
        # Component health events
        self.event_bus.subscribe(EventType.COMPONENT_HEALTH_CHANGED, self._handle_component_health_changed)

    async def _handle_resource_exhausted(self, event: MLEvent) -> None:
        """Handle resource exhaustion events."""
        resource_type = event.data.get("resource_type")
        self.logger.warning(f"Resource exhausted: {resource_type}")
        
        # Implement resource management strategies through resource manager
        try:
            await self.resource_manager.handle_resource_exhaustion(resource_type)
        except Exception as e:
            self.logger.error(f"Failed to handle resource exhaustion: {e}")

    async def _handle_component_health_changed(self, event: MLEvent) -> None:
        """Handle component health change events."""
        component_name = event.data.get("component_name")
        is_healthy = event.data.get("is_healthy", False)
        
        if component_name:
            self.logger.info(f"Component {component_name} health changed: {is_healthy}")
            
            # Additional monitoring logic could be added here
            # Such as triggering alerts, updating dashboards, etc.

    async def _check_external_services_health(self) -> dict[str, Any]:
        """Check health of external services."""
        external_services_health = {}
        
        # Check MLflow service health
        if self.mlflow_service and hasattr(self.mlflow_service, 'health_check'):
            try:
                mlflow_health = await self.mlflow_service.health_check()
                external_services_health['mlflow'] = mlflow_health.value if hasattr(mlflow_health, 'value') else str(mlflow_health)
            except Exception as e:
                external_services_health['mlflow'] = f"error: {e}"
        
        # Check cache service health
        if self.cache_service and hasattr(self.cache_service, 'health_check'):
            try:
                cache_health = await self.cache_service.health_check()
                external_services_health['cache'] = cache_health.value if hasattr(cache_health, 'value') else str(cache_health)
            except Exception as e:
                external_services_health['cache'] = f"error: {e}"
        
        # Check database service health
        if self.database_service and hasattr(self.database_service, 'health_check'):
            try:
                db_health = await self.database_service.health_check()
                external_services_health['database'] = db_health.value if hasattr(db_health, 'value') else str(db_health)
            except Exception as e:
                external_services_health['database'] = f"error: {e}"
        
        return external_services_health