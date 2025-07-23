"""
ML Orchestration Health Checkers for integration with existing health monitoring system.

Extends the existing health monitoring infrastructure to include ML pipeline components.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from .base import HealthChecker, HealthResult, HealthStatus
from .metrics import instrument_health_check


class MLOrchestratorHealthChecker(HealthChecker):
    """Health checker for the main ML Pipeline Orchestrator."""
    
    def __init__(self):
        """Initialize the ML orchestrator health checker."""
        super().__init__(name="ml_orchestrator")
        self.logger = logging.getLogger(__name__)
        self._orchestrator = None
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator instance to monitor."""
        self._orchestrator = orchestrator
    
    @instrument_health_check("ml_orchestrator")
    async def check(self) -> HealthResult:
        """Perform health check on the ML orchestrator."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self._orchestrator:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="ML orchestrator not initialized",
                    details={},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Check orchestrator state
            if not self._orchestrator._is_initialized:
                return HealthResult(
                    status=HealthStatus.FAILED,
                    component=self.name,
                    message="ML orchestrator not initialized",
                    details={"state": "uninitialized"},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Check component health
            component_health = await self._orchestrator.get_component_health()
            healthy_components = sum(1 for h in component_health.values() if h)
            total_components = len(component_health)
            
            # Check resource usage
            resource_usage = await self._orchestrator.get_resource_usage()
            
            # Check active workflows
            workflows = await self._orchestrator.list_workflows()
            active_workflows = len([w for w in workflows if w.state.value == "running"])
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            # Determine health status
            health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 100
            
            if health_percentage >= 90:
                status = HealthStatus.HEALTHY
                message = f"ML orchestrator healthy ({healthy_components}/{total_components} components)"
            elif health_percentage >= 70:
                status = HealthStatus.WARNING
                message = f"ML orchestrator degraded ({healthy_components}/{total_components} components)"
            else:
                status = HealthStatus.FAILED
                message = f"ML orchestrator unhealthy ({healthy_components}/{total_components} components)"
            
            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "total_components": total_components,
                    "healthy_components": healthy_components,
                    "health_percentage": health_percentage,
                    "active_workflows": active_workflows,
                    "resource_usage": resource_usage,
                    "orchestrator_state": self._orchestrator.state.value if hasattr(self._orchestrator, 'state') else "unknown"
                },
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"ML orchestrator health check failed: {e}")
            
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )


class MLComponentRegistryHealthChecker(HealthChecker):
    """Health checker for the ML Component Registry."""
    
    def __init__(self):
        """Initialize the component registry health checker."""
        super().__init__(name="ml_component_registry")
        self.logger = logging.getLogger(__name__)
        self._registry = None
    
    def set_registry(self, registry):
        """Set the component registry instance to monitor."""
        self._registry = registry
    
    @instrument_health_check("ml_component_registry")
    async def check(self) -> HealthResult:
        """Perform health check on the component registry."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self._registry:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="Component registry not initialized",
                    details={},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Get health summary
            health_summary = await self._registry.get_health_summary()
            
            # List all components
            components = await self._registry.list_components()
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            # Determine health status based on component health
            health_percentage = health_summary.get("overall_health_percentage", 0)
            
            if health_percentage >= 90:
                status = HealthStatus.HEALTHY
                message = f"Component registry healthy ({health_percentage:.1f}%)"
            elif health_percentage >= 70:
                status = HealthStatus.WARNING
                message = f"Component registry degraded ({health_percentage:.1f}%)"
            else:
                status = HealthStatus.FAILED
                message = f"Component registry unhealthy ({health_percentage:.1f}%)"
            
            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "health_summary": health_summary,
                    "total_registered_components": len(components),
                    "registry_response_time": response_time
                },
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Component registry health check failed: {e}")
            
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )


class MLResourceManagerHealthChecker(HealthChecker):
    """Health checker for the ML Resource Manager."""
    
    def __init__(self):
        """Initialize the resource manager health checker."""
        super().__init__(name="ml_resource_manager")
        self.logger = logging.getLogger(__name__)
        self._resource_manager = None
    
    def set_resource_manager(self, resource_manager):
        """Set the resource manager instance to monitor."""
        self._resource_manager = resource_manager
    
    @instrument_health_check("ml_resource_manager")
    async def check(self) -> HealthResult:
        """Perform health check on the resource manager."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self._resource_manager:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="Resource manager not initialized",
                    details={},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Get usage statistics
            usage_stats = await self._resource_manager.get_usage_stats()
            
            # Get current allocations
            allocations = await self._resource_manager.get_allocations()
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            # Check for resource exhaustion
            critical_resources = []
            warning_resources = []
            
            for resource_type, stats in usage_stats.items():
                usage_pct = stats.get("usage_percentage", 0)
                if usage_pct >= 90:
                    critical_resources.append(f"{resource_type}: {usage_pct:.1f}%")
                elif usage_pct >= 75:
                    warning_resources.append(f"{resource_type}: {usage_pct:.1f}%")
            
            # Determine health status
            if critical_resources:
                status = HealthStatus.FAILED
                message = f"Critical resource usage: {', '.join(critical_resources)}"
            elif warning_resources:
                status = HealthStatus.WARNING
                message = f"High resource usage: {', '.join(warning_resources)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Resource usage within normal limits"
            
            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "usage_stats": usage_stats,
                    "total_allocations": len(allocations),
                    "critical_resources": critical_resources,
                    "warning_resources": warning_resources,
                    "monitoring_active": self._resource_manager.is_monitoring
                },
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Resource manager health check failed: {e}")
            
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )


class MLWorkflowEngineHealthChecker(HealthChecker):
    """Health checker for the ML Workflow Execution Engine."""
    
    def __init__(self):
        """Initialize the workflow engine health checker."""
        super().__init__(name="ml_workflow_engine")
        self.logger = logging.getLogger(__name__)
        self._workflow_engine = None
    
    def set_workflow_engine(self, workflow_engine):
        """Set the workflow engine instance to monitor."""
        self._workflow_engine = workflow_engine
    
    @instrument_health_check("ml_workflow_engine")
    async def check(self) -> HealthResult:
        """Perform health check on the workflow engine."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self._workflow_engine:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="Workflow engine not initialized",
                    details={},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Get workflow definitions
            workflow_definitions = await self._workflow_engine.list_workflow_definitions()
            
            # Get active executors
            active_executors = len(self._workflow_engine.active_executors)
            
            # Check for failed workflows
            failed_workflows = []
            for workflow_id, executor in self._workflow_engine.active_executors.items():
                if hasattr(executor, 'is_cancelled') and executor.is_cancelled:
                    failed_workflows.append(workflow_id)
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            # Determine health status
            if failed_workflows:
                status = HealthStatus.WARNING
                message = f"Workflow engine operational with {len(failed_workflows)} failed workflows"
            elif active_executors > 0:
                status = HealthStatus.HEALTHY
                message = f"Workflow engine healthy with {active_executors} active workflows"
            else:
                status = HealthStatus.HEALTHY
                message = "Workflow engine idle and ready"
            
            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "workflow_definitions": len(workflow_definitions),
                    "active_executors": active_executors,
                    "failed_workflows": len(failed_workflows),
                    "available_workflow_types": workflow_definitions
                },
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Workflow engine health check failed: {e}")
            
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )


class MLEventBusHealthChecker(HealthChecker):
    """Health checker for the ML Event Bus."""
    
    def __init__(self):
        """Initialize the event bus health checker."""
        super().__init__(name="ml_event_bus")
        self.logger = logging.getLogger(__name__)
        self._event_bus = None
    
    def set_event_bus(self, event_bus):
        """Set the event bus instance to monitor."""
        self._event_bus = event_bus
    
    @instrument_health_check("ml_event_bus")
    async def check(self) -> HealthResult:
        """Perform health check on the event bus."""
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self._event_bus:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="Event bus not initialized",
                    details={},
                    timestamp=start_time,
                    response_time_ms=0.0
                )
            
            # Check event bus statistics
            stats = self._event_bus.get_statistics()
            
            # Test event emission (simple health check event)
            test_start = datetime.now(timezone.utc)
            await self._event_bus.emit_health_check_event("ml_event_bus_health_test")
            test_end = datetime.now(timezone.utc)
            test_response_time = (test_end - test_start).total_seconds()
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            # Determine health status based on metrics
            total_events = stats.get("total_events", 0)
            failed_events = stats.get("failed_events", 0)
            active_handlers = stats.get("active_handlers", 0)
            
            failure_rate = (failed_events / total_events * 100) if total_events > 0 else 0
            
            if test_response_time > 1.0:  # Event emission took too long
                status = HealthStatus.FAILED
                message = f"Event bus slow (test emission: {test_response_time:.2f}s)"
            elif failure_rate > 5:  # High failure rate
                status = HealthStatus.WARNING
                message = f"Event bus degraded (failure rate: {failure_rate:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Event bus healthy ({active_handlers} handlers active)"
            
            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "event_statistics": stats,
                    "test_emission_time": test_response_time,
                    "failure_rate": failure_rate
                },
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Event bus health check failed: {e}")
            
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=start_time,
                response_time_ms=response_time * 1000
            )