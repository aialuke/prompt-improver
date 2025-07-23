"""
Component Health Monitor for ML Pipeline Orchestration.

Monitors the health of individual ML components registered with the orchestrator.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

from ..events.event_types import EventType, MLEvent
from ..connectors.component_connector import ComponentStatus


class ComponentHealthStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISCONNECTED = "disconnected"


@dataclass
class ComponentHealthCheck:
    """Component health check result."""
    component_name: str
    health_status: ComponentHealthStatus
    response_time: Optional[float]
    error_message: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime


class ComponentHealthMonitor:
    """
    Monitors the health of individual ML components.
    
    Performs regular health checks on registered components and tracks
    their health status over time.
    """
    
    def __init__(self, component_registry=None, event_bus=None, config=None):
        """Initialize the component health monitor."""
        self.component_registry = component_registry
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task = None
        
        # Health check results
        self.health_history: Dict[str, List[ComponentHealthCheck]] = {}
        self.current_health: Dict[str, ComponentHealthStatus] = {}
        
        # Configuration
        self.health_check_interval = self.config.get("health_check_interval", 60)  # seconds
        self.health_check_timeout = self.config.get("health_check_timeout", 10)  # seconds
        self.unhealthy_threshold = self.config.get("unhealthy_threshold", 3)  # consecutive failures
        self.history_limit = self.config.get("history_limit", 100)  # per component
        
        # Component tracking
        self.monitored_components: Set[str] = set()
        self.failed_checks: Dict[str, int] = {}
    
    async def start_monitoring(self) -> None:
        """Start component health monitoring."""
        if self.is_monitoring:
            self.logger.warning("Component health monitoring is already running")
            return
        
        self.logger.info("Starting component health monitoring")
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Emit monitoring started event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.MONITORING_STARTED,
                source="component_health_monitor",
                data={
                    "monitor_type": "component_health",
                    "started_at": datetime.now(timezone.utc).isoformat()
                }
            ))
    
    async def stop_monitoring(self) -> None:
        """Stop component health monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping component health monitoring")
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Emit monitoring stopped event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.MONITORING_STOPPED,
                source="component_health_monitor",
                data={
                    "monitor_type": "component_health",
                    "stopped_at": datetime.now(timezone.utc).isoformat()
                }
            ))
    
    async def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        try:
            while self.is_monitoring:
                # Get list of registered components
                if self.component_registry:
                    await self._update_monitored_components()
                
                # Perform health checks on all monitored components
                await self._perform_health_checks()
                
                # Wait for next check interval
                await asyncio.sleep(self.health_check_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Component health monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in component health monitoring loop: {e}")
            self.is_monitoring = False
    
    async def _update_monitored_components(self) -> None:
        """Update the list of components to monitor."""
        try:
            if not self.component_registry:
                return
            
            # Get currently registered components
            registered_components = await self.component_registry.list_components()
            new_components = set(registered_components) - self.monitored_components
            removed_components = self.monitored_components - set(registered_components)
            
            # Add new components
            for component_name in new_components:
                self.monitored_components.add(component_name)
                self.current_health[component_name] = ComponentHealthStatus.UNKNOWN
                self.failed_checks[component_name] = 0
                self.health_history[component_name] = []
                self.logger.info(f"Started monitoring component: {component_name}")
            
            # Remove components that are no longer registered
            for component_name in removed_components:
                self.monitored_components.discard(component_name)
                self.current_health[component_name] = ComponentHealthStatus.DISCONNECTED
                self.logger.info(f"Stopped monitoring component: {component_name}")
            
        except Exception as e:
            self.logger.error(f"Error updating monitored components: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all monitored components."""
        if not self.monitored_components:
            return
        
        # Perform health checks in parallel
        health_check_tasks = [
            self._check_component_health(component_name)
            for component_name in self.monitored_components
        ]
        
        await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_component_health(self, component_name: str) -> None:
        """Perform health check on a single component."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.component_registry:
                return
            
            # Get component connector
            connector = await self.component_registry.get_connector(component_name)
            if not connector:
                await self._record_health_check(
                    component_name,
                    ComponentHealthStatus.DISCONNECTED,
                    None,
                    "Component connector not found",
                    {}
                )
                return
            
            # Perform health check with timeout
            try:
                health_result = await asyncio.wait_for(
                    connector.health_check(),
                    timeout=self.health_check_timeout
                )
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Determine health status from result
                health_status = self._interpret_health_result(health_result)
                
                await self._record_health_check(
                    component_name,
                    health_status,
                    response_time,
                    None,
                    health_result
                )
                
            except asyncio.TimeoutError:
                response_time = asyncio.get_event_loop().time() - start_time
                await self._record_health_check(
                    component_name,
                    ComponentHealthStatus.UNHEALTHY,
                    response_time,
                    f"Health check timeout after {self.health_check_timeout}s",
                    {}
                )
            
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            await self._record_health_check(
                component_name,
                ComponentHealthStatus.UNHEALTHY,
                response_time,
                str(e),
                {}
            )
    
    def _interpret_health_result(self, health_result: Dict[str, Any]) -> ComponentHealthStatus:
        """Interpret health check result to determine status."""
        if not health_result:
            return ComponentHealthStatus.UNKNOWN
        
        status = health_result.get("status", "unknown").lower()
        
        if status == "healthy":
            return ComponentHealthStatus.HEALTHY
        elif status == "warning":
            return ComponentHealthStatus.WARNING
        elif status in ["unhealthy", "error", "failed"]:
            return ComponentHealthStatus.UNHEALTHY
        else:
            return ComponentHealthStatus.UNKNOWN
    
    async def _record_health_check(self, component_name: str, health_status: ComponentHealthStatus,
                                 response_time: Optional[float], error_message: Optional[str],
                                 details: Dict[str, Any]) -> None:
        """Record the result of a health check."""
        try:
            # Create health check record
            health_check = ComponentHealthCheck(
                component_name=component_name,
                health_status=health_status,
                response_time=response_time,
                error_message=error_message,
                details=details,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add to history
            if component_name not in self.health_history:
                self.health_history[component_name] = []
            
            self.health_history[component_name].append(health_check)
            
            # Limit history size
            if len(self.health_history[component_name]) > self.history_limit:
                self.health_history[component_name] = self.health_history[component_name][-self.history_limit:]
            
            # Update current health status
            previous_health = self.current_health.get(component_name, ComponentHealthStatus.UNKNOWN)
            self.current_health[component_name] = health_status
            
            # Track consecutive failures
            if health_status in [ComponentHealthStatus.UNHEALTHY, ComponentHealthStatus.DISCONNECTED]:
                self.failed_checks[component_name] = self.failed_checks.get(component_name, 0) + 1
            else:
                self.failed_checks[component_name] = 0
            
            # Check if component became unhealthy
            if (previous_health != health_status and 
                health_status in [ComponentHealthStatus.UNHEALTHY, ComponentHealthStatus.DISCONNECTED]):
                
                await self._handle_component_unhealthy(component_name, health_check)
            
            # Check if component recovered
            elif (previous_health in [ComponentHealthStatus.UNHEALTHY, ComponentHealthStatus.DISCONNECTED] and
                  health_status == ComponentHealthStatus.HEALTHY):
                
                await self._handle_component_recovered(component_name, health_check)
            
            # Emit health check event
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.COMPONENT_HEALTH_CHECK,
                    source="component_health_monitor",
                    data={
                        "component_name": component_name,
                        "health_status": health_status.value,
                        "response_time": response_time,
                        "error_message": error_message,
                        "consecutive_failures": self.failed_checks.get(component_name, 0),
                        "timestamp": health_check.timestamp.isoformat()
                    }
                ))
            
        except Exception as e:
            self.logger.error(f"Error recording health check for {component_name}: {e}")
    
    async def _handle_component_unhealthy(self, component_name: str, health_check: ComponentHealthCheck) -> None:
        """Handle component becoming unhealthy."""
        consecutive_failures = self.failed_checks.get(component_name, 0)
        
        self.logger.warning(f"Component {component_name} is unhealthy: {health_check.error_message}")
        
        # Emit unhealthy component alert
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.COMPONENT_UNHEALTHY,
                source="component_health_monitor",
                data={
                    "component_name": component_name,
                    "health_status": health_check.health_status.value,
                    "error_message": health_check.error_message,
                    "consecutive_failures": consecutive_failures,
                    "severity": "critical" if consecutive_failures >= self.unhealthy_threshold else "warning",
                    "timestamp": health_check.timestamp.isoformat()
                }
            ))
        
        # If component has failed too many times, mark for intervention
        if consecutive_failures >= self.unhealthy_threshold:
            self.logger.error(f"Component {component_name} has failed {consecutive_failures} consecutive health checks")
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.COMPONENT_INTERVENTION_REQUIRED,
                    source="component_health_monitor",
                    data={
                        "component_name": component_name,
                        "consecutive_failures": consecutive_failures,
                        "threshold": self.unhealthy_threshold,
                        "timestamp": health_check.timestamp.isoformat()
                    }
                ))
    
    async def _handle_component_recovered(self, component_name: str, health_check: ComponentHealthCheck) -> None:
        """Handle component recovery."""
        self.logger.info(f"Component {component_name} has recovered")
        
        # Emit recovery event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.COMPONENT_RECOVERED,
                source="component_health_monitor",
                data={
                    "component_name": component_name,
                    "health_status": health_check.health_status.value,
                    "response_time": health_check.response_time,
                    "timestamp": health_check.timestamp.isoformat()
                }
            ))
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get current health status of a component."""
        if component_name not in self.current_health:
            return None
        
        recent_checks = self.health_history.get(component_name, [])[-10:]  # Last 10 checks
        
        return {
            "component_name": component_name,
            "current_health": self.current_health[component_name].value,
            "consecutive_failures": self.failed_checks.get(component_name, 0),
            "last_check": recent_checks[-1].timestamp.isoformat() if recent_checks else None,
            "recent_checks_count": len(recent_checks),
            "average_response_time": sum(
                check.response_time for check in recent_checks 
                if check.response_time is not None
            ) / len([check for check in recent_checks if check.response_time is not None]) if recent_checks else None
        }
    
    async def get_all_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all monitored components."""
        result = {}
        for component_name in self.monitored_components:
            health_info = await self.get_component_health(component_name)
            if health_info:
                result[component_name] = health_info
        return result
    
    async def get_unhealthy_components(self) -> List[str]:
        """Get list of currently unhealthy components."""
        return [
            component_name for component_name, health_status in self.current_health.items()
            if health_status in [ComponentHealthStatus.UNHEALTHY, ComponentHealthStatus.DISCONNECTED]
        ]
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_components = len(self.monitored_components)
        healthy_count = sum(
            1 for status in self.current_health.values()
            if status == ComponentHealthStatus.HEALTHY
        )
        unhealthy_count = sum(
            1 for status in self.current_health.values()
            if status in [ComponentHealthStatus.UNHEALTHY, ComponentHealthStatus.DISCONNECTED]
        )
        
        return {
            "total_components": total_components,
            "healthy_components": healthy_count,
            "unhealthy_components": unhealthy_count,
            "health_percentage": (healthy_count / total_components * 100) if total_components > 0 else 0,
            "monitoring_active": self.is_monitoring,
            "health_check_interval": self.health_check_interval,
            "components_requiring_intervention": sum(
                1 for failures in self.failed_checks.values()
                if failures >= self.unhealthy_threshold
            )
        }