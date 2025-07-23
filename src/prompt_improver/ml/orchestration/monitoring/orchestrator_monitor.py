"""
Orchestrator Monitor for ML Pipeline Orchestration.

Monitors the health and performance of the central ML pipeline orchestrator.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

from ..events.event_types import EventType, MLEvent


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    memory_total: float
    disk_usage: float
    disk_total: float
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    timestamp: datetime


@dataclass
class OrchestratorMetrics:
    """Orchestrator-specific metrics."""
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    active_components: int
    event_queue_size: int
    event_processing_rate: float
    average_workflow_duration: float
    resource_utilization: Dict[str, float]
    timestamp: datetime


class OrchestratorMonitor:
    """
    Monitors the health and performance of the ML pipeline orchestrator.
    
    Tracks system resources, orchestrator metrics, component health,
    and provides alerting capabilities.
    """
    
    def __init__(self, orchestrator=None, event_bus=None, config=None):
        """Initialize the orchestrator monitor."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task = None
        
        # Health status
        self.current_health = HealthStatus.UNKNOWN
        self.last_health_check = None
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.orchestrator_metrics_history: List[OrchestratorMetrics] = []
        
        # Alert thresholds
        self.cpu_threshold = self.config.get("cpu_threshold", 80.0)
        self.memory_threshold = self.config.get("memory_threshold", 85.0)
        self.disk_threshold = self.config.get("disk_threshold", 90.0)
        self.queue_size_threshold = self.config.get("queue_size_threshold", 1000)
        
        # Monitoring intervals
        self.system_check_interval = self.config.get("system_check_interval", 30)  # seconds
        self.orchestrator_check_interval = self.config.get("orchestrator_check_interval", 10)  # seconds
        
        # Alert cooldown
        self.alert_cooldown = self.config.get("alert_cooldown", 300)  # 5 minutes
        self.last_alerts = {}
    
    async def start_monitoring(self) -> None:
        """Start the monitoring process."""
        if self.is_monitoring:
            self.logger.warning("Orchestrator monitoring is already running")
            return
        
        self.logger.info("Starting orchestrator monitoring")
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Emit monitoring started event
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.MONITORING_STARTED,
                source="orchestrator_monitor",
                data={
                    "monitor_type": "orchestrator",
                    "started_at": datetime.now(timezone.utc).isoformat()
                }
            ))
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping orchestrator monitoring")
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
                source="orchestrator_monitor",
                data={
                    "monitor_type": "orchestrator",
                    "stopped_at": datetime.now(timezone.utc).isoformat()
                }
            ))
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        system_check_counter = 0
        orchestrator_check_counter = 0
        
        try:
            while self.is_monitoring:
                current_time = asyncio.get_event_loop().time()
                
                # System metrics check
                if system_check_counter <= 0:
                    await self._collect_system_metrics()
                    system_check_counter = self.system_check_interval
                
                # Orchestrator metrics check
                if orchestrator_check_counter <= 0:
                    await self._collect_orchestrator_metrics()
                    orchestrator_check_counter = self.orchestrator_check_interval
                
                # Perform health assessment
                await self._assess_health()
                
                # Wait for 1 second
                await asyncio.sleep(1)
                system_check_counter -= 1
                orchestrator_check_counter -= 1
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_total = memory.total
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            disk_total = disk.total
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Thread count (current process)
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # Create metrics object
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_total=memory_total,
                disk_usage=disk_usage,
                disk_total=disk_total,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            
            # Keep only recent metrics (last 1000 entries)
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-1000:]
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_orchestrator_metrics(self) -> None:
        """Collect orchestrator-specific metrics."""
        try:
            if not self.orchestrator:
                return
            
            # Get orchestrator status
            active_workflows = 0
            completed_workflows = 0
            failed_workflows = 0
            active_components = 0
            event_queue_size = 0
            event_processing_rate = 0.0
            average_workflow_duration = 0.0
            resource_utilization = {}
            
            # Collect metrics from orchestrator components
            if hasattr(self.orchestrator, 'workflow_engine'):
                workflow_stats = await self.orchestrator.workflow_engine.get_statistics()
                active_workflows = workflow_stats.get("active_workflows", 0)
                completed_workflows = workflow_stats.get("completed_workflows", 0)
                failed_workflows = workflow_stats.get("failed_workflows", 0)
                average_workflow_duration = workflow_stats.get("average_duration", 0.0)
            
            if hasattr(self.orchestrator, 'component_registry'):
                component_stats = await self.orchestrator.component_registry.get_statistics()
                active_components = component_stats.get("active_components", 0)
            
            if hasattr(self.orchestrator, 'event_bus'):
                event_stats = await self.orchestrator.event_bus.get_statistics()
                event_queue_size = event_stats.get("queue_size", 0)
                event_processing_rate = event_stats.get("processing_rate", 0.0)
            
            if hasattr(self.orchestrator, 'resource_manager'):
                resource_stats = await self.orchestrator.resource_manager.get_utilization()
                resource_utilization = resource_stats
            
            # Create metrics object
            metrics = OrchestratorMetrics(
                active_workflows=active_workflows,
                completed_workflows=completed_workflows,
                failed_workflows=failed_workflows,
                active_components=active_components,
                event_queue_size=event_queue_size,
                event_processing_rate=event_processing_rate,
                average_workflow_duration=average_workflow_duration,
                resource_utilization=resource_utilization,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Store metrics
            self.orchestrator_metrics_history.append(metrics)
            
            # Keep only recent metrics (last 1000 entries)
            if len(self.orchestrator_metrics_history) > 1000:
                self.orchestrator_metrics_history = self.orchestrator_metrics_history[-1000:]
            
            # Check for alerts
            await self._check_orchestrator_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting orchestrator metrics: {e}")
    
    async def _assess_health(self) -> None:
        """Assess overall orchestrator health."""
        try:
            health_factors = []
            
            # System health factors
            if self.system_metrics_history:
                latest_system = self.system_metrics_history[-1]
                
                if latest_system.cpu_usage > self.cpu_threshold:
                    health_factors.append("high_cpu")
                if latest_system.memory_usage > self.memory_threshold:
                    health_factors.append("high_memory")
                if latest_system.disk_usage > self.disk_threshold:
                    health_factors.append("high_disk")
            
            # Orchestrator health factors
            if self.orchestrator_metrics_history:
                latest_orchestrator = self.orchestrator_metrics_history[-1]
                
                if latest_orchestrator.event_queue_size > self.queue_size_threshold:
                    health_factors.append("high_queue_size")
                if latest_orchestrator.failed_workflows > latest_orchestrator.completed_workflows * 0.1:
                    health_factors.append("high_failure_rate")
            
            # Determine overall health
            if not health_factors:
                new_health = HealthStatus.HEALTHY
            elif len(health_factors) == 1:
                new_health = HealthStatus.WARNING
            else:
                new_health = HealthStatus.CRITICAL
            
            # Check if health status changed
            if new_health != self.current_health:
                self.logger.info(f"Orchestrator health changed from {self.current_health.value} to {new_health.value}")
                
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(
                        event_type=EventType.HEALTH_STATUS_CHANGED,
                        source="orchestrator_monitor",
                        data={
                            "previous_health": self.current_health.value,
                            "current_health": new_health.value,
                            "health_factors": health_factors,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                
                self.current_health = new_health
            
            self.last_health_check = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error assessing health: {e}")
            self.current_health = HealthStatus.UNKNOWN
    
    async def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check for system-related alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.cpu_threshold:
            alerts.append(("high_cpu", f"CPU usage {metrics.cpu_usage:.1f}% exceeds threshold {self.cpu_threshold}%"))
        
        if metrics.memory_usage > self.memory_threshold:
            alerts.append(("high_memory", f"Memory usage {metrics.memory_usage:.1f}% exceeds threshold {self.memory_threshold}%"))
        
        if metrics.disk_usage > self.disk_threshold:
            alerts.append(("high_disk", f"Disk usage {metrics.disk_usage:.1f}% exceeds threshold {self.disk_threshold}%"))
        
        # Send alerts with cooldown
        for alert_type, message in alerts:
            await self._send_alert_with_cooldown(alert_type, message, "system")
    
    async def _check_orchestrator_alerts(self, metrics: OrchestratorMetrics) -> None:
        """Check for orchestrator-related alerts."""
        alerts = []
        
        if metrics.event_queue_size > self.queue_size_threshold:
            alerts.append(("high_queue_size", f"Event queue size {metrics.event_queue_size} exceeds threshold {self.queue_size_threshold}"))
        
        # Check failure rate
        total_workflows = metrics.completed_workflows + metrics.failed_workflows
        if total_workflows > 0:
            failure_rate = metrics.failed_workflows / total_workflows
            if failure_rate > 0.1:  # 10% failure rate
                alerts.append(("high_failure_rate", f"Workflow failure rate {failure_rate:.2%} is concerning"))
        
        # Send alerts with cooldown
        for alert_type, message in alerts:
            await self._send_alert_with_cooldown(alert_type, message, "orchestrator")
    
    async def _send_alert_with_cooldown(self, alert_type: str, message: str, category: str) -> None:
        """Send alert with cooldown to prevent spam."""
        now = datetime.now(timezone.utc)
        last_alert_time = self.last_alerts.get(alert_type)
        
        if last_alert_time is None or (now - last_alert_time).total_seconds() > self.alert_cooldown:
            self.logger.warning(f"Alert: {message}")
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.ORCHESTRATOR_ALERT,
                    source="orchestrator_monitor",
                    data={
                        "alert_type": alert_type,
                        "message": message,
                        "category": category,
                        "severity": "warning",
                        "timestamp": now.isoformat()
                    }
                ))
            
            self.last_alerts[alert_type] = now
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
        latest_orchestrator = self.orchestrator_metrics_history[-1] if self.orchestrator_metrics_history else None
        
        return {
            "health_status": self.current_health.value,
            "is_monitoring": self.is_monitoring,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "system_metrics": asdict(latest_system) if latest_system else None,
            "orchestrator_metrics": asdict(latest_orchestrator) if latest_orchestrator else None,
            "metrics_history_count": {
                "system": len(self.system_metrics_history),
                "orchestrator": len(self.orchestrator_metrics_history)
            }
        }
    
    async def get_metrics_history(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history."""
        if metric_type == "system":
            history = self.system_metrics_history[-limit:]
            return [asdict(metrics) for metrics in history]
        elif metric_type == "orchestrator":
            history = self.orchestrator_metrics_history[-limit:]
            return [asdict(metrics) for metrics in history]
        else:
            return []
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            "current_health": self.current_health.value,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "monitoring_active": self.is_monitoring,
            "alert_thresholds": {
                "cpu": self.cpu_threshold,
                "memory": self.memory_threshold,
                "disk": self.disk_threshold,
                "queue_size": self.queue_size_threshold
            },
            "recent_alerts": len([
                alert_time for alert_time in self.last_alerts.values()
                if (datetime.now(timezone.utc) - alert_time).total_seconds() < 3600  # Last hour
            ])
        }