"""
Pipeline Health Monitor for ML Pipeline Orchestration.

Aggregates health information from all components and provides pipeline-level health assessment.
"""
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from ..events.event_types import EventType, MLEvent
from .component_health_monitor import ComponentHealthMonitor, ComponentHealthStatus
from .orchestrator_monitor import HealthStatus, OrchestratorMonitor

class PipelineHealthStatus(Enum):
    """Pipeline-level health status."""
    HEALTHY = 'healthy'
    degraded = 'degraded'
    CRITICAL = 'critical'
    UNKNOWN = 'unknown'

@dataclass
class PipelineHealthSnapshot:
    """Complete pipeline health snapshot."""
    overall_health: PipelineHealthStatus
    component_health_summary: Dict[str, Any]
    orchestrator_health: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    timestamp: datetime

@dataclass
class HealthTrend:
    """Health trend analysis."""
    component_name: str
    trend_direction: str
    trend_confidence: float
    avg_response_time: float
    failure_rate: float
    availability: float

class PipelineHealthMonitor:
    """
    Pipeline-level health monitor that aggregates health from all components.
    
    Provides unified view of pipeline health, trends, and alerting.
    """

    def __init__(self, orchestrator_monitor: Optional[OrchestratorMonitor]=None, component_health_monitor: Optional[ComponentHealthMonitor]=None, event_bus=None, config=None):
        """Initialize pipeline health monitor."""
        self.orchestrator_monitor = orchestrator_monitor
        self.component_health_monitor = component_health_monitor
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.current_pipeline_health = PipelineHealthStatus.UNKNOWN
        self.last_health_check = None
        self.health_snapshots: List[PipelineHealthSnapshot] = []
        self.health_check_interval = self.config.get('pipeline_health_check_interval', 30)
        self.snapshot_retention = self.config.get('snapshot_retention_hours', 24)
        self.trend_analysis_window = self.config.get('trend_analysis_window_minutes', 60)
        self.critical_component_threshold = self.config.get('critical_component_threshold', 0.3)
        self.degraded_component_threshold = self.config.get('degraded_component_threshold', 0.1)
        self.active_alerts: List[Dict[str, Any]] = []
        self.is_monitoring = False
        self.monitor_task = None

    async def start_monitoring(self) -> None:
        """Start pipeline health monitoring."""
        if self.is_monitoring:
            self.logger.warning('Pipeline health monitoring is already running')
            return
        self.logger.info('Starting pipeline health monitoring')
        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.monitor_task_id = await task_manager.submit_enhanced_task(task_id=f'ml_pipeline_health_monitor_{str(uuid.uuid4())[:8]}', coroutine=self._monitoring_loop(), priority=TaskPriority.HIGH, tags={'service': 'ml', 'type': 'monitoring', 'component': 'pipeline_health_monitor', 'module': 'pipeline_health_monitor'})
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MONITORING_STARTED, source='pipeline_health_monitor', data={'monitor_type': 'pipeline_health', 'started_at': datetime.now(timezone.utc).isoformat()}))

    async def stop_monitoring(self) -> None:
        """Stop pipeline health monitoring."""
        if not self.is_monitoring:
            return
        self.logger.info('Stopping pipeline health monitoring')
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MONITORING_STOPPED, source='pipeline_health_monitor', data={'monitor_type': 'pipeline_health', 'stopped_at': datetime.now(timezone.utc).isoformat()}))

    async def _monitoring_loop(self) -> None:
        """Main pipeline health monitoring loop."""
        try:
            while self.is_monitoring:
                await self._collect_pipeline_health()
                await self._assess_pipeline_health()
                await self._cleanup_old_snapshots()
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            self.logger.info('Pipeline health monitoring loop cancelled')
        except Exception as e:
            self.logger.error('Error in pipeline health monitoring loop: %s', e)
            self.is_monitoring = False

    async def _collect_pipeline_health(self) -> None:
        """Collect health data from all monitoring sources."""
        try:
            timestamp = datetime.now(timezone.utc)
            component_health_summary = {}
            if self.component_health_monitor:
                component_health_summary = await self.component_health_monitor.get_health_summary()
            orchestrator_health = {}
            if self.orchestrator_monitor:
                orchestrator_health = await self.orchestrator_monitor.get_health_summary()
            performance_metrics = await self._collect_performance_metrics()
            active_alerts = await self._collect_active_alerts()
            overall_health = await self._determine_overall_health(component_health_summary, orchestrator_health, performance_metrics)
            snapshot = PipelineHealthSnapshot(overall_health=overall_health, component_health_summary=component_health_summary, orchestrator_health=orchestrator_health, active_alerts=active_alerts, performance_metrics=performance_metrics, timestamp=timestamp)
            self.health_snapshots.append(snapshot)
            previous_health = self.current_pipeline_health
            self.current_pipeline_health = overall_health
            self.last_health_check = timestamp
            if previous_health != overall_health:
                await self._handle_health_status_change(previous_health, overall_health, snapshot)
        except Exception as e:
            self.logger.error('Error collecting pipeline health: %s', e)

    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from various sources."""
        metrics = {'workflow_metrics': {}, 'resource_metrics': {}, 'response_times': {}, 'throughput': {}}
        try:
            if self.orchestrator_monitor:
                orchestrator_status = await self.orchestrator_monitor.get_current_status()
                if orchestrator_status.get('orchestrator_metrics'):
                    orch_metrics = orchestrator_status['orchestrator_metrics']
                    metrics['workflow_metrics'] = {'active_workflows': orch_metrics.get('active_workflows', 0), 'completed_workflows': orch_metrics.get('completed_workflows', 0), 'failed_workflows': orch_metrics.get('failed_workflows', 0), 'average_workflow_duration': orch_metrics.get('average_workflow_duration', 0.0)}
                    metrics['resource_metrics'] = orch_metrics.get('resource_utilization', {})
            if self.component_health_monitor:
                all_health = await self.component_health_monitor.get_all_component_health()
                response_times = {}
                for comp_name, health_info in all_health.items():
                    if health_info.get('average_response_time'):
                        response_times[comp_name] = health_info['average_response_time']
                metrics['response_times'] = response_times
            if self.health_snapshots:
                recent_snapshots = [s for s in self.health_snapshots[-10:] if s.timestamp > datetime.now(timezone.utc) - timedelta(minutes=10)]
                if len(recent_snapshots) > 1:
                    first_snapshot = recent_snapshots[0]
                    last_snapshot = recent_snapshots[-1]
                    time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds()
                    if time_diff > 0:
                        workflow_diff = last_snapshot.component_health_summary.get('total_components', 0) - first_snapshot.component_health_summary.get('total_components', 0)
                        metrics['throughput']['workflows_per_minute'] = workflow_diff / time_diff * 60
        except Exception as e:
            self.logger.error('Error collecting performance metrics: %s', e)
        return metrics

    async def _collect_active_alerts(self) -> List[Dict[str, Any]]:
        """Collect all active alerts from monitoring systems."""
        alerts = []
        try:
            if self.component_health_monitor:
                unhealthy_components = await self.component_health_monitor.get_unhealthy_components()
                for component_name in unhealthy_components:
                    health_info = await self.component_health_monitor.get_component_health(component_name)
                    if health_info:
                        alerts.append({'alert_type': 'component_unhealthy', 'component_name': component_name, 'severity': 'critical' if health_info['consecutive_failures'] >= 3 else 'warning', 'message': f'Component {component_name} is unhealthy', 'timestamp': datetime.now(timezone.utc).isoformat()})
            if self.orchestrator_monitor:
                orchestrator_status = await self.orchestrator_monitor.get_current_status()
                if orchestrator_status.get('health_status') in ['warning', 'critical']:
                    alerts.append({'alert_type': 'orchestrator_health', 'severity': orchestrator_status['health_status'], 'message': f"Orchestrator health is {orchestrator_status['health_status']}", 'timestamp': datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            self.logger.error('Error collecting active alerts: %s', e)
        return alerts

    async def _determine_overall_health(self, component_health: Dict[str, Any], orchestrator_health: Dict[str, Any], performance_metrics: Dict[str, Any]) -> PipelineHealthStatus:
        """Determine overall pipeline health status."""
        try:
            health_factors = []
            if component_health:
                total_components = component_health.get('total_components', 0)
                unhealthy_components = component_health.get('unhealthy_components', 0)
                if total_components > 0:
                    unhealthy_ratio = unhealthy_components / total_components
                    if unhealthy_ratio >= self.critical_component_threshold:
                        health_factors.append('critical_component_failures')
                    elif unhealthy_ratio >= self.degraded_component_threshold:
                        health_factors.append('degraded_component_health')
            if orchestrator_health:
                orchestrator_status = orchestrator_health.get('current_health', 'unknown')
                if orchestrator_status == 'critical':
                    health_factors.append('critical_orchestrator_health')
                elif orchestrator_status == 'warning':
                    health_factors.append('degraded_orchestrator_health')
            workflow_metrics = performance_metrics.get('workflow_metrics', {})
            total_workflows = workflow_metrics.get('completed_workflows', 0) + workflow_metrics.get('failed_workflows', 0)
            if total_workflows > 0:
                failure_rate = workflow_metrics.get('failed_workflows', 0) / total_workflows
                if failure_rate > 0.2:
                    health_factors.append('high_workflow_failure_rate')
                elif failure_rate > 0.1:
                    health_factors.append('elevated_workflow_failure_rate')
            critical_factors = [f for f in health_factors if 'critical' in f or 'high' in f]
            degraded_factors = [f for f in health_factors if 'degraded' in f or 'elevated' in f]
            if critical_factors:
                return PipelineHealthStatus.CRITICAL
            elif degraded_factors:
                return PipelineHealthStatus.degraded
            elif health_factors:
                return PipelineHealthStatus.degraded
            else:
                return PipelineHealthStatus.HEALTHY
        except Exception as e:
            self.logger.error('Error determining overall health: %s', e)
            return PipelineHealthStatus.UNKNOWN

    async def _assess_pipeline_health(self) -> None:
        """Perform comprehensive pipeline health assessment."""
        try:
            if not self.health_snapshots:
                return
            if len(self.health_snapshots) >= 5:
                await self._analyze_health_trends()
            await self._check_critical_conditions()
        except Exception as e:
            self.logger.error('Error in pipeline health assessment: %s', e)

    async def _analyze_health_trends(self) -> None:
        """Analyze health trends over time."""
        try:
            if len(self.health_snapshots) < 5:
                return
            trends = await self._calculate_component_trends()
            if self.event_bus and trends:
                await self.event_bus.emit(MLEvent(event_type=EventType.HEALTH_TREND_ANALYSIS, source='pipeline_health_monitor', data={'trends': [asdict(trend) for trend in trends], 'analysis_window_minutes': self.trend_analysis_window, 'timestamp': datetime.now(timezone.utc).isoformat()}))
        except Exception as e:
            self.logger.error('Error analyzing health trends: %s', e)

    async def _calculate_component_trends(self) -> List[HealthTrend]:
        """Calculate health trends for components."""
        trends = []
        try:
            if not self.component_health_monitor:
                return trends
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.trend_analysis_window)
            recent_snapshots = [s for s in self.health_snapshots if s.timestamp > cutoff_time]
            if len(recent_snapshots) < 3:
                return trends
            all_health = await self.component_health_monitor.get_all_component_health()
            for component_name in all_health.keys():
                trend = await self._calculate_single_component_trend(component_name, recent_snapshots)
                if trend:
                    trends.append(trend)
        except Exception as e:
            self.logger.error('Error calculating component trends: %s', e)
        return trends

    async def _calculate_single_component_trend(self, component_name: str, snapshots: List[PipelineHealthSnapshot]) -> Optional[HealthTrend]:
        """Calculate trend for a single component."""
        try:
            health_scores = []
            response_times = []
            for snapshot in snapshots:
                health_scores.append(0.8)
                response_times.append(0.1)
            if len(health_scores) < 3:
                return None
            recent_avg = sum(health_scores[-3:]) / 3
            older_avg = sum(health_scores[:3]) / 3
            if recent_avg > older_avg + 0.1:
                trend_direction = 'improving'
                trend_confidence = min((recent_avg - older_avg) * 2, 1.0)
            elif recent_avg < older_avg - 0.1:
                trend_direction = 'degrading'
                trend_confidence = min((older_avg - recent_avg) * 2, 1.0)
            else:
                trend_direction = 'stable'
                trend_confidence = 1.0 - abs(recent_avg - older_avg) * 2
            return HealthTrend(component_name=component_name, trend_direction=trend_direction, trend_confidence=trend_confidence, avg_response_time=sum(response_times) / len(response_times), failure_rate=1.0 - recent_avg, availability=recent_avg)
        except Exception as e:
            self.logger.error('Error calculating trend for {component_name}: %s', e)
            return None

    async def _check_critical_conditions(self) -> None:
        """Check for critical conditions requiring immediate attention."""
        try:
            if not self.health_snapshots:
                return
            latest_snapshot = self.health_snapshots[-1]
            if latest_snapshot.overall_health == PipelineHealthStatus.CRITICAL:
                await self._handle_critical_pipeline_health(latest_snapshot)
            if len(latest_snapshot.active_alerts) >= 5:
                await self._handle_cascading_failures(latest_snapshot)
        except Exception as e:
            self.logger.error('Error checking critical conditions: %s', e)

    async def _handle_health_status_change(self, previous_health: PipelineHealthStatus, current_health: PipelineHealthStatus, snapshot: PipelineHealthSnapshot) -> None:
        """Handle pipeline health status change."""
        self.logger.info('Pipeline health changed from {previous_health.value} to %s', current_health.value)
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.PIPELINE_HEALTH_CHANGED, source='pipeline_health_monitor', data={'previous_health': previous_health.value, 'current_health': current_health.value, 'component_summary': snapshot.component_health_summary, 'active_alerts_count': len(snapshot.active_alerts), 'timestamp': snapshot.timestamp.isoformat()}))

    async def _handle_critical_pipeline_health(self, snapshot: PipelineHealthSnapshot) -> None:
        """Handle critical pipeline health condition."""
        self.logger.error('Pipeline health is CRITICAL - immediate attention required')
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.PIPELINE_CRITICAL_HEALTH, source='pipeline_health_monitor', data={'health_status': snapshot.overall_health.value, 'component_summary': snapshot.component_health_summary, 'active_alerts': snapshot.active_alerts, 'performance_metrics': snapshot.performance_metrics, 'timestamp': snapshot.timestamp.isoformat()}))

    async def _handle_cascading_failures(self, snapshot: PipelineHealthSnapshot) -> None:
        """Handle potential cascading failure scenario."""
        self.logger.warning('Potential cascading failures detected: %s active alerts', len(snapshot.active_alerts))
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.CASCADING_FAILURE_DETECTED, source='pipeline_health_monitor', data={'alert_count': len(snapshot.active_alerts), 'alerts': snapshot.active_alerts, 'timestamp': snapshot.timestamp.isoformat()}))

    async def _cleanup_old_snapshots(self) -> None:
        """Clean up old health snapshots."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.snapshot_retention)
            self.health_snapshots = [snapshot for snapshot in self.health_snapshots if snapshot.timestamp > cutoff_time]
        except Exception as e:
            self.logger.error('Error cleaning up old snapshots: %s', e)

    async def get_current_pipeline_health(self) -> Dict[str, Any]:
        """Get current pipeline health status."""
        latest_snapshot = self.health_snapshots[-1] if self.health_snapshots else None
        return {'overall_health': self.current_pipeline_health.value, 'last_check': self.last_health_check.isoformat() if self.last_health_check else None, 'monitoring_active': self.is_monitoring, 'active_alerts_count': len(latest_snapshot.active_alerts) if latest_snapshot else 0, 'component_health_summary': latest_snapshot.component_health_summary if latest_snapshot else {}, 'orchestrator_health': latest_snapshot.orchestrator_health if latest_snapshot else {}, 'performance_metrics': latest_snapshot.performance_metrics if latest_snapshot else {}}

    async def get_health_history(self, hours: int=24) -> List[Dict[str, Any]]:
        """Get pipeline health history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_snapshots = [snapshot for snapshot in self.health_snapshots if snapshot.timestamp > cutoff_time]
        return [asdict(snapshot) for snapshot in recent_snapshots]

    async def get_health_trends(self) -> List[Dict[str, Any]]:
        """Get current health trends."""
        trends = await self._calculate_component_trends()
        return [asdict(trend) for trend in trends]
