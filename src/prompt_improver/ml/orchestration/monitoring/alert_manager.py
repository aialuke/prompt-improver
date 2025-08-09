"""
Alert Manager for ML Pipeline Orchestration.

Handles alerting for component failures, performance degradation, and resource exhaustion.
"""
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Set
import uuid
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from ..events.event_types import EventType, MLEvent

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
    emergency = 'emergency'

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = 'active'
    acknowledged = 'acknowledged'
    resolved = 'resolved'
    suppressed = 'suppressed'

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source_component: Optional[str]
    affected_components: List[str]
    metrics: Dict[str, Any]
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_at: Optional[datetime]
    resolution_note: Optional[str]

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    rule_name: str
    metric_name: str
    condition: str
    threshold: float
    duration_seconds: int
    severity: AlertSeverity
    enabled: bool
    tags: Dict[str, str]

@dataclass
class AlertStats:
    """Alert statistics."""
    total_alerts: int
    active_alerts: int
    critical_alerts: int
    warning_alerts: int
    alerts_last_hour: int
    alerts_last_day: int
    avg_resolution_time_hours: float
    top_alert_types: List[Dict[str, Any]]

class AlertManager:
    """
    Manages alerting for the ML pipeline orchestration system.
    
    Handles alert creation, escalation, acknowledgment, and resolution.
    Integrates with external alerting systems.
    """

    def __init__(self, event_bus=None, config=None):
        """Initialize alert manager."""
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_retention_days = self.config.get('alert_retention_days', 30)
        self.max_alerts_in_memory = self.config.get('max_alerts_in_memory', 1000)
        self.escalation_timeout_minutes = self.config.get('escalation_timeout_minutes', 30)
        self.suppression_timeout_minutes = self.config.get('suppression_timeout_minutes', 60)
        self.enable_email_alerts = self.config.get('enable_email_alerts', False)
        self.enable_slack_alerts = self.config.get('enable_slack_alerts', False)
        self.enable_pagerduty_alerts = self.config.get('enable_pagerduty_alerts', False)
        self.is_monitoring = False
        self.monitor_task = None
        self.alert_stats = AlertStats(total_alerts=0, active_alerts=0, critical_alerts=0, warning_alerts=0, alerts_last_hour=0, alerts_last_day=0, avg_resolution_time_hours=0.0, top_alert_types=[])
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [AlertRule(rule_id='component_unhealthy', rule_name='Component Unhealthy', metric_name='component_health_status', condition='equals', threshold=0, duration_seconds=60, severity=AlertSeverity.CRITICAL, enabled=True, tags={'category': 'component_health'}), AlertRule(rule_id='high_error_rate', rule_name='High Error Rate', metric_name='error_rate_percentage', condition='greater_than', threshold=10.0, duration_seconds=300, severity=AlertSeverity.WARNING, enabled=True, tags={'category': 'performance'}), AlertRule(rule_id='high_response_time', rule_name='High Response Time', metric_name='average_response_time_ms', condition='greater_than', threshold=1000.0, duration_seconds=180, severity=AlertSeverity.WARNING, enabled=True, tags={'category': 'performance'}), AlertRule(rule_id='resource_exhaustion', rule_name='Resource Exhaustion', metric_name='resource_utilization_percentage', condition='greater_than', threshold=90.0, duration_seconds=120, severity=AlertSeverity.CRITICAL, enabled=True, tags={'category': 'resources'}), AlertRule(rule_id='workflow_failure_rate', rule_name='High Workflow Failure Rate', metric_name='workflow_failure_rate_percentage', condition='greater_than', threshold=20.0, duration_seconds=600, severity=AlertSeverity.CRITICAL, enabled=True, tags={'category': 'workflow'})]
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    async def start_monitoring(self) -> None:
        """Start alert monitoring."""
        if self.is_monitoring:
            self.logger.warning('Alert monitoring is already running')
            return
        self.logger.info('Starting alert monitoring')
        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.monitor_task_id = await task_manager.submit_enhanced_task(task_id=f'ml_alert_monitor_{str(uuid.uuid4())[:8]}', coroutine=self._monitoring_loop(), priority=TaskPriority.HIGH, tags={'service': 'ml', 'type': 'monitoring', 'component': 'alert_monitor', 'module': 'alert_manager'})
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MONITORING_STARTED, source='alert_manager', data={'monitor_type': 'alert_manager', 'started_at': datetime.now(timezone.utc).isoformat()}))

    async def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        if not self.is_monitoring:
            return
        self.logger.info('Stopping alert monitoring')
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        if self.event_bus:
            await self.event_bus.emit(MLEvent(event_type=EventType.MONITORING_STOPPED, source='alert_manager', data={'monitor_type': 'alert_manager', 'stopped_at': datetime.now(timezone.utc).isoformat()}))

    async def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        try:
            while self.is_monitoring:
                await self._process_escalations()
                await self._cleanup_alerts()
                await self._update_alert_stats()
                await self._check_alert_storms()
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.logger.info('Alert monitoring loop cancelled')
        except Exception as e:
            self.logger.error('Error in alert monitoring loop: %s', e)
            self.is_monitoring = False

    async def create_alert(self, alert_type: str, severity: AlertSeverity, title: str, description: str, source_component: str=None, affected_components: List[str]=None, metrics: Dict[str, Any]=None, tags: Dict[str, str]=None) -> str:
        """Create a new alert."""
        try:
            alert_id = f'alert_{int(datetime.now().timestamp())}_{alert_type}'
            timestamp = datetime.now(timezone.utc)
            alert = Alert(alert_id=alert_id, alert_type=alert_type, severity=severity, status=AlertStatus.ACTIVE, title=title, description=description, source_component=source_component, affected_components=affected_components or [], metrics=metrics or {}, tags=tags or {}, created_at=timestamp, updated_at=timestamp, acknowledged_at=None, acknowledged_by=None, resolved_at=None, resolution_note=None)
            duplicate_alert = await self._check_duplicate_alert(alert)
            if duplicate_alert:
                self.logger.debug('Suppressing duplicate alert: %s', alert_id)
                return duplicate_alert.alert_id
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.alert_stats.total_alerts += 1
            self.alert_stats.active_alerts += 1
            if severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts += 1
            elif severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts += 1
            self.logger.warning('Alert created: {title} ({severity.value}) - %s', description)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_CREATED, source='alert_manager', data={'alert': asdict(alert), 'timestamp': timestamp.isoformat()}))
            await self._send_external_notifications(alert)
            return alert_id
        except Exception as e:
            self.logger.error('Error creating alert: %s', e)
            return ''

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, note: str=None) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.active_alerts:
                self.logger.warning('Alert %s not found for acknowledgment', alert_id)
                return False
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.acknowledged
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now(timezone.utc)
            if note:
                alert.description += f'\n\nAcknowledgment note: {note}'
            self.logger.info('Alert {alert_id} acknowledged by %s', acknowledged_by)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_ACKNOWLEDGED, source='alert_manager', data={'alert_id': alert_id, 'acknowledged_by': acknowledged_by, 'note': note, 'timestamp': datetime.now(timezone.utc).isoformat()}))
            return True
        except Exception as e:
            self.logger.error('Error acknowledging alert {alert_id}: %s', e)
            return False

    async def resolve_alert(self, alert_id: str, resolution_note: str=None) -> bool:
        """Resolve an alert."""
        try:
            if alert_id not in self.active_alerts:
                self.logger.warning('Alert %s not found for resolution', alert_id)
                return False
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.resolved
            alert.resolved_at = datetime.now(timezone.utc)
            alert.resolution_note = resolution_note
            alert.updated_at = datetime.now(timezone.utc)
            del self.active_alerts[alert_id]
            self.alert_stats.active_alerts -= 1
            if alert.severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts -= 1
            elif alert.severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts -= 1
            self.logger.info('Alert {alert_id} resolved: %s', resolution_note or 'No note provided')
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_RESOLVED, source='alert_manager', data={'alert_id': alert_id, 'resolution_note': resolution_note, 'duration_minutes': (alert.resolved_at - alert.created_at).total_seconds() / 60, 'timestamp': datetime.now(timezone.utc).isoformat()}))
            return True
        except Exception as e:
            self.logger.error('Error resolving alert {alert_id}: %s', e)
            return False

    async def _check_duplicate_alert(self, new_alert: Alert) -> Optional[Alert]:
        """Check for duplicate alerts to prevent spam."""
        for alert in self.active_alerts.values():
            if alert.alert_type == new_alert.alert_type and alert.source_component == new_alert.source_component and (alert.status == AlertStatus.ACTIVE):
                time_diff = (new_alert.created_at - alert.created_at).total_seconds()
                if time_diff < self.suppression_timeout_minutes * 60:
                    return alert
        return None

    async def _process_escalations(self) -> None:
        """Process alert escalations."""
        try:
            current_time = datetime.now(timezone.utc)
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    time_since_creation = (current_time - alert.created_at).total_seconds()
                    escalation_threshold = self.escalation_timeout_minutes * 60
                    if time_since_creation > escalation_threshold:
                        await self._escalate_alert(alert)
        except Exception as e:
            self.logger.error('Error processing escalations: %s', e)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an unacknowledged alert."""
        try:
            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.CRITICAL
                self.alert_stats.warning_alerts -= 1
                self.alert_stats.critical_alerts += 1
            elif alert.severity == AlertSeverity.CRITICAL:
                alert.severity = AlertSeverity.emergency
            alert.updated_at = datetime.now(timezone.utc)
            self.logger.warning('Alert {alert.alert_id} escalated to %s', alert.severity.value)
            await self._send_escalation_notifications(alert)
            if self.event_bus:
                await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_ESCALATED, source='alert_manager', data={'alert_id': alert.alert_id, 'new_severity': alert.severity.value, 'timestamp': datetime.now(timezone.utc).isoformat()}))
        except Exception as e:
            self.logger.error('Error escalating alert {alert.alert_id}: %s', e)

    async def _send_external_notifications(self, alert: Alert) -> None:
        """Send notifications to external systems."""
        try:
            if self.enable_email_alerts:
                await self._send_email_notification(alert)
            if self.enable_slack_alerts:
                await self._send_slack_notification(alert)
            if self.enable_pagerduty_alerts and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.emergency]:
                await self._send_pagerduty_notification(alert)
        except Exception as e:
            self.logger.error('Error sending external notifications: %s', e)

    async def _send_escalation_notifications(self, alert: Alert) -> None:
        """Send escalation notifications."""
        try:
            if self.enable_email_alerts:
                await self._send_email_notification(alert, is_escalation=True)
            if self.enable_slack_alerts:
                await self._send_slack_notification(alert, is_escalation=True)
            if self.enable_pagerduty_alerts:
                await self._send_pagerduty_notification(alert, is_escalation=True)
        except Exception as e:
            self.logger.error('Error sending escalation notifications: %s', e)

    async def _send_email_notification(self, alert: Alert, is_escalation: bool=False) -> None:
        """Send email notification."""
        try:
            subject = f"{('ESCALATED: ' if is_escalation else '')}{alert.severity.value.upper()}: {alert.title}"
            self.logger.debug('Email notification: %s', subject)
        except Exception as e:
            self.logger.error('Error sending email notification: %s', e)

    async def _send_slack_notification(self, alert: Alert, is_escalation: bool=False) -> None:
        """Send Slack notification."""
        try:
            message = f"{('🚨 ESCALATED: ' if is_escalation else '⚠️ ')}{alert.title}"
            self.logger.debug('Slack notification: %s', message)
        except Exception as e:
            self.logger.error('Error sending Slack notification: %s', e)

    async def _send_pagerduty_notification(self, alert: Alert, is_escalation: bool=False) -> None:
        """Send PagerDuty notification."""
        try:
            incident_key = f'ml_pipeline_{alert.alert_id}'
            self.logger.debug('PagerDuty notification: %s', incident_key)
        except Exception as e:
            self.logger.error('Error sending PagerDuty notification: %s', e)

    async def _cleanup_alerts(self) -> None:
        """Clean up old alerts."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.alert_retention_days)
            original_count = len(self.alert_history)
            self.alert_history = [alert for alert in self.alert_history if alert.created_at > cutoff_time]
            if len(self.alert_history) > self.max_alerts_in_memory:
                self.alert_history = self.alert_history[-self.max_alerts_in_memory:]
            cleaned_count = original_count - len(self.alert_history)
            if cleaned_count > 0:
                self.logger.debug('Cleaned up %s old alerts', cleaned_count)
        except Exception as e:
            self.logger.error('Error cleaning up alerts: %s', e)

    async def _update_alert_stats(self) -> None:
        """Update alert statistics."""
        try:
            current_time = datetime.now(timezone.utc)
            one_hour_ago = current_time - timedelta(hours=1)
            one_day_ago = current_time - timedelta(days=1)
            self.alert_stats.alerts_last_hour = sum((1 for alert in self.alert_history if alert.created_at > one_hour_ago))
            self.alert_stats.alerts_last_day = sum((1 for alert in self.alert_history if alert.created_at > one_day_ago))
            resolved_alerts = [alert for alert in self.alert_history if alert.resolved_at is not None and alert.created_at > one_day_ago]
            if resolved_alerts:
                total_resolution_time = sum(((alert.resolved_at - alert.created_at).total_seconds() for alert in resolved_alerts))
                self.alert_stats.avg_resolution_time_hours = total_resolution_time / len(resolved_alerts) / 3600
            alert_type_counts = {}
            for alert in self.alert_history:
                if alert.created_at > one_day_ago:
                    alert_type_counts[alert.alert_type] = alert_type_counts.get(alert.alert_type, 0) + 1
            self.alert_stats.top_alert_types = [{'alert_type': alert_type, 'count': count} for alert_type, count in sorted(alert_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        except Exception as e:
            self.logger.error('Error updating alert stats: %s', e)

    async def _check_alert_storms(self) -> None:
        """Check for alert storms and implement suppression."""
        try:
            current_time = datetime.now(timezone.utc)
            last_hour = current_time - timedelta(hours=1)
            recent_alerts = [alert for alert in self.alert_history if alert.created_at > last_hour]
            if len(recent_alerts) > 50:
                self.logger.warning('Alert storm detected: %s alerts in the last hour', len(recent_alerts))
                if self.event_bus:
                    await self.event_bus.emit(MLEvent(event_type=EventType.ALERT_STORM_DETECTED, source='alert_manager', data={'alert_count': len(recent_alerts), 'time_window_hours': 1, 'timestamp': current_time.isoformat()}))
        except Exception as e:
            self.logger.error('Error checking alert storms: %s', e)

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [asdict(alert) for alert in self.active_alerts.values()]

    async def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get specific alert by ID."""
        if alert_id in self.active_alerts:
            return asdict(self.active_alerts[alert_id])
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                return asdict(alert)
        return None

    async def get_alerts_by_component(self, component_name: str) -> List[Dict[str, Any]]:
        """Get alerts for specific component."""
        alerts = []
        for alert in self.active_alerts.values():
            if alert.source_component == component_name or component_name in alert.affected_components:
                alerts.append(asdict(alert))
        return alerts

    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return asdict(self.alert_stats)

    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules."""
        return [asdict(rule) for rule in self.alert_rules.values()]

    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule."""
        try:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info('Added alert rule: %s', rule.rule_name)
            return True
        except Exception as e:
            self.logger.error('Error adding alert rule: %s', e)
            return False

    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info('Removed alert rule: %s', rule_id)
                return True
            return False
        except Exception as e:
            self.logger.error('Error removing alert rule: %s', e)
            return False
