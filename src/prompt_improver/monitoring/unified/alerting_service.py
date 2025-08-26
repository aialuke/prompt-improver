"""Alerting Service.

Focused service for managing alerts, notifications, and escalation policies.
Extracted from unified_monitoring_manager.py.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True

    alerting_tracer = trace.get_tracer(__name__ + ".alerting")
    alerting_meter = metrics.get_meter(__name__ + ".alerting")

    alerts_created_total = alerting_meter.create_counter(
        "alerts_created_total",
        description="Total alerts created by severity",
        unit="1",
    )

    alerts_resolved_total = alerting_meter.create_counter(
        "alerts_resolved_total",
        description="Total alerts resolved",
        unit="1",
    )

    alert_processing_duration = alerting_meter.create_histogram(
        "alert_processing_duration_seconds",
        description="Time taken to process alerts",
        unit="s",
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    alerting_tracer = None
    alerting_meter = None
    alerts_created_total = None
    alerts_resolved_total = None
    alert_processing_duration = None

import contextlib

from prompt_improver.monitoring.unified.types import MonitoringConfig
from prompt_improver.shared.interfaces.protocols.monitoring import (
    MonitoringRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status values."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source_component: str | None = None
    affected_components: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    metric_name: str
    enabled: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    cooldown_seconds: int = 300
    escalation_delay_seconds: int = 1800


@dataclass
class AlertStats:
    """Alert statistics."""
    total_alerts: int = 0
    active_alerts: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0
    resolved_alerts: int = 0
    suppressed_alerts: int = 0
    avg_resolution_time_minutes: float = 0.0


class AlertingService:
    """Service for managing alerts, notifications, and escalations.

    Provides:
    - Alert creation and lifecycle management
    - Alert rules and threshold monitoring
    - Escalation policies and notifications
    - Alert suppression and grouping
    - Alert metrics and reporting
    """

    def __init__(
        self,
        config: MonitoringConfig,
        repository: MonitoringRepositoryProtocol | None = None,
    ) -> None:
        self.config = config
        self.repository = repository

        # Alert storage
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: dict[str, AlertRule] = {}

        # Statistics
        self.alert_stats = AlertStats()

        # Notification handlers
        self._notification_handlers: list[Callable] = []
        self._escalation_handlers: list[Callable] = []

        # Suppression and grouping
        self._suppression_rules: dict[str, dict] = {}
        self._alert_groups: dict[str, list[str]] = defaultdict(list)

        # Background monitoring
        self.is_monitoring = False
        self.monitor_task: asyncio.Task | None = None
        self.monitor_task_id: str | None = None

        # Alert storm detection
        self._alert_rate_window = deque(maxlen=100)
        self._storm_threshold = 10  # alerts per minute
        self._storm_cooldown = 600  # 10 minutes
        self._last_storm_time: datetime | None = None

        # Initialize default alert rules
        self._setup_default_alert_rules()

        logger.info("AlertingService initialized")

    def _setup_default_alert_rules(self) -> None:
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="system_cpu_high",
                name="High CPU Usage",
                condition="cpu_usage > threshold",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                metric_name="system.cpu.usage_percent",
                tags={"category": "system"},
            ),
            AlertRule(
                rule_id="system_memory_high",
                name="High Memory Usage",
                condition="memory_usage > threshold",
                severity=AlertSeverity.CRITICAL,
                threshold=90.0,
                metric_name="system.memory.usage_percent",
                tags={"category": "system"},
            ),
            AlertRule(
                rule_id="health_component_down",
                name="Component Unhealthy",
                condition="component_status == unhealthy",
                severity=AlertSeverity.CRITICAL,
                threshold=1.0,
                metric_name="health.component.status",
                tags={"category": "health"},
            ),
            AlertRule(
                rule_id="cache_hit_rate_low",
                name="Low Cache Hit Rate",
                condition="cache_hit_rate < threshold",
                severity=AlertSeverity.WARNING,
                threshold=70.0,
                metric_name="cache.hit_rate",
                tags={"category": "performance"},
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    async def start_monitoring(self) -> None:
        """Start alert monitoring background task."""
        if self.is_monitoring:
            logger.warning("Alert monitoring is already running")
            return

        logger.info("Starting alert monitoring")
        self.is_monitoring = True

        try:
            # Start monitoring loop
            self.monitor_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Alert monitoring started successfully")

        except Exception as e:
            logger.exception(f"Failed to start alert monitoring: {e}")
            self.is_monitoring = False
            raise

    async def stop_monitoring(self) -> None:
        """Stop alert monitoring background task."""
        if not self.is_monitoring:
            return

        logger.info("Stopping alert monitoring")
        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
            self.monitor_task = None

        logger.info("Alert monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        try:
            while self.is_monitoring:
                await self._process_alert_escalations()
                await self._cleanup_old_alerts()
                await self._update_alert_statistics()
                await self._check_alert_storms()
                await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            logger.info("Alert monitoring loop cancelled")
        except Exception as e:
            logger.exception(f"Error in alert monitoring loop: {e}")
            self.is_monitoring = False

    async def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        source_component: str | None = None,
        affected_components: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create a new alert."""
        start_time = time.time()

        try:
            alert_id = f"alert_{int(datetime.now().timestamp())}_{alert_type}_{str(uuid.uuid4())[:8]}"
            timestamp = datetime.now(UTC)

            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                status=AlertStatus.ACTIVE,
                title=title,
                description=description,
                source_component=source_component,
                affected_components=affected_components or [],
                metrics=metrics or {},
                tags=tags or {},
                created_at=timestamp,
                updated_at=timestamp,
            )

            # Check for duplicate/similar alerts
            if await self._is_duplicate_alert(alert):
                logger.debug(f"Suppressing duplicate alert: {alert_id}")
                return ""

            # Check suppression rules
            if await self._is_suppressed_alert(alert):
                logger.debug(f"Alert suppressed by rules: {alert_id}")
                alert.status = AlertStatus.SUPPRESSED
                return alert_id

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Update statistics
            self._update_alert_stats(alert, "created")

            # Store in repository
            if self.repository:
                await self.repository.store_alert(alert)

            # Track alert rate for storm detection
            self._alert_rate_window.append(time.time())

            # Send notifications
            await self._send_notifications(alert)

            # Record telemetry
            if OPENTELEMETRY_AVAILABLE and alerts_created_total:
                alerts_created_total.add(
                    1,
                    {"severity": severity.value, "type": alert_type}
                )

            if OPENTELEMETRY_AVAILABLE and alert_processing_duration:
                alert_processing_duration.record(
                    time.time() - start_time,
                    {"operation": "create_alert"}
                )

            logger.warning(f"Alert created: {title} ({severity.value}) - {description}")
            return alert_id

        except Exception as e:
            logger.exception(f"Error creating alert: {e}")
            return ""

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        note: str | None = None,
    ) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False

        try:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(UTC)
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now(UTC)

            if note:
                alert.description += f"\n\nAcknowledgment note: {note}"

            # Update statistics
            self._update_alert_stats(alert, "acknowledged")

            # Store in repository
            if self.repository:
                await self.repository.update_alert(alert)

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.exception(f"Error acknowledging alert {alert_id}: {e}")
            return False

    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: str | None = None,
    ) -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False

        try:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(UTC)
            alert.resolved_by = resolved_by
            alert.updated_at = datetime.now(UTC)

            if resolution_note:
                alert.description += f"\n\nResolution note: {resolution_note}"

            # Remove from active alerts
            del self.active_alerts[alert_id]

            # Update statistics
            self._update_alert_stats(alert, "resolved")

            # Calculate resolution time
            if alert.created_at:
                resolution_time = (alert.resolved_at - alert.created_at).total_seconds() / 60
                self._update_avg_resolution_time(resolution_time)

            # Store in repository
            if self.repository:
                await self.repository.update_alert(alert)

            # Record telemetry
            if OPENTELEMETRY_AVAILABLE and alerts_resolved_total:
                alerts_resolved_total.add(
                    1,
                    {"severity": alert.severity.value, "type": alert.alert_type}
                )

            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.exception(f"Error resolving alert {alert_id}: {e}")
            return False

    async def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Check if alert is a duplicate of an existing active alert."""
        for existing_alert in self.active_alerts.values():
            if (
                existing_alert.alert_type == alert.alert_type
                and existing_alert.source_component == alert.source_component
                and existing_alert.severity == alert.severity
                and existing_alert.status == AlertStatus.ACTIVE
            ):
                # Check if created within cooldown period
                time_diff = (alert.created_at - existing_alert.created_at).total_seconds()
                cooldown = self.alert_rules.get(alert.alert_type, AlertRule("", "", "", AlertSeverity.INFO, 0, "")).cooldown_seconds

                if time_diff < cooldown:
                    return True

        return False

    async def _is_suppressed_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed based on rules."""
        # Check for alert storm suppression
        if self._is_alert_storm():
            return True

        # Check custom suppression rules
        for rule in self._suppression_rules.values():
            if self._matches_suppression_rule(alert, rule):
                return True

        return False

    def _is_alert_storm(self) -> bool:
        """Check if we're in an alert storm condition."""
        if not self._alert_rate_window:
            return False

        # Check if we're in storm cooldown
        if (
            self._last_storm_time
            and (datetime.now(UTC) - self._last_storm_time).total_seconds() < self._storm_cooldown
        ):
            return True

        # Calculate alerts per minute in recent window
        now = time.time()
        recent_alerts = [t for t in self._alert_rate_window if now - t < 60]
        alerts_per_minute = len(recent_alerts)

        if alerts_per_minute > self._storm_threshold:
            self._last_storm_time = datetime.now(UTC)
            logger.warning(f"Alert storm detected: {alerts_per_minute} alerts in last minute")
            return True

        return False

    def _matches_suppression_rule(self, alert: Alert, rule: dict[str, Any]) -> bool:
        """Check if alert matches a suppression rule."""
        # Implement suppression rule matching logic
        return False

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.exception(f"Notification handler failed: {e}")

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations based on rules."""
        now = datetime.now(UTC)

        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                # Check if escalation is needed
                time_since_created = (now - alert.created_at).total_seconds()
                rule = self.alert_rules.get(alert.alert_type)

                if rule and time_since_created > rule.escalation_delay_seconds:
                    await self._escalate_alert(alert)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to higher severity or different handlers."""
        try:
            logger.warning(f"Escalating alert {alert.alert_id}: {alert.title}")

            # Run escalation handlers
            for handler in self._escalation_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.exception(f"Escalation handler failed: {e}")

        except Exception as e:
            logger.exception(f"Alert escalation failed: {e}")

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved/suppressed alerts."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        # Clean up alert history
        original_len = len(self.alert_history)
        self.alert_history = deque([
            alert for alert in self.alert_history
            if alert.created_at > cutoff_time
        ], maxlen=10000)

        cleaned_count = original_len - len(self.alert_history)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old alerts from history")

    async def _update_alert_statistics(self) -> None:
        """Update alert statistics."""
        self.alert_stats.active_alerts = len(self.active_alerts)

        # Count by severity
        critical_count = sum(
            1 for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL
        )
        warning_count = sum(
            1 for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.WARNING
        )

        self.alert_stats.critical_alerts = critical_count
        self.alert_stats.warning_alerts = warning_count

    async def _check_alert_storms(self) -> None:
        """Check for and handle alert storms."""
        if self._is_alert_storm():
            # Implement storm handling logic
            # This could involve suppressing alerts, sending summary notifications, etc.
            logger.warning("Alert storm in progress - suppressing non-critical alerts")

    def _update_alert_stats(self, alert: Alert, action: str) -> None:
        """Update alert statistics for an action."""
        if action == "created":
            self.alert_stats.total_alerts += 1
            if alert.severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts += 1
            elif alert.severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts += 1
        elif action == "resolved":
            self.alert_stats.resolved_alerts += 1
            if alert.severity == AlertSeverity.CRITICAL:
                self.alert_stats.critical_alerts = max(0, self.alert_stats.critical_alerts - 1)
            elif alert.severity == AlertSeverity.WARNING:
                self.alert_stats.warning_alerts = max(0, self.alert_stats.warning_alerts - 1)

    def _update_avg_resolution_time(self, resolution_time_minutes: float) -> None:
        """Update average resolution time."""
        current_avg = self.alert_stats.avg_resolution_time_minutes
        resolved_count = self.alert_stats.resolved_alerts

        if resolved_count > 0:
            self.alert_stats.avg_resolution_time_minutes = (
                (current_avg * (resolved_count - 1) + resolution_time_minutes) / resolved_count
            )

    def add_notification_handler(self, handler: Callable) -> None:
        """Add a notification handler."""
        self._notification_handlers.append(handler)
        logger.info(f"Added notification handler: {handler.__name__}")

    def add_escalation_handler(self, handler: Callable) -> None:
        """Add an escalation handler."""
        self._escalation_handlers.append(handler)
        logger.info(f"Added escalation handler: {handler.__name__}")

    def add_suppression_rule(self, rule_id: str, rule: dict[str, Any]) -> None:
        """Add an alert suppression rule."""
        self._suppression_rules[rule_id] = rule
        logger.info(f"Added suppression rule: {rule_id}")

    def get_alert_statistics(self) -> AlertStats:
        """Get current alert statistics."""
        return self.alert_stats

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return alerts

    def get_alert_history(self, hours: int = 24) -> list[Alert]:
        """Get alert history for specified time window."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        return [
            alert for alert in self.alert_history
            if alert.created_at > cutoff_time
        ]
