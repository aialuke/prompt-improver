"""Redis Alerting Service

Focused Redis alerting service for incident detection and notification management.
Designed for real-time incident response following SRE best practices with configurable thresholds.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry

from .protocols import RedisAlertingServiceProtocol
from .types import AlertEvent, AlertLevel, HealthMetrics

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Alerting metrics
ALERTS_TRIGGERED = _metrics_registry.get_or_create_counter(
    "redis_alerts_triggered_total",
    "Total Redis alerts triggered",
    ["alert_level", "alert_type"]
)

ALERTS_RESOLVED = _metrics_registry.get_or_create_counter(
    "redis_alerts_resolved_total", 
    "Total Redis alerts resolved",
    ["alert_level", "alert_type"]
)

ALERT_DURATION = _metrics_registry.get_or_create_histogram(
    "redis_alert_duration_seconds",
    "Redis alert duration in seconds",
    ["alert_level"]
)

ACTIVE_ALERTS_COUNT = _metrics_registry.get_or_create_gauge(
    "redis_active_alerts_count",
    "Number of active Redis alerts",
    ["alert_level"]
)


class RedisAlertingService:
    """Redis alerting service for incident detection and notification management.
    
    Provides comprehensive alerting with configurable thresholds, escalation policies,
    and notification channels following SRE incident response best practices.
    """
    
    def __init__(
        self,
        notification_channels: Optional[List[str]] = None,
        escalation_timeout_minutes: int = 15,
        alert_deduplication_window_minutes: int = 5
    ):
        """Initialize Redis alerting service.
        
        Args:
            notification_channels: List of notification channels to use
            escalation_timeout_minutes: Time before escalating unresolved alerts
            alert_deduplication_window_minutes: Window for alert deduplication
        """
        self.notification_channels = notification_channels or ["log"]
        self.escalation_timeout_minutes = escalation_timeout_minutes
        self.alert_deduplication_window_minutes = alert_deduplication_window_minutes
        
        # Alert state management
        self._active_alerts: Dict[str, AlertEvent] = {}
        self._alert_history: List[AlertEvent] = []
        self._max_history_size = 1000
        
        # Deduplication tracking
        self._recent_alerts: Set[str] = set()
        self._dedup_cleanup_task: Optional[asyncio.Task] = None
        
        # Default thresholds (SRE recommended values)
        self._thresholds = {
            "ping_latency_ms": {
                "warning": 50.0,
                "critical": 100.0,
            },
            "connection_utilization_percent": {
                "warning": 70.0,
                "critical": 85.0,
            },
            "hit_rate_percent": {
                "warning": 80.0,
                "critical": 50.0,
            },
            "memory_usage_percent": {
                "warning": 70.0,
                "critical": 85.0,
            },
            "fragmentation_ratio": {
                "warning": 2.0,
                "critical": 3.0,
            },
            "consecutive_failures": {
                "warning": 3,
                "critical": 5,
            },
            "availability": {
                "critical": False,  # Any unavailability is critical
            },
        }
        
        # Alert configuration
        self._alert_enabled = True
        self._is_running = False
        
    async def check_thresholds(self, metrics: HealthMetrics) -> List[AlertEvent]:
        """Check metrics against alerting thresholds.
        
        Args:
            metrics: Current health metrics to evaluate
            
        Returns:
            List of triggered alert events
        """
        if not self._alert_enabled:
            return []
        
        alerts = []
        
        try:
            # Check availability (highest priority)
            if not metrics.is_available:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    "Redis unavailable",
                    "availability_check",
                    {
                        "is_available": metrics.is_available,
                        "consecutive_failures": metrics.consecutive_failures,
                        "last_error": metrics.last_error,
                    }
                )
                if alert:
                    alerts.append(alert)
            
            # Check ping latency
            if metrics.ping_latency_ms > self._thresholds["ping_latency_ms"]["critical"]:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"High Redis latency: {metrics.ping_latency_ms:.2f}ms",
                    "ping_latency",
                    {"ping_latency_ms": metrics.ping_latency_ms}
                )
                if alert:
                    alerts.append(alert)
            elif metrics.ping_latency_ms > self._thresholds["ping_latency_ms"]["warning"]:
                alert = await self._create_alert(
                    AlertLevel.WARNING,
                    f"Elevated Redis latency: {metrics.ping_latency_ms:.2f}ms",
                    "ping_latency",
                    {"ping_latency_ms": metrics.ping_latency_ms}
                )
                if alert:
                    alerts.append(alert)
            
            # Check connection utilization
            if metrics.connection_utilization > self._thresholds["connection_utilization_percent"]["critical"]:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical connection utilization: {metrics.connection_utilization:.1f}%",
                    "connection_utilization",
                    {"connection_utilization": metrics.connection_utilization}
                )
                if alert:
                    alerts.append(alert)
            elif metrics.connection_utilization > self._thresholds["connection_utilization_percent"]["warning"]:
                alert = await self._create_alert(
                    AlertLevel.WARNING,
                    f"High connection utilization: {metrics.connection_utilization:.1f}%",
                    "connection_utilization",
                    {"connection_utilization": metrics.connection_utilization}
                )
                if alert:
                    alerts.append(alert)
            
            # Check hit rate (inverted - low hit rate is bad)
            if metrics.hit_rate < self._thresholds["hit_rate_percent"]["critical"]:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical low hit rate: {metrics.hit_rate:.1f}%",
                    "hit_rate",
                    {"hit_rate": metrics.hit_rate}
                )
                if alert:
                    alerts.append(alert)
            elif metrics.hit_rate < self._thresholds["hit_rate_percent"]["warning"]:
                alert = await self._create_alert(
                    AlertLevel.WARNING,
                    f"Low hit rate: {metrics.hit_rate:.1f}%",
                    "hit_rate",
                    {"hit_rate": metrics.hit_rate}
                )
                if alert:
                    alerts.append(alert)
            
            # Check memory fragmentation
            if metrics.memory_fragmentation_ratio > self._thresholds["fragmentation_ratio"]["critical"]:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical memory fragmentation: {metrics.memory_fragmentation_ratio:.2f}",
                    "memory_fragmentation",
                    {"fragmentation_ratio": metrics.memory_fragmentation_ratio}
                )
                if alert:
                    alerts.append(alert)
            elif metrics.memory_fragmentation_ratio > self._thresholds["fragmentation_ratio"]["warning"]:
                alert = await self._create_alert(
                    AlertLevel.WARNING,
                    f"High memory fragmentation: {metrics.memory_fragmentation_ratio:.2f}",
                    "memory_fragmentation",
                    {"fragmentation_ratio": metrics.memory_fragmentation_ratio}
                )
                if alert:
                    alerts.append(alert)
            
            # Check consecutive failures
            if metrics.consecutive_failures >= self._thresholds["consecutive_failures"]["critical"]:
                alert = await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Multiple consecutive failures: {metrics.consecutive_failures}",
                    "consecutive_failures",
                    {"consecutive_failures": metrics.consecutive_failures}
                )
                if alert:
                    alerts.append(alert)
            elif metrics.consecutive_failures >= self._thresholds["consecutive_failures"]["warning"]:
                alert = await self._create_alert(
                    AlertLevel.WARNING,
                    f"Consecutive failures detected: {metrics.consecutive_failures}",
                    "consecutive_failures",
                    {"consecutive_failures": metrics.consecutive_failures}
                )
                if alert:
                    alerts.append(alert)
            
            # Update metrics for resolved alerts
            await self._check_for_resolved_alerts(metrics)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check alert thresholds: {e}")
            return []
    
    async def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert notification through configured channels.
        
        Args:
            alert: Alert event to send
            
        Returns:
            True if alert was sent successfully through at least one channel
        """
        success = False
        
        for channel in self.notification_channels:
            try:
                if await self._send_to_channel(channel, alert):
                    success = True
            except Exception as e:
                logger.error(f"Failed to send alert to channel {channel}: {e}")
        
        if success:
            # Update metrics
            ALERTS_TRIGGERED.labels(
                alert_level=alert.level.value,
                alert_type=self._extract_alert_type(alert)
            ).inc()
            
            # Add to active alerts
            self._active_alerts[alert.id] = alert
            self._update_active_alerts_metrics()
            
            logger.info(f"Alert sent: {alert.level.value} - {alert.message}")
        
        return success
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved successfully
        """
        try:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now(UTC)
                
                # Calculate duration and update metrics
                if alert.duration_seconds:
                    ALERT_DURATION.labels(alert_level=alert.level.value).observe(
                        alert.duration_seconds
                    )
                
                ALERTS_RESOLVED.labels(
                    alert_level=alert.level.value,
                    alert_type=self._extract_alert_type(alert)
                ).inc()
                
                # Move to history and remove from active
                self._alert_history.append(alert)
                del self._active_alerts[alert_id]
                
                # Keep history size manageable
                if len(self._alert_history) > self._max_history_size:
                    self._alert_history = self._alert_history[-self._max_history_size:]
                
                self._update_active_alerts_metrics()
                
                # Send resolution notification
                await self._send_resolution_notification(alert)
                
                logger.info(f"Alert resolved: {alert_id} - {alert.message}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active unresolved alerts.
        
        Returns:
            List of active alert events sorted by severity and time
        """
        alerts = list(self._active_alerts.values())
        
        # Sort by level (critical first) then by timestamp
        level_priority = {
            AlertLevel.EMERGENCY: 0,
            AlertLevel.CRITICAL: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3,
        }
        
        alerts.sort(key=lambda a: (level_priority[a.level], a.timestamp))
        return alerts
    
    def configure_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """Configure alerting thresholds.
        
        Args:
            thresholds: Threshold configuration dictionary
        """
        try:
            # Merge with existing thresholds
            for metric, config in thresholds.items():
                if metric in self._thresholds:
                    if isinstance(config, dict):
                        self._thresholds[metric].update(config)
                    else:
                        self._thresholds[metric] = config
                else:
                    self._thresholds[metric] = config
            
            logger.info(f"Alert thresholds updated: {list(thresholds.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to configure thresholds: {e}")
    
    async def start_alerting(self) -> None:
        """Start alerting service background tasks."""
        if self._is_running:
            logger.warning("Alerting service already running")
            return
        
        self._is_running = True
        self._alert_enabled = True
        
        # Start deduplication cleanup task
        self._dedup_cleanup_task = asyncio.create_task(self._deduplication_cleanup_loop())
        
        logger.info("Redis alerting service started")
    
    async def stop_alerting(self) -> None:
        """Stop alerting service background tasks."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._alert_enabled = False
        
        if self._dedup_cleanup_task:
            self._dedup_cleanup_task.cancel()
            try:
                await self._dedup_cleanup_task
            except asyncio.CancelledError:
                pass
            self._dedup_cleanup_task = None
        
        logger.info("Redis alerting service stopped")
    
    async def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[AlertEvent]:
        """Create alert with deduplication logic.
        
        Args:
            level: Alert severity level
            message: Alert message
            source: Source service/check that triggered alert
            metrics: Additional metrics context
            
        Returns:
            AlertEvent if new alert, None if deduplicated
        """
        # Create deduplication key
        dedup_key = f"{level.value}:{source}:{hash(message)}"
        
        # Check for recent duplicates
        if dedup_key in self._recent_alerts:
            return None
        
        # Create new alert
        alert = AlertEvent(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            source_service=f"redis_alerting.{source}",
            metrics=metrics
        )
        
        # Add to deduplication tracking
        self._recent_alerts.add(dedup_key)
        
        return alert
    
    async def _send_to_channel(self, channel: str, alert: AlertEvent) -> bool:
        """Send alert to specific notification channel.
        
        Args:
            channel: Notification channel name
            alert: Alert event to send
            
        Returns:
            True if sent successfully
        """
        try:
            if channel == "log":
                # Log-based notification
                log_level = logging.ERROR if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] else logging.WARNING
                logger.log(log_level, f"REDIS ALERT [{alert.level.value.upper()}]: {alert.message}")
                return True
            
            elif channel == "webhook":
                # Webhook notification (placeholder)
                # In production, implement actual webhook sending
                logger.info(f"Would send webhook alert: {alert.message}")
                return True
            
            elif channel == "email":
                # Email notification (placeholder)
                # In production, implement actual email sending
                logger.info(f"Would send email alert: {alert.message}")
                return True
            
            elif channel == "slack":
                # Slack notification (placeholder)
                # In production, implement actual Slack integration
                logger.info(f"Would send Slack alert: {alert.message}")
                return True
            
            else:
                logger.warning(f"Unknown notification channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send alert to {channel}: {e}")
            return False
    
    async def _send_resolution_notification(self, alert: AlertEvent) -> None:
        """Send alert resolution notification.
        
        Args:
            alert: Resolved alert event
        """
        try:
            resolution_message = f"RESOLVED: {alert.message}"
            if alert.duration_seconds:
                resolution_message += f" (Duration: {alert.duration_seconds:.1f}s)"
            
            for channel in self.notification_channels:
                if channel == "log":
                    logger.info(f"REDIS ALERT RESOLVED: {resolution_message}")
                # Add other channels as needed
                
        except Exception as e:
            logger.error(f"Failed to send resolution notification: {e}")
    
    async def _check_for_resolved_alerts(self, metrics: HealthMetrics) -> None:
        """Check if any active alerts should be automatically resolved.
        
        Args:
            metrics: Current health metrics
        """
        try:
            alerts_to_resolve = []
            
            for alert_id, alert in self._active_alerts.items():
                should_resolve = False
                
                # Check resolution conditions based on alert type
                if "availability" in alert.source_service and metrics.is_available:
                    should_resolve = True
                elif "ping_latency" in alert.source_service:
                    if alert.level == AlertLevel.CRITICAL:
                        should_resolve = metrics.ping_latency_ms <= self._thresholds["ping_latency_ms"]["critical"]
                    else:
                        should_resolve = metrics.ping_latency_ms <= self._thresholds["ping_latency_ms"]["warning"]
                elif "connection_utilization" in alert.source_service:
                    if alert.level == AlertLevel.CRITICAL:
                        should_resolve = metrics.connection_utilization <= self._thresholds["connection_utilization_percent"]["critical"]
                    else:
                        should_resolve = metrics.connection_utilization <= self._thresholds["connection_utilization_percent"]["warning"]
                elif "hit_rate" in alert.source_service:
                    if alert.level == AlertLevel.CRITICAL:
                        should_resolve = metrics.hit_rate >= self._thresholds["hit_rate_percent"]["critical"]
                    else:
                        should_resolve = metrics.hit_rate >= self._thresholds["hit_rate_percent"]["warning"]
                
                if should_resolve:
                    alerts_to_resolve.append(alert_id)
            
            # Resolve alerts
            for alert_id in alerts_to_resolve:
                await self.resolve_alert(alert_id)
                
        except Exception as e:
            logger.error(f"Failed to check for resolved alerts: {e}")
    
    async def _deduplication_cleanup_loop(self) -> None:
        """Background task to clean up deduplication tracking."""
        while self._is_running:
            try:
                # Clear deduplication tracking every window
                await asyncio.sleep(self.alert_deduplication_window_minutes * 60)
                self._recent_alerts.clear()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in deduplication cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    def _extract_alert_type(self, alert: AlertEvent) -> str:
        """Extract alert type from source service for metrics.
        
        Args:
            alert: Alert event
            
        Returns:
            Alert type string
        """
        if "." in alert.source_service:
            return alert.source_service.split(".")[-1]
        return alert.source_service
    
    def _update_active_alerts_metrics(self) -> None:
        """Update Prometheus metrics for active alerts."""
        try:
            # Count alerts by level
            level_counts = {level: 0 for level in AlertLevel}
            
            for alert in self._active_alerts.values():
                level_counts[alert.level] += 1
            
            # Update gauges
            for level, count in level_counts.items():
                ACTIVE_ALERTS_COUNT.labels(alert_level=level.value).set(count)
                
        except Exception as e:
            logger.debug(f"Failed to update alert metrics: {e}")
    
    def get_alerting_status(self) -> Dict[str, Any]:
        """Get alerting service status and statistics.
        
        Returns:
            Alerting status dictionary
        """
        return {
            "enabled": self._alert_enabled,
            "is_running": self._is_running,
            "active_alerts_count": len(self._active_alerts),
            "total_alerts_in_history": len(self._alert_history),
            "notification_channels": self.notification_channels,
            "thresholds": self._thresholds,
            "deduplication_window_minutes": self.alert_deduplication_window_minutes,
            "escalation_timeout_minutes": self.escalation_timeout_minutes,
            "recent_alerts_count": len(self._recent_alerts),
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alert activity.
        
        Returns:
            Alert activity summary
        """
        now = datetime.now(UTC)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Count recent alerts
        recent_hour_alerts = [
            alert for alert in self._alert_history
            if alert.timestamp >= last_hour
        ]
        
        recent_day_alerts = [
            alert for alert in self._alert_history
            if alert.timestamp >= last_day
        ]
        
        # Count by severity
        severity_counts = {level.value: 0 for level in AlertLevel}
        for alert in recent_day_alerts:
            severity_counts[alert.level.value] += 1
        
        return {
            "active_alerts": len(self._active_alerts),
            "alerts_last_hour": len(recent_hour_alerts),
            "alerts_last_day": len(recent_day_alerts),
            "severity_breakdown_24h": severity_counts,
            "top_alert_sources": self._get_top_alert_sources(recent_day_alerts),
        }
    
    def _get_top_alert_sources(self, alerts: List[AlertEvent]) -> Dict[str, int]:
        """Get top alert sources from alert list.
        
        Args:
            alerts: List of alerts to analyze
            
        Returns:
            Dictionary of source counts
        """
        source_counts = {}
        for alert in alerts:
            source = self._extract_alert_type(alert)
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Return top 5 sources
        return dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5])