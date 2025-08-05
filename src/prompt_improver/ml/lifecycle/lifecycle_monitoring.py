"""ML Lifecycle Monitoring and Alerting System (2025)

Comprehensive monitoring and alerting for complete ML model lifecycle:
- Real-time performance monitoring across all lifecycle stages
- Automated alerting for performance degradation and failures
- Advanced metrics collection and analytics
- Integration with platform health monitoring
- SLA tracking and compliance reporting
- Predictive failure detection and prevention
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
from pydantic import BaseModel, Field as PydanticField

# REMOVED prometheus_client for OpenTelemetry consolidation
from prompt_improver.performance.monitoring.health.enhanced_base import EnhancedHealthMonitor
from prompt_improver.performance.monitoring.health.service import HealthService
from prompt_improver.performance.monitoring.performance_monitor import PerformanceMonitor
from prompt_improver.utils.datetime_utils import aware_utc_now
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DEPLOYMENT_FAILURE = "deployment_failure"
    MODEL_DRIFT = "model_drift"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXPERIMENT_FAILURE = "experiment_failure"
    SERVING_ERROR = "serving_error"
    SLA_VIOLATION = "sla_violation"
    SYSTEM_HEALTH = "system_health"

class MonitoringScope(Enum):
    """Monitoring scope levels."""
    PLATFORM = "platform"
    COMPONENT = "component"
    MODEL = "model"
    EXPERIMENT = "experiment"
    DEPLOYMENT = "deployment"

@dataclass
class AlertRule:
    """Configuration for monitoring alert rules."""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    scope: MonitoringScope
    
    # Conditions
    metric_name: str
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    evaluation_window_minutes: int = 5
    trigger_threshold: int = 1  # Number of violations to trigger alert
    
    # Actions
    notification_channels: List[str] = field(default_factory=list)
    auto_remediation_enabled: bool = False
    remediation_actions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=aware_utc_now)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    scope: MonitoringScope
    
    # Alert details
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    
    # Timing
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=aware_utc_now)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Status
    status: str = "active"  # active, resolved, acknowledged, suppressed
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

@dataclass
class SLATarget:
    """Service Level Agreement target definition."""
    sla_id: str
    name: str
    description: str
    
    # Metrics
    metric_name: str
    target_value: float
    target_operator: str  # "gt", "lt", "gte", "lte"
    measurement_window_hours: int = 24
    
    # Compliance
    compliance_threshold: float = 99.0  # Percentage
    reporting_interval_hours: int = 24
    
    # Context
    scope: MonitoringScope
    component: Optional[str] = None
    created_at: datetime = field(default_factory=aware_utc_now)
    enabled: bool = True

@dataclass
class MetricDataPoint:
    """Individual metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LifecycleMetrics:
    """Comprehensive lifecycle metrics."""
    
    # Platform-level metrics
    platform_uptime_seconds: float = 0.0
    total_experiments: int = 0
    active_experiments: int = 0
    total_models: int = 0
    active_models: int = 0
    total_deployments: int = 0
    active_deployments: int = 0
    
    # Performance metrics
    experiment_success_rate: float = 100.0
    deployment_success_rate: float = 100.0
    avg_experiment_duration_minutes: float = 0.0
    avg_deployment_time_minutes: float = 0.0
    
    # Throughput metrics
    experiments_per_hour: float = 0.0
    deployments_per_hour: float = 0.0
    inference_requests_per_second: float = 0.0
    
    # Quality metrics
    model_accuracy_avg: float = 0.0
    model_drift_score: float = 0.0
    data_quality_score: float = 100.0
    
    # Resource metrics
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    disk_utilization_percent: float = 0.0
    
    # Error metrics
    error_rate_percent: float = 0.0
    alert_count_24h: int = 0
    critical_alert_count_24h: int = 0
    
    # SLA metrics
    sla_compliance_percent: float = 100.0
    availability_percent: float = 100.0
    mean_time_to_recovery_minutes: float = 0.0
    
    # Timestamp
    collected_at: datetime = field(default_factory=aware_utc_now)

class LifecycleMonitor:
    """Comprehensive ML Lifecycle Monitoring and Alerting System.
    
    Features:
    - Real-time metrics collection across all lifecycle components
    - Configurable alerting rules with auto-remediation
    - SLA tracking and compliance reporting
    - Performance trend analysis and prediction
    - Integration with existing health monitoring systems
    - Prometheus metrics export for external monitoring tools
    """
    
    def __init__(self,
                 storage_path: Path = Path("./monitoring"),
                 metrics_retention_days: int = 30,
                 alert_retention_days: int = 90):
        """Initialize lifecycle monitoring system.
        
        Args:
            storage_path: Path for monitoring data storage
            metrics_retention_days: How long to retain metric data
            alert_retention_days: How long to retain alert history
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_retention_days = metrics_retention_days
        self.metrics_buffer: deque = deque(maxlen=10000)  # In-memory buffer
        self.metrics_history: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        
        # Alert management
        self.alert_retention_days = alert_retention_days
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # SLA tracking
        self.sla_targets: Dict[str, SLATarget] = {}
        self.sla_compliance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Component integrations
        self.health_service: Optional[HealthService] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Background task IDs for EnhancedBackgroundTaskManager
        self._background_task_ids: List[str] = []
        self._is_running = False
        
        # Performance tracking
        self.start_time = time.time()
        self.last_metrics_collection = time.time()
        
        logger.info("ML Lifecycle Monitor initialized")
        logger.info(f"Metrics retention: {metrics_retention_days} days")
        logger.info(f"Alert retention: {alert_retention_days} days")
    
    async def start_monitoring(self):
        """Start the lifecycle monitoring system."""
        
        if self._is_running:
            return
        
        self._is_running = True
        
        # Initialize component integrations
        await self._initialize_component_integrations()
        
        # Load existing configuration
        await self._load_configuration()
        
        # Start background monitoring tasks using EnhancedBackgroundTaskManager
        task_manager = get_background_task_manager()
        
        # Start metrics collector with NORMAL priority for regular data collection
        metrics_task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_metrics_collector_{str(uuid.uuid4())[:8]}",
            coroutine=self._metrics_collector(),
            priority=TaskPriority.NORMAL,
            tags={
                "service": "ml",
                "type": "monitoring",
                "component": "metrics_collector",
                "module": "lifecycle_monitoring"
            }
        )
        
        # Start alert evaluator with HIGH priority for real-time alerting
        alert_task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_alert_evaluator_{str(uuid.uuid4())[:8]}",
            coroutine=self._alert_evaluator(),
            priority=TaskPriority.HIGH,
            tags={
                "service": "ml",
                "type": "monitoring",
                "component": "alert_evaluator",
                "module": "lifecycle_monitoring"
            }
        )
        
        # Start SLA tracker with HIGH priority for compliance monitoring
        sla_task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_sla_tracker_{str(uuid.uuid4())[:8]}",
            coroutine=self._sla_tracker(),
            priority=TaskPriority.HIGH,
            tags={
                "service": "ml",
                "type": "monitoring",
                "component": "sla_tracker",
                "module": "lifecycle_monitoring"
            }
        )
        
        # Start data retention manager with LOW priority for cleanup operations
        retention_task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_data_retention_{str(uuid.uuid4())[:8]}",
            coroutine=self._data_retention_manager(),
            priority=TaskPriority.LOW,
            tags={
                "service": "ml",
                "type": "maintenance",
                "component": "data_retention",
                "module": "lifecycle_monitoring"
            }
        )
        
        # Start health checker with HIGH priority for system health monitoring
        health_task_id = await task_manager.submit_enhanced_task(
            task_id=f"ml_health_checker_{str(uuid.uuid4())[:8]}",
            coroutine=self._health_checker(),
            priority=TaskPriority.HIGH,
            tags={
                "service": "ml",
                "type": "monitoring",
                "component": "health_checker",
                "module": "lifecycle_monitoring"
            }
        )
        
        # Store task IDs for cleanup
        self._background_task_ids = [metrics_task_id, alert_task_id, sla_task_id, retention_task_id, health_task_id]
        
        logger.info("✅ ML Lifecycle Monitoring started")
        
        # Setup default alert rules if none exist
        if not self.alert_rules:
            await self._setup_default_alert_rules()
        
        # Setup default SLA targets
        if not self.sla_targets:
            await self._setup_default_sla_targets()
    
    async def stop_monitoring(self):
        """Stop the lifecycle monitoring system."""
        
        self._is_running = False
        
        # Cancel background tasks using task manager
        task_manager = get_background_task_manager()
        for task_id in self._background_task_ids:
            await task_manager.cancel_task(task_id)
        self._background_task_ids.clear()
        
        # Save configuration
        await self._save_configuration()
        
        logger.info("🛑 ML Lifecycle Monitoring stopped")
    
    async def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule to monitoring system."""
        
        self.alert_rules[rule.rule_id] = rule
        
        logger.info(f"Added alert rule: {rule.name} ({rule.severity.value})")
        await self._save_configuration()
    
    async def add_sla_target(self, target: SLATarget):
        """Add new SLA target for monitoring."""
        
        self.sla_targets[target.sla_id] = target
        
        logger.info(f"Added SLA target: {target.name} ({target.target_value}{target.target_operator})")
        await self._save_configuration()
    
    async def record_metric(self, 
                          metric_name: str, 
                          value: float,
                          labels: Dict[str, str] = None,
                          context: Dict[str, Any] = None):
        """Record a metric data point."""
        
        data_point = MetricDataPoint(
            timestamp=aware_utc_now(),
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            context=context or {}
        )
        
        # Add to buffer and history
        self.metrics_buffer.append(data_point)
        self.metrics_history[metric_name].append(data_point)
        
        # Update Prometheus metrics
        await self._update_prometheus_metrics(data_point)
        
        # Trigger immediate alert evaluation for critical metrics
        if metric_name in ["deployment_failure_rate", "experiment_failure_rate", "error_rate"]:
            await self._evaluate_alerts_for_metric(metric_name, value)
    
    async def get_lifecycle_metrics(self) -> LifecycleMetrics:
        """Get comprehensive lifecycle metrics."""
        
        metrics = LifecycleMetrics()
        
        # Calculate platform uptime
        metrics.platform_uptime_seconds = time.time() - self.start_time
        
        # Aggregate metrics from history
        current_time = aware_utc_now()
        hour_ago = current_time - timedelta(hours=1)
        day_ago = current_time - timedelta(days=1)
        
        # Get recent metrics
        recent_metrics = [
            dp for dp in self.metrics_buffer 
            if dp.timestamp >= hour_ago
        ]
        
        if recent_metrics:
            # Calculate averages and totals
            for metric_name in ["cpu_utilization", "memory_utilization", "gpu_utilization"]:
                values = [dp.value for dp in recent_metrics if dp.metric_name == metric_name]
                if values:
                    if metric_name == "cpu_utilization":
                        metrics.cpu_utilization_percent = np.mean(values)
                    elif metric_name == "memory_utilization":
                        metrics.memory_utilization_percent = np.mean(values)
                    elif metric_name == "gpu_utilization":
                        metrics.gpu_utilization_percent = np.mean(values)
            
            # Calculate rates
            experiment_count = len([dp for dp in recent_metrics if dp.metric_name == "experiment_completed"])
            deployment_count = len([dp for dp in recent_metrics if dp.metric_name == "deployment_completed"])
            
            metrics.experiments_per_hour = experiment_count
            metrics.deployments_per_hour = deployment_count
        
        # Calculate SLA compliance
        sla_compliances = []
        for sla_id, compliance_history in self.sla_compliance_history.items():
            if compliance_history:
                sla_compliances.append(compliance_history[-1])
        
        if sla_compliances:
            metrics.sla_compliance_percent = np.mean(sla_compliances)
        
        # Count alerts
        day_alerts = [
            alert for alert in self.alert_history 
            if alert.triggered_at >= day_ago
        ]
        metrics.alert_count_24h = len(day_alerts)
        metrics.critical_alert_count_24h = len([
            alert for alert in day_alerts 
            if alert.severity == AlertSeverity.CRITICAL
        ])
        
        return metrics
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts and alert history."""
        
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        # Recent alert trends
        current_time = aware_utc_now()
        day_ago = current_time - timedelta(days=1)
        week_ago = current_time - timedelta(days=7)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.triggered_at >= day_ago
        ]
        
        weekly_alerts = [
            alert for alert in self.alert_history 
            if alert.triggered_at >= week_ago
        ]
        
        return {
            "active_alerts": {
                "total": len(self.active_alerts),
                "by_severity": dict(active_by_severity),
                "by_type": dict(defaultdict(int, {
                    alert.alert_type.value: len([
                        a for a in self.active_alerts.values() 
                        if a.alert_type == alert.alert_type
                    ])
                    for alert in self.active_alerts.values()
                }))
            },
            "recent_trends": {
                "alerts_24h": len(recent_alerts),
                "alerts_7d": len(weekly_alerts),
                "critical_alerts_24h": len([
                    a for a in recent_alerts 
                    if a.severity == AlertSeverity.CRITICAL
                ]),
                "most_common_type_24h": max([
                    (alert.alert_type.value, len([
                        a for a in recent_alerts 
                        if a.alert_type == alert.alert_type
                    ]))
                    for alert in recent_alerts
                ], key=lambda x: x[1], default=("none", 0))[0]
            },
            "alert_rules": {
                "total_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled])
            }
        }
    
    async def get_sla_report(self) -> Dict[str, Any]:
        """Generate comprehensive SLA compliance report."""
        
        report = {
            "sla_targets": len(self.sla_targets),
            "overall_compliance": 0.0,
            "targets": {}
        }
        
        total_compliance = 0.0
        compliant_targets = 0
        
        for sla_id, target in self.sla_targets.items():
            compliance_history = self.sla_compliance_history.get(sla_id, [])
            
            if compliance_history:
                current_compliance = compliance_history[-1]
                avg_compliance = np.mean(compliance_history[-24:])  # Last 24 measurements
                
                target_report = {
                    "name": target.name,
                    "target_value": target.target_value,
                    "current_compliance": current_compliance,
                    "average_compliance_24h": avg_compliance,
                    "is_compliant": current_compliance >= target.compliance_threshold,
                    "violations_24h": len([
                        c for c in compliance_history[-24:] 
                        if c < target.compliance_threshold
                    ])
                }
                
                report["targets"][sla_id] = target_report
                
                if current_compliance >= target.compliance_threshold:
                    compliant_targets += 1
                total_compliance += current_compliance
        
        if len(self.sla_targets) > 0:
            report["overall_compliance"] = total_compliance / len(self.sla_targets)
            report["compliant_targets"] = compliant_targets
            report["compliance_rate"] = compliant_targets / len(self.sla_targets) * 100
        
        return report
    
    async def _metrics_collector(self):
        """Background metrics collection task."""
        
        while self._is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect component metrics
                await self._collect_component_metrics()
                
                # Update collection timestamp
                self.last_metrics_collection = time.time()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_evaluator(self):
        """Background alert evaluation task."""
        
        while self._is_running:
            try:
                for rule in self.alert_rules.values():
                    if rule.enabled:
                        await self._evaluate_alert_rule(rule)
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _sla_tracker(self):
        """Background SLA compliance tracking task."""
        
        while self._is_running:
            try:
                for sla_id, target in self.sla_targets.items():
                    if target.enabled:
                        compliance = await self._calculate_sla_compliance(target)
                        self.sla_compliance_history[sla_id].append(compliance)
                        
                        # Keep only recent history
                        max_history = target.reporting_interval_hours * 7  # One week
                        if len(self.sla_compliance_history[sla_id]) > max_history:
                            self.sla_compliance_history[sla_id] = \
                                self.sla_compliance_history[sla_id][-max_history:]
                
                await asyncio.sleep(3600)  # Check SLA every hour
                
            except Exception as e:
                logger.error(f"SLA tracking error: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        
        import psutil
        
        # CPU and Memory
        await self.record_metric("cpu_utilization", psutil.cpu_percent())
        await self.record_metric("memory_utilization", psutil.virtual_memory().percent)
        
        # Disk
        disk_usage = psutil.disk_usage('/')
        await self.record_metric("disk_utilization", 
                                (disk_usage.used / disk_usage.total) * 100)
        
        # Platform uptime
        uptime_hours = (time.time() - self.start_time) / 3600
        await self.record_metric("platform_uptime_hours", uptime_hours)
    
    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        
        # Get recent metrics for the rule
        current_time = aware_utc_now()
        window_start = current_time - timedelta(minutes=rule.evaluation_window_minutes)
        
        recent_metrics = [
            dp for dp in self.metrics_history.get(rule.metric_name, [])
            if dp.timestamp >= window_start
        ]
        
        if not recent_metrics:
            return
        
        # Calculate metric value based on aggregation
        metric_values = [dp.value for dp in recent_metrics]
        current_value = np.mean(metric_values)  # Could be configurable (avg, max, min, etc.)
        
        # Evaluate condition
        violated = self._evaluate_condition(
            current_value, 
            rule.threshold_value, 
            rule.comparison_operator
        )
        
        alert_key = f"{rule.rule_id}_{rule.scope.value}"
        
        if violated:
            if alert_key not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    scope=rule.scope,
                    title=rule.name,
                    message=f"{rule.description} - Current: {current_value:.2f}, Threshold: {rule.threshold_value}",
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold_value=rule.threshold_value,
                    triggered_at=current_time,
                    context={"rule": rule.__dict__},
                    tags=rule.tags.copy()
                )
                
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"🚨 Alert triggered: {rule.name} ({rule.severity.value})")
                
                # Execute remediation if enabled
                if rule.auto_remediation_enabled:
                    await self._execute_remediation(alert, rule)
        else:
            # Resolve alert if it exists
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.status = "resolved"
                alert.resolved_at = current_time
                
                del self.active_alerts[alert_key]
                
                logger.info(f"✅ Alert resolved: {rule.name}")
    
    def _evaluate_condition(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition."""
        
        if operator == "gt":
            return current_value > threshold
        elif operator == "lt":
            return current_value < threshold
        elif operator == "gte":
            return current_value >= threshold
        elif operator == "lte":
            return current_value <= threshold
        elif operator == "eq":
            return abs(current_value - threshold) < 1e-6
        else:
            return False
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules for ML lifecycle monitoring."""
        
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU utilization is above 80%",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.WARNING,
                scope=MonitoringScope.PLATFORM,
                metric_name="cpu_utilization",
                threshold_value=80.0,
                comparison_operator="gt",
                evaluation_window_minutes=5
            ),
            AlertRule(
                rule_id="deployment_failure_rate",
                name="High Deployment Failure Rate",
                description="Deployment failure rate is above 10%",
                alert_type=AlertType.DEPLOYMENT_FAILURE,
                severity=AlertSeverity.ERROR,
                scope=MonitoringScope.PLATFORM,
                metric_name="deployment_failure_rate",
                threshold_value=10.0,
                comparison_operator="gt",
                evaluation_window_minutes=10
            ),
            AlertRule(
                rule_id="experiment_throughput_low",
                name="Low Experiment Throughput",
                description="Experiment throughput is below target",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.WARNING,
                scope=MonitoringScope.PLATFORM,
                metric_name="experiments_per_hour",
                threshold_value=10.0,
                comparison_operator="lt",
                evaluation_window_minutes=15
            ),
            AlertRule(
                rule_id="memory_exhaustion",
                name="Memory Exhaustion",
                description="Memory utilization is above 90%",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.CRITICAL,
                scope=MonitoringScope.PLATFORM,
                metric_name="memory_utilization",
                threshold_value=90.0,
                comparison_operator="gt",
                evaluation_window_minutes=3
            )
        ]
        
        for rule in default_rules:
            await self.add_alert_rule(rule)
        
        logger.info(f"✅ Setup {len(default_rules)} default alert rules")
    
    async def _setup_default_sla_targets(self):
        """Setup default SLA targets for ML lifecycle."""
        
        default_slas = [
            SLATarget(
                sla_id="platform_availability",
                name="Platform Availability",
                description="Platform should be available 99.9% of the time",
                metric_name="platform_uptime_hours",
                target_value=99.9,
                target_operator="gte",
                compliance_threshold=99.9,
                scope=MonitoringScope.PLATFORM
            ),
            SLATarget(
                sla_id="deployment_success_rate",
                name="Deployment Success Rate",
                description="95% of deployments should succeed",
                metric_name="deployment_success_rate",
                target_value=95.0,
                target_operator="gte",
                compliance_threshold=95.0,
                scope=MonitoringScope.PLATFORM
            ),
            SLATarget(
                sla_id="experiment_completion_rate",
                name="Experiment Completion Rate",
                description="90% of experiments should complete successfully",
                metric_name="experiment_success_rate",
                target_value=90.0,
                target_operator="gte",
                compliance_threshold=90.0,
                scope=MonitoringScope.PLATFORM
            )
        ]
        
        for sla in default_slas:
            await self.add_sla_target(sla)
        
        logger.info(f"✅ Setup {len(default_slas)} default SLA targets")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for external monitoring."""
        
        self.prometheus_metrics = {
            "platform_uptime": Gauge(
                "ml_platform_uptime_seconds",
                "Platform uptime in seconds",
                registry=self.prometheus_registry
            ),
            "active_alerts": Gauge(
                "ml_platform_active_alerts_total",
                "Number of active alerts",
                ["severity"],
                registry=self.prometheus_registry
            ),
            "sla_compliance": Gauge(
                "ml_platform_sla_compliance_percent",
                "SLA compliance percentage",
                ["sla_id"],
                registry=self.prometheus_registry
            ),
            "metrics_collected": Counter(
                "ml_platform_metrics_collected_total",
                "Total metrics collected",
                ["metric_name"],
                registry=self.prometheus_registry
            )
        }
    
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        
        # Update Prometheus metrics
        self.prometheus_metrics["platform_uptime"].set(time.time() - self.start_time)
        
        # Update alert counts by severity
        for severity in AlertSeverity:
            count = len([
                alert for alert in self.active_alerts.values()
                if alert.severity == severity
            ])
            self.prometheus_metrics["active_alerts"].labels(severity=severity.value).set(count)
        
        # Update SLA compliance
        for sla_id, compliance_history in self.sla_compliance_history.items():
            if compliance_history:
                self.prometheus_metrics["sla_compliance"].labels(sla_id=sla_id).set(compliance_history[-1])
        
        return generate_latest(self.prometheus_registry)

# Factory Functions

async def create_lifecycle_monitor(
    storage_path: Path = Path("./monitoring"),
    metrics_retention_days: int = 30,
    alert_retention_days: int = 90
) -> LifecycleMonitor:
    """Create and start lifecycle monitoring system.
    
    Args:
        storage_path: Monitoring storage path
        metrics_retention_days: Metric retention period
        alert_retention_days: Alert retention period
        
    Returns:
        Started LifecycleMonitor instance
    """
    
    monitor = LifecycleMonitor(
        storage_path=storage_path,
        metrics_retention_days=metrics_retention_days,
        alert_retention_days=alert_retention_days
    )
    
    await monitor.start_monitoring()
    return monitor

# Utility functions for common monitoring patterns

def create_performance_alert_rule(
    metric_name: str,
    threshold: float,
    severity: AlertSeverity = AlertSeverity.WARNING
) -> AlertRule:
    """Create a performance-based alert rule."""
    
    return AlertRule(
        rule_id=f"perf_{metric_name}_{int(time.time())}",
        name=f"Performance Alert: {metric_name}",
        description=f"{metric_name} performance threshold exceeded",
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        severity=severity,
        scope=MonitoringScope.PLATFORM,
        metric_name=metric_name,
        threshold_value=threshold,
        comparison_operator="gt"
    )

def create_sla_target(
    name: str,
    metric_name: str,
    target_value: float,
    compliance_threshold: float = 99.0
) -> SLATarget:
    """Create an SLA target."""
    
    return SLATarget(
        sla_id=f"sla_{name.lower().replace(' ', '_')}_{int(time.time())}",
        name=name,
        description=f"SLA target for {name}",
        metric_name=metric_name,
        target_value=target_value,
        target_operator="gte",
        compliance_threshold=compliance_threshold,
        scope=MonitoringScope.PLATFORM
    )