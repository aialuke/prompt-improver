"""
Performance Telemetry System for ML Pipeline Orchestrator.

Provides real-time performance monitoring, metrics collection, and adaptive tuning
for the event bus and connection pool optimizations.

Key Features:
- Real-time performance metrics collection
- Adaptive threshold tuning based on performance trends
- Performance anomaly detection
- Auto-tuning recommendations
- Comprehensive performance reporting
"""
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
import statistics
import time
from typing import Any, Dict, List, Optional
from collections.abc import Callable
import uuid
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager

class PerformanceLevel(Enum):
    """Performance level classifications."""
    EXCELLENT = 'excellent'
    GOOD = 'good'
    ACCEPTABLE = 'acceptable'
    DEGRADED = 'degraded'
    CRITICAL = 'critical'

class MetricType(Enum):
    """Types of performance metrics."""
    THROUGHPUT = 'throughput'
    LATENCY = 'latency'
    UTILIZATION = 'utilization'
    ERROR_RATE = 'error_rate'
    QUEUE_DEPTH = 'queue_depth'
    CONNECTION_POOL = 'connection_pool'

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    component: str
    tags: dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceThresholds:
    """Dynamic performance thresholds."""
    excellent: float
    good: float
    acceptable: float
    degraded: float
    critical: float

    def classify(self, value: float, greater_is_better: bool=True) -> PerformanceLevel:
        """Classify a value based on thresholds."""
        if greater_is_better:
            if value >= self.excellent:
                return PerformanceLevel.EXCELLENT
            elif value >= self.good:
                return PerformanceLevel.GOOD
            elif value >= self.acceptable:
                return PerformanceLevel.ACCEPTABLE
            elif value >= self.degraded:
                return PerformanceLevel.DEGRADED
            else:
                return PerformanceLevel.CRITICAL
        elif value <= self.excellent:
            return PerformanceLevel.EXCELLENT
        elif value <= self.good:
            return PerformanceLevel.GOOD
        elif value <= self.acceptable:
            return PerformanceLevel.ACCEPTABLE
        elif value <= self.degraded:
            return PerformanceLevel.DEGRADED
        else:
            return PerformanceLevel.CRITICAL

@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly."""
    component: str
    metric_name: str
    current_value: float
    expected_range: tuple
    severity: PerformanceLevel
    timestamp: datetime
    description: str
    recommendation: str

class PerformanceTelemetrySystem:
    """
    Comprehensive performance telemetry and monitoring system.
    
    Collects, analyzes, and reports on ML Pipeline Orchestrator performance
    with adaptive tuning and anomaly detection capabilities.
    """

    def __init__(self):
        """Initialize performance telemetry system."""
        self.logger = logging.getLogger(__name__)
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics: dict[str, PerformanceMetric] = {}
        self.thresholds: dict[str, PerformanceThresholds] = {}
        self._initialize_default_thresholds()
        self.anomalies: deque = deque(maxlen=1000)
        self.anomaly_callbacks: list[Callable[[PerformanceAnomaly], None]] = []
        self.trend_window = 300
        self.baseline_window = 3600
        self.is_monitoring = False
        self.monitoring_task: asyncio.Task | None = None
        self.collection_interval = 10
        self.event_bus = None
        self._unified_manager = None
        self.logger.info('Performance telemetry system initialized')

    def _initialize_default_thresholds(self) -> None:
        """Initialize default performance thresholds."""
        self.thresholds.update({'event_throughput': PerformanceThresholds(excellent=8000.0, good=5000.0, acceptable=2000.0, degraded=1000.0, critical=500.0), 'event_latency': PerformanceThresholds(excellent=1.0, good=5.0, acceptable=10.0, degraded=50.0, critical=100.0), 'queue_utilization': PerformanceThresholds(excellent=0.3, good=0.5, acceptable=0.7, degraded=0.85, critical=0.95), 'connection_utilization': PerformanceThresholds(excellent=0.4, good=0.6, acceptable=0.75, degraded=0.85, critical=0.95), 'connection_latency': PerformanceThresholds(excellent=5.0, good=10.0, acceptable=25.0, degraded=50.0, critical=100.0), 'query_latency': PerformanceThresholds(excellent=10.0, good=25.0, acceptable=50.0, degraded=100.0, critical=500.0)})

    def register_event_bus(self, event_bus) -> None:
        """Register event bus for monitoring."""
        self.event_bus = event_bus
        self.logger.info('Event bus registered for telemetry')

    def register_unified_manager(self, unified_manager) -> None:
        """Register DatabaseServices for consolidated pool monitoring.
        
        This replaces independent connection pool registration with unified
        pool monitoring, eliminating duplication across ML orchestration.
        """
        self._unified_manager = unified_manager
        self.logger.info('DatabaseServices registered for consolidated telemetry')

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        self.logger.info('Starting performance monitoring...')
        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.monitoring_task_id = await task_manager.submit_enhanced_task(task_id=f'ml_telemetry_monitor_{str(uuid.uuid4())[:8]}', coroutine=self._monitoring_loop(), priority=TaskPriority.HIGH, tags={'service': 'ml', 'type': 'monitoring', 'component': 'performance_telemetry', 'module': 'telemetry'})

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        self.logger.info('Stopping performance monitoring...')
        self.is_monitoring = False
        if hasattr(self, 'monitoring_task_id') and self.monitoring_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self.monitoring_task_id)

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.collection_interval)
                await self._collect_metrics()
                await self._analyze_performance()
                await self._detect_anomalies()
                await self._update_adaptive_thresholds()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error('Monitoring loop error: %s', e)

    async def _collect_metrics(self) -> None:
        """Collect performance metrics from all components."""
        timestamp = datetime.now(timezone.utc)
        if self.event_bus:
            await self._collect_event_bus_metrics(timestamp)
        if self._unified_manager:
            await self._collect_unified_pool_metrics(timestamp)
        await self._collect_system_metrics(timestamp)

    async def _collect_event_bus_metrics(self, timestamp: datetime) -> None:
        """Collect event bus performance metrics."""
        try:
            bus_metrics = self.event_bus.get_performance_metrics()
            self._record_metric('event_throughput', bus_metrics.get('throughput_per_second', 0), 'events/sec', timestamp, MetricType.THROUGHPUT, 'event_bus')
            self._record_metric('event_latency', bus_metrics.get('avg_processing_time_ms', 0), 'ms', timestamp, MetricType.LATENCY, 'event_bus')
            queue_utilization = bus_metrics.get('queue_utilization', 0)
            self._record_metric('queue_utilization', queue_utilization, 'ratio', timestamp, MetricType.UTILIZATION, 'event_bus')
            total_events = bus_metrics.get('events_processed', 0) + bus_metrics.get('events_failed', 0)
            error_rate = bus_metrics.get('events_failed', 0) / max(total_events, 1)
            self._record_metric('event_error_rate', error_rate, 'ratio', timestamp, MetricType.ERROR_RATE, 'event_bus')
        except Exception as e:
            self.logger.error('Error collecting event bus metrics: %s', e)

    async def _collect_unified_pool_metrics(self, timestamp: datetime) -> None:
        """Collect unified connection pool performance metrics.
        
        Uses DatabaseServices's consolidated pool monitoring,
        eliminating the need for independent pool registration patterns.
        """
        try:
            pool_metrics = await self._unified_manager.get_ml_telemetry_metrics()
            self._record_metric('connection_utilization', pool_metrics.get('pool_utilization', 0), 'ratio', timestamp, MetricType.UTILIZATION, 'unified_pool')
            self._record_metric('connection_latency', pool_metrics.get('avg_connection_time_ms', 0), 'ms', timestamp, MetricType.LATENCY, 'unified_pool')
            self._record_metric('query_latency', pool_metrics.get('avg_query_time_ms', 0), 'ms', timestamp, MetricType.LATENCY, 'unified_pool')
            self._record_metric('connection_pool_size', pool_metrics.get('pool_size', 0), 'connections', timestamp, MetricType.CONNECTION_POOL, 'unified_pool')
            self._record_metric('active_connections', pool_metrics.get('active_connections', 0), 'connections', timestamp, MetricType.CONNECTION_POOL, 'unified_pool')
            self._record_metric('available_connections', pool_metrics.get('available_connections', 0), 'connections', timestamp, MetricType.CONNECTION_POOL, 'unified_pool')
        except Exception as e:
            self.logger.error('Error collecting unified pool metrics: %s', e)

    async def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system-level performance metrics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            self._record_metric('cpu_utilization', cpu_percent / 100.0, 'ratio', timestamp, MetricType.UTILIZATION, 'system')
            memory = psutil.virtual_memory()
            self._record_metric('memory_utilization', memory.percent / 100.0, 'ratio', timestamp, MetricType.UTILIZATION, 'system')
        except ImportError:
            pass
        except Exception as e:
            self.logger.error('Error collecting system metrics: %s', e)

    def _record_metric(self, name: str, value: float, unit: str, timestamp: datetime, metric_type: MetricType, component: str) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(name=name, value=value, unit=unit, timestamp=timestamp, metric_type=metric_type, component=component)
        self.metrics[name].append(metric)
        self.current_metrics[name] = metric

    async def _analyze_performance(self) -> None:
        """Analyze current performance levels."""
        for metric_name, metric in self.current_metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                greater_is_better = metric.metric_type in [MetricType.THROUGHPUT]
                level = threshold.classify(metric.value, greater_is_better)
                if level in [PerformanceLevel.DEGRADED, PerformanceLevel.CRITICAL]:
                    self.logger.warning('Performance %s: %s=%s %s', level.value, metric_name, format(metric.value, '.2f'), metric.unit)

    async def _detect_anomalies(self) -> None:
        """Detect performance anomalies."""
        for metric_name, metric_history in self.metrics.items():
            if len(metric_history) < 10:
                continue
            try:
                await self._detect_metric_anomaly(metric_name, metric_history)
            except Exception as e:
                self.logger.error('Error detecting anomalies for {metric_name}: %s', e)

    async def _detect_metric_anomaly(self, metric_name: str, metric_history: deque) -> None:
        """Detect anomalies for a specific metric."""
        recent_values = [m.value for m in list(metric_history)[-30:]]
        baseline_values = [m.value for m in list(metric_history)[-300:-30]]
        if len(baseline_values) < 10:
            return
        baseline_mean = statistics.mean(baseline_values)
        baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
        current_value = recent_values[-1] if recent_values else 0
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            if z_score > 3:
                severity = PerformanceLevel.CRITICAL if z_score > 5 else PerformanceLevel.DEGRADED
                anomaly = PerformanceAnomaly(component=self.current_metrics[metric_name].component, metric_name=metric_name, current_value=current_value, expected_range=(baseline_mean - 2 * baseline_std, baseline_mean + 2 * baseline_std), severity=severity, timestamp=datetime.now(timezone.utc), description=f'{metric_name} is {z_score:.1f} standard deviations from baseline', recommendation=self._get_anomaly_recommendation(metric_name, current_value, baseline_mean))
                self.anomalies.append(anomaly)
                for callback in self.anomaly_callbacks:
                    try:
                        callback(anomaly)
                    except Exception as e:
                        self.logger.error('Error in anomaly callback: %s', e)

    def _get_anomaly_recommendation(self, metric_name: str, current_value: float, baseline_mean: float) -> str:
        """Get recommendation for handling anomaly."""
        if 'throughput' in metric_name and current_value < baseline_mean:
            return 'Consider scaling up event bus workers or connection pool size'
        elif 'latency' in metric_name and current_value > baseline_mean:
            return 'Investigate performance bottlenecks and consider optimization'
        elif 'utilization' in metric_name and current_value > baseline_mean:
            return 'Scale up resources to handle increased load'
        else:
            return 'Monitor closely and investigate root cause'

    async def _update_adaptive_thresholds(self) -> None:
        """Update performance thresholds based on historical data."""
        if len(self.metrics.get('event_throughput', [])) % 360 != 0:
            return
        for metric_name, metric_history in self.metrics.items():
            if metric_name in self.thresholds and len(metric_history) > 100:
                await self._update_metric_threshold(metric_name, metric_history)

    async def _update_metric_threshold(self, metric_name: str, metric_history: deque) -> None:
        """Update threshold for a specific metric."""
        try:
            recent_values = [m.value for m in list(metric_history)[-360:]]
            if len(recent_values) < 50:
                return
            recent_values.sort()
            p10 = recent_values[int(len(recent_values) * 0.1)]
            p25 = recent_values[int(len(recent_values) * 0.25)]
            p50 = recent_values[int(len(recent_values) * 0.5)]
            p75 = recent_values[int(len(recent_values) * 0.75)]
            p90 = recent_values[int(len(recent_values) * 0.9)]
            if 'throughput' in metric_name:
                self.thresholds[metric_name] = PerformanceThresholds(excellent=p75, good=p50, acceptable=p25, degraded=p10, critical=p10 * 0.5)
            else:
                self.thresholds[metric_name] = PerformanceThresholds(excellent=p25, good=p50, acceptable=p75, degraded=p90, critical=p90 * 1.5)
            self.logger.debug('Updated adaptive thresholds for %s', metric_name)
        except Exception as e:
            self.logger.error('Error updating threshold for {metric_name}: %s', e)

    def add_anomaly_callback(self, callback: Callable[[PerformanceAnomaly], None]) -> None:
        """Add callback for anomaly notifications."""
        self.anomaly_callbacks.append(callback)

    def get_performance_report(self, hours: int=1) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'period_hours': hours, 'metrics': {}, 'anomalies': [], 'recommendations': []}
        for metric_name, metric_history in self.metrics.items():
            recent_metrics = [m for m in metric_history if m.timestamp >= cutoff_time]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                report['metrics'][metric_name] = {'current': values[-1] if values else 0, 'average': statistics.mean(values), 'min': min(values), 'max': max(values), 'count': len(values), 'unit': recent_metrics[0].unit, 'component': recent_metrics[0].component}
        recent_anomalies = [a for a in self.anomalies if a.timestamp >= cutoff_time]
        report['anomalies'] = [{'component': a.component, 'metric': a.metric_name, 'severity': a.severity.value, 'description': a.description, 'recommendation': a.recommendation, 'timestamp': a.timestamp.isoformat()} for a in recent_anomalies]
        report['recommendations'] = self._generate_recommendations(report['metrics'])
        return report

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        if 'event_throughput' in metrics:
            throughput = metrics['event_throughput']['current']
            if throughput < 2000:
                recommendations.append('Consider increasing event bus worker count')
            elif throughput > 8000:
                recommendations.append('Excellent event throughput - consider this as baseline')
        if 'connection_utilization' in metrics:
            utilization = metrics['connection_utilization']['current']
            if utilization > 0.8:
                recommendations.append('High connection pool utilization - consider scaling up')
            elif utilization < 0.3:
                recommendations.append('Low connection pool utilization - consider scaling down')
        if 'event_latency' in metrics:
            latency = metrics['event_latency']['current']
            if latency > 50:
                recommendations.append('High event processing latency - investigate bottlenecks')
        return recommendations
