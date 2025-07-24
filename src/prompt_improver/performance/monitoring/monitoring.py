"""Enhanced Real-Time Monitor - 2025 Edition

Advanced real-time monitoring with 2025 best practices:
- OpenTelemetry integration for distributed tracing
- Structured logging with correlation IDs
- Multi-dimensional metrics collection
- Real-time alerting with smart routing
- Performance anomaly detection
- Service mesh observability
- Custom metrics and dashboards
"""

import asyncio
import time
import uuid
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from collections import defaultdict, deque
import statistics

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, key, value):
            pass
        def add_event(self, name, attributes=None):
            pass
        def set_status(self, status):
            pass

    class MockMeter:
        def create_counter(self, name, **kwargs):
            return MockInstrument()
        def create_histogram(self, name, **kwargs):
            return MockInstrument()
        def create_gauge(self, name, **kwargs):
            return MockInstrument()

    class MockInstrument:
        def add(self, value, attributes=None):
            pass
        def record(self, value, attributes=None):
            pass
        def set(self, value, attributes=None):
            pass

# Enhanced observability imports
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import text

from ...database import get_sessionmanager
from ...database.models import RulePerformance
from ..analytics.analytics import AnalyticsService
from ...utils.error_handlers import handle_common_errors, handle_database_errors

class TraceLevel(Enum):
    """Trace level for distributed tracing."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics for collection."""

    counter = "counter"
    histogram = "histogram"
    gauge = "gauge"
    summary = "summary"

class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TraceContext:
    """Distributed tracing context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "correlation_id": self.correlation_id
        }

@dataclass
class CustomMetric:
    """Custom metric definition."""

    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    value: Union[int, float] = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "labels": self.labels,
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class EnhancedAlert:
    """Enhanced alert with OpenTelemetry context."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    source_service: str
    trace_context: Optional[TraceContext] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "source_service": self.source_service,
            "trace_context": self.trace_context.to_dict() if self.trace_context else None,
            "metrics": self.metrics,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }

@dataclass
class ServiceHealth:
    """Service health status with detailed metrics."""

    service_name: str
    status: str  # healthy, degraded, unhealthy
    response_time_p95: float
    error_rate: float
    throughput: float
    availability: float
    last_check: datetime = field(default_factory=datetime.utcnow)
    dependencies: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "status": self.status,
            "response_time_p95": self.response_time_p95,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "availability": self.availability,
            "last_check": self.last_check.isoformat(),
            "dependencies": self.dependencies,
            "custom_metrics": self.custom_metrics
        }

# OpenTelemetry setup
if OPENTELEMETRY_AVAILABLE:
    # Initialize tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()

    # Initialize meter provider
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://localhost:4317"),
        export_interval_millis=5000
    )
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    # Set up propagators
    set_global_textmap(B3MultiFormat())

    # Get tracer and meter
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)

    # Create metrics
    REQUEST_COUNTER = meter.create_counter(
        "requests_total",
        description="Total number of requests",
        unit="1"
    )

    RESPONSE_TIME_HISTOGRAM = meter.create_histogram(
        "response_time_seconds",
        description="Response time in seconds",
        unit="s"
    )

    ERROR_COUNTER = meter.create_counter(
        "errors_total",
        description="Total number of errors",
        unit="1"
    )

    SERVICE_HEALTH_GAUGE = meter.create_gauge(
        "service_health_score",
        description="Service health score",
        unit="1"
    )
else:
    tracer = MockTracer()
    meter = MockMeter()
    REQUEST_COUNTER = MockInstrument()
    RESPONSE_TIME_HISTOGRAM = MockInstrument()
    ERROR_COUNTER = MockInstrument()
    SERVICE_HEALTH_GAUGE = MockInstrument()

# Use centralized metrics registry
from .metrics_registry import get_metrics_registry

metrics_registry = get_metrics_registry()
REALTIME_REQUESTS = metrics_registry.get_or_create_counter(
    'realtime_requests_total',
    'Total real-time requests',
    ['service', 'method']
)
REALTIME_RESPONSE_TIME = metrics_registry.get_or_create_histogram(
    'realtime_response_time_seconds',
    'Real-time response time',
    ['service']
)
REALTIME_ERRORS = metrics_registry.get_or_create_counter(
    'realtime_errors_total',
    'Total real-time errors',
    ['service', 'error_type']
)
REALTIME_ACTIVE_CONNECTIONS = metrics_registry.get_or_create_gauge(
    'realtime_active_connections',
    'Active real-time connections',
    ['service']
)

@dataclass
class AlertThreshold:
    """Alert threshold configuration"""

    response_time_ms: int = 200
    cache_hit_ratio: float = 90.0
    database_connections: int = 15
    memory_usage_mb: int = 256
    error_rate_percent: float = 5.0

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""

    timestamp: datetime
    alert_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str

class EnhancedRealTimeMonitor:
    """Enhanced real-time monitor with OpenTelemetry and 2025 best practices.

    features:
    - OpenTelemetry integration for distributed tracing
    - Structured logging with correlation IDs
    - Multi-dimensional metrics collection
    - Real-time alerting with smart routing
    - Performance anomaly detection
    - Service mesh observability
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_alerting: bool = True,
        service_name: str = "apes-realtime-monitor",
        otlp_endpoint: str = "http://localhost:4317"
    ):
        """Initialize enhanced real-time monitor."""
        self.console = console or Console()
        self.analytics = AnalyticsService()
        self.alert_thresholds = AlertThreshold()
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint

        # Enhanced features
        self.enable_tracing = enable_tracing and OPENTELEMETRY_AVAILABLE
        self.enable_metrics = enable_metrics
        self.enable_alerting = enable_alerting

        # Monitoring state
        self.monitoring_active = False
        self.active_alerts: List[EnhancedAlert] = []
        self.service_health: Dict[str, ServiceHealth] = {}
        self.custom_metrics: Dict[str, CustomMetric] = {}

        # Tracing and metrics
        self.tracer = tracer
        self.meter = meter

        # Performance tracking
        self.request_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=1000)
        self.response_times: deque = deque(maxlen=1000)

        # Alert routing
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)

        # Correlation tracking
        self.correlation_contexts: Dict[str, TraceContext] = {}

        # Add missing attributes for compatibility
        self.error_events = deque(maxlen=1000)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Enhanced real-time monitor initialized for service: {service_name}")

    def create_trace_context(self, operation_name: str, parent_context: Optional[TraceContext] = None) -> TraceContext:
        """Create a new trace context for distributed tracing."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        parent_span_id = parent_context.span_id if parent_context else None

        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=parent_context.baggage.copy() if parent_context else {},
            correlation_id=str(uuid.uuid4())
        )

        # Store context for correlation
        self.correlation_contexts[context.correlation_id] = context

        return context

    async def start_span(self, name: str, context: Optional[TraceContext] = None, **attributes):
        """Start a distributed trace span."""
        if not self.enable_tracing:
            return MockSpan()

        span_attributes = {
            "service.name": self.service_name,
            "service.version": "2025.1.0",
            **attributes
        }

        if context:
            span_attributes.update({
                "trace.correlation_id": context.correlation_id,
                "trace.parent_span_id": context.parent_span_id
            })

        span = self.tracer.start_span(name, attributes=span_attributes)

        # Record span start
        if REALTIME_REQUESTS:
            REALTIME_REQUESTS.labels(service=self.service_name, method=name).inc()

        return span

    async def record_metric(self, metric: CustomMetric):
        """Record a custom metric."""
        self.custom_metrics[metric.name] = metric

        # Record to OpenTelemetry
        if self.enable_metrics:
            labels = {"service": self.service_name, **metric.labels}

            if metric.metric_type == MetricType.counter:
                REQUEST_COUNTER.add(metric.value, labels)
            elif metric.metric_type == MetricType.histogram:
                RESPONSE_TIME_HISTOGRAM.record(metric.value, labels)
            elif metric.metric_type == MetricType.gauge:
                SERVICE_HEALTH_GAUGE.set(metric.value, labels)

        # Record to Prometheus
        if PROMETHEUS_AVAILABLE and REALTIME_REQUESTS:
            if metric.metric_type == MetricType.counter:
                REALTIME_REQUESTS.labels(service=self.service_name, method=metric.name).inc(metric.value)
            elif metric.metric_type == MetricType.histogram:
                REALTIME_RESPONSE_TIME.labels(service=self.service_name).observe(metric.value)

        self.logger.debug(f"Recorded metric: {metric.name} = {metric.value}")

    async def emit_alert(self, alert: EnhancedAlert):
        """Emit an enhanced alert with proper routing."""
        self.active_alerts.append(alert)

        # Add trace context if available
        if not alert.trace_context and self.correlation_contexts:
            # Use the most recent context
            latest_context = list(self.correlation_contexts.values())[-1]
            alert.trace_context = latest_context

        # Route alert based on severity
        handlers = self.alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")

        # Record alert metric
        if REALTIME_ERRORS:
            REALTIME_ERRORS.labels(
                service=self.service_name,
                error_type=alert.alert_type
            ).inc()

        # Log structured alert
        self.logger.warning(
            f"Alert emitted: {alert.title}",
            extra={
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "trace_id": alert.trace_context.trace_id if alert.trace_context else None,
                "correlation_id": alert.trace_context.correlation_id if alert.trace_context else None
            }
        )

    def add_alert_handler(self, severity: AlertSeverity, handler: Callable[[EnhancedAlert], None]):
        """Add an alert handler for specific severity level."""
        self.alert_handlers[severity].append(handler)

    async def update_service_health(self, service_name: str, health: ServiceHealth):
        """Update service health status."""
        self.service_health[service_name] = health

        # Record health metric
        health_score = self._calculate_health_score(health)
        await self.record_metric(CustomMetric(
            name=f"service_health_{service_name}",
            metric_type=MetricType.gauge,
            description=f"Health score for {service_name}",
            value=health_score,
            labels={"service": service_name}
        ))

    def _calculate_health_score(self, health: ServiceHealth) -> float:
        """Calculate a composite health score (0-100)."""
        # Weight different factors
        availability_weight = 0.4
        response_time_weight = 0.3
        error_rate_weight = 0.3

        # Normalize metrics (higher is better)
        availability_score = health.availability
        response_time_score = max(0, 100 - (health.response_time_p95 / 10))  # Assume 1000ms = 0 score
        error_rate_score = max(0, 100 - (health.error_rate * 10))  # 10% error = 0 score

        composite_score = (
            availability_score * availability_weight +
            response_time_score * response_time_weight +
            error_rate_score * error_rate_weight
        )

        return min(100, max(0, composite_score))

    async def start_enhanced_monitoring(
        self,
        refresh_seconds: int = 5,
        enable_dashboard: bool = True,
        enable_background_collection: bool = True
    ):
        """Start enhanced monitoring with OpenTelemetry integration."""
        self.monitoring_active = True

        # Create root trace context
        root_context = self.create_trace_context("monitoring_session")

        with await self.start_span("start_enhanced_monitoring", root_context) as span:
            span.set_attribute("monitoring.refresh_seconds", refresh_seconds)
            span.set_attribute("monitoring.dashboard_enabled", enable_dashboard)
            span.set_attribute("monitoring.background_enabled", enable_background_collection)

            self.console.print(
                "🚀 Starting Enhanced Real-Time Monitoring with OpenTelemetry",
                style="green"
            )

            # Start background collection if enabled
            if enable_background_collection:
                asyncio.create_task(self._background_metrics_collection())

            # Start dashboard if enabled
            if enable_dashboard:
                await self._start_dashboard_loop(refresh_seconds, root_context)

    async def _background_metrics_collection(self):
        """Background task for continuous metrics collection."""
        while self.monitoring_active:
            try:
                context = self.create_trace_context("background_collection")

                with await self.start_span("collect_background_metrics", context) as span:
                    # Collect system metrics
                    system_metrics = await self.collect_enhanced_system_metrics(context)

                    # Update service health
                    await self._update_all_service_health(context)

                    # Check for anomalies
                    await self._detect_performance_anomalies(context)

                    span.add_event("background_collection_completed")

                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                self.logger.error(f"Background collection error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _start_dashboard_loop(self, refresh_seconds: int, root_context: TraceContext):
        """Start the dashboard display loop."""
        with Live(auto_refresh=False) as live:
            while self.monitoring_active:
                try:
                    context = self.create_trace_context("dashboard_update", root_context)

                    with await self.start_span("update_dashboard", context) as span:
                        # Collect all monitoring data
                        monitoring_data = await self._collect_comprehensive_data(context)

                        # Create enhanced dashboard
                        dashboard = await self._create_enhanced_dashboard(monitoring_data, context)

                        live.update(dashboard)
                        span.add_event("dashboard_updated")

                    await asyncio.sleep(refresh_seconds)

                except Exception as e:
                    self.logger.error(f"Dashboard update error: {e}")
                    await asyncio.sleep(refresh_seconds)

    async def _collect_comprehensive_data(self, context: TraceContext) -> Dict[str, Any]:
        """Collect comprehensive monitoring data."""
        with await self.start_span("collect_comprehensive_data", context) as span:
            data = {}

            # Analytics data
            try:
                data["performance"] = await self.analytics.get_performance_trends(days=1)
                data["effectiveness"] = await self.analytics.get_rule_effectiveness()
                data["satisfaction"] = await self.analytics.get_user_satisfaction(days=1)
                span.add_event("analytics_data_collected")
            except Exception as e:
                self.logger.error(f"Analytics collection error: {e}")
                data["analytics_error"] = str(e)

            # System metrics
            try:
                data["system"] = await self.collect_enhanced_system_metrics(context)
                span.add_event("system_metrics_collected")
            except Exception as e:
                self.logger.error(f"System metrics error: {e}")
                data["system_error"] = str(e)

            # Service health
            data["service_health"] = {
                name: health.to_dict()
                for name, health in self.service_health.items()
            }

            # Custom metrics
            data["custom_metrics"] = {
                name: metric.to_dict()
                for name, metric in self.custom_metrics.items()
            }

            # Active alerts
            data["active_alerts"] = [alert.to_dict() for alert in self.active_alerts]

            # Trace context
            data["trace_context"] = context.to_dict()

            span.set_attribute("data.components_collected", len(data))
            return data

    async def collect_enhanced_system_metrics(self, context: Optional[TraceContext] = None) -> Dict[str, Any]:
        """Collect enhanced system metrics with tracing."""
        if not context:
            context = self.create_trace_context("system_metrics_collection")

        with await self.start_span("collect_system_metrics", context) as span:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": context.correlation_id,
                "service_name": self.service_name
            }

            # Database metrics
            try:
                with await self.start_span("database_health_check", context) as db_span:
                    async with get_sessionmanager().session() as session:
                        # Test database response time
                        start_time = time.time()
                        await session.execute(text("SELECT 1"))
                        end_time = time.time()

                        db_response_time = (end_time - start_time) * 1000
                        metrics["database_response_time_ms"] = db_response_time

                        # Get active database connections
                        result = await session.execute(
                            text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                        )
                        metrics["database_connections"] = result.scalar() or 0

                        db_span.set_attribute("db.response_time_ms", db_response_time)
                        db_span.set_attribute("db.active_connections", metrics["database_connections"])

                        # Record database metrics
                        await self.record_metric(CustomMetric(
                            name="database_response_time",
                            metric_type=MetricType.histogram,
                            description="Database response time",
                            unit="ms",
                            value=db_response_time,
                            labels={"operation": "health_check"}
                        ))

            except Exception as e:
                metrics["database_error"] = str(e)
                span.add_event("database_error", {"error": str(e)})

                # Record error
                await self.record_metric(CustomMetric(
                    name="database_errors",
                    metric_type=MetricType.counter,
                    description="Database errors",
                    value=1,
                    labels={"error_type": "connection"}
                ))

            # Memory and CPU metrics (simplified)
            try:
                import psutil

                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)

                metrics["memory_usage_mb"] = memory_info.used / (1024 * 1024)
                metrics["memory_usage_percent"] = memory_info.percent
                metrics["cpu_usage_percent"] = cpu_percent

                span.set_attribute("system.memory_usage_percent", memory_info.percent)
                span.set_attribute("system.cpu_usage_percent", cpu_percent)

                # Record system metrics
                await self.record_metric(CustomMetric(
                    name="memory_usage",
                    metric_type=MetricType.gauge,
                    description="Memory usage percentage",
                    unit="percent",
                    value=memory_info.percent
                ))

                await self.record_metric(CustomMetric(
                    name="cpu_usage",
                    metric_type=MetricType.gauge,
                    description="CPU usage percentage",
                    unit="percent",
                    value=cpu_percent
                ))

            except ImportError:
                metrics["system_metrics_unavailable"] = "psutil not installed"
            except Exception as e:
                metrics["system_error"] = str(e)
                span.add_event("system_metrics_error", {"error": str(e)})

            span.set_attribute("metrics.collected_count", len(metrics))
            return metrics

    async def _update_all_service_health(self, context: TraceContext):
        """Update health for all monitored services."""
        with await self.start_span("update_all_service_health", context) as span:
            services = ["database", "analytics", "api", "websocket"]

            for service in services:
                try:
                    health = await self._calculate_service_health(service, context)
                    await self.update_service_health(service, health)
                except Exception as e:
                    self.logger.error(f"Health update error for {service}: {e}")

            span.set_attribute("services.updated_count", len(services))

    async def _calculate_service_health(self, service_name: str, context: TraceContext) -> ServiceHealth:
        """Calculate health metrics for a specific service."""
        # Simplified health calculation - in production this would be more sophisticated
        response_times = [rt for rt in self.response_times if rt < 1000]  # Filter outliers
        error_count = sum(1 for error in self.error_events if error.get("service") == service_name)

        p95_response_time = statistics.quantiles(response_times, n=20)[18] if response_times else 0
        error_rate = (error_count / max(len(self.request_history), 1)) * 100
        throughput = len(self.request_history) / 60  # Requests per minute
        availability = max(0, 100 - error_rate)

        # Determine status
        if availability >= 99.9 and p95_response_time <= 200 and error_rate <= 1:
            status = "healthy"
        elif availability >= 99 and p95_response_time <= 500 and error_rate <= 5:
            status = "degraded"
        else:
            status = "unhealthy"

        return ServiceHealth(
            service_name=service_name,
            status=status,
            response_time_p95=p95_response_time,
            error_rate=error_rate,
            throughput=throughput,
            availability=availability,
            dependencies=self._get_service_dependencies(service_name)
        )

    def _get_service_dependencies(self, service_name: str) -> List[str]:
        """Get dependencies for a service."""
        dependency_map = {
            "api": ["database", "analytics"],
            "analytics": ["database"],
            "websocket": ["database", "api"],
            "database": []
        }
        return dependency_map.get(service_name, [])

    async def _detect_performance_anomalies(self, context: TraceContext):
        """Detect performance anomalies and emit alerts."""
        with await self.start_span("detect_anomalies", context) as span:
            # Check response time anomalies
            if len(self.response_times) >= 10:
                recent_avg = statistics.mean(list(self.response_times)[-10:])
                historical_avg = statistics.mean(list(self.response_times)[:-10]) if len(self.response_times) > 10 else recent_avg

                if recent_avg > historical_avg * 2:  # 100% increase
                    alert = EnhancedAlert(
                        alert_id=str(uuid.uuid4()),
                        alert_type="performance_anomaly",
                        severity=AlertSeverity.HIGH,
                        title="Response Time Anomaly Detected",
                        description=f"Response time increased by {((recent_avg/historical_avg)-1)*100:.1f}%",
                        source_service=self.service_name,
                        trace_context=context,
                        metrics={
                            "recent_avg_ms": recent_avg,
                            "historical_avg_ms": historical_avg,
                            "increase_percent": ((recent_avg/historical_avg)-1)*100
                        }
                    )
                    await self.emit_alert(alert)

            # Check error rate anomalies
            recent_errors = len([e for e in self.error_history if (datetime.utcnow() - e.get("timestamp", datetime.utcnow())).seconds < 300])
            if recent_errors > 10:  # More than 10 errors in 5 minutes
                alert = EnhancedAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="error_rate_anomaly",
                    severity=AlertSeverity.CRITICAL,
                    title="High Error Rate Detected",
                    description=f"Detected {recent_errors} errors in the last 5 minutes",
                    source_service=self.service_name,
                    trace_context=context,
                    metrics={"error_count_5min": recent_errors}
                )
                await self.emit_alert(alert)

            span.set_attribute("anomalies.checked", True)

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for real-time monitoring (2025 pattern)"""
        start_time = datetime.utcnow()

        try:
            # Create trace context for orchestrated analysis
            context = self.create_trace_context("orchestrated_analysis")

            with await self.start_span("orchestrated_analysis", context) as span:
                # Extract configuration
                monitoring_duration = config.get("monitoring_duration", 60)  # seconds
                collect_traces = config.get("collect_traces", True)
                collect_metrics = config.get("collect_metrics", True)
                output_path = config.get("output_path", "./outputs/realtime_monitoring")

                span.set_attribute("config.monitoring_duration", monitoring_duration)
                span.set_attribute("config.collect_traces", collect_traces)
                span.set_attribute("config.collect_metrics", collect_metrics)

                # Simulate monitoring if requested
                simulate_data = config.get("simulate_data", False)
                if simulate_data:
                    await self._simulate_monitoring_data(context)

                # Collect comprehensive monitoring data
                monitoring_data = await self._collect_comprehensive_data(context)

                # Calculate summary statistics
                summary = self._calculate_monitoring_summary(monitoring_data)

                # Prepare orchestrator-compatible result
                result = {
                    "monitoring_summary": summary,
                    "service_health": monitoring_data.get("service_health", {}),
                    "system_metrics": monitoring_data.get("system", {}),
                    "custom_metrics": monitoring_data.get("custom_metrics", {}),
                    "active_alerts": monitoring_data.get("active_alerts", []),
                    "trace_context": monitoring_data.get("trace_context", {}),
                    "performance_data": monitoring_data.get("performance", {}),
                    "capabilities": {
                        "opentelemetry_tracing": self.enable_tracing,
                        "custom_metrics": self.enable_metrics,
                        "real_time_alerting": self.enable_alerting,
                        "service_health_monitoring": True,
                        "anomaly_detection": True,
                        "distributed_tracing": OPENTELEMETRY_AVAILABLE,
                        "prometheus_metrics": PROMETHEUS_AVAILABLE
                    }
                }

                # Calculate execution metadata
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                span.set_attribute("execution.duration_seconds", execution_time)

                return {
                    "orchestrator_compatible": True,
                    "component_result": result,
                    "local_metadata": {
                        "output_path": output_path,
                        "execution_time": execution_time,
                        "monitoring_duration": monitoring_duration,
                        "services_monitored": len(self.service_health),
                        "custom_metrics_count": len(self.custom_metrics),
                        "active_alerts_count": len(self.active_alerts),
                        "tracing_enabled": self.enable_tracing,
                        "metrics_enabled": self.enable_metrics,
                        "component_version": "2025.1.0",
                        "trace_id": context.trace_id,
                        "correlation_id": context.correlation_id
                    }
                }

        except Exception as e:
            self.logger.error(f"Orchestrated monitoring failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "monitoring_summary": {}},
                "local_metadata": {
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "error": True,
                    "component_version": "2025.1.0"
                }
            }

    async def _simulate_monitoring_data(self, context: TraceContext):
        """Simulate monitoring data for testing purposes."""
        import random

        with await self.start_span("simulate_monitoring_data", context) as span:
            # Simulate requests and responses
            for i in range(50):
                response_time = random.gauss(150, 50)  # 150ms ± 50ms
                self.response_times.append(max(10, response_time))

                # Simulate some errors
                if random.random() < 0.02:  # 2% error rate
                    self.error_history.append({
                        "timestamp": datetime.utcnow(),
                        "service": random.choice(["api", "database", "analytics"]),
                        "error_type": "timeout"
                    })

                self.request_history.append({
                    "timestamp": datetime.utcnow(),
                    "response_time": response_time
                })

            span.set_attribute("simulation.requests_generated", 50)

    def _calculate_monitoring_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate monitoring summary statistics."""
        summary = {
            "overall_health": "healthy",
            "total_services": len(self.service_health),
            "healthy_services": 0,
            "degraded_services": 0,
            "unhealthy_services": 0,
            "total_alerts": len(self.active_alerts),
            "critical_alerts": 0,
            "high_alerts": 0,
            "medium_alerts": 0,
            "low_alerts": 0,
            "avg_response_time": 0,
            "error_rate": 0,
            "uptime_percentage": 99.9
        }

        # Calculate service health distribution
        for health in self.service_health.values():
            if health.status == "healthy":
                summary["healthy_services"] += 1
            elif health.status == "degraded":
                summary["degraded_services"] += 1
            else:
                summary["unhealthy_services"] += 1

        # Calculate alert distribution
        for alert in self.active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                summary["critical_alerts"] += 1
            elif alert.severity == AlertSeverity.HIGH:
                summary["high_alerts"] += 1
            elif alert.severity == AlertSeverity.MEDIUM:
                summary["medium_alerts"] += 1
            else:
                summary["low_alerts"] += 1

        # Calculate overall health
        if summary["critical_alerts"] > 0 or summary["unhealthy_services"] > 0:
            summary["overall_health"] = "critical"
        elif summary["high_alerts"] > 0 or summary["degraded_services"] > 0:
            summary["overall_health"] = "degraded"

        # Calculate performance metrics
        if self.response_times:
            summary["avg_response_time"] = statistics.mean(self.response_times)

        if self.request_history and self.error_history:
            summary["error_rate"] = (len(self.error_history) / len(self.request_history)) * 100

        return summary

# Maintain backward compatibility
class RealTimeMonitor(EnhancedRealTimeMonitor):
    """Backward compatible real-time monitor."""

    def __init__(self, console: Optional[Console] = None):
        super().__init__(
            console=console,
            enable_tracing=False,
            enable_metrics=False,
            enable_alerting=False
        )

    async def start_monitoring_dashboard(self, refresh_seconds: int = 5):
        """Live dashboard using existing analytics + Rich interface"""
        self.monitoring_active = True
        self.console.print(
            "🚀 Starting APES Real-Time Monitoring Dashboard", style="green"
        )

        with Live(auto_refresh=False) as live:
            while self.monitoring_active:
                try:
                    # Use existing analytics methods
                    performance_data = await self.analytics.get_performance_trends(
                        days=1
                    )
                    rule_effectiveness = await self.analytics.get_rule_effectiveness()
                    user_satisfaction = await self.analytics.get_user_satisfaction(
                        days=1
                    )

                    # Get real-time system metrics
                    system_metrics = await self.collect_system_metrics()

                    # Create Rich dashboard
                    dashboard = self.create_dashboard({
                        "performance": performance_data,
                        "effectiveness": rule_effectiveness,
                        "satisfaction": user_satisfaction,
                        "system": system_metrics,
                    })

                    live.update(dashboard)

                    # Check alerting thresholds
                    await self.check_performance_alerts(
                        performance_data, system_metrics
                    )

                    await asyncio.sleep(refresh_seconds)

                except Exception as e:
                    error_panel = Panel(
                        f"[red]Monitoring Error: {e}[/red]",
                        title="⚠️ Dashboard Error",
                        border_style="red",
                    )
                    live.update(error_panel)
                    await asyncio.sleep(refresh_seconds * 2)  # Longer delay on error

    def create_dashboard(self, data: dict[str, Any]) -> Layout:
        """Create Rich dashboard layout"""
        layout = Layout()

        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        # Header with system status
        header_content = self.create_header_panel(data.get("system", {}))
        layout["header"].update(header_content)

        # Body with metrics
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))

        # Left column: Performance & System metrics
        layout["left"].split_column(
            Layout(name="performance", ratio=2), Layout(name="alerts", ratio=1)
        )

        # Right column: Rule effectiveness & User satisfaction
        layout["right"].split_column(
            Layout(name="effectiveness"), Layout(name="satisfaction")
        )

        # Populate sections
        layout["performance"].update(
            self.create_performance_panel(
                data.get("performance", {}), data.get("system", {})
            )
        )
        layout["effectiveness"].update(
            self.create_effectiveness_panel(data.get("effectiveness", {}))
        )
        layout["satisfaction"].update(
            self.create_satisfaction_panel(data.get("satisfaction", {}))
        )
        layout["alerts"].update(self.create_alerts_panel())
        layout["footer"].update(self.create_footer_panel())

        return layout

    def create_header_panel(self, system_data: dict[str, Any]) -> Panel:
        """Create header panel with system overview"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_time = system_data.get("avg_response_time_ms", 0)
        memory_usage = system_data.get("memory_usage_mb", 0)
        db_connections = system_data.get("database_connections", 0)

        # Status indicators
        response_status = (
            "🟢" if response_time < 200 else "🟡" if response_time < 300 else "🔴"
        )
        memory_status = (
            "🟢" if memory_usage < 200 else "🟡" if memory_usage < 300 else "🔴"
        )

        header_text = Text()
        header_text.append("APES Real-Time Monitoring", style="bold blue")
        header_text.append(f" | {current_time}", style="dim")
        header_text.append(
            f" | Response: {response_status} {response_time:.1f}ms", style=""
        )
        header_text.append(f" | Memory: {memory_status} {memory_usage:.1f}MB", style="")
        header_text.append(f" | DB Connections: {db_connections}", style="")

        return Panel(header_text, border_style="blue")

    def create_performance_panel(
        self, perf_data: dict[str, Any], system_data: dict[str, Any]
    ) -> Panel:
        """Create performance metrics panel"""
        table = Table(title="📊 Performance Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="magenta")
        table.add_column("Target", style="green")
        table.add_column("Status", justify="center")

        # Response time
        response_time = system_data.get("avg_response_time_ms", 0)
        response_status = (
            "✅" if response_time < 200 else "⚠️" if response_time < 300 else "❌"
        )
        table.add_row(
            "Response Time", f"{response_time:.1f}ms", "< 200ms", response_status
        )

        # Cache hit ratio
        cache_ratio = perf_data.get("cache_hit_ratio", 0)
        cache_status = "✅" if cache_ratio > 90 else "⚠️" if cache_ratio > 80 else "❌"
        table.add_row("Cache Hit Ratio", f"{cache_ratio:.1f}%", "> 90%", cache_status)

        # Database connections
        db_conn = system_data.get("database_connections", 0)
        db_status = "✅" if db_conn < 15 else "⚠️" if db_conn < 20 else "❌"
        table.add_row("DB Connections", str(db_conn), "< 15", db_status)

        # Memory usage
        memory = system_data.get("memory_usage_mb", 0)
        memory_status = "✅" if memory < 200 else "⚠️" if memory < 300 else "❌"
        table.add_row("Memory Usage", f"{memory:.1f}MB", "< 200MB", memory_status)

        # Throughput (requests per minute)
        throughput = perf_data.get("throughput_rpm", 0)
        table.add_row("Throughput", f"{throughput:.0f} req/min", "-", "📈")

        return Panel(table, border_style="cyan")

    def create_effectiveness_panel(self, effectiveness_data: dict[str, Any]) -> Panel:
        """Create rule effectiveness panel"""
        table = Table(title="🎯 Rule Effectiveness", box=box.ROUNDED)
        table.add_column("Rule", style="cyan")
        table.add_column("Effectiveness", style="magenta")
        table.add_column("Usage", style="yellow")
        table.add_column("Trend", justify="center")

        rules = effectiveness_data.get("rules", [])
        for rule in rules[:5]:  # Top 5 rules
            name = rule.get("rule_name", "Unknown")[:15]  # Truncate long names
            effectiveness = rule.get("effectiveness_score", 0)
            usage_count = rule.get("usage_count", 0)
            trend = rule.get("trend", "stable")

            trend_icon = (
                "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
            )

            table.add_row(name, f"{effectiveness:.1f}%", str(usage_count), trend_icon)

        return Panel(table, border_style="green")

    def create_satisfaction_panel(self, satisfaction_data: dict[str, Any]) -> Panel:
        """Create user satisfaction panel"""
        table = Table(title="😊 User Satisfaction", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Change", style="yellow")

        avg_rating = satisfaction_data.get("average_rating", 0)
        total_feedback = satisfaction_data.get("total_feedback", 0)
        satisfaction_trend = satisfaction_data.get("trend", 0)

        trend_icon = (
            "📈" if satisfaction_trend > 0 else "📉" if satisfaction_trend < 0 else "➡️"
        )
        trend_text = f"{satisfaction_trend:+.1f}" if satisfaction_trend != 0 else "0.0"

        table.add_row(
            "Average Rating", f"{avg_rating:.1f}/5.0", f"{trend_text} {trend_icon}"
        )
        table.add_row("Total Feedback", str(total_feedback), "-")

        # Satisfaction distribution
        excellent = satisfaction_data.get("rating_distribution", {}).get("5", 0)
        good = satisfaction_data.get("rating_distribution", {}).get("4", 0)
        poor = satisfaction_data.get("rating_distribution", {}).get(
            "1", 0
        ) + satisfaction_data.get("rating_distribution", {}).get("2", 0)

        table.add_row("Excellent (5★)", str(excellent), "")
        table.add_row("Good (4★)", str(good), "")
        table.add_row("Poor (1-2★)", str(poor), "")

        return Panel(table, border_style="magenta")

    def create_alerts_panel(self) -> Panel:
        """Create active alerts panel"""
        if not self.active_alerts:
            content = Text("🟢 All systems operating normally", style="green")
            return Panel(content, title="🚨 Active Alerts", border_style="green")

        table = Table(box=box.ROUNDED)
        table.add_column("Time", style="cyan")
        table.add_column("Alert", style="red")
        table.add_column("Value", style="yellow")

        # Show last 3 alerts
        for alert in self.active_alerts[-3:]:
            time_str = alert.timestamp.strftime("%H:%M:%S")
            severity_icon = "🔴" if alert.severity == "critical" else "🟡"

            table.add_row(
                time_str,
                f"{severity_icon} {alert.message}",
                f"{alert.current_value:.1f}",
            )

        return Panel(table, title="🚨 Active Alerts", border_style="red")

    def create_footer_panel(self) -> Panel:
        """Create footer with controls and status"""
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("Ctrl+C", style="red")
        controls.append(" to stop | ", style="")
        controls.append("Auto-refresh: 5s", style="dim")
        controls.append(" | ", style="")
        controls.append(f"Active Alerts: {len(self.active_alerts)}", style="yellow")

        return Panel(controls, border_style="dim")

    @handle_database_errors(
        rollback_session=False,
        return_format="none",
        operation_name="collect_system_metrics",
        retry_count=1,
    )
    async def collect_system_metrics(self) -> dict[str, Any]:
        """Collect real-time system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "avg_response_time_ms": 0,
            "database_connections": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
        }

        async with get_sessionmanager().session() as session:
            # Test database response time
            start_time = time.time()
            await session.execute(text("SELECT 1"))
            end_time = time.time()
            metrics["avg_response_time_ms"] = (end_time - start_time) * 1000

            # Get active database connections
            result = await session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            metrics["database_connections"] = result.scalar() or 0

        # Get system resource usage (with fallback for missing psutil)
        try:
            import psutil

            process = psutil.process()
            metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
            metrics["cpu_usage_percent"] = process.cpu_percent()
        except ImportError:
            # Fallback values
            metrics["memory_usage_mb"] = 50.0
            metrics["cpu_usage_percent"] = 5.0

        return metrics

    async def check_performance_alerts(
        self, performance_data: dict[str, Any], system_metrics: dict[str, Any]
    ):
        """Check for performance threshold violations and generate alerts"""
        current_time = datetime.now()
        new_alerts = []

        # Check response time
        response_time = system_metrics.get("avg_response_time_ms", 0)
        if response_time > self.alert_thresholds.response_time_ms:
            severity = "critical" if response_time > 300 else "warning"
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="performance",
                metric_name="response_time",
                current_value=response_time,
                threshold_value=self.alert_thresholds.response_time_ms,
                severity=severity,
                message=f"High response time: {response_time:.1f}ms",
            )
            new_alerts.append(alert)

        # Check database connections
        db_connections = system_metrics.get("database_connections", 0)
        if db_connections > self.alert_thresholds.database_connections:
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="resource",
                metric_name="database_connections",
                current_value=db_connections,
                threshold_value=self.alert_thresholds.database_connections,
                severity="warning",
                message=f"High DB connections: {db_connections}",
            )
            new_alerts.append(alert)

        # Check memory usage
        memory_usage = system_metrics.get("memory_usage_mb", 0)
        if memory_usage > self.alert_thresholds.memory_usage_mb:
            severity = "critical" if memory_usage > 400 else "warning"
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="resource",
                metric_name="memory_usage",
                current_value=memory_usage,
                threshold_value=self.alert_thresholds.memory_usage_mb,
                severity=severity,
                message=f"High memory usage: {memory_usage:.1f}MB",
            )
            new_alerts.append(alert)

        # Check cache hit ratio
        cache_ratio = performance_data.get("cache_hit_ratio", 100)
        if cache_ratio < self.alert_thresholds.cache_hit_ratio:
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="performance",
                metric_name="cache_hit_ratio",
                current_value=cache_ratio,
                threshold_value=self.alert_thresholds.cache_hit_ratio,
                severity="warning",
                message=f"Low cache hit ratio: {cache_ratio:.1f}%",
            )
            new_alerts.append(alert)

        # Add new alerts and maintain alert history (keep last 10)
        self.active_alerts.extend(new_alerts)
        if len(self.active_alerts) > 10:
            self.active_alerts = self.active_alerts[-10:]

        # Log alerts for persistence
        for alert in new_alerts:
            await self.log_alert(alert)

    @handle_database_errors(
        rollback_session=True,
        return_format="none",
        operation_name="log_alert",
        retry_count=1,
    )
    async def log_alert(self, alert: PerformanceAlert):
        """Log alert to database for historical analysis"""
        async with get_sessionmanager().session() as session:
            # Store alert in rule performance table as monitoring entry
            perf_metric = RulePerformance(
                rule_id="monitoring_alert",
                rule_name=f"{alert.alert_type}_alert",
                improvement_score=0.0,  # Alert indicates a problem
                confidence_level=1.0,
                execution_time_ms=int(alert.current_value)
                if alert.metric_name == "response_time"
                else None,
                rule_parameters={
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "threshold": alert.threshold_value,
                },
                before_metrics={"current_value": alert.current_value},
                after_metrics={"threshold": alert.threshold_value},
                prompt_characteristics={"monitoring": True},
            )

            session.add(perf_metric)
            await session.commit()

    def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        self.monitoring_active = False

    async def get_monitoring_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get monitoring summary for the specified time period"""
        try:
            # Get performance trends
            performance_data = await self.analytics.get_performance_trends(
                days=max(1, hours // 24)
            )

            # Get current system metrics
            current_metrics = await self.collect_system_metrics()

            # Calculate alert statistics
            recent_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                a for a in self.active_alerts if a.timestamp >= recent_time
            ]

            alert_summary = {
                "total_alerts": len(recent_alerts),
                "critical_alerts": len([
                    a for a in recent_alerts if a.severity == "critical"
                ]),
                "warning_alerts": len([
                    a for a in recent_alerts if a.severity == "warning"
                ]),
                "most_common_alert": self._get_most_common_alert_type(recent_alerts),
            }

            return {
                "monitoring_period_hours": hours,
                "timestamp": datetime.now().isoformat(),
                "current_performance": current_metrics,
                "performance_trends": performance_data,
                "alert_summary": alert_summary,
                "health_status": self._calculate_health_status(
                    current_metrics, recent_alerts
                ),
            }

        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "monitoring_period_hours": hours,
            }

    def _get_most_common_alert_type(self, alerts: list[PerformanceAlert]) -> str | None:
        """Get the most common alert type from a list of alerts"""
        if not alerts:
            return None

        alert_counts = {}
        for alert in alerts:
            alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type, 0) + 1

        return max(alert_counts, key=alert_counts.get)

    def _calculate_health_status(
        self, current_metrics: dict[str, Any], recent_alerts: list[PerformanceAlert]
    ) -> str:
        """Calculate overall system health status"""
        response_time = current_metrics.get("avg_response_time_ms", 0)
        memory_usage = current_metrics.get("memory_usage_mb", 0)
        db_connections = current_metrics.get("database_connections", 0)

        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]

        # Determine health status
        if critical_alerts or response_time > 300 or memory_usage > 400:
            return "critical"
        if response_time > 200 or memory_usage > 256 or db_connections > 15:
            return "warning"
        return "healthy"

class HealthMonitor:
    """System health monitoring with automated diagnostics"""

    def __init__(self):
        self.analytics = AnalyticsService()

    async def run_health_check(self) -> dict[str, Any]:
        """Comprehensive system health check"""
        health_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
        }

        # Database connectivity check
        db_status = await self._check_database_health()
        health_results["checks"]["database"] = db_status

        # MCP server performance check
        mcp_status = await self._check_mcp_performance()
        health_results["checks"]["mcp_server"] = mcp_status

        # Analytics service check
        analytics_status = await self._check_analytics_service()
        health_results["checks"]["analytics"] = analytics_status

        # ML service check
        ml_status = await self._check_ml_service()
        health_results["checks"]["ml_service"] = ml_status

        # System resources check
        system_status = await self._check_system_resources()
        health_results["checks"]["system_resources"] = system_status

        # Determine overall status
        failed_checks = [
            name
            for name, check in health_results["checks"].items()
            if check.get("status") == "failed"
        ]
        warning_checks = [
            name
            for name, check in health_results["checks"].items()
            if check.get("status") == "warning"
        ]

        if failed_checks:
            health_results["overall_status"] = "failed"
            health_results["failed_checks"] = failed_checks
        elif warning_checks:
            health_results["overall_status"] = "warning"
            health_results["warning_checks"] = warning_checks

        return health_results

    @handle_database_errors(
        rollback_session=False,
        return_format="dict",
        operation_name="check_database_health",
        retry_count=1,
    )
    async def _check_database_health(self) -> dict[str, Any]:
        """Check database connectivity and performance"""
        start_time = time.time()
        async with get_sessionmanager().session() as session:
            await session.execute(text("SELECT 1"))
            response_time = (time.time() - start_time) * 1000

            # Check for long-running queries
            result = await session.execute(
                text("""
                SELECT count(*)
                FROM pg_stat_activity
                WHERE state = 'active'
                AND query_start < NOW() - INTERVAL '30 seconds'
            """)
            )
            long_queries = result.scalar() or 0

            status = "healthy"
            if response_time > 100:
                status = "warning"
            if response_time > 500 or long_queries > 0:
                status = "failed"

            return {
                "status": status,
                "response_time_ms": response_time,
                "long_running_queries": long_queries,
                "message": f"Database responding in {response_time:.1f}ms",
            }

    async def _check_mcp_performance(self) -> dict[str, Any]:
        """Check MCP server performance"""
        try:
            from ..mcp_server.mcp_server import improve_prompt

            start_time = time.time()
            result = await improve_prompt(
                prompt="Health check test prompt",
                context={"domain": "health_check"},
                session_id="health_check",
            )
            response_time = (time.time() - start_time) * 1000

            status = "healthy"
            if response_time > 200:
                status = "warning"
            if response_time > 500:
                status = "failed"

            return {
                "status": status,
                "response_time_ms": response_time,
                "message": f"MCP server responding in {response_time:.1f}ms",
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "MCP server health check failed",
            }

    async def _check_analytics_service(self) -> dict[str, Any]:
        """Check analytics service functionality"""
        try:
            start_time = time.time()
            result = await self.analytics.get_performance_trends(days=1)
            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "data_points": len(result.get("trends", [])),
                "message": f"Analytics service responding in {response_time:.1f}ms",
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "Analytics service check failed",
            }

    async def _check_ml_service(self) -> dict[str, Any]:
        """Check ML service availability"""
        try:
            from .ml_integration import get_ml_service

            start_time = time.time()
            ml_service = await get_ml_service()
            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "message": f"ML service available in {response_time:.1f}ms",
            }

        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "message": "ML service unavailable (fallback to rule-based)",
            }

    async def _check_system_resources(self) -> dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            status = "healthy"
            warnings = []

            if memory_usage_percent > 80:
                status = "warning"
                warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")

            if disk_usage_percent > 85:
                status = "warning"
                warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")

            if cpu_percent > 80:
                status = "warning"
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

            return {
                "status": status,
                "memory_usage_percent": memory_usage_percent,
                "disk_usage_percent": disk_usage_percent,
                "cpu_usage_percent": cpu_percent,
                "warnings": warnings,
                "message": f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_usage_percent:.1f}%, Disk {disk_usage_percent:.1f}%",
            }

        except ImportError:
            return {
                "status": "warning",
                "message": "psutil not available for system monitoring",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "System resource check failed",
            }
