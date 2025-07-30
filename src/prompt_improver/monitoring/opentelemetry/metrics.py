"""
OpenTelemetry Metrics Collection for Production Monitoring
=========================================================

Provides comprehensive metrics collection following the RED method
(Rate, Errors, Duration) and custom business metrics for ML operations.
"""

import time
from contextlib import contextmanager
from typing import Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import (
        Counter, Histogram, ObservableCounter,
        ObservableGauge, ObservableUpDownCounter, UpDownCounter
    )
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = Counter = Histogram = None
    ObservableCounter = ObservableGauge = ObservableUpDownCounter = UpDownCounter = None

from .setup import get_meter

logger = logging.getLogger(__name__)

# ML framework functions will be imported later to avoid circular imports
ML_FRAMEWORK_AVAILABLE = False


class MetricType(Enum):
    """Standard metric types for consistent naming."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    UP_DOWN_COUNTER = "up_down_counter"
    OBSERVABLE_COUNTER = "observable_counter"
    OBSERVABLE_GAUGE = "observable_gauge"
    OBSERVABLE_UP_DOWN_COUNTER = "observable_up_down_counter"


@dataclass
class MetricDefinition:
    """Metric definition with metadata."""
    name: str
    description: str
    unit: str
    metric_type: MetricType
    labels: list[str] | None = None
    buckets: list[float] | None = None


class StandardBuckets:
    """Standard histogram buckets for different types of measurements."""
    
    # HTTP request duration (milliseconds)
    HTTP_DURATION_MS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    
    # Database query duration (milliseconds)
    DB_DURATION_MS = [0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    # ML inference duration (milliseconds)
    ML_DURATION_MS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000]
    
    # Cache operation duration (milliseconds)
    CACHE_DURATION_MS = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100]
    
    # Generic duration (seconds)
    DURATION_SECONDS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    # Request/response sizes (bytes)
    SIZE_BYTES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
    
    # Token counts for LLM operations
    TOKEN_COUNT = [1, 10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]


class BaseMetrics:
    """Base class for metric collections."""
    
    def __init__(self, meter_name: str, component: str):
        self.meter_name = meter_name
        self.component = component
        self.meter = get_meter(meter_name) if OTEL_AVAILABLE else None
        self._instruments: dict[str, Any] = {}
    
    def _create_instrument(self, definition: MetricDefinition) -> Any:
        """Create an OpenTelemetry instrument based on definition."""
        if not self.meter:
            return None
        
        try:
            if definition.metric_type == MetricType.COUNTER:
                return self.meter.create_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                return self.meter.create_histogram(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.GAUGE:
                return self.meter.create_gauge(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.UP_DOWN_COUNTER:
                return self.meter.create_up_down_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.OBSERVABLE_COUNTER:
                return self.meter.create_observable_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.OBSERVABLE_GAUGE:
                return self.meter.create_observable_gauge(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            elif definition.metric_type == MetricType.OBSERVABLE_UP_DOWN_COUNTER:
                return self.meter.create_observable_up_down_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )
            else:
                logger.warning(f"Unknown metric type: {definition.metric_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create instrument {definition.name}: {e}")
            return None
    
    def _get_instrument(self, name: str) -> Any:
        """Get or create an instrument by name."""
        return self._instruments.get(name)
    
    def add_common_labels(self, labels: dict[str, str]) -> dict[str, str]:
        """Add common labels to all metrics."""
        common_labels = {
            "component": self.component,
            "service": "prompt-improver"
        }
        common_labels.update(labels or {})
        return common_labels


class HttpMetrics(BaseMetrics):
    """HTTP request/response metrics following RED method."""
    
    def __init__(self):
        super().__init__("http_metrics", "http")
        self._setup_instruments()
    
    def _setup_instruments(self):
        """Create HTTP-specific instruments."""
        definitions = [
            MetricDefinition(
                name="http_requests_total",
                description="Total number of HTTP requests",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="http_request_duration_ms",
                description="HTTP request duration in milliseconds",
                unit="ms",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.HTTP_DURATION_MS
            ),
            MetricDefinition(
                name="http_request_size_bytes",
                description="HTTP request size in bytes",
                unit="bytes",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.SIZE_BYTES
            ),
            MetricDefinition(
                name="http_response_size_bytes",
                description="HTTP response size in bytes",
                unit="bytes",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.SIZE_BYTES
            ),
            MetricDefinition(
                name="http_active_requests",
                description="Number of active HTTP requests",
                unit="1",
                metric_type=MetricType.UP_DOWN_COUNTER
            )
        ]
        
        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        request_size: int | None = None,
        response_size: int | None = None
    ):
        """Record HTTP request metrics."""
        labels = self.add_common_labels({
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx"
        })
        
        # Request count
        if counter := self._get_instrument("http_requests_total"):
            counter.add(1, labels)
        
        # Request duration
        if histogram := self._get_instrument("http_request_duration_ms"):
            histogram.record(duration_ms, labels)
        
        # Request size
        if request_size and (histogram := self._get_instrument("http_request_size_bytes")):
            histogram.record(request_size, labels)
        
        # Response size
        if response_size and (histogram := self._get_instrument("http_response_size_bytes")):
            histogram.record(response_size, labels)
    
    @contextmanager
    def track_active_request(self, method: str, endpoint: str):
        """Context manager to track active requests."""
        labels = self.add_common_labels({
            "method": method,
            "endpoint": endpoint
        })
        
        if counter := self._get_instrument("http_active_requests"):
            counter.add(1, labels)
        
        try:
            yield
        finally:
            if counter := self._get_instrument("http_active_requests"):
                counter.add(-1, labels)


class DatabaseMetrics(BaseMetrics):
    """Database operation metrics."""
    
    def __init__(self):
        super().__init__("database_metrics", "database")
        self._setup_instruments()
    
    def _setup_instruments(self):
        """Create database-specific instruments."""
        definitions = [
            MetricDefinition(
                name="db_queries_total",
                description="Total number of database queries",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="db_query_duration_ms",
                description="Database query duration in milliseconds",
                unit="ms",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.DB_DURATION_MS
            ),
            MetricDefinition(
                name="db_connections_active",
                description="Number of active database connections",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="db_connection_pool_size",
                description="Database connection pool size",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="db_rows_affected",
                description="Number of rows affected by database operations",
                unit="1",
                metric_type=MetricType.HISTOGRAM
            )
        ]
        
        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument
    
    def record_query(
        self,
        operation: str,
        table: str,
        duration_ms: float,
        rows_affected: int | None = None,
        success: bool = True
    ):
        """Record database query metrics."""
        labels = self.add_common_labels({
            "operation": operation,
            "table": table,
            "status": "success" if success else "error"
        })
        
        # Query count
        if counter := self._get_instrument("db_queries_total"):
            counter.add(1, labels)
        
        # Query duration
        if histogram := self._get_instrument("db_query_duration_ms"):
            histogram.record(duration_ms, labels)
        
        # Rows affected
        if rows_affected and (histogram := self._get_instrument("db_rows_affected")):
            histogram.record(rows_affected, labels)
    
    def set_connection_metrics(self, active_connections: int, pool_size: int, pool_name: str = "default"):
        """Set database connection metrics."""
        labels = self.add_common_labels({"pool_name": pool_name})
        
        if gauge := self._get_instrument("db_connections_active"):
            gauge.set(active_connections, labels)
        
        if gauge := self._get_instrument("db_connection_pool_size"):
            gauge.set(pool_size, labels)


class MLMetrics(BaseMetrics):
    """Machine Learning operation metrics - Enhanced for failure analysis and classification."""

    def __init__(self):
        super().__init__("ml_metrics", "ml")
        self._setup_instruments()

    def _setup_instruments(self):
        """Create ML-specific instruments including failure analysis metrics."""
        definitions = [
            # Original ML metrics
            MetricDefinition(
                name="ml_inferences_total",
                description="Total number of ML model inferences",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="ml_inference_duration_ms",
                description="ML model inference duration in milliseconds",
                unit="ms",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.ML_DURATION_MS
            ),
            MetricDefinition(
                name="ml_prompt_tokens",
                description="Number of tokens in ML prompts",
                unit="1",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.TOKEN_COUNT
            ),
            MetricDefinition(
                name="ml_completion_tokens",
                description="Number of tokens in ML completions",
                unit="1",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.TOKEN_COUNT
            ),
            MetricDefinition(
                name="ml_model_accuracy",
                description="ML model accuracy score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_training_iterations",
                description="Total number of ML training iterations",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            # Enhanced failure analysis metrics (replacing prometheus_client)
            MetricDefinition(
                name="ml_failure_rate",
                description="Current ML system failure rate",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_failures_total",
                description="Total number of ML failures",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="ml_response_time_seconds",
                description="ML system response time distribution",
                unit="s",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.DURATION_SECONDS
            ),
            MetricDefinition(
                name="ml_anomaly_score",
                description="Current anomaly detection score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_risk_priority_number",
                description="ML FMEA Risk Priority Number",
                unit="1",
                metric_type=MetricType.GAUGE
            )
        ]

        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument
    
    def record_inference(
        self,
        model_name: str,
        model_version: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        success: bool = True
    ):
        """Record ML inference metrics."""
        labels = self.add_common_labels({
            "model_name": model_name,
            "model_version": model_version,
            "status": "success" if success else "error"
        })
        
        # Inference count
        if counter := self._get_instrument("ml_inferences_total"):
            counter.add(1, labels)
        
        # Inference duration
        if histogram := self._get_instrument("ml_inference_duration_ms"):
            histogram.record(duration_ms, labels)
        
        # Token counts
        if prompt_tokens and (histogram := self._get_instrument("ml_prompt_tokens")):
            histogram.record(prompt_tokens, labels)
        
        if completion_tokens and (histogram := self._get_instrument("ml_completion_tokens")):
            histogram.record(completion_tokens, labels)
    
    def record_training_iteration(self, model_name: str, accuracy: float | None = None):
        """Record ML training iteration."""
        labels = self.add_common_labels({"model_name": model_name})

        if counter := self._get_instrument("ml_training_iterations"):
            counter.add(1, labels)

        if accuracy is not None and (gauge := self._get_instrument("ml_model_accuracy")):
            gauge.set(accuracy, labels)

    def record_failure_analysis(
        self,
        failure_rate: float,
        failure_type: str = "general",
        severity: str = "unknown",
        total_failures: int = 1,
        anomaly_rate: float | None = None,
        rpn_score: float | None = None,
        response_time: float | None = None
    ):
        """Record ML failure analysis metrics (replaces prometheus_client functionality)."""
        labels = self.add_common_labels({
            "failure_type": failure_type,
            "severity": severity
        })

        # Update failure rate
        if gauge := self._get_instrument("ml_failure_rate"):
            gauge.set(failure_rate, labels)

        # Update failure count
        if counter := self._get_instrument("ml_failures_total"):
            counter.add(total_failures, labels)

        # Update anomaly score if available
        if anomaly_rate is not None and (gauge := self._get_instrument("ml_anomaly_score")):
            gauge.set(anomaly_rate, labels)

        # Update RPN score if available
        if rpn_score is not None and (gauge := self._get_instrument("ml_risk_priority_number")):
            gauge.set(rpn_score, labels)

        # Update response time if available
        if response_time is not None and (histogram := self._get_instrument("ml_response_time_seconds")):
            histogram.record(response_time, labels)

    def set_failure_rate(self, failure_rate: float, failure_type: str = "general"):
        """Set current ML failure rate."""
        labels = self.add_common_labels({"failure_type": failure_type})

        if gauge := self._get_instrument("ml_failure_rate"):
            gauge.set(failure_rate, labels)

    def set_anomaly_score(self, anomaly_score: float, detector_type: str = "ensemble"):
        """Set current anomaly detection score."""
        labels = self.add_common_labels({"detector_type": detector_type})

        if gauge := self._get_instrument("ml_anomaly_score"):
            gauge.set(anomaly_score, labels)

    def set_rpn_score(self, rpn_score: float, failure_mode: str = "general"):
        """Set ML FMEA Risk Priority Number."""
        labels = self.add_common_labels({"failure_mode": failure_mode})

        if gauge := self._get_instrument("ml_risk_priority_number"):
            gauge.set(rpn_score, labels)

    def record_response_time(self, response_time_seconds: float, operation: str = "inference"):
        """Record ML system response time."""
        labels = self.add_common_labels({"operation": operation})

        if histogram := self._get_instrument("ml_response_time_seconds"):
            histogram.record(response_time_seconds, labels)


class BusinessMetrics(BaseMetrics):
    """Custom business logic metrics."""
    
    def __init__(self):
        super().__init__("business_metrics", "business")
        self._setup_instruments()
    
    def _setup_instruments(self):
        """Create business-specific instruments."""
        definitions = [
            MetricDefinition(
                name="prompt_improvements_total",
                description="Total number of prompt improvements processed",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="prompt_improvement_quality_score",
                description="Quality score of prompt improvements",
                unit="1",
                metric_type=MetricType.HISTOGRAM
            ),
            MetricDefinition(
                name="active_sessions",
                description="Number of active user sessions",
                unit="1",
                metric_type=MetricType.UP_DOWN_COUNTER
            ),
            MetricDefinition(
                name="feature_flag_evaluations_total",
                description="Total number of feature flag evaluations",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="circuit_breaker_state",
                description="Circuit breaker state (0=closed, 1=half_open, 2=open)",
                unit="1",
                metric_type=MetricType.GAUGE
            )
        ]
        
        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument
    
    def record_prompt_improvement(
        self,
        improvement_type: str,
        quality_score: float,
        user_id: str | None = None
    ):
        """Record prompt improvement metrics."""
        labels = self.add_common_labels({
            "improvement_type": improvement_type
        })
        
        if user_id:
            labels["user_type"] = "authenticated" if user_id != "anonymous" else "anonymous"
        
        # Improvement count
        if counter := self._get_instrument("prompt_improvements_total"):
            counter.add(1, labels)
        
        # Quality score
        if histogram := self._get_instrument("prompt_improvement_quality_score"):
            histogram.record(quality_score, labels)
    
    def update_active_sessions(self, change: int, session_type: str = "default"):
        """Update active session count."""
        labels = self.add_common_labels({"session_type": session_type})
        
        if counter := self._get_instrument("active_sessions"):
            counter.add(change, labels)
    
    def record_feature_flag_evaluation(self, flag_name: str, enabled: bool):
        """Record feature flag evaluation."""
        labels = self.add_common_labels({
            "flag_name": flag_name,
            "enabled": str(enabled).lower()
        })
        
        if counter := self._get_instrument("feature_flag_evaluations_total"):
            counter.add(1, labels)
    
    def set_circuit_breaker_state(self, operation: str, state: str):
        """Set circuit breaker state."""
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
        labels = self.add_common_labels({"operation": operation})
        
        if gauge := self._get_instrument("circuit_breaker_state"):
            gauge.set(state_value, labels)


@dataclass
class OTelAlert:
    """OpenTelemetry-native alert definition (replaces PrometheusAlert)."""
    alert_name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    duration: str = "5m"  # Alert firing duration
    cooldown: str = "10m"  # Alert cooldown period
    triggered_at: datetime | None = None
    callback: Callable[[dict[str, Any]], None] | None = None


class MLAlertingMetrics(BaseMetrics):
    """ML alerting and monitoring system using OpenTelemetry (replaces prometheus alerting)."""

    def __init__(self, alert_thresholds: dict[str, float] | None = None):
        super().__init__("ml_alerting", "ml_alerting")
        self.alert_thresholds = alert_thresholds or {
            "failure_rate": 0.15,
            "response_time_ms": 200,
            "error_rate": 0.05,
            "anomaly_score": 0.8,
        }
        self.alert_definitions: list[OTelAlert] = []
        self.active_alerts: list[OTelAlert] = []
        self.alert_history: list[OTelAlert] = []
        self.alert_cooldown_seconds = 300
        self._setup_instruments()
        self._initialize_alert_definitions()

    def _setup_instruments(self):
        """Create alerting-specific instruments."""
        definitions = [
            MetricDefinition(
                name="ml_alerts_triggered_total",
                description="Total number of ML alerts triggered",
                unit="1",
                metric_type=MetricType.COUNTER
            ),
            MetricDefinition(
                name="ml_alerts_active",
                description="Number of currently active ML alerts",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_alert_duration_seconds",
                description="Duration of ML alerts",
                unit="s",
                metric_type=MetricType.HISTOGRAM,
                buckets=StandardBuckets.DURATION_SECONDS
            )
        ]

        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument

    def _initialize_alert_definitions(self):
        """Initialize alert definitions for ML monitoring."""
        self.alert_definitions = [
            OTelAlert(
                alert_name="HighFailureRate",
                metric_name="ml_failure_rate",
                threshold=self.alert_thresholds["failure_rate"],
                comparison="gt",
                severity="critical",
                description="ML system failure rate exceeds threshold",
            ),
            OTelAlert(
                alert_name="SlowResponseTime",
                metric_name="ml_response_time_seconds",
                threshold=self.alert_thresholds["response_time_ms"] / 1000.0,
                comparison="gt",
                severity="warning",
                description="ML system response time exceeds threshold",
            ),
            OTelAlert(
                alert_name="HighAnomalyScore",
                metric_name="ml_anomaly_score",
                threshold=self.alert_thresholds["anomaly_score"],
                comparison="gt",
                severity="warning",
                description="Anomaly detection score is elevated",
            ),
        ]

        logger.info(f"Initialized {len(self.alert_definitions)} OTel alert definitions")

    def check_alerts(self, failure_analysis: dict[str, Any]) -> list[OTelAlert]:
        """Check alert conditions and trigger alerts if necessary."""
        triggered_alerts = []
        current_time = datetime.now()

        for alert_def in self.alert_definitions:
            metric_value = self._extract_metric_value(failure_analysis, alert_def.metric_name)

            if metric_value is not None and self._check_threshold(metric_value, alert_def):
                if self._is_alert_ready_to_trigger(alert_def, current_time):
                    triggered_alert = self._trigger_alert(alert_def, metric_value, current_time)
                    triggered_alerts.append(triggered_alert)

        return triggered_alerts

    def _extract_metric_value(self, failure_analysis: dict[str, Any], metric_name: str) -> float | None:
        """Extract metric value from failure analysis results."""
        metadata = failure_analysis.get("metadata", {})

        if metric_name == "ml_failure_rate":
            return metadata.get("failure_rate", 0.0)
        elif metric_name == "ml_response_time_seconds":
            return metadata.get("avg_response_time", 0.1)  # Default placeholder
        elif metric_name == "ml_anomaly_score":
            anomaly_detection = failure_analysis.get("anomaly_detection", {})
            if "anomaly_summary" in anomaly_detection:
                return anomaly_detection["anomaly_summary"].get("consensus_anomaly_rate", 0) / 100.0
            return 0.0

        return None

    def _check_threshold(self, value: float, alert_def: OTelAlert) -> bool:
        """Check if metric value exceeds alert threshold."""
        if alert_def.comparison == "gt":
            return value > alert_def.threshold
        elif alert_def.comparison == "lt":
            return value < alert_def.threshold
        elif alert_def.comparison == "eq":
            return abs(value - alert_def.threshold) < 0.001
        return False

    def _is_alert_ready_to_trigger(self, alert_def: OTelAlert, current_time: datetime) -> bool:
        """Check if alert is not in cooldown period."""
        for active_alert in self.active_alerts:
            if active_alert.alert_name == alert_def.alert_name:
                if active_alert.triggered_at:
                    time_since_trigger = (current_time - active_alert.triggered_at).total_seconds()
                    if time_since_trigger < self.alert_cooldown_seconds:
                        return False
        return True

    def _trigger_alert(self, alert_def: OTelAlert, metric_value: float, current_time: datetime) -> OTelAlert:
        """Trigger an alert and record metrics."""
        triggered_alert = OTelAlert(
            alert_name=alert_def.alert_name,
            metric_name=alert_def.metric_name,
            threshold=alert_def.threshold,
            comparison=alert_def.comparison,
            severity=alert_def.severity,
            description=alert_def.description,
            triggered_at=current_time,
            callback=alert_def.callback
        )

        self.active_alerts.append(triggered_alert)
        self.alert_history.append(triggered_alert)

        # Record alert metrics
        labels = self.add_common_labels({
            "alert_name": alert_def.alert_name,
            "severity": alert_def.severity,
            "metric_name": alert_def.metric_name
        })

        if counter := self._get_instrument("ml_alerts_triggered_total"):
            counter.add(1, labels)

        if gauge := self._get_instrument("ml_alerts_active"):
            gauge.set(len(self.active_alerts), labels)

        # Execute callback if provided
        if triggered_alert.callback:
            try:
                triggered_alert.callback({
                    "alert": triggered_alert,
                    "metric_value": metric_value,
                    "timestamp": current_time
                })
            except Exception as e:
                logger.error(f"Alert callback failed for {alert_def.alert_name}: {e}")

        logger.warning(
            f"Alert triggered: {alert_def.alert_name} - {alert_def.description} "
            f"(value: {metric_value}, threshold: {alert_def.threshold})"
        )

        return triggered_alert


# Global metric instances
_http_metrics: HttpMetrics | None = None
_database_metrics: DatabaseMetrics | None = None
_ml_metrics: MLMetrics | None = None
_business_metrics: BusinessMetrics | None = None
_ml_alerting_metrics: MLAlertingMetrics | None = None


def get_http_metrics() -> HttpMetrics:
    """Get global HTTP metrics instance."""
    global _http_metrics
    if _http_metrics is None:
        _http_metrics = HttpMetrics()
    return _http_metrics


def get_database_metrics() -> DatabaseMetrics:
    """Get global database metrics instance."""
    global _database_metrics
    if _database_metrics is None:
        _database_metrics = DatabaseMetrics()
    return _database_metrics


def get_ml_metrics() -> MLMetrics:
    """Get global ML metrics instance."""
    global _ml_metrics
    if _ml_metrics is None:
        _ml_metrics = MLMetrics()
    return _ml_metrics


def get_business_metrics() -> BusinessMetrics:
    """Get global business metrics instance."""
    global _business_metrics
    if _business_metrics is None:
        _business_metrics = BusinessMetrics()
    return _business_metrics


def get_ml_alerting_metrics(alert_thresholds: dict[str, float] | None = None) -> MLAlertingMetrics:
    """Get global ML alerting metrics instance."""
    global _ml_alerting_metrics
    if _ml_alerting_metrics is None:
        _ml_alerting_metrics = MLAlertingMetrics(alert_thresholds)
    return _ml_alerting_metrics


# Convenience functions for quick metric recording
def record_counter(
    name: str,
    value: int | float = 1,
    labels: dict[str, str] | None = None,
    meter_name: str = "default"
):
    """Record a counter metric."""
    if not OTEL_AVAILABLE:
        return
        
    meter = get_meter(meter_name)
    if meter:
        counter = meter.create_counter(name)
        counter.add(value, labels or {})


def record_histogram(
    name: str,
    value: int | float,
    labels: dict[str, str] | None = None,
    meter_name: str = "default"
):
    """Record a histogram metric."""
    if not OTEL_AVAILABLE:
        return
        
    meter = get_meter(meter_name)
    if meter:
        histogram = meter.create_histogram(name)
        histogram.record(value, labels or {})


def record_gauge(
    name: str,
    value: int | float,
    labels: dict[str, str] | None = None,
    meter_name: str = "default"
):
    """Record a gauge metric."""
    if not OTEL_AVAILABLE:
        return
        
    meter = get_meter(meter_name)
    if meter:
        gauge = meter.create_gauge(name)
        gauge.set(value, labels or {})


@contextmanager
def time_operation(
    operation_name: str,
    labels: dict[str, str] | None = None,
    meter_name: str = "default"
):
    """Context manager to time an operation."""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_histogram(
            f"{operation_name}_duration_ms",
            duration_ms,
            labels,
            meter_name
        )
