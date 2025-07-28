"""
OpenTelemetry Metrics Collection for Production Monitoring
=========================================================

Provides comprehensive metrics collection following the RED method
(Rate, Errors, Duration) and custom business metrics for ML operations.
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import (
        Counter, Histogram, Gauge, ObservableCounter, 
        ObservableGauge, ObservableUpDownCounter, UpDownCounter
    )
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = Counter = Histogram = Gauge = None
    ObservableCounter = ObservableGauge = ObservableUpDownCounter = UpDownCounter = None

from .setup import get_meter

logger = logging.getLogger(__name__)


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
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None


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
        self._instruments: Dict[str, Any] = {}
    
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
    
    def add_common_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
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
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
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
        rows_affected: Optional[int] = None,
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
    """Machine Learning operation metrics."""
    
    def __init__(self):
        super().__init__("ml_metrics", "ml")
        self._setup_instruments()
    
    def _setup_instruments(self):
        """Create ML-specific instruments."""
        definitions = [
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
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
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
    
    def record_training_iteration(self, model_name: str, accuracy: Optional[float] = None):
        """Record ML training iteration."""
        labels = self.add_common_labels({"model_name": model_name})
        
        if counter := self._get_instrument("ml_training_iterations"):
            counter.add(1, labels)
        
        if accuracy is not None and (gauge := self._get_instrument("ml_model_accuracy")):
            gauge.set(accuracy, labels)


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
        user_id: Optional[str] = None
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


# Global metric instances
_http_metrics: Optional[HttpMetrics] = None
_database_metrics: Optional[DatabaseMetrics] = None
_ml_metrics: Optional[MLMetrics] = None
_business_metrics: Optional[BusinessMetrics] = None


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


# Convenience functions for quick metric recording
def record_counter(
    name: str,
    value: Union[int, float] = 1,
    labels: Optional[Dict[str, str]] = None,
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
    value: Union[int, float],
    labels: Optional[Dict[str, str]] = None,
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
    value: Union[int, float],
    labels: Optional[Dict[str, str]] = None,
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
    labels: Optional[Dict[str, str]] = None,
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
