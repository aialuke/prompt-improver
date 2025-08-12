"""OpenTelemetry Metrics Collection for Production Monitoring
=========================================================

Provides comprehensive metrics collection following the RED method
(Rate, Errors, Duration) and custom business metrics for ML operations.
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from prompt_improver.monitoring.opentelemetry.setup import get_meter

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import (
        Counter,
        Histogram,
        ObservableCounter,
        ObservableGauge,
        ObservableUpDownCounter,
        UpDownCounter,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = Counter = Histogram = None
    ObservableCounter = ObservableGauge = ObservableUpDownCounter = UpDownCounter = None
logger = logging.getLogger(__name__)
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


class MetricDefinition(BaseModel):
    """Metric definition with metadata."""

    name: str
    description: str
    unit: str
    metric_type: MetricType
    labels: list[str] | None = Field(default=None)
    buckets: list[float] | None = Field(default=None)


class StandardBuckets:
    """Standard histogram buckets for OpenTelemetry metrics distribution analysis."""

    REQUEST_DURATION = [
        1.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1000.0,
        2500.0,
        5000.0,
    ]
    DATABASE_DURATION = [
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1000.0,
    ]
    CACHE_DURATION = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    ML_INFERENCE_DURATION = [
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ]


class HttpMetrics:
    """HTTP-specific metrics collection for OpenTelemetry."""

    def __init__(self, service_name: str = "prompt-improver"):
        self.service_name = service_name
        self.meter = get_meter(__name__)
        # Test visibility holders
        self._test_last_request: dict | None = None
        self._test_request_count: int = 0
        if OTEL_AVAILABLE and self.meter is not None:
            self.request_counter = self.meter.create_counter(
                name="http_requests_total",
                description="Total number of HTTP requests",
                unit="1",
            )
            self.request_duration = self.meter.create_histogram(
                name="http_request_duration_ms",
                description="HTTP request duration in milliseconds",
                unit="ms",
            )
            self.response_size = self.meter.create_histogram(
                name="http_response_size_bytes",
                description="HTTP response size in bytes",
                unit="bytes",
            )
        else:
            self.request_counter = self._create_noop_instrument()
            self.request_duration = self._create_noop_instrument()
            self.response_size = self._create_noop_instrument()

    def _create_noop_instrument(self):
        """Create no-op instrument when OpenTelemetry is not available."""

        class NoOpInstrument:
            def add(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

        return NoOpInstrument()

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        response_size_bytes: int = 0,
    ):
        """Record HTTP request metrics."""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "service": self.service_name,
        }
        self.request_counter.add(1, labels)
        self.request_duration.record(duration_ms, labels)
        if response_size_bytes > 0:
            self.response_size.record(response_size_bytes, labels)
        # Test visibility (safe side channel for integration tests without mocks)
        self._test_request_count += 1
        self._test_last_request = {
            "method": method,
            "endpoint": endpoint,
            "status_code": int(status_code),
            "duration_ms": float(duration_ms),
            "response_size_bytes": int(response_size_bytes),
        }


class DatabaseMetrics:
    """Database-specific metrics collection for OpenTelemetry."""

    def __init__(self, service_name: str = "prompt-improver"):
        self.service_name = service_name
        self.meter = get_meter(__name__)
        if OTEL_AVAILABLE and self.meter is not None:
            self.query_counter = self.meter.create_counter(
                name="database_queries_total",
                description="Total number of database queries",
                unit="1",
            )
            self.query_duration = self.meter.create_histogram(
                name="database_query_duration_ms",
                description="Database query duration in milliseconds",
                unit="ms",
            )
            self.connection_pool = self.meter.create_gauge(
                name="database_connection_pool_active",
                description="Active database connections",
                unit="1",
            )
        else:
            self.query_counter = self._create_noop_instrument()
            self.query_duration = self._create_noop_instrument()
            self.connection_pool = self._create_noop_instrument()

    def _create_noop_instrument(self):
        """Create no-op instrument when OpenTelemetry is not available."""

        class NoOpInstrument:
            def add(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

        return NoOpInstrument()

    def record_query(
        self, operation: str, table: str, duration_ms: float, success: bool = True
    ):
        """Record database query metrics."""
        labels = {
            "operation": operation,
            "table": table,
            "success": str(success),
            "service": self.service_name,
        }
        self.query_counter.add(1, labels)
        self.query_duration.record(duration_ms, labels)


class MLMetrics:
    """ML-specific metrics collection for OpenTelemetry."""

    def __init__(self, service_name: str = "prompt-improver"):
        self.service_name = service_name
        self.meter = get_meter(__name__)
        if OTEL_AVAILABLE and self.meter is not None:
            self.inference_counter = self.meter.create_counter(
                name="ml_inferences_total",
                description="Total number of ML inferences",
                unit="1",
            )
            self.inference_duration = self.meter.create_histogram(
                name="ml_inference_duration_s",
                description="ML inference duration in seconds",
                unit="s",
            )
            self.model_accuracy = self.meter.create_histogram(
                name="ml_model_accuracy_score",
                description="ML model accuracy score",
                unit="1",
            )
            self.prompt_improvements = self.meter.create_counter(
                name="prompt_improvements_total",
                description="Total number of prompt improvements",
                unit="1",
            )
        else:
            self.inference_counter = self._create_noop_instrument()
            self.inference_duration = self._create_noop_instrument()
            self.model_accuracy = self._create_noop_instrument()
            self.prompt_improvements = self._create_noop_instrument()

    def _create_noop_instrument(self):
        """Create no-op instrument when OpenTelemetry is not available."""

        class NoOpInstrument:
            def add(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

        return NoOpInstrument()

    def record_inference(
        self, model_name: str, duration_s: float, success: bool = True
    ):
        """Record ML inference metrics."""
        labels = {
            "model": model_name,
            "success": str(success),
            "service": self.service_name,
        }
        self.inference_counter.add(1, labels)
        self.inference_duration.record(duration_s, labels)

    def record_prompt_improvement(self, category: str, improvement_score: float):
        """Record prompt improvement metrics."""
        labels = {"category": category, "service": self.service_name}
        self.prompt_improvements.add(1, labels)
        if improvement_score > 0:
            self.model_accuracy.record(improvement_score, labels)


class BusinessMetrics:
    """Business-specific metrics collection for OpenTelemetry."""

    def __init__(self, service_name: str = "prompt-improver"):
        self.service_name = service_name
        self.meter = get_meter(__name__)
        # Test visibility holders
        self._test_last_journey: dict | None = None
        self._test_journey_count: int = 0
        if OTEL_AVAILABLE and self.meter is not None:
            # Existing BI instruments
            self.feature_usage = self.meter.create_counter(
                name="feature_usage_total",
                description="Total feature usage count",
                unit="1",
            )
            self.user_sessions = self.meter.create_counter(
                name="user_sessions_total", description="Total user sessions", unit="1"
            )
            self.operational_costs = self.meter.create_histogram(
                name="operational_costs_usd",
                description="Operational costs in USD",
                unit="USD",
            )
            # New user journey instruments
            self.journey_events = self.meter.create_counter(
                name="user_journey_events_total",
                description="Total number of user journey events",
                unit="1",
            )
            self.journey_stage_gauge = self.meter.create_gauge(
                name="user_journey_stage",
                description="Current user journey stage as numeric code",
                unit="1",
            )
        else:
            self.feature_usage = self._create_noop_instrument()
            self.user_sessions = self._create_noop_instrument()
            self.operational_costs = self._create_noop_instrument()
            self.journey_events = self._create_noop_instrument()
            self.journey_stage_gauge = self._create_noop_instrument()

    def _create_noop_instrument(self):
        """Create no-op instrument when OpenTelemetry is not available."""

        class NoOpInstrument:
            def add(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

        return NoOpInstrument()

    def record_feature_usage(self, feature: str, user_tier: str = "unknown"):
        """Record feature usage metrics."""
        labels = {
            "feature": feature,
            "user_tier": user_tier,
            "service": self.service_name,
        }
        self.feature_usage.add(1, labels)

    def record_session(self, user_id: str, session_duration_s: float):
        """Record user session metrics."""
        labels = {"service": self.service_name, "user_id": user_id}
        self.user_sessions.add(1, labels)

    # New: user journey OTEL emission helpers
    def record_journey_event(
        self,
        user_id: str,
        session_id: str,
        stage: str,
        event_type: str,
        success: bool = True,
    ):
        labels = {
            "service": self.service_name,
            "user_id": user_id,
            "session_id": session_id,
            "stage": stage,
            "event_type": event_type,
            "success": str(success),
        }
        self.journey_events.add(1, labels)
        # Optional: map stage to a stable numeric code for gauge
        stage_map = {
            "ONBOARDING": 1,
            "FIRST_USE": 2,
            "REGULAR_USE": 3,
            "POWER_USER": 4,
        }
        code = stage_map.get(stage, 0)
        self.journey_stage_gauge.record(
            code, {"service": self.service_name, "user_id": user_id, "stage": stage}
        )
        # Test visibility (safe side channel)
        self._test_journey_count += 1
        self._test_last_journey = dict(labels) | {"stage_code": code}


_http_metrics: HttpMetrics | None = None
_database_metrics: DatabaseMetrics | None = None
_ml_metrics: MLMetrics | None = None
_business_metrics: BusinessMetrics | None = None


def get_http_metrics(service_name: str = "prompt-improver") -> HttpMetrics:
    """Get or create HTTP metrics instance."""
    global _http_metrics
    if _http_metrics is None:
        _http_metrics = HttpMetrics(service_name)
    return _http_metrics


def get_database_metrics(service_name: str = "prompt-improver") -> DatabaseMetrics:
    """Get or create database metrics instance."""
    global _database_metrics
    if _database_metrics is None:
        _database_metrics = DatabaseMetrics(service_name)
    return _database_metrics


def get_ml_metrics(service_name: str = "prompt-improver") -> MLMetrics:
    """Get or create ML metrics instance."""
    global _ml_metrics
    if _ml_metrics is None:
        _ml_metrics = MLMetrics(service_name)
    return _ml_metrics


def get_business_metrics(service_name: str = "prompt-improver") -> BusinessMetrics:
    """Get or create business metrics instance."""
    global _business_metrics
    if _business_metrics is None:
        _business_metrics = BusinessMetrics(service_name)
    return _business_metrics


def record_counter(name: str, value: float, labels: dict | None = None):
    """Record a counter metric."""
    if not OTEL_AVAILABLE:
        return
    meter = get_meter(__name__)
    counter = meter.create_counter(name, description=f"Counter for {name}")
    counter.add(value, labels or {})


def record_histogram(name: str, value: float, labels: dict | None = None):
    """Record a histogram metric."""
    if not OTEL_AVAILABLE:
        return
    meter = get_meter(__name__)
    histogram = meter.create_histogram(name, description=f"Histogram for {name}")
    histogram.record(value, labels or {})


def record_gauge(name: str, value: float, labels: dict | None = None):
    """Record a gauge metric."""
    if not OTEL_AVAILABLE:
        return
    meter = get_meter(__name__)
    gauge = meter.create_gauge(name, description=f"Gauge for {name}")
    gauge.record(value, labels or {})
