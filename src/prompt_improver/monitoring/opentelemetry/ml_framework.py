"""
OpenTelemetry ML Monitoring Framework
====================================

Comprehensive ML-specific monitoring framework to replace prometheus_client
usage in ML algorithms. Provides OTel-native metrics collection, HTTP server
replacement, and alerting system for ML failures.

This framework enables migration of:
- failure_analyzer.py (3,163 lines, 47 prometheus refs)
- failure_classifier.py (976 lines, 38 prometheus refs)
- ML ecosystem monitoring across the codebase
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from threading import Lock

try:
    from opentelemetry import metrics
    from opentelemetry.metrics import (
        Counter, Histogram, ObservableCounter,
        ObservableGauge, UpDownCounter
    )
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    metrics = None

from .setup import get_meter
from .metrics import BaseMetrics, MetricType

logger = logging.getLogger(__name__)


class MLMetricType(Enum):
    """ML-specific metric types for failure analysis."""
    FAILURE_RATE = "failure_rate"
    ANOMALY_SCORE = "anomaly_score"
    RPN_SCORE = "rpn_score"
    RESPONSE_TIME = "response_time"
    MODEL_ACCURACY = "model_accuracy"
    DRIFT_SCORE = "drift_score"
    ROBUSTNESS_SCORE = "robustness_score"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class OTelAlert:
    """OpenTelemetry-native alert definition (replaces PrometheusAlert)."""
    alert_name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: AlertSeverity
    description: str
    duration: str = "5m"  # Alert firing duration
    cooldown: str = "10m"  # Alert cooldown period
    triggered_at: datetime | None = None
    last_checked: datetime | None = None
    
    def is_in_cooldown(self) -> bool:
        """Check if alert is in cooldown period."""
        if not self.triggered_at:
            return False
        
        cooldown_seconds = self._parse_duration(self.cooldown)
        return (datetime.now() - self.triggered_at).total_seconds() < cooldown_seconds
    
    def _parse_duration(self, duration: str) -> int:
        """Parse duration string to seconds."""
        if duration.endswith('m'):
            return int(duration[:-1]) * 60
        elif duration.endswith('s'):
            return int(duration[:-1])
        elif duration.endswith('h'):
            return int(duration[:-1]) * 3600
        return 300  # Default 5 minutes


class MLMetricsCollector(BaseMetrics):
    """ML-specific metrics collector using OpenTelemetry."""

    def __init__(self, service_name: str = "ml-monitoring"):
        super().__init__(f"{service_name}_ml_metrics", "ml")
        self.service_name = service_name
        self._lock = Lock()

        if OTEL_AVAILABLE:
            self._initialize_ml_instruments()
        else:
            logger.warning("OpenTelemetry not available, using no-op metrics")
    
    def _initialize_ml_instruments(self):
        """Initialize ML-specific OpenTelemetry instruments."""
        from .metrics import MetricDefinition

        # Define ML-specific metrics
        definitions = [
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
                description="ML operation response time",
                unit="s",
                metric_type=MetricType.HISTOGRAM
            ),
            MetricDefinition(
                name="ml_anomaly_score",
                description="Current anomaly detection score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_rpn_score",
                description="Risk Priority Number score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_model_accuracy",
                description="Current model accuracy",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_drift_score",
                description="Data drift detection score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_robustness_score",
                description="Model robustness validation score",
                unit="1",
                metric_type=MetricType.GAUGE
            ),
            MetricDefinition(
                name="ml_operations_total",
                description="Total ML operations performed",
                unit="1",
                metric_type=MetricType.COUNTER
            )
        ]

        # Create instruments using BaseMetrics pattern
        for definition in definitions:
            instrument = self._create_instrument(definition)
            if instrument:
                self._instruments[definition.name] = instrument

        logger.info(f"Initialized {len(self._instruments)} ML monitoring instruments")
    
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
        if not OTEL_AVAILABLE:
            return
        
        labels = {
            "failure_type": failure_type,
            "severity": severity,
            "service": self.service_name
        }
        
        with self._lock:
            # Update failure rate
            if gauge := self._instruments.get("ml_failure_rate"):
                gauge.set(failure_rate, labels)
            
            # Update failure count
            if counter := self._instruments.get("ml_failures_total"):
                counter.add(total_failures, labels)
            
            # Update anomaly rate if provided
            if anomaly_rate is not None:
                if gauge := self._instruments.get("ml_anomaly_score"):
                    gauge.set(anomaly_rate, labels)
            
            # Update RPN score if provided
            if rpn_score is not None:
                if gauge := self._instruments.get("ml_rpn_score"):
                    gauge.set(rpn_score, labels)
            
            # Update response time if provided
            if response_time is not None:
                if histogram := self._instruments.get("ml_response_time"):
                    histogram.record(response_time, labels)
        
        logger.debug(f"Recorded ML failure analysis: rate={failure_rate}, type={failure_type}")
    
    def record_model_performance(
        self,
        accuracy: float | None = None,
        drift_score: float | None = None,
        robustness_score: float | None = None,
        model_name: str = "default"
    ):
        """Record ML model performance metrics."""
        if not OTEL_AVAILABLE:
            return
        
        labels = {
            "model_name": model_name,
            "service": self.service_name
        }
        
        with self._lock:
            if accuracy is not None:
                if gauge := self._instruments.get("ml_model_accuracy"):
                    gauge.set(accuracy, labels)
            
            if drift_score is not None:
                if gauge := self._instruments.get("ml_drift_score"):
                    gauge.set(drift_score, labels)
            
            if robustness_score is not None:
                if gauge := self._instruments.get("ml_robustness_score"):
                    gauge.set(robustness_score, labels)
        
        logger.debug(f"Recorded model performance for {model_name}")
    
    @asynccontextmanager
    async def measure_operation(self, operation_name: str):
        """Context manager to measure ML operation duration."""
        start_time = time.time()
        labels = {
            "operation": operation_name,
            "service": self.service_name
        }
        
        try:
            yield
            # Record successful operation
            if counter := self._instruments.get("ml_operations_total"):
                counter.add(1, {**labels, "status": "success"})
        except Exception as e:
            # Record failed operation
            if counter := self._instruments.get("ml_operations_total"):
                counter.add(1, {**labels, "status": "error", "error_type": type(e).__name__})
            raise
        finally:
            # Record operation duration
            duration = time.time() - start_time
            if histogram := self._instruments.get("ml_response_time"):
                histogram.record(duration, labels)


class MLAlertingSystem:
    """OpenTelemetry-native alerting system for ML monitoring."""
    
    def __init__(self, alert_thresholds: dict[str, float]):
        self.alert_thresholds = alert_thresholds
        self.active_alerts: list[OTelAlert] = []
        self.alert_history: list[OTelAlert] = []
        self._lock = Lock()
        
        self._initialize_default_alerts()
        logger.info(f"Initialized ML alerting system with {len(self.active_alerts)} alerts")
    
    def _initialize_default_alerts(self):
        """Initialize default ML alert definitions."""
        default_alerts = [
            OTelAlert(
                alert_name="HighFailureRate",
                metric_name="ml_failure_rate",
                threshold=self.alert_thresholds.get("failure_rate", 0.15),
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                description="ML system failure rate exceeds threshold"
            ),
            OTelAlert(
                alert_name="SlowResponseTime",
                metric_name="ml_response_time_seconds",
                threshold=self.alert_thresholds.get("response_time_ms", 200) / 1000.0,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="ML system response time exceeds threshold"
            ),
            OTelAlert(
                alert_name="HighAnomalyScore",
                metric_name="ml_anomaly_score",
                threshold=self.alert_thresholds.get("anomaly_score", 0.8),
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="Anomaly detection score is elevated"
            ),
            OTelAlert(
                alert_name="ModelDrift",
                metric_name="ml_drift_score",
                threshold=self.alert_thresholds.get("drift_score", 0.7),
                comparison="gt",
                severity=AlertSeverity.WARNING,
                description="Model drift detected"
            )
        ]
        
        self.active_alerts.extend(default_alerts)

    def check_alerts(self, failure_analysis: dict[str, Any]) -> list[OTelAlert]:
        """Check all alerts against current metrics and return triggered alerts."""
        triggered_alerts = []
        current_time = datetime.now()

        with self._lock:
            for alert in self.active_alerts:
                # Skip if in cooldown
                if alert.is_in_cooldown():
                    continue

                # Extract metric value
                metric_value = self._extract_metric_value(failure_analysis, alert.metric_name)
                if metric_value is None:
                    continue

                # Check threshold
                if self._check_threshold(metric_value, alert):
                    alert.triggered_at = current_time
                    alert.last_checked = current_time
                    triggered_alerts.append(alert)

                    # Add to history
                    self.alert_history.append(alert)

                    logger.warning(
                        f"Alert triggered: {alert.alert_name} "
                        f"(value={metric_value}, threshold={alert.threshold})"
                    )

        return triggered_alerts

    def _extract_metric_value(self, failure_analysis: dict[str, Any], metric_name: str) -> float | None:
        """Extract metric value from failure analysis results."""
        metadata = failure_analysis.get("metadata", {})

        if metric_name == "ml_failure_rate":
            return metadata.get("failure_rate", 0.0)
        elif metric_name == "ml_response_time_seconds":
            return metadata.get("avg_response_time", 0.1)
        elif metric_name == "ml_anomaly_score":
            anomaly_detection = failure_analysis.get("anomaly_detection", {})
            if "anomaly_summary" in anomaly_detection:
                return (
                    anomaly_detection["anomaly_summary"].get("consensus_anomaly_rate", 0) / 100.0
                )
            return 0.0
        elif metric_name == "ml_drift_score":
            return metadata.get("drift_score", 0.0)

        return None

    def _check_threshold(self, value: float, alert: OTelAlert) -> bool:
        """Check if metric value exceeds alert threshold."""
        if alert.comparison == "gt":
            return value > alert.threshold
        elif alert.comparison == "lt":
            return value < alert.threshold
        elif alert.comparison == "eq":
            return abs(value - alert.threshold) < 0.001
        return False

    def get_alert_recommendations(self, alert_name: str) -> list[str]:
        """Get recommended actions for specific alert."""
        recommendations = {
            "HighFailureRate": [
                "Investigate recent system changes",
                "Check for data quality issues",
                "Review model performance metrics",
                "Consider temporary rollback if needed",
            ],
            "SlowResponseTime": [
                "Check system resource utilization",
                "Review database performance",
                "Monitor network latency",
                "Consider scaling resources",
            ],
            "HighAnomalyScore": [
                "Investigate anomalous patterns",
                "Check for data drift",
                "Review recent input changes",
                "Validate model predictions",
            ],
            "ModelDrift": [
                "Retrain model with recent data",
                "Investigate data distribution changes",
                "Review feature engineering pipeline",
                "Consider model ensemble approach",
            ],
        }

        return recommendations.get(
            alert_name, ["Investigate the issue", "Check system logs"]
        )


class OTelHTTPServer:
    """OpenTelemetry HTTP server replacement for prometheus start_http_server."""

    def __init__(self, port: int = 8000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.server = None
        self._running = False

        # Initialize Prometheus exporter for compatibility
        if OTEL_AVAILABLE:
            self.prometheus_reader = PrometheusMetricReader()
            logger.info(f"OTel HTTP server initialized on {host}:{port}")
        else:
            logger.warning("OpenTelemetry not available, HTTP server disabled")

    async def start(self):
        """Start the HTTP server for metrics export."""
        if not OTEL_AVAILABLE:
            logger.warning("Cannot start HTTP server: OpenTelemetry not available")
            return

        try:
            from aiohttp import web, web_runner

            app = web.Application()
            app.router.add_get('/metrics', self._metrics_handler)
            app.router.add_get('/health', self._health_handler)
            app.router.add_get('/ready', self._ready_handler)

            runner = web_runner.AppRunner(app)
            await runner.setup()

            site = web_runner.TCPSite(runner, self.host, self.port)
            await site.start()

            self._running = True
            logger.info(f"OTel HTTP server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start OTel HTTP server: {e}")
            raise

    async def _metrics_handler(self, request):
        """Handle /metrics endpoint (Prometheus format)."""
        try:
            from aiohttp import web

            # Export metrics in Prometheus format
            metrics_data = self.prometheus_reader.get_metrics_data()

            # Convert to Prometheus text format
            prometheus_text = self._convert_to_prometheus_format(metrics_data)

            return web.Response(
                text=prometheus_text,
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )
        except Exception as e:
            logger.error(f"Error in metrics handler: {e}")
            return web.Response(text="Error retrieving metrics", status=500)

    async def _health_handler(self, request):
        """Handle /health endpoint."""
        from aiohttp import web
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})

    async def _ready_handler(self, request):
        """Handle /ready endpoint."""
        from aiohttp import web
        return web.json_response({"status": "ready", "server_running": self._running})

    def _convert_to_prometheus_format(self, metrics_data) -> str:
        """Convert OpenTelemetry metrics to Prometheus text format."""
        # Simplified conversion - in production, use proper OTel Prometheus exporter
        lines = []
        lines.append("# OpenTelemetry ML Metrics")
        lines.append(f"# Generated at {datetime.now().isoformat()}")

        # Add basic metrics info
        lines.append("otel_ml_server_info{version=\"1.0.0\"} 1")

        return "\n".join(lines)

    async def stop(self):
        """Stop the HTTP server."""
        if self.server:
            await self.server.stop()
            self._running = False
            logger.info("OTel HTTP server stopped")


# Factory functions for ML algorithms to use
def get_ml_metrics(service_name: str = "ml-monitoring") -> MLMetricsCollector:
    """Get ML metrics collector instance (replaces prometheus_client usage)."""
    return MLMetricsCollector(service_name)


def get_ml_alerting_metrics(alert_thresholds: dict[str, float]) -> MLAlertingSystem:
    """Get ML alerting system instance (replaces PrometheusAlert usage)."""
    return MLAlertingSystem(alert_thresholds)


def start_ml_http_server(port: int = 8000, host: str = "0.0.0.0") -> OTelHTTPServer:
    """Start OTel HTTP server (replaces prometheus start_http_server)."""
    return OTelHTTPServer(port, host)


# Backward compatibility aliases
PrometheusAlert = OTelAlert  # For existing code compatibility
