"""OpenTelemetry ML Monitoring Utilities
====================================

Utility functions and decorators for ML monitoring integration.
Provides easy-to-use patterns for ML algorithm monitoring.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from prompt_improver.monitoring.opentelemetry.metrics import MLMetrics, get_ml_metrics
from prompt_improver.monitoring.opentelemetry.setup import get_tracer

logger = logging.getLogger(__name__)


class MLMonitoringMixin:
    """Mixin class to add ML monitoring capabilities to any class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ml_metrics = None
        self._ml_alerting = None
        self._monitoring_enabled = True

    def enable_ml_monitoring(
        self,
        service_name: str = "ml-service",
        alert_thresholds: dict[str, float] | None = None,
    ):
        """Enable ML monitoring for this instance."""
        try:
            self._ml_metrics = get_ml_metrics(service_name)
            self._ml_alerting = None
            self._monitoring_enabled = True
            logger.info(f"ML monitoring enabled for {service_name}")
        except Exception as e:
            logger.error(f"Failed to enable ML monitoring: {e}")
            self._monitoring_enabled = False

    def record_ml_operation(
        self,
        operation_name: str,
        duration: float | None = None,
        success: bool = True,
        **metrics,
    ):
        """Record ML operation metrics."""
        if not self._monitoring_enabled or not self._ml_metrics:
            return
        try:
            if hasattr(self._ml_metrics, "record_inference") and duration is not None:
                self._ml_metrics.record_inference(operation_name, duration, success)
            for metric_name, value in metrics.items():
                if hasattr(self._ml_metrics, f"record_{metric_name}"):
                    getattr(self._ml_metrics, f"record_{metric_name}")(value)
        except Exception as e:
            logger.error(f"Failed to record ML operation: {e}")

    def check_ml_alerts(self, analysis_data: dict[str, Any]) -> list[Any]:
        """Check ML alerts and return triggered alerts."""
        if not self._monitoring_enabled or not self._ml_alerting:
            return []
        try:
            return self._ml_alerting.check_alerts(analysis_data)
        except Exception as e:
            logger.error(f"Failed to check ML alerts: {e}")
            return []


def ml_monitor(
    operation_name: str | None = None,
    record_duration: bool = True,
    record_errors: bool = True,
    service_name: str = "ml-operation",
):
    """Decorator to automatically monitor ML operations."""

    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics_collector = get_ml_metrics(service_name)
            start_time = time.time() if record_duration else None
            try:
                result = func(*args, **kwargs)
                if hasattr(metrics_collector, "_instruments"):
                    if counter := metrics_collector._instruments.get(
                        "ml_operations_total"
                    ):
                        counter.add(
                            1,
                            {
                                "operation": op_name,
                                "status": "success",
                                "service": service_name,
                            },
                        )
                if record_duration and start_time:
                    duration = time.time() - start_time
                    if histogram := metrics_collector._instruments.get(
                        "ml_response_time"
                    ):
                        histogram.record(
                            duration, {"operation": op_name, "service": service_name}
                        )
                return result
            except Exception as e:
                if record_errors and hasattr(metrics_collector, "_instruments"):
                    if counter := metrics_collector._instruments.get(
                        "ml_operations_total"
                    ):
                        counter.add(
                            1,
                            {
                                "operation": op_name,
                                "status": "error",
                                "error_type": type(e).__name__,
                                "service": service_name,
                            },
                        )
                logger.error(f"ML operation {op_name} failed: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics_collector = get_ml_metrics(service_name)
            start_time = time.time() if record_duration else None
            try:
                result = await func(*args, **kwargs)
                if hasattr(metrics_collector, "_instruments"):
                    if counter := metrics_collector._instruments.get(
                        "ml_operations_total"
                    ):
                        counter.add(
                            1,
                            {
                                "operation": op_name,
                                "status": "success",
                                "service": service_name,
                            },
                        )
                if record_duration and start_time:
                    duration = time.time() - start_time
                    if histogram := metrics_collector._instruments.get(
                        "ml_response_time"
                    ):
                        histogram.record(
                            duration, {"operation": op_name, "service": service_name}
                        )
                return result
            except Exception as e:
                if record_errors and hasattr(metrics_collector, "_instruments"):
                    if counter := metrics_collector._instruments.get(
                        "ml_operations_total"
                    ):
                        counter.add(
                            1,
                            {
                                "operation": op_name,
                                "status": "error",
                                "error_type": type(e).__name__,
                                "service": service_name,
                            },
                        )
                logger.error(f"ML operation {op_name} failed: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def create_ml_monitoring_context(
    service_name: str,
    alert_thresholds: dict[str, float] | None = None,
    enable_http_server: bool = False,
    http_port: int = 8000,
) -> dict[str, Any]:
    """Create a complete ML monitoring context."""
    context = {
        "metrics": get_ml_metrics(service_name),
        "alerting": None,
        "http_server": None,
        "service_name": service_name,
    }
    if alert_thresholds:
        context["alerting"] = None
    if enable_http_server:
        context["http_server"] = None
    logger.info(f"Created ML monitoring context for {service_name}")
    return context


class MLPerformanceTracker:
    """Track ML model performance over time."""

    def __init__(self, model_name: str, service_name: str = "ml-performance"):
        self.model_name = model_name
        self.service_name = service_name
        self.metrics = get_ml_metrics(service_name)
        self._performance_history = []

    def record_prediction_batch(
        self,
        batch_size: int,
        accuracy: float | None = None,
        latency: float | None = None,
        confidence: float | None = None,
    ):
        """Record metrics for a batch of predictions."""
        labels = {"model_name": self.model_name, "service": self.service_name}
        if hasattr(self.metrics, "_instruments"):
            if counter := self.metrics._instruments.get("ml_operations_total"):
                counter.add(batch_size, {**labels, "operation": "prediction"})
        self.metrics.record_model_performance(
            accuracy=accuracy, model_name=self.model_name
        )
        if latency and hasattr(self.metrics, "_instruments"):
            if histogram := self.metrics._instruments.get("ml_response_time"):
                histogram.record(latency, labels)
        self._performance_history.append({
            "timestamp": datetime.now(),
            "batch_size": batch_size,
            "accuracy": accuracy,
            "latency": latency,
            "confidence": confidence,
        })
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary statistics."""
        if not self._performance_history:
            return {}
        recent_entries = self._performance_history[-100:]
        accuracies = [
            e["accuracy"] for e in recent_entries if e["accuracy"] is not None
        ]
        latencies = [e["latency"] for e in recent_entries if e["latency"] is not None]
        summary = {
            "model_name": self.model_name,
            "total_predictions": sum(e["batch_size"] for e in recent_entries),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "avg_latency": sum(latencies) / len(latencies) if latencies else None,
            "recent_entries": len(recent_entries),
        }
        return summary


def monitor_failure_analysis(func: Callable) -> Callable:
    """Decorator specifically for failure analysis functions."""
    return ml_monitor(
        operation_name="failure_analysis",
        record_duration=True,
        record_errors=True,
        service_name="failure-analyzer",
    )(func)


def monitor_classification(func: Callable) -> Callable:
    """Decorator specifically for classification functions."""
    return ml_monitor(
        operation_name="classification",
        record_duration=True,
        record_errors=True,
        service_name="failure-classifier",
    )(func)
