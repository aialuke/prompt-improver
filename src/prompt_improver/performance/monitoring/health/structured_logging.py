"""Enhanced Structured Logging for Health Monitoring
2025 Best Practices for Observability.
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)
try:
    from middleware.correlation_context import get_correlation_id
except (ImportError, ModuleNotFoundError):

    def get_correlation_id():
        return None


@dataclass
class LogContext:
    """Structured context for enhanced logging."""

    component: str
    operation: str
    health_check_type: str
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    environment: str = "production"

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredLogger:
    """Enhanced logger with structured output for 2025 observability standards."""

    def __init__(self, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)
        self._context_stack = []

    @contextmanager
    def context(self, **kwargs):
        """Add contextual information to all logs within this block."""
        self._context_stack.append(kwargs)
        try:
            yield
        finally:
            self._context_stack.pop()

    def _build_log_entry(self, level: str, message: str, **kwargs) -> dict[str, Any]:
        """Build structured log entry with all context."""
        correlation_id = get_correlation_id() or "unknown"
        log_entry = {
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "level": level,
            "message": message,
            "correlation_id": correlation_id,
        }
        for context in self._context_stack:
            log_entry.update(context)
        log_entry.update(kwargs)
        if "error" in kwargs and isinstance(kwargs["error"], Exception):
            error = kwargs["error"]
            log_entry["error_details"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        return log_entry

    def _make_json_safe(self, obj):
        """Convert non-JSON-serializable objects to strings recursively."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (Exception, BaseException, type)):
                    obj[key] = {"type": type(value).__name__, "message": str(value)}
                elif isinstance(value, dict):
                    self._make_json_safe(value)
                elif isinstance(value, list):
                    obj[key] = [
                        {"type": type(item).__name__, "message": str(item)}
                        if isinstance(item, (Exception, BaseException, type))
                        else item
                        for item in value
                    ]
        return obj

    def info(self, message: str, **kwargs):
        """Log info with structured data."""
        log_entry = self._build_log_entry("INFO", message, **kwargs)
        self.logger.info(json.dumps(log_entry))

    def warning(self, message: str, **kwargs):
        """Log warning with structured data."""
        log_entry = self._build_log_entry("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(log_entry))

    def error(self, message: str, **kwargs):
        """Log error with structured data."""
        log_entry = self._build_log_entry("ERROR", message, **kwargs)
        self._make_json_safe(log_entry)
        self.logger.error(json.dumps(log_entry))

    def debug(self, message: str, **kwargs):
        """Log debug with structured data."""
        log_entry = self._build_log_entry("DEBUG", message, **kwargs)
        self.logger.debug(json.dumps(log_entry))


def log_health_check(component_name: str, check_type: str = "health_check"):
    """Decorator for health check methods with comprehensive logging."""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            logger = StructuredLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
            start_time = time.time()
            logger.info(f"Health check started: {component_name}")
            try:
                result = await func(self, *args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Health check completed: {component_name}")
                logger.info(
                    "Health check result",
                    extra={
                        "component": component_name,
                        "status": result.status.name
                        if hasattr(result, "status")
                        else "unknown",
                        "healthy": (result.status.name == "HEALTHY")
                        if hasattr(result, "status")
                        else None,
                        "response_time_ms": getattr(result, "response_time_ms", None),
                        "details": getattr(result, "details", {}),
                        "duration_ms": duration_ms,
                    },
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(f"Health check failed: {component_name}")
                raise

        return wrapper

    return decorator


class HealthMetricsLogger:
    """Specialized logger for health metrics with aggregation support."""

    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        self.logger = StructuredLogger(f"health_metrics.{component_name}")
        self._metrics_buffer = []
        self._last_flush = time.time()
        self._flush_interval = 60

    def record_check(
        self,
        status: str,
        response_time_ms: float,
        details: dict[str, Any] | None = None,
    ):
        """Record a health check result."""
        metric = {
            "timestamp": time.time(),
            "component": self.component_name,
            "status": status,
            "response_time_ms": response_time_ms,
            "details": details or {},
        }
        self._metrics_buffer.append(metric)
        if time.time() - self._last_flush > self._flush_interval:
            self.flush()

    def flush(self):
        """Flush aggregated metrics."""
        if not self._metrics_buffer:
            return
        response_times = [m["response_time_ms"] for m in self._metrics_buffer]
        status_counts = {}
        for metric in self._metrics_buffer:
            status = metric["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        self.logger.info(
            f"Health metrics summary for {self.component_name}",
            extra={
                "avg_response_time_ms": sum(response_times) / len(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "p95_response_time_ms": sorted(response_times)[
                    max(0, int(len(response_times) * 0.95) - 1)
                ],
                "status_counts": status_counts,
            },
        )
        self._metrics_buffer = []
        self._last_flush = time.time()


health_metrics_loggers: dict[str, HealthMetricsLogger] = {}


def get_metrics_logger(component_name: str) -> HealthMetricsLogger:
    """Get or create a metrics logger for a component."""
    if component_name not in health_metrics_loggers:
        health_metrics_loggers[component_name] = HealthMetricsLogger(component_name)
    return health_metrics_loggers[component_name]
