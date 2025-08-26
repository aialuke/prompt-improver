"""OpenTelemetry Integration with Existing Monitoring Infrastructure.
================================================================

Provides integration adapters to connect OpenTelemetry with existing
OpenTelemetry metrics, health checks, and monitoring systems.
"""

import logging

from prompt_improver.monitoring.opentelemetry.metrics import (
    get_business_metrics,
    get_database_metrics,
    get_http_metrics,
)
from prompt_improver.monitoring.opentelemetry.setup import get_meter, get_tracer

try:
    from opentelemetry import (
        metrics as otel_metrics,
        trace,
    )
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = otel_metrics = Status = StatusCode = None
logger = logging.getLogger(__name__)


class MetricsIntegration:
    """Integrates OpenTelemetry metrics with existing monitoring systems."""

    def __init__(self) -> None:
        """Initialize metrics integration with OpenTelemetry collectors."""
        self.http_metrics = get_http_metrics()
        self.database_metrics = get_database_metrics()
        self.business_metrics = get_business_metrics()
        self.meter = get_meter(__name__)
        self.tracer = get_tracer(__name__)
        logger.info("OpenTelemetry metrics integration initialized")
