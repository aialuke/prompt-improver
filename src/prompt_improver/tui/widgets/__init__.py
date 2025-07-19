"""Dashboard widgets for APES TUI interface."""

from .ab_testing import ABTestingWidget
from .automl_status import AutoMLStatusWidget
from .performance_metrics import PerformanceMetricsWidget
from .service_control import ServiceControlWidget
from .system_overview import SystemOverviewWidget

__all__ = [
    "ABTestingWidget",
    "AutoMLStatusWidget",
    "PerformanceMetricsWidget",
    "ServiceControlWidget",
    "SystemOverviewWidget",
]
