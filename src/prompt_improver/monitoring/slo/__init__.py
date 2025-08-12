"""Service Level Objective (SLO) and Service Level Agreement (SLA) Monitoring System

This module provides comprehensive SLO/SLA monitoring following Google SRE practices:
- Configurable SLO targets with multi-window calculations
- Error budget tracking and burn rate alerting
- Customer-specific SLA monitoring
- Integration with OpenTelemetry
- Automated reporting and dashboards
"""

from prompt_improver.monitoring.opentelemetry.integration import (
    MetricsIntegration as OpenTelemetryIntegration,
)
from prompt_improver.monitoring.slo.calculator import (
    AvailabilityCalculator,
    MultiWindowSLICalculator,
    PercentileCalculator,
    SLICalculator,
)
from prompt_improver.monitoring.slo.framework import (
    BurnRate,
    ErrorBudget,
    SLODefinition,
    SLOTarget,
    SLOTimeWindow,
    SLOType,
)
from prompt_improver.monitoring.slo.monitor import (
    BurnRateAlert,
    ErrorBudgetMonitor,
    SLAMonitor,
    SLOMonitor,
)
from prompt_improver.monitoring.slo.reporting import (
    DashboardGenerator,
    ExecutiveReporter,
    SLAReporter,
    SLOReporter,
)

__all__ = [
    "AvailabilityCalculator",
    "BurnRate",
    "BurnRateAlert",
    "DashboardGenerator",
    "ErrorBudget",
    "ErrorBudgetMonitor",
    "ExecutiveReporter",
    "MetricsCollector",
    "MultiWindowSLICalculator",
    "OpenTelemetryIntegration",
    "PercentileCalculator",
    "SLAMonitor",
    "SLAReporter",
    "SLICalculator",
    "SLODefinition",
    "SLOMonitor",
    "SLOReporter",
    "SLOTarget",
    "SLOTimeWindow",
    "SLOType",
]
