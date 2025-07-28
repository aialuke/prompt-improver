"""
Service Level Objective (SLO) and Service Level Agreement (SLA) Monitoring System

This module provides comprehensive SLO/SLA monitoring following Google SRE practices:
- Configurable SLO targets with multi-window calculations
- Error budget tracking and burn rate alerting  
- Customer-specific SLA monitoring
- Integration with OpenTelemetry and Prometheus
- Automated reporting and dashboards
"""

from .framework import (
    SLODefinition,
    SLOTarget,
    SLOTimeWindow,
    SLOType,
    ErrorBudget,
    BurnRate,
)

from .calculator import (
    SLICalculator,
    MultiWindowSLICalculator,
    PercentileCalculator,
    AvailabilityCalculator,
)

from .monitor import (
    SLOMonitor,
    SLAMonitor,
    ErrorBudgetMonitor,
    BurnRateAlert,
)

from .reporting import (
    SLOReporter,
    SLAReporter,
    DashboardGenerator,
    ExecutiveReporter,
)

from .integration import (
    OpenTelemetryIntegration,
    PrometheusRecordingRules,
    MetricsCollector,
)

__all__ = [
    # Framework
    "SLODefinition",
    "SLOTarget", 
    "SLOTimeWindow",
    "SLOType",
    "ErrorBudget",
    "BurnRate",
    
    # Calculators
    "SLICalculator",
    "MultiWindowSLICalculator", 
    "PercentileCalculator",
    "AvailabilityCalculator",
    
    # Monitors
    "SLOMonitor",
    "SLAMonitor", 
    "ErrorBudgetMonitor",
    "BurnRateAlert",
    
    # Reporting
    "SLOReporter",
    "SLAReporter",
    "DashboardGenerator", 
    "ExecutiveReporter",
    
    # Integration
    "OpenTelemetryIntegration",
    "PrometheusRecordingRules",
    "MetricsCollector",
]