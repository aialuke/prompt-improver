"""Performance Regression Prevention Framework.
==========================================

Comprehensive monitoring and prevention system for architectural violations
that could reintroduce performance degradation.

Key Components:
- Protocol compliance monitoring (TYPE_CHECKING violations)
- Startup performance tracking (import time, dependency contamination)
- Automated regression detection and prevention
- CI/CD integration for blocking violations
- Real-time monitoring with VS Code diagnostics integration

Protects against:
- Direct imports in protocol files (134-1007ms startup penalty)
- Missing TYPE_CHECKING guards (dependency contamination)
- Heavy dependency loading during import (NumPy/torch contamination)
- Circular import patterns
- God object violations (>500 lines)
"""

from prompt_improver.monitoring.regression.architectural_compliance import (
    ArchitecturalComplianceMonitor,
)
from prompt_improver.monitoring.regression.ci_integration import CIIntegration
from prompt_improver.monitoring.regression.diagnostics_integration import (
    VSCodeDiagnosticsMonitor,
)
from prompt_improver.monitoring.regression.regression_prevention import (
    RegressionPreventionFramework,
)
from prompt_improver.monitoring.regression.startup_performance import (
    StartupPerformanceTracker,
)

__all__ = [
    "ArchitecturalComplianceMonitor",
    "CIIntegration",
    "RegressionPreventionFramework",
    "StartupPerformanceTracker",
    "VSCodeDiagnosticsMonitor"
]
