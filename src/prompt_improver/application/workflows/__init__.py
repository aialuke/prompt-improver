"""Application Workflows.

This module contains workflow definitions that orchestrate complex business processes
by coordinating multiple domain services and repositories within transaction boundaries.
"""

from prompt_improver.application.workflows.analytics_workflows import (
    DashboardDataWorkflow,
    SessionComparisonWorkflow,
    TrendAnalysisWorkflow,
)
from prompt_improver.application.workflows.ml_workflows import (
    ModelDeploymentWorkflow,
    ModelInferenceWorkflow,
    PatternDiscoveryWorkflow,
    TrainingWorkflow,
)
from prompt_improver.application.workflows.prompt_workflows import (
    PromptImprovementWorkflow,
    RuleApplicationWorkflow,
    SessionManagementWorkflow,
)

__all__ = [
    "DashboardDataWorkflow",
    "ModelDeploymentWorkflow",
    "ModelInferenceWorkflow",
    "PatternDiscoveryWorkflow",
    "PromptImprovementWorkflow",
    "RuleApplicationWorkflow",
    "SessionComparisonWorkflow",
    "SessionManagementWorkflow",
    "TrainingWorkflow",
    "TrendAnalysisWorkflow",
]
