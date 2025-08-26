"""Application Service Protocols.

Defines the interfaces for application services that orchestrate business workflows
and coordinate between presentation and domain layers.
"""

from typing import Any, Protocol

from prompt_improver.core.domain.types import (
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    AssociationRuleFilterData,
    PatternDiscoveryFilterData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
)

# These are already imported above as domain DTOs - no database models needed


class ApplicationServiceProtocol(Protocol):
    """Base protocol for all application services."""

    async def initialize(self) -> None:
        """Initialize the application service."""
        ...

    async def cleanup(self) -> None:
        """Clean up application service resources."""
        ...


class PromptApplicationServiceProtocol(Protocol):
    """Protocol for prompt improvement application service."""

    async def improve_prompt(
        self,
        prompt: str,
        session_id: str,
        improvement_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Orchestrate the complete prompt improvement workflow."""
        ...

    async def apply_rules_to_prompt(
        self,
        prompt: str,
        rule_ids: list[str],
        session_id: str,
    ) -> dict[str, Any]:
        """Apply specific rules to a prompt with session tracking."""
        ...

    async def create_improvement_session(
        self,
        initial_prompt: str,
        user_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new prompt improvement session."""
        ...

    async def finalize_improvement_session(
        self,
        session_id: str,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize an improvement session with user feedback."""
        ...


class MLApplicationServiceProtocol(Protocol):
    """Protocol for ML training and inference application service."""

    async def execute_training_workflow(
        self,
        training_config: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a complete ML training workflow."""
        ...

    async def execute_pattern_discovery(
        self,
        request: PatternDiscoveryRequestData,
        session_id: str | None = None,
    ) -> PatternDiscoveryResponseData:
        """Execute comprehensive pattern discovery workflow."""
        ...

    async def execute_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        session_id: str | None = None,
    ) -> AprioriAnalysisResponseData:
        """Execute Apriori association rule mining workflow."""
        ...

    async def deploy_model(
        self,
        model_id: str,
        deployment_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Deploy a trained model to production."""
        ...

    async def execute_inference(
        self,
        model_id: str,
        input_data: dict[str, Any],
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute model inference with proper error handling."""
        ...


class AnalyticsApplicationServiceProtocol(Protocol):
    """Protocol for analytics and reporting application service."""

    async def generate_dashboard_data(
        self,
        time_range_hours: int = 24,
        include_comparisons: bool = True,
    ) -> dict[str, Any]:
        """Generate comprehensive dashboard data."""
        ...

    async def execute_trend_analysis(
        self,
        metric_type: str,
        time_range: dict[str, Any],
        granularity: str = "day",
        session_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute trend analysis workflow."""
        ...

    async def execute_session_comparison(
        self,
        session_a_id: str,
        session_b_id: str,
        comparison_dimension: str = "performance",
        method: str = "t_test",
    ) -> dict[str, Any]:
        """Execute session comparison analysis."""
        ...

    async def generate_session_summary(
        self,
        session_id: str,
        include_insights: bool = True,
    ) -> dict[str, Any]:
        """Generate comprehensive session summary."""
        ...

    async def export_session_report(
        self,
        session_id: str,
        export_format: str = "json",
        include_detailed_metrics: bool = True,
    ) -> dict[str, Any]:
        """Export session report in specified format."""
        ...


class TrainingApplicationServiceProtocol(Protocol):
    """Protocol for training workflow orchestration."""

    async def start_training_workflow(
        self,
        workflow_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Start a new training workflow."""
        ...

    async def monitor_training_progress(
        self,
        workflow_id: str,
    ) -> dict[str, Any]:
        """Monitor training workflow progress."""
        ...

    async def pause_training_workflow(
        self,
        workflow_id: str,
    ) -> dict[str, Any]:
        """Pause an active training workflow."""
        ...

    async def resume_training_workflow(
        self,
        workflow_id: str,
    ) -> dict[str, Any]:
        """Resume a paused training workflow."""
        ...

    async def stop_training_workflow(
        self,
        workflow_id: str,
        graceful: bool = True,
    ) -> dict[str, Any]:
        """Stop a training workflow with optional graceful shutdown."""
        ...


class HealthApplicationServiceProtocol(Protocol):
    """Protocol for health monitoring and system status."""

    async def perform_comprehensive_health_check(
        self,
        include_detailed_metrics: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive system health check."""
        ...

    async def monitor_system_performance(
        self,
        duration_seconds: int = 60,
    ) -> dict[str, Any]:
        """Monitor system performance over specified duration."""
        ...

    async def diagnose_system_issues(
        self,
        component_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        """Diagnose potential system issues."""
        ...

    async def execute_performance_benchmark(
        self,
        benchmark_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute performance benchmark workflow."""
        ...


class AprioriApplicationServiceProtocol(Protocol):
    """Protocol for Apriori analysis application service."""

    async def execute_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        session_id: str | None = None,
    ) -> AprioriAnalysisResponseData:
        """Execute complete Apriori analysis workflow."""
        ...

    async def get_association_rules(
        self,
        filters: AssociationRuleFilterData | None = None,
        sort_by: str = "lift",
        sort_desc: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve association rules with filtering and sorting."""
        ...

    async def get_contextualized_patterns(
        self,
        context_items: list[str],
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Get patterns relevant to specific context items."""
        ...


class PatternApplicationServiceProtocol(Protocol):
    """Protocol for pattern discovery application service."""

    async def execute_comprehensive_pattern_discovery(
        self,
        request: PatternDiscoveryRequestData,
        session_id: str | None = None,
    ) -> PatternDiscoveryResponseData:
        """Execute comprehensive pattern discovery workflow."""
        ...

    async def get_pattern_discoveries(
        self,
        filters: PatternDiscoveryFilterData | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve historical pattern discovery runs."""
        ...

    async def get_discovery_insights(
        self,
        discovery_run_id: str,
    ) -> dict[str, Any]:
        """Get detailed insights for a specific discovery run."""
        ...
