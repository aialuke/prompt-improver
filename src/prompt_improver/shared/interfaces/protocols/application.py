"""Application layer protocol definitions.

Consolidated protocols for application services that orchestrate business workflows
and coordinate between presentation, domain, and infrastructure layers.
"""

from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# TYPE_CHECKING imports - these are not imported at runtime, only for static analysis
if TYPE_CHECKING:
    from prompt_improver.core.domain.types import (
        AprioriAnalysisRequestData,
        AprioriAnalysisResponseData,
        AssociationRuleFilterData,
        PatternDiscoveryFilterData,
        PatternDiscoveryRequestData,
        PatternDiscoveryResponseData,
    )


# Base Application Service Protocols

@runtime_checkable
class ApplicationServiceProtocol(Protocol):
    """Base protocol for all application services."""

    async def initialize(self) -> None:
        """Initialize the application service."""
        ...

    async def cleanup(self) -> None:
        """Clean up application service resources."""
        ...


# Domain-Specific Application Service Protocols

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
        request: "PatternDiscoveryRequestData",
        session_id: str | None = None,
    ) -> "PatternDiscoveryResponseData":
        """Execute comprehensive pattern discovery workflow."""
        ...

    async def execute_apriori_analysis(
        self,
        request: "AprioriAnalysisRequestData",
        session_id: str | None = None,
    ) -> "AprioriAnalysisResponseData":
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
        request: "AprioriAnalysisRequestData",
        session_id: str | None = None,
    ) -> "AprioriAnalysisResponseData":
        """Execute complete Apriori analysis workflow."""
        ...

    async def get_association_rules(
        self,
        filters: "AssociationRuleFilterData | None" = None,
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
        request: "PatternDiscoveryRequestData",
        session_id: str | None = None,
    ) -> "PatternDiscoveryResponseData":
        """Execute comprehensive pattern discovery workflow."""
        ...

    async def get_pattern_discoveries(
        self,
        filters: "PatternDiscoveryFilterData | None" = None,
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


# Extended Application Protocols for Workflow Orchestration

@runtime_checkable
class WorkflowOrchestratorProtocol(Protocol):
    """Protocol for workflow orchestration services."""

    @abstractmethod
    async def orchestrate_workflow(self, workflow_type: str, context: dict[str, Any]) -> dict[str, Any]:
        """Orchestrate a business workflow."""
        ...

    @abstractmethod
    def get_workflow_definition(self, workflow_type: str) -> dict[str, Any]:
        """Get definition of a workflow type."""
        ...


@runtime_checkable
class RetryStrategyProtocol(Protocol):
    """Protocol for retry and resilience strategies."""

    @abstractmethod
    async def execute_with_retry(self, operation: Any, retry_config: dict[str, Any]) -> Any:
        """Execute operation with retry strategy."""
        ...

    @abstractmethod
    def get_retry_config(self, operation_type: str) -> dict[str, Any]:
        """Get retry configuration for operation type."""
        ...


# Prompt Service Protocols (migrated from core.protocols.prompt_service.prompt_protocols)

from uuid import UUID

# Import these inside TYPE_CHECKING to avoid circular imports during protocol consolidation
if TYPE_CHECKING:
    from prompt_improver.core.config.validation import ValidationResult
    from prompt_improver.core.domain.types import (
        ImprovementSessionData,
        UserFeedbackData,
    )
    from prompt_improver.rule_engine.base import BasePromptRule
else:
    # During runtime, we'll use Any for these types to avoid the circular import
    BasePromptRule = Any
    ImprovementSessionData = Any
    UserFeedbackData = Any
    ValidationResult = Any


@runtime_checkable
class PromptAnalysisServiceProtocol(Protocol):
    """Protocol for prompt analysis and improvement logic."""

    async def analyze_prompt(
        self,
        prompt_id: UUID,
        session_id: UUID | None = None,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analyze a prompt and generate improvement suggestions."""
        ...

    async def generate_improvements(
        self,
        prompt: str,
        rules: "list[BasePromptRule]",
        context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Generate improvements for a prompt using specified rules."""
        ...

    async def evaluate_improvement_quality(
        self,
        original_prompt: str,
        improved_prompt: str,
        metrics: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Evaluate the quality of an improvement."""
        ...

    async def get_ml_recommendations(
        self,
        prompt: str,
        session_id: UUID,
        model_version: str | None = None
    ) -> dict[str, Any]:
        """Get ML-based recommendations for prompt improvement."""
        ...


@runtime_checkable
class RuleApplicationServiceProtocol(Protocol):
    """Protocol for rule execution and validation."""

    async def apply_rules(
        self,
        prompt: str,
        rules: "list[BasePromptRule]",
        session_id: UUID | None = None,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Apply a set of rules to a prompt."""
        ...

    async def validate_rule_compatibility(
        self,
        rules: "list[BasePromptRule]",
        prompt_type: str | None = None
    ) -> dict[str, bool]:
        """Validate if rules are compatible with each other."""
        ...

    async def execute_rule_chain(
        self,
        prompt: str,
        rule_chain: "list[BasePromptRule]",
        stop_on_error: bool = False
    ) -> list[dict[str, Any]]:
        """Execute a chain of rules in sequence."""
        ...

    async def get_rule_performance_metrics(
        self,
        rule_id: str,
        time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """Get performance metrics for a specific rule."""
        ...


@runtime_checkable
class ValidationServiceProtocol(Protocol):
    """Protocol for input validation and business rule checking."""

    async def validate_prompt_input(
        self,
        prompt: str,
        constraints: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate prompt input against constraints."""
        ...

    async def check_business_rules(
        self,
        operation: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> bool:
        """Check if an operation complies with business rules."""
        ...

    async def validate_improvement_session(
        self,
        session: "ImprovementSessionData"
    ) -> dict[str, Any]:
        """Validate an improvement session."""
        ...

    async def sanitize_prompt_content(
        self,
        prompt: str,
        sanitization_level: str = "standard"
    ) -> str:
        """Sanitize prompt content for safety."""
        ...

    async def validate_startup_configuration(
        self,
        environment: str | None = None
    ) -> "ValidationResult":
        """Validate overall startup configuration integrity."""
        ...

    async def validate_database_configuration(
        self,
        test_connectivity: bool = True
    ) -> "ValidationResult":
        """Validate database configuration and connectivity."""
        ...

    async def validate_security_configuration(
        self,
        security_profile: str | None = None
    ) -> "ValidationResult":
        """Validate security configuration and settings."""
        ...

    async def validate_monitoring_configuration(
        self,
        include_connectivity_tests: bool = False
    ) -> "ValidationResult":
        """Validate monitoring and observability configuration."""
        ...

    # Legacy application-level validation methods
    @abstractmethod
    async def validate_request(self, request: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        """Validate request against schema."""
        ...

    @abstractmethod
    def get_validation_rules(self, context: str) -> list[dict[str, Any]]:
        """Get validation rules for a context."""
        ...


@runtime_checkable
class PromptServiceFacadeProtocol(Protocol):
    """Protocol for the unified PromptServiceFacade."""

    async def improve_prompt(
        self,
        prompt: str,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
        rules: "list[BasePromptRule] | None" = None,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Main method to improve a prompt."""
        ...

    async def get_session_summary(
        self,
        session_id: UUID
    ) -> dict[str, Any]:
        """Get summary of an improvement session."""
        ...

    async def process_feedback(
        self,
        feedback: "UserFeedbackData",
        session_id: UUID
    ) -> dict[str, Any]:
        """Process user feedback for a session."""
        ...

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all prompt services."""
        ...


# Extended Application Protocols (legacy)

@runtime_checkable
class WorkflowOrchestratorProtocol(Protocol):
    """Protocol for workflow orchestration services."""

    @abstractmethod
    async def orchestrate_workflow(self, workflow_type: str, context: dict[str, Any]) -> dict[str, Any]:
        """Orchestrate a business workflow."""
        ...

    @abstractmethod
    def get_workflow_definition(self, workflow_type: str) -> dict[str, Any]:
        """Get definition of a workflow type."""
        ...


@runtime_checkable
class RetryStrategyProtocol(Protocol):
    """Protocol for retry and resilience strategies."""

    @abstractmethod
    async def execute_with_retry(self, operation: Any, retry_config: dict[str, Any]) -> Any:
        """Execute operation with retry strategy."""
        ...

    @abstractmethod
    def get_retry_config(self, operation_type: str) -> dict[str, Any]:
        """Get retry configuration for operation type."""
        ...


# All application service protocols consolidated from /application/protocols/application_service_protocols.py
