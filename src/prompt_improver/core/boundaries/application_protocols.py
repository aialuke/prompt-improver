"""Application Layer Boundary Protocols.

These protocols define the contracts for application services that orchestrate
business workflows and coordinate between presentation and domain layers.

Clean Architecture Rule: Application layer depends only on domain layer.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime

from prompt_improver.core.domain.types import (
    SessionId,
    UserId,
    ModelId,
    RuleId,
    AnalysisId,
    ImprovementSessionData,
    PromptSessionData,
    TrainingSessionData,
    UserFeedbackData,
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
    HealthCheckResultData,
    ModelMetricsData,
    TrainingResultData,
)
from prompt_improver.core.domain.enums import (
    SessionStatus,
    AnalysisStatus,
    ValidationLevel,
)


@runtime_checkable
class ApplicationServiceProtocol(Protocol):
    """Base protocol for all application services."""
    
    async def initialize(self) -> None:
        """Initialize the application service and its dependencies."""
        ...

    async def cleanup(self) -> None:
        """Clean up application service resources."""
        ...
    
    async def get_health_status(self) -> HealthCheckResultData:
        """Get health status of this application service.
        
        Returns:
            Health check result for this service
        """
        ...


@runtime_checkable
class WorkflowOrchestratorProtocol(Protocol):
    """Protocol for orchestrating complex business workflows."""
    
    async def start_workflow(
        self,
        workflow_type: str,
        workflow_data: Dict[str, Any],
        user_id: Optional[UserId] = None,
    ) -> SessionId:
        """Start a new workflow instance.
        
        Args:
            workflow_type: Type of workflow to start
            workflow_data: Initial workflow data
            user_id: Optional user identifier
            
        Returns:
            Session ID for the workflow instance
        """
        ...
    
    async def get_workflow_status(
        self,
        session_id: SessionId
    ) -> Dict[str, Any]:
        """Get current status of a workflow.
        
        Args:
            session_id: Workflow session identifier
            
        Returns:
            Workflow status and progress information
        """
        ...
    
    async def cancel_workflow(
        self,
        session_id: SessionId,
        reason: Optional[str] = None,
    ) -> bool:
        """Cancel a running workflow.
        
        Args:
            session_id: Workflow session identifier
            reason: Optional cancellation reason
            
        Returns:
            Whether cancellation was successful
        """
        ...
    
    async def resume_workflow(
        self,
        session_id: SessionId,
        checkpoint_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resume a paused or failed workflow.
        
        Args:
            session_id: Workflow session identifier
            checkpoint_data: Optional checkpoint data to resume from
            
        Returns:
            Whether resume was successful
        """
        ...


@runtime_checkable
class BusinessProcessProtocol(Protocol):
    """Protocol for executing business processes."""
    
    async def execute_process(
        self,
        process_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a business process.
        
        Args:
            process_name: Name of the process to execute
            input_data: Input data for the process
            context: Optional execution context
            
        Returns:
            Process execution results
        """
        ...
    
    async def validate_process_input(
        self,
        process_name: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate input data for a business process.
        
        Args:
            process_name: Name of the process
            input_data: Input data to validate
            
        Returns:
            Validation results
        """
        ...
    
    async def get_process_definition(
        self,
        process_name: str
    ) -> Dict[str, Any]:
        """Get definition of a business process.
        
        Args:
            process_name: Name of the process
            
        Returns:
            Process definition and metadata
        """
        ...


@runtime_checkable
class ApplicationEventProtocol(Protocol):
    """Protocol for application-level event handling."""
    
    async def publish_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Publish an application event.
        
        Args:
            event_type: Type of event
            event_data: Event payload
            correlation_id: Optional correlation identifier
        """
        ...
    
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: Any,  # Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> str:
        """Subscribe to application events.
        
        Args:
            event_types: List of event types to subscribe to
            callback: Callback function for events
            
        Returns:
            Subscription identifier
        """
        ...
    
    async def unsubscribe(
        self,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from events.
        
        Args:
            subscription_id: Subscription to cancel
            
        Returns:
            Whether unsubscription was successful
        """
        ...


# Domain-specific application service protocols

@runtime_checkable
class PromptApplicationServiceProtocol(ApplicationServiceProtocol, Protocol):
    """Protocol for prompt improvement application service."""
    
    async def improve_prompt(
        self,
        prompt: str,
        session_id: SessionId,
        rule_ids: Optional[List[RuleId]] = None,
        improvement_options: Optional[Dict[str, Any]] = None,
    ) -> ImprovementSessionData:
        """Orchestrate the complete prompt improvement workflow.
        
        Args:
            prompt: Original prompt text
            session_id: Session identifier
            rule_ids: Optional specific rules to apply
            improvement_options: Optional improvement configuration
            
        Returns:
            Improvement session data with results
        """
        ...

    async def apply_rules_to_prompt(
        self,
        prompt: str,
        rule_ids: List[RuleId],
        session_id: SessionId,
    ) -> Dict[str, Any]:
        """Apply specific rules to a prompt.
        
        Args:
            prompt: Prompt text
            rule_ids: Rules to apply
            session_id: Session identifier
            
        Returns:
            Rule application results
        """
        ...
    
    async def get_improvement_history(
        self,
        user_id: UserId,
        limit: Optional[int] = None,
    ) -> List[ImprovementSessionData]:
        """Get improvement history for a user.
        
        Args:
            user_id: User identifier
            limit: Optional limit on results
            
        Returns:
            List of improvement sessions
        """
        ...


@runtime_checkable
class MLApplicationServiceProtocol(ApplicationServiceProtocol, Protocol):
    """Protocol for ML training and inference application service."""
    
    async def start_training(
        self,
        model_id: ModelId,
        training_config: Dict[str, Any],
        user_id: Optional[UserId] = None,
    ) -> TrainingSessionData:
        """Start a new ML model training session.
        
        Args:
            model_id: Model identifier
            training_config: Training configuration
            user_id: Optional user identifier
            
        Returns:
            Training session data
        """
        ...
    
    async def get_training_status(
        self,
        session_id: SessionId
    ) -> TrainingSessionData:
        """Get status of a training session.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Current training session data
        """
        ...
    
    async def run_inference(
        self,
        model_id: ModelId,
        input_data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run inference with a trained model.
        
        Args:
            model_id: Model identifier
            input_data: Input data for inference
            options: Optional inference options
            
        Returns:
            Inference results
        """
        ...
    
    async def get_model_metrics(
        self,
        model_id: ModelId
    ) -> ModelMetricsData:
        """Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model performance metrics
        """
        ...


@runtime_checkable
class AnalyticsApplicationServiceProtocol(ApplicationServiceProtocol, Protocol):
    """Protocol for analytics and pattern discovery application service."""
    
    async def run_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        user_id: Optional[UserId] = None,
    ) -> AprioriAnalysisResponseData:
        """Run Apriori algorithm analysis.
        
        Args:
            request: Analysis request data
            user_id: Optional user identifier
            
        Returns:
            Analysis response with results
        """
        ...
    
    async def discover_patterns(
        self,
        request: PatternDiscoveryRequestData,
        user_id: Optional[UserId] = None,
    ) -> PatternDiscoveryResponseData:
        """Discover patterns in data.
        
        Args:
            request: Pattern discovery request
            user_id: Optional user identifier
            
        Returns:
            Pattern discovery results
        """
        ...
    
    async def get_analysis_history(
        self,
        user_id: UserId,
        analysis_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get analysis history for a user.
        
        Args:
            user_id: User identifier
            analysis_type: Optional filter by analysis type
            
        Returns:
            List of previous analyses
        """
        ...


@runtime_checkable
class HealthApplicationServiceProtocol(ApplicationServiceProtocol, Protocol):
    """Protocol for system health monitoring application service."""
    
    async def get_system_health(self) -> Dict[str, HealthCheckResultData]:
        """Get overall system health status.
        
        Returns:
            Health status of all system components
        """
        ...
    
    async def run_health_check(
        self,
        component_name: str
    ) -> HealthCheckResultData:
        """Run health check for a specific component.
        
        Args:
            component_name: Name of component to check
            
        Returns:
            Health check result for the component
        """
        ...
    
    async def get_health_history(
        self,
        component_name: str,
        time_range: Optional[tuple[datetime, datetime]] = None,
    ) -> List[HealthCheckResultData]:
        """Get health check history for a component.
        
        Args:
            component_name: Component name
            time_range: Optional time range filter
            
        Returns:
            Historical health check results
        """
        ...