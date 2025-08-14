"""Protocol interfaces for decomposed ML Pipeline Orchestrator services.

This module defines the protocol interfaces for the five focused services that 
replace the MLPipelineOrchestrator god object, following clean architecture 
and single responsibility principles.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .orchestrator_types import PipelineState, WorkflowInstance


@runtime_checkable
class WorkflowOrchestratorProtocol(Protocol):
    """Protocol for workflow orchestration operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the workflow orchestrator."""
        ...
    
    @abstractmethod
    async def start_workflow(self, workflow_type: str, parameters: dict[str, Any]) -> str:
        """Start a new ML workflow and return workflow ID."""
        ...
    
    @abstractmethod
    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a running workflow."""
        ...
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> WorkflowInstance:
        """Get the status of a specific workflow."""
        ...
    
    @abstractmethod
    async def list_workflows(self) -> list[WorkflowInstance]:
        """List all active workflows."""
        ...
    
    @abstractmethod
    async def run_training_workflow(self, training_data: Any) -> dict[str, Any]:
        """Execute a complete training workflow."""
        ...
    
    @abstractmethod
    async def run_evaluation_workflow(self, evaluation_data: Any) -> dict[str, Any]:
        """Execute a complete evaluation workflow."""
        ...


@runtime_checkable
class ComponentManagerProtocol(Protocol):
    """Protocol for component lifecycle management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component manager."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component manager."""
        ...
    
    @abstractmethod
    async def discover_components(self) -> None:
        """Discover and register all ML components."""
        ...
    
    @abstractmethod
    async def load_direct_components(self) -> None:
        """Load ML components directly for integration."""
        ...
    
    @abstractmethod
    async def get_component_health(self) -> dict[str, bool]:
        """Get health status of all registered components."""
        ...
    
    @abstractmethod
    async def invoke_component(self, component_name: str, method_name: str, *args, **kwargs) -> Any:
        """Invoke a method on a loaded ML component."""
        ...
    
    @abstractmethod
    def get_loaded_components(self) -> list[str]:
        """Get list of loaded component names."""
        ...
    
    @abstractmethod
    def get_component_methods(self, component_name: str) -> list[str]:
        """Get available methods for a component."""
        ...


@runtime_checkable
class SecurityIntegrationServiceProtocol(Protocol):
    """Protocol for security validation and integration."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the security service."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the security service."""
        ...
    
    @abstractmethod
    async def validate_input_secure(self, input_data: Any, context: dict[str, Any] = None) -> Any:
        """Validate input data using integrated security sanitizer."""
        ...
    
    @abstractmethod
    async def monitor_memory_usage(self, operation_name: str = "operation", component_name: str = None) -> Any:
        """Monitor memory usage for operations."""
        ...
    
    @abstractmethod
    def monitor_operation_memory(self, operation_name: str, component_name: str = None) -> Any:
        """Context manager for monitoring memory during operations."""
        ...
    
    @abstractmethod
    async def validate_operation_memory(self, data: Any, operation_name: str, component_name: str = None) -> bool:
        """Validate memory requirements for ML operations."""
        ...
    
    @abstractmethod
    async def run_training_workflow_secure(self, training_data: Any, context: dict[str, Any] = None) -> dict[str, Any]:
        """Run a complete training workflow with security validation."""
        ...


@runtime_checkable
class DeploymentPipelineServiceProtocol(Protocol):
    """Protocol for model deployment and release management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the deployment service."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the deployment service."""
        ...
    
    @abstractmethod
    async def run_native_deployment_workflow(self, model_id: str, deployment_config: dict[str, Any]) -> dict[str, Any]:
        """Run native ML model deployment workflow without Docker containers."""
        ...
    
    @abstractmethod
    async def run_complete_ml_pipeline(self, 
                                     training_data: Any,
                                     model_config: dict[str, Any],
                                     deployment_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run complete ML pipeline: training, evaluation, and native deployment."""
        ...


@runtime_checkable 
class MonitoringCoordinatorProtocol(Protocol):
    """Protocol for health monitoring and performance tracking."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the monitoring coordinator."""
        ...
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the monitoring coordinator."""
        ...
    
    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of orchestrator and all components."""
        ...
    
    @abstractmethod
    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        ...
    
    @abstractmethod
    async def setup_monitoring(self) -> None:
        """Setup monitoring for the orchestrator."""
        ...
    
    @abstractmethod
    def get_invocation_history(self, component_name: str | None = None) -> list[dict[str, Any]]:
        """Get component invocation history."""
        ...