"""
Central ML Pipeline Orchestrator (Decomposed Architecture)

Modern ML pipeline orchestrator using decomposed architecture with focused services.
Provides comprehensive ML workflow coordination through specialized components.

Architecture: 5 focused services replacing the original 1,043-line monolith:
1. WorkflowOrchestrator - Core workflow execution and pipeline coordination  
2. ComponentManager - Component loading, lifecycle management, and registry operations
3. SecurityIntegrationService - Security validation, input sanitization, and access control
4. DeploymentPipelineService - Model deployment, versioning, and release management
5. MonitoringCoordinator - Health monitoring, metrics collection, and performance tracking
"""

# Import shared types to avoid circular imports
from .orchestrator_types import PipelineState, WorkflowInstance

# Export the primary orchestrator facade
from .ml_pipeline_orchestrator_facade import MLPipelineOrchestrator

# Export the classes that other modules expect
__all__ = [
    "MLPipelineOrchestrator",
    "PipelineState", 
    "WorkflowInstance"
]