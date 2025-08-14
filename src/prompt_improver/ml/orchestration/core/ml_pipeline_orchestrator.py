"""
Central ML Pipeline Orchestrator (Decomposed Architecture)

This module provides backward compatibility for the MLPipelineOrchestrator
while internally using the new decomposed architecture with focused services.

The original 1,043-line god object has been decomposed into 5 focused services:
1. WorkflowOrchestrator - Core workflow execution and pipeline coordination  
2. ComponentManager - Component loading, lifecycle management, and registry operations
3. SecurityIntegrationService - Security validation, input sanitization, and access control
4. DeploymentPipelineService - Model deployment, versioning, and release management
5. MonitoringCoordinator - Health monitoring, metrics collection, and performance tracking
"""

# Import shared types to avoid circular imports
from .orchestrator_types import PipelineState, WorkflowInstance

# Re-export the facade as the main orchestrator for backward compatibility  
from .ml_pipeline_orchestrator_facade import MLPipelineOrchestrator

# Export the classes that other modules expect
__all__ = [
    "MLPipelineOrchestrator",
    "PipelineState", 
    "WorkflowInstance"
]