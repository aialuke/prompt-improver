"""
Workflow type definitions for ML Pipeline orchestration.

Separated to avoid circular imports.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from sqlmodel import SQLModel, Field
from pydantic import BaseModel

class WorkflowStepStatus(Enum):
    """Status of individual workflow steps."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    RETRYING = 'retrying'

class WorkflowStep(BaseModel):
    """Individual step in a workflow."""
    step_id: str = Field(description='Unique step identifier')
    name: str = Field(description='Human-readable step name')
    component_name: str = Field(description='Component to execute')
    parameters: dict[str, Any] = Field(default_factory=dict, description='Step parameters')
    dependencies: list[str] = Field(default_factory=list, description='Step dependencies')
    timeout: int | None = Field(default=None, ge=1, description='Step timeout in seconds')
    retry_count: int = Field(default=0, ge=0, description='Current retry count')
    max_retries: int = Field(default=3, ge=0, description='Maximum retry attempts')
    status: WorkflowStepStatus = Field(default=WorkflowStepStatus.PENDING, description='Current step status')
    started_at: datetime | None = Field(default=None, description='Step start timestamp')
    completed_at: datetime | None = Field(default=None, description='Step completion timestamp')
    error_message: str | None = Field(default=None, description='Error message if step failed')
    result: Any | None = Field(default=None, description='Step execution result')

class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow."""
    workflow_type: str = Field(description='Type of workflow')
    name: str = Field(description='Workflow name')
    description: str = Field(description='Workflow description')
    steps: list[WorkflowStep] = Field(default_factory=list, description='Workflow steps')
    global_timeout: int | None = Field(default=None, ge=1, description='Global workflow timeout in seconds')
    parallel_execution: bool = Field(default=False, description='Whether to execute steps in parallel')
    on_failure: str = Field(default='stop', description='Action on failure: stop, continue, or retry')
    max_iterations: int | None = Field(default=None, ge=1, description='Maximum iterations for continuous training')
    continuous: bool = Field(default=False, description='Whether workflow is continuous')
    retry_policy: dict[str, Any] | None = Field(default=None, description='Retry policy configuration')
    metadata: dict[str, Any] | None = Field(default=None, description='Additional workflow metadata')
