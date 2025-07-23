"""
Workflow type definitions for ML Pipeline orchestration.

Separated to avoid circular imports.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class WorkflowStepStatus(Enum):
    """Status of individual workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    name: str
    component_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Runtime state
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    workflow_type: str
    name: str
    description: str
    steps: List[WorkflowStep]
    global_timeout: Optional[int] = None
    parallel_execution: bool = False
    on_failure: str = "stop"  # "stop", "continue", "retry"