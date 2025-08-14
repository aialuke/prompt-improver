"""
Shared Types for ML Pipeline Orchestrator

Contains shared data types and enums used across the orchestrator architecture
to avoid circular imports between the main orchestrator and facade.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class PipelineState(Enum):
    """Pipeline execution states."""
    idle = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class WorkflowInstance:
    """Represents a running workflow instance."""
    workflow_id: str
    workflow_type: str
    state: PipelineState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None