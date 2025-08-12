"""Deployment metrics for ML orchestration system.

This module provides metrics tracking for deployment workflows including
blue-green deployments, rolling updates, and canary releases.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field


class DeploymentStrategy(Enum):
    """Available deployment strategies."""
    BLUE_GREEN = 'blue_green'
    ROLLING_UPDATE = 'rolling_update'
    CANARY = 'canary'
    RECREATE = 'recreate'


class DeploymentMetrics(BaseModel):
    """Metrics for deployment workflow tracking."""
    
    workflow_id: str = Field(description="Unique workflow identifier")
    deployment_strategy: DeploymentStrategy = Field(description="Deployment strategy used")
    deployment_time: float = Field(ge=0, description="Total deployment time in seconds")
    success_rate: float = Field(ge=0, le=1, description="Deployment success rate")
    rollback_count: int = Field(ge=0, description="Number of rollbacks performed")
    health_check_duration: float = Field(ge=0, description="Health check duration in seconds")
    traffic_migration_time: float = Field(ge=0, description="Time to migrate traffic in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics collection timestamp")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DeploymentMetrics':
        """Create instance from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('deployment_strategy'), str):
            data['deployment_strategy'] = DeploymentStrategy(data['deployment_strategy'])
        return cls(**data)