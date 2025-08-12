"""Training metrics for ML orchestration system.

This module provides metrics tracking for model training workflows
including accuracy, loss, training time, and convergence indicators.
"""
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TrainingMetrics(BaseModel):
    """Metrics for training workflow tracking."""
    
    workflow_id: str = Field(description="Unique workflow identifier")
    accuracy: float = Field(ge=0, le=1, description="Training accuracy")
    loss: float = Field(ge=0, description="Training loss")
    training_time: float = Field(ge=0, description="Training time in seconds")
    epochs_completed: int = Field(ge=0, description="Number of epochs completed")
    samples_processed: int = Field(ge=0, description="Number of samples processed")
    convergence_reached: bool | None = Field(None, description="Whether training converged")
    final_learning_rate: float | None = Field(None, gt=0, description="Final learning rate")
    dataset_size: int | None = Field(None, ge=0, description="Size of training dataset")
    batch_size: int | None = Field(None, ge=1, description="Training batch size")
    model_parameters: int | None = Field(None, ge=0, description="Number of model parameters")
    validation_accuracy: float | None = Field(None, ge=0, le=1, description="Validation accuracy")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics collection timestamp")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TrainingMetrics':
        """Create instance from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)