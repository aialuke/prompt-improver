"""Evaluation metrics for ML orchestration system.

This module provides metrics tracking for model evaluation workflows
including accuracy, precision, recall, and other performance indicators.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EvaluationType(Enum):
    """Types of model evaluation."""
    VALIDATION = 'validation'
    TEST = 'test'
    CROSS_VALIDATION = 'cross_validation'
    HOLDOUT = 'holdout'
    STATISTICAL_VALIDATION = 'statistical_validation'


class EvaluationMetrics(BaseModel):
    """Metrics for evaluation workflow tracking."""
    
    workflow_id: str = Field(description="Unique workflow identifier")
    evaluation_type: EvaluationType = Field(description="Type of evaluation performed")
    accuracy: float = Field(ge=0, le=1, description="Model accuracy")
    precision: float = Field(ge=0, le=1, description="Model precision")
    recall: float = Field(ge=0, le=1, description="Model recall")
    f1_score: float = Field(ge=0, le=1, description="F1 score")
    statistical_significance: bool = Field(description="Whether results are statistically significant")
    p_value: float = Field(ge=0, le=1, description="Statistical p-value")
    effect_size: float = Field(description="Effect size of the evaluation")
    evaluation_time: float | None = Field(None, ge=0, description="Evaluation time in seconds")
    dataset_size: int | None = Field(None, ge=0, description="Size of evaluation dataset")
    model_version: str | None = Field(None, description="Version of evaluated model")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics collection timestamp")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EvaluationMetrics':
        """Create instance from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('evaluation_type'), str):
            data['evaluation_type'] = EvaluationType(data['evaluation_type'])
        return cls(**data)