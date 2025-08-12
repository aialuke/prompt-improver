"""Persistence repository protocol for data persistence operations.

Defines clean interfaces for core persistence operations without coupling to
database implementation details. Uses domain models instead of database models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel


class SessionData(BaseModel):
    """Domain model for improvement session data."""
    
    session_id: str
    original_prompt: str
    final_prompt: str
    rules_applied: list[dict[str, Any]]
    user_context: dict[str, Any] | None = None
    performance_metrics: dict[str, float] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RulePerformanceData(BaseModel):
    """Domain model for rule performance tracking."""
    
    rule_id: str
    session_id: str
    improvement_score: float
    confidence_level: float
    execution_time_ms: int
    prompt_type: str | None = None
    prompt_category: str | None = None
    context_data: dict[str, Any] | None = None
    created_at: datetime | None = None


class FeedbackData(BaseModel):
    """Domain model for user feedback."""
    
    session_id: str
    user_id: str | None = None
    rating: int  # 1-5 scale
    feedback_text: str | None = None
    improvement_areas: list[str] | None = None
    is_processed: bool = False
    model_id: str | None = None
    created_at: datetime | None = None


class ModelPerformanceData(BaseModel):
    """Domain model for ML model performance metrics."""
    
    model_id: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    evaluation_data_size: int
    hyperparameters: dict[str, Any] | None = None
    evaluation_date: datetime | None = None
    created_at: datetime | None = None


class ExperimentData(BaseModel):
    """Domain model for A/B experiment data."""
    
    experiment_id: str
    experiment_name: str
    description: str | None = None
    experiment_type: str
    status: str = "active"  # active, paused, completed, cancelled
    configuration: dict[str, Any]
    metrics: dict[str, Any] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    created_at: datetime | None = None


@runtime_checkable
class PersistenceRepositoryProtocol(Protocol):
    """Protocol for core persistence operations without database coupling."""

    # Session Management
    async def store_session(self, session_data: SessionData) -> bool:
        """Store improvement session data."""
        ...

    async def get_session(self, session_id: str) -> SessionData | None:
        """Retrieve session by ID."""
        ...

    async def get_recent_sessions(
        self, 
        limit: int = 100,
        user_context_filter: dict[str, Any] | None = None
    ) -> list[SessionData]:
        """Get recent sessions with optional filtering."""
        ...

    async def update_session(
        self, 
        session_id: str, 
        update_data: dict[str, Any]
    ) -> bool:
        """Update session data."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete session and related data."""
        ...

    # Rule Performance Tracking
    async def store_rule_performance(
        self, 
        performance_data: RulePerformanceData
    ) -> bool:
        """Store rule performance metrics."""
        ...

    async def get_rule_performance_history(
        self,
        rule_id: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 1000
    ) -> list[RulePerformanceData]:
        """Get performance history for a rule."""
        ...

    async def get_performance_by_session(
        self, 
        session_id: str
    ) -> list[RulePerformanceData]:
        """Get all rule performances for a session."""
        ...

    # Feedback Management
    async def store_feedback(self, feedback_data: FeedbackData) -> bool:
        """Store user feedback."""
        ...

    async def get_feedback_by_session(
        self, 
        session_id: str
    ) -> list[FeedbackData]:
        """Get feedback for a session."""
        ...

    async def get_recent_feedback(
        self,
        limit: int = 100,
        processed_only: bool = False
    ) -> list[FeedbackData]:
        """Get recent feedback with optional filtering."""
        ...

    async def mark_feedback_processed(
        self, 
        feedback_ids: list[str]
    ) -> int:
        """Mark feedback as processed, return count updated."""
        ...

    # Model Performance Tracking  
    async def store_model_performance(
        self, 
        performance_data: ModelPerformanceData
    ) -> bool:
        """Store ML model performance metrics."""
        ...

    async def get_model_performance_history(
        self,
        model_id: str,
        limit: int = 100
    ) -> list[ModelPerformanceData]:
        """Get performance history for a model."""
        ...

    async def get_latest_model_performance(
        self, 
        model_type: str
    ) -> ModelPerformanceData | None:
        """Get latest performance for model type."""
        ...

    # A/B Experiment Management
    async def store_experiment(self, experiment_data: ExperimentData) -> bool:
        """Store A/B experiment configuration."""
        ...

    async def get_experiment(self, experiment_id: str) -> ExperimentData | None:
        """Get experiment by ID."""
        ...

    async def get_active_experiments(self) -> list[ExperimentData]:
        """Get all active experiments."""
        ...

    async def update_experiment_metrics(
        self,
        experiment_id: str,
        metrics: dict[str, Any]
    ) -> bool:
        """Update experiment metrics."""
        ...

    async def complete_experiment(
        self, 
        experiment_id: str,
        final_metrics: dict[str, Any]
    ) -> bool:
        """Mark experiment as completed with final metrics."""
        ...

    # Analytics and Reporting
    async def get_session_analytics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, Any]:
        """Get session analytics for date range."""
        ...

    async def get_rule_effectiveness_summary(
        self,
        rule_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, dict[str, float]]:
        """Get rule effectiveness summary."""
        ...

    async def get_user_satisfaction_metrics(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None
    ) -> dict[str, Any]:
        """Get user satisfaction metrics."""
        ...

    # Cleanup and Maintenance
    async def cleanup_old_sessions(self, days_old: int = 90) -> int:
        """Clean up old sessions, return count deleted."""
        ...

    async def cleanup_old_performance_data(self, days_old: int = 180) -> int:
        """Clean up old performance data, return count deleted."""
        ...

    async def get_storage_metrics(self) -> dict[str, int]:
        """Get storage usage metrics."""
        ...