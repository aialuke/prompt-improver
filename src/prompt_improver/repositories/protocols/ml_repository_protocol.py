"""ML repository protocol for machine learning operations and model management.

Defines the interface for ML-specific data access operations, including:
- Training session management
- Model performance tracking
- Synthetic data generation
- ML pipeline analytics
- Model versioning and deployment
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

# Domain models - no database coupling


class TrainingSessionFilter(BaseModel):
    """Filter criteria for training session queries."""

    status: str | None = None
    continuous_mode: bool | None = None
    min_performance: float | None = None
    max_performance: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    active_workflow_id: str | None = None


class ModelPerformanceFilter(BaseModel):
    """Filter criteria for model performance queries."""

    model_type: str | None = None
    min_accuracy: float | None = None
    min_precision: float | None = None
    min_recall: float | None = None
    min_training_samples: int | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class GenerationSessionFilter(BaseModel):
    """Filter criteria for generation session queries."""

    session_type: str | None = None
    generation_method: str | None = None
    status: str | None = None
    min_quality_threshold: float | None = None
    training_session_id: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class TrainingMetrics(BaseModel):
    """Training session metrics summary."""

    session_id: str
    total_iterations: int
    current_performance: float | None
    best_performance: float | None
    improvement_rate: float | None
    efficiency_score: float | None
    resource_utilization: dict[str, float] | None
    status_summary: dict[str, Any]


class ModelVersionInfo(BaseModel):
    """Model version information."""

    model_id: str
    version: str
    performance_metrics: dict[str, float]
    deployment_status: str
    created_at: datetime
    metadata: dict[str, Any] | None


class SyntheticDataMetrics(BaseModel):
    """Synthetic data generation metrics."""

    session_id: str
    total_samples: int
    avg_quality_score: float
    generation_efficiency: float
    method_performance: dict[str, dict[str, float]]
    quality_distribution: dict[str, int]


@runtime_checkable
class MLRepositoryProtocol(Protocol):
    """Protocol for ML operations data access."""

    # Training Session Management
    async def create_training_session(
        self, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new training session."""
        ...

    async def get_training_sessions(
        self,
        filters: TrainingSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve training sessions with filtering."""
        ...

    async def get_training_session_by_id(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get training session by ID."""
        ...

    async def update_training_session(
        self, session_id: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update training session."""
        ...

    async def get_active_training_sessions(self) -> list[dict[str, Any]]:
        """Get all currently active training sessions."""
        ...

    async def get_training_session_metrics(
        self, session_id: str
    ) -> TrainingMetrics | None:
        """Get comprehensive metrics for training session."""
        ...

    # Training Iteration Management
    async def create_training_iteration(
        self, iteration_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create training iteration record."""
        ...

    async def get_training_iterations(
        self,
        session_id: str,
        start_iteration: int | None = None,
        end_iteration: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get training iterations for session."""
        ...

    async def get_latest_iteration(self, session_id: str) -> dict[str, Any] | None:
        """Get latest iteration for training session."""
        ...

    async def get_iteration_performance_trend(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Get performance trend across iterations."""
        ...

    # Model Performance Management
    async def create_model_performance(
        self, performance_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Record model performance metrics."""
        ...

    async def get_model_performances(
        self,
        filters: ModelPerformanceFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get model performance records."""
        ...

    async def get_model_performance_by_id(
        self, model_id: str
    ) -> list[dict[str, Any]]:
        """Get performance history for specific model."""
        ...

    async def get_best_performing_models(
        self, metric: str = "accuracy", model_type: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get top performing models by metric."""
        ...

    # Training Data Management
    async def create_training_prompt(
        self, prompt_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create training prompt record."""
        ...

    async def get_training_prompts(
        self,
        data_source: str | None = None,
        is_active: bool = True,
        min_priority: int | None = None,
        session_id: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get training prompts with filters."""
        ...

    async def update_training_prompt(
        self, prompt_id: int, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update training prompt."""
        ...

    async def deactivate_training_prompts(self, prompt_ids: list[int]) -> int:
        """Deactivate training prompts, returns count updated."""
        ...

    # Synthetic Data Generation Management
    async def create_generation_session(
        self, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create synthetic data generation session."""
        ...

    async def get_generation_sessions(
        self,
        filters: GenerationSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get generation sessions with filters."""
        ...

    async def get_generation_session_by_id(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get generation session by ID."""
        ...

    async def update_generation_session(
        self, session_id: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update generation session."""
        ...

    # Generation Batch Management
    async def create_generation_batch(
        self, batch_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create generation batch record."""
        ...

    async def get_generation_batches(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get generation batches for session."""
        ...

    async def update_generation_batch(
        self, batch_id: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update generation batch."""
        ...

    # Synthetic Data Sample Management
    async def create_synthetic_data_samples(
        self, samples_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create multiple synthetic data samples."""
        ...

    async def get_synthetic_data_samples(
        self,
        session_id: str | None = None,
        batch_id: str | None = None,
        min_quality_score: float | None = None,
        domain_category: str | None = None,
        status: str = "active",
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get synthetic data samples with filters."""
        ...

    async def update_synthetic_data_sample(
        self, sample_id: str, update_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update synthetic data sample."""
        ...

    async def archive_synthetic_samples(self, sample_ids: list[str]) -> int:
        """Archive synthetic data samples, returns count updated."""
        ...

    # Quality Assessment Management
    async def create_quality_assessment(
        self, assessment_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create quality assessment record."""
        ...

    async def get_quality_assessments(
        self,
        session_id: str | None = None,
        batch_id: str | None = None,
        assessment_type: str | None = None,
        min_quality_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get quality assessments with filters."""
        ...

    # Analytics and Insights
    async def get_synthetic_data_metrics(
        self, session_id: str
    ) -> SyntheticDataMetrics | None:
        """Get comprehensive synthetic data metrics."""
        ...

    async def get_training_analytics(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Get training analytics for date range."""
        ...

    async def get_model_version_history(self, model_id: str) -> list[ModelVersionInfo]:
        """Get version history for model."""
        ...

    async def get_generation_method_performance(
        self,
        session_id: str | None = None,
        method_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get generation method performance data."""
        ...

    # Cleanup and Maintenance
    async def cleanup_old_iterations(
        self, session_id: str, keep_latest: int = 100
    ) -> int:
        """Clean up old iterations, keeping latest N."""
        ...

    async def cleanup_failed_sessions(self, days_old: int = 7) -> int:
        """Clean up failed training sessions older than specified days."""
        ...

    async def archive_completed_sessions(self, days_old: int = 30) -> int:
        """Archive completed sessions older than specified days."""
        ...
    
    # Intelligence Processing Methods (migrated from MLIntelligenceProcessor)
    
    async def get_prompt_characteristics_batch(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get batch of prompt characteristics for ML processing."""
        ...
    
    async def get_rule_performance_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule performance data for intelligence processing."""
        ...
    
    async def cache_rule_intelligence(
        self, intelligence_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule intelligence results with upsert logic."""
        ...
    
    async def get_rule_combinations_data(
        self, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Get rule combination data for analysis."""
        ...
    
    async def cache_combination_intelligence(
        self, combination_data: list[dict[str, Any]]
    ) -> None:
        """Cache rule combination intelligence results."""
        ...
    
    async def cache_pattern_discovery(
        self, pattern_data: dict[str, Any]
    ) -> None:
        """Cache pattern discovery results."""
        ...
    
    async def cleanup_expired_cache(self) -> dict[str, Any]:
        """Clean up expired intelligence cache entries."""
        ...
    
    async def check_rule_intelligence_freshness(
        self, rule_id: str
    ) -> bool:
        """Check if rule intelligence cache is fresh."""
        ...
    
    async def get_rule_historical_performance(
        self, rule_id: str
    ) -> list[dict[str, Any]]:
        """Get historical performance data for rule."""
        ...
    
    async def process_ml_predictions_batch(
        self, batch_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process ML predictions for batch of data."""
        ...
    
    async def update_rule_intelligence_incremental(
        self, rule_id: str, performance_data: dict[str, Any]
    ) -> None:
        """Update rule intelligence with incremental data."""
        ...
    
    async def get_intelligence_processing_stats(self) -> dict[str, Any]:
        """Get statistics for intelligence processing operations."""
        ...
