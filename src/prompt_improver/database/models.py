"""SQLModel data models for APES following 2025 best practices
Combines SQLAlchemy 2.0 async with Pydantic validation
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Column, Index
from sqlalchemy.dialects.postgresql import (
    JSONB,
    UUID as PGUUID,
)
from sqlmodel import Field, SQLModel

# ===================================
# Base Models and Mixins
# ===================================


class TimestampMixin(SQLModel):
    """Mixin for automatic timestamp tracking"""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default=None)


class UUIDMixin(SQLModel):
    """Mixin for UUID primary keys"""

    id: UUID = Field(default_factory=uuid4, sa_column=Column(PGUUID, primary_key=True))


# ===================================
# Rule Performance Tracking Models
# ===================================


class RulePerformanceBase(SQLModel):
    """Base model for rule performance tracking"""

    rule_id: str = Field(max_length=50, index=True)
    rule_name: str = Field(max_length=100)
    prompt_id: UUID = Field(default_factory=uuid4, sa_column=Column(PGUUID, index=True))
    prompt_type: str | None = Field(default=None, max_length=50, index=True)
    prompt_category: str | None = Field(default=None, max_length=50)
    improvement_score: float = Field(ge=0, le=1, index=True)
    confidence_level: float = Field(ge=0, le=1)
    execution_time_ms: int | None = Field(default=None)
    rule_parameters: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    prompt_characteristics: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    before_metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    after_metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))


class RulePerformance(RulePerformanceBase, TimestampMixin, table=True):
    """Rule performance tracking with full audit trail"""

    __tablename__ = "rule_performance"
    id: int | None = Field(default=None, primary_key=True)

    __table_args__ = (
        CheckConstraint("improvement_score >= 0 AND improvement_score <= 1"),
        CheckConstraint("confidence_level >= 0 AND confidence_level <= 1"),
        Index("idx_rule_perf_score_type", "improvement_score", "prompt_type"),
        Index(
            "idx_rule_perf_characteristics",
            "prompt_characteristics",
            postgresql_using="gin",
        ),
    )


class RulePerformanceCreate(RulePerformanceBase):
    """Model for creating rule performance records"""


class RulePerformanceRead(RulePerformanceBase, TimestampMixin):
    """Model for reading rule performance records"""

    id: int


# ===================================
# Rule Combination Models
# ===================================


class RuleCombinationBase(SQLModel):
    """Base model for rule combination tracking"""

    combination_id: UUID = Field(
        default_factory=uuid4, sa_column=Column(PGUUID, unique=True)
    )
    rule_set: dict[str, Any] = Field(sa_column=Column(JSONB))
    prompt_type: str | None = Field(default=None, max_length=50)
    combined_effectiveness: float | None = Field(default=None)
    individual_scores: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    sample_size: int = Field(default=1)
    statistical_confidence: float | None = Field(default=None)


class RuleCombination(RuleCombinationBase, TimestampMixin, table=True):
    """Rule combination effectiveness tracking"""

    id: int | None = Field(default=None, primary_key=True)


class RuleCombinationCreate(RuleCombinationBase):
    """Model for creating rule combinations"""


class RuleCombinationRead(RuleCombinationBase, TimestampMixin):
    """Model for reading rule combinations"""

    id: int


# ===================================
# User Feedback Models
# ===================================


class UserFeedbackBase(SQLModel):
    """Base model for user feedback"""

    feedback_id: UUID = Field(
        default_factory=uuid4, sa_column=Column(PGUUID, unique=True)
    )
    original_prompt: str
    improved_prompt: str
    user_rating: int = Field(ge=1, le=5)
    applied_rules: dict[str, Any] = Field(sa_column=Column(JSONB))
    user_context: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    improvement_areas: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    user_notes: str | None = Field(default=None)
    session_id: str | None = Field(default=None, max_length=100, index=True)


class UserFeedback(UserFeedbackBase, table=True):
    """User feedback on prompt improvements"""

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (
        CheckConstraint("user_rating BETWEEN 1 AND 5"),
        Index(
            "idx_user_feedback_applied_rules", "applied_rules", postgresql_using="gin"
        ),
        Index("idx_user_feedback_rating_date", "user_rating", "created_at"),
    )


class UserFeedbackCreate(UserFeedbackBase):
    """Model for creating user feedback"""


class UserFeedbackRead(UserFeedbackBase):
    """Model for reading user feedback"""

    id: int
    created_at: datetime


# ===================================
# Improvement Session Models
# ===================================


class ImprovementSessionBase(SQLModel):
    """Base model for improvement sessions"""

    session_id: str = Field(max_length=100, unique=True, index=True)
    user_id: str | None = Field(default=None, max_length=50, index=True)
    original_prompt: str
    final_prompt: str | None = Field(default=None)
    rules_applied: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    iteration_count: int = Field(default=1)
    total_improvement_score: float | None = Field(default=None)
    session_metadata: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    started_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    completed_at: datetime | None = Field(default=None)
    status: str = Field(default="active", max_length=20, index=True)


class ImprovementSession(ImprovementSessionBase, table=True):
    """Prompt improvement session tracking"""

    __tablename__ = "improvement_sessions"
    id: int | None = Field(default=None, primary_key=True)

    __table_args__ = (
        CheckConstraint("status IN ('active', 'completed', 'abandoned')"),
    )


class ImprovementSessionCreate(ImprovementSessionBase):
    """Model for creating improvement sessions"""


class ImprovementSessionRead(ImprovementSessionBase):
    """Model for reading improvement sessions"""

    id: int


# ===================================
# ML Optimization Models
# ===================================


class MLModelPerformanceBase(SQLModel):
    """Base model for ML model performance"""

    model_version: str = Field(max_length=50, index=True)
    model_type: str = Field(max_length=50, index=True)
    accuracy_score: float | None = Field(default=None)
    precision_score: float | None = Field(default=None)
    recall_score: float | None = Field(default=None)
    f1_score: float | None = Field(default=None)
    training_data_size: int | None = Field(default=None)
    validation_data_size: int | None = Field(default=None)
    hyperparameters: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    feature_importance: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    model_artifacts_path: str | None = Field(default=None)
    mlflow_run_id: str | None = Field(default=None, max_length=100)


class MLModelPerformance(MLModelPerformanceBase, table=True):
    """ML model performance tracking"""

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MLModelPerformanceCreate(MLModelPerformanceBase):
    """Model for creating ML model performance records"""


class MLModelPerformanceRead(MLModelPerformanceBase):
    """Model for reading ML model performance records"""

    id: int
    created_at: datetime


# ===================================
# Pattern Discovery Models
# ===================================


class DiscoveredPatternBase(SQLModel):
    """Base model for discovered patterns"""

    pattern_id: UUID = Field(
        default_factory=uuid4, sa_column=Column(PGUUID, unique=True)
    )
    pattern_name: str | None = Field(default=None, max_length=100)
    pattern_description: str | None = Field(default=None)
    pattern_rule: dict[str, Any] = Field(sa_column=Column(JSONB))
    discovery_method: str | None = Field(default=None, max_length=50)
    effectiveness_score: float | None = Field(default=None, index=True)
    support_count: int | None = Field(default=None)
    confidence_interval: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    validation_status: str = Field(default="pending", max_length=20, index=True)
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: datetime | None = Field(default=None)


class DiscoveredPattern(DiscoveredPatternBase, table=True):
    """ML-discovered rule patterns"""

    id: int | None = Field(default=None, primary_key=True)

    __table_args__ = (
        CheckConstraint("validation_status IN ('pending', 'validated', 'rejected')"),
    )


class DiscoveredPatternCreate(DiscoveredPatternBase):
    """Model for creating discovered patterns"""


class DiscoveredPatternRead(DiscoveredPatternBase):
    """Model for reading discovered patterns"""

    id: int


# ===================================
# Rule Metadata Models
# ===================================


class RuleMetadataBase(SQLModel):
    """Base model for rule metadata"""

    rule_id: str = Field(max_length=50, unique=True, index=True)
    rule_name: str = Field(max_length=100)
    rule_category: str | None = Field(default=None, max_length=50)
    rule_description: str | None = Field(default=None)
    default_parameters: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    parameter_constraints: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB)
    )
    enabled: bool = Field(default=True)
    priority: int = Field(default=100)
    rule_version: str = Field(default="1.0.0", max_length=20)


class RuleMetadata(RuleMetadataBase, TimestampMixin, table=True):
    """Rule configuration and metadata"""

    __tablename__ = "rule_metadata"
    id: int | None = Field(default=None, primary_key=True)


class RuleMetadataCreate(RuleMetadataBase):
    """Model for creating rule metadata"""


class RuleMetadataRead(RuleMetadataBase, TimestampMixin):
    """Model for reading rule metadata"""

    id: int


# ===================================
# A/B Testing Models
# ===================================


class ABExperimentBase(SQLModel):
    """Base model for A/B experiments"""

    experiment_id: UUID = Field(
        default_factory=uuid4, sa_column=Column(PGUUID, unique=True)
    )
    experiment_name: str = Field(max_length=100)
    description: str | None = Field(default=None)
    control_rules: dict[str, Any] = Field(sa_column=Column(JSONB))
    treatment_rules: dict[str, Any] = Field(sa_column=Column(JSONB))
    target_metric: str | None = Field(default=None, max_length=50)
    sample_size_per_group: int | None = Field(default=None)
    current_sample_size: int = Field(default=0)
    significance_threshold: float = Field(default=0.05)
    status: str = Field(default="running", max_length=20, index=True)
    results: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)


class ABExperiment(ABExperimentBase, table=True):
    """A/B testing experiments for rule combinations"""

    id: int | None = Field(default=None, primary_key=True)

    __table_args__ = (
        CheckConstraint("status IN ('planning', 'running', 'completed', 'stopped')"),
    )


class ABExperimentCreate(ABExperimentBase):
    """Model for creating A/B experiments"""


class ABExperimentRead(ABExperimentBase):
    """Model for reading A/B experiments"""

    id: int


# ===================================
# Analytics and Response Models
# ===================================


class RuleEffectivenessStats(SQLModel):
    """Statistics for rule effectiveness"""

    rule_id: str
    rule_name: str
    usage_count: int
    avg_improvement: float
    score_stddev: float | None
    min_improvement: float
    max_improvement: float
    avg_confidence: float
    avg_execution_time: float | None
    prompt_types_count: int


class UserSatisfactionStats(SQLModel):
    """Statistics for user satisfaction"""

    feedback_date: datetime
    total_feedback: int
    avg_rating: float
    positive_feedback: int
    negative_feedback: int
    rules_used: list[str]


class PromptImprovementRequest(SQLModel):
    """Request model for prompt improvement"""

    prompt: str
    user_context: dict[str, Any] | None = None
    session_id: str | None = None
    preferred_rules: list[str] | None = None


class PromptImprovementResponse(SQLModel):
    """Response model for prompt improvement"""

    original_prompt: str
    improved_prompt: str
    applied_rules: list[dict[str, Any]]
    processing_time_ms: int
    session_id: str
    improvement_summary: dict[str, Any]
    confidence_score: float


class UserFeedbackRequest(SQLModel):
    """Request model for user feedback"""

    session_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: str | None = None
    improvement_areas: list[str] | None = None
