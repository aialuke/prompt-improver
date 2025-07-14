"""Database models for prompt improvement system.
Enhanced with Apriori association rules and pattern discovery tracking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import sqlmodel
from sqlalchemy import JSON, Index, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


class PromptSession(SQLModel, table=True):
    """Table for tracking prompt improvement sessions"""

    __tablename__: str = "prompt_sessions"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    session_id: str = Field(unique=True, index=True)
    original_prompt: str
    improved_prompt: str
    user_context: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    quality_score: Optional[float] = Field(default=None)
    improvement_score: Optional[float] = Field(default=None)
    confidence_level: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)

    # Relationships
    rule_performances: List["RulePerformance"] = Relationship(back_populates="session")
    user_feedback: Optional["UserFeedback"] = Relationship(back_populates="session")


class ABExperiment(SQLModel, table=True):
    """Table for A/B testing experiments"""

    __tablename__: str = "ab_experiments"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()), unique=True, index=True)
    experiment_name: str = Field(max_length=100)
    description: Optional[str] = Field(default=None)
    control_rules: Dict[str, Any] = Field(sa_column=sqlmodel.Column(JSON))
    treatment_rules: Dict[str, Any] = Field(sa_column=sqlmodel.Column(JSON))
    target_metric: str = Field(default="improvement_score", max_length=50)
    sample_size_per_group: Optional[int] = Field(default=None)
    current_sample_size: int = Field(default=0)
    significance_threshold: float = Field(default=0.05)
    status: str = Field(default="running", max_length=20)  # planning, running, completed, stopped
    results: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    experiment_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))

    __table_args__ = (
        Index('idx_ab_experiments_status', 'status', 'started_at'),
        Index('idx_ab_experiments_target_metric', 'target_metric'),
        {'extend_existing': True}
    )


class RuleMetadata(SQLModel, table=True):
    """Table for rule metadata and configuration"""

    __tablename__: str = "rule_metadata"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    rule_id: str = Field(unique=True, index=True)
    rule_name: str
    description: Optional[str] = Field(default=None)
    category: str = Field(default="general")
    is_enabled: bool = Field(default=True)
    priority: int = Field(default=1)
    default_parameters: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)

    # Relationships
    performances: List["RulePerformance"] = Relationship(back_populates="rule")


class RulePerformance(SQLModel, table=True):
    """Table for tracking rule performance metrics"""

    __tablename__: str = "rule_performance"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    session_id: str = Field(foreign_key="prompt_sessions.session_id", index=True)
    rule_id: str = Field(foreign_key="rule_metadata.rule_id", index=True)
    improvement_score: float
    execution_time_ms: float
    confidence_level: float
    parameters_used: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    session: PromptSession = Relationship(back_populates="rule_performances")
    rule: RuleMetadata = Relationship(back_populates="performances")


class UserFeedback(SQLModel, table=True):
    """Table for user feedback on prompt improvements"""

    __tablename__: str = "user_feedback"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    session_id: str = Field(foreign_key="prompt_sessions.session_id", unique=True, index=True)
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = Field(default=None)
    improvement_areas: Optional[List[str]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    is_processed: bool = Field(default=False)
    ml_optimized: bool = Field(default=False)
    model_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    session: PromptSession = Relationship(back_populates="user_feedback")


class MLModelPerformance(SQLModel, table=True):
    """Table for tracking ML model performance metrics"""

    __tablename__: str = "ml_model_performance"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    model_id: str = Field(index=True)
    model_type: str = Field(default="sklearn")
    performance_score: float
    accuracy: float
    precision: float
    recall: float
    training_samples: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ImprovementSession(SQLModel, table=True):
    """Enhanced session model with additional metadata"""

    __tablename__: str = "improvement_sessions"
    __table_args__ = {'extend_existing': True}

    id: int = Field(primary_key=True)
    session_id: str = Field(unique=True, index=True)
    original_prompt: str
    final_prompt: str
    rules_applied: Optional[List[str]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    user_context: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    improvement_metrics: Optional[Dict[str, float]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Pydantic models for API requests/responses
class ImprovementSessionCreate(SQLModel):
    """Model for creating improvement sessions"""

    session_id: str
    original_prompt: str
    final_prompt: str
    rules_applied: Optional[List[str]] = None
    user_context: Optional[Dict[str, Any]] = None
    improvement_metrics: Optional[Dict[str, float]] = None


class ABExperimentCreate(SQLModel):
    """Model for creating A/B testing experiments"""

    experiment_name: str
    description: Optional[str] = None
    control_rules: Dict[str, Any]
    treatment_rules: Dict[str, Any]
    target_metric: str = "improvement_score"
    sample_size_per_group: int = 100
    status: str = "running"


class RulePerformanceCreate(SQLModel):
    """Model for creating rule performance records"""

    session_id: str
    rule_id: str
    improvement_score: float
    execution_time_ms: float
    confidence_level: float
    parameters_used: Optional[Dict[str, Any]] = None


class UserFeedbackCreate(SQLModel):
    """Model for creating user feedback records"""

    session_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    improvement_areas: Optional[List[str]] = None


# NEW: Apriori Association Rules Schema
class AprioriAssociationRule(SQLModel, table=True):
    """Table for storing Apriori association rules and their metrics"""
    
    __tablename__: str = "apriori_association_rules"
    
    id: int = Field(primary_key=True)
    antecedents: str = Field(index=True)  # JSON string of antecedent items
    consequents: str = Field(index=True)  # JSON string of consequent items
    support: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    lift: float = Field(gt=0.0)
    conviction: Optional[float] = Field(default=None, gt=0.0)
    rule_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    business_insight: Optional[str] = Field(default=None)
    pattern_category: str = Field(default="general")
    
    # Metadata
    discovery_run_id: Optional[str] = Field(default=None, index=True)
    data_window_days: Optional[int] = Field(default=30)
    min_support_threshold: Optional[float] = Field(default=None)
    min_confidence_threshold: Optional[float] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    
    __table_args__ = (
        UniqueConstraint('antecedents', 'consequents', name='unique_rule_pair'),
        Index('idx_association_rules_performance', 'support', 'confidence', 'lift'),
        Index('idx_association_rules_discovery', 'discovery_run_id', 'created_at'),
        {'extend_existing': True}
    )


class AprioriPatternDiscovery(SQLModel, table=True):
    """Table for tracking Apriori pattern discovery runs and metadata"""
    
    __tablename__: str = "apriori_pattern_discovery"
    
    id: int = Field(primary_key=True)
    discovery_run_id: str = Field(unique=True, index=True)
    
    # Configuration parameters
    min_support: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    min_lift: float = Field(gt=0.0)
    max_itemset_length: int = Field(ge=1, le=10, default=5)
    data_window_days: int = Field(ge=1, default=30)
    
    # Results metadata
    transaction_count: int = Field(ge=0, default=0)
    frequent_itemsets_count: int = Field(ge=0, default=0)
    association_rules_count: int = Field(ge=0, default=0)
    execution_time_seconds: Optional[float] = Field(default=None, ge=0.0)
    
    # Discovery results summary
    top_patterns_summary: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    pattern_insights: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    quality_metrics: Optional[Dict[str, float]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    
    # Status tracking
    status: str = Field(default="running")  # running, completed, failed
    error_message: Optional[str] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    
    __table_args__ = (
        Index('idx_discovery_status', 'status', 'created_at'),
        Index('idx_discovery_config', 'min_support', 'min_confidence'),
        {'extend_existing': True}
    )


class FrequentItemset(SQLModel, table=True):
    """Table for storing frequent itemsets discovered by Apriori algorithm"""
    
    __tablename__: str = "frequent_itemsets"
    __table_args__ = {'extend_existing': True}
    
    id: int = Field(primary_key=True)
    discovery_run_id: str = Field(foreign_key="apriori_pattern_discovery.discovery_run_id", index=True)
    
    itemset: str = Field(index=True)  # JSON string of items in the itemset
    itemset_length: int = Field(ge=1, le=10)
    support: float = Field(ge=0.0, le=1.0)
    
    # Context and metadata
    itemset_type: str = Field(default="mixed")  # rule_combination, prompt_characteristic, outcome_pattern
    business_relevance: Optional[str] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_itemset_support', 'support', 'itemset_length'),
        Index('idx_itemset_type', 'itemset_type', 'discovery_run_id'),
        {'extend_existing': True}
    )


class PatternEvaluation(SQLModel, table=True):
    """Table for tracking evaluation and validation of discovered patterns"""
    
    __tablename__: str = "pattern_evaluations"
    __table_args__ = {'extend_existing': True}
    
    id: int = Field(primary_key=True)
    
    # Pattern reference
    pattern_type: str = Field(index=True)  # apriori_rule, frequent_itemset, traditional_pattern
    pattern_reference_id: Optional[int] = Field(default=None)  # ID in related table
    discovery_run_id: str = Field(index=True)
    
    # Evaluation metrics
    validation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    business_impact_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    implementation_difficulty: Optional[int] = Field(default=None, ge=1, le=5)
    
    # Validation results
    cross_validation_results: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    a_b_test_results: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    
    # Status and feedback
    evaluation_status: str = Field(default="pending")  # pending, validated, rejected, implemented
    evaluator_notes: Optional[str] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    
    __table_args__ = (
        Index('idx_pattern_eval_scores', 'validation_score', 'business_impact_score'),
        Index('idx_pattern_eval_status', 'evaluation_status', 'created_at'),
        {'extend_existing': True}
    )


class AdvancedPatternResults(SQLModel, table=True):
    """Table for storing results from advanced pattern discovery (HDBSCAN, FP-Growth, etc.)"""
    
    __tablename__: str = "advanced_pattern_results"
    __table_args__ = {'extend_existing': True}
    
    id: int = Field(primary_key=True)
    discovery_run_id: str = Field(unique=True, index=True)
    
    # Algorithm configuration
    algorithms_used: List[str] = Field(sa_column=sqlmodel.Column(JSON))
    discovery_modes: List[str] = Field(sa_column=sqlmodel.Column(JSON))
    
    # Pattern discovery results
    parameter_patterns: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    sequence_patterns: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    performance_patterns: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    semantic_patterns: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    apriori_patterns: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    
    # Cross-validation and ensemble results
    cross_validation: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    ensemble_analysis: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    
    # Unified insights
    unified_recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    business_insights: Optional[Dict[str, Any]] = Field(default=None, sa_column=sqlmodel.Column(JSON))
    
    # Execution metadata
    execution_time_seconds: float = Field(ge=0.0)
    total_patterns_discovered: int = Field(ge=0, default=0)
    discovery_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    algorithms_count: int = Field(ge=1, default=1)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_advanced_patterns_algorithms', 'algorithms_count', 'execution_time_seconds'),
        Index('idx_advanced_patterns_quality', 'discovery_quality_score', 'total_patterns_discovered'),
        Index('idx_advanced_patterns_created', 'created_at'),
        {'extend_existing': True}
    )


# API Models for Apriori functionality
class AprioriAnalysisRequest(SQLModel):
    """Request model for Apriori analysis"""
    
    window_days: int = Field(ge=1, le=365, default=30)
    min_support: float = Field(ge=0.01, le=1.0, default=0.1)
    min_confidence: float = Field(ge=0.1, le=1.0, default=0.6)
    min_lift: float = Field(gt=0.0, default=1.0)
    max_itemset_length: int = Field(ge=1, le=10, default=5)
    save_to_database: bool = Field(default=True)


class AprioriAnalysisResponse(SQLModel):
    """Response model for Apriori analysis results"""
    
    discovery_run_id: str
    transaction_count: int
    frequent_itemsets_count: int
    association_rules_count: int
    execution_time_seconds: float
    
    # Pattern summaries
    top_itemsets: List[Dict[str, Any]]
    top_rules: List[Dict[str, Any]]
    pattern_insights: Dict[str, Any]
    
    # Configuration used
    config: Dict[str, Any]
    status: str
    timestamp: str


class PatternDiscoveryRequest(SQLModel):
    """Request model for comprehensive pattern discovery"""
    
    min_effectiveness: float = Field(ge=0.0, le=1.0, default=0.7)
    min_support: int = Field(ge=1, default=5)
    use_advanced_discovery: bool = Field(default=True)
    include_apriori: bool = Field(default=True)
    pattern_types: Optional[List[str]] = Field(default=None)
    use_ensemble: bool = Field(default=True)


class PatternDiscoveryResponse(SQLModel):
    """Response model for comprehensive pattern discovery results"""
    
    status: str
    discovery_run_id: str
    
    # Discovery results
    traditional_patterns: Optional[Dict[str, Any]]
    advanced_patterns: Optional[Dict[str, Any]]
    apriori_patterns: Optional[Dict[str, Any]]
    
    # Analysis results
    cross_validation: Optional[Dict[str, Any]]
    unified_recommendations: List[Dict[str, Any]]
    business_insights: Dict[str, Any]
    
    # Metadata
    discovery_metadata: Dict[str, Any]


# --- Analytics Response Models (2025 Best Practice) ---

from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class RuleEffectivenessStats(BaseModel):
    rule_id: str
    rule_name: str
    usage_count: int
    avg_improvement: float
    score_stddev: float
    min_improvement: float
    max_improvement: float
    avg_confidence: float
    avg_execution_time: float
    prompt_types_count: int

class UserSatisfactionStats(BaseModel):
    feedback_date: date
    total_feedback: int
    avg_rating: float
    positive_feedback: int
    negative_feedback: int
    rules_used: List[str]
