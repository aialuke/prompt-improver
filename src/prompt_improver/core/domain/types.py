"""Domain value objects and data types.

These types represent domain concepts without infrastructure dependencies.
They are used in protocols to ensure clean architecture boundaries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NewType
from uuid import UUID

# Type aliases for clarity and type safety
SessionId = NewType('SessionId', UUID)
UserId = NewType('UserId', UUID)
ModelId = NewType('ModelId', str)
RuleId = NewType('RuleId', str)
AnalysisId = NewType('AnalysisId', UUID)


@dataclass(frozen=True)
class ImprovementSessionData:
    """Domain representation of an improvement session."""
    id: SessionId
    user_id: UserId | None
    original_prompt: str
    improved_prompt: str | None
    improvement_rules: list[str]
    metrics: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PromptSessionData:
    """Domain representation of a prompt session."""
    id: SessionId
    prompt_text: str
    context: dict[str, Any]
    analysis_results: dict[str, Any]
    performance_metrics: dict[str, float]
    created_at: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingSessionData:
    """Domain representation of a training session."""
    id: SessionId
    model_id: ModelId
    dataset_info: dict[str, Any]
    training_config: dict[str, Any]
    metrics: dict[str, float]
    status: str
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None


@dataclass(frozen=True)
class UserFeedbackData:
    """Domain representation of user feedback."""
    id: UUID
    session_id: SessionId
    user_id: UserId | None
    rating: int
    comments: str
    feedback_type: str
    created_at: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AprioriAnalysisRequestData:
    """Domain representation of Apriori analysis request."""
    id: AnalysisId
    dataset_path: str
    min_support: float
    min_confidence: float
    parameters: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class AprioriAnalysisResponseData:
    """Domain representation of Apriori analysis response."""
    request_id: AnalysisId
    rules: list[dict[str, Any]]
    frequent_itemsets: list[dict[str, Any]]
    metrics: dict[str, float]
    execution_time: float
    status: str
    completed_at: datetime


@dataclass(frozen=True)
class PatternDiscoveryRequestData:
    """Domain representation of pattern discovery request."""
    id: AnalysisId
    data_source: str
    algorithm_config: dict[str, Any]
    filters: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class PatternDiscoveryResponseData:
    """Domain representation of pattern discovery response."""
    request_id: AnalysisId
    patterns: list[dict[str, Any]]
    insights: dict[str, Any]
    confidence_scores: dict[str, float]
    completed_at: datetime


@dataclass(frozen=True)
class HealthCheckResultData:
    """Domain representation of health check result."""
    service_name: str
    status: str
    timestamp: datetime
    response_time_ms: float
    details: dict[str, Any]
    dependencies: list[str]


@dataclass(frozen=True)
class HealthStatusData:
    """Domain representation of overall health status."""
    overall_status: str
    service_statuses: dict[str, HealthCheckResultData]
    timestamp: datetime
    summary: dict[str, Any]


@dataclass(frozen=True)
class ModelMetricsData:
    """Domain representation of ML model metrics."""
    model_id: ModelId
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    custom_metrics: dict[str, float]
    evaluation_timestamp: datetime


@dataclass(frozen=True)
class RuleEffectivenessData:
    """Domain representation of rule effectiveness statistics."""
    rule_id: RuleId
    total_applications: int
    success_rate: float
    average_improvement: float
    confidence_level: float
    performance_metrics: dict[str, float]
    last_updated: datetime


@dataclass(frozen=True)
class UserSatisfactionData:
    """Domain representation of user satisfaction statistics."""
    metric_period: str
    total_responses: int
    average_rating: float
    satisfaction_distribution: dict[str, int]
    improvement_feedback: dict[str, Any]
    trends: dict[str, float]
    calculated_at: datetime


@dataclass(frozen=True)
class TrainingResultData:
    """Domain representation of training results."""
    model_id: ModelId
    final_metrics: ModelMetricsData
    training_history: list[dict[str, float]]
    best_checkpoint: str
    training_duration: float
    hyperparameters: dict[str, Any]
    validation_results: dict[str, Any]


# Filter and configuration types
@dataclass(frozen=True)
class AssociationRuleFilterData:
    """Domain representation of association rule filters."""
    min_support: float
    min_confidence: float
    min_lift: float
    max_rules: int | None
    item_constraints: list[str]


@dataclass(frozen=True)
class PatternDiscoveryFilterData:
    """Domain representation of pattern discovery filters."""
    min_frequency: int
    max_pattern_length: int
    pattern_types: list[str]
    exclusion_rules: list[str]


# Validation and constraint types
@dataclass(frozen=True)
class ValidationConstraintData:
    """Domain representation of validation constraints."""
    max_length: int | None
    min_length: int | None
    allowed_patterns: list[str]
    forbidden_patterns: list[str]
    custom_rules: dict[str, Any]


@dataclass(frozen=True)
class BusinessRuleData:
    """Domain representation of business rules."""
    rule_id: RuleId
    name: str
    description: str
    conditions: dict[str, Any]
    actions: dict[str, Any]
    priority: int
    enabled: bool


# Additional ML-specific types for protocol compatibility
@dataclass(frozen=True)
class GenerationAnalyticsData:
    """Domain representation of generation analytics."""
    session_id: SessionId
    method_name: str
    success_rate: float
    avg_quality_score: float
    total_generations: int
    timestamp: datetime


@dataclass(frozen=True)
class GenerationBatchData:
    """Domain representation of generation batch."""
    batch_id: str
    session_id: SessionId
    prompt_count: int
    completion_status: str
    created_at: datetime


@dataclass(frozen=True)
class GenerationMethodPerformanceData:
    """Domain representation of generation method performance."""
    method_name: str
    avg_response_time: float
    success_rate: float
    quality_metrics: dict[str, float]


@dataclass(frozen=True)
class GenerationQualityAssessmentData:
    """Domain representation of quality assessment."""
    generation_id: str
    quality_score: float
    assessment_criteria: dict[str, float]
    feedback: str | None


@dataclass(frozen=True)
class GenerationSessionData:
    """Domain representation of generation session."""
    session_id: SessionId
    user_id: UserId | None
    parameters: dict[str, Any]
    status: str
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class MLModelPerformanceData:
    """Domain representation of ML model performance."""
    model_id: ModelId
    performance_metrics: dict[str, float]
    benchmark_scores: dict[str, float]
    evaluation_date: datetime


@dataclass(frozen=True)
class SyntheticDataSampleData:
    """Domain representation of synthetic data sample."""
    sample_id: str
    generation_method: str
    data_content: dict[str, Any]
    quality_indicators: dict[str, float]


@dataclass(frozen=True)
class TrainingIterationData:
    """Domain representation of training iteration."""
    iteration_id: str
    session_id: SessionId
    iteration_number: int
    metrics: dict[str, float]
    timestamp: datetime


@dataclass(frozen=True)
class TrainingPromptData:
    """Domain representation of training prompt."""
    prompt_id: str
    content: str
    category: str
    quality_score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingSessionCreateData:
    """Domain representation for training session creation."""
    model_id: ModelId
    dataset_config: dict[str, Any]
    training_parameters: dict[str, Any]
    user_id: UserId | None


@dataclass(frozen=True)
class TrainingSessionUpdateData:
    """Domain representation for training session update."""
    status: str
    current_epoch: int
    metrics: dict[str, float]
    updated_at: datetime


@dataclass(frozen=True)
class MLExperimentData:
    """Domain representation of ML experiment."""
    experiment_id: str
    name: str
    parameters: dict[str, Any]
    results: dict[str, Any]
    status: str
    created_at: datetime


# Apriori and Pattern Mining Domain Types
@dataclass(frozen=True)
class AprioriAssociationRuleData:
    """Domain representation of Apriori association rule."""
    id: int
    discovery_run_id: str
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float
    conviction: float | None
    rule_metadata: dict[str, Any]
    business_relevance: str | None
    created_at: datetime


@dataclass(frozen=True)
class FrequentItemsetData:
    """Domain representation of frequent itemset."""
    id: int
    discovery_run_id: str
    itemset: str
    support: float
    itemset_length: int
    itemset_type: str | None
    business_relevance: str | None
    created_at: datetime


@dataclass(frozen=True)
class AprioriPatternDiscoveryData:
    """Domain representation of Apriori pattern discovery run."""
    id: str
    dataset_source: str
    algorithm_config: dict[str, Any]
    execution_status: str
    total_patterns_found: int
    execution_time: float
    quality_metrics: dict[str, float]
    created_at: datetime
    completed_at: datetime | None
    error_message: str | None


@dataclass(frozen=True)
class AdvancedPatternResultsData:
    """Domain representation of advanced pattern results."""
    id: int
    discovery_run_id: str
    analysis_type: str
    results_summary: dict[str, Any]
    quality_score: float
    algorithms_used: list[str]
    performance_metrics: dict[str, float]
    recommendations: list[str]
    created_at: datetime


@dataclass(frozen=True)
class PatternEvaluationData:
    """Domain representation of pattern evaluation."""
    id: int
    pattern_id: str
    pattern_type: str
    evaluation_criteria: dict[str, Any]
    validation_score: float
    evaluation_status: str
    business_impact_score: float | None
    recommendations: list[str]
    evaluated_at: datetime


# Additional domain types for prompt repository protocol
@dataclass(frozen=True)
class ABExperimentData:
    """Domain representation of A/B experiment."""
    id: str
    name: str
    description: str
    hypothesis: str
    target_metric: str
    control_version: str
    treatment_version: str
    sample_size_target: int
    current_sample_size: int
    status: str  # 'draft', 'running', 'completed', 'stopped'
    statistical_power: float
    confidence_level: float
    results_summary: dict[str, Any]
    metadata: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class DiscoveredPatternData:
    """Domain representation of discovered pattern."""
    id: str
    pattern_type: str
    pattern_description: str
    pattern_config: dict[str, Any]
    avg_effectiveness: float
    support_count: int
    confidence_level: float
    business_context: str | None
    discovery_run_id: str
    validation_status: str
    pattern_metadata: dict[str, Any]
    discovered_at: datetime
    last_validated: datetime | None


@dataclass(frozen=True)
class ImprovementSessionCreateData:
    """Domain representation for creating improvement session."""
    original_prompt: str
    user_id: UserId | None
    context: dict[str, Any]
    expected_outcome: str | None
    priority: int
    session_config: dict[str, Any]
