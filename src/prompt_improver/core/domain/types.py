"""Domain value objects and data types.

These types represent domain concepts without infrastructure dependencies.
They are used in protocols to ensure clean architecture boundaries.
"""

from datetime import datetime
from typing import Any, Dict, List, NewType, Optional
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

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
    user_id: Optional[UserId]
    original_prompt: str
    improved_prompt: Optional[str]
    improvement_rules: List[str]
    metrics: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PromptSessionData:
    """Domain representation of a prompt session."""
    id: SessionId
    prompt_text: str
    context: Dict[str, Any]
    analysis_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TrainingSessionData:
    """Domain representation of a training session."""
    id: SessionId
    model_id: ModelId
    dataset_info: Dict[str, Any]
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


@dataclass(frozen=True)
class UserFeedbackData:
    """Domain representation of user feedback."""
    id: UUID
    session_id: SessionId
    user_id: Optional[UserId]
    rating: int
    comments: str
    feedback_type: str
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class AprioriAnalysisRequestData:
    """Domain representation of Apriori analysis request."""
    id: AnalysisId
    dataset_path: str
    min_support: float
    min_confidence: float
    parameters: Dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class AprioriAnalysisResponseData:
    """Domain representation of Apriori analysis response."""
    request_id: AnalysisId
    rules: List[Dict[str, Any]]
    frequent_itemsets: List[Dict[str, Any]]
    metrics: Dict[str, float]
    execution_time: float
    status: str
    completed_at: datetime


@dataclass(frozen=True)
class PatternDiscoveryRequestData:
    """Domain representation of pattern discovery request."""
    id: AnalysisId
    data_source: str
    algorithm_config: Dict[str, Any]
    filters: Dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class PatternDiscoveryResponseData:
    """Domain representation of pattern discovery response."""
    request_id: AnalysisId
    patterns: List[Dict[str, Any]]
    insights: Dict[str, Any]
    confidence_scores: Dict[str, float]
    completed_at: datetime


@dataclass(frozen=True)
class HealthCheckResultData:
    """Domain representation of health check result."""
    service_name: str
    status: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]
    dependencies: List[str]


@dataclass(frozen=True)
class HealthStatusData:
    """Domain representation of overall health status."""
    overall_status: str
    service_statuses: Dict[str, HealthCheckResultData]
    timestamp: datetime
    summary: Dict[str, Any]


@dataclass(frozen=True)
class ModelMetricsData:
    """Domain representation of ML model metrics."""
    model_id: ModelId
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    custom_metrics: Dict[str, float]
    evaluation_timestamp: datetime


@dataclass(frozen=True)
class RuleEffectivenessData:
    """Domain representation of rule effectiveness statistics."""
    rule_id: RuleId
    total_applications: int
    success_rate: float
    average_improvement: float
    confidence_level: float
    performance_metrics: Dict[str, float]
    last_updated: datetime


@dataclass(frozen=True)
class UserSatisfactionData:
    """Domain representation of user satisfaction statistics."""
    metric_period: str
    total_responses: int
    average_rating: float
    satisfaction_distribution: Dict[str, int]
    improvement_feedback: Dict[str, Any]
    trends: Dict[str, float]
    calculated_at: datetime


@dataclass(frozen=True)
class TrainingResultData:
    """Domain representation of training results."""
    model_id: ModelId
    final_metrics: ModelMetricsData
    training_history: List[Dict[str, float]]
    best_checkpoint: str
    training_duration: float
    hyperparameters: Dict[str, Any]
    validation_results: Dict[str, Any]


# Filter and configuration types
@dataclass(frozen=True)
class AssociationRuleFilterData:
    """Domain representation of association rule filters."""
    min_support: float
    min_confidence: float
    min_lift: float
    max_rules: Optional[int]
    item_constraints: List[str]


@dataclass(frozen=True)
class PatternDiscoveryFilterData:
    """Domain representation of pattern discovery filters."""
    min_frequency: int
    max_pattern_length: int
    pattern_types: List[str]
    exclusion_rules: List[str]


# Validation and constraint types
@dataclass(frozen=True)
class ValidationConstraintData:
    """Domain representation of validation constraints."""
    max_length: Optional[int]
    min_length: Optional[int]
    allowed_patterns: List[str]
    forbidden_patterns: List[str]
    custom_rules: Dict[str, Any]


@dataclass(frozen=True)
class BusinessRuleData:
    """Domain representation of business rules."""
    rule_id: RuleId
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
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
    quality_metrics: Dict[str, float]


@dataclass(frozen=True)
class GenerationQualityAssessmentData:
    """Domain representation of quality assessment."""
    generation_id: str
    quality_score: float
    assessment_criteria: Dict[str, float]
    feedback: Optional[str]


@dataclass(frozen=True)
class GenerationSessionData:
    """Domain representation of generation session."""
    session_id: SessionId
    user_id: Optional[UserId]
    parameters: Dict[str, Any]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]


@dataclass(frozen=True)
class MLModelPerformanceData:
    """Domain representation of ML model performance."""
    model_id: ModelId
    performance_metrics: Dict[str, float]
    benchmark_scores: Dict[str, float]
    evaluation_date: datetime


@dataclass(frozen=True)
class SyntheticDataSampleData:
    """Domain representation of synthetic data sample."""
    sample_id: str
    generation_method: str
    data_content: Dict[str, Any]
    quality_indicators: Dict[str, float]


@dataclass(frozen=True)
class TrainingIterationData:
    """Domain representation of training iteration."""
    iteration_id: str
    session_id: SessionId
    iteration_number: int
    metrics: Dict[str, float]
    timestamp: datetime


@dataclass(frozen=True)
class TrainingPromptData:
    """Domain representation of training prompt."""
    prompt_id: str
    content: str
    category: str
    quality_score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TrainingSessionCreateData:
    """Domain representation for training session creation."""
    model_id: ModelId
    dataset_config: Dict[str, Any]
    training_parameters: Dict[str, Any]
    user_id: Optional[UserId]


@dataclass(frozen=True)
class TrainingSessionUpdateData:
    """Domain representation for training session update."""
    status: str
    current_epoch: int
    metrics: Dict[str, float]
    updated_at: datetime


@dataclass(frozen=True)
class MLExperimentData:
    """Domain representation of ML experiment."""
    experiment_id: str
    name: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    status: str
    created_at: datetime


# Apriori and Pattern Mining Domain Types
@dataclass(frozen=True)
class AprioriAssociationRuleData:
    """Domain representation of Apriori association rule."""
    id: int
    discovery_run_id: str
    antecedents: List[str]
    consequents: List[str]
    support: float
    confidence: float
    lift: float
    conviction: Optional[float]
    rule_metadata: Dict[str, Any]
    business_relevance: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class FrequentItemsetData:
    """Domain representation of frequent itemset."""
    id: int
    discovery_run_id: str
    itemset: str
    support: float
    itemset_length: int
    itemset_type: Optional[str]
    business_relevance: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class AprioriPatternDiscoveryData:
    """Domain representation of Apriori pattern discovery run."""
    id: str
    dataset_source: str
    algorithm_config: Dict[str, Any]
    execution_status: str
    total_patterns_found: int
    execution_time: float
    quality_metrics: Dict[str, float]
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


@dataclass(frozen=True)
class AdvancedPatternResultsData:
    """Domain representation of advanced pattern results."""
    id: int
    discovery_run_id: str
    analysis_type: str
    results_summary: Dict[str, Any]
    quality_score: float
    algorithms_used: List[str]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    created_at: datetime


@dataclass(frozen=True)
class PatternEvaluationData:
    """Domain representation of pattern evaluation."""
    id: int
    pattern_id: str
    pattern_type: str
    evaluation_criteria: Dict[str, Any]
    validation_score: float
    evaluation_status: str
    business_impact_score: Optional[float]
    recommendations: List[str]
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
    results_summary: Dict[str, Any]
    metadata: Dict[str, Any]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


@dataclass(frozen=True)
class DiscoveredPatternData:
    """Domain representation of discovered pattern."""
    id: str
    pattern_type: str
    pattern_description: str
    pattern_config: Dict[str, Any]
    avg_effectiveness: float
    support_count: int
    confidence_level: float
    business_context: Optional[str]
    discovery_run_id: str
    validation_status: str
    pattern_metadata: Dict[str, Any]
    discovered_at: datetime
    last_validated: Optional[datetime]


@dataclass(frozen=True)
class ImprovementSessionCreateData:
    """Domain representation for creating improvement session."""
    original_prompt: str
    user_id: Optional[UserId]
    context: Dict[str, Any]
    expected_outcome: Optional[str]
    priority: int
    session_config: Dict[str, Any]