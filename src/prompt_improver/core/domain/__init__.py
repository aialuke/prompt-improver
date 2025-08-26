"""Core domain types and value objects.

This package contains domain-specific types that are independent of infrastructure
concerns. These types are used in protocols to eliminate circular import risks
and ensure clean architecture boundaries.

Clean Architecture Principle: Domain types should be infrastructure-agnostic.
"""

from prompt_improver.core.domain.enums import (
    AnalysisStatus,
    HealthStatus,
    ModelStatus,
    SessionStatus,
    TrainingStatus,
    ValidationLevel,
)
from prompt_improver.core.domain.types import (
    AnalysisId,
    # Request/Response types
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    # Health types
    HealthCheckResultData,
    HealthStatusData,
    # Session types
    ImprovementSessionData,
    ModelId,
    # ML types
    ModelMetricsData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
    PromptSessionData,
    RuleId,
    # Common types
    SessionId,
    TrainingResultData,
    TrainingSessionData,
    # Feedback types
    UserFeedbackData,
    UserId,
)

__all__ = [
    "AnalysisId",
    "AnalysisStatus",
    # Request/Response types
    "AprioriAnalysisRequestData",
    "AprioriAnalysisResponseData",
    # Health types
    "HealthCheckResultData",
    "HealthStatus",
    "HealthStatusData",
    # Session types
    "ImprovementSessionData",
    "ModelId",
    # ML types
    "ModelMetricsData",
    "ModelStatus",
    "PatternDiscoveryRequestData",
    "PatternDiscoveryResponseData",
    "PromptSessionData",
    "RuleId",
    # Common types
    "SessionId",
    # Enums
    "SessionStatus",
    "TrainingResultData",
    "TrainingSessionData",
    "TrainingStatus",
    # Feedback types
    "UserFeedbackData",
    "UserId",
    "ValidationLevel",
]
