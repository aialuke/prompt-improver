"""Core domain types and value objects.

This package contains domain-specific types that are independent of infrastructure
concerns. These types are used in protocols to eliminate circular import risks
and ensure clean architecture boundaries.

Clean Architecture Principle: Domain types should be infrastructure-agnostic.
"""

from prompt_improver.core.domain.types import (
    # Session types
    ImprovementSessionData,
    PromptSessionData,
    TrainingSessionData,
    
    # Request/Response types
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
    
    # Feedback types
    UserFeedbackData,
    
    # Health types
    HealthCheckResultData,
    HealthStatusData,
    
    # ML types
    ModelMetricsData,
    TrainingResultData,
    
    # Common types
    SessionId,
    UserId,
    ModelId,
    RuleId,
    AnalysisId,
)

from prompt_improver.core.domain.enums import (
    SessionStatus,
    AnalysisStatus,
    HealthStatus,
    ModelStatus,
    TrainingStatus,
    ValidationLevel,
)

__all__ = [
    # Session types
    "ImprovementSessionData",
    "PromptSessionData", 
    "TrainingSessionData",
    
    # Request/Response types
    "AprioriAnalysisRequestData",
    "AprioriAnalysisResponseData",
    "PatternDiscoveryRequestData", 
    "PatternDiscoveryResponseData",
    
    # Feedback types
    "UserFeedbackData",
    
    # Health types
    "HealthCheckResultData",
    "HealthStatusData",
    
    # ML types
    "ModelMetricsData",
    "TrainingResultData",
    
    # Common types
    "SessionId",
    "UserId", 
    "ModelId",
    "RuleId",
    "AnalysisId",
    
    # Enums
    "SessionStatus",
    "AnalysisStatus",
    "HealthStatus",
    "ModelStatus",
    "TrainingStatus",
    "ValidationLevel",
]