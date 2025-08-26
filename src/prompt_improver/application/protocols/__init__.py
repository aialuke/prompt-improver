"""Application Service Protocols.

This module defines the protocols (interfaces) for application services,
establishing clear contracts for business workflow orchestration.
"""

from prompt_improver.application.protocols.application_service_protocols import (
    AnalyticsApplicationServiceProtocol,
    ApplicationServiceProtocol,
    HealthApplicationServiceProtocol,
    MLApplicationServiceProtocol,
    PromptApplicationServiceProtocol,
    TrainingApplicationServiceProtocol,
)

__all__ = [
    "AnalyticsApplicationServiceProtocol",
    "ApplicationServiceProtocol",
    "HealthApplicationServiceProtocol",
    "MLApplicationServiceProtocol",
    "PromptApplicationServiceProtocol",
    "TrainingApplicationServiceProtocol",
]
