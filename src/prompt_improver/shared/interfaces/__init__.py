"""Common interfaces and protocols for dependency injection"""

from .improvement import IImprovementService, IMLService
from .repository import IRepository, IPromptRepository
from .automl import IAutoMLCallback, IAutoMLOrchestrator
from .emergency import IEmergencyOperations

__all__ = [
    "IImprovementService", 
    "IMLService",
    "IRepository", 
    "IPromptRepository",
    "IAutoMLCallback", 
    "IAutoMLOrchestrator",
    "IEmergencyOperations"
]