"""Common interfaces and protocols for dependency injection"""
from prompt_improver.shared.interfaces.automl import IAutoMLCallback, IAutoMLOrchestrator
from prompt_improver.shared.interfaces.emergency import IEmergencyOperations
from prompt_improver.shared.interfaces.improvement import IImprovementService, IMLService
from prompt_improver.shared.interfaces.repository import IPromptRepository, IRepository
__all__ = ['IAutoMLCallback', 'IAutoMLOrchestrator', 'IEmergencyOperations', 'IImprovementService', 'IMLService', 'IPromptRepository', 'IRepository']
