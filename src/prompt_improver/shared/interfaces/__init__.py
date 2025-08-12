"""Common interfaces and protocols for dependency injection"""

from prompt_improver.shared.interfaces.ab_testing import (
    IABTestingService,
    NoOpABTestingService,
)
from prompt_improver.shared.interfaces.automl import (
    IAutoMLCallback,
    IAutoMLOrchestrator,
)
from prompt_improver.shared.interfaces.emergency import IEmergencyOperations
from prompt_improver.shared.interfaces.improvement import (
    IImprovementService,
    IMLService,
    IRuleEngine,
)
from prompt_improver.shared.interfaces.protocols import (
    CacheProtocol,
    ConfigManagerProtocol,
    ConnectionManagerProtocol,
    DatabaseManagerProtocol,
    EventBusProtocol,
    HealthCheckerProtocol,
    LoggerProtocol,
    MetricsCollectorProtocol,
    MLModelProtocol,
    RepositoryProtocol,
    RetryManagerProtocol,
    SecurityManagerProtocol,
    ServiceProtocol,
    WorkflowProtocol,
)
from prompt_improver.shared.interfaces.repository import IPromptRepository, IRepository

__all__ = [
    # Business interfaces
    "IABTestingService",
    "IAutoMLCallback",
    "IAutoMLOrchestrator",
    "IEmergencyOperations",
    "IImprovementService",
    "IMLService",
    "IPromptRepository",
    "IRepository",
    "IRuleEngine",
    "NoOpABTestingService",
    # Infrastructure protocols
    "CacheProtocol",
    "ConfigManagerProtocol",
    "ConnectionManagerProtocol",
    "DatabaseManagerProtocol",
    "EventBusProtocol",
    "HealthCheckerProtocol",
    "LoggerProtocol",
    "MetricsCollectorProtocol",
    "MLModelProtocol",
    "RepositoryProtocol",
    "RetryManagerProtocol",
    "SecurityManagerProtocol",
    "ServiceProtocol",
    "WorkflowProtocol",
]
