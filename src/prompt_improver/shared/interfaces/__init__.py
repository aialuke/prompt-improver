"""Common interfaces and protocols for dependency injection."""

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
from prompt_improver.shared.interfaces.protocols.cache import (
    BasicCacheProtocol as CacheProtocol,
)
from prompt_improver.shared.interfaces.protocols.cli import (
    WorkflowManagerProtocol as WorkflowProtocol,
)

# Import from consolidated protocols in protocols/ directory
from prompt_improver.shared.interfaces.protocols.core import (
    EventBusProtocol,
    HealthCheckProtocol as LoggerProtocol,  # Temporary mapping
    ServiceProtocol,
)
from prompt_improver.shared.interfaces.protocols.database import (
    ConnectionPoolCoreProtocol as ConnectionManagerProtocol,
    DatabaseProtocol as DatabaseManagerProtocol,
    DatabaseProtocol as RepositoryProtocol,
)
from prompt_improver.shared.interfaces.protocols.security import (
    AuthenticationProtocol as SecurityManagerProtocol,
)


# Heavy protocols available via lazy loading to avoid import penalty
def get_ml_model_protocol():
    """Lazy load ML model protocol to avoid heavy dependencies."""
    from prompt_improver.shared.interfaces.protocols.ml import (
        MLflowServiceProtocol as MLModelProtocol,
    )
    return MLModelProtocol


def get_health_checker_protocol():
    """Lazy load health checker protocol to avoid monitoring dependencies."""
    from prompt_improver.shared.interfaces.protocols.monitoring import (
        HealthCheckComponentProtocol as HealthCheckerProtocol,
    )
    return HealthCheckerProtocol


def get_metrics_collector_protocol():
    """Lazy load metrics collector protocol to avoid monitoring dependencies."""
    from prompt_improver.shared.interfaces.protocols.monitoring import (
        MetricsCollectorProtocol,
    )
    return MetricsCollectorProtocol


def get_retry_manager_protocol():
    """Lazy load retry manager protocol to avoid application dependencies."""
    from prompt_improver.shared.interfaces.protocols.application import (
        RetryStrategyProtocol as RetryManagerProtocol,
    )
    return RetryManagerProtocol


def get_config_manager_protocol():
    """Lazy load configuration manager protocol to avoid monitoring dependencies."""
    from prompt_improver.shared.interfaces.protocols.monitoring import (
        ConfigurationServiceProtocol as ConfigManagerProtocol,
    )
    return ConfigManagerProtocol


from prompt_improver.shared.interfaces.repository import IPromptRepository, IRepository

__all__ = [
    # Infrastructure protocols (consolidated)
    "CacheProtocol",
    "ConnectionManagerProtocol",
    "DatabaseManagerProtocol",
    "EventBusProtocol",
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
    "LoggerProtocol",
    "NoOpABTestingService",
    "RepositoryProtocol",
    "SecurityManagerProtocol",
    "ServiceProtocol",
    "WorkflowProtocol",
    "get_config_manager_protocol",
    "get_health_checker_protocol",
    "get_metrics_collector_protocol",
    # Lazy-loaded protocols (avoid heavy imports)
    "get_ml_model_protocol",
    "get_retry_manager_protocol",
]
