"""Repository interface definitions for dependency injection.

Concrete interface classes that combine protocol definitions with
implementation requirements for clean dependency injection.
"""

from abc import ABC
from typing import Protocol

from prompt_improver.repositories.protocols import (
    AnalyticsRepositoryProtocol,
    AprioriRepositoryProtocol,
    BaseRepositoryProtocol,
    HealthRepositoryProtocol,
    MLRepositoryProtocol,
    MonitoringRepositoryProtocol,
    PersistenceRepositoryProtocol,
    RulesRepositoryProtocol,
    SessionManagerProtocol,
    TransactionManagerProtocol,
    UserFeedbackRepositoryProtocol,
)
from prompt_improver.repositories.protocols.base_repository_protocol import (
    RepositoryFactoryProtocol,
)


class IAnalyticsRepository(AnalyticsRepositoryProtocol, Protocol):
    """Interface for analytics repository implementations."""


class IAprioriRepository(AprioriRepositoryProtocol, Protocol):
    """Interface for Apriori repository implementations."""


class IMLRepository(MLRepositoryProtocol, Protocol):
    """Interface for ML repository implementations."""


class IRulesRepository(RulesRepositoryProtocol, Protocol):
    """Interface for rules repository implementations."""


class IUserFeedbackRepository(UserFeedbackRepositoryProtocol, Protocol):
    """Interface for user feedback repository implementations."""


class IHealthRepository(HealthRepositoryProtocol, Protocol):
    """Interface for health repository implementations."""


class ITransactionManager(TransactionManagerProtocol, Protocol):
    """Interface for transaction manager implementations."""


class IPersistenceRepository(PersistenceRepositoryProtocol, Protocol):
    """Interface for persistence repository implementations."""


class IMonitoringRepository(MonitoringRepositoryProtocol, Protocol):
    """Interface for monitoring repository implementations."""


class ISessionManager(SessionManagerProtocol, Protocol):
    """Interface for session manager implementations."""


class IRepositoryFactory(RepositoryFactoryProtocol, Protocol):
    """Interface for repository factory implementations."""
