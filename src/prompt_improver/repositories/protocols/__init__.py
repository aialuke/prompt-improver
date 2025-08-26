"""Repository protocol definitions for clean architecture.

This module provides abstract interfaces for the repository layer,
enabling dependency inversion and improved testability following SOLID principles.
"""

from prompt_improver.repositories.protocols.analytics_repository_protocol import (
    AnalyticsRepositoryProtocol,
)
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
)
from prompt_improver.repositories.protocols.base_repository_protocol import (
    BaseRepositoryProtocol,
    QueryBuilderProtocol,
    TransactionManagerProtocol,
)
from prompt_improver.repositories.protocols.health_repository_protocol import (
    HealthRepositoryProtocol,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    MLRepositoryProtocol,
)
from prompt_improver.repositories.protocols.monitoring_repository_protocol import (
    MonitoringRepositoryProtocol,
)
from prompt_improver.repositories.protocols.persistence_repository_protocol import (
    PersistenceRepositoryProtocol,
)
from prompt_improver.repositories.protocols.rules_repository_protocol import (
    RulesRepositoryProtocol,
)
from prompt_improver.repositories.protocols.user_feedback_repository_protocol import (
    UserFeedbackRepositoryProtocol,
)
from prompt_improver.shared.interfaces.protocols.database import (
    QueryExecutorProtocol,
    SessionManagerProtocol,
    SessionProtocol,
)

__all__ = [
    # Domain-specific protocols
    "AnalyticsRepositoryProtocol",
    "AprioriRepositoryProtocol",
    # Base protocols
    "BaseRepositoryProtocol",
    "HealthRepositoryProtocol",
    "MLRepositoryProtocol",
    "MonitoringRepositoryProtocol",
    "PersistenceRepositoryProtocol",
    "QueryBuilderProtocol",
    "QueryExecutorProtocol",
    "RulesRepositoryProtocol",
    # Session management protocols
    "SessionManagerProtocol",
    "SessionProtocol",
    "TransactionManagerProtocol",
    "UserFeedbackRepositoryProtocol",
]
