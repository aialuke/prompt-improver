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
from prompt_improver.repositories.protocols.rules_repository_protocol import (
    RulesRepositoryProtocol,
)
from prompt_improver.repositories.protocols.user_feedback_repository_protocol import (
    UserFeedbackRepositoryProtocol,
)
from prompt_improver.repositories.protocols.persistence_repository_protocol import (
    PersistenceRepositoryProtocol,
)
from prompt_improver.repositories.protocols.monitoring_repository_protocol import (
    MonitoringRepositoryProtocol,
)
from prompt_improver.repositories.protocols.session_manager_protocol import (
    SessionManagerProtocol,
    SessionProtocol,
    QueryExecutorProtocol,
)

__all__ = [
    # Base protocols
    "BaseRepositoryProtocol",
    "QueryBuilderProtocol",
    "TransactionManagerProtocol",
    # Session management protocols
    "SessionManagerProtocol",
    "SessionProtocol", 
    "QueryExecutorProtocol",
    # Domain-specific protocols
    "AnalyticsRepositoryProtocol",
    "AprioriRepositoryProtocol",
    "MLRepositoryProtocol",
    "RulesRepositoryProtocol",
    "UserFeedbackRepositoryProtocol",
    "HealthRepositoryProtocol",
    "PersistenceRepositoryProtocol",
    "MonitoringRepositoryProtocol",
]
