"""PostgreSQL connection pool manager with advanced scaling and monitoring.

Clean architecture implementation with decomposed components:
- ConnectionPoolCore: Core connection management and session creation
- PoolScalingManager: Dynamic scaling and optimization logic
- PoolMonitoringService: Health monitoring and metrics collection
- PostgreSQLPoolManager: Unified facade for orchestrated functionality

Follows single responsibility principle enabling better testability,
maintainability, and performance optimization.
"""

# Export the facade as the main interface
# Re-export supporting types and configurations
from prompt_improver.database.services.connection.pool_shared_context import (
    ConnectionInfo,
    ConnectionMode,
    DatabaseConfig,
    HealthStatus,
    PoolConfiguration,
    PoolState,
)
from prompt_improver.database.services.connection.postgres_pool_manager_facade import (
    PostgreSQLPoolManager,
)

# Export all public components for access if needed
__all__ = [
    "ConnectionInfo",
    "ConnectionMode",
    "DatabaseConfig",
    "HealthStatus",
    "PoolConfiguration",
    "PoolState",
    "PostgreSQLPoolManager",
]
