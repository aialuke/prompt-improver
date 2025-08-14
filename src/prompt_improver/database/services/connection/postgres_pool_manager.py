"""PostgreSQL connection pool manager with advanced scaling and monitoring.

Re-architected from monolithic design into clean architecture components:
- ConnectionPoolCore: Core connection management and session creation  
- PoolScalingManager: Dynamic scaling and optimization logic
- PoolMonitoringService: Health monitoring and metrics collection
- PostgreSQLPoolManager: Unified facade maintaining backward compatibility

This maintains identical functionality while following single responsibility
principle and enabling better testability and maintainability.
"""

# Export the facade as the main interface for backward compatibility
from prompt_improver.database.services.connection.postgres_pool_manager_facade import (
    PostgreSQLPoolManager,
)

# Re-export supporting types and configurations for backward compatibility  
from prompt_improver.database.services.connection.pool_shared_context import (
    ConnectionInfo,
    ConnectionMode,
    DatabaseConfig,
    HealthStatus,
    PoolConfiguration,
    PoolState,
)

# Export all public components for access if needed
__all__ = [
    "PostgreSQLPoolManager",
    "DatabaseConfig", 
    "PoolConfiguration",
    "ConnectionMode",
    "PoolState",
    "HealthStatus",
    "ConnectionInfo",
]