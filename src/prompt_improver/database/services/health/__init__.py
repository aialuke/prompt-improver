"""Health monitoring services for database connections and components.

This package provides comprehensive health monitoring functionality extracted from
unified_connection_manager.py, implementing:

- HealthManager: Unified health monitoring with multi-component support
- HealthChecker: Base interface for component-specific health checks
- CircuitBreaker: Fault tolerance patterns with automatic recovery
- HealthResult: Standardized result types for health status reporting
- Background monitoring with configurable intervals and thresholds

Designed for production-ready health monitoring with sub-10ms response times.
"""

from prompt_improver.database.services.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)
from prompt_improver.database.services.health.health_checker import (
    HealthChecker,
    HealthResult,
    HealthStatus,
)
from prompt_improver.database.services.health.health_manager import (
    HealthManager,
    HealthManagerConfig,
    create_health_manager,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "HealthChecker",
    "HealthManager",
    "HealthManagerConfig",
    "HealthResult",
    "HealthStatus",
    "create_health_manager",
]
