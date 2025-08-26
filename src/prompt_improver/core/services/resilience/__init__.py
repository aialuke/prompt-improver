"""Resilience Services Package.

Centralized resilience and retry configuration management following clean architecture principles.

This package provides:
- RetryConfigurationService for centralized retry configuration management
- BackoffStrategyService for optimized delay calculations (<1ms performance)
- CircuitBreakerService for state management and protection
- RetryOrchestratorService for coordination of all retry operations (<10ms decisions)
- Domain-specific retry templates (database, ML, API operations)
- Performance-optimized configuration caching (<1ms lookups)
- Runtime configuration updates and composition
- Integration with existing retry patterns

2025 Architecture Standards:
- Protocol-based interfaces for dependency injection
- Elimination of redundant configuration classes
- Template-based approach for common scenarios
- Clean separation of concerns
- Service coordination with comprehensive observability
"""

from prompt_improver.core.services.resilience.backoff_strategy_service import (
    BackoffStrategyProtocol,
    BackoffStrategyService,
    StrategyMetrics,
    calculate_delay,
    get_backoff_strategy_service,
)
from prompt_improver.core.services.resilience.circuit_breaker_service import (
    CallRecord,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerService,
    CircuitState,
    create_circuit_breaker_service,
)
from prompt_improver.core.services.resilience.retry_configuration_service import (
    ConfigTemplate,
    RetryConfigurationProtocol,
    RetryConfigurationService,
    get_retry_configuration_service,
)
from prompt_improver.core.services.resilience.retry_orchestrator_service import (
    RetryExecutionContext,
    RetryOrchestratorProtocol,
    RetryOrchestratorService,
    get_retry_orchestrator_service,
    set_retry_orchestrator_service,
)

__all__ = [
    "BackoffStrategyProtocol",
    "BackoffStrategyService",
    "CallRecord",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreakerService",
    "CircuitState",
    "ConfigTemplate",
    "RetryConfigurationProtocol",
    "RetryConfigurationService",
    "RetryExecutionContext",
    "RetryOrchestratorProtocol",
    "RetryOrchestratorService",
    "StrategyMetrics",
    "calculate_delay",
    "create_circuit_breaker_service",
    "get_backoff_strategy_service",
    "get_retry_configuration_service",
    "get_retry_orchestrator_service",
    "set_retry_orchestrator_service",
]
