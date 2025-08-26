"""Retry Service Facade - Clean Break Implementation.

Complete replacement for the legacy retry system with zero backwards compatibility.
Provides unified interface for all retry operations using decomposed services.

Performance Target: <5ms retry decision coordination
Memory Target: <5MB for service orchestration
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prompt_improver.core.services.resilience.backoff_strategy_service import (
    get_backoff_strategy_service,
)
from prompt_improver.core.services.resilience.circuit_breaker_service import (
    create_circuit_breaker_service,
)
from prompt_improver.core.services.resilience.retry_configuration_service import (
    get_retry_configuration_service,
)
from prompt_improver.core.services.resilience.retry_orchestrator_service import (
    RetryExecutionContext,
    RetryOrchestratorService,
)
from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
from prompt_improver.shared.interfaces.protocols.core import (
    RetryConfigProtocol,
    RetryObserverProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryOperationResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    attempts_made: int = 0
    total_time_ms: float = 0.0
    error: Exception | None = None
    final_config_used: RetryConfigProtocol | None = None


class RetryServiceFacade:
    """Unified Retry Service Facade.

    Clean break replacement for legacy retry system. Coordinates all retry operations
    through decomposed, focused services with zero legacy support.
    """

    def __init__(self) -> None:
        """Initialize retry service facade with all dependencies."""
        # Initialize all services - no legacy support
        self._config_service = get_retry_configuration_service()
        self._backoff_service = get_backoff_strategy_service()
        self._circuit_breaker_service = create_circuit_breaker_service()

        self._orchestrator = RetryOrchestratorService(
            config_service=self._config_service,
            backoff_service=self._backoff_service,
            circuit_breaker_service=self._circuit_breaker_service,
            metrics_registry=get_metrics_registry()
        )

        self._observers: list[RetryObserverProtocol] = []

        logger.info("RetryServiceFacade initialized with clean architecture services")

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        *,
        domain: str | None = None,
        operation_type: str | None = None,
        custom_config: RetryConfigProtocol | None = None,
        **kwargs
    ) -> RetryOperationResult:
        """Execute operation with retry logic.

        Args:
            operation: Operation to execute with retry
            domain: Domain for configuration template (database, ml, api, etc.)
            operation_type: Specific operation type within domain
            custom_config: Optional custom retry configuration
            **kwargs: Arguments for the operation

        Returns:
            Retry operation result
        """
        try:
            # Get or create configuration
            if custom_config:
                config = custom_config
            elif domain:
                config = self._config_service.create_config(
                    domain=domain,
                    operation=operation_type,
                    **kwargs
                )
            else:
                config = self._config_service.get_default_config()

            # Execute with orchestrator
            context = RetryExecutionContext(
                operation_name=f"{domain}.{operation_type}" if domain and operation_type else "unknown",
                config=config
            )

            result = await self._orchestrator.execute_with_retry(
                operation=operation,
                context=context,
                *kwargs.get('operation_args', []),
                **kwargs.get('operation_kwargs', {})
            )

            return RetryOperationResult(
                success=True,
                result=result.result,
                attempts_made=result.attempts_made,
                total_time_ms=result.total_time_ms,
                final_config_used=result.final_config_used
            )

        except Exception as e:
            logger.exception(f"Retry execution failed: {e}")
            return RetryOperationResult(
                success=False,
                error=e,
                attempts_made=0,
                total_time_ms=0.0
            )

    def execute_with_retry_sync(
        self,
        operation: Callable[..., Any],
        *,
        domain: str | None = None,
        operation_type: str | None = None,
        custom_config: RetryConfigProtocol | None = None,
        **kwargs
    ) -> RetryOperationResult:
        """Execute operation with retry logic (synchronous).

        Args:
            operation: Synchronous operation to execute with retry
            domain: Domain for configuration template
            operation_type: Specific operation type within domain
            custom_config: Optional custom retry configuration
            **kwargs: Arguments for the operation

        Returns:
            Retry operation result
        """
        try:
            # Run async method in event loop
            return asyncio.run(
                self.execute_with_retry(
                    operation=operation,
                    domain=domain,
                    operation_type=operation_type,
                    custom_config=custom_config,
                    **kwargs
                )
            )
        except Exception as e:
            logger.exception(f"Synchronous retry execution failed: {e}")
            return RetryOperationResult(
                success=False,
                error=e
            )

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        circuit_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with circuit breaker protection.

        Args:
            operation: Operation to execute
            circuit_name: Circuit breaker identifier
            *args, **kwargs: Operation arguments

        Returns:
            Operation result
        """
        return await self._circuit_breaker_service.execute_with_circuit_breaker(
            operation=operation,
            circuit_name=circuit_name,
            *args,
            **kwargs
        )

    def add_observer(self, observer: RetryObserverProtocol) -> None:
        """Add retry observer.

        Args:
            observer: Observer to add for retry events
        """
        self._observers.append(observer)
        self._orchestrator.add_observer(observer)
        logger.debug(f"Added retry observer: {type(observer).__name__}")

    def remove_observer(self, observer: RetryObserverProtocol) -> None:
        """Remove retry observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
            self._orchestrator.remove_observer(observer)
            logger.debug(f"Removed retry observer: {type(observer).__name__}")

    def create_domain_config(
        self,
        domain: str,
        **overrides
    ) -> RetryConfigProtocol:
        """Create domain-specific retry configuration.

        Args:
            domain: Domain name (database, ml, api, etc.)
            **overrides: Configuration overrides

        Returns:
            Domain-specific retry configuration
        """
        return self._config_service.create_config(domain=domain, **overrides)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of retry system.

        Returns:
            Health status information
        """
        try:
            circuit_states = self._circuit_breaker_service.get_all_circuit_states()
            orchestrator_health = self._orchestrator.get_health_status()

            return {
                "overall_status": "healthy",
                "services": {
                    "configuration": {"status": "healthy"},
                    "backoff_strategy": {"status": "healthy"},
                    "circuit_breakers": {
                        "status": "healthy" if not any(
                            state.get("is_open", False) for state in circuit_states.values()
                        ) else "degraded",
                        "circuits": circuit_states
                    },
                    "orchestrator": orchestrator_health
                },
                "observers_count": len(self._observers),
                "facade_info": {
                    "implementation": "clean_architecture_2025",
                    "legacy_support": False,
                    "performance_target_ms": 5
                }
            }
        except Exception as e:
            logger.exception(f"Health status check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e)
            }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns:
            Performance metrics from all services
        """
        return {
            "facade_metrics": self._get_facade_metrics(),
            "orchestrator_metrics": self._orchestrator.get_comprehensive_metrics(),
            "service_health": self.get_health_status()
        }

    def _get_facade_metrics(self) -> dict[str, Any]:
        """Get facade-level metrics.

        Returns:
            Facade metrics
        """
        metrics_registry = get_metrics_registry()

        return {
            "total_operations": metrics_registry.get_counter_value(
                "retry_facade_operations_total"
            ) or 0,
            "successful_operations": metrics_registry.get_counter_value(
                "retry_facade_operations_total",
                tags={"result": "success"}
            ) or 0,
            "failed_operations": metrics_registry.get_counter_value(
                "retry_facade_operations_total",
                tags={"result": "error"}
            ) or 0,
            "observers_count": len(self._observers)
        }


# Global service instance - Clean Break Pattern
_retry_service_instance: RetryServiceFacade | None = None


def get_retry_service() -> RetryServiceFacade:
    """Get global retry service instance.

    Clean break replacement for legacy retry manager.
    No backwards compatibility - pure 2025 architecture.

    Returns:
        Retry service facade instance
    """
    global _retry_service_instance

    if _retry_service_instance is None:
        _retry_service_instance = RetryServiceFacade()
        logger.info("Initialized global retry service with clean architecture")

    return _retry_service_instance


def reset_retry_service() -> None:
    """Reset global retry service instance.

    Used for testing and clean initialization.
    """
    global _retry_service_instance
    _retry_service_instance = None
    logger.info("Reset global retry service instance")


# Clean Break Migration Helpers
def migrate_from_legacy() -> dict[str, str]:
    """Provide migration guidance from legacy retry system.

    Returns:
        Migration mapping from old to new patterns
    """
    return {
        "old_pattern": "from prompt_improver.core.retry_manager import get_retry_manager",
        "new_pattern": "from prompt_improver.core.services.resilience.retry_service_facade import get_retry_service",

        "old_usage": "retry_manager.execute_with_retry(operation, config)",
        "new_usage": "retry_service.execute_with_retry(operation, domain='database', operation_type='query')",

        "old_decorator": "@retry(max_attempts=3, backoff_strategy='exponential')",
        "new_decorator": "await retry_service.execute_with_retry(operation, domain='api', max_attempts=3)",

        "configuration": "Use domain-specific templates instead of manual config creation",
        "circuit_breakers": "Built-in circuit breaker protection, no separate setup needed",
        "observability": "Enhanced metrics and tracing built-in, observers for custom monitoring"
    }


# Performance Validation
async def validate_performance_targets() -> dict[str, Any]:
    """Validate performance targets for retry service.

    Returns:
        Performance validation results
    """
    import time

    retry_service = get_retry_service()

    async def test_operation() -> str:
        return "success"

    # Test retry decision time
    start_time = time.perf_counter()

    result = await retry_service.execute_with_retry(
        test_operation,
        domain="test",
        operation_type="validation"
    )

    decision_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "decision_time_ms": decision_time_ms,
        "target_ms": 5.0,
        "performance_met": decision_time_ms < 5.0,
        "result_success": result.success,
        "architecture": "clean_2025",
        "legacy_eliminated": True
    }
