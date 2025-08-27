"""Performance Benchmark Factory - 2025 Unified Service Pattern.

Factory functions for performance benchmarking with service locator pattern.
Follows the established architecture from analytics_factory.py to eliminate
circular dependencies while maintaining clean dependency injection.
"""

import logging
from typing import Any

from prompt_improver.performance.monitoring.performance_service_locator import (
    PerformanceServiceLocator,
    create_performance_service_locator,
)

logger = logging.getLogger(__name__)


async def create_performance_benchmark(
    service_locator: PerformanceServiceLocator | None = None,
    config: dict[str, Any] | None = None
) -> Any:
    """Create performance benchmark with service locator dependencies.

    Modern pattern: Uses service locator for dependency injection to avoid
    circular imports while maintaining full functionality.

    Args:
        service_locator: Configured service locator instance
        config: Optional configuration dictionary

    Returns:
        MCPPerformanceBenchmark instance with injected dependencies
    """
    try:
        from prompt_improver.performance.monitoring.performance_benchmark_enhanced import (
            MCPPerformanceBenchmarkEnhanced,
        )

        if service_locator is None:
            # Create default service locator with no-op implementations
            service_locator = create_performance_service_locator(config=config)

        return MCPPerformanceBenchmarkEnhanced(service_locator=service_locator)

    except ImportError as e:
        logger.exception(f"Failed to import enhanced performance benchmark: {e}")
        return None


async def create_performance_benchmark_with_dependencies(
    database_session_factory: Any | None = None,
    prompt_service: Any | None = None,
    config: dict[str, Any] | None = None,
    event_bus_getter: Any | None = None,
    session_store: Any | None = None
) -> Any:
    """Create performance benchmark with explicit dependencies.

    This function follows the analytics factory pattern and creates a fully
    configured performance benchmark with all required dependencies.

    Args:
        database_session_factory: Function to get database sessions
        prompt_service: Prompt improvement service instance
        config: Configuration dictionary
        event_bus_getter: Function to get event bus
        session_store: Session store instance

    Returns:
        Configured performance benchmark instance
    """
    try:
        # Create and configure service locator
        service_locator = create_performance_service_locator(
            database_session_factory=database_session_factory,
            prompt_service=prompt_service,
            config=config,
            event_bus_getter=event_bus_getter,
            session_store=session_store
        )

        # Create performance benchmark with configured locator
        return await create_performance_benchmark(
            service_locator=service_locator,
            config=config
        )

    except Exception as e:
        logger.exception(f"Failed to create performance benchmark with dependencies: {e}")
        return None


async def create_performance_benchmark_from_container(container: Any) -> Any:
    """Create performance benchmark using DI container dependencies.

    This function bridges the gap between the DI container and the service
    locator pattern, allowing the monitoring container to provide dependencies
    without creating circular imports.

    Args:
        container: DI container instance (MonitoringContainer or similar)

    Returns:
        Configured performance benchmark instance
    """
    try:
        # Extract dependencies from container using safe methods
        database_session_factory = None
        prompt_service = None
        config = {}
        event_bus_getter = None
        session_store = None

        # Try to get database session factory
        try:
            if hasattr(container, 'get_database_session_factory'):
                database_session_factory = await container.get_database_session_factory()
            else:
                # Fallback to importing database session directly
                from prompt_improver.database import get_session
                database_session_factory = get_session
        except Exception as e:
            logger.warning(f"Could not get database session factory: {e}")

        # Try to get prompt service
        try:
            if hasattr(container, 'get_prompt_service'):
                prompt_service = await container.get_prompt_service()
            else:
                # Fallback to importing prompt service facade
                from prompt_improver.services.prompt.facade import PromptServiceFacade
                # Create with minimal dependencies for benchmarking
                prompt_service = PromptServiceFacade()
        except Exception as e:
            logger.warning(f"Could not get prompt service: {e}")

        # Try to get configuration
        try:
            if hasattr(container, 'get_config'):
                config = await container.get_config()
            elif hasattr(container, 'get_configuration'):
                config = await container.get_configuration()
        except Exception as e:
            logger.warning(f"Could not get configuration: {e}")

        # Try to get event bus getter
        try:
            from prompt_improver.core.events.ml_event_bus import get_ml_event_bus
            event_bus_getter = get_ml_event_bus
        except Exception as e:
            logger.warning(f"Could not get ML event bus: {e}")

        # Try to get session store (this might not be available in monitoring container)
        try:
            if hasattr(container, 'get_session_store'):
                session_store = await container.get_session_store()
        except Exception as e:
            logger.debug(f"Session store not available in container: {e}")

        # Create benchmark with available dependencies
        return await create_performance_benchmark_with_dependencies(
            database_session_factory=database_session_factory,
            prompt_service=prompt_service,
            config=config,
            event_bus_getter=event_bus_getter,
            session_store=session_store
        )

    except Exception as e:
        logger.exception(f"Failed to create performance benchmark from container: {e}")
        # Return no-op implementation as fallback
        return await create_performance_benchmark()


# Compatibility functions for existing code

async def get_performance_optimizer():
    """Get performance optimizer instance.

    This is a compatibility function for existing code that uses
    get_performance_optimizer() directly.
    """
    try:
        from prompt_improver.performance.optimization.performance_optimizer import (
            get_performance_optimizer,
        )
        return get_performance_optimizer()
    except ImportError:
        logger.warning("Performance optimizer not available, using no-op implementation")

        class NoOpPerformanceOptimizer:
            async def run_performance_benchmark(self, name, operation, sample_count):
                # Return mock baseline
                class MockBaseline:
                    def __init__(self) -> None:
                        self.avg_duration_ms = 10.0
                        self.p95_duration_ms = 15.0
                        self.p99_duration_ms = 20.0
                        self.success_rate = 1.0
                        self.sample_count = sample_count

                    def meets_target(self, target_ms):
                        return self.avg_duration_ms < target_ms

                    def model_dump(self):
                        return {
                            "avg_duration_ms": self.avg_duration_ms,
                            "p95_duration_ms": self.p95_duration_ms,
                            "p99_duration_ms": self.p99_duration_ms,
                            "success_rate": self.success_rate,
                            "sample_count": self.sample_count,
                            "meets_200ms_target": self.meets_target(200)
                        }

                return MockBaseline()

            async def get_all_baselines(self):
                return {}

            @property
            def _measurements(self):
                return {}

        return NoOpPerformanceOptimizer()
