"""Unified Performance System with Facade Pattern - Reduced Coupling Implementation.

This is the modernized version of performance/baseline/__init__.py that uses facade patterns
to reduce coupling from 12 to 2 internal imports while maintaining full functionality.

Key improvements:
- 83% reduction in internal imports (12 → 2)
- Facade-based performance system coordination
- Protocol-based interfaces for loose coupling
- Streamlined monitoring and baseline management
- Zero circular import possibilities
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from prompt_improver.core.facades import get_performance_facade

if TYPE_CHECKING:
    from prompt_improver.shared.interfaces.protocols.monitoring import (
        PerformanceFacadeProtocol,
    )

logger = logging.getLogger(__name__)


class UnifiedPerformanceManager:
    """Unified performance manager using facade pattern for loose coupling.

    This manager provides the same interface as the original performance baseline system
    but with dramatically reduced coupling through facade patterns.

    Coupling reduction: 12 → 2 internal imports (83% reduction)
    """

    def __init__(self) -> None:
        """Initialize the unified performance manager."""
        self._performance_facade: PerformanceFacadeProtocol = get_performance_facade()
        self._system_initialized = False
        self._system_running = False
        logger.debug("UnifiedPerformanceManager initialized with facade pattern")

    async def initialize_system(self) -> None:
        """Initialize performance system through facade."""
        if self._system_initialized:
            return

        await self._performance_facade.initialize_system()
        self._system_initialized = True
        logger.info("UnifiedPerformanceManager system initialization complete")

    async def start_system(self) -> None:
        """Start performance system through facade."""
        if not self._system_initialized:
            await self.initialize_system()

        if self._system_running:
            return

        await self._performance_facade.start_system()
        self._system_running = True
        logger.info("UnifiedPerformanceManager system started")

    async def shutdown_system(self) -> None:
        """Shutdown performance system through facade."""
        if not self._system_running:
            return

        await self._performance_facade.shutdown_system()
        self._system_running = False
        self._system_initialized = False
        logger.info("UnifiedPerformanceManager system shutdown complete")

    async def get_baseline_system(self) -> Any:
        """Get baseline system through facade."""
        return await self._performance_facade.get_baseline_system()

    async def get_baseline_collector(self) -> Any:
        """Get baseline collector through facade."""
        return await self._performance_facade.get_baseline_collector()

    async def get_profiler(self) -> Any:
        """Get profiler through facade."""
        return await self._performance_facade.get_profiler()

    async def get_regression_detector(self) -> Any:
        """Get regression detector through facade."""
        return await self._performance_facade.get_regression_detector()

    async def get_performance_dashboard(self) -> Any:
        """Get performance dashboard through facade."""
        return await self._performance_facade.get_performance_dashboard()

    async def record_operation(self, operation_name: str, response_time_ms: float, **kwargs) -> None:
        """Record operation performance through facade."""
        await self._performance_facade.record_operation(operation_name, response_time_ms, **kwargs)

    async def track_production_operation(self, operation_name: str, **metadata):
        """Get context manager for tracking production operations through facade."""
        return await self._performance_facade.track_production_operation(operation_name, **metadata)

    async def analyze_performance_trends(self, hours: int = 24) -> dict[str, Any]:
        """Analyze performance trends through facade."""
        return await self._performance_facade.analyze_trends(hours)

    async def check_for_regressions(self) -> list[dict[str, Any]]:
        """Check for performance regressions through facade."""
        return await self._performance_facade.check_regressions()

    async def generate_performance_report(self, report_type: str = "daily") -> dict[str, Any]:
        """Generate performance report through facade."""
        return await self._performance_facade.generate_report(report_type)

    async def validate_system_performance(self) -> dict[str, Any]:
        """Validate system performance through facade."""
        return await self._performance_facade.validate_system_performance()

    async def run_load_test(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run integrated load test through facade."""
        return await self._performance_facade.run_load_test(config)

    async def profile_block(self, block_name: str):
        """Get context manager for profiling code blocks through facade."""
        return await self._performance_facade.profile_block(block_name)

    def get_system_status(self) -> dict[str, Any]:
        """Get performance system status through facade."""
        status = self._performance_facade.get_system_status()
        status.update({
            "manager_initialized": self._system_initialized,
            "manager_running": self._system_running,
            "facade_type": type(self._performance_facade).__name__,
        })
        return status

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on performance system through facade."""
        return await self._performance_facade.health_check()


class track_production_operation:
    """Context manager for tracking production operations with facade pattern."""

    def __init__(self, operation_name: str, **metadata) -> None:
        """Initialize operation tracker."""
        self.operation_name = operation_name
        self.metadata = metadata
        self.start_time = None
        self._manager = get_performance_manager()

    def __enter__(self):
        """Enter tracking context."""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit tracking context and record operation."""
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            is_error = exc_type is not None

            # Submit tracking task
            asyncio.create_task(
                self._submit_tracking_task(duration_ms, is_error)
            )

    async def _submit_tracking_task(self, duration_ms: float, is_error: bool) -> None:
        """Submit tracking task through facade."""
        await self._manager.record_operation(
            self.operation_name, duration_ms, is_error=is_error, **self.metadata
        )


# Global performance manager instance
_performance_manager: UnifiedPerformanceManager | None = None


def get_performance_manager() -> UnifiedPerformanceManager:
    """Get the global unified performance manager instance.

    Returns:
        UnifiedPerformanceManager: Global performance manager with facade pattern
    """
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = UnifiedPerformanceManager()
    return _performance_manager


async def initialize_performance_manager() -> None:
    """Initialize the global performance manager."""
    manager = get_performance_manager()
    await manager.initialize_system()


async def start_performance_manager() -> None:
    """Start the global performance manager."""
    manager = get_performance_manager()
    await manager.start_system()


async def shutdown_performance_manager() -> None:
    """Shutdown the global performance manager."""
    global _performance_manager
    if _performance_manager:
        await _performance_manager.shutdown_system()
        _performance_manager = None


# Convenience functions with facade pattern
async def initialize_baseline_system(**kwargs) -> None:
    """Initialize the performance baseline system."""
    await initialize_performance_manager()


async def start_baseline_system() -> None:
    """Start the performance baseline system."""
    await start_performance_manager()


async def stop_baseline_system() -> None:
    """Stop the performance baseline system."""
    await shutdown_performance_manager()


def is_baseline_system_running() -> bool:
    """Check if the baseline system is running."""
    return get_performance_manager()._system_running


async def record_production_request(
    operation_name: str, response_time_ms: float, is_error: bool = False, **kwargs
) -> None:
    """Record a production request for baseline analysis."""
    await get_performance_manager().record_operation(
        operation_name, response_time_ms, is_error=is_error, **kwargs
    )


async def get_baseline_system() -> Any:
    """Get baseline system."""
    return await get_performance_manager().get_baseline_system()


async def get_baseline_collector() -> Any:
    """Get baseline collector."""
    return await get_performance_manager().get_baseline_collector()


async def get_profiler() -> Any:
    """Get profiler."""
    return await get_performance_manager().get_profiler()


async def get_regression_detector() -> Any:
    """Get regression detector."""
    return await get_performance_manager().get_regression_detector()


async def get_performance_dashboard() -> Any:
    """Get performance dashboard."""
    return await get_performance_manager().get_performance_dashboard()


async def analyze_performance_trends(hours: int = 24) -> dict[str, Any]:
    """Analyze performance trends."""
    return await get_performance_manager().analyze_performance_trends(hours)


async def check_for_regressions() -> list[dict[str, Any]]:
    """Check for performance regressions."""
    return await get_performance_manager().check_for_regressions()


async def generate_performance_report(report_type: str = "daily") -> dict[str, Any]:
    """Generate performance report."""
    return await get_performance_manager().generate_performance_report(report_type)


def get_system_status() -> dict[str, Any]:
    """Get performance system status."""
    return get_performance_manager().get_system_status()


async def health_check() -> dict[str, Any]:
    """Perform health check on performance system."""
    return await get_performance_manager().health_check()


__all__ = [
    # Manager class
    "UnifiedPerformanceManager",
    "analyze_performance_trends",
    "check_for_regressions",
    "generate_performance_report",
    "get_baseline_collector",
    "get_baseline_system",
    "get_performance_dashboard",
    "get_performance_manager",
    "get_profiler",
    "get_regression_detector",
    "get_system_status",
    "health_check",
    # Convenience functions
    "initialize_baseline_system",
    "initialize_performance_manager",
    "is_baseline_system_running",
    "record_production_request",
    "shutdown_performance_manager",
    "start_baseline_system",
    "start_performance_manager",
    "stop_baseline_system",
    # Context manager
    "track_production_operation",
]
