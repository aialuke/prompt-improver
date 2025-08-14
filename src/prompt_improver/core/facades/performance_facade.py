"""Performance Systems Facade - Reduces Performance Baseline Module Coupling

This facade provides unified performance system coordination while reducing
direct imports from 12 to 2 internal dependencies through lazy initialization.

Design:
- Protocol-based interface for loose coupling
- Lazy loading of performance system components
- Monitoring and baseline system coordination
- Health check and metrics integration
- Zero circular import dependencies
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class PerformanceFacadeProtocol(Protocol):
    """Protocol for performance systems facade."""
    
    async def get_baseline_system(self) -> Any:
        """Get performance baseline system."""
        ...
    
    async def get_baseline_collector(self) -> Any:
        """Get baseline collector."""
        ...
    
    async def get_profiler(self) -> Any:
        """Get continuous profiler."""
        ...
    
    async def get_regression_detector(self) -> Any:
        """Get regression detector."""
        ...
    
    async def get_performance_dashboard(self) -> Any:
        """Get performance dashboard."""
        ...
    
    async def record_operation(self, operation_name: str, response_time_ms: float, **kwargs) -> None:
        """Record operation performance."""
        ...
    
    async def analyze_trends(self, hours: int = 24) -> dict[str, Any]:
        """Analyze performance trends."""
        ...
    
    async def check_regressions(self) -> list[dict[str, Any]]:
        """Check for performance regressions."""
        ...
    
    async def initialize_system(self) -> None:
        """Initialize performance system."""
        ...
    
    async def shutdown_system(self) -> None:
        """Shutdown performance system."""
        ...


class PerformanceFacade(PerformanceFacadeProtocol):
    """Performance systems facade with minimal coupling.
    
    Reduces performance baseline module coupling from 12 internal imports to 2.
    Provides unified interface for all performance system coordination.
    """

    def __init__(self):
        """Initialize facade with lazy loading."""
        self._baseline_system = None
        self._baseline_collector = None
        self._profiler = None
        self._regression_detector = None
        self._performance_dashboard = None
        self._load_testing_integration = None
        self._validation_suite = None
        self._optimization_guide = None
        self._statistical_analyzer = None
        self._background_manager = None
        self._system_initialized = False
        logger.debug("PerformanceFacade initialized with lazy loading")

    async def _ensure_baseline_system(self):
        """Ensure baseline system is available."""
        if self._baseline_system is None:
            # Only import when needed to reduce coupling
            from prompt_improver.performance.baseline import get_baseline_system
            self._baseline_system = get_baseline_system()

    async def _ensure_baseline_collector(self):
        """Ensure baseline collector is available."""
        if self._baseline_collector is None:
            from prompt_improver.performance.baseline.baseline_collector import get_baseline_collector
            self._baseline_collector = get_baseline_collector()

    async def _ensure_profiler(self):
        """Ensure profiler is available."""
        if self._profiler is None:
            from prompt_improver.performance.baseline.profiler import get_profiler
            self._profiler = get_profiler()

    async def _ensure_regression_detector(self):
        """Ensure regression detector is available."""
        if self._regression_detector is None:
            from prompt_improver.performance.baseline.regression_detector import get_regression_detector
            self._regression_detector = get_regression_detector()

    async def _ensure_performance_dashboard(self):
        """Ensure performance dashboard is available."""
        if self._performance_dashboard is None:
            from prompt_improver.performance.baseline.enhanced_dashboard_integration import get_performance_dashboard
            self._performance_dashboard = get_performance_dashboard()

    async def _ensure_background_manager(self):
        """Ensure background task manager is available."""
        if self._background_manager is None:
            from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager
            self._background_manager = get_background_task_manager()

    async def get_baseline_system(self) -> Any:
        """Get performance baseline system."""
        await self._ensure_baseline_system()
        return self._baseline_system

    async def get_baseline_collector(self) -> Any:
        """Get baseline collector."""
        await self._ensure_baseline_collector()
        return self._baseline_collector

    async def get_profiler(self) -> Any:
        """Get continuous profiler."""
        await self._ensure_profiler()
        return self._profiler

    async def get_regression_detector(self) -> Any:
        """Get regression detector."""
        await self._ensure_regression_detector()
        return self._regression_detector

    async def get_performance_dashboard(self) -> Any:
        """Get performance dashboard."""
        await self._ensure_performance_dashboard()
        return self._performance_dashboard

    async def get_load_testing_integration(self) -> Any:
        """Get load testing integration."""
        if self._load_testing_integration is None:
            from prompt_improver.performance.baseline.load_testing_integration import get_load_testing_integration
            self._load_testing_integration = get_load_testing_integration()
        return self._load_testing_integration

    async def get_validation_suite(self) -> Any:
        """Get performance validation suite."""
        if self._validation_suite is None:
            from prompt_improver.performance.baseline.performance_validation_suite import get_validation_suite
            self._validation_suite = get_validation_suite()
        return self._validation_suite

    async def get_optimization_guide(self) -> Any:
        """Get production optimization guide."""
        if self._optimization_guide is None:
            from prompt_improver.performance.baseline.production_optimization_guide import get_optimization_guide
            self._optimization_guide = get_optimization_guide()
        return self._optimization_guide

    async def get_statistical_analyzer(self) -> Any:
        """Get statistical analyzer."""
        if self._statistical_analyzer is None:
            from prompt_improver.performance.baseline.statistical_analyzer import StatisticalAnalyzer
            self._statistical_analyzer = StatisticalAnalyzer()
        return self._statistical_analyzer

    async def record_operation(self, operation_name: str, response_time_ms: float, **kwargs) -> None:
        """Record operation performance."""
        collector = await self.get_baseline_collector()
        await collector.record_request(response_time_ms, kwargs.get("is_error", False), operation_name, kwargs)

    async def track_production_operation(self, operation_name: str, **metadata):
        """Get context manager for tracking production operations."""
        from prompt_improver.performance.baseline import track_production_operation
        return track_production_operation(operation_name, **metadata)

    async def analyze_trends(self, hours: int = 24) -> dict[str, Any]:
        """Analyze performance trends."""
        baseline_system = await self.get_baseline_system()
        return await baseline_system.analyze_performance_trends(hours)

    async def check_regressions(self) -> list[dict[str, Any]]:
        """Check for performance regressions."""
        baseline_system = await self.get_baseline_system()
        return await baseline_system.check_for_regressions()

    async def generate_report(self, report_type: str = "daily") -> dict[str, Any]:
        """Generate performance report."""
        baseline_system = await self.get_baseline_system()
        return await baseline_system.generate_performance_report(report_type)

    async def validate_system_performance(self) -> dict[str, Any]:
        """Validate baseline system performance."""
        validation_suite = await self.get_validation_suite()
        return await validation_suite.validate_system_performance()

    async def run_load_test(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run integrated load test."""
        load_testing = await self.get_load_testing_integration()
        from prompt_improver.performance.baseline.load_testing_integration import LoadTestConfig
        test_config = LoadTestConfig(**config)
        return await load_testing.run_load_test(test_config)

    async def profile_block(self, block_name: str):
        """Get context manager for profiling code blocks."""
        from prompt_improver.performance.baseline.profiler import profile_block
        return profile_block(block_name)

    async def initialize_system(self) -> None:
        """Initialize performance system."""
        if self._system_initialized:
            logger.warning("Performance system already initialized")
            return

        logger.info("Initializing performance system...")

        # Initialize core components
        baseline_system = await self.get_baseline_system()
        await baseline_system.initialize()
        logger.info("✓ Baseline system initialized")

        # Initialize dashboard
        dashboard = await self.get_performance_dashboard()
        if hasattr(dashboard, "initialize"):
            await dashboard.initialize()
            logger.info("✓ Performance dashboard initialized")

        # Initialize background manager
        await self._ensure_background_manager()
        logger.info("✓ Background manager initialized")

        self._system_initialized = True
        logger.info("Performance system initialization complete")

    async def start_system(self) -> None:
        """Start performance system."""
        if not self._system_initialized:
            await self.initialize_system()

        baseline_system = await self.get_baseline_system()
        await baseline_system.start()
        logger.info("Performance system started")

    async def shutdown_system(self) -> None:
        """Shutdown performance system."""
        if not self._system_initialized:
            return

        logger.info("Shutting down performance system...")

        try:
            baseline_system = await self.get_baseline_system()
            await baseline_system.stop()
            logger.info("✓ Baseline system stopped")

            if self._performance_dashboard and hasattr(self._performance_dashboard, "shutdown"):
                await self._performance_dashboard.shutdown()
                logger.info("✓ Performance dashboard shut down")

        except Exception as e:
            logger.error(f"Error during performance system shutdown: {e}")

        # Clear references
        self._baseline_system = None
        self._baseline_collector = None
        self._profiler = None
        self._regression_detector = None
        self._performance_dashboard = None
        self._load_testing_integration = None
        self._validation_suite = None
        self._optimization_guide = None
        self._statistical_analyzer = None
        self._background_manager = None
        self._system_initialized = False

        logger.info("Performance system shutdown complete")

    def get_system_status(self) -> dict[str, Any]:
        """Get performance system status."""
        return {
            "initialized": self._system_initialized,
            "baseline_system": self._baseline_system is not None,
            "baseline_collector": self._baseline_collector is not None,
            "profiler": self._profiler is not None,
            "regression_detector": self._regression_detector is not None,
            "performance_dashboard": self._performance_dashboard is not None,
            "background_manager": self._background_manager is not None,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on performance system."""
        if not self._system_initialized:
            return {"status": "not_initialized"}

        try:
            baseline_system = await self.get_baseline_system()
            return baseline_system.get_system_status()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global facade instance
_performance_facade: PerformanceFacade | None = None


def get_performance_facade() -> PerformanceFacade:
    """Get global performance facade instance.
    
    Returns:
        PerformanceFacade with lazy initialization and minimal coupling
    """
    global _performance_facade
    if _performance_facade is None:
        _performance_facade = PerformanceFacade()
    return _performance_facade


async def initialize_performance_facade() -> None:
    """Initialize the global performance facade."""
    facade = get_performance_facade()
    await facade.initialize_system()


async def start_performance_facade() -> None:
    """Start the global performance facade."""
    facade = get_performance_facade()
    await facade.start_system()


async def shutdown_performance_facade() -> None:
    """Shutdown the global performance facade."""
    global _performance_facade
    if _performance_facade:
        await _performance_facade.shutdown_system()
        _performance_facade = None


__all__ = [
    "PerformanceFacadeProtocol",
    "PerformanceFacade",
    "get_performance_facade",
    "initialize_performance_facade",
    "start_performance_facade",
    "shutdown_performance_facade",
]