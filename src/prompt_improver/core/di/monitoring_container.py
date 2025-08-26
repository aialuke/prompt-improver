"""Monitoring Services Dependency Injection Container (2025).

Specialized DI container for monitoring services including health checks,
metrics collection, observability, and performance monitoring services.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from prompt_improver.core.di.protocols import (
    ContainerRegistryProtocol,
    MonitoringContainerProtocol,
)
from prompt_improver.shared.interfaces.ab_testing import IABTestingService
from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class MonitoringServiceRegistration:
    """Monitoring service registration information."""
    interface: type[Any]
    implementation: type[Any] | None
    lifetime: ServiceLifetime
    factory: Callable[[], Any] | None = None
    initialized: bool = False
    instance: Any = None
    tags: set[str] = field(default_factory=set)
    health_check: Callable[[], Any] | None = None


class MonitoringContainer(ContainerRegistryProtocol, MonitoringContainerProtocol):
    """Specialized DI container for monitoring services.

    Manages monitoring and observability services including:
    - Health check systems and unified health monitoring
    - Metrics collection and registry (OpenTelemetry)
    - A/B testing and experimentation services
    - Performance monitoring and benchmarking
    - Alerting and notification systems
    - Distributed tracing and logging

    Follows clean architecture with protocol-based dependencies.
    """

    def __init__(self, name: str = "monitoring") -> None:
        """Initialize monitoring services container.

        Args:
            name: Container identifier for logging
        """
        self.name = name
        self.logger = logger.getChild(f"container.{name}")
        self._services: dict[type[Any], MonitoringServiceRegistration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_order: list[type[Any]] = []
        self._register_default_services()
        self.logger.debug(f"Monitoring container '{self.name}' initialized")

    def _register_default_services(self) -> None:
        """Register default monitoring services."""
        # Self-registration for dependency injection
        self.register_instance(MonitoringContainer, self, tags={"container", "monitoring"})

        # Metrics registry factory (OpenTelemetry)
        self.register_metrics_collector_factory()

        # Health monitor factory
        self.register_health_monitor_factory()

        # A/B testing service factory
        self.register_ab_testing_service_factory()

        # Performance monitoring factory
        self.register_performance_monitoring_factory()

        # Alert manager factory
        self.register_alert_manager_factory()

        # Tracing service factory
        self.register_tracing_service_factory()

        # Observability dashboard factory
        self.register_observability_dashboard_factory()

    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T],
        tags: set[str] | None = None,
    ) -> None:
        """Register a singleton service.

        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = MonitoringServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or set(),
        )
        self.logger.debug(
            f"Registered singleton: {interface.__name__} -> {implementation.__name__}"
        )

    def register_transient(
        self,
        interface: type[T],
        implementation_or_factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a transient service.

        Args:
            interface: Service interface/protocol
            implementation_or_factory: Implementation class or factory
            tags: Optional tags for service categorization
        """
        self._services[interface] = MonitoringServiceRegistration(
            interface=interface,
            implementation=implementation_or_factory if not callable(implementation_or_factory) else None,
            factory=implementation_or_factory if callable(implementation_or_factory) else None,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set(),
        )
        self.logger.debug(f"Registered transient: {interface.__name__}")

    def register_factory(
        self,
        interface: type[T],
        factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a service factory.

        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
            tags: Optional tags for service categorization
        """
        self._services[interface] = MonitoringServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory,
            tags=tags or set(),
        )
        interface_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
        self.logger.debug(f"Registered factory: {interface_name}")

    def register_instance(
        self,
        interface: type[T],
        instance: T,
        tags: set[str] | None = None,
    ) -> None:
        """Register a pre-created service instance.

        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
            tags: Optional tags for service categorization
        """
        registration = MonitoringServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            tags=tags or set(),
        )
        self._services[interface] = registration
        self.logger.debug(f"Registered instance: {interface.__name__}")

    async def get(self, interface: type[T]) -> T:
        """Resolve service instance.

        Args:
            interface: Service interface to resolve

        Returns:
            Service instance

        Raises:
            KeyError: If service is not registered
        """
        async with self._lock:
            return await self._resolve_service(interface)

    async def _resolve_service(self, interface: type[T]) -> T:
        """Internal service resolution with lifecycle management.

        Args:
            interface: Service interface to resolve

        Returns:
            Service instance
        """
        if interface not in self._services:
            raise KeyError(f"Monitoring service not registered: {interface.__name__}")

        registration = self._services[interface]

        # Return existing singleton instance
        if (registration.lifetime == ServiceLifetime.SINGLETON and
            registration.initialized and registration.instance is not None):
            return registration.instance

        # Create new instance
        if registration.factory:
            instance = await self._create_from_factory(registration.factory)
        elif registration.implementation:
            instance = await self._create_from_class(registration.implementation)
        else:
            raise ValueError(f"No factory or implementation for {interface.__name__}")

        # Initialize if needed
        if hasattr(instance, "initialize") and asyncio.iscoroutinefunction(instance.initialize):
            await instance.initialize()

        # Store singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            registration.instance = instance
            registration.initialized = True
            self._initialization_order.append(interface)

        self.logger.debug(f"Resolved monitoring service: {interface.__name__}")
        return instance

    async def _create_from_factory(self, factory: Callable[[], Any]) -> Any:
        """Create service instance from factory.

        Args:
            factory: Factory function

        Returns:
            Service instance
        """
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        return factory()

    async def _create_from_class(self, implementation: type[Any]) -> Any:
        """Create service instance from class constructor.

        Args:
            implementation: Implementation class

        Returns:
            Service instance
        """
        import inspect

        sig = inspect.signature(implementation.__init__)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = await self._resolve_service(param.annotation)
                    kwargs[param_name] = dependency
                except KeyError:
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise

        return implementation(**kwargs)

    def _create_noop_performance_monitor(self) -> Any:
        """Create a no-op performance monitor for fallback scenarios."""
        class NoOpPerformanceMonitor:
            """No-op performance monitor that provides safe fallback behavior."""

            async def run_baseline_benchmark(self, samples_per_operation: int = 50):
                return {}

            async def health_check(self):
                return {"status": "healthy", "type": "noop"}

            def generate_performance_report(self, baselines) -> str:
                return "No-op performance monitor: no metrics available"

        return NoOpPerformanceMonitor()

    def is_registered(self, interface: type[T]) -> bool:
        """Check if service is registered.

        Args:
            interface: Service interface to check

        Returns:
            True if registered, False otherwise
        """
        return interface in self._services

    # Monitoring service factory methods
    def register_metrics_collector_factory(self, collector_type: str = "opentelemetry") -> None:
        """Register factory for metrics collectors optimized for ML workloads.

        Args:
            collector_type: Type of metrics collector (opentelemetry only)
        """
        def create_metrics_collector() -> MetricsRegistryProtocol:
            if collector_type == "opentelemetry":
                try:
                    from prompt_improver.performance.monitoring.metrics_registry import (
                        MetricsRegistry,
                    )
                    return MetricsRegistry()
                except ImportError:
                    # Fallback to no-op metrics registry
                    from prompt_improver.core.services.noop_metrics import (
                        NoOpMetricsRegistry,
                    )
                    return NoOpMetricsRegistry()
            raise ValueError(
                f"Unsupported metrics collector type: {collector_type}. Only 'opentelemetry' is supported."
            )

        self.register_factory(
            MetricsRegistryProtocol,
            create_metrics_collector,
            tags={"metrics", "monitoring", "opentelemetry"}
        )
        self.logger.debug(f"Registered metrics collector factory: {collector_type}")

    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitors with circuit breaker patterns."""
        def create_health_monitor():
            try:
                from prompt_improver.performance.monitoring.health.unified_health_system import (
                    get_unified_health_monitor,
                )
                return get_unified_health_monitor()
            except ImportError:
                # Fallback to basic health monitor
                from prompt_improver.monitoring.health_check import BasicHealthMonitor
                return BasicHealthMonitor()

        self.register_factory(
            "health_monitor",
            create_health_monitor,
            tags={"health", "monitoring", "circuit_breaker"}
        )
        self.logger.debug("Registered health monitor factory")

    def register_ab_testing_service_factory(self) -> None:
        """Register factory for A/B testing service.

        This creates an IABTestingService instance that uses the performance layer
        implementation through the adapter pattern, maintaining clean architecture.
        """
        def create_ab_testing_service() -> IABTestingService:
            try:
                from prompt_improver.performance.testing.ab_testing_adapter import (
                    create_ab_testing_service_adapter,
                )
                from prompt_improver.performance.testing.ab_testing_service import (
                    ModernABConfig,
                )

                # Use default configuration optimized for prompt improvement
                config = ModernABConfig(
                    confidence_level=0.95,
                    statistical_power=0.8,
                    minimum_detectable_effect=0.05,
                    practical_significance_threshold=0.02,
                    enable_early_stopping=True,
                    enable_sequential_testing=True,
                )

                return create_ab_testing_service_adapter(config)
            except ImportError:
                # Fallback to no-op implementation if performance layer is unavailable
                from prompt_improver.shared.interfaces.ab_testing import (
                    NoOpABTestingService,
                )
                self.logger.warning("A/B testing service unavailable, using no-op implementation")
                return NoOpABTestingService()

        self.register_factory(
            IABTestingService,
            create_ab_testing_service,
            tags={"ab_testing", "statistics", "experimentation"}
        )
        self.logger.debug("Registered A/B testing service factory")

    def register_performance_monitoring_factory(self) -> None:
        """Register factory for performance monitoring service.

        Uses service locator pattern to eliminate circular dependencies while
        maintaining full functionality. No longer imports MCP server components directly.
        """
        async def create_performance_monitoring():
            try:
                # Use the factory pattern to create performance benchmark with service locator
                from prompt_improver.performance.monitoring.performance_benchmark_factory import (
                    create_performance_benchmark_from_container,
                )
                # Pass this container to extract dependencies safely
                return await create_performance_benchmark_from_container(self)
            except ImportError:
                # Fallback to no-op performance monitor
                return self._create_noop_performance_monitor()

        self.register_factory(
            "performance_monitor",
            create_performance_monitoring,
            tags={"performance", "monitoring", "benchmark", "service_locator"}
        )
        self.logger.debug("Registered performance monitoring factory with service locator pattern")

    def register_alert_manager_factory(self) -> None:
        """Register factory for alert management service."""
        def create_alert_manager():
            try:
                from prompt_improver.monitoring.alert_manager import AlertManager
                return AlertManager()
            except ImportError:
                # Fallback to no-op alert manager
                from prompt_improver.monitoring.noop_alert_manager import (
                    NoOpAlertManager,
                )
                return NoOpAlertManager()

        self.register_factory(
            "alert_manager",
            create_alert_manager,
            tags={"alerts", "monitoring", "notifications"}
        )
        self.logger.debug("Registered alert manager factory")

    def register_tracing_service_factory(self) -> None:
        """Register factory for distributed tracing service."""
        def create_tracing_service():
            try:
                from prompt_improver.monitoring.opentelemetry.instrumentation import (
                    TracingService,
                )
                return TracingService()
            except ImportError:
                # Fallback to no-op tracing
                from prompt_improver.monitoring.noop_tracing import NoOpTracingService
                return NoOpTracingService()

        self.register_factory(
            "tracing_service",
            create_tracing_service,
            tags={"tracing", "monitoring", "observability"}
        )
        self.logger.debug("Registered tracing service factory")

    def register_observability_dashboard_factory(self) -> None:
        """Register factory for observability dashboard service."""
        def create_observability_dashboard():
            try:
                from prompt_improver.monitoring.dashboard.observability_dashboard import (
                    ObservabilityDashboard,
                )
                return ObservabilityDashboard()
            except ImportError:
                # Fallback to basic dashboard
                from prompt_improver.monitoring.basic_dashboard import BasicDashboard
                return BasicDashboard()

        self.register_factory(
            "observability_dashboard",
            create_observability_dashboard,
            tags={"dashboard", "monitoring", "visualization"}
        )
        self.logger.debug("Registered observability dashboard factory")

    # Service access methods
    async def get_metrics_registry(self) -> MetricsRegistryProtocol:
        """Get metrics registry instance."""
        return await self.get(MetricsRegistryProtocol)

    async def get_health_monitor(self) -> Any:
        """Get health monitor instance."""
        return await self.get("health_monitor")

    async def get_ab_testing_service(self) -> IABTestingService:
        """Get A/B testing service instance."""
        return await self.get(IABTestingService)

    async def get_performance_monitor(self) -> Any:
        """Get performance monitor instance."""
        return await self.get("performance_monitor")

    async def get_alert_manager(self) -> Any:
        """Get alert manager instance."""
        return await self.get("alert_manager")

    async def get_tracing_service(self) -> Any:
        """Get tracing service instance."""
        return await self.get("tracing_service")

    async def get_observability_dashboard(self) -> Any:
        """Get observability dashboard instance."""
        return await self.get("observability_dashboard")

    # Container lifecycle management
    async def initialize(self) -> None:
        """Initialize all monitoring services."""
        if self._initialized:
            return

        self.logger.info(f"Initializing monitoring container '{self.name}'")

        # Initialize all registered services
        for interface in list(self._services.keys()):
            try:
                await self.get(interface)
            except Exception as e:
                self.logger.exception(f"Failed to initialize {interface}: {e}")
                raise

        self._initialized = True
        self.logger.info(f"Monitoring container '{self.name}' initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all monitoring services gracefully."""
        self.logger.info(f"Shutting down monitoring container '{self.name}'")

        # Shutdown in reverse initialization order
        for interface in reversed(self._initialization_order):
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    if hasattr(registration.instance, "shutdown"):
                        if asyncio.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                    self.logger.debug(f"Shutdown service: {interface.__name__}")
                except Exception as e:
                    self.logger.exception(f"Error shutting down {interface.__name__}: {e}")

        self._services.clear()
        self._initialization_order.clear()
        self._initialized = False
        self.logger.info(f"Monitoring container '{self.name}' shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all monitoring services."""
        results = {
            "container_status": "healthy",
            "container_name": self.name,
            "initialized": self._initialized,
            "registered_services": len(self._services),
            "services": {},
        }

        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            try:
                if (registration.initialized and registration.instance and
                    hasattr(registration.instance, "health_check")):

                    health_check = registration.instance.health_check
                    if asyncio.iscoroutinefunction(health_check):
                        service_health = await health_check()
                    else:
                        service_health = health_check()
                    results["services"][service_name] = service_health
                else:
                    results["services"][service_name] = {
                        "status": "healthy",
                        "initialized": registration.initialized,
                    }
            except Exception as e:
                results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                results["container_status"] = "degraded"

        return results

    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered monitoring services."""
        info = {
            "container_name": self.name,
            "initialized": self._initialized,
            "services": {},
        }

        for interface, registration in self._services.items():
            service_name = interface.__name__ if hasattr(interface, "__name__") else str(interface)
            info["services"][service_name] = {
                "implementation": registration.implementation.__name__ if registration.implementation else "Factory",
                "lifetime": registration.lifetime.value,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None,
                "tags": list(registration.tags),
            }

        return info

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle context for monitoring container."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global monitoring container instance
_monitoring_container: MonitoringContainer | None = None


def get_monitoring_container() -> MonitoringContainer:
    """Get the global monitoring container instance.

    Returns:
        MonitoringContainer: Global monitoring container instance
    """
    global _monitoring_container
    if _monitoring_container is None:
        _monitoring_container = MonitoringContainer()
    return _monitoring_container


async def shutdown_monitoring_container() -> None:
    """Shutdown the global monitoring container."""
    global _monitoring_container
    if _monitoring_container:
        await _monitoring_container.shutdown()
        _monitoring_container = None
