"""Core Services Dependency Injection Container (2025).

Specialized DI container for core system services including configuration,
metrics, health monitoring, and essential system utilities.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol
from prompt_improver.core.services.datetime_service import DateTimeService
from prompt_improver.shared.interfaces.protocols.core import (
    ContainerRegistryProtocol,
    CoreContainerProtocol,
    MetricsRegistryProtocol,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class CoreServiceRegistration:
    """Core service registration information."""
    interface: type[Any]
    implementation: type[Any] | None
    lifetime: ServiceLifetime
    factory: Callable[[], Any] | None = None
    initialized: bool = False
    instance: Any = None
    tags: set[str] = field(default_factory=set)
    health_check: Callable[[], Any] | None = None


class CoreContainer(CoreContainerProtocol, ContainerRegistryProtocol):
    """Specialized DI container for core system services.

    Manages essential system services including:
    - Configuration management
    - Metrics collection and registry
    - Health monitoring
    - DateTime services
    - Logging infrastructure

    Follows clean architecture with protocol-based dependencies.
    """

    def __init__(self, name: str = "core") -> None:
        """Initialize core services container.

        Args:
            name: Container identifier for logging
        """
        self.name = name
        self.logger = logger.getChild(f"container.{name}")
        self._services: dict[type[Any], CoreServiceRegistration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_order: list[type[Any]] = []
        self._register_default_services()
        self.logger.debug(f"Core container '{self.name}' initialized")

    def _register_default_services(self) -> None:
        """Register default core services."""
        # DateTime service
        self.register_singleton(
            DateTimeServiceProtocol,
            DateTimeService,
            tags={"core", "datetime"}
        )

        # Self-registration for dependency injection
        self.register_instance(CoreContainer, self, tags={"container", "core"})

        # Metrics registry factory
        self.register_metrics_collector_factory()

        # Health monitor factory
        self.register_health_monitor_factory()

        # Configuration service factory
        self.register_config_service_factory()

        # Logging infrastructure
        self.register_logging_service_factory()

        # Feature flags service
        self.register_feature_flags_service_factory()

        # External services config service
        self.register_external_services_config_factory()

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
        self._services[interface] = CoreServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or set(),
        )
        interface_name = self._get_interface_name(interface)
        impl_name = self._get_interface_name(implementation)
        self.logger.debug(f"Registered singleton: {interface_name} -> {impl_name}")

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
        self._services[interface] = CoreServiceRegistration(
            interface=interface,
            implementation=implementation_or_factory if not callable(implementation_or_factory) else None,
            factory=implementation_or_factory if callable(implementation_or_factory) else None,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set(),
        )
        interface_name = self._get_interface_name(interface)
        self.logger.debug(f"Registered transient: {interface_name}")

    def register_factory(
        self,
        interface: type[T],
        factory: Any,
        tags: set[str] | None = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> None:
        """Register a service factory.

        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
            tags: Optional tags for service categorization
            lifetime: Service lifetime (singleton by default)
        """
        self._services[interface] = CoreServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=lifetime,
            factory=factory,
            initialized=False,  # Explicitly set for factory registrations
            instance=None,  # Explicitly set for factory registrations
            tags=tags or set(),
        )
        interface_name = self._get_interface_name(interface)
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
        registration = CoreServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            tags=tags or set(),
        )
        self._services[interface] = registration
        interface_name = self._get_interface_name(interface)
        self.logger.debug(f"Registered instance: {interface_name}")

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
        # Handle interface name for logging/error messages
        interface_name = self._get_interface_name(interface)

        if interface not in self._services:
            raise KeyError(f"Service not registered: {interface_name}")

        registration = self._services[interface]

        # Return existing singleton instance
        if (registration.lifetime == ServiceLifetime.SINGLETON and
            registration.initialized and registration.instance is not None):
            self.logger.debug(f"Returning cached singleton for {interface_name}")
            return registration.instance

        # Create new instance
        if registration.factory:
            instance = await self._create_from_factory(registration.factory)
        elif registration.implementation:
            instance = await self._create_from_class(registration.implementation)
        else:
            raise ValueError(f"No factory or implementation for {interface_name}")

        # Initialize if needed
        if hasattr(instance, "initialize") and asyncio.iscoroutinefunction(instance.initialize):
            await instance.initialize()

        # Store singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            registration.instance = instance
            registration.initialized = True
            self._initialization_order.append(interface)
            self.logger.debug(f"Cached new singleton instance for {interface_name}")

        self.logger.debug(f"Resolved service: {interface_name}")
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

    def _get_interface_name(self, interface: Any) -> str:
        """Get a readable name for an interface, handling Union types and other special cases.

        Args:
            interface: Interface type or annotation

        Returns:
            str: Readable interface name
        """
        if hasattr(interface, "__name__"):
            return interface.__name__

        # Handle Union types (Python 3.10+ syntax like `logging.Logger | None`)
        if hasattr(interface, "__args__"):
            args_names = []
            for arg in interface.__args__:
                if hasattr(arg, "__name__"):
                    args_names.append(arg.__name__)
                else:
                    args_names.append(str(arg))
            return f"Union[{', '.join(args_names)}]"

        return str(interface)

    def _is_optional_parameter(self, annotation: Any) -> tuple[bool, Any]:
        """Check if parameter annotation is optional (Union with None).

        Args:
            annotation: Parameter annotation

        Returns:
            tuple: (is_optional, non_none_type_if_optional)
        """
        # Handle Union types with None (Optional)
        if hasattr(annotation, "__args__"):
            args = annotation.__args__
            if len(args) == 2 and type(None) in args:
                # This is Optional[T] or T | None
                non_none_type = next(arg for arg in args if arg is not type(None))
                return True, non_none_type

        return False, annotation

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
                # Check if this is an optional parameter
                is_optional, actual_type = self._is_optional_parameter(param.annotation)

                try:
                    # Try to resolve the actual type (non-None part for optional parameters)
                    dependency = await self._resolve_service(actual_type)
                    kwargs[param_name] = dependency
                except KeyError:
                    # If optional or has default, skip this parameter
                    if is_optional or param.default != inspect.Parameter.empty:
                        continue
                    # If not optional and no default, re-raise the error
                    raise

        return implementation(**kwargs)

    def is_registered(self, interface: type[T]) -> bool:
        """Check if service is registered.

        Args:
            interface: Service interface to check

        Returns:
            True if registered, False otherwise
        """
        return interface in self._services

    def register_metrics_collector_factory(self) -> None:
        """Register factory for metrics collector."""
        def create_metrics_collector() -> MetricsRegistryProtocol:
            try:
                from prompt_improver.performance.monitoring.metrics_registry import (
                    MetricsRegistry,
                )
                return MetricsRegistry()
            except ImportError:
                # Fallback to no-op implementation
                from prompt_improver.core.services.noop_metrics import (
                    NoOpMetricsRegistry,
                )
                return NoOpMetricsRegistry()

        self.register_factory(
            MetricsRegistryProtocol,
            create_metrics_collector,
            tags={"metrics", "monitoring", "core"}
        )
        self.logger.debug("Registered metrics collector factory")

    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitor."""
        def create_health_monitor():
            try:
                from prompt_improver.performance.monitoring.health.unified_health_system import (
                    get_unified_health_monitor,
                )
                return get_unified_health_monitor()
            except ImportError:
                # Fallback to basic health monitor
                from prompt_improver.core.services.basic_health_monitor import (
                    BasicHealthMonitor,
                )
                return BasicHealthMonitor()

        self.register_factory(
            "health_monitor",
            create_health_monitor,
            tags={"health", "monitoring", "core"}
        )
        self.logger.debug("Registered health monitor factory")

    def register_config_service_factory(self) -> None:
        """Register factory for configuration service."""
        def create_config_service():
            try:
                from prompt_improver.core.config.unified_config import UnifiedConfig
                return UnifiedConfig()
            except ImportError:
                try:
                    from prompt_improver.core.facades.minimal_config_facade import (
                        MinimalConfigFacade,
                    )
                    return MinimalConfigFacade()
                except ImportError:
                    # Simple fallback using dictionary
                    import os
                    return {
                        'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/prompt_improver'),
                        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
                        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                        'environment': os.getenv('ENVIRONMENT', 'development')
                    }

        self.register_factory(
            "config_service",
            create_config_service,
            tags={"config", "core"}
        )
        self.logger.debug("Registered config service factory")

    def register_logging_service_factory(self) -> None:
        """Register factory for logging service with structured logging."""
        def create_logging_service():
            try:
                from prompt_improver.core.common.logging_utils import LoggingService
                return LoggingService()
            except ImportError:
                # Fallback to basic logging
                import logging
                logger = logging.getLogger("prompt_improver")
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)
                    logger.setLevel(logging.INFO)
                return logger

        self.register_factory(
            "logging_service",
            create_logging_service,
            tags={"logging", "core"}
        )
        self.logger.debug("Registered logging service factory")

    def register_feature_flags_service_factory(self) -> None:
        """Register factory for feature flags service."""
        def create_feature_flags_service():
            try:
                from prompt_improver.core.feature_flags import FeatureFlagsManager
                return FeatureFlagsManager()
            except ImportError:
                # Simple fallback feature flags
                class SimpleFeatureFlags:
                    def __init__(self) -> None:
                        self.flags = {}

                    def is_enabled(self, flag_name: str) -> bool:
                        return self.flags.get(flag_name, False)

                    def enable(self, flag_name: str) -> None:
                        self.flags[flag_name] = True

                    def disable(self, flag_name: str) -> None:
                        self.flags[flag_name] = False

                return SimpleFeatureFlags()

        self.register_factory(
            "feature_flags_service",
            create_feature_flags_service,
            tags={"feature_flags", "core"}
        )
        self.logger.debug("Registered feature flags service factory")

    def register_external_services_config_factory(self, environment: str = "production") -> None:
        """Register factory for external services configuration.

        Args:
            environment: Deployment environment (development, staging, production)
        """
        def create_external_services_config():
            try:
                from prompt_improver.core.config.unified_config import UnifiedConfig
                config = UnifiedConfig()
                return config.get_external_services_config()
            except ImportError:
                # Simple fallback using environment variables
                import os
                return {
                    'environment': environment,
                    'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/prompt_improver'),
                    'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
                    'ml_service_url': os.getenv('ML_SERVICE_URL', 'http://localhost:8001'),
                    'monitoring_enabled': os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
                    'metrics_port': int(os.getenv('METRICS_PORT', '9090')),
                }

        self.register_factory(
            "external_services_config",
            create_external_services_config,
            tags={"config", "external", "services"}
        )
        self.logger.debug(f"Registered external services config factory (env: {environment})")

    # CoreContainerProtocol implementation
    async def get_datetime_service(self) -> DateTimeServiceProtocol:
        """Get datetime service instance."""
        return await self.get(DateTimeServiceProtocol)

    async def get_metrics_registry(self) -> MetricsRegistryProtocol:
        """Get metrics registry instance."""
        return await self.get(MetricsRegistryProtocol)

    async def get_health_monitor(self) -> Any:
        """Get health monitor instance."""
        return await self.get("health_monitor")

    async def get_config_service(self) -> Any:
        """Get configuration service instance."""
        return await self.get("config_service")

    async def get_logging_service(self) -> Any:
        """Get logging service instance."""
        return await self.get("logging_service")

    async def get_feature_flags_service(self) -> Any:
        """Get feature flags service instance."""
        return await self.get("feature_flags_service")

    async def get_external_services_config(self) -> Any:
        """Get external services config instance."""
        return await self.get("external_services_config")

    # Container lifecycle management
    async def initialize(self) -> None:
        """Initialize all core services."""
        if self._initialized:
            return

        self.logger.info(f"Initializing core container '{self.name}'")

        # Initialize all registered services
        for interface in list(self._services.keys()):
            try:
                await self.get(interface)
            except Exception as e:
                self.logger.exception(f"Failed to initialize {interface}: {e}")
                raise

        self._initialized = True
        self.logger.info(f"Core container '{self.name}' initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all core services gracefully."""
        self.logger.info(f"Shutting down core container '{self.name}'")

        # Shutdown in reverse initialization order
        for interface in reversed(self._initialization_order):
            interface_name = self._get_interface_name(interface)
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    if hasattr(registration.instance, "shutdown"):
                        if asyncio.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                    self.logger.debug(f"Shutdown service: {interface_name}")
                except Exception as e:
                    self.logger.exception(f"Error shutting down {interface_name}: {e}")

        self._services.clear()
        self._initialization_order.clear()
        self._initialized = False
        self.logger.info(f"Core container '{self.name}' shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all core services."""
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
        """Get information about registered services."""
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

    # ContainerRegistryProtocol implementation
    def register_container(self, name: str, container: Any) -> None:
        """Register a container with the registry.

        Args:
            name: Container identifier
            container: Container instance to register
        """
        # For now, core container doesn't maintain other containers
        # This could be extended if needed for container orchestration
        self.logger.debug(f"Container registration not implemented for core container: {name}")

    def get_container(self, name: str) -> Any:
        """Get a container from the registry.

        Args:
            name: Container identifier

        Returns:
            Container instance

        Raises:
            KeyError: If container not found
        """
        if name == self.name:
            return self
        raise KeyError(f"Container not found: {name}")

    def list_containers(self) -> list[str]:
        """List all registered container names.

        Returns:
            List of container names
        """
        return [self.name]

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle context for core container."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global core container instance
_core_container: CoreContainer | None = None


def get_core_container() -> CoreContainer:
    """Get the global core container instance.

    Returns:
        CoreContainer: Global core container instance
    """
    global _core_container
    if _core_container is None:
        _core_container = CoreContainer()
    return _core_container


async def shutdown_core_container() -> None:
    """Shutdown the global core container."""
    global _core_container
    if _core_container:
        await _core_container.shutdown()
        _core_container = None
