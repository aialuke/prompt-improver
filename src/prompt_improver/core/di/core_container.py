"""Core Services Dependency Injection Container (2025).

Specialized DI container for core system services including configuration,
metrics, health monitoring, and essential system utilities.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar

from prompt_improver.core.di.protocols import CoreContainerProtocol, ContainerRegistryProtocol
from prompt_improver.core.interfaces.datetime_service import DateTimeServiceProtocol
from prompt_improver.core.protocols.retry_protocols import MetricsRegistryProtocol
from prompt_improver.core.services.datetime_service import DateTimeService

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
    interface: Type[Any]
    implementation: Type[Any] | None
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

    def __init__(self, name: str = "core"):
        """Initialize core services container.
        
        Args:
            name: Container identifier for logging
        """
        self.name = name
        self.logger = logger.getChild(f"container.{name}")
        self._services: dict[Type[Any], CoreServiceRegistration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_order: list[Type[Any]] = []
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
        interface: Type[T],
        implementation: Type[T],
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
        self.logger.debug(
            f"Registered singleton: {interface.__name__} -> {implementation.__name__}"
        )

    def register_transient(
        self,
        interface: Type[T],
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
        self.logger.debug(f"Registered transient: {interface.__name__}")

    def register_factory(
        self,
        interface: Type[T],
        factory: Any,
        tags: set[str] | None = None,
    ) -> None:
        """Register a service factory.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create service
            tags: Optional tags for service categorization
        """
        self._services[interface] = CoreServiceRegistration(
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
        interface: Type[T],
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
        self.logger.debug(f"Registered instance: {interface.__name__}")

    async def get(self, interface: Type[T]) -> T:
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

    async def _resolve_service(self, interface: Type[T]) -> T:
        """Internal service resolution with lifecycle management.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
        """
        if interface not in self._services:
            raise KeyError(f"Service not registered: {interface.__name__}")
            
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
            
        self.logger.debug(f"Resolved service: {interface.__name__}")
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

    async def _create_from_class(self, implementation: Type[Any]) -> Any:
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

    def is_registered(self, interface: Type[T]) -> bool:
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
                from prompt_improver.performance.monitoring.metrics_registry import MetricsRegistry
                return MetricsRegistry()
            except ImportError:
                # Fallback to no-op implementation
                from prompt_improver.core.services.noop_metrics import NoOpMetricsRegistry
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
                from prompt_improver.core.services.basic_health_monitor import BasicHealthMonitor
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
                from prompt_improver.core.config_schema import ConfigService
                return ConfigService()
            except ImportError:
                # Fallback to environment-based config
                from prompt_improver.core.services.env_config_service import EnvConfigService
                return EnvConfigService()

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
                from prompt_improver.core.services.logging_service import LoggingService
                return LoggingService()
            except ImportError:
                # Fallback to basic logging
                import logging
                return logging.getLogger("prompt_improver")

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
                from prompt_improver.core.feature_flags import FeatureFlagService
                return FeatureFlagService()
            except ImportError:
                # Fallback to no-op feature flags
                from prompt_improver.core.services.noop_feature_flags import NoOpFeatureFlagService
                return NoOpFeatureFlagService()

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
                from prompt_improver.integrations.services_config import ExternalServicesConfigImpl
                return ExternalServicesConfigImpl.from_environment(environment)
            except ImportError:
                # Fallback to environment-based config
                from prompt_improver.core.services.env_external_services_config import EnvExternalServicesConfig
                return EnvExternalServicesConfig(environment)

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
                self.logger.error(f"Failed to initialize {interface}: {e}")
                raise
                
        self._initialized = True
        self.logger.info(f"Core container '{self.name}' initialization complete")

    async def shutdown(self) -> None:
        """Shutdown all core services gracefully."""
        self.logger.info(f"Shutting down core container '{self.name}'")
        
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
                    self.logger.error(f"Error shutting down {interface.__name__}: {e}")
                    
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

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Managed lifecycle context for core container."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global core container instance
_core_container: Optional[CoreContainer] = None


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