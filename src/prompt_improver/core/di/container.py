"""Enhanced Dependency Injection Container following 2025 best practices.

This module provides a comprehensive dependency injection system that combines
the power of python-dependency-injector with async-compatible extensions
for ML pipeline orchestration and modern Python applications.

Features:
- Modern dependency-injector integration
- Async service initialization and lifecycle management
- ML component-specific factory patterns
- Resource management for database connections
- Type-safe service resolution with protocols
- Circular dependency detection
- Service health monitoring
- Easy testing with mock services
- Performance metrics collection
"""

import asyncio
import logging

from typing import Dict, Type, Any, TypeVar, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect
from contextlib import asynccontextmanager

try:
    from dependency_injector import containers, providers
    from dependency_injector.wiring import Provide, inject
    DEPENDENCY_INJECTOR_AVAILABLE = True
except ImportError:
    DEPENDENCY_INJECTOR_AVAILABLE = False
    containers = None
    providers = None
    Provide = None
    inject = None

from ..interfaces.datetime_service import DateTimeServiceProtocol
from ..services.datetime_service import DateTimeService
from ..protocols.retry_protocols import MetricsRegistryProtocol

T = TypeVar('T')

class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    RESOURCE = "resource"  # For resources requiring cleanup

@dataclass
class ServiceRegistration:
    """Enhanced service registration information."""
    interface: Type
    implementation: Optional[Type]
    lifetime: ServiceLifetime
    factory: Optional[Callable] = None
    initializer: Optional[Callable] = None  # Async initializer
    finalizer: Optional[Callable] = None    # Cleanup function
    initialized: bool = False
    instance: Any = None
    dependencies: Set[Type] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    priority: int = 0  # For service ordering
    health_check: Optional[Callable] = None

class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    def __init__(self, cycle: str):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {cycle}")

class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve unregistered service."""
    def __init__(self, service_type: Type):
        self.service_type = service_type
        super().__init__(f"Service not registered: {service_type.__name__}")

class ServiceInitializationError(Exception):
    """Raised when service initialization fails."""
    def __init__(self, service_type: Type, original_error: Exception):
        self.service_type = service_type
        self.original_error = original_error
        super().__init__(f"Failed to initialize {service_type.__name__}: {original_error}")

class ResourceManagementError(Exception):
    """Raised when resource management operations fail."""
    pass

class DIContainer:
    """Enhanced dependency injection container with 2025 best practices.
    
    Provides comprehensive service registration, resolution, and lifecycle management
    for ML Pipeline Orchestrator systems with async support, resource management,
    and factory patterns optimized for ML workloads.
    
    Features:
    - Async service initialization and cleanup
    - Resource lifecycle management (connections, models, etc.)
    - Factory patterns for metrics collectors and health monitors
    - Scoped service lifetimes for request/session isolation
    - Performance monitoring and health checks
    - Integration with python-dependency-injector when available
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, name: str = "default"):
        """Initialize the enhanced DI container.
        
        Args:
            logger: Optional logger for debugging and monitoring
            name: Container name for identification in logs
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self._services: Dict[Type, ServiceRegistration] = {}
        self._scoped_services: Dict[str, Dict[Type, Any]] = {}  # scope_id -> services
        self._resources: Set[Any] = set()  # Resources requiring cleanup
        self._resolution_stack: Set[Type] = set()
        self._lock = asyncio.Lock()
        self._metrics_registry: Optional[MetricsRegistryProtocol] = None
        self._health_checks: Dict[str, Callable] = {}
        
        # Performance tracking
        self._resolution_times: Dict[Type, float] = {}
        self._initialization_order: list[Type] = []
        
        # Register default services
        self._register_default_services()
        
        self.logger.debug(f"Enhanced DIContainer '{self.name}' initialized")
    
    def _register_default_services(self):
        """Register default system services."""
        # Register datetime service as singleton
        self.register_singleton(DateTimeServiceProtocol, DateTimeService)
        
        # Register container self-reference for dependency injection
        self.register_instance(DIContainer, self)
    
    def register_singleton(self, interface: Type[T], implementation: Type[T], 
                          tags: Optional[Set[str]] = None,
                          initializer: Optional[Callable] = None,
                          finalizer: Optional[Callable] = None,
                          health_check: Optional[Callable] = None) -> None:
        """Register a service as singleton.
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
            initializer: Optional async initializer function
            finalizer: Optional cleanup function
            health_check: Optional health check function
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set(),
            health_check=health_check
        )
        
        self.logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")
    
    def register_transient(self, interface: Type[T], implementation: Type[T],
                          tags: Optional[Set[str]] = None) -> None:
        """Register a service as transient (new instance each time).
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or set()
        )
        
        self.logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], 
                        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                        tags: Optional[Set[str]] = None,
                        initializer: Optional[Callable] = None,
                        finalizer: Optional[Callable] = None) -> None:
        """Register a service with custom factory function.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create service instance
            lifetime: Service lifetime (singleton, transient, or scoped)
            tags: Optional tags for service categorization
            initializer: Optional async initializer function
            finalizer: Optional cleanup function
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=lifetime,
            factory=factory,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set()
        )
        
        self.logger.debug(f"Registered factory: {interface.__name__} (lifetime: {lifetime.value})")
    
    def register_instance(self, interface: Type[T], instance: T,
                         tags: Optional[Set[str]] = None,
                         finalizer: Optional[Callable] = None,
                         health_check: Optional[Callable] = None) -> None:
        """Register a pre-created service instance.
        
        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
            tags: Optional tags for service categorization
            finalizer: Optional cleanup function
            health_check: Optional health check function
        """
        registration = ServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance,
            finalizer=finalizer,
            tags=tags or set(),
            health_check=health_check
        )
        
        self._services[interface] = registration
        
        self.logger.debug(f"Registered instance: {interface.__name__}")
    
    async def get(self, interface: Type[T], scope_id: Optional[str] = None) -> T:
        """Resolve service instance.
        
        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier for scoped services
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotRegisteredError: If service is not registered
            CircularDependencyError: If circular dependency detected
            ServiceInitializationError: If service initialization fails
        """
        import time
        start_time = time.perf_counter()
        
        try:
            async with self._lock:
                result = await self._resolve_service(interface, scope_id)
                
                # Track resolution time for performance monitoring
                resolution_time = time.perf_counter() - start_time
                self._resolution_times[interface] = resolution_time
                
                # Report metrics if available
                if self._metrics_registry:
                    self._metrics_registry.record_histogram(
                        "di_container_resolution_time",
                        resolution_time,
                        {"service": interface.__name__, "container": self.name}
                    )
                
                return result
        except Exception as e:
            if self._metrics_registry:
                self._metrics_registry.increment_counter(
                    "di_container_resolution_errors",
                    {"service": interface.__name__, "container": self.name}
                )
            raise
    
    async def _resolve_service(self, interface: Type[T], scope_id: Optional[str] = None) -> T:
        """Internal service resolution with enhanced lifecycle management.
        
        Args:
            interface: Service interface to resolve
            scope_id: Optional scope identifier for scoped services
            
        Returns:
            Service instance
        """
        # Check if service is registered
        if interface not in self._services:
            raise ServiceNotRegisteredError(interface)
        
        # Check for circular dependencies
        if interface in self._resolution_stack:
            cycle = " -> ".join([t.__name__ for t in self._resolution_stack]) + f" -> {interface.__name__}"
            raise CircularDependencyError(cycle)
        
        registration = self._services[interface]
        
        # Handle different service lifetimes
        if registration.lifetime == ServiceLifetime.SINGLETON:
            # Return existing singleton instance if available
            if registration.initialized and registration.instance is not None:
                return registration.instance
        elif registration.lifetime == ServiceLifetime.SCOPED:
            # Handle scoped services
            if scope_id and scope_id in self._scoped_services:
                if interface in self._scoped_services[scope_id]:
                    return self._scoped_services[scope_id][interface]
        
        # Add to resolution stack for circular dependency detection
        self._resolution_stack.add(interface)
        
        try:
            # Create new instance
            if registration.factory:
                instance = await self._create_from_factory(registration.factory)
            else:
                instance = await self._create_from_class(registration.implementation)
            
            # Run initializer if present
            if registration.initializer:
                try:
                    if inspect.iscoroutinefunction(registration.initializer):
                        await registration.initializer(instance)
                    else:
                        registration.initializer(instance)
                except Exception as e:
                    raise ServiceInitializationError(interface, e)
            
            # Store instance based on lifetime
            if registration.lifetime == ServiceLifetime.SINGLETON:
                registration.instance = instance
                registration.initialized = True
                self._initialization_order.append(interface)
            elif registration.lifetime == ServiceLifetime.SCOPED and scope_id:
                if scope_id not in self._scoped_services:
                    self._scoped_services[scope_id] = {}
                self._scoped_services[scope_id][interface] = instance
            elif registration.lifetime == ServiceLifetime.RESOURCE:
                # Track resources for cleanup
                self._resources.add(instance)
                registration.instance = instance
                registration.initialized = True
            
            self.logger.debug(f"Resolved service: {interface.__name__} (lifetime: {registration.lifetime.value})")
            return instance
            
        except Exception as e:
            if not isinstance(e, (ServiceInitializationError, CircularDependencyError)):
                raise ServiceInitializationError(interface, e)
            raise
        finally:
            # Remove from resolution stack
            self._resolution_stack.discard(interface)
    
    async def _create_from_factory(self, factory: Callable) -> Any:
        """Create service instance from factory function.
        
        Args:
            factory: Factory function
            
        Returns:
            Service instance
        """
        if inspect.iscoroutinefunction(factory):
            return await factory()
        else:
            return factory()
    
    async def _create_from_class(self, implementation: Type) -> Any:
        """Create service instance from class constructor.
        
        Args:
            implementation: Implementation class
            
        Returns:
            Service instance
        """
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        
        # Resolve constructor dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Check if parameter has type annotation
            if param.annotation != inspect.Parameter.empty:
                # Try to resolve dependency
                try:
                    dependency = await self._resolve_service(param.annotation)
                    kwargs[param_name] = dependency
                except ServiceNotRegisteredError:
                    # Skip optional dependencies
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise
        
        # Create instance
        instance = implementation(**kwargs)
        
        # Initialize if it has an async initialize method
        if hasattr(instance, 'initialize') and inspect.iscoroutinefunction(instance.initialize):
            await instance.initialize()
        
        return instance
    
    async def health_check(self) -> dict:
        """Perform health check on all registered services.
        
        Returns:
            dict: Health check results
        """
        results = {
            "container_status": "healthy",
            "registered_services": len(self._services),
            "services": {}
        }
        
        for interface, registration in self._services.items():
            service_name = interface.__name__
            
            try:
                # Only check initialized singletons
                if (registration.lifetime == ServiceLifetime.SINGLETON and 
                    registration.initialized and 
                    registration.instance is not None):
                    
                    instance = registration.instance
                    
                    # Check if service has health_check method
                    if hasattr(instance, 'health_check') and callable(instance.health_check):
                        if inspect.iscoroutinefunction(instance.health_check):
                            service_health = await instance.health_check()
                        else:
                            service_health = instance.health_check()
                        
                        results["services"][service_name] = service_health
                    else:
                        results["services"][service_name] = {
                            "status": "healthy",
                            "note": "No health check method available"
                        }
                else:
                    results["services"][service_name] = {
                        "status": "not_initialized",
                        "lifetime": registration.lifetime.value
                    }
                    
            except Exception as e:
                results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                results["container_status"] = "degraded"
        
        return results
    
    def get_registration_info(self) -> dict:
        """Get information about registered services.
        
        Returns:
            dict: Registration information
        """
        info = {}
        
        for interface, registration in self._services.items():
            info[interface.__name__] = {
                "implementation": registration.implementation.__name__ if registration.implementation else "Factory",
                "lifetime": registration.lifetime.value,
                "initialized": registration.initialized,
                "has_instance": registration.instance is not None
            }
        
        return info
    
    # Enhanced methods for 2025 best practices
    
    def register_scoped(self, interface: Type[T], implementation: Type[T],
                       tags: Optional[Set[str]] = None) -> None:
        """Register a service as scoped (one instance per scope).
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED,
            tags=tags or set()
        )
        
        self.logger.debug(f"Registered scoped: {interface.__name__} -> {implementation.__name__}")
    
    def register_resource(self, interface: Type[T], 
                         factory: Callable[[], T],
                         initializer: Optional[Callable] = None,
                         finalizer: Optional[Callable] = None,
                         tags: Optional[Set[str]] = None) -> None:
        """Register a resource that requires lifecycle management.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create resource
            initializer: Optional async initializer function
            finalizer: Required cleanup function for resource
            tags: Optional tags for service categorization
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=ServiceLifetime.RESOURCE,
            factory=factory,
            initializer=initializer,
            finalizer=finalizer,
            tags=tags or set()
        )
        
        self.logger.debug(f"Registered resource: {interface.__name__}")
    
    def register_metrics_collector_factory(self, collector_type: str = "prometheus") -> None:
        """Register factory for metrics collectors optimized for ML workloads.
        
        Args:
            collector_type: Type of metrics collector (prometheus, opentelemetry, etc.)
        """
        def create_metrics_collector() -> MetricsRegistryProtocol:
            if collector_type == "prometheus":
                from ...performance.monitoring.metrics_registry import MetricsRegistry
                return MetricsRegistry()
            else:
                raise ValueError(f"Unsupported metrics collector type: {collector_type}")
        
        self.register_factory(
            MetricsRegistryProtocol,
            create_metrics_collector,
            ServiceLifetime.SINGLETON,
            tags={"metrics", "monitoring"}
        )
        
        self.logger.debug(f"Registered metrics collector factory: {collector_type}")
    
    def register_health_monitor_factory(self) -> None:
        """Register factory for health monitors with circuit breaker patterns."""
        def create_health_monitor():
            from ...performance.monitoring.health.service import HealthService
            return HealthService()
        
        from ...performance.monitoring.health.service import HealthService
        self.register_factory(
            HealthService,
            create_health_monitor,
            ServiceLifetime.SINGLETON,
            tags={"health", "monitoring"}
        )
        
        self.logger.debug("Registered health monitor factory")
    
    def register_ml_model_factory(self, model_type: str = "sklearn") -> None:
        """Register factory for ML model instances with proper lifecycle.
        
        Args:
            model_type: Type of ML framework (sklearn, tensorflow, pytorch, etc.)
        """
        def create_ml_model():
            # Factory will be implemented based on model_type
            # This is a placeholder for the pattern
            if model_type == "sklearn":
                from ...ml.lifecycle.model_registry import ModelRegistry
                return ModelRegistry()
            else:
                raise ValueError(f"Unsupported ML model type: {model_type}")
        
        from ...ml.lifecycle.model_registry import ModelRegistry
        self.register_factory(
            ModelRegistry,
            create_ml_model,
            ServiceLifetime.SINGLETON,
            tags={"ml", "model"},
            finalizer=lambda instance: getattr(instance, 'cleanup', lambda: None)()
        )
        
        self.logger.debug(f"Registered ML model factory: {model_type}")
    
    @asynccontextmanager
    async def scope(self, scope_id: str):
        """Create a scoped context for scoped services.
        
        Args:
            scope_id: Unique identifier for this scope
        """
        self._scoped_services[scope_id] = {}
        try:
            yield
        finally:
            # Clean up scoped services
            scoped_services = self._scoped_services.pop(scope_id, {})
            for service in scoped_services.values():
                if hasattr(service, 'cleanup') and callable(service.cleanup):
                    try:
                        if inspect.iscoroutinefunction(service.cleanup):
                            await service.cleanup()
                        else:
                            service.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up scoped service: {e}")
    
    def set_metrics_registry(self, metrics_registry: MetricsRegistryProtocol) -> None:
        """Set the metrics registry for performance tracking.
        
        Args:
            metrics_registry: Metrics registry implementation
        """
        self._metrics_registry = metrics_registry
        self.logger.debug("Metrics registry set for DI container")
    
    def get_services_by_tag(self, tag: str) -> Dict[Type, ServiceRegistration]:
        """Get all services registered with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            Dictionary of service types to their registrations
        """
        return {
            service_type: registration
            for service_type, registration in self._services.items()
            if tag in registration.tags
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the DI container.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "resolution_times": dict(self._resolution_times),
            "initialization_order": [t.__name__ for t in self._initialization_order],
            "registered_services_count": len(self._services),
            "scoped_contexts_count": len(self._scoped_services),
            "active_resources_count": len(self._resources)
        }
    
    async def shutdown(self):
        """Enhanced shutdown with comprehensive resource cleanup."""
        self.logger.info(f"Shutting down enhanced DI container '{self.name}'")
        
        # Shutdown resources first (in reverse order of creation)
        for resource in reversed(list(self._resources)):
            try:
                # Find the registration for this resource
                for registration in self._services.values():
                    if registration.instance is resource and registration.finalizer:
                        if inspect.iscoroutinefunction(registration.finalizer):
                            await registration.finalizer(resource)
                        else:
                            registration.finalizer(resource)
                        break
                else:
                    # Fallback to generic shutdown if available
                    if hasattr(resource, 'shutdown') and callable(resource.shutdown):
                        if inspect.iscoroutinefunction(resource.shutdown):
                            await resource.shutdown()
                        else:
                            resource.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down resource {type(resource).__name__}: {e}")
        
        # Shutdown singleton services (in reverse initialization order)
        for interface in reversed(self._initialization_order):
            registration = self._services.get(interface)
            if registration and registration.instance:
                try:
                    # Use finalizer if available
                    if registration.finalizer:
                        if inspect.iscoroutinefunction(registration.finalizer):
                            await registration.finalizer(registration.instance)
                        else:
                            registration.finalizer(registration.instance)
                    # Fallback to generic shutdown
                    elif hasattr(registration.instance, 'shutdown') and callable(registration.instance.shutdown):
                        if inspect.iscoroutinefunction(registration.instance.shutdown):
                            await registration.instance.shutdown()
                        else:
                            registration.instance.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down service {interface.__name__}: {e}")
        
        # Clean up all scoped services
        for scope_id, scoped_services in self._scoped_services.items():
            self.logger.debug(f"Cleaning up scoped services for scope: {scope_id}")
            for service in scoped_services.values():
                try:
                    if hasattr(service, 'cleanup') and callable(service.cleanup):
                        if inspect.iscoroutinefunction(service.cleanup):
                            await service.cleanup()
                        else:
                            service.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up scoped service: {e}")
        
        # Clear all collections
        self._services.clear()
        self._scoped_services.clear()
        self._resources.clear()
        self._resolution_stack.clear()
        self._resolution_times.clear()
        self._initialization_order.clear()
        self._health_checks.clear()
        
        self.logger.info(f"Enhanced DI container '{self.name}' shutdown complete")

# Global container instance
_container: Optional[DIContainer] = None

async def get_container() -> DIContainer:
    """Get the global DI container instance.
    
    Returns:
        DIContainer: Global container instance
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container

async def get_datetime_service() -> DateTimeServiceProtocol:
    """Get datetime service instance.
    
    Convenience function for getting the datetime service.
    
    Returns:
        DateTimeServiceProtocol: DateTime service instance
    """
    container = await get_container()
    return await container.get(DateTimeServiceProtocol)

async def shutdown_container():
    """Shutdown the global container."""
    global _container
    if _container:
        await _container.shutdown()
        _container = None
