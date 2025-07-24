"""Dependency injection container for the ML Pipeline Orchestrator.

This module provides a lightweight, async-compatible dependency injection
container following 2025 best practices for Python applications.

Features:
- Singleton and transient service lifetimes
- Async service initialization
- Type-safe service resolution
- Circular dependency detection
- Service health monitoring
- Easy testing with mock services
"""

import asyncio
import logging
from typing import Dict, Type, Any, TypeVar, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect

from ..interfaces.datetime_service import DateTimeServiceProtocol
from ..services.datetime_service import DateTimeService


T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"


@dataclass
class ServiceRegistration:
    """Service registration information."""
    interface: Type
    implementation: Type
    lifetime: ServiceLifetime
    factory: Optional[Callable] = None
    initialized: bool = False
    instance: Any = None


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve unregistered service."""
    pass


class DIContainer:
    """Lightweight dependency injection container.
    
    Provides service registration, resolution, and lifecycle management
    for the ML Pipeline Orchestrator system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the DI container.
        
        Args:
            logger: Optional logger for debugging and monitoring
        """
        self.logger = logger or logging.getLogger(__name__)
        self._services: Dict[Type, ServiceRegistration] = {}
        self._resolution_stack: Set[Type] = set()
        self._lock = asyncio.Lock()
        
        # Register default services
        self._register_default_services()
        
        self.logger.debug("DIContainer initialized")
    
    def _register_default_services(self):
        """Register default system services."""
        # Register datetime service as singleton
        self.register_singleton(DateTimeServiceProtocol, DateTimeService)
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a service as singleton.
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
        
        self.logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a service as transient (new instance each time).
        
        Args:
            interface: Service interface/protocol
            implementation: Concrete implementation class
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        
        self.logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], 
                        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON) -> None:
        """Register a service with custom factory function.
        
        Args:
            interface: Service interface/protocol
            factory: Factory function to create service instance
            lifetime: Service lifetime (singleton or transient)
        """
        self._services[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=lifetime,
            factory=factory
        )
        
        self.logger.debug(f"Registered factory: {interface.__name__} (lifetime: {lifetime.value})")
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a pre-created service instance.
        
        Args:
            interface: Service interface/protocol
            instance: Pre-created service instance
        """
        registration = ServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True,
            instance=instance
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
            ServiceNotRegisteredError: If service is not registered
            CircularDependencyError: If circular dependency detected
        """
        async with self._lock:
            return await self._resolve_service(interface)
    
    async def _resolve_service(self, interface: Type[T]) -> T:
        """Internal service resolution with circular dependency detection.
        
        Args:
            interface: Service interface to resolve
            
        Returns:
            Service instance
        """
        # Check if service is registered
        if interface not in self._services:
            raise ServiceNotRegisteredError(f"Service not registered: {interface.__name__}")
        
        # Check for circular dependencies
        if interface in self._resolution_stack:
            cycle = " -> ".join([t.__name__ for t in self._resolution_stack]) + f" -> {interface.__name__}"
            raise CircularDependencyError(f"Circular dependency detected: {cycle}")
        
        registration = self._services[interface]
        
        # Return existing singleton instance if available
        if (registration.lifetime == ServiceLifetime.SINGLETON and 
            registration.initialized and 
            registration.instance is not None):
            return registration.instance
        
        # Add to resolution stack for circular dependency detection
        self._resolution_stack.add(interface)
        
        try:
            # Create new instance
            if registration.factory:
                instance = await self._create_from_factory(registration.factory)
            else:
                instance = await self._create_from_class(registration.implementation)
            
            # Store singleton instance
            if registration.lifetime == ServiceLifetime.SINGLETON:
                registration.instance = instance
                registration.initialized = True
            
            self.logger.debug(f"Resolved service: {interface.__name__}")
            return instance
            
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
    
    async def shutdown(self):
        """Shutdown container and cleanup resources."""
        self.logger.info("Shutting down DI container")
        
        # Call shutdown on services that support it
        for registration in self._services.values():
            if (registration.instance and 
                hasattr(registration.instance, 'shutdown') and 
                callable(registration.instance.shutdown)):
                
                try:
                    if inspect.iscoroutinefunction(registration.instance.shutdown):
                        await registration.instance.shutdown()
                    else:
                        registration.instance.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down service: {e}")
        
        # Clear all registrations
        self._services.clear()
        self._resolution_stack.clear()


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
