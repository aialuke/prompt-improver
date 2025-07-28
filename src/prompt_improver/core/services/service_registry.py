"""Service Registry Pattern for Circular Import Resolution (2025 Best Practice)

This module implements a modern service registry pattern that eliminates circular imports
by providing centralized service discovery and lazy instantiation.

Key Features:
- Lazy service instantiation
- Type-safe service registration
- Dependency injection support
- Circular import prevention
- Thread-safe singleton management
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceScope(Enum):
    """Service lifecycle scopes"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceDefinition:
    """Service definition with metadata"""
    factory: Callable[[], Any]
    scope: ServiceScope
    dependencies: list[str]
    instance: Optional[Any] = None
    initialized: bool = False

class ServiceRegistry:
    """
    Modern service registry implementing 2025 best practices for dependency management.
    
    Eliminates circular imports by providing:
    1. Lazy service instantiation
    2. Centralized service discovery
    3. Dependency injection
    4. Thread-safe singleton management
    """
    
    _instance: Optional['ServiceRegistry'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ServiceRegistry':
        """Singleton pattern with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._services: Dict[str, ServiceDefinition] = {}
            self._instances: Dict[str, Any] = {}
            self._initialization_lock = threading.Lock()
            self._initialized = True
            logger.info("Service registry initialized")
    
    def register(
        self,
        service_name: str,
        factory: Callable[[], T],
        scope: ServiceScope = ServiceScope.SINGLETON,
        dependencies: Optional[list[str]] = None
    ) -> None:
        """
        Register a service with the registry.
        
        Args:
            service_name: Unique service identifier
            factory: Factory function to create service instance
            scope: Service lifecycle scope
            dependencies: List of dependency service names
        """
        if dependencies is None:
            dependencies = []
            
        self._services[service_name] = ServiceDefinition(
            factory=factory,
            scope=scope,
            dependencies=dependencies
        )
        logger.debug(f"Registered service: {service_name} with scope: {scope.value}")
    
    def get(self, service_name: str) -> Any:
        """
        Get service instance with lazy instantiation.
        
        Args:
            service_name: Service identifier
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
        """
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")
        
        service_def = self._services[service_name]
        
        # Handle singleton scope
        if service_def.scope == ServiceScope.SINGLETON:
            if service_name in self._instances:
                return self._instances[service_name]
            
            with self._initialization_lock:
                # Double-check pattern
                if service_name in self._instances:
                    return self._instances[service_name]
                
                instance = self._create_instance(service_name, service_def)
                self._instances[service_name] = instance
                return instance
        
        # Handle transient scope
        elif service_def.scope == ServiceScope.TRANSIENT:
            return self._create_instance(service_name, service_def)
        
        # Handle scoped (for future extension)
        else:
            raise NotImplementedError(f"Scope {service_def.scope} not implemented")
    
    def _create_instance(self, service_name: str, service_def: ServiceDefinition) -> Any:
        """Create service instance with dependency injection"""
        try:
            # Resolve dependencies first
            dependencies = {}
            for dep_name in service_def.dependencies:
                dependencies[dep_name] = self.get(dep_name)
            
            # Create instance
            if dependencies:
                # If factory accepts dependencies, pass them
                instance = service_def.factory(**dependencies)
            else:
                instance = service_def.factory()
            
            logger.debug(f"Created instance for service: {service_name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance for service '{service_name}': {e}")
            raise
    
    def is_registered(self, service_name: str) -> bool:
        """Check if service is registered"""
        return service_name in self._services
    
    def clear(self) -> None:
        """Clear all services (mainly for testing)"""
        self._services.clear()
        self._instances.clear()
        logger.debug("Service registry cleared")

# Global registry instance
_registry = ServiceRegistry()

def register_service(
    name: str,
    factory: Callable[[], T],
    scope: ServiceScope = ServiceScope.SINGLETON,
    dependencies: Optional[list[str]] = None
) -> None:
    """Register a service with the global registry"""
    _registry.register(name, factory, scope, dependencies)

def get_service(name: str) -> Any:
    """Get service from the global registry"""
    return _registry.get(name)

def service_provider(
    name: str,
    scope: ServiceScope = ServiceScope.SINGLETON,
    dependencies: Optional[list[str]] = None
):
    """
    Decorator for automatic service registration.
    
    Usage:
        @service_provider("analytics_service")
        def create_analytics_service():
            return AnalyticsQueryInterface()
    """
    def decorator(factory_func: Callable[[], T]) -> Callable[[], T]:
        register_service(name, factory_func, scope, dependencies)
        
        @wraps(factory_func)
        def wrapper(*args, **kwargs):
            return get_service(name)
        
        return wrapper
    
    return decorator

# Convenience functions for common service types
def register_analytics_service(factory: Callable[[], Any]) -> None:
    """Register analytics service"""
    register_service("analytics", factory, ServiceScope.SINGLETON)

def register_real_time_analytics_service(factory: Callable[[], Any]) -> None:
    """Register real-time analytics service"""
    register_service("real_time_analytics", factory, ServiceScope.SINGLETON)

def get_analytics_service() -> Any:
    """Get analytics service"""
    return get_service("analytics")

def get_real_time_analytics_service() -> Any:
    """Get real-time analytics service"""
    return get_service("real_time_analytics")
