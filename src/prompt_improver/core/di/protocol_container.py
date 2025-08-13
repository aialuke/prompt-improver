"""Protocol-based Dependency Injection Container.

This module provides a clean dependency injection framework that uses protocols
to eliminate circular import risks and enforce clean architecture boundaries.

CRITICAL: All dependency registration and resolution uses protocols only,
never concrete implementations, to ensure clean architecture compliance.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar, get_origin, get_args
from dataclasses import dataclass
from enum import Enum
import logging

from prompt_improver.core.domain.enums import HealthStatus


T = TypeVar('T')
P = TypeVar('P')


class LifecycleScope(Enum):
    """Dependency lifecycle management scopes."""
    SINGLETON = "singleton"      # One instance for entire application
    SCOPED = "scoped"           # One instance per scope (e.g., request)
    TRANSIENT = "transient"     # New instance every time


@dataclass
class DependencyRegistration:
    """Registration information for a dependency."""
    protocol_type: Type[Any]
    implementation_factory: Callable[..., Any]
    lifecycle: LifecycleScope
    dependencies: List[Type[Any]]
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResolutionContext:
    """Context for dependency resolution."""
    scope_id: str
    resolved_instances: Dict[str, Any]
    resolution_stack: List[str]
    metadata: Dict[str, Any]


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class DependencyNotFoundError(Exception):
    """Raised when a dependency cannot be resolved."""
    pass


class ProtocolContainer:
    """Protocol-based dependency injection container."""
    
    def __init__(self, name: str = "default"):
        """Initialize the protocol container.
        
        Args:
            name: Container name for identification
        """
        self.name = name
        self._registrations: Dict[str, DependencyRegistration] = {}
        self._singletons: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        self._logger = logging.getLogger(__name__)
        
    def register_protocol(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        lifecycle: LifecycleScope = LifecycleScope.TRANSIENT,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a protocol implementation.
        
        Args:
            protocol_type: Protocol interface to register
            implementation_factory: Factory function to create implementation
            lifecycle: Lifecycle management scope
            name: Optional name for named registration
            metadata: Optional metadata
        """
        if not self._is_protocol(protocol_type):
            raise ValueError(f"{protocol_type} is not a Protocol")
        
        # Analyze dependencies from factory signature
        dependencies = self._analyze_factory_dependencies(implementation_factory)
        
        registration_key = self._get_registration_key(protocol_type, name)
        
        registration = DependencyRegistration(
            protocol_type=protocol_type,
            implementation_factory=implementation_factory,
            lifecycle=lifecycle,
            dependencies=dependencies,
            name=name,
            metadata=metadata or {}
        )
        
        self._registrations[registration_key] = registration
        self._logger.info(f"Registered {protocol_type.__name__} with {lifecycle.value} lifecycle")
    
    def register_singleton(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a protocol implementation as singleton.
        
        Args:
            protocol_type: Protocol interface to register
            implementation_factory: Factory function to create implementation
            name: Optional name for named registration
            metadata: Optional metadata
        """
        self.register_protocol(
            protocol_type=protocol_type,
            implementation_factory=implementation_factory,
            lifecycle=LifecycleScope.SINGLETON,
            name=name,
            metadata=metadata
        )
    
    def register_scoped(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a protocol implementation as scoped.
        
        Args:
            protocol_type: Protocol interface to register
            implementation_factory: Factory function to create implementation
            name: Optional name for named registration
            metadata: Optional metadata
        """
        self.register_protocol(
            protocol_type=protocol_type,
            implementation_factory=implementation_factory,
            lifecycle=LifecycleScope.SCOPED,
            name=name,
            metadata=metadata
        )
    
    def resolve(
        self,
        protocol_type: Type[P],
        name: Optional[str] = None,
        scope_id: str = "default",
    ) -> P:
        """Resolve a protocol implementation.
        
        Args:
            protocol_type: Protocol interface to resolve
            name: Optional name for named resolution
            scope_id: Scope identifier for scoped dependencies
            
        Returns:
            Implementation instance
            
        Raises:
            DependencyNotFoundError: If dependency cannot be resolved
            CircularDependencyError: If circular dependency detected
        """
        context = ResolutionContext(
            scope_id=scope_id,
            resolved_instances={},
            resolution_stack=[],
            metadata={}
        )
        
        return self._resolve_with_context(protocol_type, name, context)
    
    def _resolve_with_context(
        self,
        protocol_type: Type[P],
        name: Optional[str],
        context: ResolutionContext,
    ) -> P:
        """Resolve dependency with resolution context.
        
        Args:
            protocol_type: Protocol to resolve
            name: Optional name
            context: Resolution context
            
        Returns:
            Resolved instance
        """
        registration_key = self._get_registration_key(protocol_type, name)
        
        # Check for circular dependencies
        if registration_key in context.resolution_stack:
            stack_str = " -> ".join(context.resolution_stack + [registration_key])
            raise CircularDependencyError(f"Circular dependency detected: {stack_str}")
        
        # Check if already resolved in this context
        if registration_key in context.resolved_instances:
            return context.resolved_instances[registration_key]
        
        # Get registration
        registration = self._registrations.get(registration_key)
        if not registration:
            raise DependencyNotFoundError(f"No registration found for {protocol_type.__name__}")
        
        # Handle different lifecycles
        if registration.lifecycle == LifecycleScope.SINGLETON:
            if registration_key in self._singletons:
                return self._singletons[registration_key]
        
        elif registration.lifecycle == LifecycleScope.SCOPED:
            scope_instances = self._scoped_instances.get(context.scope_id, {})
            if registration_key in scope_instances:
                return scope_instances[registration_key]
        
        # Add to resolution stack
        context.resolution_stack.append(registration_key)
        
        try:
            # Resolve dependencies
            dependency_instances = []
            for dep_protocol in registration.dependencies:
                dep_instance = self._resolve_with_context(dep_protocol, None, context)
                dependency_instances.append(dep_instance)
            
            # Create instance
            instance = registration.implementation_factory(*dependency_instances)
            
            # Cache based on lifecycle
            if registration.lifecycle == LifecycleScope.SINGLETON:
                self._singletons[registration_key] = instance
            elif registration.lifecycle == LifecycleScope.SCOPED:
                if context.scope_id not in self._scoped_instances:
                    self._scoped_instances[context.scope_id] = {}
                self._scoped_instances[context.scope_id][registration_key] = instance
            
            # Add to context
            context.resolved_instances[registration_key] = instance
            
            self._logger.debug(f"Resolved {protocol_type.__name__} ({registration.lifecycle.value})")
            return instance
            
        finally:
            # Remove from resolution stack
            context.resolution_stack.pop()
    
    def _is_protocol(self, obj: Any) -> bool:
        """Check if an object is a Protocol.
        
        Args:
            obj: Object to check
            
        Returns:
            Whether object is a Protocol
        """
        return (
            hasattr(obj, '__annotations__') and
            getattr(obj, '_is_protocol', False) or
            (hasattr(obj, '__protocol__') and obj.__protocol__)
        )
    
    def _analyze_factory_dependencies(
        self,
        factory: Callable[..., Any]
    ) -> List[Type[Any]]:
        """Analyze factory function to extract protocol dependencies.
        
        Args:
            factory: Factory function to analyze
            
        Returns:
            List of protocol dependencies
        """
        dependencies = []
        
        try:
            sig = inspect.signature(factory)
            for param in sig.parameters.values():
                if param.annotation != inspect.Parameter.empty:
                    if self._is_protocol(param.annotation):
                        dependencies.append(param.annotation)
        except Exception as e:
            self._logger.warning(f"Could not analyze dependencies for {factory}: {e}")
        
        return dependencies
    
    def _get_registration_key(
        self,
        protocol_type: Type[Any],
        name: Optional[str]
    ) -> str:
        """Get registration key for a protocol.
        
        Args:
            protocol_type: Protocol type
            name: Optional name
            
        Returns:
            Registration key
        """
        base_key = f"{protocol_type.__module__}.{protocol_type.__name__}"
        if name:
            return f"{base_key}#{name}"
        return base_key
    
    def clear_scope(self, scope_id: str) -> None:
        """Clear all scoped instances for a scope.
        
        Args:
            scope_id: Scope to clear
        """
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
            self._logger.debug(f"Cleared scope: {scope_id}")
    
    def get_registrations(self) -> List[DependencyRegistration]:
        """Get all registrations in this container.
        
        Returns:
            List of all registrations
        """
        return list(self._registrations.values())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the container.
        
        Returns:
            Health status information
        """
        total_registrations = len(self._registrations)
        singleton_instances = len(self._singletons)
        scoped_scopes = len(self._scoped_instances)
        total_scoped_instances = sum(len(instances) for instances in self._scoped_instances.values())
        
        health_status = HealthStatus.HEALTHY
        issues = []
        
        # Check for potential issues
        if total_registrations == 0:
            health_status = HealthStatus.DEGRADED
            issues.append("No dependencies registered")
        
        if total_scoped_instances > 1000:
            health_status = HealthStatus.DEGRADED
            issues.append("High number of scoped instances may indicate memory leak")
        
        return {
            "container_name": self.name,
            "health_status": health_status.value,
            "total_registrations": total_registrations,
            "singleton_instances": singleton_instances,
            "scoped_scopes": scoped_scopes,
            "total_scoped_instances": total_scoped_instances,
            "issues": issues
        }
    
    def validate_registrations(self) -> Dict[str, Any]:
        """Validate all registrations for potential issues.
        
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check for missing dependencies
        for reg_key, registration in self._registrations.items():
            for dep_protocol in registration.dependencies:
                dep_key = self._get_registration_key(dep_protocol, None)
                if dep_key not in self._registrations:
                    issues.append(f"Missing dependency: {dep_protocol.__name__} required by {reg_key}")
        
        # Check for potential circular dependencies
        try:
            # Try to build dependency graph
            self._build_dependency_graph()
        except CircularDependencyError as e:
            issues.append(f"Circular dependency detected: {e}")
        
        # Check lifecycle compatibility
        for registration in self._registrations.values():
            if registration.lifecycle == LifecycleScope.SINGLETON:
                for dep_protocol in registration.dependencies:
                    dep_key = self._get_registration_key(dep_protocol, None)
                    dep_reg = self._registrations.get(dep_key)
                    if dep_reg and dep_reg.lifecycle == LifecycleScope.SCOPED:
                        warnings.append(
                            f"Singleton {registration.protocol_type.__name__} "
                            f"depends on scoped {dep_protocol.__name__}"
                        )
        
        return {
            "validation_passed": len(issues) == 0,
            "total_issues": len(issues),
            "total_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings
        }
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for validation.
        
        Returns:
            Dependency graph
            
        Raises:
            CircularDependencyError: If circular dependency found
        """
        graph = {}
        
        for reg_key, registration in self._registrations.items():
            dependencies = []
            for dep_protocol in registration.dependencies:
                dep_key = self._get_registration_key(dep_protocol, None)
                dependencies.append(dep_key)
            graph[reg_key] = dependencies
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    raise CircularDependencyError(f"Circular dependency involving {node}")
        
        return graph


class ProtocolContainerBuilder:
    """Builder for creating and configuring protocol containers."""
    
    def __init__(self, name: str = "default"):
        """Initialize container builder.
        
        Args:
            name: Container name
        """
        self._container = ProtocolContainer(name)
        self._auto_register_enabled = False
    
    def enable_auto_registration(self) -> 'ProtocolContainerBuilder':
        """Enable automatic registration of protocols by convention.
        
        Returns:
            Builder instance for method chaining
        """
        self._auto_register_enabled = True
        return self
    
    def register(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        lifecycle: LifecycleScope = LifecycleScope.TRANSIENT,
        name: Optional[str] = None,
    ) -> 'ProtocolContainerBuilder':
        """Register a protocol implementation.
        
        Args:
            protocol_type: Protocol to register
            implementation_factory: Factory function
            lifecycle: Lifecycle scope
            name: Optional name
            
        Returns:
            Builder instance for method chaining
        """
        self._container.register_protocol(
            protocol_type=protocol_type,
            implementation_factory=implementation_factory,
            lifecycle=lifecycle,
            name=name
        )
        return self
    
    def register_singleton(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        name: Optional[str] = None,
    ) -> 'ProtocolContainerBuilder':
        """Register singleton implementation.
        
        Args:
            protocol_type: Protocol to register
            implementation_factory: Factory function
            name: Optional name
            
        Returns:
            Builder instance for method chaining
        """
        return self.register(protocol_type, implementation_factory, LifecycleScope.SINGLETON, name)
    
    def register_scoped(
        self,
        protocol_type: Type[P],
        implementation_factory: Callable[..., P],
        name: Optional[str] = None,
    ) -> 'ProtocolContainerBuilder':
        """Register scoped implementation.
        
        Args:
            protocol_type: Protocol to register
            implementation_factory: Factory function
            name: Optional name
            
        Returns:
            Builder instance for method chaining
        """
        return self.register(protocol_type, implementation_factory, LifecycleScope.SCOPED, name)
    
    def build(self) -> ProtocolContainer:
        """Build the configured container.
        
        Returns:
            Configured protocol container
        """
        # Validate container before returning
        validation = self._container.validate_registrations()
        if not validation["validation_passed"]:
            logging.warning(f"Container validation issues: {validation['issues']}")
        
        return self._container


# Global default container instance
_default_container: Optional[ProtocolContainer] = None


def get_default_container() -> ProtocolContainer:
    """Get the default global container instance.
    
    Returns:
        Default protocol container
    """
    global _default_container
    if _default_container is None:
        _default_container = ProtocolContainer("global_default")
    return _default_container


def set_default_container(container: ProtocolContainer) -> None:
    """Set the default global container.
    
    Args:
        container: Container to set as default
    """
    global _default_container
    _default_container = container


def resolve(
    protocol_type: Type[P],
    name: Optional[str] = None,
    container: Optional[ProtocolContainer] = None,
) -> P:
    """Convenience function to resolve from default container.
    
    Args:
        protocol_type: Protocol to resolve
        name: Optional name
        container: Optional specific container
        
    Returns:
        Resolved instance
    """
    if container is None:
        container = get_default_container()
    return container.resolve(protocol_type, name)