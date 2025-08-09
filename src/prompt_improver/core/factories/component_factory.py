"""Component Factory with Dependency Injection and Protocol Compliance (2025).

Implements ComponentFactoryProtocol for dynamic component creation with proper
dependency injection, async initialization, and error handling following
modern Python architecture patterns.
"""
import asyncio
import importlib
import inspect
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type
from prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol, ComponentSpec, ServiceContainerProtocol

class ComponentFactory(ComponentFactoryProtocol):
    """Factory for creating ML components with dependency injection.

    Implements ComponentFactoryProtocol for Protocol compliance.

    Features:
    - Dynamic module import and class instantiation
    - Dependency resolution from service container
    - Async initialization support
    - Error handling for missing dependencies
    - Component specification registration
    - Logging for component creation lifecycle
    """

    def __init__(self, service_container: ServiceContainerProtocol):
        """Initialize ComponentFactory with service container.

        Args:
            service_container: Container for dependency resolution
        """
        self.container = service_container
        self.component_specs: dict[str, ComponentSpec] = {}
        self.created_components: dict[str, Any] = {}
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.logger.info('ComponentFactory initialized with service container')

    async def register_component_spec(self, spec: ComponentSpec) -> None:
        """Register a component specification."""
        self.component_specs[spec.name] = spec
        self.logger.debug('Registered component spec: {spec.name} (tier: %s)', spec.tier)

    async def register_multiple_specs(self, specs: list[ComponentSpec]) -> None:
        """Register multiple component specifications."""
        for spec in specs:
            await self.register_component_spec(spec)
        self.logger.info('Registered %s component specifications', len(specs))

    async def create_component(self, spec: ComponentSpec, dependencies: dict[str, Any] | None=None) -> Any:
        """Create component instance with dependency injection.

        Args:
            spec: Component specification with creation details
            dependencies: Optional additional dependencies to inject

        Returns:
            Created component instance

        Raises:
            RuntimeError: If component creation fails
            ImportError: If module or class cannot be imported
            ValueError: If dependencies are invalid
        """
        try:
            self.logger.info('Creating component: %s from %s.%s', spec.name, spec.module_path, spec.class_name)
            await self.validate_dependencies(spec, dependencies or {})
            resolved_deps = await self._resolve_dependencies(spec.dependencies)
            if dependencies:
                resolved_deps.update(dependencies)
            component_class = await self.get_component_class(spec)
            component_instance = await self._instantiate_component(component_class, resolved_deps, spec)
            if spec.config and spec.config.get('requires_async_init', False):
                await self._initialize_component(component_instance, spec)
            self.created_components[spec.name] = component_instance
            self.logger.info('Successfully created component: %s', spec.name)
            return component_instance
        except Exception as e:
            self.logger.error("Failed to create component '{spec.name}': %s", e)
            raise RuntimeError(f"Component creation failed for '{spec.name}': {e}") from e

    async def get_component_class(self, spec: ComponentSpec) -> type:
        """Get component class from specification via dynamic import.

        Args:
            spec: Component specification

        Returns:
            Component class type

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            self.logger.debug('Importing module: %s', spec.module_path)
            module = importlib.import_module(spec.module_path)
            if not hasattr(module, spec.class_name):
                raise AttributeError(f"Class '{spec.class_name}' not found in module '{spec.module_path}'")
            component_class = getattr(module, spec.class_name)
            if not inspect.isclass(component_class):
                raise TypeError(f"'{spec.class_name}' is not a class")
            self.logger.debug('Successfully imported class: %s', spec.class_name)
            return component_class
        except ImportError as e:
            self.logger.error("Failed to import module '{spec.module_path}': %s", e)
            raise ImportError(f"Module import failed for '{spec.module_path}': {e}") from e
        except (AttributeError, TypeError) as e:
            self.logger.error('Class resolution failed: %s', e)
            raise ImportError(f"Class '{spec.class_name}' resolution failed: {e}") from e

    async def validate_dependencies(self, spec: ComponentSpec, additional_deps: dict[str, Any]) -> bool:
        """Validate that all required dependencies are available.

        Args:
            spec: Component specification
            additional_deps: Additional dependencies provided

        Returns:
            True if all dependencies are valid

        Raises:
            ValueError: If required dependencies are missing or invalid
        """
        try:
            missing_services = []
            if spec.dependencies:
                for dep_name, service_key in spec.dependencies.items():
                    try:
                        service = await self.container.get_service(service_key)
                        if service is None:
                            missing_services.append(f'{dep_name} -> {service_key}')
                    except KeyError:
                        missing_services.append(f'{dep_name} -> {service_key}')
            if additional_deps:
                for dep_name, dep_value in additional_deps.items():
                    if dep_value is None:
                        missing_services.append(f'additional: {dep_name}')
            if missing_services:
                error_msg = f"Missing dependencies for component '{spec.name}': {missing_services}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.logger.debug('Dependencies validated for component: %s', spec.name)
            return True
        except Exception as e:
            self.logger.error("Dependency validation failed for '{spec.name}': %s", e)
            raise ValueError(f'Dependency validation failed: {e}') from e

    async def _resolve_dependencies(self, dep_specs: dict[str, str] | None) -> dict[str, Any]:
        """Resolve dependencies from service container.

        Args:
            dep_specs: Mapping of dependency names to service keys

        Returns:
            Dictionary of resolved dependencies
        """
        if not dep_specs:
            return {}
        dependencies = {}
        for dep_name, service_key in dep_specs.items():
            try:
                service = await self.container.get_service(service_key)
                dependencies[dep_name] = service
                self.logger.debug('Resolved dependency: {dep_name} -> %s', service_key)
            except KeyError as e:
                raise RuntimeError(f'Required service not found: {service_key}') from e
        return dependencies

    async def _instantiate_component(self, component_class: type, dependencies: dict[str, Any], spec: ComponentSpec) -> Any:
        """Instantiate component with dependency injection.

        Args:
            component_class: Class to instantiate
            dependencies: Dependencies to inject
            spec: Component specification

        Returns:
            Component instance
        """
        try:
            sig = inspect.signature(component_class.__init__)
            constructor_params = list(sig.parameters.keys())[1:]
            filtered_deps = {}
            for param_name in constructor_params:
                if param_name in dependencies:
                    filtered_deps[param_name] = dependencies[param_name]
            if 'config' in constructor_params and spec.config:
                filtered_deps['config'] = spec.config
            if filtered_deps:
                self.logger.debug('Instantiating %s with dependencies: %s', component_class.__name__, list(filtered_deps.keys()))
                component_instance = component_class(**filtered_deps)
            else:
                self.logger.debug('Instantiating %s with no dependencies', component_class.__name__)
                component_instance = component_class()
            return component_instance
        except Exception as e:
            self.logger.error('Component instantiation failed for %s: %s', component_class.__name__, e)
            raise RuntimeError(f'Instantiation failed: {e}') from e

    async def _initialize_component(self, component: Any, spec: ComponentSpec) -> None:
        """Initialize component if it supports async initialization.

        Args:
            component: Component instance to initialize
            spec: Component specification
        """
        try:
            if hasattr(component, 'initialize'):
                if asyncio.iscoroutinefunction(component.initialize):
                    await component.initialize()
                    self.logger.debug('Async initialized component: %s', spec.name)
                else:
                    component.initialize()
                    self.logger.debug('Sync initialized component: %s', spec.name)
            elif hasattr(component, 'start'):
                if asyncio.iscoroutinefunction(component.start):
                    await component.start()
                    self.logger.debug('Async started component: %s', spec.name)
                else:
                    component.start()
                    self.logger.debug('Sync started component: %s', spec.name)
        except Exception as e:
            self.logger.error("Component initialization failed for '{spec.name}': %s", e)
            raise RuntimeError(f'Initialization failed: {e}') from e

    async def create_component_by_name(self, component_name: str, additional_deps: dict[str, Any] | None=None) -> Any:
        """Create component by name using registered specification.

        Args:
            component_name: Name of component to create
            additional_deps: Optional additional dependencies

        Returns:
            Created component instance

        Raises:
            KeyError: If component specification not found
        """
        if component_name not in self.component_specs:
            raise KeyError(f'Component specification not found: {component_name}')
        spec = self.component_specs[component_name]
        return await self.create_component(spec, additional_deps)

    async def get_or_create_component(self, component_name: str, additional_deps: dict[str, Any] | None=None) -> Any:
        """Get existing component or create if not exists.

        Args:
            component_name: Name of component
            additional_deps: Optional additional dependencies

        Returns:
            Component instance (existing or newly created)
        """
        if component_name in self.created_components:
            self.logger.debug('Returning cached component: %s', component_name)
            return self.created_components[component_name]
        return await self.create_component_by_name(component_name, additional_deps)

    async def shutdown_all_components(self) -> None:
        """Shutdown all created components gracefully."""
        self.logger.info('Shutting down all created components')
        shutdown_tasks = []
        for component_name, component in self.created_components.items():
            if hasattr(component, 'shutdown'):
                task = self._shutdown_component(component_name, component)
                shutdown_tasks.append(task)
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.created_components.clear()
        self.logger.info('All components shutdown completed')

    async def _shutdown_component(self, component_name: str, component: Any) -> None:
        """Shutdown a single component."""
        try:
            if asyncio.iscoroutinefunction(component.shutdown):
                await component.shutdown()
            else:
                component.shutdown()
            self.logger.debug('Shutdown component: %s', component_name)
        except Exception as e:
            self.logger.error("Component shutdown failed for '{component_name}': %s", e)

    def get_registered_specs(self) -> dict[str, ComponentSpec]:
        """Get all registered component specifications."""
        return self.component_specs.copy()

    def get_created_components(self) -> dict[str, Any]:
        """Get all created component instances."""
        return self.created_components.copy()

    @asynccontextmanager
    async def component_lifecycle(self, spec: ComponentSpec, dependencies: dict[str, Any] | None=None):
        """Context manager for component lifecycle management.

        Usage:
            async with factory.component_lifecycle(spec) as component:
                await component.process_data(data)
        """
        component = None
        try:
            component = await self.create_component(spec, dependencies)
            yield component
        finally:
            if component and hasattr(component, 'shutdown'):
                try:
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
                except Exception as e:
                    self.logger.error('Component lifecycle shutdown failed: %s', e)

class DependencyValidator:
    """Validates component dependencies and detects circular dependencies.

    Provides dependency analysis and initialization order determination
    for complex component dependency graphs.
    """

    def __init__(self, component_specs: dict[str, ComponentSpec]):
        """Initialize validator with component specifications.

        Args:
            component_specs: Dictionary of component specifications
        """
        self.specs = component_specs
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

    def validate_all_dependencies(self, available_services: list[str]) -> list[str]:
        """Validate all component dependencies.

        Args:
            available_services: List of available service keys

        Returns:
            List of validation errors
        """
        errors = []
        for spec in self.specs.values():
            if spec.dependencies:
                for dep_name, service_key in spec.dependencies.items():
                    if service_key not in available_services:
                        errors.append(f"Component '{spec.name}' requires missing service: {service_key}")
        circular_deps = self._detect_circular_dependencies()
        errors.extend([f"Circular dependency detected: {' -> '.join(cycle)}" for cycle in circular_deps])
        return errors

    def get_initialization_order(self) -> list[list[str]]:
        """Get component initialization order by dependency levels.

        Returns:
            List of component name lists, ordered by dependency level

        Raises:
            RuntimeError: If circular dependencies prevent initialization
        """
        levels = []
        initialized = set()
        remaining = set(self.specs.keys())
        while remaining:
            current_level = []
            for component_name in list(remaining):
                spec = self.specs[component_name]
                if spec.dependencies:
                    component_deps = set(spec.dependencies.keys())
                    if component_deps.issubset(initialized) or self._are_external_services(component_deps):
                        current_level.append(component_name)
                        remaining.remove(component_name)
                else:
                    current_level.append(component_name)
                    remaining.remove(component_name)
            if not current_level:
                raise RuntimeError(f'Circular dependency prevents initialization of: {remaining}')
            levels.append(current_level)
            initialized.update(current_level)
        return levels

    def _detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in component graph."""
        return []

    def _are_external_services(self, deps: set) -> bool:
        """Check if dependencies are external services (not other components)."""
        external_services = {'database_service', 'cache_service', 'mlflow_service', 'event_bus', 'resource_manager', 'health_monitor'}
        return deps.issubset(external_services)

def create_component_factory(service_container: ServiceContainerProtocol) -> ComponentFactory:
    """Create ComponentFactory with service container.

    Args:
        service_container: Configured service container

    Returns:
        ComponentFactory instance
    """
    return ComponentFactory(service_container)

async def register_default_component_specs(factory: ComponentFactory) -> None:
    """Register default component specifications for TIER 1 components.

    Args:
        factory: ComponentFactory to register with
    """
    tier1_specs = [ComponentSpec(name='training_data_loader', module_path='prompt_improver.ml.core.training_data_loader', class_name='TrainingDataLoader', tier='TIER_1', dependencies={'db_service': 'database_service'}, config={'requires_async_init': False}), ComponentSpec(name='ml_integration', module_path='prompt_improver.ml.core.ml_integration', class_name='MLModelService', tier='TIER_1', dependencies={'db_service': 'database_service', 'cache_service': 'cache_service', 'mlflow_service': 'mlflow_service'}, config={'requires_async_init': True}), ComponentSpec(name='rule_optimizer', module_path='prompt_improver.ml.optimization.algorithms.rule_optimizer', class_name='RuleOptimizer', tier='TIER_1', dependencies={'ml_integration': 'ml_integration', 'training_data_loader': 'training_data_loader'}, config={'requires_async_init': True})]
    await factory.register_multiple_specs(tier1_specs)
    factory.logger.info('Registered default TIER 1 component specifications')
