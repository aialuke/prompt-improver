"""
Direct Component Loader for ML Pipeline Orchestrator.

Loads actual ML components and makes them available for orchestration.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol, ComponentRegistryProtocol, ComponentSpec
from ..core.component_registry import ComponentInfo
from ..shared.component_types import ComponentTier

@dataclass
class LoadedComponent:
    """Represents a loaded ML component with its metadata."""
    name: str
    component_class: type
    instance: Any | None = None
    module_path: str = ''
    dependencies: list[str] = None
    is_initialized: bool = False

class DirectComponentLoader:
    """
    Loads ML components directly from the codebase for orchestration.
    
    This provides actual component integration rather than placeholder connectors.
    """

    def __init__(self):
        """Initialize the direct component loader."""
        self.logger = logging.getLogger(__name__)
        self.loaded_components: dict[str, LoadedComponent] = {}
        self.component_registry: ComponentRegistryProtocol | None = None
        self.component_factory: ComponentFactoryProtocol | None = None
        self.component_specs: dict[str, ComponentSpec] = {}
        self.tier_specifications = {ComponentTier.TIER_1: 'critical_path_components', ComponentTier.TIER_2: 'important_components', ComponentTier.TIER_3: 'optional_components'}

    def set_component_registry(self, registry: ComponentRegistryProtocol) -> None:
        """Set the component registry for discovery."""
        self.component_registry = registry

    def set_component_factory(self, factory: ComponentFactoryProtocol) -> None:
        """Set the component factory for instantiation."""
        self.component_factory = factory

    async def discover_components(self, tier: ComponentTier | None=None) -> list[ComponentSpec]:
        """
        Discover available components using registry.
        
        Args:
            tier: Optional tier filter
            
        Returns:
            List of discovered component specifications
        """
        if not self.component_registry:
            raise RuntimeError('Component registry not injected')
        tier_str = tier.value if tier else None
        discovered_specs = await self.component_registry.discover_components(tier_str)
        for spec in discovered_specs:
            self.component_specs[spec.name] = spec
        self.logger.info('Discovered {len(discovered_specs)} components for tier %s', tier)
        return discovered_specs

    async def load_component(self, component_name: str, dependencies: dict[str, Any] | None=None) -> LoadedComponent | None:
        """
        Load component using factory pattern with dependency injection.
        
        Args:
            component_name: Name of component to load
            dependencies: Optional dependencies to inject
            
        Returns:
            LoadedComponent instance or None if loading failed
        """
        if component_name in self.loaded_components:
            return self.loaded_components[component_name]
        spec = self.component_specs.get(component_name)
        if not spec:
            await self.discover_components()
            spec = self.component_specs.get(component_name)
        if not spec:
            self.logger.error('Component specification for %s not found', component_name)
            return None
        if not self.component_factory:
            raise RuntimeError('Component factory not injected')
        try:
            component_instance = await self.component_factory.create_component(spec=spec, dependencies=dependencies or {})
            loaded_component = LoadedComponent(name=component_name, component_class=type(component_instance), instance=component_instance, module_path=spec.module_path, dependencies=list(spec.dependencies.keys()) if spec.dependencies else [], is_initialized=True)
            self.loaded_components[component_name] = loaded_component
            self.logger.info('Successfully loaded component: %s', component_name)
            return loaded_component
        except Exception as e:
            self.logger.error('Failed to load component {component_name}: %s', e)
            return None

    async def load_tier_components(self, tier: ComponentTier, dependencies: dict[str, Any] | None=None) -> dict[str, LoadedComponent]:
        """
        Load all components in a specific tier using modern factory pattern.
        
        Args:
            tier: Component tier to load
            dependencies: Optional dependencies to inject into all components
            
        Returns:
            Dictionary of component name to LoadedComponent
        """
        loaded_tier_components = {}
        try:
            tier_specs = await self.discover_components(tier)
            for spec in tier_specs:
                loaded_component = await self.load_component(spec.name, dependencies)
                if loaded_component:
                    loaded_tier_components[spec.name] = loaded_component
            self.logger.info('Loaded {len(loaded_tier_components)}/{len(tier_specs)} components for %s', tier)
            return loaded_tier_components
        except Exception as e:
            self.logger.error('Failed to load tier {tier} components: %s', e)
            return loaded_tier_components

    async def load_all_components(self) -> dict[str, LoadedComponent]:
        """
        Load all components across all tiers.
        
        Returns:
            Dictionary of all loaded components
        """
        all_loaded = {}
        for tier in ComponentTier:
            tier_components = await self.load_tier_components(tier)
            all_loaded.update(tier_components)
        self.logger.info('Loaded %s total components', len(all_loaded))
        return all_loaded

    def get_loaded_component(self, component_name: str) -> LoadedComponent | None:
        """Get a loaded component by name."""
        return self.loaded_components.get(component_name)

    def get_all_loaded_components(self) -> dict[str, LoadedComponent]:
        """Get all loaded components."""
        return self.loaded_components.copy()

    def is_component_loaded(self, component_name: str) -> bool:
        """Check if a component is loaded."""
        return component_name in self.loaded_components

    def is_component_initialized(self, component_name: str) -> bool:
        """Check if a component is loaded and initialized."""
        loaded_component = self.loaded_components.get(component_name)
        return loaded_component is not None and loaded_component.is_initialized
