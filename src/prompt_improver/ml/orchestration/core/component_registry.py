"""
Component Registry for ML Pipeline orchestration.

Implements ComponentRegistryProtocol for Protocol compliance and modern 3-tier system support.
Manages registration, discovery, and health monitoring of all ML components.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid
from ....core.protocols.ml_protocols import ComponentRegistryProtocol, ComponentSpec, ServiceStatus
from ....performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
from ..config.orchestrator_config import OrchestratorConfig
from ..shared.component_types import ComponentCapability, ComponentInfo, ComponentTier

class ComponentStatus(Enum):
    """Component status states."""
    UNKNOWN = 'unknown'
    HEALTHY = 'healthy'
    unhealthy = 'unhealthy'
    starting = 'starting'
    stopping = 'stopping'
    ERROR = 'error'

class ComponentRegistry(ComponentRegistryProtocol):
    """
    Registry for all ML pipeline components implementing ComponentRegistryProtocol.

    Manages components across the modern 3-tier system:
    - TIER_1: Critical path components (core ML pipeline)
    - TIER_2: Important but not critical components (optimization & learning)  
    - TIER_3: Optional/experimental components (evaluation, performance, security)
    
    Features:
    - Protocol compliance for dependency injection
    - Component discovery and registration
    - Health monitoring
    - Capability tracking
    - Dependency management
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize the component registry with Protocol compliance."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.components: Dict[str, ComponentInfo] = {}
        self.components_by_tier: Dict[ComponentTier, List[str]] = {tier: [] for tier in ComponentTier}
        self.component_specs: Dict[str, ComponentSpec] = {}
        self.specs_by_tier: Dict[str, List[str]] = {'TIER_1': [], 'TIER_2': [], 'TIER_3': []}
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    async def initialize(self) -> None:
        """Initialize the component registry."""
        self.logger.info('Initializing component registry')
        await self._load_component_specifications()
        await self._load_component_definitions()
        await self._start_health_monitoring()
        self.logger.info('Component registry initialized with {len(self.component_specs)} specifications and %s components', len(self.components))

    async def shutdown(self) -> None:
        """Shutdown the component registry."""
        self.logger.info('Shutting down component registry')
        await self._stop_health_monitoring()
        self.logger.info('Component registry shutdown complete')

    async def discover_components(self, tier: Optional[str]=None) -> List[ComponentSpec]:
        """
        Discover available components, optionally filtered by tier.
        
        Args:
            tier: Optional tier filter ("TIER_1", "TIER_2", "TIER_3")
            
        Returns:
            List of discovered component specifications
        """
        self.logger.info('Discovering components for tier: %s', tier or 'all')
        try:
            if not self.component_specs:
                await self._load_component_specifications()
            if tier:
                if tier in self.specs_by_tier:
                    component_names = self.specs_by_tier[tier]
                    return [self.component_specs[name] for name in component_names if name in self.component_specs]
                else:
                    self.logger.warning('Invalid tier specified: %s', tier)
                    return []
            return list(self.component_specs.values())
        except Exception as e:
            self.logger.error('Component discovery failed: %s', e)
            return []

    async def register_component(self, spec: ComponentSpec) -> None:
        """
        Register a component specification (Protocol method).
        
        Args:
            spec: Component specification to register
        """
        try:
            name = spec.name
            if name in self.component_specs:
                self.logger.warning('Component spec %s already registered, updating', name)
            self.component_specs[name] = spec
            if spec.tier in self.specs_by_tier:
                if name not in self.specs_by_tier[spec.tier]:
                    self.specs_by_tier[spec.tier].append(name)
            self.logger.info('Registered component spec {name} in %s', spec.tier)
        except Exception as e:
            self.logger.error('Failed to register component spec {spec.name}: %s', e)
            raise

    async def get_component_spec(self, component_name: str) -> Optional[ComponentSpec]:
        """
        Get component specification by name (Protocol method).
        
        Args:
            component_name: Name of component to retrieve
            
        Returns:
            Component specification if found
        """
        return self.component_specs.get(component_name)

    async def list_components_by_tier(self, tier: str) -> List[ComponentSpec]:
        """
        List all components for a specific tier (Protocol method).
        
        Args:
            tier: Tier identifier ("TIER_1", "TIER_2", "TIER_3")
            
        Returns:
            List of component specifications for the tier
        """
        if tier not in self.specs_by_tier:
            self.logger.warning('Invalid tier: %s', tier)
            return []
        component_names = self.specs_by_tier[tier]
        return [self.component_specs[name] for name in component_names if name in self.component_specs]

    async def register_component_info(self, component_info: ComponentInfo) -> None:
        """
        Register a new ML component.

        Args:
            component_info: Component information and capabilities
        """
        name = component_info.name
        if name in self.components:
            self.logger.warning('Component %s already registered, updating', name)
        self.components[name] = component_info
        self.components_by_tier[component_info.tier].append(name)
        self.logger.info('Registered component {name} in %s', component_info.tier.value)
        spec = self._convert_info_to_spec(component_info)
        await self.register_component(spec)

    def _convert_info_to_spec(self, info: ComponentInfo) -> ComponentSpec:
        """Convert ComponentInfo to ComponentSpec for Protocol compatibility."""
        return ComponentSpec(name=info.name, module_path=info.config.get('module_path', ''), class_name=info.config.get('class_name', ''), tier=info.tier.value, dependencies=info.config.get('dependencies'), config=info.config, enabled=info.enabled if hasattr(info, 'enabled') else True)

    async def unregister_component(self, component_name: str) -> bool:
        """
        Unregister a component (both ComponentSpec and ComponentInfo).

        Args:
            component_name: Name of component to unregister

        Returns:
            True if component was found and removed
        """
        removed = False
        if component_name in self.component_specs:
            spec = self.component_specs[component_name]
            if spec.tier in self.specs_by_tier and component_name in self.specs_by_tier[spec.tier]:
                self.specs_by_tier[spec.tier].remove(component_name)
            del self.component_specs[component_name]
            removed = True
        if component_name in self.components:
            component_info = self.components[component_name]
            if component_name in self.components_by_tier[component_info.tier]:
                self.components_by_tier[component_info.tier].remove(component_name)
            del self.components[component_name]
            removed = True
        if removed:
            self.logger.info('Unregistered component %s', component_name)
        return removed

    async def get_component(self, component_name: str) -> Optional[ComponentInfo]:
        """Get component information by name."""
        return self.components.get(component_name)

    async def list_components(self, tier: Optional[ComponentTier]=None) -> List[ComponentInfo]:
        """
        List registered components.

        Args:
            tier: Filter by specific tier (optional)

        Returns:
            List of component information
        """
        if tier:
            component_names = self.components_by_tier[tier]
            return [self.components[name] for name in component_names]
        return list(self.components.values())

    async def get_components_by_capability(self, capability_name: str) -> List[ComponentInfo]:
        """
        Find components that provide a specific capability.

        Args:
            capability_name: Name of the capability to search for

        Returns:
            List of components that provide the capability
        """
        matching_components = []
        for component in self.components.values():
            for capability in component.capabilities:
                if capability.name == capability_name:
                    matching_components.append(component)
                    break
        return matching_components

    async def check_component_health(self, component_name: str) -> ComponentStatus:
        """
        Check the health of a specific component.

        Args:
            component_name: Name of component to check

        Returns:
            Current health status
        """
        component = self.components.get(component_name)
        if not component:
            return ComponentStatus.UNKNOWN
        try:
            status = await self._perform_health_check(component)
            component.status = status.value
            component.last_health_check = datetime.now(timezone.utc)
            component.error_message = None
            return status
        except Exception as e:
            component.status = ComponentStatus.ERROR.value
            component.error_message = str(e)
            component.last_health_check = datetime.now(timezone.utc)
            self.logger.error('Health check failed for {component_name}: %s', e)
            return ComponentStatus.ERROR

    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get overall health summary of all components.

        Returns:
            Health summary with statistics
        """
        total_components = len(self.components)
        status_counts = {status.value: 0 for status in ComponentStatus}
        tier_health = {tier.value: {'total': 0, 'healthy': 0} for tier in ComponentTier}
        for component in self.components.values():
            status = component.status or ComponentStatus.UNKNOWN.value
            if status in status_counts:
                status_counts[status] += 1
            tier_health[component.tier.value]['total'] += 1
            if component.status == ComponentStatus.HEALTHY.value:
                tier_health[component.tier.value]['healthy'] += 1
        return {'total_components': total_components, 'status_distribution': status_counts, 'tier_health': tier_health, 'overall_health_percentage': status_counts['healthy'] / total_components * 100 if total_components > 0 else 0}

    async def discover_components(self) -> List[ComponentInfo]:
        """
        Discover ComponentInfo objects from the codebase.

        This method loads components from component definitions and registers them.

        Returns:
            List of discovered components
        """
        self.logger.info('Discovering ML components')
        discovered_components = []
        try:
            predefined_components = self._get_predefined_components()
            for component_name, component_data in predefined_components.items():
                try:
                    capability_enums = []
                    for cap in component_data['capabilities']:
                        if 'optimization' in cap.name.lower() or 'bayesian' in cap.name.lower():
                            capability_enums.append(ComponentCapability.OPTIMIZATION)
                        elif 'validation' in cap.name.lower() or 'statistical' in cap.name.lower():
                            capability_enums.append(ComponentCapability.VALIDATION)
                        elif 'model' in cap.name.lower() and 'training' in cap.name.lower():
                            capability_enums.append(ComponentCapability.MODEL_TRAINING)
                        elif 'deployment' in cap.name.lower():
                            capability_enums.append(ComponentCapability.DEPLOYMENT)
                        elif 'monitoring' in cap.name.lower():
                            capability_enums.append(ComponentCapability.MONITORING)
                        elif 'feature' in cap.name.lower():
                            capability_enums.append(ComponentCapability.FEATURE_ENGINEERING)
                        else:
                            capability_enums.append(ComponentCapability.DATA_PROCESSING)
                    component_info = ComponentInfo(name=component_name, tier=component_data['tier'], capabilities=list(set(capability_enums)), dependencies=component_data.get('dependencies', []), config={'description': component_data.get('description', ''), 'version': component_data.get('version', '1.0.0'), 'module_path': component_data.get('module_path', ''), 'class_name': component_data.get('class_name', '')})
                    discovered_components.append(component_info)
                    self.logger.debug('Discovered component: {component_name} (Tier: %s)', component_data['tier'].value)
                except Exception as e:
                    self.logger.warning('Failed to create ComponentInfo for {component_name}: %s', e)
                    continue
            self.logger.info('Discovered %s components', len(discovered_components))
        except Exception as e:
            self.logger.error('Legacy component discovery failed: %s', e)
        return discovered_components

    def _get_predefined_components(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined components following modern 3-tier system."""

        class SimpleCapability:

            def __init__(self, name: str, description: str=''):
                self.name = name
                self.description = description
        return {'ml_model_service': {'tier': ComponentTier.TIER_1, 'capabilities': [SimpleCapability('model_training', 'ML model training with production deployment')], 'description': 'Core ML model service', 'version': '1.0.0'}, 'rule_analyzer': {'tier': ComponentTier.TIER_2, 'capabilities': [SimpleCapability('optimization', 'Bayesian effectiveness analysis')], 'description': 'Rule analyzer with optimization capabilities', 'version': '1.1.0'}, 'automl_orchestrator': {'tier': ComponentTier.TIER_2, 'capabilities': [SimpleCapability('optimization', 'Automated ML optimization')], 'description': 'AutoML orchestration controller', 'version': '1.1.0'}, 'failure_analyzer': {'tier': ComponentTier.TIER_2, 'capabilities': [SimpleCapability('validation', 'Failure pattern analysis')], 'description': 'Advanced failure mode analyzer', 'version': '1.0.0'}, 'experiment_orchestrator': {'tier': ComponentTier.TIER_3, 'capabilities': [SimpleCapability('validation', 'Bayesian A/B testing')], 'description': 'Enhanced experiment orchestrator', 'version': '1.1.0'}, 'enhanced_quality_scorer': {'tier': ComponentTier.TIER_3, 'capabilities': [SimpleCapability('validation', 'Multi-dimensional quality assessment')], 'description': 'Advanced quality scoring system', 'version': '1.0.0'}, 'causal_inference_analyzer': {'tier': ComponentTier.TIER_3, 'capabilities': [SimpleCapability('validation', 'Advanced causal inference analysis')], 'description': 'Causal inference analyzer', 'version': '1.0.0'}}

    async def _load_component_specifications(self) -> None:
        """Load component specifications from configuration for Protocol compliance."""
        from ..config.component_definitions import ComponentDefinitions
        component_defs = ComponentDefinitions()
        all_definitions = component_defs.get_all_component_definitions()
        for component_name, definition in all_definitions.items():
            try:
                tier = self._determine_component_tier(component_name, component_defs)
                spec = ComponentSpec(name=component_name, module_path=self._get_module_path(definition.get('file_path', '')), class_name=self._get_class_name(component_name, definition), tier=tier, dependencies=self._convert_dependencies(definition.get('dependencies', [])), config=definition.get('local_config', {}), enabled=True)
                await self.register_component(spec)
            except Exception as e:
                self.logger.warning('Failed to create ComponentSpec for {component_name}: %s', e)
                continue
        total_specs = len(self.component_specs)
        tier_counts = {tier: len(names) for tier, names in self.specs_by_tier.items()}
        self.logger.info('Loaded {total_specs} component specifications: %s', tier_counts)

    async def _load_component_definitions(self) -> None:
        """Load component definitions from configuration."""
        from ..config.component_definitions import ComponentDefinitions
        component_defs = ComponentDefinitions()
        tier1_defs = component_defs.get_tier_components(ComponentTier.TIER_1)
        for name, definition in tier1_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_1)
            await self.register_component_info(component_info)
        tier2_defs = component_defs.get_tier_components(ComponentTier.TIER_2)
        for name, definition in tier2_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_2)
            await self.register_component_info(component_info)
        tier3_defs = component_defs.get_tier_components(ComponentTier.TIER_3)
        for name, definition in tier3_defs.items():
            component_info = component_defs.create_component_info(name, definition, ComponentTier.TIER_3)
            await self.register_component_info(component_info)
        total_components = len(tier1_defs) + len(tier2_defs) + len(tier3_defs)
        self.logger.info('Loaded definitions for {total_components} components ({len(tier1_defs)} Tier 1, {len(tier2_defs)} Tier 2, %s Tier 3)', len(tier3_defs))

    def _determine_component_tier(self, component_name: str, component_defs: Any) -> str:
        """Determine component tier based on component definitions."""
        if component_name in component_defs.tier1_core_components:
            return 'TIER_1'
        elif component_name in component_defs.tier2_optimization_components:
            return 'TIER_2'
        elif component_name in component_defs.tier3_evaluation_components or component_name in component_defs.tier4_performance_components or component_name in component_defs.tier6_security_components:
            return 'TIER_3'
        else:
            return 'TIER_3'

    def _get_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        if not file_path:
            return ''
        module_path = file_path.replace('/', '.').replace('.py', '')
        return f'prompt_improver.{module_path}'

    def _get_class_name(self, component_name: str, definition: Dict[str, Any]) -> str:
        """Get class name from component name or definition."""
        if 'class_name' in definition:
            return definition['class_name']
        words = component_name.split('_')
        return ''.join((word.capitalize() for word in words))

    def _convert_dependencies(self, dependencies: List[str]) -> Optional[Dict[str, str]]:
        """Convert dependency list to dependency mapping."""
        if not dependencies:
            return None
        return {dep: dep for dep in dependencies}

    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring of components."""
        if self.is_monitoring:
            return
        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.health_check_task_id = await task_manager.submit_enhanced_task(task_id=f'ml_component_registry_health_{str(uuid.uuid4())[:8]}', coroutine=self._health_monitoring_loop(), priority=TaskPriority.HIGH, tags={'service': 'ml', 'type': 'monitoring', 'component': 'component_registry_health', 'module': 'component_registry'})
        self.logger.info('Started component health monitoring')

    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        self.is_monitoring = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        self.logger.info('Stopped component health monitoring')

    async def _health_monitoring_loop(self) -> None:
        """Periodic health monitoring loop."""
        while self.is_monitoring:
            try:
                for component_name in self.components:
                    await self.check_component_health(component_name)
                await asyncio.sleep(self.config.component_health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error('Error in health monitoring loop: %s', e)
                await asyncio.sleep(5)

    async def _perform_health_check(self, component: ComponentInfo) -> ComponentStatus:
        """
        Perform health check for a component.

        Args:
            component: Component to check

        Returns:
            Health status
        """
        if component.health_check_endpoint:
            return ComponentStatus.HEALTHY
        if component.registered_at:
            time_since_registration = datetime.now(timezone.utc) - component.registered_at
            if time_since_registration.total_seconds() < 300:
                return ComponentStatus.HEALTHY
        return ComponentStatus.UNKNOWN
