"""Component Manager Service.

Focused service responsible for component loading, lifecycle management, and registry operations.
Handles component discovery, initialization, health tracking, and method invocation.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ....core.protocols.ml_protocols import (
    ComponentFactoryProtocol,
    ComponentInvokerProtocol,
    ComponentLoaderProtocol,
    ComponentRegistryProtocol,
    EventBusProtocol,
)
from ..core.orchestrator_service_protocols import ComponentManagerProtocol
from ..events.event_types import EventType, MLEvent
from ..integration.direct_component_loader import DirectComponentLoader


class ComponentService:
    """
    Component Service.
    
    Responsible for:
    - Component discovery and registration
    - Component loading and initialization
    - Component health tracking and lifecycle management
    - Component method invocation and coordination
    """

    def __init__(
        self,
        component_registry: ComponentRegistryProtocol,
        component_factory: ComponentFactoryProtocol,
        event_bus: EventBusProtocol,
        component_loader: ComponentLoaderProtocol | None = None,
        component_invoker: ComponentInvokerProtocol | None = None,
    ):
        """Initialize ComponentManager with required dependencies.
        
        Args:
            component_registry: Component registration and discovery
            component_factory: Factory for creating components
            event_bus: Event bus for inter-component communication
            component_loader: Optional component loader for direct integration
            component_invoker: Optional component invoker for method calls
        """
        self.component_registry = component_registry
        self.component_factory = component_factory
        self.event_bus = event_bus
        self.component_loader = component_loader
        self.component_invoker = component_invoker
        self.logger = logging.getLogger(__name__)
        
        # Component health tracking
        self.component_health: dict[str, bool] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the component manager."""
        if self._is_initialized:
            return
        
        self.logger.info("Initializing Component Manager")
        
        try:
            # Initialize core component services
            await self.component_registry.initialize()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self._is_initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="component_manager",
                data={"component": "component_manager", "timestamp": datetime.now(timezone.utc).isoformat()}
            ))
            
            self.logger.info("Component Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize component manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the component manager."""
        self.logger.info("Shutting down Component Manager")
        
        try:
            # Shutdown component services
            await self.component_registry.shutdown()
            
            self._is_initialized = False
            
            self.logger.info("Component Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during component manager shutdown: {e}")
            raise

    async def discover_components(self) -> None:
        """Discover and register all ML components."""
        self.logger.info("Discovering ML components across all tiers")
        
        try:
            # Discover components from all 6 tiers
            components = await self.component_registry.discover_components()
            
            # Initialize component health tracking
            for component in components:
                self.component_health[component.name] = False  # Initially unhealthy until verified
            
            self.logger.info(f"Discovered {len(components)} ML components")
            
        except Exception as e:
            self.logger.error(f"Failed to discover components: {e}")
            raise

    async def load_direct_components(self) -> None:
        """Load ML components directly for Phase 6 integration."""
        self.logger.info("Loading direct ML components (Phase 6)")
        
        if not self.component_loader:
            self.logger.warning("Component loader not available, skipping direct component loading")
            return
        
        try:
            # Load all components across all tiers
            loaded_components = await self.component_loader.load_all_components()
            
            self.logger.info(f"Successfully loaded {len(loaded_components)} ML components")
            
            # Initialize core components for immediate use
            await self._initialize_core_components(loaded_components)
            
            # Initialize evaluation components for workflows
            await self._initialize_evaluation_components(loaded_components)
            
            # Emit component loading complete event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.COMPONENT_REGISTERED,
                source="component_manager",
                data={
                    "loaded_components": len(loaded_components),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to load direct components: {e}")
            raise

    async def get_component_health(self) -> dict[str, bool]:
        """Get health status of all registered components."""
        # Get current health from component registry
        components = await self.component_registry.list_components()
        health_status = {}
        
        for component in components:
            # Convert component status to boolean health status
            health_status[component.name] = component.status.value in ["healthy", "starting"]
        
        return health_status

    async def invoke_component(self, component_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        Invoke a method on a loaded ML component.
        
        Args:
            component_name: Name of the component
            method_name: Method to invoke
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the component method invocation
        """
        if not self._is_initialized:
            raise RuntimeError("Component manager not initialized")
        
        if not self.component_invoker:
            raise RuntimeError("Component invoker not available")
        
        result = await self.component_invoker.invoke_component_method(
            component_name, method_name, *args, **kwargs
        )
        
        if not result.success:
            self.logger.error(f"Component invocation failed: {result.error}")
            raise RuntimeError(f"Component {component_name}.{method_name} failed: {result.error}")
        
        return result.result

    def get_loaded_components(self) -> list[str]:
        """Get list of loaded component names."""
        if not self.component_loader:
            self.logger.warning("Component loader not available")
            return []
        
        return list(self.component_loader.get_all_loaded_components().keys())

    def get_component_methods(self, component_name: str) -> list[str]:
        """Get available methods for a component."""
        if not self.component_invoker:
            self.logger.warning("Component invoker not available")
            return []
        
        return self.component_invoker.get_available_methods(component_name)

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for component events."""
        # Component health events
        self.event_bus.subscribe(EventType.COMPONENT_HEALTH_CHANGED, self._handle_component_health)

    async def _handle_component_health(self, event: MLEvent) -> None:
        """Handle component health change events."""
        component_name = event.data.get("component_name")
        is_healthy = event.data.get("is_healthy", False)
        
        if component_name:
            self.component_health[component_name] = is_healthy
            self.logger.info(f"Component {component_name} health: {is_healthy}")

    async def _initialize_core_components(self, loaded_components: dict[str, Any]) -> None:
        """Initialize core components for immediate use."""
        core_components = [
            "training_data_loader", "ml_integration", "rule_optimizer",
            "multi_armed_bandit", "batch_processor", "apes_service_manager",
            "unified_retry_manager", "input_sanitizer", "memory_guard",
            "apriori_analyzer", "context_learner"
        ]
        
        for component_name in core_components:
            if component_name in loaded_components:
                try:
                    # Special initialization for APESServiceManager with event bus
                    if component_name == "apes_service_manager":
                        success = await self.component_loader.initialize_component(
                            component_name,
                            event_bus=self.event_bus
                        )
                    else:
                        success = await self.component_loader.initialize_component(component_name)
                    
                    if success:
                        self.logger.info(f"Initialized core component: {component_name}")
                        # Update component health
                        self.component_health[component_name] = True
                    else:
                        self.logger.warning(f"Failed to initialize core component: {component_name}")
                        self.component_health[component_name] = False
                        
                except Exception as e:
                    self.logger.error(f"Error initializing core component {component_name}: {e}")
                    self.component_health[component_name] = False

    async def _initialize_evaluation_components(self, loaded_components: dict[str, Any]) -> None:
        """Initialize evaluation components for workflows."""
        evaluation_components = [
            "statistical_analyzer", "advanced_statistical_validator",
            "causal_inference_analyzer", "pattern_significance_analyzer",
            "structural_analyzer"
        ]
        
        for component_name in evaluation_components:
            if component_name in loaded_components:
                try:
                    success = await self.component_loader.initialize_component(component_name)
                    if success:
                        self.logger.info(f"Initialized evaluation component: {component_name}")
                        self.component_health[component_name] = True
                    else:
                        self.logger.warning(f"Failed to initialize evaluation component: {component_name}")
                        self.component_health[component_name] = False
                        
                except Exception as e:
                    self.logger.error(f"Error initializing evaluation component {component_name}: {e}")
                    self.component_health[component_name] = False