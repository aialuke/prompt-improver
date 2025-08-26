"""
Simple component registry implementation for testing.
"""
from typing import Dict, List, Any
from prompt_improver.shared.interfaces.protocols.ml import ComponentRegistryProtocol


class SimpleComponentRegistry:
    """Simple component registry for testing."""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the component registry."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the component registry."""
        self._components.clear()
        self._initialized = False
    
    async def discover_components(self) -> List[str]:
        """Discover available components."""
        if not self._initialized:
            raise RuntimeError("Component registry not initialized")
        return list(self._components.keys())
    
    async def list_components(self) -> List[Dict[str, Any]]:
        """List all registered components."""
        if not self._initialized:
            raise RuntimeError("Component registry not initialized")
        return [
            {"name": name, "component": comp}
            for name, comp in self._components.items()
        ]
    
    def register_component(self, name: str, component: Any):
        """Register a component."""
        self._components[name] = component
    
    def get_component(self, name: str) -> Any:
        """Get a registered component."""
        return self._components.get(name)