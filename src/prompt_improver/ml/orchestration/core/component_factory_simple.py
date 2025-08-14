"""
Simple component factory implementation for testing.
"""
from typing import Any, Dict
from prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol


class SimpleComponentFactory:
    """Simple component factory for testing."""
    
    def __init__(self):
        self._component_classes = {}
    
    def register_component_class(self, name: str, component_class: type):
        """Register a component class."""
        self._component_classes[name] = component_class
    
    def create_component(self, name: str, config: Dict[str, Any] = None) -> Any:
        """Create a component instance."""
        if name not in self._component_classes:
            raise ValueError(f"Component class '{name}' not registered")
        
        component_class = self._component_classes[name]
        if config:
            return component_class(**config)
        return component_class()
    
    def get_available_components(self) -> list[str]:
        """Get list of available component names."""
        return list(self._component_classes.keys())