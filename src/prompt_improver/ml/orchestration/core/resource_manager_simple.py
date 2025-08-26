"""
Simple resource manager implementation for testing.
"""
import psutil
from typing import Dict
from prompt_improver.shared.interfaces.protocols.ml import ResourceManagerProtocol


class SimpleResourceManager:
    """Simple resource manager for testing."""
    
    def __init__(self):
        self._initialized = False
    
    async def initialize(self):
        """Initialize the resource manager."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the resource manager."""
        self._initialized = False
    
    async def get_usage_stats(self) -> Dict[str, float]:
        """Get real system resource usage stats."""
        if not self._initialized:
            raise RuntimeError("Resource manager not initialized")
        
        try:
            return {
                "memory_usage_percent": psutil.virtual_memory().percent,
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                "disk_usage_percent": psutil.disk_usage('/').percent,
            }
        except Exception:
            # Fallback values if psutil fails
            return {
                "memory_usage_percent": 50.0,
                "cpu_usage_percent": 25.0,
                "disk_usage_percent": 30.0,
            }
    
    async def handle_resource_exhaustion(self, resource_type: str):
        """Handle resource exhaustion."""
        if self.event_bus:
            await self.event_bus.emit("resource_exhaustion", {"type": resource_type})