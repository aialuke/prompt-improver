"""
Simple workflow engine implementation for testing.
"""
import asyncio
from typing import Any, Dict
from prompt_improver.shared.interfaces.protocols.ml import WorkflowEngineProtocol


class SimpleWorkflowEngine:
    """Simple workflow engine for testing."""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self._initialized = False
    
    async def initialize(self):
        """Initialize the workflow engine."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the workflow engine."""
        self._initialized = False
    
    async def start_workflow(self, workflow_id: str, workflow_config: Dict[str, Any]):
        """Start a workflow."""
        if not self._initialized:
            raise RuntimeError("Workflow engine not initialized")
        return {"workflow_id": workflow_id, "status": "started"}
    
    async def stop_workflow(self, workflow_id: str):
        """Stop a workflow.""" 
        if not self._initialized:
            raise RuntimeError("Workflow engine not initialized")
        return {"workflow_id": workflow_id, "status": "stopped"}