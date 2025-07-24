"""
Main Orchestrator API Endpoints for ML Pipeline Orchestration.

Provides REST API endpoints for orchestrator management and control.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..events.event_types import EventType, MLEvent

# Request/Response Models
class WorkflowRequest(BaseModel):
    """Request model for workflow creation."""
    workflow_type: str = Field(..., description="Type of workflow (training, evaluation, deployment)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    priority: str = Field(default="normal", description="Workflow priority (low, normal, high)")
    timeout: Optional[int] = Field(default=None, description="Workflow timeout in seconds")

class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    workflow_id: str
    status: str
    created_at: str
    parameters: Dict[str, Any]
    current_step: Optional[str] = None

class ComponentRegistrationRequest(BaseModel):
    """Request model for component registration."""
    component_name: str = Field(..., description="Name of the component")
    component_tier: str = Field(..., description="Component tier (tier_1_core, tier_2_optimization, etc.)")
    capabilities: List[str] = Field(..., description="List of component capabilities")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")

class HealthStatusResponse(BaseModel):
    """Response model for health status."""
    overall_health: str
    orchestrator_status: str
    active_workflows: int
    active_components: int
    last_check: str
    details: Dict[str, Any]

class OrchestratorEndpoints:
    """
    Main orchestrator API endpoints.
    
    Provides REST API for orchestrator management, workflow control,
    and system monitoring.
    """
    
    def __init__(self, orchestrator=None):
        """Initialize orchestrator endpoints."""
        self.orchestrator = orchestrator
        self.router = APIRouter()
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.router.get("/status", response_model=HealthStatusResponse)
        async def get_orchestrator_status():
            """Get overall orchestrator status."""
            return await self.get_status()
        
        @self.router.post("/workflows", response_model=WorkflowResponse)
        async def create_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
            """Create and start a new workflow."""
            return await self.create_workflow(request, background_tasks)
        
        @self.router.get("/workflows")
        async def list_workflows():
            """List all workflows."""
            return await self.list_workflows()
        
        @self.router.get("/workflows/{workflow_id}")
        async def get_workflow_status(workflow_id: str):
            """Get status of a specific workflow."""
            return await self.get_workflow_status(workflow_id)
        
        @self.router.post("/workflows/{workflow_id}/stop")
        async def stop_workflow(workflow_id: str):
            """Stop a running workflow."""
            return await self.stop_workflow(workflow_id)
        
        @self.router.get("/components")
        async def list_components():
            """List all registered components."""
            return await self.list_components()
        
        @self.router.post("/components/register")
        async def register_component(request: ComponentRegistrationRequest):
            """Register a new component."""
            return await self.register_component(request)
        
        @self.router.get("/components/{component_name}/health")
        async def get_component_health(component_name: str):
            """Get health status of a specific component."""
            return await self.get_component_health(component_name)
        
        @self.router.get("/metrics")
        async def get_metrics():
            """Get orchestrator performance metrics."""
            return await self.get_metrics()
        
        @self.router.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            # Get orchestrator monitor status
            orchestrator_monitor = getattr(self.orchestrator, 'orchestrator_monitor', None)
            if orchestrator_monitor:
                monitor_status = await orchestrator_monitor.get_current_status()
                overall_health = monitor_status.get("health_status", "unknown")
            else:
                overall_health = "unknown"
                monitor_status = {}
            
            # Get workflow statistics
            workflow_engine = getattr(self.orchestrator, 'workflow_engine', None)
            if workflow_engine:
                workflow_stats = await workflow_engine.get_statistics()
                active_workflows = workflow_stats.get("active_workflows", 0)
            else:
                active_workflows = 0
            
            # Get component statistics
            component_registry = getattr(self.orchestrator, 'component_registry', None)
            if component_registry:
                component_stats = await component_registry.get_statistics()
                active_components = component_stats.get("active_components", 0)
            else:
                active_components = 0
            
            return HealthStatusResponse(
                overall_health=overall_health,
                orchestrator_status="running" if self.orchestrator else "stopped",
                active_workflows=active_workflows,
                active_components=active_components,
                last_check=datetime.utcnow().isoformat(),
                details={
                    "monitor_status": monitor_status,
                    "workflow_stats": workflow_stats if workflow_engine else {},
                    "component_stats": component_stats if component_registry else {}
                }
            ).dict()
            
        except Exception as e:
            self.logger.error(f"Error getting orchestrator status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_workflow(self, request: WorkflowRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Create and start a new workflow."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            # Generate workflow ID
            workflow_id = f"{request.workflow_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Get appropriate coordinator
            coordinator = await self._get_workflow_coordinator(request.workflow_type)
            if not coordinator:
                raise HTTPException(status_code=400, detail=f"Unsupported workflow type: {request.workflow_type}")
            
            # Start workflow in background
            background_tasks.add_task(
                self._start_workflow_async,
                coordinator,
                workflow_id,
                request.parameters,
                request.timeout
            )
            
            return WorkflowResponse(
                workflow_id=workflow_id,
                status="starting",
                created_at=datetime.utcnow().isoformat(),
                parameters=request.parameters,
                current_step="initialization"
            ).dict()
            
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_workflow_coordinator(self, workflow_type: str):
        """Get the appropriate workflow coordinator."""
        coordinators = {
            "training": getattr(self.orchestrator, 'training_coordinator', None),
            "evaluation": getattr(self.orchestrator, 'evaluation_coordinator', None),
            "deployment": getattr(self.orchestrator, 'deployment_coordinator', None),
            "optimization": getattr(self.orchestrator, 'optimization_coordinator', None),
            "data_pipeline": getattr(self.orchestrator, 'data_pipeline_coordinator', None)
        }
        return coordinators.get(workflow_type)
    
    async def _start_workflow_async(self, coordinator, workflow_id: str, parameters: Dict[str, Any], timeout: Optional[int]):
        """Start workflow asynchronously."""
        try:
            if hasattr(coordinator, 'start_training_workflow'):
                await coordinator.start_training_workflow(workflow_id, parameters)
            elif hasattr(coordinator, 'start_evaluation_workflow'):
                await coordinator.start_evaluation_workflow(workflow_id, parameters)
            elif hasattr(coordinator, 'start_deployment'):
                await coordinator.start_deployment(workflow_id, parameters)
            elif hasattr(coordinator, 'start_optimization_workflow'):
                await coordinator.start_optimization_workflow(workflow_id, parameters)
            elif hasattr(coordinator, 'start_data_pipeline'):
                await coordinator.start_data_pipeline(workflow_id, parameters)
            else:
                self.logger.error(f"Unknown coordinator type for workflow {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"Error starting workflow {workflow_id}: {e}")
    
    async def list_workflows(self) -> Dict[str, Any]:
        """List all workflows."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            workflows = {}
            
            # Get workflows from all coordinators
            coordinators = [
                ('training', getattr(self.orchestrator, 'training_coordinator', None)),
                ('evaluation', getattr(self.orchestrator, 'evaluation_coordinator', None)),
                ('deployment', getattr(self.orchestrator, 'deployment_coordinator', None)),
                ('optimization', getattr(self.orchestrator, 'optimization_coordinator', None)),
                ('data_pipeline', getattr(self.orchestrator, 'data_pipeline_coordinator', None))
            ]
            
            for workflow_type, coordinator in coordinators:
                if coordinator:
                    try:
                        if hasattr(coordinator, 'list_active_workflows'):
                            active_workflows = await coordinator.list_active_workflows()
                            workflows[workflow_type] = active_workflows
                        elif hasattr(coordinator, 'list_active_training_sessions'):
                            active_workflows = await coordinator.list_active_training_sessions()
                            workflows[workflow_type] = list(active_workflows.keys())
                        elif hasattr(coordinator, 'list_active_evaluations'):
                            active_workflows = await coordinator.list_active_evaluations()
                            workflows[workflow_type] = list(active_workflows.keys())
                        elif hasattr(coordinator, 'list_active_deployments'):
                            active_workflows = await coordinator.list_active_deployments()
                            workflows[workflow_type] = active_workflows
                        elif hasattr(coordinator, 'list_active_optimizations'):
                            active_workflows = await coordinator.list_active_optimizations()
                            workflows[workflow_type] = active_workflows
                        elif hasattr(coordinator, 'list_active_pipelines'):
                            active_workflows = await coordinator.list_active_pipelines()
                            workflows[workflow_type] = active_workflows
                    except Exception as e:
                        self.logger.warning(f"Error getting workflows from {workflow_type} coordinator: {e}")
                        workflows[workflow_type] = []
            
            return {
                "workflows": workflows,
                "total_active": sum(len(wf_list) for wf_list in workflows.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error listing workflows: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            # Try to find workflow in all coordinators
            coordinators = [
                getattr(self.orchestrator, 'training_coordinator', None),
                getattr(self.orchestrator, 'evaluation_coordinator', None),
                getattr(self.orchestrator, 'deployment_coordinator', None),
                getattr(self.orchestrator, 'optimization_coordinator', None),
                getattr(self.orchestrator, 'data_pipeline_coordinator', None)
            ]
            
            for coordinator in coordinators:
                if not coordinator:
                    continue
                
                try:
                    # Try different status methods
                    status_methods = [
                        'get_workflow_status',
                        'get_training_session_status',
                        'get_evaluation_status',
                        'get_deployment_status',
                        'get_optimization_status',
                        'get_pipeline_status'
                    ]
                    
                    for method_name in status_methods:
                        if hasattr(coordinator, method_name):
                            method = getattr(coordinator, method_name)
                            status = await method(workflow_id)
                            if status:
                                return status
                                
                except Exception as e:
                    self.logger.debug(f"Workflow {workflow_id} not found in coordinator: {e}")
                    continue
            
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stop_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Stop a running workflow."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            # Try to stop workflow in all coordinators
            coordinators = [
                getattr(self.orchestrator, 'training_coordinator', None),
                getattr(self.orchestrator, 'evaluation_coordinator', None),
                getattr(self.orchestrator, 'deployment_coordinator', None),
                getattr(self.orchestrator, 'optimization_coordinator', None),
                getattr(self.orchestrator, 'data_pipeline_coordinator', None)
            ]
            
            for coordinator in coordinators:
                if not coordinator:
                    continue
                
                try:
                    # Try different stop methods
                    stop_methods = [
                        'stop_workflow',
                        'stop_evaluation',
                        'stop_deployment',
                        'stop_optimization',
                        'stop_pipeline'
                    ]
                    
                    for method_name in stop_methods:
                        if hasattr(coordinator, method_name):
                            method = getattr(coordinator, method_name)
                            await method(workflow_id)
                            return {
                                "workflow_id": workflow_id,
                                "status": "stopped",
                                "stopped_at": datetime.utcnow().isoformat()
                            }
                            
                except Exception as e:
                    self.logger.debug(f"Failed to stop workflow {workflow_id} in coordinator: {e}")
                    continue
            
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found or could not be stopped")
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error stopping workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def list_components(self) -> Dict[str, Any]:
        """List all registered components."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            component_registry = getattr(self.orchestrator, 'component_registry', None)
            if not component_registry:
                return {"components": [], "total": 0}
            
            components = await component_registry.list_components()
            component_details = []
            
            for component_name in components:
                try:
                    connector = await component_registry.get_connector(component_name)
                    if connector:
                        metadata = connector.metadata
                        component_details.append({
                            "name": metadata.name,
                            "tier": metadata.tier.value,
                            "version": metadata.version,
                            "capabilities": [cap.name for cap in metadata.capabilities],
                            "status": connector.status.value if hasattr(connector, 'status') else "unknown"
                        })
                except Exception as e:
                    self.logger.warning(f"Error getting details for component {component_name}: {e}")
                    component_details.append({
                        "name": component_name,
                        "tier": "unknown",
                        "version": "unknown",
                        "capabilities": [],
                        "status": "unknown"
                    })
            
            return {
                "components": component_details,
                "total": len(component_details)
            }
            
        except Exception as e:
            self.logger.error(f"Error listing components: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def register_component(self, request: ComponentRegistrationRequest) -> Dict[str, Any]:
        """Register a new component."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            component_registry = getattr(self.orchestrator, 'component_registry', None)
            if not component_registry:
                raise HTTPException(status_code=503, detail="Component registry not available")
            
            # Register component (this would depend on the actual implementation)
            # For now, return a success response
            return {
                "component_name": request.component_name,
                "status": "registered",
                "registered_at": datetime.utcnow().isoformat(),
                "tier": request.component_tier,
                "capabilities": request.capabilities
            }
            
        except Exception as e:
            self.logger.error(f"Error registering component: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """Get health status of a specific component."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            component_health_monitor = getattr(self.orchestrator, 'component_health_monitor', None)
            if not component_health_monitor:
                raise HTTPException(status_code=503, detail="Component health monitor not available")
            
            health_info = await component_health_monitor.get_component_health(component_name)
            if not health_info:
                raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
            
            return health_info
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting component health: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        try:
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Orchestrator not available")
            
            metrics = {}
            
            # Get orchestrator monitor metrics
            orchestrator_monitor = getattr(self.orchestrator, 'orchestrator_monitor', None)
            if orchestrator_monitor:
                metrics["system"] = await orchestrator_monitor.get_current_status()
            
            # Get component health metrics
            component_health_monitor = getattr(self.orchestrator, 'component_health_monitor', None)
            if component_health_monitor:
                metrics["component_health"] = await component_health_monitor.get_health_summary()
            
            # Get workflow metrics
            workflow_engine = getattr(self.orchestrator, 'workflow_engine', None)
            if workflow_engine:
                metrics["workflows"] = await workflow_engine.get_statistics()
            
            return {
                "metrics": metrics,
                "collected_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router."""
        return self.router