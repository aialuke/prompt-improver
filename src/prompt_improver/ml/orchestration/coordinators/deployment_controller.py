"""
Deployment Controller for ML Pipeline Orchestration.

Coordinates model deployment workflows and manages deployment lifecycle.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from ..events.event_types import EventType, MLEvent

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    immediate = "immediate"

@dataclass
class DeploymentConfig:
    """Configuration for deployment workflows."""
    default_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    default_timeout: int = 600  # 10 minutes
    health_check_interval: int = 30  # seconds
    rollback_timeout: int = 300  # 5 minutes
    canary_traffic_percentage: float = 0.1  # 10%

class DeploymentController:
    """
    Coordinates model deployment workflows and manages deployment lifecycle.
    
    Handles deployment to ProductionModelRegistry and manages deployment
    strategies, health checks, and rollback capabilities.
    """
    
    def __init__(self, config: DeploymentConfig, event_bus=None, resource_manager=None):
        """Initialize the deployment controller."""
        self.config = config
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Active deployments
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        
        # Deployment history
        self.deployment_history: List[Dict[str, Any]] = []
        
    async def start_deployment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Start a new model deployment."""
        self.logger.info(f"Starting deployment {deployment_id}")
        
        # Validate deployment parameters
        await self._validate_deployment_parameters(parameters)
        
        # Register deployment
        deployment_strategy = DeploymentStrategy(
            parameters.get("strategy", self.config.default_strategy.value)
        )
        
        self.active_deployments[deployment_id] = {
            "status": "running",
            "strategy": deployment_strategy,
            "started_at": datetime.now(timezone.utc),
            "parameters": parameters,
            "current_step": None,
            "health_checks": [],
            "rollback_point": None
        }
        
        try:
            # Step 1: Prepare deployment environment
            await self._prepare_deployment_environment(deployment_id, parameters)
            
            # Step 2: Execute deployment strategy
            await self._execute_deployment_strategy(deployment_id, deployment_strategy, parameters)
            
            # Step 3: Verify deployment health
            await self._verify_deployment_health(deployment_id, parameters)
            
            # Step 4: Register with production registry
            await self._register_with_production_registry(deployment_id, parameters)
            
            # Mark deployment as completed
            self.active_deployments[deployment_id]["status"] = "completed"
            self.active_deployments[deployment_id]["completed_at"] = datetime.now(timezone.utc)
            
            # Add to deployment history
            self.deployment_history.append(self.active_deployments[deployment_id].copy())
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.DEPLOYMENT_COMPLETED,
                    source="deployment_controller",
                    data={
                        "deployment_id": deployment_id,
                        "model_version": parameters.get("model_version"),
                        "strategy": deployment_strategy.value
                    }
                ))
            
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            await self._handle_deployment_failure(deployment_id, e)
            raise
    
    async def _validate_deployment_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate deployment parameters."""
        required_params = ["model_version", "model_artifact_path"]
        missing_params = [param for param in required_params if param not in parameters]
        
        if missing_params:
            raise ValueError(f"Missing required deployment parameters: {missing_params}")
        
        # Validate strategy
        strategy = parameters.get("strategy", self.config.default_strategy.value)
        try:
            DeploymentStrategy(strategy)
        except ValueError:
            raise ValueError(f"Invalid deployment strategy: {strategy}")
    
    async def _prepare_deployment_environment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Prepare the deployment environment."""
        self.logger.info(f"Preparing deployment environment for {deployment_id}")
        self.active_deployments[deployment_id]["current_step"] = "environment_preparation"
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DEPLOYMENT_STARTED,
                source="deployment_controller",
                data={
                    "deployment_id": deployment_id,
                    "step": "environment_preparation"
                }
            ))
        
        # Simulate environment preparation (resource allocation, config setup, etc.)
        await asyncio.sleep(0.1)
        
        # Create rollback point
        self.active_deployments[deployment_id]["rollback_point"] = {
            "created_at": datetime.now(timezone.utc),
            "previous_version": parameters.get("previous_model_version"),
            "environment_snapshot": "snapshot_data"  # Simulated
        }
        
        self.logger.info(f"Deployment environment prepared for {deployment_id}")
    
    async def _execute_deployment_strategy(self, deployment_id: str, strategy: DeploymentStrategy, 
                                         parameters: Dict[str, Any]) -> None:
        """Execute the specific deployment strategy."""
        self.logger.info(f"Executing {strategy.value} deployment strategy for {deployment_id}")
        self.active_deployments[deployment_id]["current_step"] = f"deployment_{strategy.value}"
        
        if strategy == DeploymentStrategy.BLUE_GREEN:
            await self._execute_blue_green_deployment(deployment_id, parameters)
        elif strategy == DeploymentStrategy.CANARY:
            await self._execute_canary_deployment(deployment_id, parameters)
        elif strategy == DeploymentStrategy.ROLLING:
            await self._execute_rolling_deployment(deployment_id, parameters)
        elif strategy == DeploymentStrategy.immediate:
            await self._execute_immediate_deployment(deployment_id, parameters)
        
        self.logger.info(f"Deployment strategy {strategy.value} executed for {deployment_id}")
    
    async def _execute_blue_green_deployment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Execute blue-green deployment strategy."""
        # Deploy to green environment
        await asyncio.sleep(0.1)  # Simulate deployment
        
        # Switch traffic to green environment
        await asyncio.sleep(0.05)  # Simulate traffic switch
        
        self.active_deployments[deployment_id]["deployment_details"] = {
            "blue_environment": "preserved",
            "green_environment": "active",
            "traffic_switched": True
        }
    
    async def _execute_canary_deployment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Execute canary deployment strategy."""
        traffic_percentage = parameters.get("canary_traffic", self.config.canary_traffic_percentage)
        
        # Deploy canary version
        await asyncio.sleep(0.1)  # Simulate canary deployment
        
        # Route small percentage of traffic to canary
        await asyncio.sleep(0.05)  # Simulate traffic routing
        
        self.active_deployments[deployment_id]["deployment_details"] = {
            "canary_deployed": True,
            "traffic_percentage": traffic_percentage,
            "monitoring_enabled": True
        }
    
    async def _execute_rolling_deployment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Execute rolling deployment strategy."""
        # Gradual rollout across instances
        await asyncio.sleep(0.1)  # Simulate rolling deployment
        
        self.active_deployments[deployment_id]["deployment_details"] = {
            "rolling_strategy": "gradual",
            "instances_updated": "all",
            "zero_downtime": True
        }
    
    async def _execute_immediate_deployment(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Execute immediate deployment strategy."""
        # Direct replacement
        await asyncio.sleep(0.05)  # Simulate immediate deployment
        
        self.active_deployments[deployment_id]["deployment_details"] = {
            "immediate_replacement": True,
            "downtime": "minimal"
        }
    
    async def _verify_deployment_health(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Verify deployment health through health checks."""
        self.logger.info(f"Verifying deployment health for {deployment_id}")
        self.active_deployments[deployment_id]["current_step"] = "health_verification"
        
        # Simulate health checks
        for i in range(3):  # Multiple health check rounds
            await asyncio.sleep(0.1)
            
            health_check = {
                "timestamp": datetime.now(timezone.utc),
                "status": "healthy",
                "response_time": 0.05 + (i * 0.01),  # Simulated response times
                "error_rate": 0.0
            }
            
            self.active_deployments[deployment_id]["health_checks"].append(health_check)
        
        # Evaluate overall health
        avg_response_time = sum(hc["response_time"] for hc in self.active_deployments[deployment_id]["health_checks"]) / 3
        overall_healthy = all(hc["status"] == "healthy" for hc in self.active_deployments[deployment_id]["health_checks"])
        
        if not overall_healthy or avg_response_time > 0.1:  # Health check failure
            raise Exception(f"Deployment {deployment_id} failed health checks")
        
        self.logger.info(f"Deployment health verification passed for {deployment_id}")
    
    async def _register_with_production_registry(self, deployment_id: str, parameters: Dict[str, Any]) -> None:
        """Register deployment with production model registry."""
        self.logger.info(f"Registering deployment {deployment_id} with production registry")
        self.active_deployments[deployment_id]["current_step"] = "registry_registration"
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.MODEL_DEPLOYED,
                source="deployment_controller",
                data={
                    "deployment_id": deployment_id,
                    "model_version": parameters.get("model_version"),
                    "registry_path": parameters.get("model_artifact_path")
                }
            ))
        
        # Simulate registry registration
        await asyncio.sleep(0.1)
        
        self.active_deployments[deployment_id]["registry_info"] = {
            "model_id": f"model_{deployment_id}",
            "version": parameters.get("model_version"),
            "artifact_path": parameters.get("model_artifact_path"),
            "registered_at": datetime.now(timezone.utc)
        }
        
        self.logger.info(f"Deployment {deployment_id} registered with production registry")
    
    async def _handle_deployment_failure(self, deployment_id: str, error: Exception) -> None:
        """Handle deployment failure and trigger rollback if necessary."""
        self.logger.error(f"Deployment {deployment_id} failed: {error}")
        
        self.active_deployments[deployment_id]["status"] = "failed"
        self.active_deployments[deployment_id]["error"] = str(error)
        self.active_deployments[deployment_id]["completed_at"] = datetime.now(timezone.utc)
        
        # Trigger rollback if rollback point exists
        if self.active_deployments[deployment_id].get("rollback_point"):
            await self._execute_rollback(deployment_id)
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DEPLOYMENT_FAILED,
                source="deployment_controller",
                data={
                    "deployment_id": deployment_id,
                    "error_message": str(error),
                    "rollback_executed": bool(self.active_deployments[deployment_id].get("rollback_point"))
                }
            ))
    
    async def _execute_rollback(self, deployment_id: str) -> None:
        """Execute rollback to previous version."""
        self.logger.info(f"Executing rollback for deployment {deployment_id}")
        
        rollback_point = self.active_deployments[deployment_id]["rollback_point"]
        
        # Simulate rollback execution
        await asyncio.sleep(0.1)
        
        self.active_deployments[deployment_id]["rollback_executed"] = {
            "executed_at": datetime.now(timezone.utc),
            "reverted_to_version": rollback_point.get("previous_version"),
            "status": "completed"
        }
        
        self.logger.info(f"Rollback completed for deployment {deployment_id}")
    
    async def stop_deployment(self, deployment_id: str) -> None:
        """Stop a running deployment."""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        self.active_deployments[deployment_id]["status"] = "stopped"
        self.active_deployments[deployment_id]["completed_at"] = datetime.now(timezone.utc)
        
        self.logger.info(f"Deployment {deployment_id} stopped")
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get the status of a deployment."""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        return self.active_deployments[deployment_id].copy()
    
    async def list_active_deployments(self) -> List[str]:
        """List all active deployments."""
        return [
            dep_id for dep_id, dep_data in self.active_deployments.items()
            if dep_data["status"] == "running"
        ]
    
    async def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history[-limit:] if limit > 0 else self.deployment_history