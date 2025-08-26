"""Deployment Pipeline Service.

Focused service responsible for model deployment, versioning, and release management.
Handles native deployment workflows, complete ML pipelines, and deployment coordination.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ....shared.interfaces.protocols.ml import EventBusProtocol
from ..config.external_services_config import ExternalServicesConfig
from ..core.orchestrator_service_protocols import DeploymentPipelineServiceProtocol
from ..events.event_types import EventType, MLEvent


class DeploymentPipelineService:
    """
    Deployment Pipeline Service.
    
    Responsible for:
    - Native ML model deployment workflows
    - Complete ML pipeline orchestration (training + evaluation + deployment)
    - Model versioning and release management
    - Deployment strategy coordination
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        external_services_config: ExternalServicesConfig,
    ):
        """Initialize DeploymentPipelineService with required dependencies.
        
        Args:
            event_bus: Event bus for deployment event communication
            external_services_config: External services configuration
        """
        self.event_bus = event_bus
        self.external_services_config = external_services_config
        self.logger = logging.getLogger(__name__)
        
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the deployment pipeline service."""
        if self._is_initialized:
            return
        
        self.logger.info("Initializing Deployment Pipeline Service")
        
        try:
            self._is_initialized = True
            
            # Emit initialization event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ORCHESTRATOR_INITIALIZED,
                source="deployment_pipeline_service",
                data={"component": "deployment_pipeline_service", "timestamp": datetime.now(timezone.utc).isoformat()}
            ))
            
            self.logger.info("Deployment Pipeline Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize deployment pipeline service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the deployment pipeline service."""
        self.logger.info("Shutting down Deployment Pipeline Service")
        
        try:
            self._is_initialized = False
            self.logger.info("Deployment Pipeline Service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during deployment pipeline service shutdown: {e}")
            raise

    async def run_native_deployment_workflow(self, 
                                           model_id: str,
                                           deployment_config: dict[str, Any]) -> dict[str, Any]:
        """
        Run native ML model deployment workflow without Docker containers.
        
        Args:
            model_id: Model to deploy from registry
            deployment_config: Native deployment configuration
            
        Returns:
            Dictionary of deployment workflow results
        """
        self.logger.info(f"Running native deployment workflow for model {model_id}")
        
        try:
            # Import native deployment pipeline
            from ...lifecycle.enhanced_model_registry import EnhancedModelRegistry
            from ...lifecycle.native_deployment_pipeline import (
                NativeDeploymentPipeline,
                NativeDeploymentStrategy,
                NativePipelineConfig,
            )
            
            # Initialize model registry with external PostgreSQL
            model_registry = EnhancedModelRegistry(
                tracking_uri=self.external_services_config.mlflow.tracking_uri,
                registry_uri=self.external_services_config.mlflow.registry_store_uri
            )
            
            # Initialize native deployment pipeline
            deployment_pipeline = NativeDeploymentPipeline(
                model_registry=model_registry,
                external_services=self.external_services_config,
                enable_parallel_deployment=True
            )
            
            # Configure deployment pipeline
            strategy = deployment_config.get('strategy', 'blue_green')
            pipeline_config = NativePipelineConfig(
                strategy=NativeDeploymentStrategy[strategy.upper()],
                environment=deployment_config.get('environment', 'production'),
                use_systemd=deployment_config.get('use_systemd', True),
                use_nginx=deployment_config.get('use_nginx', True),
                enable_monitoring=deployment_config.get('enable_monitoring', True),
                parallel_deployment=deployment_config.get('parallel_deployment', True),
                enable_caching=deployment_config.get('enable_caching', True)
            )
            
            # Execute native deployment
            result = await deployment_pipeline.deploy_model_pipeline(
                model_id=model_id,
                pipeline_config=pipeline_config
            )
            
            deployment_results = {
                'deployment_id': result.pipeline_id,
                'model_id': model_id,
                'status': result.status.value,
                'deployment_time_seconds': result.total_pipeline_time_seconds,
                'active_endpoints': result.active_endpoints,
                'service_names': result.service_names,
                'performance_metrics': {
                    'preparation_time': result.preparation_time_seconds,
                    'build_time': result.build_time_seconds,
                    'deployment_time': result.deployment_time_seconds,
                    'verification_time': result.verification_time_seconds
                },
                'deployment_type': 'native'
            }
            
            if result.error_message:
                deployment_results['error'] = result.error_message
            
            # Emit deployment completion event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                source="deployment_pipeline_service",
                data={
                    "deployment_id": result.pipeline_id,
                    "model_id": model_id,
                    "status": result.status.value
                }
            ))
            
            self.logger.info(f"Native deployment workflow completed for {model_id}")
            return deployment_results
            
        except Exception as e:
            # Emit deployment failure event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_FAILED,
                source="deployment_pipeline_service",
                data={
                    "model_id": model_id,
                    "error_message": str(e)
                }
            ))
            
            self.logger.error(f"Native deployment workflow failed: {e}")
            raise

    async def run_complete_ml_pipeline(self, 
                                     training_data: Any,
                                     model_config: dict[str, Any],
                                     deployment_config: dict[str, Any] | None = None,
                                     training_runner: callable = None,
                                     evaluation_runner: callable = None) -> dict[str, Any]:
        """
        Run complete ML pipeline: training, evaluation, and native deployment.
        
        Args:
            training_data: Input training data
            model_config: Model configuration
            deployment_config: Optional deployment configuration
            training_runner: Function to execute training workflow
            evaluation_runner: Function to execute evaluation workflow
            
        Returns:
            Dictionary of complete pipeline results
        """
        self.logger.info("Running complete ML pipeline with native deployment")
        
        pipeline_start = time.time()
        results = {}
        
        try:
            # Phase 1: Model training
            if training_runner:
                self.logger.info("Phase 1: Model training")
                training_results = await training_runner(training_data)
                results['training'] = training_results
            else:
                self.logger.warning("No training runner provided, skipping training phase")
                results['training'] = {"status": "skipped", "message": "No training runner provided"}
            
            # Phase 2: Model evaluation
            if evaluation_runner:
                self.logger.info("Phase 2: Model evaluation")
                evaluation_results = await evaluation_runner(training_data)
                results['evaluation'] = evaluation_results
            else:
                self.logger.warning("No evaluation runner provided, skipping evaluation phase")
                results['evaluation'] = {"status": "skipped", "message": "No evaluation runner provided"}
            
            # Phase 3: Native deployment (if configured)
            if deployment_config:
                self.logger.info("Phase 3: Native model deployment")
                
                # Simulate model registration (in real implementation, this would register the trained model)
                model_id = model_config.get('model_name', f"model_{int(time.time())}")
                
                deployment_results = await self.run_native_deployment_workflow(
                    model_id=model_id,
                    deployment_config=deployment_config
                )
                results['deployment'] = deployment_results
            else:
                self.logger.info("No deployment configuration provided, skipping deployment phase")
                results['deployment'] = {"status": "skipped", "message": "No deployment configuration provided"}
            
            # Pipeline summary
            total_time = time.time() - pipeline_start
            results['pipeline_summary'] = {
                'total_time_seconds': total_time,
                'phases_completed': len([r for r in results.values() if r.get('status') != 'skipped']),
                'deployment_type': 'native',
                'external_services': {
                    'postgresql': self.external_services_config.postgresql.host,
                    'redis': self.external_services_config.redis.host
                }
            }
            
            # Emit pipeline completion event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                source="deployment_pipeline_service",
                data={
                    "pipeline_type": "complete_ml_pipeline",
                    "total_time_seconds": total_time,
                    "phases": list(results.keys())
                }
            ))
            
            self.logger.info("Complete ML pipeline finished in %.2fs", total_time)
            return results
            
        except Exception as e:
            # Emit pipeline failure event
            await self.event_bus.emit(MLEvent(
                event_type=EventType.WORKFLOW_FAILED,
                source="deployment_pipeline_service",
                data={
                    "pipeline_type": "complete_ml_pipeline",
                    "error_message": str(e)
                }
            ))
            
            self.logger.error(f"Complete ML pipeline failed: {e}")
            raise