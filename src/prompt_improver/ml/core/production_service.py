"""ML Production Service for deployment and production model management.

Handles production deployment with blue-green strategies, rollback capabilities,
health monitoring, and production model lifecycle management.
"""

import logging
from typing import Any, Dict, List

from prompt_improver.ml.models.production_registry import (
    DeploymentStrategy,
    ModelAlias,
    ModelDeploymentConfig,
    ModelMetrics,
    ProductionModelRegistry,
    get_production_registry,
)
from prompt_improver.utils.datetime_utils import aware_utc_now
from .protocols import ProductionServiceProtocol, ModelRegistryProtocol

logger = logging.getLogger(__name__)


class MLProductionService(ProductionServiceProtocol):
    """Service for production ML model deployment and management."""

    def __init__(
        self, 
        model_registry: ModelRegistryProtocol,
        orchestrator_event_bus=None
    ):
        """Initialize production service.
        
        Args:
            model_registry: Registry for accessing cached models
            orchestrator_event_bus: Event bus for orchestrator integration
        """
        self.model_registry = model_registry
        self.orchestrator_event_bus = orchestrator_event_bus
        
        # Production model registry with alias-based deployment
        self.production_registry: ProductionModelRegistry | None = None
        self._production_enabled = False
        
        logger.info("ML Production Service initialized")

    async def enable_production_deployment(
        self, tracking_uri: str | None = None
    ) -> Dict[str, Any]:
        """Enable production deployment capabilities.

        Args:
            tracking_uri: MLflow tracking URI for production (defaults to local)

        Returns:
            Status of production enablement
        """
        try:
            self.production_registry = await get_production_registry(tracking_uri)
            self._production_enabled = True

            logger.info("Production deployment enabled")
            return {
                "status": "enabled",
                "tracking_uri": tracking_uri or "local",
                "capabilities": [
                    "Alias-based deployment (@champion, @production)",
                    "Blue-green deployments",
                    "Automatic rollback",
                    "Health monitoring",
                    "Performance tracking",
                ],
            }

        except Exception as e:
            logger.error(f"Failed to enable production deployment: {e}")
            return {"status": "failed", "error": str(e)}

    async def deploy_to_production(
        self,
        model_name: str,
        version: str,
        alias: ModelAlias = ModelAlias.PRODUCTION,
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
    ) -> Dict[str, Any]:
        """Deploy model to production with specified strategy.

        Args:
            model_name: Name of the model to deploy
            version: Model version to deploy
            alias: Deployment alias (@production, @champion, etc.)
            strategy: Deployment strategy (blue-green, canary, etc.)

        Returns:
            Deployment result with status and metrics
        """
        if not self._production_enabled or self.production_registry is None:
            return {
                "status": "failed",
                "error": "Production deployment not enabled. Call enable_production_deployment() first.",
            }

        try:
            # Create deployment configuration
            config = ModelDeploymentConfig(
                model_name=model_name,
                alias=alias,
                strategy=strategy,
                health_check_interval=60,
                rollback_threshold=0.05,  # 5% performance degradation
                max_latency_ms=500,
                min_accuracy=0.8,
            )

            # Deploy using production registry
            result = await self.production_registry.deploy_model(
                model_name=model_name, version=version, alias=alias, config=config
            )

            # Emit deployment event
            await self._emit_orchestrator_event("MODEL_DEPLOYED", {
                "model_name": model_name,
                "version": version,
                "alias": alias.value,
                "strategy": strategy.value,
                "deployment_result": result
            })

            logger.info(
                f"Production deployment initiated: {model_name}:{version}@{alias.value}"
            )
            return result

        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return {"status": "failed", "error": str(e), "rollback_required": True}

    async def rollback_production(
        self,
        model_name: str,
        alias: ModelAlias = ModelAlias.PRODUCTION,
        reason: str = "Performance degradation detected",
    ) -> Dict[str, Any]:
        """Rollback production deployment to previous version.

        Args:
            model_name: Name of the model to rollback
            alias: Alias to rollback (@production, @champion, etc.)
            reason: Reason for rollback

        Returns:
            Rollback result
        """
        if not self._production_enabled or self.production_registry is None:
            return {"status": "failed", "error": "Production deployment not enabled."}

        try:
            result = await self.production_registry.rollback_deployment(
                model_name=model_name, alias=alias, reason=reason
            )

            # Emit rollback event
            await self._emit_orchestrator_event("MODEL_ROLLBACK", {
                "model_name": model_name,
                "alias": alias.value,
                "reason": reason,
                "rollback_result": result
            })

            logger.warning(f"Production rollback completed: {model_name}@{alias.value}")
            return result

        except Exception as e:
            logger.error(f"Production rollback failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def monitor_production_health(
        self, model_name: str, alias: ModelAlias = ModelAlias.PRODUCTION
    ) -> Dict[str, Any]:
        """Monitor production model health and performance.

        Args:
            model_name: Name of the model to monitor
            alias: Model alias to monitor

        Returns:
            Health status and metrics
        """
        if not self._production_enabled or self.production_registry is None:
            return {"status": "failed", "error": "Production deployment not enabled."}

        try:
            # Simulate performance metrics (in production, these would come from real monitoring)
            current_metrics = ModelMetrics(
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                latency_p95=120.0,
                latency_p99=250.0,
                error_rate=0.005,
                prediction_count=1000,
                timestamp=aware_utc_now(),
            )

            # Monitor health using production registry
            health_result = await self.production_registry.monitor_model_health(
                model_name=model_name, alias=alias, metrics=current_metrics
            )

            return health_result

        except Exception as e:
            logger.error(f"Production health monitoring failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def get_production_model(
        self, model_name: str, alias: ModelAlias = ModelAlias.PRODUCTION
    ) -> Any:
        """Load production model by alias.

        Args:
            model_name: Name of the model
            alias: Model alias to load

        Returns:
            Loaded production model
        """
        if not self._production_enabled or self.production_registry is None:
            # Fallback to regular model loading from registry
            return self.model_registry.get_model(f"{model_name}:{alias.value}")

        try:
            return await self.production_registry.get_production_model(
                model_name=model_name, alias=alias
            )

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            # Fallback to regular model loading
            return self.model_registry.get_model(f"{model_name}:{alias.value}")

    async def list_production_deployments(self) -> List[Dict[str, Any]]:
        """List all production deployments with their status.

        Returns:
            List of deployment information
        """
        if not self._production_enabled or self.production_registry is None:
            return []

        try:
            return await self.production_registry.list_deployments()

        except Exception as e:
            logger.error(f"Failed to list production deployments: {e}")
            return []

    async def _emit_orchestrator_event(self, event_type_name: str, data: Dict[str, Any]) -> None:
        """Emit event to orchestrator if event bus is available (backward compatible)."""
        if self.orchestrator_event_bus:
            try:
                from ..orchestration.events.event_types import EventType, MLEvent

                # Map string event type to enum
                event_type_map = {
                    "MODEL_DEPLOYED": EventType.MODEL_DEPLOYED,
                    "MODEL_ROLLBACK": EventType.TRAINING_FAILED,  # Reuse existing event type
                }

                event_type = event_type_map.get(event_type_name)
                if event_type:
                    await self.orchestrator_event_bus.emit(MLEvent(
                        event_type=event_type,
                        source="ml_production_service",
                        data=data
                    ))
            except Exception as e:
                # Log error but don't fail the operation
                logger.warning(f"Failed to emit orchestrator event {event_type_name}: {e}")