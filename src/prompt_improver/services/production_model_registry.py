"""Production Model Registry with Alias-Based Deployment.

Advanced MLflow model registry implementation following best practices:
- Alias-based deployment (@champion, @production, @challenger)
- Blue-green deployment capabilities
- Automatic rollback and health monitoring
- Model versioning and environment management
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelAlias(Enum):
    """Model deployment aliases following MLflow best practices."""
    
    CHAMPION = "champion"  # Current best-performing production model
    PRODUCTION = "production"  # Active production model (may be same as champion)
    CHALLENGER = "challenger"  # New model being tested against champion
    STAGING = "staging"  # Pre-production testing
    SHADOW = "shadow"  # Shadow deployment for A/B testing


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    
    BLUE_GREEN = "blue_green"  # Instant traffic switch
    CANARY = "canary"  # Gradual traffic ramp
    SHADOW = "shadow"  # Shadow traffic for testing
    A_B_TEST = "a_b_test"  # A/B split testing


@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment."""
    
    model_name: str
    alias: ModelAlias
    strategy: DeploymentStrategy
    traffic_percentage: float = 100.0
    health_check_interval: int = 60  # seconds
    rollback_threshold: float = 0.05  # 5% performance degradation
    max_latency_ms: int = 500
    min_accuracy: float = 0.8


@dataclass
class ModelMetrics:
    """Model performance metrics for monitoring."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    prediction_count: int
    timestamp: datetime


class ProductionModelRegistry:
    """Production-ready model registry with alias-based deployment."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize production model registry.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to local file store)
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.model_configs: Dict[str, ModelDeploymentConfig] = {}
        self.model_metrics: Dict[str, List[ModelMetrics]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Configure MLflow for production
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        logger.info("Production model registry initialized")
    
    async def register_model(
        self,
        model,
        model_name: str,
        run_id: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """Register a new model version with enhanced metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model in registry
            run_id: MLflow run ID where model was trained
            description: Model description
            tags: Additional model tags
            
        Returns:
            ModelVersion: Registered model version
        """
        try:
            # Register model with MLflow (no description parameter supported)
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_name
            )
            
            # Enhance description with deployment metadata using client
            enhanced_description = f"{description or 'Production model deployment'}\n\nRegistered: {datetime.utcnow().isoformat()}\nRun ID: {run_id}"
            if tags:
                enhanced_description += f"\nTags: {', '.join([f'{k}={v}' for k, v in tags.items()])}"
            
            # Update model version with enhanced description using client
            client = mlflow.MlflowClient()
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=enhanced_description
            )
            
            # Refresh model version to get updated description
            model_version = client.get_model_version(name=model_name, version=model_version.version)
            
            # Add production tags
            production_tags = {
                "deployment_ready": "true",
                "registered_at": datetime.utcnow().isoformat(),
                "python_version": self._get_python_version(),
                "environment": "production",
                **(tags or {})
            }
            
            # Set model version tags
            model_version = self.client.get_model_version(
                name=model_name,
                version=model_info.registered_model_version
            )
            
            for key, value in production_tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    async def deploy_model(
        self,
        model_name: str,
        version: str,
        alias: ModelAlias,
        config: Optional[ModelDeploymentConfig] = None
    ) -> Dict[str, Any]:
        """Deploy model with specified alias using blue-green strategy.
        
        Args:
            model_name: Name of the model
            version: Model version to deploy
            alias: Deployment alias (@champion, @production, etc.)
            config: Deployment configuration
            
        Returns:
            Deployment result with status and metrics
        """
        try:
            deployment_start = time.time()
            
            # Validate model version exists
            model_version = self.client.get_model_version(model_name, version)
            
            # Create deployment config if not provided
            if config is None:
                config = ModelDeploymentConfig(
                    model_name=model_name,
                    alias=alias,
                    strategy=DeploymentStrategy.BLUE_GREEN
                )
            
            # Store configuration
            self.model_configs[f"{model_name}:{alias.value}"] = config
            
            # Perform health check before deployment
            health_check = await self._perform_health_check(model_name, version)
            if not health_check["healthy"]:
                raise ValueError(f"Model health check failed: {health_check['errors']}")
            
            # Get current alias assignment (if any)
            current_alias = await self._get_current_alias(model_name, alias)
            
            # Set new alias
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias.value,
                version=version
            )
            
            # Record deployment
            deployment_record = {
                "model_name": model_name,
                "version": version,
                "alias": alias.value,
                "strategy": config.strategy.value,
                "previous_version": current_alias,
                "deployment_time": datetime.utcnow().isoformat(),
                "duration_seconds": time.time() - deployment_start,
                "health_check": health_check
            }
            
            self.deployment_history.append(deployment_record)
            
            logger.info(f"Successfully deployed {model_name}:{version} with alias @{alias.value}")
            
            return {
                "status": "success",
                "deployment": deployment_record,
                "next_steps": [
                    f"Monitor model performance at alias @{alias.value}",
                    f"Run health checks every {config.health_check_interval} seconds",
                    "Consider A/B testing against previous version"
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}:{version}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rollback_required": True
            }
    
    async def rollback_deployment(
        self,
        model_name: str,
        alias: ModelAlias,
        reason: str = "Performance degradation"
    ) -> Dict[str, Any]:
        """Rollback deployment to previous version.
        
        Args:
            model_name: Name of the model
            alias: Alias to rollback
            reason: Reason for rollback
            
        Returns:
            Rollback result
        """
        try:
            # Find previous deployment
            previous_deployment = None
            for deployment in reversed(self.deployment_history):
                if (deployment["model_name"] == model_name and 
                    deployment["alias"] == alias.value and
                    deployment.get("previous_version")):
                    previous_deployment = deployment
                    break
            
            if not previous_deployment:
                raise ValueError(f"No previous deployment found for {model_name}:@{alias.value}")
            
            previous_version = previous_deployment["previous_version"]
            
            # Rollback to previous version
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias.value,
                version=previous_version
            )
            
            # Record rollback
            rollback_record = {
                "model_name": model_name,
                "alias": alias.value,
                "rolled_back_to": previous_version,
                "reason": reason,
                "rollback_time": datetime.utcnow().isoformat()
            }
            
            self.deployment_history.append(rollback_record)
            
            logger.warning(f"Rolled back {model_name}:@{alias.value} to version {previous_version}: {reason}")
            
            return {
                "status": "success",
                "rollback": rollback_record,
                "current_version": previous_version
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback {model_name}:@{alias.value}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_production_model(
        self,
        model_name: str,
        alias: ModelAlias = ModelAlias.PRODUCTION
    ) -> Any:
        """Load production model by alias.
        
        Args:
            model_name: Name of the model
            alias: Model alias to load
            
        Returns:
            Loaded model object
        """
        try:
            # Get model version by alias
            model_version = self.client.get_model_version_by_alias(
                name=model_name,
                alias=alias.value
            )
            
            # Load model
            model_uri = f"models:/{model_name}@{alias.value}"
            model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"Loaded model {model_name}@{alias.value} (version {model_version.version})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}@{alias.value}: {e}")
            raise
    
    async def monitor_model_health(
        self,
        model_name: str,
        alias: ModelAlias,
        metrics: ModelMetrics
    ) -> Dict[str, Any]:
        """Monitor model health and trigger alerts if needed.
        
        Args:
            model_name: Name of the model
            alias: Model alias
            metrics: Current model metrics
            
        Returns:
            Health status and recommendations
        """
        try:
            model_key = f"{model_name}:{alias.value}"
            
            # Store metrics
            if model_key not in self.model_metrics:
                self.model_metrics[model_key] = []
            
            self.model_metrics[model_key].append(metrics)
            
            # Keep only last 100 metrics
            self.model_metrics[model_key] = self.model_metrics[model_key][-100:]
            
            # Get deployment config
            config = self.model_configs.get(model_key)
            if not config:
                return {"status": "no_config", "healthy": True}
            
            # Check health thresholds
            health_issues = []
            
            if metrics.accuracy < config.min_accuracy:
                health_issues.append(f"Accuracy {metrics.accuracy:.3f} below threshold {config.min_accuracy}")
            
            if metrics.latency_p95 > config.max_latency_ms:
                health_issues.append(f"P95 latency {metrics.latency_p95:.1f}ms above threshold {config.max_latency_ms}ms")
            
            if metrics.error_rate > 0.01:  # 1% error rate
                health_issues.append(f"Error rate {metrics.error_rate:.1%} too high")
            
            # Check for performance degradation
            if len(self.model_metrics[model_key]) >= 10:
                recent_accuracy = np.mean([m.accuracy for m in self.model_metrics[model_key][-5:]])
                baseline_accuracy = np.mean([m.accuracy for m in self.model_metrics[model_key][-10:-5]])
                
                if baseline_accuracy - recent_accuracy > config.rollback_threshold:
                    health_issues.append(f"Performance degraded by {(baseline_accuracy - recent_accuracy):.1%}")
            
            # Determine health status
            is_healthy = len(health_issues) == 0
            
            result = {
                "healthy": is_healthy,
                "issues": health_issues,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "latency_p95": metrics.latency_p95,
                    "error_rate": metrics.error_rate,
                    "predictions": metrics.prediction_count
                },
                "timestamp": metrics.timestamp.isoformat()
            }
            
            # Trigger alerts if unhealthy
            if not is_healthy:
                logger.warning(f"Model {model_name}@{alias.value} health issues: {health_issues}")
                result["recommendations"] = [
                    "Consider rolling back to previous version",
                    "Check training data quality",
                    "Verify feature engineering pipeline",
                    "Investigate recent system changes"
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to monitor model health: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all current deployments with their status.
        
        Returns:
            List of deployment information
        """
        deployments = []
        
        try:
            # Get all registered models
            for rm in self.client.search_registered_models():
                model_name = rm.name
                
                # Get all aliases for this model
                try:
                    aliases = self.client.get_model_version_by_alias(model_name, "")
                except:
                    # No aliases set
                    continue
                
                # Get latest versions
                latest_versions = self.client.get_latest_versions(model_name)
                
                for version in latest_versions:
                    deployment_info = {
                        "model_name": model_name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "created_at": version.creation_timestamp,
                        "description": version.description
                    }
                    
                    # Add health metrics if available
                    model_key = f"{model_name}:production"
                    if model_key in self.model_metrics and self.model_metrics[model_key]:
                        latest_metrics = self.model_metrics[model_key][-1]
                        deployment_info["health"] = {
                            "accuracy": latest_metrics.accuracy,
                            "latency_p95": latest_metrics.latency_p95,
                            "error_rate": latest_metrics.error_rate
                        }
                    
                    deployments.append(deployment_info)
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            return []
    
    # Helper methods
    
    async def _perform_health_check(self, model_name: str, version: str) -> Dict[str, Any]:
        """Perform health check on model version."""
        try:
            # Load model to verify it works
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Test prediction with sample data
            sample_input = self._generate_input_example()
            start_time = time.time()
            prediction = model.predict(sample_input)
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                "healthy": True,
                "latency_ms": latency,
                "prediction_shape": prediction.shape,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "errors": [str(e)],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_current_alias(self, model_name: str, alias: ModelAlias) -> Optional[str]:
        """Get current version assigned to alias."""
        try:
            model_version = self.client.get_model_version_by_alias(
                name=model_name,
                alias=alias.value
            )
            return model_version.version
        except:
            return None
    
    def _infer_model_signature(self, model) -> Any:
        """Infer model signature for MLflow."""
        # Implementation would depend on your specific model types
        # For now, return None to use auto-inference
        return None
    
    def _generate_input_example(self) -> Any:
        """Generate sample input for model testing."""
        # Return sample feature vector for your models
        # Adjust based on your actual feature dimensions
        return np.array([[0.5] * 31])  # 31-dimensional feature vector
    
    def _get_pip_requirements(self) -> List[str]:
        """Get pip requirements for model environment."""
        return [
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "mlflow>=2.8.0"
        ]
    
    def _get_python_version(self) -> str:
        """Get current Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# Factory function for easy integration
async def get_production_registry(tracking_uri: Optional[str] = None) -> ProductionModelRegistry:
    """Get production model registry instance.
    
    Args:
        tracking_uri: MLflow tracking URI
        
    Returns:
        ProductionModelRegistry instance
    """
    return ProductionModelRegistry(tracking_uri=tracking_uri) 