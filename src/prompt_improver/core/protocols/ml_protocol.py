"""Protocol definitions for ML operations.

Provides type-safe interface contracts for machine learning components,
enabling dependency inversion and improved testability.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for ML model operations"""

    async def predict(self, inputs: Any) -> Any:
        """Make prediction with the model"""
        ...

    async def predict_batch(self, inputs: list[Any]) -> list[Any]:
        """Make batch predictions"""
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata and information"""
        ...

    def get_model_version(self) -> str:
        """Get model version"""
        ...


@runtime_checkable
class ModelRegistryProtocol(Protocol):
    """Protocol for model registry operations"""

    async def register_model(
        self, name: str, model: Any, metadata: dict[str, Any]
    ) -> str:
        """Register a new model"""
        ...

    async def get_model(self, name: str, version: str | None = None) -> ModelProtocol:
        """Get model by name and version"""
        ...

    async def list_models(self) -> list[dict[str, Any]]:
        """List all registered models"""
        ...

    async def get_model_versions(self, name: str) -> list[str]:
        """Get all versions of a model"""
        ...

    async def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a stage (staging, production, etc.)"""
        ...


@runtime_checkable
class ExperimentTrackingProtocol(Protocol):
    """Protocol for experiment tracking"""

    async def start_experiment(self, name: str, description: str = "") -> str:
        """Start a new experiment"""
        ...

    async def log_parameters(self, experiment_id: str, params: dict[str, Any]) -> None:
        """Log experiment parameters"""
        ...

    async def log_metrics(self, experiment_id: str, metrics: dict[str, float]) -> None:
        """Log experiment metrics"""
        ...

    async def log_artifacts(
        self, experiment_id: str, artifacts: dict[str, Path]
    ) -> None:
        """Log experiment artifacts"""
        ...

    async def end_experiment(self, experiment_id: str, status: str) -> None:
        """End experiment with status"""
        ...


@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Protocol for feature store operations"""

    async def get_features(
        self, feature_names: list[str], entity_ids: list[str]
    ) -> dict[str, Any]:
        """Get features for entities"""
        ...

    async def get_historical_features(
        self,
        feature_names: list[str],
        entity_ids: list[str],
        timestamp_range: tuple[datetime, datetime],
    ) -> dict[str, Any]:
        """Get historical features"""
        ...

    async def register_feature_group(self, name: str, schema: dict[str, Any]) -> bool:
        """Register a new feature group"""
        ...


@runtime_checkable
class DataPipelineProtocol(Protocol):
    """Protocol for data pipeline operations"""

    async def load_data(self, source: str, **kwargs) -> Any:
        """Load data from source"""
        ...

    async def transform_data(self, data: Any, transformations: list[str]) -> Any:
        """Apply transformations to data"""
        ...

    async def validate_data(self, data: Any, schema: dict[str, Any]) -> bool:
        """Validate data against schema"""
        ...

    async def save_data(self, data: Any, destination: str, **kwargs) -> bool:
        """Save data to destination"""
        ...


@runtime_checkable
class ModelTrainingProtocol(Protocol):
    """Protocol for model training operations"""

    async def train_model(
        self, algorithm: str, training_data: Any, hyperparameters: dict[str, Any]
    ) -> tuple[ModelProtocol, dict[str, Any]]:
        """Train a model and return model + metrics"""
        ...

    async def validate_model(
        self, model: ModelProtocol, validation_data: Any
    ) -> dict[str, float]:
        """Validate model performance"""
        ...

    async def tune_hyperparameters(
        self, algorithm: str, training_data: Any, param_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Tune hyperparameters"""
        ...


@runtime_checkable
class ModelServingProtocol(Protocol):
    """Protocol for model serving operations"""

    async def deploy_model(
        self, model_name: str, version: str, config: dict[str, Any]
    ) -> str:
        """Deploy model for serving"""
        ...

    async def undeploy_model(self, deployment_id: str) -> bool:
        """Undeploy model"""
        ...

    async def serve_prediction(self, deployment_id: str, inputs: Any) -> Any:
        """Serve prediction from deployed model"""
        ...

    async def get_serving_metrics(self, deployment_id: str) -> dict[str, Any]:
        """Get serving performance metrics"""
        ...


@runtime_checkable
class AutoMLProtocol(Protocol):
    """Protocol for AutoML operations"""

    async def auto_train(
        self, dataset: Any, target_column: str, task_type: str, time_budget: int
    ) -> tuple[ModelProtocol, dict[str, Any]]:
        """Automatically train best model"""
        ...

    async def suggest_features(self, dataset: Any, target_column: str) -> list[str]:
        """Suggest relevant features"""
        ...

    async def suggest_algorithms(self, dataset: Any, task_type: str) -> list[str]:
        """Suggest appropriate algorithms"""
        ...


@runtime_checkable
class MLMonitoringProtocol(Protocol):
    """Protocol for ML monitoring operations"""

    async def monitor_model_drift(
        self, model_name: str, new_data: Any
    ) -> dict[str, Any]:
        """Monitor for model drift"""
        ...

    async def monitor_data_quality(
        self, data: Any, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Monitor data quality"""
        ...

    async def get_model_performance(
        self, model_name: str, timeframe: str
    ) -> dict[str, Any]:
        """Get model performance metrics"""
        ...


@runtime_checkable
class MLPlatformProtocol(
    ModelRegistryProtocol,
    ExperimentTrackingProtocol,
    FeatureStoreProtocol,
    DataPipelineProtocol,
    ModelTrainingProtocol,
    ModelServingProtocol,
    AutoMLProtocol,
    MLMonitoringProtocol,
    Protocol,
):
    """Combined protocol for complete ML platform"""

    async def get_platform_health(self) -> dict[str, Any]:
        """Get overall ML platform health"""

    async def initialize_platform(self, config: dict[str, Any]) -> bool:
        """Initialize ML platform"""

    async def shutdown_platform(self) -> bool:
        """Shutdown ML platform gracefully"""
