"""Consolidated ML protocol definitions with enhanced lazy loading.

Consolidates 1,317 lines of ML protocols from 4 separate files with
lazy loading to avoid torch/tensorflow import dependencies.

Consolidated from:
- /core/protocols/ml_protocols.py (397 lines) - Infrastructure protocols
- /core/protocols/ml_protocol.py (241 lines) - Business domain protocols
- /ml/core/protocols.py (250 lines) - Service layer protocols
- /repositories/protocols/ml_repository_protocol.py (429 lines) - Repository protocols
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Type hints available during type checking without runtime import
    from sqlalchemy.ext.asyncio import AsyncSession

# =============================================================================
# CORE INFRASTRUCTURE PROTOCOLS (from ml_protocols.py)
# =============================================================================


class ServiceStatus(Enum):
    """Service health status enumeration."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ComponentSpec:
    """Specification for a component that can be loaded."""
    name: str
    module_path: str
    class_name: str
    tier: str
    dependencies: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    enabled: bool = True


@dataclass
class ServiceConnectionInfo:
    """Information about service connections."""
    service_name: str
    connection_status: ServiceStatus
    connection_details: dict[str, Any] | None = None
    last_check: str | None = None
    error_message: str | None = None


@runtime_checkable
class MLflowServiceProtocol(Protocol):
    """Protocol for MLflow service interactions."""

    @abstractmethod
    async def log_experiment(
        self, experiment_name: str, parameters: dict[str, Any]
    ) -> str:
        """Log an ML experiment and return run ID."""
        ...

    @abstractmethod
    async def log_model(
        self, model_name: str, model_data: Any, metadata: dict[str, Any]
    ) -> str:
        """Log a model and return model URI."""
        ...

    @abstractmethod
    async def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Retrieve model metadata by ID."""
        ...

    @abstractmethod
    async def start_trace(
        self, trace_name: str, attributes: dict[str, Any] | None = None
    ) -> str:
        """Start MLflow tracing and return trace ID."""
        ...

    @abstractmethod
    async def end_trace(
        self, trace_id: str, outputs: dict[str, Any] | None = None
    ) -> None:
        """End MLflow trace with outputs."""
        ...


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for cache service (Redis) interactions."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set key-value with optional TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key and return success status."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    async def health_check(self) -> ServiceStatus:
        """Check cache service health."""
        ...


@runtime_checkable
class DatabaseServiceProtocol(Protocol):
    """Protocol for database service interactions."""

    @abstractmethod
    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return results."""
        ...

    @abstractmethod
    async def execute_transaction(
        self, queries: list[str], parameters: list[dict[str, Any]] | None = None
    ) -> None:
        """Execute multiple queries in transaction."""
        ...

    @abstractmethod
    async def health_check(self) -> ServiceStatus:
        """Check database service health."""
        ...

    @abstractmethod
    async def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        ...


@runtime_checkable
class ExternalServicesConfigProtocol(Protocol):
    """Protocol for external services configuration."""

    @abstractmethod
    def get_service_endpoint(self, service_name: str) -> str:
        """Get endpoint URL for external service."""
        ...

    @abstractmethod
    def get_service_config(self, service_name: str) -> dict[str, Any]:
        """Get configuration for external service."""
        ...

    @abstractmethod
    def is_service_enabled(self, service_name: str) -> bool:
        """Check if external service is enabled."""
        ...


@runtime_checkable
class ResourceManagerProtocol(Protocol):
    """Protocol for ML resource management."""

    @abstractmethod
    async def allocate_resources(self, task_id: str, requirements: dict[str, Any]) -> dict[str, Any]:
        """Allocate resources for ML task."""
        ...

    @abstractmethod
    async def release_resources(self, task_id: str) -> None:
        """Release resources allocated to task."""
        ...

    @abstractmethod
    def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        ...

    @abstractmethod
    async def check_resource_availability(self, requirements: dict[str, Any]) -> bool:
        """Check if resources are available for task."""
        ...


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Protocol for ML workflow orchestration."""

    @abstractmethod
    async def execute_workflow(self, workflow_id: str, parameters: dict[str, Any]) -> str:
        """Execute ML workflow and return execution ID."""
        ...

    @abstractmethod
    async def get_workflow_status(self, execution_id: str) -> dict[str, Any]:
        """Get workflow execution status."""
        ...

    @abstractmethod
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel running workflow execution."""
        ...

    @abstractmethod
    def register_workflow(self, workflow_definition: dict[str, Any]) -> str:
        """Register new workflow definition."""
        ...


@runtime_checkable
class ComponentFactoryProtocol(Protocol):
    """Protocol for ML component factory operations."""

    @abstractmethod
    def create_component(self, component_type: str, config: dict[str, Any]) -> Any:
        """Create ML component instance."""
        ...

    @abstractmethod
    def register_component_type(self, component_type: str, factory_func: Any) -> None:
        """Register new component type with factory function."""
        ...

    @abstractmethod
    def get_supported_types(self) -> list[str]:
        """Get list of supported component types."""
        ...


@runtime_checkable
class ComponentRegistryProtocol(Protocol):
    """Protocol for ML component registry operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component registry."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component registry."""
        ...

    @abstractmethod
    async def discover_components(self, tier: str | None = None) -> list[ComponentSpec]:
        """Discover available components, optionally filtered by tier."""
        ...

    @abstractmethod
    async def register_component(self, spec: ComponentSpec) -> None:
        """Register a component specification."""
        ...

    @abstractmethod
    async def get_component_spec(self, component_name: str) -> ComponentSpec | None:
        """Get component specification by name."""
        ...

    @abstractmethod
    async def get_component_status(self, component_name: str) -> ServiceStatus:
        """Get component health status."""
        ...

    @abstractmethod
    async def list_registered_components(self) -> list[str]:
        """List all registered component names."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for ML event bus operations."""

    @abstractmethod
    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to the bus."""
        ...

    @abstractmethod
    async def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to events of a specific type."""
        ...

    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe from events."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the event bus."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        ...


@runtime_checkable
class ComponentLoaderProtocol(Protocol):
    """Protocol for ML component loader operations."""

    @abstractmethod
    async def load_component(self, spec: ComponentSpec) -> Any:
        """Load a component from specification."""
        ...

    @abstractmethod
    async def unload_component(self, component_name: str) -> None:
        """Unload a component."""
        ...

    @abstractmethod
    def is_component_loaded(self, component_name: str) -> bool:
        """Check if component is loaded."""
        ...


@runtime_checkable
class ComponentInvokerProtocol(Protocol):
    """Protocol for ML component invoker operations."""

    @abstractmethod
    async def invoke_component(
        self, component_name: str, method_name: str, **kwargs: Any
    ) -> Any:
        """Invoke a method on a component."""
        ...

    @abstractmethod
    def get_component_methods(self, component_name: str) -> list[str]:
        """Get available methods for a component."""
        ...


@runtime_checkable
class HealthMonitorProtocol(Protocol):
    """Protocol for ML health monitoring operations."""

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Check overall system health."""
        ...

    @abstractmethod
    async def check_component_health(self, component_name: str) -> ServiceStatus:
        """Check health of specific component."""
        ...

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        ...

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        ...


# =============================================================================
# BUSINESS DOMAIN PROTOCOLS (from ml_protocol.py)
# =============================================================================

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for ML model operations."""

    async def predict(self, inputs: Any) -> Any:
        """Make prediction with the model."""
        ...

    async def predict_batch(self, inputs: list[Any]) -> list[Any]:
        """Make batch predictions."""
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata and information."""
        ...

    def get_model_version(self) -> str:
        """Get model version."""
        ...


@runtime_checkable
class BusinessModelRegistryProtocol(Protocol):
    """Protocol for model registry operations with @champion/@production/@challenger support."""

    async def register_model(
        self, name: str, model: Any, metadata: dict[str, Any]
    ) -> str:
        """Register a new model."""
        ...

    async def get_model(self, name: str, version: str | None = None) -> ModelProtocol:
        """Get model by name and version."""
        ...

    async def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        ...

    async def get_model_versions(self, name: str) -> list[str]:
        """Get all versions of a model."""
        ...

    async def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a stage (@champion, @production, @challenger)."""
        ...


@runtime_checkable
class ExperimentTrackingProtocol(Protocol):
    """Protocol for experiment tracking."""

    async def start_experiment(self, name: str, description: str = "") -> str:
        """Start a new experiment."""
        ...

    async def log_parameters(self, experiment_id: str, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        ...

    async def log_metrics(self, experiment_id: str, metrics: dict[str, float]) -> None:
        """Log experiment metrics."""
        ...

    async def log_artifacts(
        self, experiment_id: str, artifacts: dict[str, "Path"]
    ) -> None:
        """Log experiment artifacts."""
        ...

    async def end_experiment(self, experiment_id: str, status: str) -> None:
        """End experiment with status."""
        ...


@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Protocol for feature store operations."""

    async def get_features(
        self, feature_names: list[str], entity_ids: list[str]
    ) -> dict[str, Any]:
        """Get features for entities."""
        ...

    async def get_historical_features(
        self,
        feature_names: list[str],
        entity_ids: list[str], 
        timestamp_range: tuple[datetime, datetime],
    ) -> dict[str, Any]:
        """Get historical features."""
        ...

    async def register_feature_group(self, name: str, schema: dict[str, Any]) -> bool:
        """Register a new feature group."""
        ...


@runtime_checkable
class MLPlatformProtocol(
    BusinessModelRegistryProtocol,
    ExperimentTrackingProtocol,
    FeatureStoreProtocol,
    Protocol,
):
    """Combined protocol for complete ML platform."""

    async def get_platform_health(self) -> dict[str, Any]:
        """Get overall ML platform health."""
        ...

    async def initialize_platform(self, config: dict[str, Any]) -> bool:
        """Initialize ML platform."""
        ...

    async def shutdown_platform(self) -> bool:
        """Shutdown ML platform gracefully."""
        ...


# =============================================================================
# SERVICE LAYER PROTOCOLS (from ml/core/protocols.py)
# =============================================================================

@runtime_checkable
class ServiceModelRegistryProtocol(Protocol):
    """Protocol for model registry implementations."""

    def get_model(self, model_id: str) -> Any | None:
        """Get model from registry."""
        ...

    def add_model(
        self,
        model_id: str,
        model: Any,
        model_type: str = "sklearn",
        ttl_minutes: int | None = None
    ) -> bool:
        """Add model to registry."""
        ...

    def remove_model(self, model_id: str) -> bool:
        """Remove model from registry."""
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        ...


@runtime_checkable
class TrainingServiceProtocol(Protocol):
    """Protocol for ML training services."""

    async def optimize_rules(
        self,
        training_data: dict[str, list],
        db_session: "AsyncSession",
        rule_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Optimize rule parameters using ML training."""
        ...

    async def optimize_ensemble_rules(
        self,
        training_data: dict[str, list],
        db_session: "AsyncSession"
    ) -> dict[str, Any]:
        """Optimize rules using ensemble methods."""
        ...


@runtime_checkable
class InferenceServiceProtocol(Protocol):
    """Protocol for ML inference services."""

    async def predict_rule_effectiveness(
        self,
        model_id: str,
        rule_features: list[float]
    ) -> dict[str, Any]:
        """Predict rule effectiveness using trained model."""
        ...


@runtime_checkable
class MLServiceProtocol(Protocol):
    """Main protocol for the ML service facade."""

    model_registry: ServiceModelRegistryProtocol

    # Training methods
    async def optimize_rules(
        self,
        training_data: dict[str, list],
        db_session: "AsyncSession",
        rule_ids: list[str] | None = None
    ) -> dict[str, Any]:
        ...

    async def optimize_ensemble_rules(
        self,
        training_data: dict[str, list],
        db_session: "AsyncSession"
    ) -> dict[str, Any]:
        ...

    # Inference methods
    async def predict_rule_effectiveness(
        self,
        model_id: str,
        rule_features: list[float]
    ) -> dict[str, Any]:
        ...


# =============================================================================
# REPOSITORY PROTOCOLS (from repositories/protocols/ml_repository_protocol.py)
# =============================================================================

class TrainingSessionFilter:
    """Filter criteria for training session queries."""
    status: str | None = None
    continuous_mode: bool | None = None
    min_performance: float | None = None
    max_performance: float | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    active_workflow_id: str | None = None


@runtime_checkable
class MLRepositoryProtocol(Protocol):
    """Protocol for ML operations data access."""

    # Training Session Management
    async def create_training_session(
        self, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new training session."""
        ...

    async def get_training_sessions(
        self,
        filters: TrainingSessionFilter | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve training sessions with filtering."""
        ...

    async def get_training_session_by_id(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get training session by ID."""
        ...


# =============================================================================
# ENHANCED LAZY LOADING IMPLEMENTATION
# =============================================================================

# Protocol cache for <2ms resolution performance
_PROTOCOL_CACHE: dict[str, Any] = {}


def get_ml_protocols():
    """Enhanced lazy load ML protocols with domain-modular loading.

    Maintains <2ms protocol resolution through caching while avoiding
    torch/tensorflow imports during regular application startup.

    Returns consolidated protocols by domain:
    - infrastructure: Core ML infrastructure protocols
    - business: Business domain protocols
    - service: Service layer protocols
    - repository: Repository protocols
    """
    if "consolidated_protocols" not in _PROTOCOL_CACHE:
        _PROTOCOL_CACHE["consolidated_protocols"] = {
            # Infrastructure Layer
            "infrastructure": {
                "MLflowServiceProtocol": MLflowServiceProtocol,
                "CacheServiceProtocol": CacheServiceProtocol,
                "DatabaseServiceProtocol": DatabaseServiceProtocol,
                "ExternalServicesConfigProtocol": ExternalServicesConfigProtocol,
                "ResourceManagerProtocol": ResourceManagerProtocol,
                "WorkflowEngineProtocol": WorkflowEngineProtocol,
                "ComponentFactoryProtocol": ComponentFactoryProtocol,
                "ComponentRegistryProtocol": ComponentRegistryProtocol,
                "EventBusProtocol": EventBusProtocol,
                "ComponentLoaderProtocol": ComponentLoaderProtocol,
                "ComponentInvokerProtocol": ComponentInvokerProtocol,
                "HealthMonitorProtocol": HealthMonitorProtocol,
                "ComponentSpec": ComponentSpec,
                "ServiceConnectionInfo": ServiceConnectionInfo,
                "ServiceStatus": ServiceStatus,
            },
            # Business Domain Layer
            "business": {
                "ModelProtocol": ModelProtocol,
                "BusinessModelRegistryProtocol": BusinessModelRegistryProtocol,
                "ExperimentTrackingProtocol": ExperimentTrackingProtocol,
                "FeatureStoreProtocol": FeatureStoreProtocol,
                "MLPlatformProtocol": MLPlatformProtocol,
            },
            # Service Layer
            "service": {
                "ServiceModelRegistryProtocol": ServiceModelRegistryProtocol,
                "TrainingServiceProtocol": TrainingServiceProtocol,
                "InferenceServiceProtocol": InferenceServiceProtocol,
                "MLServiceProtocol": MLServiceProtocol,
            },
            # Repository Layer
            "repository": {
                "MLRepositoryProtocol": MLRepositoryProtocol,
                "TrainingSessionFilter": TrainingSessionFilter,
            },
        }

    return _PROTOCOL_CACHE["consolidated_protocols"]


def get_ml_protocol_by_name(protocol_name: str) -> Any | None:
    """Get specific ML protocol by name with <2ms resolution."""
    protocols = get_ml_protocols()
    for domain_protocols in protocols.values():
        if protocol_name in domain_protocols:
            return domain_protocols[protocol_name]
    return None


def get_production_model_registry_protocols() -> dict[str, Any]:
    """Get model registry protocols supporting @champion/@production/@challenger aliases."""
    protocols = get_ml_protocols()
    return {
        "BusinessModelRegistryProtocol": protocols["business"]["BusinessModelRegistryProtocol"],
        "ServiceModelRegistryProtocol": protocols["service"]["ServiceModelRegistryProtocol"],
    }