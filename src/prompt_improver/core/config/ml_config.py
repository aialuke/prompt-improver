"""ML Configuration Module

Configuration for machine learning pipelines, model serving,
external ML services, and orchestration settings.
"""

import os
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ModelServingConfig(BaseModel):
    """Model serving and inference configuration."""

    model_timeout: float = Field(
        default=120.0, gt=0, description="Model inference timeout in seconds"
    )
    max_batch_size: int = Field(
        default=32, ge=1, le=1000, description="Maximum batch size for inference"
    )
    model_cache_size: int = Field(
        default=100, ge=1, le=1000, description="Model cache size"
    )
    enable_model_caching: bool = Field(
        default=True, description="Enable model caching"
    )
    cache_ttl_seconds: int = Field(
        default=3600, gt=0, description="Model cache TTL"
    )
    enable_gpu_inference: bool = Field(
        default=False, description="Enable GPU inference"
    )
    max_workers: int = Field(
        default=4, ge=1, le=32, description="Maximum worker processes"
    )


class TrainingConfig(BaseModel):
    """ML training pipeline configuration."""

    training_data_path: str = Field(
        default="data/training", description="Training data directory"
    )
    model_storage_path: str = Field(
        default="models", description="Model storage directory"
    )
    experiment_tracking_enabled: bool = Field(
        default=True, description="Enable experiment tracking"
    )
    auto_hyperparameter_tuning: bool = Field(
        default=True, description="Enable automatic hyperparameter tuning"
    )
    max_training_time_hours: int = Field(
        default=24, ge=1, le=168, description="Maximum training time"
    )
    early_stopping_patience: int = Field(
        default=10, ge=1, description="Early stopping patience"
    )
    checkpoint_frequency: int = Field(
        default=100, ge=1, description="Model checkpoint frequency"
    )


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration."""

    max_feature_dimension: int = Field(
        default=10000, ge=1, description="Maximum feature dimension"
    )
    enable_automatic_feature_selection: bool = Field(
        default=True, description="Enable automatic feature selection"
    )
    feature_selection_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Feature selection threshold"
    )
    enable_feature_scaling: bool = Field(
        default=True, description="Enable feature scaling"
    )
    scaling_method: str = Field(
        default="standard", description="Feature scaling method"
    )

    @field_validator("scaling_method")
    @classmethod
    def validate_scaling_method(cls, v):
        """Validate scaling method."""
        allowed = ["standard", "minmax", "robust", "quantile"]
        if v not in allowed:
            raise ValueError(f"Scaling method must be one of {allowed}")
        return v


class ExternalServicesConfig(BaseModel):
    """External ML services configuration."""

    # PostgreSQL for ML (optional)
    postgres_host: str = Field(
        default_factory=lambda: os.getenv("ML_POSTGRES_HOST", "ml-postgres"),
        description="ML PostgreSQL host",
    )
    postgres_port: int = Field(
        default=5432, ge=1, le=65535, description="ML PostgreSQL port"
    )
    postgres_database: str = Field(
        default="ml_automl", description="ML PostgreSQL database name"
    )
    postgres_username: str = Field(
        default="postgres", description="ML PostgreSQL username"
    )
    postgres_password: str = Field(
        default="postgres", description="ML PostgreSQL password"
    )

    # Redis for ML caching (optional)
    redis_host: str = Field(
        default_factory=lambda: os.getenv("ML_REDIS_HOST", "ml-redis"),
        description="ML Redis host",
    )
    redis_port: int = Field(
        default=6379, ge=1, le=65535, description="ML Redis port"
    )
    redis_password: str | None = Field(default=None, description="ML Redis password")
    redis_db: int = Field(
        default=1, ge=0, le=15, description="ML Redis database number"
    )

    # MLFlow (optional)
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("ML_MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        description="MLFlow tracking server URI",
    )
    mlflow_host: str = Field(
        default_factory=lambda: os.getenv("ML_MLFLOW_HOST", "mlflow"),
        description="MLFlow server host",
    )
    mlflow_port: int = Field(
        default=5000, ge=1, le=65535, description="MLFlow server port"
    )


class OrchestrationConfig(BaseModel):
    """ML orchestration and pipeline configuration."""

    enable_ml_orchestration: bool = Field(
        default=True, description="Enable ML orchestration system"
    )
    max_concurrent_experiments: int = Field(
        default=5, ge=1, le=50, description="Maximum concurrent experiments"
    )
    experiment_timeout_hours: int = Field(
        default=6, ge=1, le=72, description="Experiment timeout in hours"
    )
    enable_distributed_training: bool = Field(
        default=False, description="Enable distributed training"
    )
    resource_monitoring_enabled: bool = Field(
        default=True, description="Enable resource monitoring for ML workloads"
    )
    auto_scaling_enabled: bool = Field(
        default=True, description="Enable auto-scaling for ML resources"
    )


class MLConfig(BaseSettings):
    """Main ML configuration with all ML components."""

    # Global ML Settings
    enabled: bool = Field(default=True, description="Enable ML features")
    debug_mode: bool = Field(default=False, description="Enable ML debug mode")

    # Component Configurations
    model_serving: ModelServingConfig = Field(default_factory=ModelServingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)

    # Performance Settings
    performance_monitoring_enabled: bool = Field(
        default=True, description="Enable ML performance monitoring"
    )
    memory_limit_gb: int = Field(
        default=8, ge=1, le=128, description="Memory limit for ML processes in GB"
    )
    cpu_limit: int = Field(
        default=4, ge=1, le=64, description="CPU limit for ML processes"
    )

    # Quality and Validation
    enable_model_validation: bool = Field(
        default=True, description="Enable model validation"
    )
    validation_split: float = Field(
        default=0.2, gt=0, lt=1, description="Validation data split ratio"
    )
    minimum_model_accuracy: float = Field(
        default=0.8, gt=0, le=1, description="Minimum acceptable model accuracy"
    )

    # Data Management
    enable_data_versioning: bool = Field(
        default=True, description="Enable data versioning"
    )
    max_dataset_size_gb: int = Field(
        default=10, ge=1, le=1000, description="Maximum dataset size in GB"
    )
    data_cleanup_enabled: bool = Field(
        default=True, description="Enable automatic data cleanup"
    )

    model_config = {
        "env_prefix": "ML_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_assignment": True,
    }