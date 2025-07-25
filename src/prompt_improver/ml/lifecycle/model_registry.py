"""Enhanced Model Registry for ML Lifecycle Management.

This module provides comprehensive model management with:
- Model versioning with semantic versioning support
- Metadata and lineage tracking
- Model validation and approval workflows
- Dependency management
- Integration with experiment tracking
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import mlflow
import numpy as np
from pydantic import BaseModel, Field as PydanticField, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import ABExperiment
from prompt_improver.ml.types import features, labels, model_config, hyper_parameters
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    training = "training"  # Backward compatibility
    VALIDATION = "validation"
    validation = "validation"  # Backward compatibility
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    approved = "approved"  # Backward compatibility
    DEPLOYED = "deployed"
    deployed = "deployed"  # Backward compatibility
    DEPRECATED = "deprecated"
    deprecated = "deprecated"  # Backward compatibility
    ARCHIVED = "archived"
    archived = "archived"  # Backward compatibility

class ModelTier(Enum):
    """Model deployment tiers."""
    development = "development"
    staging = "staging"
    production = "production"
    champion = "champion"
    challenger = "challenger"

class ModelFormat(Enum):
    """Supported model formats."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    MLFLOW_PYFUNC = "mlflow_pyfunc"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    model_name: str
    version: str
    created_at: datetime
    created_by: str
    status: ModelStatus
    tier: Optional[ModelTier] = None
    model_format: ModelFormat = ModelFormat.SKLEARN
    
    # Training metadata
    training_dataset_id: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    training_framework: Optional[str] = None
    training_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    production_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Lineage and dependencies
    parent_model_id: Optional[str] = None
    experiment_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # Deployment info
    deployment_timestamp: Optional[datetime] = None
    deployment_endpoint: Optional[str] = None
    serving_container: Optional[str] = None
    
    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    notes: List[str] = field(default_factory=list)

@dataclass
class ModelLineage:
    """Model lineage tracking."""
    model_id: str
    parent_models: List[str]
    derived_models: List[str]
    training_data_sources: List[str]
    feature_transformations: List[str]
    experiment_history: List[str]
    creation_timestamp: datetime

@dataclass
class ModelValidation:
    """Model validation results."""
    model_id: str
    validation_timestamp: datetime
    is_valid: bool
    validation_checks: Dict[str, bool]
    performance_benchmarks: Dict[str, float]
    data_drift_score: Optional[float] = None
    concept_drift_score: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ModelApprovalRequest(BaseModel):
    """Request for model approval."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_id: str
    requester: str
    reason: str
    target_tier: ModelTier
    validation_results: Optional[ModelValidation] = None
    reviewer_notes: Optional[str] = None
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None

class EnhancedModelRegistry:
    """Enhanced Model Registry with comprehensive lifecycle management."""
    
    def __init__(self, 
                 mlflow_tracking_uri: Optional[str] = None,
                 storage_path: Path = Path("./model_registry")):
        """Initialize enhanced model registry.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            storage_path: Local storage path for model artifacts
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches for fast access
        self._model_metadata: Dict[str, ModelMetadata] = {}
        self._model_lineage: Dict[str, ModelLineage] = {}
        self._approval_requests: Dict[str, ModelApprovalRequest] = {}
        self._active_experiments: Dict[str, Set[str]] = {}
        
        # MLflow client
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
        
        logger.info(f"Enhanced Model Registry initialized at {storage_path}")
    
    async def register_model(self,
                           model: Any,
                           model_name: str,
                           version: str,
                           metadata: ModelMetadata,
                           experiment_id: Optional[str] = None,
                           parent_model_id: Optional[str] = None) -> str:
        """Register a new model with comprehensive metadata.
        
        Args:
            model: The model object to register
            model_name: Name of the model
            version: Semantic version (e.g., "1.2.3")
            metadata: Model metadata
            experiment_id: Associated experiment ID
            parent_model_id: Parent model ID for lineage tracking
            
        Returns:
            Unique model ID
        """
        # Generate unique model ID
        model_id = self._generate_model_id(model_name, version)
        metadata.model_id = model_id
        metadata.model_name = model_name
        metadata.version = version
        metadata.created_at = aware_utc_now()
        
        # Update lineage
        if parent_model_id:
            metadata.parent_model_id = parent_model_id
            await self._update_lineage(model_id, parent_model_id, experiment_id)
        
        # Store metadata
        self._model_metadata[model_id] = metadata
        
        # Register with MLflow if available
        if self.mlflow_tracking_uri:
            try:
                with mlflow.start_run(experiment_id=experiment_id) as run:
                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                    
                    # Log hyperparameters
                    mlflow.log_params(metadata.training_hyperparameters)
                    
                    # Log metrics
                    for metric_name, value in metadata.validation_metrics.items():
                        mlflow.log_metric(f"validation_{metric_name}", value)
                    
                    # Set tags
                    mlflow.set_tags({
                        "model_id": model_id,
                        "version": version,
                        "status": metadata.status.value,
                        **metadata.tags
                    })
                    
                    metadata.tags["mlflow_run_id"] = run.info.run_id
                    
            except Exception as e:
                logger.error(f"Failed to register model with MLflow: {e}")
        
        # Save model artifacts locally
        model_path = self.storage_path / model_name / version
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata_to_dict(metadata), f, indent=2)
        
        logger.info(f"Registered model {model_name} version {version} with ID {model_id}")
        
        # Track in active experiments
        if experiment_id:
            if experiment_id not in self._active_experiments:
                self._active_experiments[experiment_id] = set()
            self._active_experiments[experiment_id].add(model_id)
        
        return model_id
    
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self._model_metadata.get(model_id)
    
    async def list_models(self,
                         status: Optional[ModelStatus] = None,
                         tier: Optional[ModelTier] = None,
                         experiment_id: Optional[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering.
        
        Args:
            status: Filter by model status
            tier: Filter by deployment tier
            experiment_id: Filter by experiment
            
        Returns:
            List of model metadata matching filters
        """
        models = list(self._model_metadata.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tier:
            models = [m for m in models if m.tier == tier]
        
        if experiment_id:
            experiment_models = self._active_experiments.get(experiment_id, set())
            models = [m for m in models if m.model_id in experiment_models]
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    async def update_model_status(self,
                                model_id: str,
                                new_status: ModelStatus,
                                notes: Optional[str] = None) -> bool:
        """Update model status in lifecycle.
        
        Args:
            model_id: Model to update
            new_status: New status
            notes: Optional notes about status change
            
        Returns:
            Success status
        """
        metadata = self._model_metadata.get(model_id)
        if not metadata:
            logger.error(f"Model {model_id} not found")
            return False
        
        old_status = metadata.status
        metadata.status = new_status
        
        if notes:
            metadata.notes.append(f"{aware_utc_now().isoformat()}: Status changed from {old_status.value} to {new_status.value} - {notes}")
        
        # Update MLflow tags if available
        if self.mlflow_tracking_uri and "mlflow_run_id" in metadata.tags:
            try:
                self.mlflow_client.set_tag(
                    metadata.tags["mlflow_run_id"],
                    "status",
                    new_status.value
                )
            except Exception as e:
                logger.error(f"Failed to update MLflow status: {e}")
        
        logger.info(f"Updated model {model_id} status from {old_status.value} to {new_status.value}")
        return True
    
    async def validate_model(self,
                           model_id: str,
                           validation_data: Tuple[features, labels],
                           validation_checks: Optional[Dict[str, Callable]] = None) -> ModelValidation:
        """Validate model before deployment.
        
        Args:
            model_id: Model to validate
            validation_data: Validation dataset (features, labels)
            validation_checks: Custom validation functions
            
        Returns:
            ModelValidation results
        """
        validation = ModelValidation(
            model_id=model_id,
            validation_timestamp=aware_utc_now(),
            is_valid=True,
            validation_checks={},
            performance_benchmarks={}
        )
        
        metadata = self._model_metadata.get(model_id)
        if not metadata:
            validation.is_valid = False
            validation.errors.append(f"Model {model_id} not found")
            return validation
        
        try:
            # Load model (simplified for example)
            # In practice, this would load from MLflow or storage
            model = await self._load_model(model_id)
            
            # Basic validation checks
            features, labels = validation_data
            
            # 1. Prediction check
            try:
                predictions = model.predict(features[:10])  # Test on sample
                validation.validation_checks["prediction_works"] = True
            except Exception as e:
                validation.validation_checks["prediction_works"] = False
                validation.errors.append(f"Prediction failed: {str(e)}")
                validation.is_valid = False
            
            # 2. Performance benchmarks
            if validation.is_valid:
                # Measure prediction latency
                start_time = time.time()
                _ = model.predict(features[:100])
                latency = (time.time() - start_time) / 100 * 1000  # ms per prediction
                validation.performance_benchmarks["latency_ms"] = latency
                
                # Check latency threshold
                if latency > 100:  # 100ms threshold
                    validation.warnings.append(f"High latency: {latency:.2f}ms")
                
                # Calculate accuracy on validation set
                predictions = model.predict(features)
                accuracy = np.mean(predictions == labels)
                validation.performance_benchmarks["accuracy"] = accuracy
                
                # Check accuracy threshold
                if accuracy < 0.8:  # 80% threshold
                    validation.warnings.append(f"Low accuracy: {accuracy:.2%}")
            
            # 3. Custom validation checks
            if validation_checks:
                for check_name, check_func in validation_checks.items():
                    try:
                        result = check_func(model, features, labels)
                        validation.validation_checks[check_name] = result
                        if not result:
                            validation.warnings.append(f"Custom check '{check_name}' failed")
                    except Exception as e:
                        validation.validation_checks[check_name] = False
                        validation.errors.append(f"Custom check '{check_name}' error: {str(e)}")
            
            # 4. Data drift detection (simplified)
            if hasattr(model, "feature_importances_"):
                # Simple feature importance change detection
                validation.data_drift_score = 0.05  # Placeholder
            
            # Overall validation result
            validation.is_valid = len(validation.errors) == 0
            
        except Exception as e:
            validation.is_valid = False
            validation.errors.append(f"Validation error: {str(e)}")
        
        # Store validation results
        if metadata:
            metadata.notes.append(f"{aware_utc_now().isoformat()}: Validation {'passed' if validation.is_valid else 'failed'}")
        
        return validation
    
    async def request_approval(self,
                             model_id: str,
                             requester: str,
                             reason: str,
                             target_tier: ModelTier,
                             validation_results: Optional[ModelValidation] = None) -> str:
        """Request model approval for deployment.
        
        Args:
            model_id: Model requesting approval
            requester: Person requesting approval
            reason: Reason for approval request
            target_tier: Target deployment tier
            validation_results: Validation results
            
        Returns:
            Approval request ID
        """
        request_id = f"approval_{model_id}_{int(time.time())}"
        
        approval_request = ModelApprovalRequest(
            model_id=model_id,
            requester=requester,
            reason=reason,
            target_tier=target_tier,
            validation_results=validation_results
        )
        
        self._approval_requests[request_id] = approval_request
        
        # Update model status
        await self.update_model_status(
            model_id,
            ModelStatus.PENDING_APPROVAL,
            f"Approval requested by {requester} for {target_tier.value}"
        )
        
        logger.info(f"Created approval request {request_id} for model {model_id}")
        return request_id
    
    async def approve_model(self,
                          request_id: str,
                          approved: bool,
                          approved_by: str,
                          reviewer_notes: Optional[str] = None) -> bool:
        """Approve or reject model deployment request.
        
        Args:
            request_id: Approval request ID
            approved: Approval decision
            approved_by: Person approving
            reviewer_notes: Optional reviewer notes
            
        Returns:
            Success status
        """
        request = self._approval_requests.get(request_id)
        if not request:
            logger.error(f"Approval request {request_id} not found")
            return False
        
        request.approved = approved
        request.approved_by = approved_by
        request.approval_timestamp = aware_utc_now()
        request.reviewer_notes = reviewer_notes
        
        # Update model status
        new_status = ModelStatus.approved if approved else ModelStatus.validation
        notes = f"{'Approved' if approved else 'Rejected'} by {approved_by}"
        if reviewer_notes:
            notes += f": {reviewer_notes}"
        
        await self.update_model_status(request.model_id, new_status, notes)
        
        # Update tier if approved
        if approved:
            metadata = self._model_metadata.get(request.model_id)
            if metadata:
                metadata.tier = request.target_tier
        
        logger.info(f"Approval request {request_id} {'approved' if approved else 'rejected'} by {approved_by}")
        return True
    
    async def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get complete model lineage."""
        return self._model_lineage.get(model_id)
    
    async def get_experiment_models(self, experiment_id: str) -> List[str]:
        """Get all models from an experiment."""
        return list(self._active_experiments.get(experiment_id, set()))
    
    async def compare_models(self,
                           model_ids: List[str],
                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple models side by side.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "models": {},
            "best_by_metric": {}
        }
        
        for model_id in model_ids:
            metadata = self._model_metadata.get(model_id)
            if metadata:
                model_info = {
                    "name": metadata.model_name,
                    "version": metadata.version,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "validation_metrics": metadata.validation_metrics,
                    "test_metrics": metadata.test_metrics,
                    "production_metrics": metadata.production_metrics
                }
                comparison["models"][model_id] = model_info
        
        # Find best model for each metric
        if metrics:
            for metric in metrics:
                best_model = None
                best_value = None
                
                for model_id, info in comparison["models"].items():
                    # Check all metric sources
                    for metric_source in ["validation_metrics", "test_metrics", "production_metrics"]:
                        if metric in info[metric_source]:
                            value = info[metric_source][metric]
                            if best_value is None or value > best_value:
                                best_value = value
                                best_model = model_id
                
                if best_model:
                    comparison["best_by_metric"][metric] = {
                        "model_id": best_model,
                        "value": best_value
                    }
        
        return comparison
    
    # Helper methods
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = int(time.time() * 1000)
        return f"{model_name}_{version}_{timestamp}"
    
    async def _update_lineage(self,
                            model_id: str,
                            parent_model_id: str,
                            experiment_id: Optional[str] = None):
        """Update model lineage tracking."""
        # Create or update lineage for new model
        if model_id not in self._model_lineage:
            self._model_lineage[model_id] = ModelLineage(
                model_id=model_id,
                parent_models=[parent_model_id] if parent_model_id else [],
                derived_models=[],
                training_data_sources=[],
                feature_transformations=[],
                experiment_history=[experiment_id] if experiment_id else [],
                creation_timestamp=aware_utc_now()
            )
        else:
            lineage = self._model_lineage[model_id]
            if parent_model_id and parent_model_id not in lineage.parent_models:
                lineage.parent_models.append(parent_model_id)
            if experiment_id and experiment_id not in lineage.experiment_history:
                lineage.experiment_history.append(experiment_id)
        
        # Update parent's lineage
        if parent_model_id and parent_model_id in self._model_lineage:
            parent_lineage = self._model_lineage[parent_model_id]
            if model_id not in parent_lineage.derived_models:
                parent_lineage.derived_models.append(model_id)
    
    async def _load_model(self, model_id: str) -> Any:
        """Load model from storage."""
        # Simplified model loading
        # In practice, this would load from MLflow or file storage
        metadata = self._model_metadata.get(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Try to load from MLflow first
        if "mlflow_model_uri" in metadata.tags:
            try:
                model_uri = metadata.tags["mlflow_model_uri"]
                if metadata.model_format == ModelFormat.SKLEARN:
                    return mlflow.sklearn.load_model(model_uri)
                elif metadata.model_format == ModelFormat.PYTORCH:
                    return mlflow.pytorch.load_model(model_uri)
                elif metadata.model_format == ModelFormat.TENSORFLOW:
                    return mlflow.tensorflow.load_model(model_uri)
                else:
                    return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load from MLflow: {e}")
        
        # Fallback to local storage
        model_path = self.storage_path / metadata.model_name / metadata.version / "model.pkl"
        if model_path.exists():
            import pickle
            with open(model_path, "rb") as f:
                return pickle.load(f)
        
        raise FileNotFoundError(f"Model artifacts not found for {model_id}")
    
    def _metadata_to_dict(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "model_id": metadata.model_id,
            "model_name": metadata.model_name,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "created_by": metadata.created_by,
            "status": metadata.status.value,
            "tier": metadata.tier.value if metadata.tier else None,
            "training_dataset_id": metadata.training_dataset_id,
            "training_duration_seconds": metadata.training_duration_seconds,
            "training_framework": metadata.training_framework,
            "training_hyperparameters": metadata.training_hyperparameters,
            "validation_metrics": metadata.validation_metrics,
            "test_metrics": metadata.test_metrics,
            "production_metrics": metadata.production_metrics,
            "parent_model_id": metadata.parent_model_id,
            "experiment_id": metadata.experiment_id,
            "dependencies": metadata.dependencies,
            "deployment_timestamp": metadata.deployment_timestamp.isoformat() if metadata.deployment_timestamp else None,
            "deployment_endpoint": metadata.deployment_endpoint,
            "serving_container": metadata.serving_container,
            "tags": metadata.tags,
            "description": metadata.description,
            "notes": metadata.notes
        }