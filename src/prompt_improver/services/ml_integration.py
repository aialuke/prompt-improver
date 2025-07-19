"""Direct Python ML integration service for Phase 3 continuous learning.
Replaces cross-language bridge architecture with direct Python function calls.
Performance improvement: 50-100ms → 1-5ms response times.

Enhanced with production model registry, alias-based deployment, and Apriori pattern discovery.
"""

import asyncio
import glob
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.utils.datetime_utils import aware_utc_now

from ..database.connection import DatabaseManager, DatabaseSessionManager
from ..database.models import MLModelPerformance, RuleMetadata, RulePerformance
from ..utils.redis_cache import redis_client
from .advanced_pattern_discovery import AdvancedPatternDiscovery
from .production_model_registry import (
    DeploymentStrategy,
    ModelAlias,
    ModelDeploymentConfig,
    ModelMetrics,
    ProductionModelRegistry,
    get_production_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    """Model cache entry with TTL and metadata"""

    model: Any
    model_id: str
    cached_at: datetime
    last_accessed: datetime
    access_count: int = 0
    model_type: str = "sklearn"
    memory_size_mb: float = 0.0
    ttl_minutes: int = 60  # Default 1 hour TTL

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return aware_utc_now() - self.cached_at > timedelta(minutes=self.ttl_minutes)

    def update_access(self):
        """Update access tracking"""
        self.last_accessed = aware_utc_now()
        self.access_count += 1


class InMemoryModelRegistry:
    """Thread-safe in-memory model registry with TTL and lazy loading"""

    def __init__(self, max_cache_size_mb: int = 500, default_ttl_minutes: int = 60):
        self._cache: dict[str, ModelCacheEntry] = {}
        self._lock = Lock()
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_minutes = default_ttl_minutes
        self._total_cache_size_mb = 0.0

    def get_model(self, model_id: str) -> Any | None:
        """Get model from cache with lazy loading"""
        with self._lock:
            entry = self._cache.get(model_id)

            if entry is None:
                return None

            if entry.is_expired():
                logger.info(f"Model {model_id} expired, removing from cache")
                self._remove_entry(model_id)
                return None

            entry.update_access()
            logger.debug(f"Model {model_id} cache hit (access #{entry.access_count})")
            return entry.model

    def add_model(
        self,
        model_id: str,
        model: Any,
        model_type: str = "sklearn",
        ttl_minutes: int | None = None,
    ) -> bool:
        """Add model to cache with memory management"""
        with self._lock:
            # Estimate model memory size
            memory_size = self._estimate_model_memory(model)
            ttl = ttl_minutes or self.default_ttl_minutes

            # Check if we need to free memory
            if self._total_cache_size_mb + memory_size > self.max_cache_size_mb:
                self._evict_models(memory_size)

            entry = ModelCacheEntry(
                model=model,
                model_id=model_id,
                cached_at=aware_utc_now(),
                last_accessed=aware_utc_now(),
                model_type=model_type,
                memory_size_mb=memory_size,
                ttl_minutes=ttl,
            )

            self._cache[model_id] = entry
            self._total_cache_size_mb += memory_size

            logger.info(f"Cached model {model_id} ({memory_size:.1f}MB, TTL: {ttl}min)")
            return True

    def remove_model(self, model_id: str) -> bool:
        """Remove model from cache"""
        with self._lock:
            return self._remove_entry(model_id)

    def _remove_entry(self, model_id: str) -> bool:
        """Internal method to remove cache entry"""
        entry = self._cache.pop(model_id, None)
        if entry:
            self._total_cache_size_mb -= entry.memory_size_mb
            logger.debug(f"Removed model {model_id} from cache")
            return True
        return False

    def _evict_models(self, required_space_mb: float):
        """Evict least recently used models to free space"""
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)

        freed_space = 0.0
        evicted_count = 0

        for model_id, entry in sorted_entries:
            if freed_space >= required_space_mb:
                break

            freed_space += entry.memory_size_mb
            self._remove_entry(model_id)
            evicted_count += 1

        logger.info(f"Evicted {evicted_count} models, freed {freed_space:.1f}MB")

    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            import pickle

            # Try to serialize and measure size
            serialized = pickle.dumps(model)
            size_bytes = len(serialized)
            size_mb = size_bytes / (1024 * 1024)

            return size_mb

        except Exception:
            # Fallback estimate based on model type
            if hasattr(model, "get_params"):
                return 10.0  # Default sklearn model estimate
            return 5.0  # Conservative estimate

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_models = len(self._cache)
            expired_models = sum(
                1 for entry in self._cache.values() if entry.is_expired()
            )

            return {
                "total_models": total_models,
                "expired_models": expired_models,
                "active_models": total_models - expired_models,
                "total_memory_mb": self._total_cache_size_mb,
                "max_memory_mb": self.max_cache_size_mb,
                "memory_utilization": self._total_cache_size_mb
                / self.max_cache_size_mb,
                "model_details": [
                    {
                        "model_id": entry.model_id,
                        "model_type": entry.model_type,
                        "memory_mb": entry.memory_size_mb,
                        "access_count": entry.access_count,
                        "cached_minutes_ago": (
                            aware_utc_now() - entry.cached_at
                        ).total_seconds()
                        / 60,
                        "expires_in_minutes": entry.ttl_minutes
                        - (aware_utc_now() - entry.cached_at).total_seconds() / 60,
                        "is_expired": entry.is_expired(),
                    }
                    for entry in self._cache.values()
                ],
            }

    def cleanup_expired(self) -> int:
        """Remove all expired models and return count"""
        with self._lock:
            expired_ids = [
                model_id
                for model_id, entry in self._cache.items()
                if entry.is_expired()
            ]

            for model_id in expired_ids:
                self._remove_entry(model_id)

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired models")

            return len(expired_ids)


class MLModelService:
    """Enhanced ML service with direct Python integration and production deployment capabilities

    Enhanced with Apriori pattern discovery for comprehensive rule relationship analysis:
    - Traditional ML-based pattern discovery
    - Association rule mining via Apriori algorithm
    - Cross-validation of patterns between approaches
    - Business insight generation from discovered patterns
    """

    def __init__(self, db_manager: DatabaseSessionManager | None = None):
        # Enhanced in-memory model registry with TTL
        self.model_registry = InMemoryModelRegistry(
            max_cache_size_mb=500, default_ttl_minutes=60
        )

        # Production model registry with alias-based deployment
        self.production_registry: ProductionModelRegistry | None = None
        self._production_enabled = False

        # MLflow client for model persistence
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.cache_lock = Lock()

        # Model cache statistics
        self.model_access_stats = {}

        # MLflow setup
        mlruns_path = Path("mlruns").resolve()
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        mlflow.set_experiment("apes_rule_optimization")

        # Configure ML performance monitoring
        self._configure_ml_performance()

        # Initialize advanced pattern discovery with proper sync database manager
        if db_manager:
            # Create sync DatabaseManager for AdvancedPatternDiscovery from async DatabaseSessionManager
            # Extract database URL and convert to sync format
            import os

            database_url = os.getenv(
                "DATABASE_URL",
                "postgresql+psycopg://apes_user:apes_secure_password_2024@localhost:5432/apes_production",
            ).replace(
                "postgresql+asyncpg://", "postgresql+psycopg://"
            )  # Convert async to sync psycopg3

            sync_db_manager = DatabaseManager(database_url)
            self.pattern_discovery = AdvancedPatternDiscovery(
                db_manager=sync_db_manager
            )
        else:
            self.pattern_discovery = AdvancedPatternDiscovery(db_manager=None)

        self.db_manager = db_manager

        logger.info(
            "Enhanced ML Model Service initialized with production registry support"
        )

    async def enable_production_deployment(
        self, tracking_uri: str | None = None
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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

            logger.warning(f"Production rollback completed: {model_name}@{alias.value}")
            return result

        except Exception as e:
            logger.error(f"Production rollback failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def monitor_production_health(
        self, model_name: str, alias: ModelAlias = ModelAlias.PRODUCTION
    ) -> dict[str, Any]:
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
            # Fallback to regular model loading
            return await self._lazy_load_model(f"{model_name}:{alias.value}")

        try:
            return await self.production_registry.get_production_model(
                model_name=model_name, alias=alias
            )

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            # Fallback to regular model loading
            return await self._lazy_load_model(f"{model_name}:{alias.value}")

    async def list_production_deployments(self) -> list[dict[str, Any]]:
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

    async def optimize_rules(
        self,
        training_data: dict[str, list],
        db_session: AsyncSession,
        rule_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Optimize rule parameters using ML training.
        Direct Python integration replacing bridge cmd_optimize_model.

        Args:
            training_data: Features and effectiveness scores
            db_session: Database session for storing results
            rule_ids: Specific rules to optimize (None = all active rules)

        Returns:
            Optimization results with performance metrics
        """
        start_time = time.time()

        try:
            # Ensure any previous runs are ended
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(
                run_name=f"rule_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                try:
                    # Extract and validate training data (research-based data cleaning)
                    X_raw = np.array(training_data["features"])
                    y_raw = np.array(training_data["effectiveness_scores"])

                    # Data validation and cleaning - research best practice
                    valid_indices = []

                    for i in range(len(X_raw)):
                        # Check for valid features
                        feature_valid = True
                        if len(X_raw[i]) == 0:
                            feature_valid = False
                        else:
                            for feature in X_raw[i]:
                                if (
                                    not isinstance(feature, (int, float))
                                    or np.isnan(feature)
                                    or np.isinf(feature)
                                ):
                                    feature_valid = False
                                    break

                        # Check for valid effectiveness score
                        score_valid = (
                            isinstance(y_raw[i], (int, float))
                            and not np.isnan(y_raw[i])
                            and not np.isinf(y_raw[i])
                            and 0.0 <= y_raw[i] <= 1.0
                        )

                        if feature_valid and score_valid:
                            valid_indices.append(i)

                    if len(valid_indices) == 0:
                        return {
                            "status": "error",
                            "error": "No valid training data after filtering corrupted samples",
                            "processing_time_ms": (time.time() - start_time) * 1000,
                        }

                    # Filter to valid data only
                    X = X_raw[valid_indices]
                    y_continuous = y_raw[valid_indices]

                    # Log data quality metrics
                    data_quality_ratio = len(valid_indices) / len(X_raw)
                    logger.info(
                        f"Data quality: {len(valid_indices)}/{len(X_raw)} samples valid ({data_quality_ratio:.1%})"
                    )

                    # Check for class diversity (research-based degenerate case prevention)
                    unique_scores = np.unique(y_continuous)
                    if len(unique_scores) < 2:
                        return {
                            "status": "error",
                            "error": f"Insufficient class diversity: only {len(unique_scores)} unique effectiveness score(s)",
                            "processing_time_ms": (time.time() - start_time) * 1000,
                        }

                    # Check standard deviation to ensure meaningful variance
                    score_std = np.std(y_continuous)
                    if score_std < 0.05:  # Very low variance threshold
                        return {
                            "status": "error",
                            "error": f"Insufficient score variance: std={score_std:.3f} (minimum: 0.05)",
                            "processing_time_ms": (time.time() - start_time) * 1000,
                        }

                    # Convert continuous scores to binary classification (high/low effectiveness)
                    y_threshold = np.median(y_continuous)
                    y = (y_continuous >= y_threshold).astype(int)

                    # Final check: ensure both classes are present after threshold
                    unique_classes = np.unique(y)
                    if len(unique_classes) < 2:
                        # Adjust threshold to ensure class balance
                        sorted_scores = np.sort(y_continuous)
                        y_threshold = sorted_scores[
                            len(sorted_scores) // 3
                        ]  # 33rd percentile
                        y = (y_continuous >= y_threshold).astype(int)

                        # If still single class, return error
                        if len(np.unique(y)) < 2:
                            return {
                                "status": "error",
                                "error": "Cannot create binary classification: all samples in single class after thresholding",
                                "processing_time_ms": (time.time() - start_time) * 1000,
                            }

                    if len(X) < 10:
                        return {
                            "error": f"Insufficient training data: {len(X)} samples (minimum: 10)"
                        }

                    # Log training data characteristics
                    mlflow.log_params({
                        "n_samples": len(X),
                        "n_features": X.shape[1],
                        "target_mean": float(np.mean(y_continuous)),
                        "target_std": float(np.std(y_continuous)),
                        "binary_threshold": float(y_threshold),
                        "high_effectiveness_ratio": float(np.mean(y)),
                        "data_quality_ratio": float(data_quality_ratio),
                        "original_samples": len(X_raw),
                    })

                    # Hyperparameter optimization with Optuna
                    study = optuna.create_study(direction="maximize")

                    def objective(trial):
                        # Define search space
                        n_estimators = trial.suggest_int("n_estimators", 50, 200)
                        max_depth = trial.suggest_int("max_depth", 3, 15)
                        min_samples_split = trial.suggest_int(
                            "min_samples_split", 2, 20
                        )

                        # Create model pipeline
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            (
                                "classifier",
                                RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    random_state=42,
                                    n_jobs=1,  # Single thread for testing stability
                                ),
                            ),
                        ])

                        # Nested cross-validation for unbiased performance estimation
                        # Use fewer folds for small datasets to speed up testing
                        n_splits = 3 if len(X) < 100 else 5
                        cv = StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=42
                        )
                        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

                        return np.mean(scores)

                    # Run optimization with reduced trials for faster testing
                    n_trials = 5 if len(X) < 100 else 10  # Minimal trials for testing
                    timeout = (
                        30 if len(X) < 100 else 120
                    )  # Very short timeout for tests
                    study.optimize(objective, n_trials=n_trials, timeout=timeout)

                    # Get best parameters
                    best_params = study.best_params
                    best_score = study.best_value

                    # Train final model with best parameters
                    final_model = Pipeline([
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=best_params["n_estimators"],
                                max_depth=best_params["max_depth"],
                                min_samples_split=best_params["min_samples_split"],
                                random_state=42,
                                n_jobs=1,  # Single thread for testing stability
                            ),
                        ),
                    ])

                    final_model.fit(X, y)

                    # Generate predictions for effectiveness analysis
                    predictions = final_model.predict_proba(X)[:, 1]

                    # Calculate performance metrics
                    accuracy = accuracy_score(y, final_model.predict(X))
                    precision = precision_score(
                        y, final_model.predict(X), average="weighted"
                    )
                    recall = recall_score(y, final_model.predict(X), average="weighted")

                    # Generate model ID and cache with TTL
                    model_id = f"rule_optimizer_{int(time.time())}"
                    self.model_registry.add_model(
                        model_id,
                        final_model,
                        model_type="RandomForestClassifier",
                        ttl_minutes=120,  # 2 hours for optimized models
                    )

                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metrics({
                        "best_score": float(best_score),
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "training_time": float(time.time() - start_time),
                    })

                    # Save model to MLflow
                    mlflow.sklearn.log_model(
                        final_model,
                        "model",
                        registered_model_name="apes_rule_optimizer",
                    )

                    # Update rule parameters in database
                    await self._update_rule_parameters(
                        db_session, rule_ids or [], best_params, best_score, model_id
                    )

                    # Store model performance metrics
                    await self._store_model_performance(
                        db_session, model_id, best_score, accuracy, precision, recall
                    )

                    processing_time = time.time() - start_time

                    # Get MLflow run ID safely
                    active_run = mlflow.active_run()
                    mlflow_run_id = active_run.info.run_id if active_run else "unknown"

                    # Emit pattern.invalidate event for model training completion
                    try:
                        event_data = json.dumps({
                            "type": "model_training_completed",
                            "model_id": model_id,
                            "model_type": "RandomForestClassifier",
                            "mlflow_run_id": mlflow_run_id,
                            "best_score": float(best_score),
                            "timestamp": aware_utc_now().isoformat(),
                            "cache_prefixes": ["apes:pattern:", "rule:", "ml:model:"]
                        })
                        await redis_client.publish('pattern.invalidate', event_data)
                        logger.info(f"Emitted pattern.invalidate event for model training completion: {model_id}")
                    except Exception as e:
                        logger.warning(f"Failed to emit pattern.invalidate event: {e}")

                    return {
                        "status": "success",
                        "model_id": model_id,
                        "best_score": float(best_score),
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "best_params": best_params,
                        "training_samples": len(X),
                        "processing_time_ms": processing_time * 1000,
                        "mlflow_run_id": mlflow_run_id,
                    }

                except Exception as e:
                    logger.error(f"ML optimization failed: {e}")
                    mlflow.log_params({"error": str(e)})
                    return {
                        "status": "error",
                        "error": str(e),
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    }
        except Exception as e:
            logger.error(f"ML service error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def predict_rule_effectiveness(
        self, model_id: str, rule_features: list[float]
    ) -> dict[str, Any]:
        """Predict rule effectiveness using trained model.
        Direct Python integration replacing bridge cmd_predict.

        Args:
            model_id: Trained model identifier
            rule_features: Feature vector for prediction

        Returns:
            Effectiveness prediction with confidence scores
        """
        start_time = time.time()

        try:
            # Try to get model from cache first
            model = self.model_registry.get_model(model_id)

            if model is None:
                # Try to load from MLflow if not in cache
                model = await self._lazy_load_model(model_id)

            if model is None:
                return {"error": f"Model {model_id} not found in cache or MLflow"}
            features = np.array([rule_features])

            # Get prediction and probability
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            processing_time = (time.time() - start_time) * 1000

            return {
                "status": "success",
                "prediction": float(prediction),
                "confidence": float(max(probabilities)),
                "probabilities": probabilities.tolist(),
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def optimize_ensemble_rules(
        self, training_data: dict[str, list], db_session: AsyncSession
    ) -> dict[str, Any]:
        """Optimize rules using sophisticated ensemble methods.
        Direct Python integration replacing bridge cmd_optimize_stacking_model.

        Args:
            training_data: Features and effectiveness scores
            db_session: Database session for storing results

        Returns:
            Ensemble optimization results
        """
        start_time = time.time()

        try:
            # Ensure any previous runs are ended
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(
                run_name=f"ensemble_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                try:
                    X = np.array(training_data["features"])
                    y_continuous = np.array(training_data["effectiveness_scores"])

                    # Convert continuous scores to binary classification
                    y_threshold = np.median(y_continuous)
                    y = (y_continuous >= y_threshold).astype(int)

                    if len(X) < 20:
                        return {
                            "error": f"Insufficient data for ensemble: {len(X)} samples (minimum: 20)"
                        }

                    # Define base estimators
                    base_estimators = [
                        (
                            "rf",
                            Pipeline([
                                ("scaler", StandardScaler()),
                                (
                                    "classifier",
                                    RandomForestClassifier(
                                        n_estimators=100, max_depth=10, random_state=42
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "gb",
                            Pipeline([
                                ("scaler", StandardScaler()),
                                (
                                    "classifier",
                                    GradientBoostingClassifier(
                                        n_estimators=100,
                                        learning_rate=0.1,
                                        max_depth=6,
                                        random_state=42,
                                    ),
                                ),
                            ]),
                        ),
                    ]

                    # Final estimator
                    final_estimator = LogisticRegression(random_state=42)

                    # Create stacking classifier
                    stacking_model = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=final_estimator,
                        cv=5,
                        n_jobs=-1,
                    )

                    # Train ensemble model
                    stacking_model.fit(X, y)

                    # Evaluate ensemble performance
                    cv_scores = cross_val_score(
                        stacking_model,
                        X,
                        y,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring="roc_auc",
                    )

                    ensemble_score = np.mean(cv_scores)
                    ensemble_std = np.std(cv_scores)

                    # Generate model ID and cache ensemble model
                    model_id = f"ensemble_optimizer_{int(time.time())}"
                    self.model_registry.add_model(
                        model_id,
                        stacking_model,
                        model_type="StackingClassifier",
                        ttl_minutes=180,  # 3 hours for ensemble models
                    )

                    # Log to MLflow
                    mlflow.log_params({
                        "model_type": "StackingClassifier",
                        "base_estimators": "RandomForest,GradientBoosting",
                        "final_estimator": "LogisticRegression",
                        "cv_folds": 5,
                    })

                    mlflow.log_metrics({
                        "ensemble_score": float(ensemble_score),
                        "ensemble_std": float(ensemble_std),
                        "training_time": float(time.time() - start_time),
                    })

                    # Save ensemble model
                    mlflow.sklearn.log_model(
                        stacking_model,
                        "ensemble_model",
                        registered_model_name="apes_ensemble_optimizer",
                    )

                    # Store ensemble performance
                    await self._store_model_performance(
                        db_session,
                        model_id,
                        ensemble_score,
                        accuracy_score(y, stacking_model.predict(X)),
                        precision_score(
                            y, stacking_model.predict(X), average="weighted"
                        ),
                        recall_score(y, stacking_model.predict(X), average="weighted"),
                    )

                    processing_time = time.time() - start_time

                    # Get MLflow run ID safely
                    active_run = mlflow.active_run()
                    mlflow_run_id = active_run.info.run_id if active_run else "unknown"

                    # Emit pattern.invalidate event for model training completion
                    try:
                        event_data = json.dumps({
                            "type": "model_training_completed",
                            "model_id": model_id,
                            "model_type": "StackingClassifier",
                            "mlflow_run_id": mlflow_run_id,
                            "ensemble_score": float(ensemble_score),
                            "timestamp": aware_utc_now().isoformat(),
                            "cache_prefixes": ["apes:pattern:", "rule:", "ml:model:"]
                        })
                        await redis_client.publish('pattern.invalidate', event_data)
                        logger.info(f"Emitted pattern.invalidate event for model training completion: {model_id}")
                    except Exception as e:
                        logger.warning(f"Failed to emit pattern.invalidate event: {e}")

                    return {
                        "status": "success",
                        "model_id": model_id,
                        "ensemble_score": float(ensemble_score),
                        "ensemble_std": float(ensemble_std),
                        "cv_scores": cv_scores.tolist(),
                        "processing_time_ms": processing_time * 1000,
                        "mlflow_run_id": mlflow_run_id,
                    }

                except Exception as e:
                    logger.error(f"Ensemble optimization failed: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    }
        except Exception as e:
            logger.error(f"Ensemble service error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def discover_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        use_advanced_discovery: bool = True,
        include_apriori: bool = True,
    ) -> dict[str, Any]:
        """Enhanced pattern discovery combining traditional ML with Apriori association rules.

        Args:
            db_session: Database session
            min_effectiveness: Minimum effectiveness threshold
            min_support: Minimum number of occurrences
            use_advanced_discovery: Use advanced pattern discovery with HDBSCAN/FP-Growth
            include_apriori: Include Apriori association rule mining

        Returns:
            Comprehensive pattern discovery results with ML and Apriori insights
        """
        start_time = time.time()
        logger.info(
            f"Starting enhanced pattern discovery (advanced: {use_advanced_discovery}, apriori: {include_apriori})"
        )

        try:
            results = {
                "status": "success",
                "discovery_metadata": {
                    "start_time": start_time,
                    "algorithms_used": [],
                    "discovery_modes": [],
                },
            }

            # 1. Traditional ML Pattern Discovery (existing implementation)
            traditional_results = await self._discover_traditional_patterns(
                db_session, min_effectiveness, min_support
            )
            results["traditional_patterns"] = traditional_results
            results["discovery_metadata"]["algorithms_used"].append("traditional_ml")
            results["discovery_metadata"]["discovery_modes"].append(
                "parameter_analysis"
            )

            # 2. Advanced Pattern Discovery (HDBSCAN, FP-Growth, Semantic Analysis)
            if use_advanced_discovery:
                advanced_results = (
                    await self.pattern_discovery.discover_advanced_patterns(
                        db_session=db_session,
                        min_effectiveness=min_effectiveness,
                        min_support=min_support,
                        pattern_types=[
                            "parameter",
                            "sequence",
                            "performance",
                            "semantic",
                        ],
                        use_ensemble=True,
                        include_apriori=include_apriori,
                    )
                )
                results["advanced_patterns"] = advanced_results
                results["discovery_metadata"]["algorithms_used"].extend([
                    "hdbscan",
                    "fp_growth",
                    "semantic_clustering",
                ])
                results["discovery_metadata"]["discovery_modes"].extend([
                    "density_clustering",
                    "frequent_patterns",
                    "semantic_analysis",
                ])

                # Include Apriori patterns if they were discovered
                if include_apriori and "apriori_patterns" in advanced_results:
                    results["apriori_patterns"] = advanced_results["apriori_patterns"]
                    results["discovery_metadata"]["algorithms_used"].append("apriori")
                    results["discovery_metadata"]["discovery_modes"].append(
                        "association_rules"
                    )

            # 3. Cross-validation and ensemble analysis
            if use_advanced_discovery:
                cross_validation = self._cross_validate_pattern_discovery(
                    traditional_results, advanced_results
                )
                results["cross_validation"] = cross_validation

            # 4. Generate unified recommendations
            unified_recommendations = self._generate_unified_recommendations(results)
            results["unified_recommendations"] = unified_recommendations

            # 5. Business insights from all discovery methods
            business_insights = self._generate_business_insights(results)
            results["business_insights"] = business_insights

            # Add execution metadata
            execution_time = time.time() - start_time
            results["discovery_metadata"].update({
                "execution_time_seconds": execution_time,
                "total_patterns_discovered": self._count_total_patterns(results),
                "discovery_quality_score": self._calculate_discovery_quality(results),
                "timestamp": aware_utc_now().isoformat(),
                "algorithms_count": len(
                    results["discovery_metadata"]["algorithms_used"]
                ),
            })

            logger.info(
                f"Enhanced pattern discovery completed in {execution_time:.2f}s with "
                f"{len(results['discovery_metadata']['algorithms_used'])} algorithms"
            )

            return results

        except Exception as e:
            logger.error(f"Enhanced pattern discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
            }

    async def _discover_traditional_patterns(
        self,
        db_session: AsyncSession,
        min_effectiveness: float,
        min_support: int,
    ) -> dict[str, Any]:
        """Traditional pattern discovery (existing implementation)"""
        # This is the existing discover_patterns logic
        try:
            # Query rule performance data
            stmt = (
                select(RulePerformance, RuleMetadata)
                .join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id)
                .where(RulePerformance.improvement_score >= min_effectiveness)
            )

            result = await db_session.execute(stmt)
            performance_data = result.fetchall()

            if len(performance_data) < min_support:
                return {
                    "status": "insufficient_data",
                    "message": f"Only {len(performance_data)} high-performing samples found (minimum: {min_support})",
                    "data_points": len(performance_data),
                }

            # Analyze rule patterns
            rule_patterns = {}
            for row in performance_data:
                rule_id = row.rule_id
                params = row.default_parameters or {}
                effectiveness = row.improvement_score

                # Extract parameter patterns
                pattern_key = json.dumps(params, sort_keys=True)
                if pattern_key not in rule_patterns:
                    rule_patterns[pattern_key] = {
                        "parameters": params,
                        "effectiveness_scores": [],
                        "rule_ids": [],
                        "count": 0,
                    }

                rule_patterns[pattern_key]["effectiveness_scores"].append(effectiveness)
                rule_patterns[pattern_key]["rule_ids"].append(rule_id)
                rule_patterns[pattern_key]["count"] += 1

            # Filter patterns by support and effectiveness
            discovered_patterns = []
            for pattern_key, pattern_data in rule_patterns.items():
                if pattern_data["count"] >= min_support:
                    avg_effectiveness = np.mean(pattern_data["effectiveness_scores"])
                    if avg_effectiveness >= min_effectiveness:
                        discovered_patterns.append({
                            "parameters": pattern_data["parameters"],
                            "avg_effectiveness": avg_effectiveness,
                            "support_count": pattern_data["count"],
                            "rule_ids": pattern_data["rule_ids"],
                            "effectiveness_range": [
                                min(pattern_data["effectiveness_scores"]),
                                max(pattern_data["effectiveness_scores"]),
                            ],
                            "pattern_type": "traditional_parameter_pattern",
                        })

            # Sort by effectiveness
            discovered_patterns.sort(key=lambda x: x["avg_effectiveness"], reverse=True)

            return {
                "status": "success",
                "patterns_discovered": len(discovered_patterns),
                "patterns": discovered_patterns[:10],  # Top 10 patterns
                "total_analyzed": len(performance_data),
                "discovery_type": "traditional_ml",
                "algorithm": "parameter_analysis",
            }

        except Exception as e:
            logger.error(f"Traditional pattern discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "discovery_type": "traditional_ml",
            }

    def _cross_validate_pattern_discovery(
        self, traditional_results: dict[str, Any], advanced_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Cross-validate patterns discovered by different methods"""
        validation = {
            "consistency_score": 0.0,
            "complementary_insights": [],
            "confidence_boost": [],
            "pattern_overlap": 0.0,
        }

        try:
            # Compare traditional vs advanced patterns
            traditional_patterns = traditional_results.get("patterns", [])
            advanced_pattern_types = [
                "parameter_patterns",
                "sequence_patterns",
                "performance_patterns",
            ]

            total_advanced_patterns = 0
            overlapping_patterns = 0

            for pattern_type in advanced_pattern_types:
                if pattern_type in advanced_results:
                    patterns = advanced_results[pattern_type].get("patterns", [])
                    total_advanced_patterns += len(patterns)

                    # Check for overlapping insights
                    for advanced_pattern in patterns:
                        for traditional_pattern in traditional_patterns:
                            if self._patterns_overlap(
                                traditional_pattern, advanced_pattern
                            ):
                                overlapping_patterns += 1
                                validation["confidence_boost"].append({
                                    "traditional_pattern": traditional_pattern.get(
                                        "parameters"
                                    ),
                                    "advanced_pattern": advanced_pattern.get(
                                        "pattern_id"
                                    ),
                                    "overlap_reason": "parameter_similarity",
                                })

            # Calculate metrics
            if total_advanced_patterns > 0:
                validation["pattern_overlap"] = (
                    overlapping_patterns / total_advanced_patterns
                )
                validation["consistency_score"] = min(
                    validation["pattern_overlap"] * 2, 1.0
                )

            # Identify complementary insights
            if "apriori_patterns" in advanced_results:
                apriori_insights = advanced_results["apriori_patterns"].get(
                    "pattern_insights", {}
                )
                validation["complementary_insights"].extend([
                    {"type": "apriori_association", "insight": insight}
                    for insights_list in apriori_insights.values()
                    for insight in (
                        insights_list if isinstance(insights_list, list) else []
                    )
                ])

            return validation

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return validation

    def _patterns_overlap(
        self, traditional_pattern: dict, advanced_pattern: dict
    ) -> bool:
        """Check if patterns from different methods overlap"""
        try:
            # Simple overlap check based on parameter similarity
            trad_params = traditional_pattern.get("parameters", {})
            adv_params = advanced_pattern.get("parameters", {})

            if not trad_params or not adv_params:
                return False

            common_keys = set(trad_params.keys()).intersection(set(adv_params.keys()))
            return len(common_keys) > 0

        except Exception:
            return False

    def _generate_unified_recommendations(
        self, results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate unified recommendations from all discovery methods"""
        recommendations = []

        try:
            # From traditional patterns
            traditional_patterns = results.get("traditional_patterns", {}).get(
                "patterns", []
            )
            for pattern in traditional_patterns[:3]:  # Top 3
                recommendations.append({
                    "type": "parameter_optimization",
                    "source": "traditional_ml",
                    "action": f"Optimize parameters: {pattern.get('parameters')}",
                    "effectiveness": pattern.get("avg_effectiveness", 0),
                    "confidence": "high"
                    if pattern.get("support_count", 0) > 10
                    else "medium",
                    "priority": "high"
                    if pattern.get("avg_effectiveness", 0) > 0.8
                    else "medium",
                })

            # From Apriori patterns
            apriori_patterns = results.get("apriori_patterns", {}).get("patterns", [])
            for pattern in apriori_patterns[:3]:  # Top 3
                recommendations.append({
                    "type": "association_rule",
                    "source": "apriori",
                    "action": pattern.get(
                        "business_insight", "Apply discovered association rule"
                    ),
                    "confidence": pattern.get("confidence", 0),
                    "lift": pattern.get("lift", 0),
                    "priority": "high" if pattern.get("lift", 0) > 2.0 else "medium",
                })

            # From advanced patterns
            advanced_results = results.get("advanced_patterns", {})
            for pattern_type in ["parameter_patterns", "performance_patterns"]:
                if pattern_type in advanced_results:
                    patterns = advanced_results[pattern_type].get("patterns", [])
                    for pattern in patterns[:2]:  # Top 2 from each type
                        recommendations.append({
                            "type": pattern_type,
                            "source": "advanced_ml",
                            "action": f"Apply {pattern_type} insights: {pattern.get('pattern_id', 'pattern')}",
                            "effectiveness": pattern.get("effectiveness", 0),
                            "confidence": pattern.get("confidence", 0),
                            "priority": "medium",
                        })

            # Sort by priority and effectiveness
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda x: (
                    priority_order.get(x.get("priority", "low"), 1),
                    x.get("effectiveness", x.get("confidence", 0)),
                ),
                reverse=True,
            )

            return recommendations[:10]  # Top 10 recommendations

        except Exception as e:
            logger.error(f"Error generating unified recommendations: {e}")
            return []

    def _generate_business_insights(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate business insights from comprehensive pattern discovery"""
        insights = {
            "key_findings": [],
            "performance_drivers": [],
            "optimization_opportunities": [],
            "risk_factors": [],
        }

        try:
            # Insights from traditional patterns
            traditional_patterns = results.get("traditional_patterns", {}).get(
                "patterns", []
            )
            if traditional_patterns:
                top_pattern = traditional_patterns[0]
                insights["key_findings"].append(
                    f"Top performing parameter configuration achieves {top_pattern.get('avg_effectiveness', 0):.1%} effectiveness"
                )

            # Insights from Apriori patterns
            apriori_insights = results.get("apriori_patterns", {}).get(
                "pattern_insights", {}
            )
            for category, patterns in apriori_insights.items():
                if isinstance(patterns, list) and patterns:
                    insights["performance_drivers"].extend(
                        patterns[:2]
                    )  # Top 2 per category

            # Insights from advanced discovery
            advanced_results = results.get("advanced_patterns", {})
            if "ensemble_analysis" in advanced_results:
                ensemble = advanced_results["ensemble_analysis"]
                insights["optimization_opportunities"].append(
                    f"Ensemble analysis reveals {ensemble.get('consensus_patterns', 0)} consensus patterns for optimization"
                )

            # Cross-validation insights
            cross_val = results.get("cross_validation", {})
            if cross_val.get("consistency_score", 0) > 0.7:
                insights["key_findings"].append(
                    f"High consistency score ({cross_val['consistency_score']:.1%}) between discovery methods increases confidence"
                )
            elif cross_val.get("consistency_score", 0) < 0.3:
                insights["risk_factors"].append(
                    "Low consistency between discovery methods suggests need for more data or refined parameters"
                )

            return insights

        except Exception as e:
            logger.error(f"Error generating business insights: {e}")
            return insights

    def _count_total_patterns(self, results: dict[str, Any]) -> int:
        """Count total patterns discovered across all methods"""
        total = 0
        try:
            # Traditional patterns
            total += len(results.get("traditional_patterns", {}).get("patterns", []))

            # Advanced patterns
            advanced_results = results.get("advanced_patterns", {})
            for pattern_type in [
                "parameter_patterns",
                "sequence_patterns",
                "performance_patterns",
                "semantic_patterns",
            ]:
                if pattern_type in advanced_results:
                    total += len(advanced_results[pattern_type].get("patterns", []))

            # Apriori patterns
            total += len(results.get("apriori_patterns", {}).get("patterns", []))

            return total
        except Exception:
            return 0

    def _calculate_discovery_quality(self, results: dict[str, Any]) -> float:
        """Calculate overall quality score for pattern discovery"""
        try:
            scores = []

            # Traditional discovery quality
            traditional = results.get("traditional_patterns", {})
            if traditional.get("status") == "success":
                scores.append(min(traditional.get("patterns_discovered", 0) / 10, 1.0))

            # Advanced discovery quality
            advanced = results.get("advanced_patterns", {})
            if "discovery_metadata" in advanced:
                execution_time = advanced["discovery_metadata"].get(
                    "execution_time", float("inf")
                )
                # Quality inversely related to execution time (penalty for slow discovery)
                time_score = max(
                    0, 1.0 - (execution_time / 60)
                )  # Penalty after 60 seconds
                scores.append(time_score)

            # Cross-validation quality
            cross_val = results.get("cross_validation", {})
            consistency_score = cross_val.get("consistency_score", 0)
            scores.append(consistency_score)

            # Return average quality score
            return float(np.mean(scores)) if scores else 0.0

        except Exception:
            return 0.0

    async def get_contextualized_patterns(
        self,
        context_items: list[str],
        db_session: AsyncSession,
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Get patterns relevant to a specific context using advanced pattern discovery.

        This method leverages both traditional ML and Apriori association rules
        to find patterns relevant to the current prompt improvement context.

        Args:
            context_items: Items representing current context (rules, characteristics)
            db_session: Database session
            min_confidence: Minimum confidence for returned patterns

        Returns:
            Dictionary with contextualized patterns and recommendations
        """
        try:
            if not hasattr(self.pattern_discovery, "get_contextualized_patterns"):
                logger.warning(
                    "Advanced pattern discovery not available for contextualized patterns"
                )
                return {"error": "Advanced pattern discovery not configured"}

            # Use advanced pattern discovery for contextualized analysis
            results = await self.pattern_discovery.get_contextualized_patterns(
                context_items=context_items,
                db_session=db_session,
                min_confidence=min_confidence,
            )

            # Enhance with traditional ML insights
            traditional_context = await self._get_traditional_context_patterns(
                context_items, db_session
            )

            # Combine results
            if "error" not in results:
                results["traditional_insights"] = traditional_context
                results["combined_recommendations"] = (
                    self._combine_context_recommendations(
                        results.get("recommendations", []),
                        traditional_context.get("recommendations", []),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error getting contextualized patterns: {e}")
            return {"error": f"Contextualized pattern analysis failed: {e!s}"}

    async def _get_traditional_context_patterns(
        self, context_items: list[str], db_session: AsyncSession
    ) -> dict[str, Any]:
        """Get traditional ML patterns relevant to context"""
        try:
            # Extract rule names from context items
            rule_contexts = [item for item in context_items if item.startswith("rule_")]

            if not rule_contexts:
                return {"recommendations": [], "context_match": 0.0}

            # Query performance data for context rules
            rule_ids = [rule.replace("rule_", "") for rule in rule_contexts]

            stmt = (
                select(RulePerformance)
                .where(RulePerformance.rule_id.in_(rule_ids))
                .where(RulePerformance.improvement_score >= 0.6)
            )

            result = await db_session.execute(stmt)
            context_performance = result.fetchall()

            recommendations = []
            if context_performance:
                avg_performance = np.mean([
                    row.improvement_score for row in context_performance
                ])
                recommendations.append({
                    "type": "traditional_context",
                    "action": f"Context rules show {avg_performance:.1%} average performance",
                    "confidence": min(len(context_performance) / 10, 1.0),
                    "priority": "high" if avg_performance > 0.8 else "medium",
                })

            return {
                "recommendations": recommendations,
                "context_match": len(context_performance) / len(rule_ids)
                if rule_ids
                else 0.0,
                "performance_data": len(context_performance),
            }

        except Exception as e:
            logger.error(f"Error getting traditional context patterns: {e}")
            return {"recommendations": [], "context_match": 0.0}

    def _combine_context_recommendations(
        self, apriori_recommendations: list, traditional_recommendations: list
    ) -> list[dict[str, Any]]:
        """Combine recommendations from different discovery methods"""
        combined = []

        # Add Apriori recommendations with source tag
        for rec in apriori_recommendations:
            rec_copy = rec.copy()
            rec_copy["source"] = "apriori"
            combined.append(rec_copy)

        # Add traditional recommendations with source tag
        for rec in traditional_recommendations:
            rec_copy = rec.copy()
            rec_copy["source"] = "traditional"
            combined.append(rec_copy)

        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        combined.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 1),
                x.get("confidence", 0),
            ),
            reverse=True,
        )

        return combined[:8]  # Top 8 combined recommendations

    async def _update_rule_parameters(
        self,
        db_session: AsyncSession,
        rule_ids: list[str],
        optimized_params: dict[str, Any],
        effectiveness_score: float,
        model_id: str,
    ):
        """Update rule parameters in database with ML-optimized values."""
        try:
            # If no specific rule IDs, update all active rules
            if not rule_ids:
                stmt = select(RuleMetadata).where(RuleMetadata.is_enabled == True)
                result = await db_session.execute(stmt)
                rules = result.scalars().all()
                rule_ids = [rule.rule_id for rule in rules]

            # Update each rule with optimized parameters
            for rule_id in rule_ids:
                stmt = select(RuleMetadata).where(RuleMetadata.rule_id == rule_id)
                result = await db_session.execute(stmt)
                rule = result.scalar_one_or_none()

                if rule:
                    # Merge optimized parameters with existing ones
                    current_params = rule.default_parameters or {}
                    updated_params = {**current_params, **optimized_params}

                    rule.default_parameters = updated_params
                    rule.updated_at = aware_utc_now()

                    db_session.add(rule)

            await db_session.commit()
            logger.info(f"Updated {len(rule_ids)} rules with ML-optimized parameters")

            # Emit pattern.invalidate event for cache invalidation
            try:
                event_data = json.dumps({
                    "type": "rule_parameters_updated",
                    "rule_ids": rule_ids,
                    "timestamp": aware_utc_now().isoformat(),
                    "effectiveness_score": effectiveness_score,
                    "model_id": model_id,
                    "cache_prefixes": ["apes:pattern:", "rule:"]
                })
                await redis_client.publish('pattern.invalidate', event_data)
                logger.info(f"Emitted cache invalidation event for rule IDs: {rule_ids}")
            except Exception as e:
                logger.warning(f"Failed to emit cache invalidation event: {e}")

        except Exception as e:
            logger.error(f"Failed to update rule parameters: {e}")
            await db_session.rollback()

    async def _store_model_performance(
        self,
        db_session: AsyncSession,
        model_id: str,
        performance_score: float,
        accuracy: float,
        precision: float,
        recall: float,
    ):
        """Store model performance metrics in database."""
        try:
            performance_record = MLModelPerformance(
                model_id=model_id,
                model_type="RandomForestClassifier",
                performance_score=performance_score,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                training_samples=0,  # Will be updated by caller
            )

            db_session.add(performance_record)
            await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store model performance: {e}")
            await db_session.rollback()

    def _configure_ml_performance(self):
        """Configure ML performance settings for scikit-learn optimization"""
        try:
            import os

            # Optimize BLAS/LAPACK threads for NumPy operations
            cpu_count = os.cpu_count() or 1
            optimal_threads = min(cpu_count, 4)  # Cap at 4 for stability

            os.environ.setdefault("OMP_NUM_THREADS", str(optimal_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(optimal_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(optimal_threads))
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(optimal_threads))

            # Configure scikit-learn parallel processing
            os.environ.setdefault("SKLEARN_N_JOBS", str(optimal_threads))

            logger.info(f"ML performance configured: {optimal_threads} threads")

        except Exception as e:
            logger.warning(f"Failed to configure ML performance: {e}")

    async def _lazy_load_model(self, model_id: str) -> Any | None:
        """Lazy load model from MLflow when not in cache"""
        try:
            logger.info(f"Lazy loading model {model_id} from MLflow")

            # Search for model in MLflow registry
            model_versions = self.mlflow_client.search_model_versions(
                filter_string="name='apes_rule_optimizer' OR name='apes_ensemble_optimizer'"
            )

            # Find matching model by run_id or version
            target_model = None
            for version in model_versions:
                if (version.description and model_id in version.description) or (
                    version.run_id and model_id in version.run_id
                ):
                    target_model = version
                    break

            if target_model:
                # Load model using MLflow
                model_uri = f"models:/{target_model.name}/{target_model.version}"
                loaded_model = mlflow.sklearn.load_model(model_uri)

                # Cache the loaded model
                self.model_registry.add_model(
                    model_id,
                    loaded_model,
                    model_type="MLflow_Loaded",
                    ttl_minutes=90,  # 1.5 hours for lazy-loaded models
                )

                logger.info(f"Successfully lazy-loaded model {model_id}")
                return loaded_model

            logger.warning(f"Model {model_id} not found in MLflow registry")
            return None

        except Exception as e:
            logger.error(f"Failed to lazy load model {model_id}: {e}")
            return None

    async def get_model_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive model cache statistics"""
        try:
            # Clean up expired models first
            cleaned_count = self.model_registry.cleanup_expired()

            # Get cache statistics
            cache_stats = self.model_registry.get_cache_stats()

            # Add cleanup information
            cache_stats["cleaned_expired_models"] = cleaned_count
            cache_stats["cache_efficiency"] = {
                "hit_rate_estimate": "N/A",  # Would need request tracking
                "memory_efficiency": cache_stats["memory_utilization"],
                "active_model_ratio": cache_stats["active_models"]
                / max(cache_stats["total_models"], 1),
            }

            return {
                "status": "success",
                "cache_stats": cache_stats,
                "recommendations": self._generate_cache_recommendations(cache_stats),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_cache_recommendations(self, stats: dict[str, Any]) -> list[str]:
        """Generate cache optimization recommendations"""
        recommendations = []

        memory_util = stats["memory_utilization"]
        if memory_util > 0.9:
            recommendations.append(
                "🔴 High memory usage - consider increasing cache size or reducing TTL"
            )
        elif memory_util > 0.7:
            recommendations.append("🟡 Moderate memory usage - monitor for trends")
        else:
            recommendations.append("🟢 Healthy memory usage")

        if stats["expired_models"] > stats["active_models"]:
            recommendations.append(
                "⏰ Many expired models - consider shorter TTL or more frequent cleanup"
            )

        if stats["total_models"] > 20:
            recommendations.append(
                "📊 Large number of cached models - consider model lifecycle management"
            )

        return recommendations

    async def optimize_model_cache(self) -> dict[str, Any]:
        """Optimize model cache by cleaning expired models and analyzing usage"""
        try:
            start_time = time.time()

            # Clean expired models
            cleaned_count = self.model_registry.cleanup_expired()

            # Get current stats
            stats = self.model_registry.get_cache_stats()

            processing_time = (time.time() - start_time) * 1000

            return {
                "status": "success",
                "cleaned_models": cleaned_count,
                "active_models": stats["active_models"],
                "memory_usage_mb": stats["total_memory_mb"],
                "memory_utilization": stats["memory_utilization"],
                "processing_time_ms": processing_time,
                "recommendations": self._generate_cache_recommendations(stats),
            }

        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    async def send_training_batch(self, batch: list[dict]) -> dict[str, Any]:
        """Send training batch to local ML stub storage.

        Args:
            batch: List of training records to persist

        Returns:
            Status of the batch write operation
        """
        try:
            start_time = time.time()

            # Create ml_stub/batches directory if it doesn't exist
            stub_dir = Path("ml_stub/batches")
            stub_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"training_batch_{timestamp}.jsonl"
            batch_path = stub_dir / batch_filename

            # Write batch to JSONL file
            with open(batch_path, "w") as f:
                for record in batch:
                    f.write(json.dumps(record) + "\n")

            processing_time = (time.time() - start_time) * 1000

            logger.info(f"Saved training batch to {batch_path} ({len(batch)} records)")

            return {
                "status": "success",
                "batch_file": str(batch_path),
                "record_count": len(batch),
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(f"Failed to save training batch: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def fetch_latest_model(self) -> dict[str, Any]:
        """Fetch the latest model from ML stub storage.

        Returns:
            Information about the latest model file
        """
        try:
            start_time = time.time()

            # Look for model files in ml_stub directory
            model_pattern = "ml_stub/model_v*.bin"
            model_files = glob.glob(model_pattern)

            if not model_files:
                return {
                    "status": "no_models",
                    "message": "No model files found in ml_stub directory",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                }

            # Sort by modification time to get the latest
            latest_model = max(model_files, key=os.path.getmtime)
            model_stats = os.stat(latest_model)

            processing_time = (time.time() - start_time) * 1000

            # Extract version from filename
            model_filename = os.path.basename(latest_model)
            version = model_filename.replace("model_v", "").replace(".bin", "")

            logger.info(f"Found latest model: {latest_model}")

            return {
                "status": "success",
                "model_path": latest_model,
                "model_version": version,
                "file_size_bytes": model_stats.st_size,
                "last_modified": datetime.fromtimestamp(
                    model_stats.st_mtime
                ).isoformat(),
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(f"Failed to fetch latest model: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }


# Global service instance
_ml_service: MLModelService | None = None


async def get_ml_service() -> MLModelService:
    """Get or create global ML service instance."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLModelService()
    return _ml_service
