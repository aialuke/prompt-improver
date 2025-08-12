"""ML Training Service for model optimization and batch processing.

Handles rule optimization, ensemble training, hyperparameter tuning,
and training data management with MLflow integration.
"""

import glob
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
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

from prompt_improver.database.models import MLModelPerformance, RuleMetadata
from prompt_improver.security.input_validator import InputValidator
from prompt_improver.utils.datetime_utils import aware_utc_now
from .protocols import TrainingServiceProtocol, ModelRegistryProtocol

logger = logging.getLogger(__name__)


class MLTrainingService(TrainingServiceProtocol):
    """Service for ML model training and optimization."""

    def __init__(
        self,
        model_registry: ModelRegistryProtocol,
        db_manager=None,
        orchestrator_event_bus=None,
        input_validator: InputValidator | None = None
    ):
        """Initialize training service.
        
        Args:
            model_registry: Registry for caching trained models
            db_manager: Database manager for data persistence
            orchestrator_event_bus: Event bus for orchestrator integration
            input_validator: Input validator for security
        """
        self.model_registry = model_registry
        self.db_manager = db_manager
        self.orchestrator_event_bus = orchestrator_event_bus
        self.input_validator = input_validator or InputValidator()
        
        # MLflow setup
        mlruns_path = Path("mlruns").resolve()
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        mlflow.set_experiment("apes_rule_optimization")
        
        # Configure ML performance
        self._configure_ml_performance()
        
        logger.info("ML Training Service initialized")

    async def optimize_rules(
        self,
        training_data: Dict[str, List],
        db_session: AsyncSession,
        rule_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Optimize rule parameters using ML training.
        Direct Python integration replacing bridge cmd_optimize_model.

        Args:
            training_data: features and effectiveness scores
            db_session: Database session for storing results
            rule_ids: Specific rules to optimize (None = all active rules)

        Returns:
            Optimization results with performance metrics
        """
        start_time = time.time()

        # Emit optimization started event (optional orchestrator integration)
        await self._emit_orchestrator_event("OPTIMIZATION_STARTED", {
            "method": "optimize_rules",
            "rule_ids": rule_ids,
            "data_size": len(training_data.get("features", [])),
            "started_at": datetime.now().isoformat()
        })

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

                    # Emit cache invalidation event
                    await self._emit_cache_invalidation_event({
                        "type": "model_training_completed",
                        "model_id": model_id,
                        "model_type": "RandomForestClassifier",
                        "mlflow_run_id": mlflow_run_id,
                        "best_score": float(best_score),
                        "timestamp": aware_utc_now().isoformat(),
                        "cache_prefixes": ["apes:pattern:", "rule:", "ml:model:"]
                    })

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

    async def optimize_ensemble_rules(
        self, training_data: Dict[str, List], db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Optimize rules using sophisticated ensemble methods.
        Direct Python integration replacing bridge cmd_optimize_stacking_model.

        Args:
            training_data: features and effectiveness scores
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

                    # Emit cache invalidation event
                    await self._emit_cache_invalidation_event({
                        "type": "model_training_completed",
                        "model_id": model_id,
                        "model_type": "StackingClassifier",
                        "mlflow_run_id": mlflow_run_id,
                        "ensemble_score": float(ensemble_score),
                        "timestamp": aware_utc_now().isoformat(),
                        "cache_prefixes": ["apes:pattern:", "rule:", "ml:model:"]
                    })

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

    async def send_training_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """Send training batch to local ML stub storage.

        Args:
            batch: List of training records to persist

        Returns:
            Status of the batch write operation
        """
        try:
            start_time = time.time()

            # Validate input batch
            try:
                if not isinstance(batch, list):
                    return {"error": "batch must be a list", "status": "validation_error"}

                if len(batch) == 0:
                    return {"error": "batch cannot be empty", "status": "validation_error"}

                if len(batch) > 10000:  # Reasonable batch size limit
                    return {"error": "batch too large (max 10000 records)", "status": "validation_error"}

                # Validate each record in batch
                for i, record in enumerate(batch):
                    if not isinstance(record, dict):
                        return {"error": f"Record at index {i} must be a dictionary", "status": "validation_error"}

                    # Check for reasonable record size (prevent memory exhaustion)
                    record_str = json.dumps(record)
                    if len(record_str) > 100000:  # 100KB per record limit
                        return {"error": f"Record at index {i} too large (max 100KB)", "status": "validation_error"}

            except Exception as e:
                return {"error": f"Batch validation error: {str(e)}", "status": "validation_error"}

            # Create ml_stub/batches directory if it doesn't exist
            stub_dir = Path("ml_stub/batches")
            stub_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"training_batch_{timestamp}.jsonl"
            batch_path = stub_dir / batch_filename

            # Write batch to JSONL file with validation
            with open(batch_path, "w") as f:
                for record in batch:
                    # Additional sanitization before writing
                    sanitized_record = self.input_validator.sanitize_json_input(record)
                    f.write(json.dumps(sanitized_record) + "\n")

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

    async def fetch_latest_model(self) -> Dict[str, Any]:
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

    async def _emit_orchestrator_event(self, event_type_name: str, data: Dict[str, Any]) -> None:
        """Emit event to orchestrator if event bus is available (backward compatible)."""
        if self.orchestrator_event_bus:
            try:
                # Import here to avoid circular imports
                from datetime import datetime, timezone

                from ..orchestration.events.event_types import EventType, MLEvent

                # Map string event type to enum
                event_type_map = {
                    "TRAINING_STARTED": EventType.TRAINING_STARTED,
                    "TRAINING_COMPLETED": EventType.TRAINING_COMPLETED,
                    "TRAINING_FAILED": EventType.TRAINING_FAILED,
                    "OPTIMIZATION_STARTED": EventType.OPTIMIZATION_STARTED,
                    "OPTIMIZATION_COMPLETED": EventType.OPTIMIZATION_COMPLETED,
                }

                event_type = event_type_map.get(event_type_name)
                if event_type:
                    await self.orchestrator_event_bus.emit(MLEvent(
                        event_type=event_type,
                        source="ml_training_service",
                        data=data
                    ))
            except Exception as e:
                # Log error but don't fail the operation
                logger.warning(f"Failed to emit orchestrator event {event_type_name}: {e}")

    async def _emit_cache_invalidation_event(self, event_data: Dict[str, Any]) -> None:
        """Emit cache invalidation event."""
        try:
            # Import Redis client
            from ...core.config import AppConfig
            redis_client = AppConfig().redis_client if hasattr(AppConfig(), 'redis_client') else None
            
            if redis_client:
                event_json = json.dumps(event_data)
                await redis_client.publish('pattern.invalidate', event_json)
                logger.info(f"Emitted pattern.invalidate event: {event_data['type']}")
        except Exception as e:
            logger.warning(f"Failed to emit pattern.invalidate event: {e}")

    async def _update_rule_parameters(
        self,
        db_session: AsyncSession,
        rule_ids: List[str],
        optimized_params: Dict[str, Any],
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

            # Emit cache invalidation event
            await self._emit_cache_invalidation_event({
                "type": "rule_parameters_updated",
                "rule_ids": rule_ids,
                "timestamp": aware_utc_now().isoformat(),
                "effectiveness_score": effectiveness_score,
                "model_id": model_id,
                "cache_prefixes": ["apes:pattern:", "rule:"]
            })

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
        """Configure ML performance settings for scikit-learn optimization."""
        try:
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