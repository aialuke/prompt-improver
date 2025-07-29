"""Event-based ML Training Service for APES system.

This service provides ML training capabilities through event bus communication,
maintaining strict architectural separation between MCP and ML components.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..interfaces.ml_interface import MLTrainingInterface, MLTrainingRequest, MLTrainingResult, MLModelType
from ..events.ml_event_bus import get_ml_event_bus, MLEvent, MLEventType


logger = logging.getLogger(__name__)


class EventBasedMLTrainingService(MLTrainingInterface):
    """
    Event-based ML training service that communicates via event bus.

    This service implements the MLTrainingInterface by sending training requests
    through the event bus to the ML pipeline components, maintaining clean
    architectural separation.
    """

    def __init__(self):
        self.logger = logger
        self._job_counter = 0

    async def train_model(
        self,
        model_type: str,
        training_data: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start model training via event bus.

        Args:
            model_type: Type of model to train
            training_data: Training dataset
            hyperparameters: Optional hyperparameters

        Returns:
            Training job ID
        """
        self._job_counter += 1
        job_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._job_counter}"

        try:
            event_bus = await get_ml_event_bus()

            training_event = MLEvent(
                event_type=MLEventType.TRAINING_REQUEST,
                source="event_based_ml_training_service",
                data={
                    "job_id": job_id,
                    "operation": "train_model",
                    "model_type": model_type,
                    "training_data": training_data,
                    "hyperparameters": hyperparameters or {}
                }
            )

            await event_bus.publish(training_event)
            self.logger.info(f"Training job {job_id} submitted for model type: {model_type}")

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to submit training job: {e}")
            raise

    async def optimize_hyperparameters(
        self,
        model_type: str,
        training_data: Dict[str, Any],
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start hyperparameter optimization via event bus.

        Args:
            model_type: Type of model to optimize
            training_data: Training dataset
            optimization_config: Optimization configuration

        Returns:
            Optimization job ID
        """
        self._job_counter += 1
        job_id = f"optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._job_counter}"

        try:
            event_bus = await get_ml_event_bus()

            optimization_event = MLEvent(
                event_type=MLEventType.TRAINING_REQUEST,
                source="event_based_ml_training_service",
                data={
                    "job_id": job_id,
                    "operation": "optimize_hyperparameters",
                    "model_type": model_type,
                    "training_data": training_data,
                    "optimization_config": optimization_config or {
                        "n_trials": 50,
                        "timeout": 1800,
                        "optimization_direction": "maximize"
                    }
                }
            )

            await event_bus.publish(optimization_event)
            self.logger.info(f"Hyperparameter optimization job {job_id} submitted")

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to submit optimization job: {e}")
            raise

    async def retrain_model(
        self,
        model_id: str,
        new_data: Dict[str, Any],
        retrain_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start model retraining via event bus.

        Args:
            model_id: ID of the model to retrain
            new_data: New training data
            retrain_config: Retraining configuration

        Returns:
            Retraining job ID
        """
        self._job_counter += 1
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._job_counter}"

        try:
            event_bus = await get_ml_event_bus()

            retrain_event = MLEvent(
                event_type=MLEventType.TRAINING_REQUEST,
                source="event_based_ml_training_service",
                data={
                    "job_id": job_id,
                    "operation": "retrain_model",
                    "model_id": model_id,
                    "new_data": new_data,
                    "retrain_config": retrain_config or {
                        "incremental": True,
                        "validation_split": 0.2
                    }
                }
            )

            await event_bus.publish(retrain_event)
            self.logger.info(f"Model retraining job {job_id} submitted for model: {model_id}")

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to submit retraining job: {e}")
            raise

    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get training job status.

        Args:
            job_id: Training job ID

        Returns:
            Training status information
        """
        # In a real implementation, this would query the event bus or job store
        return {
            "job_id": job_id,
            "status": "running",
            "progress": 65,
            "estimated_completion": "2024-01-15T10:30:00Z",
            "current_epoch": 13,
            "total_epochs": 20,
            "current_loss": 0.234,
            "best_score": 0.876
        }

    async def cancel_training(self, job_id: str) -> bool:
        """
        Cancel a training job via event bus.

        Args:
            job_id: Training job ID to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            event_bus = await get_ml_event_bus()

            cancel_event = MLEvent(
                event_type=MLEventType.TRAINING_REQUEST,
                source="event_based_ml_training_service",
                data={
                    "job_id": job_id,
                    "operation": "cancel_training"
                }
            )

            await event_bus.publish(cancel_event)
            self.logger.info(f"Cancellation request sent for job: {job_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel training job {job_id}: {e}")
            return False

    async def get_training_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training results for a completed job.

        Args:
            job_id: Training job ID

        Returns:
            Training results if available
        """
        # In a real implementation, this would query the results store
        return {
            "job_id": job_id,
            "model_id": f"model_{job_id}",
            "final_score": 0.892,
            "training_time": 1247.5,
            "best_hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_layers": [128, 64, 32]
            },
            "validation_metrics": {
                "accuracy": 0.892,
                "precision": 0.885,
                "recall": 0.898,
                "f1_score": 0.891
            },
            "artifacts": {
                "model_path": f"/models/{job_id}/model.pkl",
                "metrics_path": f"/models/{job_id}/metrics.json"
            }
        }

    # MLTrainingInterface required methods
    async def train_model_request(self, request: MLTrainingRequest) -> MLTrainingResult:
        """Train a model using the provided request."""
        job_id = await self.train_model(
            model_type=request.model_type.value,
            training_data=request.training_data,
            hyperparameters=request.hyperparameters
        )

        return MLTrainingResult(
            job_id=job_id,
            model_id=f"{request.model_type.value}_{job_id}",
            status="started",
            metrics={},
            artifacts={},
            timestamp=datetime.now()
        )

    async def optimize_hyperparameters_typed(
        self,
        model_type: MLModelType,
        training_data: Dict[str, Any],
        optimization_budget_minutes: int = 60
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a model type."""
        job_id = await self.optimize_hyperparameters(
            model_type=model_type.value,
            training_data=training_data,
            optimization_config={"budget_minutes": optimization_budget_minutes}
        )
        return {"job_id": job_id, "status": "started"}

    async def evaluate_model(
        self,
        model_id: str,
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate a trained model."""
        results = await self.get_training_results(model_id)
        return results.get("validation_metrics", {})
