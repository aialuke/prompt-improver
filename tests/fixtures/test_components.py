"""
Test Components for ML Pipeline Testing Infrastructure (2025).

Provides realistic test components that implement common ML pipeline patterns
for comprehensive testing of dependency injection, component lifecycle,
and Protocol compliance.
"""

import asyncio
import logging
from typing import Any

from prompt_improver.utils.datetime_utils import aware_utc_now


class TestMLComponent:
    """Test ML component for dependency injection testing."""

    def __init__(
        self,
        database_service=None,
        cache_service=None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize test ML component with dependencies."""
        self.database_service = database_service
        self.cache_service = cache_service
        self.config = config or {}
        self.is_initialized = False
        self.is_shutdown = False
        self.operation_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize component asynchronously."""
        if self.is_initialized:
            return
        await asyncio.sleep(0.01)
        self.is_initialized = True
        self.operation_history.append({
            "operation": "initialize",
            "timestamp": aware_utc_now().isoformat(),
            "success": True,
        })
        self.logger.debug("TestMLComponent initialized")

    async def process_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process data using dependencies."""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        result = {
            "original_data": data,
            "processed_at": aware_utc_now().isoformat(),
            "component_id": "test_ml_component",
            "processing_result": "success",
        }
        if self.database_service:
            db_stats = await self.database_service.get_connection_pool_stats()
            result["db_stats"] = db_stats
        if self.cache_service:
            cache_key = f"processed_data_{hash(str(data))}"
            await self.cache_service.set(cache_key, result, ttl=300)
            result["cached"] = True
        self.operation_history.append({
            "operation": "process_data",
            "timestamp": aware_utc_now().isoformat(),
            "data_size": len(str(data)),
            "success": True,
        })
        return result

    async def get_metrics(self) -> dict[str, Any]:
        """Get component performance metrics."""
        return {
            "operations_count": len(self.operation_history),
            "is_initialized": self.is_initialized,
            "is_shutdown": self.is_shutdown,
            "config": self.config,
            "last_operation": self.operation_history[-1]
            if self.operation_history
            else None,
        }

    async def shutdown(self) -> None:
        """Shutdown component gracefully."""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        self.operation_history.append({
            "operation": "shutdown",
            "timestamp": aware_utc_now().isoformat(),
            "success": True,
        })
        self.logger.debug("TestMLComponent shutdown")


class TestTrainingComponent:
    """Test training component for ML pipeline testing."""

    def __init__(
        self,
        mlflow_service=None,
        database_service=None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize test training component with dependencies."""
        self.mlflow_service = mlflow_service
        self.database_service = database_service
        self.config = config or {}
        self.training_history = []
        self.is_initialized = False
        self.is_shutdown = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize training component."""
        if self.is_initialized:
            return
        await asyncio.sleep(0.01)
        self.is_initialized = True
        self.logger.debug("TestTrainingComponent initialized")

    async def train_model(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train a model with given data."""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        training_samples = self.config.get("training_samples", 100)
        await asyncio.sleep(0.02)
        training_result = {
            "model_id": f"model_{len(self.training_history)}",
            "training_samples": training_samples,
            "accuracy": 0.85 + len(self.training_history) * 0.01,
            "precision": 0.82,
            "recall": 0.88,
            "training_duration_ms": 20,
            "trained_at": aware_utc_now().isoformat(),
        }
        if self.mlflow_service:
            experiment_name = f"test_experiment_{len(self.training_history)}"
            run_id = await self.mlflow_service.log_experiment(
                experiment_name,
                {"training_samples": training_samples, "model_type": "test_model"},
            )
            training_result["mlflow_run_id"] = run_id
            model_uri = await self.mlflow_service.log_model(
                training_result["model_id"],
                {"model_type": "sklearn_test"},
                training_result,
            )
            training_result["model_uri"] = model_uri
        if self.database_service:
            await self.database_service.execute_query(
                "INSERT INTO training_results (model_id, accuracy, trained_at) VALUES ($1, $2, $3)",
                {
                    "model_id": training_result["model_id"],
                    "accuracy": training_result["accuracy"],
                    "trained_at": training_result["trained_at"],
                },
            )
        self.training_history.append(training_result)
        return training_result

    async def evaluate_model(
        self, model_id: str, test_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a trained model."""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        model = None
        for training in self.training_history:
            if training["model_id"] == model_id:
                model = training
                break
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        await asyncio.sleep(0.01)
        evaluation_result = {
            "model_id": model_id,
            "test_accuracy": model["accuracy"] - 0.02,
            "test_precision": 0.8,
            "test_recall": 0.85,
            "evaluation_samples": len(test_data.get("samples", [])),
            "evaluated_at": aware_utc_now().isoformat(),
        }
        if self.mlflow_service and "mlflow_run_id" in model:
            trace_id = await self.mlflow_service.start_trace(
                f"evaluation_{model_id}",
                {
                    "model_id": model_id,
                    "test_samples": evaluation_result["evaluation_samples"],
                },
            )
            await self.mlflow_service.end_trace(trace_id, evaluation_result)
            evaluation_result["trace_id"] = trace_id
        return evaluation_result

    async def get_training_history(self) -> list[dict[str, Any]]:
        """Get complete training history."""
        return self.training_history.copy()

    async def shutdown(self) -> None:
        """Shutdown training component."""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        self.logger.debug("TestTrainingComponent shutdown")


class TestOptimizationComponent:
    """Test optimization component for advanced ML pipeline testing."""

    def __init__(
        self, cache_service=None, event_bus=None, config: dict[str, Any] | None = None
    ):
        """Initialize test optimization component."""
        self.cache_service = cache_service
        self.event_bus = event_bus
        self.config = config or {}
        self.optimization_runs = []
        self.is_initialized = False
        self.is_shutdown = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize optimization component."""
        if self.is_initialized:
            return
        await asyncio.sleep(0.01)
        self.is_initialized = True
        if self.event_bus:
            await self.event_bus.subscribe(
                "optimization_request", self._handle_optimization_event
            )
        self.logger.debug("TestOptimizationComponent initialized")

    async def optimize_hyperparameters(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize hyperparameters for given parameters."""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        optimization_id = f"opt_{len(self.optimization_runs)}"
        cache_key = f"optimization_{hash(str(parameters))}"
        if self.cache_service:
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
        await asyncio.sleep(0.03)
        optimization_result = {
            "optimization_id": optimization_id,
            "input_parameters": parameters,
            "optimized_parameters": {
                "learning_rate": 0.001 + len(self.optimization_runs) * 0.0001,
                "batch_size": 32 if len(self.optimization_runs) % 2 == 0 else 64,
                "epochs": 100,
                "regularization": 0.01,
            },
            "optimization_score": 0.9 + len(self.optimization_runs) * 0.005,
            "optimization_time_ms": 30,
            "optimized_at": aware_utc_now().isoformat(),
            "from_cache": False,
        }
        if self.cache_service:
            await self.cache_service.set(cache_key, optimization_result, ttl=600)
        if self.event_bus:
            await self.event_bus.publish(
                "optimization_completed",
                {
                    "optimization_id": optimization_id,
                    "score": optimization_result["optimization_score"],
                    "parameters": optimization_result["optimized_parameters"],
                },
            )
        self.optimization_runs.append(optimization_result)
        return optimization_result

    async def _handle_optimization_event(self, event_data: dict[str, Any]) -> None:
        """Handle optimization event from event bus."""
        self.logger.debug(f"Received optimization event: {event_data}")

    async def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get complete optimization history."""
        return self.optimization_runs.copy()

    async def shutdown(self) -> None:
        """Shutdown optimization component."""
        if self.is_shutdown:
            return
        if self.event_bus:
            pass
        self.is_shutdown = True
        self.logger.debug("TestOptimizationComponent shutdown")


class TestWorkflowComponent:
    """Test workflow component for pipeline orchestration testing."""

    def __init__(self, **dependencies):
        """Initialize workflow component with any dependencies."""
        self.dependencies = dependencies
        self.workflows = {}
        self.workflow_counter = 0
        self.is_initialized = False
        self.is_shutdown = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize workflow component."""
        if self.is_initialized:
            return
        await asyncio.sleep(0.01)
        self.is_initialized = True
        self.logger.debug("TestWorkflowComponent initialized")

    async def execute_workflow(
        self, workflow_definition: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a workflow with given definition."""
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        workflow_id = f"workflow_{self.workflow_counter}"
        self.workflow_counter += 1
        workflow_steps = workflow_definition.get("steps", [])
        step_results = []
        for i, step in enumerate(workflow_steps):
            await asyncio.sleep(0.005)
            step_result = {
                "step_id": i,
                "step_name": step.get("name", f"step_{i}"),
                "status": "completed",
                "duration_ms": 5,
                "output": f"Step {i} completed successfully",
            }
            step_results.append(step_result)
        workflow_result = {
            "workflow_id": workflow_id,
            "definition": workflow_definition,
            "steps": step_results,
            "status": "completed",
            "total_duration_ms": len(workflow_steps) * 5,
            "executed_at": aware_utc_now().isoformat(),
            "success": True,
        }
        self.workflows[workflow_id] = workflow_result
        return workflow_result

    async def get_workflow_status(self, workflow_id: str) -> str:
        """Get workflow execution status."""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id]["status"]
        return "not_found"

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id]["status"] = "cancelled"
            return True
        return False

    async def get_all_workflows(self) -> dict[str, dict[str, Any]]:
        """Get all workflow results."""
        return self.workflows.copy()

    async def shutdown(self) -> None:
        """Shutdown workflow component."""
        if self.is_shutdown:
            return
        for workflow in self.workflows.values():
            if workflow["status"] == "running":
                workflow["status"] = "cancelled"
        self.is_shutdown = True
        self.logger.debug("TestWorkflowComponent shutdown")
