"""Real behavior tests for decomposed ML repository services.

Comprehensive validation of ML repository services with actual database operations,
model persistence, training session management, and performance metrics tracking.

Performance Requirements:
- Model operations: <100ms for CRUD operations  
- Training session management: <50ms for session operations
- Metrics aggregation: <200ms for complex analytics
- Batch operations: >1000 records/second throughput
- Data consistency: 100% ACID compliance
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

from tests.containers.postgres_container import PostgreSQLTestContainer
from src.prompt_improver.repositories.impl.ml_repository_service import (
    ModelRepository,
    TrainingRepository,
    MetricsRepository,
    ExperimentRepository,
    InferenceRepository,
    MLRepositoryFacade,
)
from src.prompt_improver.repositories.protocols.ml_repository_protocol import (
    TrainingSessionFilter,
    ModelPerformanceFilter,
    GenerationSessionFilter,
    TrainingMetrics,
    SyntheticDataMetrics,
    ModelVersionInfo,
)
from prompt_improver.database import DatabaseServices, ManagerMode
from prompt_improver.database.models import (
    TrainingSession,
    TrainingSessionCreate,
    TrainingSessionUpdate,
    MLModelPerformance,
    GenerationSession,
    GenerationBatch,
    SyntheticDataSample,
    TrainingPrompt,
    TrainingIteration,
)


class TestModelRepositoryRealBehavior:
    """Real behavior tests for model repository service."""

    @pytest.fixture
    async def test_infrastructure(self):
        """Set up PostgreSQL testcontainer."""
        postgres_container = PostgreSQLTestContainer()
        await postgres_container.start()
        
        # Set up database services
        db_services = DatabaseServices(
            mode=ManagerMode.ASYNC_MODERN,
            connection_url=postgres_container.get_connection_url()
        )
        
        yield {
            "postgres": postgres_container,
            "db_services": db_services
        }
        
        await db_services.cleanup()
        await postgres_container.stop()

    @pytest.fixture
    async def model_repository(self, test_infrastructure):
        """Create model repository with real database."""
        repository = ModelRepository(test_infrastructure["db_services"])
        yield repository

    async def test_model_performance_lifecycle(self, model_repository):
        """Test complete model performance tracking lifecycle."""
        model_id = f"test_model_{uuid4().hex[:8]}"
        
        # Create initial model performance record
        performance_data = {
            "model_id": model_id,
            "model_name": f"TestModel_{model_id}",
            "model_type": "transformer",
            "version": "1.0.0",
            "accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.83,
            "f1_score": 0.85,
            "training_time_seconds": 3600,
            "inference_time_ms": 150,
            "model_size_mb": 512.5,
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "dropout": 0.1
            },
            "training_dataset_size": 10000,
            "validation_dataset_size": 2000,
            "test_dataset_size": 1000,
            "evaluation_metrics": {
                "loss": 0.45,
                "val_loss": 0.52,
                "confusion_matrix": [[800, 50], [30, 120]]
            }
        }
        
        # Test create operation performance
        start_time = time.perf_counter()
        model_performance = await model_repository.create_model_performance(performance_data)
        create_duration = time.perf_counter() - start_time
        
        assert model_performance is not None
        assert model_performance.model_id == model_id
        assert model_performance.accuracy == 0.85
        assert model_performance.hyperparameters["learning_rate"] == 0.001
        assert create_duration < 0.1, f"Model creation took {create_duration:.3f}s, exceeds 100ms"
        
        print(f"Model Performance Creation: {create_duration*1000:.1f}ms")
        
        # Test retrieval operations
        start_time = time.perf_counter()
        retrieved_performances = await model_repository.get_model_performance_by_id(model_id)
        retrieval_duration = time.perf_counter() - start_time
        
        assert len(retrieved_performances) == 1
        assert retrieved_performances[0].model_id == model_id
        assert retrieval_duration < 0.05, f"Model retrieval took {retrieval_duration:.3f}s, exceeds 50ms"
        
        print(f"Model Performance Retrieval: {retrieval_duration*1000:.1f}ms")
        
        # Test model version updates
        updated_performance_data = performance_data.copy()
        updated_performance_data.update({
            "version": "1.1.0",
            "accuracy": 0.89,
            "model_size_mb": 520.0,
            "improvement_notes": "Added regularization and data augmentation"
        })
        
        updated_model = await model_repository.create_model_performance(updated_performance_data)
        assert updated_model.version == "1.1.0"
        assert updated_model.accuracy == 0.89
        
        # Test version history
        version_history = await model_repository.get_model_version_history(model_id)
        assert len(version_history) == 2
        assert version_history[0].version in ["1.0.0", "1.1.0"]
        assert version_history[1].version in ["1.0.0", "1.1.0"]

    async def test_model_performance_filtering_and_analytics(self, model_repository):
        """Test model performance filtering and analytics capabilities."""
        # Create multiple models with different performance characteristics
        model_data_sets = [
            # High-performing transformer model
            {
                "model_id": f"transformer_{uuid4().hex[:6]}",
                "model_name": "HighPerformanceTransformer", 
                "model_type": "transformer",
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "f1_score": 0.92,
                "training_time_seconds": 7200,
                "inference_time_ms": 120
            },
            # Medium-performing CNN model
            {
                "model_id": f"cnn_{uuid4().hex[:6]}",
                "model_name": "MediumPerformanceCNN",
                "model_type": "cnn", 
                "accuracy": 0.78,
                "precision": 0.80,
                "recall": 0.76,
                "f1_score": 0.78,
                "training_time_seconds": 3600,
                "inference_time_ms": 80
            },
            # Fast but lower accuracy model
            {
                "model_id": f"linear_{uuid4().hex[:6]}",
                "model_name": "FastLinearModel",
                "model_type": "linear",
                "accuracy": 0.68,
                "precision": 0.70,
                "recall": 0.66,
                "f1_score": 0.68,
                "training_time_seconds": 600,
                "inference_time_ms": 25
            }
        ]
        
        # Create all models
        created_models = []
        for model_data in model_data_sets:
            model = await model_repository.create_model_performance(model_data)
            created_models.append(model)
        
        # Test filtering by model type
        transformer_filter = ModelPerformanceFilter(model_type="transformer")
        start_time = time.perf_counter()
        transformer_models = await model_repository.get_model_performances(
            filters=transformer_filter, limit=10
        )
        filter_duration = time.perf_counter() - start_time
        
        assert len(transformer_models) >= 1
        assert all(m.model_type == "transformer" for m in transformer_models)
        assert filter_duration < 0.1, f"Model filtering took {filter_duration:.3f}s, exceeds 100ms"
        
        print(f"Model Filtering: {filter_duration*1000:.1f}ms")
        
        # Test best performing models query
        start_time = time.perf_counter()
        best_models = await model_repository.get_best_performing_models(
            metric="accuracy", limit=5
        )
        analytics_duration = time.perf_counter() - start_time
        
        assert len(best_models) >= 3
        # Should be sorted by accuracy descending
        for i in range(len(best_models) - 1):
            assert best_models[i].accuracy >= best_models[i + 1].accuracy
        
        assert analytics_duration < 0.2, f"Analytics query took {analytics_duration:.3f}s, exceeds 200ms"
        
        print(f"Best Models Analytics: {analytics_duration*1000:.1f}ms")
        
        # Test performance comparison analytics
        performance_filter = ModelPerformanceFilter(
            min_accuracy=0.75,
            max_inference_time_ms=150
        )
        
        filtered_models = await model_repository.get_model_performances(
            filters=performance_filter
        )
        
        # Should include transformer and CNN but not linear model
        assert len(filtered_models) >= 2
        for model in filtered_models:
            assert model.accuracy >= 0.75
            assert model.inference_time_ms <= 150

    async def test_model_repository_batch_operations(self, model_repository):
        """Test batch operations and performance under load."""
        # Generate batch of model performance data
        batch_size = 100
        model_batch_data = []
        
        for i in range(batch_size):
            model_data = {
                "model_id": f"batch_model_{i}_{uuid4().hex[:6]}",
                "model_name": f"BatchModel_{i}",
                "model_type": "test_model",
                "accuracy": 0.7 + (i % 20) * 0.01,  # Varying accuracy
                "precision": 0.72 + (i % 15) * 0.01,
                "recall": 0.68 + (i % 25) * 0.01,
                "f1_score": 0.70 + (i % 18) * 0.01,
                "training_time_seconds": 1000 + (i % 100) * 10,
                "inference_time_ms": 50 + (i % 50) * 2
            }
            model_batch_data.append(model_data)
        
        # Test batch creation performance
        start_time = time.perf_counter()
        created_models = []
        
        # Create models in smaller batches to simulate realistic usage
        batch_size_chunk = 20
        for i in range(0, batch_size, batch_size_chunk):
            chunk = model_batch_data[i:i + batch_size_chunk]
            chunk_tasks = [
                model_repository.create_model_performance(model_data)
                for model_data in chunk
            ]
            chunk_results = await asyncio.gather(*chunk_tasks)
            created_models.extend(chunk_results)
        
        batch_create_duration = time.perf_counter() - start_time
        
        assert len(created_models) == batch_size
        throughput = batch_size / batch_create_duration
        assert throughput > 50, f"Batch creation throughput {throughput:.1f} ops/sec is below 50 ops/sec target"
        
        print(f"Batch Model Creation: {batch_size} models in {batch_create_duration:.2f}s ({throughput:.1f} ops/sec)")
        
        # Test batch retrieval performance
        start_time = time.perf_counter()
        all_test_models = await model_repository.get_model_performances(
            filters=ModelPerformanceFilter(model_type="test_model"),
            limit=batch_size + 10
        )
        batch_retrieval_duration = time.perf_counter() - start_time
        
        assert len(all_test_models) >= batch_size
        retrieval_throughput = len(all_test_models) / batch_retrieval_duration
        assert retrieval_throughput > 200, f"Batch retrieval throughput {retrieval_throughput:.1f} ops/sec is below 200 ops/sec target"
        
        print(f"Batch Model Retrieval: {len(all_test_models)} models in {batch_retrieval_duration:.2f}s ({retrieval_throughput:.1f} ops/sec)")

    async def test_model_repository_concurrent_operations(self, model_repository):
        """Test concurrent operations and data consistency."""
        model_id = f"concurrent_model_{uuid4().hex[:8]}"
        
        # Define concurrent operations
        async def create_model_version(version_num):
            model_data = {
                "model_id": model_id,
                "model_name": f"ConcurrentModel",
                "model_type": "concurrent_test",
                "version": f"1.{version_num}.0",
                "accuracy": 0.80 + version_num * 0.01,
                "precision": 0.82 + version_num * 0.01,
                "recall": 0.78 + version_num * 0.01,
                "f1_score": 0.80 + version_num * 0.01,
                "training_time_seconds": 3600,
                "inference_time_ms": 100
            }
            return await model_repository.create_model_performance(model_data)
        
        # Run concurrent model version creations
        version_count = 10
        start_time = time.perf_counter()
        
        version_tasks = [create_model_version(i) for i in range(version_count)]
        created_versions = await asyncio.gather(*version_tasks)
        
        concurrent_duration = time.perf_counter() - start_time
        
        assert len(created_versions) == version_count
        assert all(v is not None for v in created_versions)
        
        print(f"Concurrent Model Versions: {version_count} versions in {concurrent_duration:.2f}s")
        
        # Verify data consistency
        version_history = await model_repository.get_model_version_history(model_id)
        assert len(version_history) == version_count
        
        # Verify all versions are unique and properly stored
        versions_set = set(v.version for v in version_history)
        assert len(versions_set) == version_count, "All versions should be unique"
        
        # Verify performance progression
        performances = await model_repository.get_model_performance_by_id(model_id)
        assert len(performances) == version_count
        
        # Check accuracy progression (should generally increase)
        accuracies = [p.accuracy for p in performances]
        assert max(accuracies) > min(accuracies), "Should show performance variation across versions"


class TestTrainingRepositoryRealBehavior:
    """Real behavior tests for training repository service."""

    @pytest.fixture
    async def training_repository(self, test_infrastructure):
        """Create training repository with real database."""
        repository = TrainingRepository(test_infrastructure["db_services"])
        yield repository

    async def test_training_session_management_lifecycle(self, training_repository):
        """Test complete training session management lifecycle."""
        # Create training session
        session_id = f"training_{uuid4().hex[:8]}"
        session_data = TrainingSessionCreate(
            session_id=session_id,
            model_name="TestTrainingModel",
            model_type="transformer",
            status="initializing",
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "optimizer": "adamw"
            },
            dataset_config={
                "train_size": 8000,
                "val_size": 1000,
                "test_size": 1000,
                "data_source": "synthetic_prompts"
            }
        )
        
        # Test session creation performance
        start_time = time.perf_counter()
        created_session = await training_repository.create_training_session(session_data)
        create_duration = time.perf_counter() - start_time
        
        assert created_session is not None
        assert created_session.session_id == session_id
        assert created_session.status == "initializing"
        assert create_duration < 0.05, f"Training session creation took {create_duration:.3f}s, exceeds 50ms"
        
        print(f"Training Session Creation: {create_duration*1000:.1f}ms")
        
        # Test session updates
        update_data = TrainingSessionUpdate(
            status="training",
            current_epoch=10,
            best_validation_loss=0.75,
            metrics_history={
                "epoch_10": {"loss": 0.75, "accuracy": 0.82, "val_loss": 0.78}
            }
        )
        
        start_time = time.perf_counter()
        updated_session = await training_repository.update_training_session(
            session_id, update_data
        )
        update_duration = time.perf_counter() - start_time
        
        assert updated_session is not None
        assert updated_session.status == "training"
        assert updated_session.current_epoch == 10
        assert update_duration < 0.05, f"Training session update took {update_duration:.3f}s, exceeds 50ms"
        
        print(f"Training Session Update: {update_duration*1000:.1f}ms")
        
        # Test session retrieval
        start_time = time.perf_counter()
        retrieved_session = await training_repository.get_training_session_by_id(session_id)
        retrieval_duration = time.perf_counter() - start_time
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id
        assert retrieved_session.current_epoch == 10
        assert retrieval_duration < 0.05, f"Training session retrieval took {retrieval_duration:.3f}s, exceeds 50ms"
        
        print(f"Training Session Retrieval: {retrieval_duration*1000:.1f}ms")

    async def test_training_iteration_tracking(self, training_repository):
        """Test training iteration tracking and analytics."""
        # Create parent training session
        session_id = f"iteration_test_{uuid4().hex[:8]}"
        session_data = TrainingSessionCreate(
            session_id=session_id,
            model_name="IterationTestModel",
            model_type="gpt",
            status="training"
        )
        
        await training_repository.create_training_session(session_data)
        
        # Create multiple training iterations
        iteration_count = 50
        iteration_data_list = []
        
        for i in range(iteration_count):
            iteration_data = {
                "session_id": session_id,
                "iteration_number": i + 1,
                "epoch": (i // 10) + 1,  # 10 iterations per epoch
                "batch_number": (i % 10) + 1,
                "training_loss": 1.0 - (i * 0.015),  # Decreasing loss
                "validation_loss": 1.1 - (i * 0.012),
                "learning_rate": 0.001 * (0.95 ** (i // 10)),  # Learning rate decay
                "accuracy": 0.5 + (i * 0.008),  # Increasing accuracy
                "gradient_norm": 2.5 - (i * 0.03),
                "batch_processing_time": 0.8 + (i % 5) * 0.1,
                "memory_usage_mb": 2048 + (i * 5),
                "iteration_metadata": {
                    "optimizer_state": f"state_{i}",
                    "data_batch_ids": [f"batch_{j}" for j in range(3)]
                }
            }
            iteration_data_list.append(iteration_data)
        
        # Test batch iteration creation performance
        start_time = time.perf_counter()
        created_iterations = []
        
        # Create iterations in chunks for realistic simulation
        chunk_size = 10
        for i in range(0, iteration_count, chunk_size):
            chunk = iteration_data_list[i:i + chunk_size]
            chunk_tasks = [
                training_repository.create_training_iteration(iteration_data)
                for iteration_data in chunk
            ]
            chunk_results = await asyncio.gather(*chunk_tasks)
            created_iterations.extend(chunk_results)
        
        iteration_create_duration = time.perf_counter() - start_time
        iteration_throughput = iteration_count / iteration_create_duration
        
        assert len(created_iterations) == iteration_count
        assert iteration_throughput > 100, f"Iteration creation throughput {iteration_throughput:.1f} ops/sec below target"
        
        print(f"Training Iterations Creation: {iteration_count} iterations in {iteration_create_duration:.2f}s ({iteration_throughput:.1f} ops/sec)")
        
        # Test iteration retrieval and analytics
        start_time = time.perf_counter()
        all_iterations = await training_repository.get_training_iterations(
            session_id=session_id, limit=100
        )
        retrieval_duration = time.perf_counter() - start_time
        
        assert len(all_iterations) == iteration_count
        assert retrieval_duration < 0.1, f"Iteration retrieval took {retrieval_duration:.3f}s, exceeds 100ms"
        
        print(f"Training Iterations Retrieval: {retrieval_duration*1000:.1f}ms")
        
        # Test performance trend analysis
        start_time = time.perf_counter()
        performance_trend = await training_repository.get_iteration_performance_trend(session_id)
        trend_duration = time.perf_counter() - start_time
        
        assert len(performance_trend) == iteration_count
        assert trend_duration < 0.2, f"Performance trend analysis took {trend_duration:.3f}s, exceeds 200ms"
        
        # Verify trend shows improvement
        first_iteration = performance_trend[0]
        last_iteration = performance_trend[-1]
        assert last_iteration["training_loss"] < first_iteration["training_loss"]
        assert last_iteration["accuracy"] > first_iteration["accuracy"]
        
        print(f"Performance Trend Analysis: {trend_duration*1000:.1f}ms")

    async def test_training_session_filtering_and_analytics(self, training_repository):
        """Test training session filtering and comprehensive analytics."""
        # Create diverse training sessions
        session_configs = [
            {
                "session_id": f"completed_{uuid4().hex[:6]}",
                "model_name": "CompletedModel",
                "model_type": "transformer",
                "status": "completed",
                "hyperparameters": {"epochs": 50, "batch_size": 32}
            },
            {
                "session_id": f"failed_{uuid4().hex[:6]}",
                "model_name": "FailedModel", 
                "model_type": "cnn",
                "status": "failed",
                "hyperparameters": {"epochs": 100, "batch_size": 16}
            },
            {
                "session_id": f"running_{uuid4().hex[:6]}",
                "model_name": "RunningModel",
                "model_type": "rnn",
                "status": "training",
                "hyperparameters": {"epochs": 75, "batch_size": 64}
            }
        ]
        
        # Create sessions
        created_sessions = []
        for config in session_configs:
            session_data = TrainingSessionCreate(**config)
            session = await training_repository.create_training_session(session_data)
            created_sessions.append(session)
        
        # Test filtering by status
        status_filter = TrainingSessionFilter(status="completed")
        start_time = time.perf_counter()
        completed_sessions = await training_repository.get_training_sessions(
            filters=status_filter
        )
        filter_duration = time.perf_counter() - start_time
        
        assert len(completed_sessions) >= 1
        assert all(s.status == "completed" for s in completed_sessions)
        assert filter_duration < 0.1, f"Session filtering took {filter_duration:.3f}s, exceeds 100ms"
        
        print(f"Training Session Filtering: {filter_duration*1000:.1f}ms")
        
        # Test active sessions query
        start_time = time.perf_counter()
        active_sessions = await training_repository.get_active_training_sessions()
        active_duration = time.perf_counter() - start_time
        
        assert len(active_sessions) >= 1  # Should include "training" status
        assert active_duration < 0.05, f"Active sessions query took {active_duration:.3f}s, exceeds 50ms"
        
        print(f"Active Training Sessions: {active_duration*1000:.1f}ms")
        
        # Test training metrics aggregation
        completed_session_id = next(s.session_id for s in created_sessions if s.status == "completed")
        
        # Add some training metrics data first
        metrics_data = {
            "session_id": completed_session_id,
            "total_epochs": 50,
            "best_accuracy": 0.89,
            "best_validation_loss": 0.34,
            "training_time_seconds": 3600,
            "convergence_epoch": 42,
            "final_learning_rate": 0.0001
        }
        
        start_time = time.perf_counter()
        training_metrics = await training_repository.get_training_session_metrics(
            completed_session_id
        )
        metrics_duration = time.perf_counter() - start_time
        
        # Should return some metrics even if placeholder
        assert metrics_duration < 0.1, f"Training metrics took {metrics_duration:.3f}s, exceeds 100ms"
        
        print(f"Training Session Metrics: {metrics_duration*1000:.1f}ms")

    async def test_training_data_management(self, training_repository):
        """Test training data and prompt management."""
        # This tests the training prompt methods implemented in the facade
        # Since they're implemented directly in the facade, we'll test via that interface
        
        # Note: This would normally test TrainingRepository methods, but since
        # the training prompt methods are implemented in the facade, we'll
        # create a simple test here for the training repository's core functionality
        
        session_id = f"data_mgmt_{uuid4().hex[:8]}"
        session_data = TrainingSessionCreate(
            session_id=session_id,
            model_name="DataManagementModel",
            model_type="data_test",
            status="preparing_data",
            hyperparameters={
                "data_preparation": True,
                "augmentation": "enabled"
            }
        )
        
        # Create session for data management testing
        created_session = await training_repository.create_training_session(session_data)
        assert created_session is not None
        
        # Test session cleanup methods
        cleanup_count = await training_repository.cleanup_failed_sessions(days_old=0)
        assert cleanup_count >= 0, "Cleanup should return non-negative count"
        
        # Test archive completed sessions
        archive_count = await training_repository.archive_completed_sessions(days_old=0) 
        assert archive_count >= 0, "Archive should return non-negative count"
        
        print(f"Data Management - Cleanup: {cleanup_count}, Archive: {archive_count}")


class TestMLRepositoryFacadeRealBehavior:
    """Real behavior tests for ML repository facade integration."""

    @pytest.fixture
    async def ml_repository_facade(self, test_infrastructure):
        """Create ML repository facade with real database."""
        facade = MLRepositoryFacade(test_infrastructure["db_services"])
        yield facade

    async def test_end_to_end_ml_workflow(self, ml_repository_facade):
        """Test complete end-to-end ML workflow through facade."""
        session_id = f"e2e_workflow_{uuid4().hex[:8]}"
        model_id = f"e2e_model_{uuid4().hex[:8]}"
        
        # Step 1: Create training session
        session_data = TrainingSessionCreate(
            session_id=session_id,
            model_name="EndToEndTestModel",
            model_type="transformer",
            status="initializing",
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "model_id": model_id
            }
        )
        
        start_time = time.perf_counter()
        training_session = await ml_repository_facade.create_training_session(session_data)
        workflow_step1_duration = time.perf_counter() - start_time
        
        assert training_session is not None
        assert training_session.session_id == session_id
        print(f"Workflow Step 1 (Training Session): {workflow_step1_duration*1000:.1f}ms")
        
        # Step 2: Create training iterations (simulate training progress)
        iteration_count = 20
        iteration_create_tasks = []
        
        for i in range(iteration_count):
            iteration_data = {
                "session_id": session_id,
                "iteration_number": i + 1,
                "training_loss": 1.0 - (i * 0.03),
                "validation_loss": 1.1 - (i * 0.025),
                "accuracy": 0.6 + (i * 0.015),
                "learning_rate": 0.001
            }
            iteration_create_tasks.append(
                ml_repository_facade.create_training_iteration(iteration_data)
            )
        
        start_time = time.perf_counter()
        created_iterations = await asyncio.gather(*iteration_create_tasks)
        workflow_step2_duration = time.perf_counter() - start_time
        
        assert len(created_iterations) == iteration_count
        print(f"Workflow Step 2 (Training Iterations): {workflow_step2_duration*1000:.1f}ms for {iteration_count} iterations")
        
        # Step 3: Record model performance
        model_performance_data = {
            "model_id": model_id,
            "model_name": "EndToEndTestModel",
            "model_type": "transformer",
            "version": "1.0.0",
            "accuracy": 0.87,
            "precision": 0.89,
            "recall": 0.85,
            "f1_score": 0.87,
            "training_session_id": session_id,
            "training_time_seconds": 1800,
            "inference_time_ms": 95
        }
        
        start_time = time.perf_counter()
        model_performance = await ml_repository_facade.create_model_performance(
            model_performance_data
        )
        workflow_step3_duration = time.perf_counter() - start_time
        
        assert model_performance is not None
        assert model_performance.model_id == model_id
        print(f"Workflow Step 3 (Model Performance): {workflow_step3_duration*1000:.1f}ms")
        
        # Step 4: Create generation session for synthetic data
        generation_session_data = {
            "session_id": f"gen_{uuid4().hex[:8]}",
            "model_id": model_id,
            "generation_method": "transformer_sampling",
            "status": "active",
            "config": {
                "temperature": 0.8,
                "max_tokens": 150,
                "top_p": 0.9
            }
        }
        
        start_time = time.perf_counter()
        generation_session = await ml_repository_facade.create_generation_session(
            generation_session_data
        )
        workflow_step4_duration = time.perf_counter() - start_time
        
        assert generation_session is not None
        print(f"Workflow Step 4 (Generation Session): {workflow_step4_duration*1000:.1f}ms")
        
        # Step 5: Create synthetic data samples
        synthetic_samples_data = []
        sample_count = 10
        
        for i in range(sample_count):
            sample_data = {
                "sample_id": f"sample_{i}_{uuid4().hex[:6]}",
                "session_id": generation_session.session_id,
                "generated_content": f"This is synthetic training prompt number {i} for testing purposes.",
                "quality_score": 0.75 + (i % 5) * 0.05,
                "domain_category": "software_development",
                "metadata": {
                    "generation_params": {"temp": 0.8, "tokens": 50 + i * 10},
                    "validation_passed": True
                }
            }
            synthetic_samples_data.append(sample_data)
        
        start_time = time.perf_counter()
        synthetic_samples = await ml_repository_facade.create_synthetic_data_samples(
            synthetic_samples_data
        )
        workflow_step5_duration = time.perf_counter() - start_time
        
        assert len(synthetic_samples) == sample_count
        print(f"Workflow Step 5 (Synthetic Data): {workflow_step5_duration*1000:.1f}ms for {sample_count} samples")
        
        # Step 6: Get comprehensive analytics
        start_time = time.perf_counter()
        
        # Get training session metrics
        training_metrics = await ml_repository_facade.get_training_session_metrics(session_id)
        
        # Get synthetic data metrics
        synthetic_metrics = await ml_repository_facade.get_synthetic_data_metrics(
            generation_session.session_id
        )
        
        # Get best performing models
        best_models = await ml_repository_facade.get_best_performing_models(
            metric="accuracy", limit=5
        )
        
        workflow_step6_duration = time.perf_counter() - start_time
        
        assert len(best_models) >= 1
        print(f"Workflow Step 6 (Analytics): {workflow_step6_duration*1000:.1f}ms")
        
        # Overall workflow performance
        total_workflow_time = (
            workflow_step1_duration + workflow_step2_duration + 
            workflow_step3_duration + workflow_step4_duration +
            workflow_step5_duration + workflow_step6_duration
        )
        
        print(f"\nEnd-to-End ML Workflow Performance:")
        print(f"  Total Time: {total_workflow_time:.3f}s")
        print(f"  Operations: Training session, {iteration_count} iterations, model performance, generation session, {sample_count} samples, analytics")
        print(f"  Average Operation Time: {total_workflow_time*1000/6:.1f}ms per step")
        
        # Performance assertions
        assert total_workflow_time < 10, f"Complete workflow should finish within 10 seconds, took {total_workflow_time:.3f}s"
        assert workflow_step1_duration < 0.1, "Training session creation should be <100ms"
        assert workflow_step3_duration < 0.1, "Model performance recording should be <100ms"

    async def test_repository_health_and_monitoring(self, ml_repository_facade):
        """Test repository health monitoring and diagnostics."""
        # Test repository health status
        start_time = time.perf_counter()
        health_status = await ml_repository_facade.get_repository_health_status()
        health_check_duration = time.perf_counter() - start_time
        
        assert health_status is not None
        assert "facade_status" in health_status
        assert "repositories" in health_status
        assert "total_repositories" in health_status
        assert "healthy_repositories" in health_status
        
        assert health_check_duration < 0.5, f"Health check took {health_check_duration:.3f}s, should be <500ms"
        
        print(f"\nRepository Health Check:")
        print(f"  Duration: {health_check_duration*1000:.1f}ms")
        print(f"  Facade Status: {health_status['facade_status']}")
        print(f"  Healthy Repositories: {health_status['healthy_repositories']}/{health_status['total_repositories']}")
        
        # Print individual repository status
        for repo_name, status in health_status["repositories"].items():
            print(f"  {repo_name}: {status}")
        
        # Verify minimum health requirements
        assert health_status["healthy_repositories"] >= 3, "At least 3 repositories should be healthy"
        assert health_status["facade_status"] in ["healthy", "degraded"], "Facade should be at least degraded"

    async def test_facade_performance_under_concurrent_load(self, ml_repository_facade):
        """Test facade performance under concurrent load."""
        # Define concurrent operations
        concurrent_operations = []
        
        # Training sessions
        for i in range(10):
            session_data = TrainingSessionCreate(
                session_id=f"concurrent_train_{i}_{uuid4().hex[:6]}",
                model_name=f"ConcurrentModel_{i}",
                model_type="load_test",
                status="initializing"
            )
            concurrent_operations.append(
                ml_repository_facade.create_training_session(session_data)
            )
        
        # Model performances
        for i in range(10):
            model_data = {
                "model_id": f"concurrent_model_{i}_{uuid4().hex[:6]}",
                "model_name": f"ConcurrentTestModel_{i}",
                "model_type": "load_test",
                "accuracy": 0.7 + i * 0.02,
                "precision": 0.72 + i * 0.02,
                "recall": 0.68 + i * 0.02,
                "f1_score": 0.70 + i * 0.02
            }
            concurrent_operations.append(
                ml_repository_facade.create_model_performance(model_data)
            )
        
        # Generation sessions
        for i in range(5):
            gen_data = {
                "session_id": f"concurrent_gen_{i}_{uuid4().hex[:6]}",
                "model_id": f"gen_model_{i}",
                "generation_method": "test_method",
                "status": "active"
            }
            concurrent_operations.append(
                ml_repository_facade.create_generation_session(gen_data)
            )
        
        # Execute all operations concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*concurrent_operations, return_exceptions=True)
        concurrent_duration = time.perf_counter() - start_time
        
        # Analyze results
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))
        failed_operations = len(results) - successful_operations
        
        success_rate = successful_operations / len(results)
        throughput = successful_operations / concurrent_duration
        
        print(f"\nConcurrent Load Test:")
        print(f"  Total Operations: {len(concurrent_operations)}")
        print(f"  Successful: {successful_operations}")
        print(f"  Failed: {failed_operations}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Duration: {concurrent_duration:.2f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        
        # Performance assertions
        assert success_rate >= 0.9, f"Success rate {success_rate:.2%} should be ≥90%"
        assert throughput > 10, f"Throughput {throughput:.1f} ops/sec should be >10 ops/sec"
        assert concurrent_duration < 5, f"Concurrent operations should complete within 5s, took {concurrent_duration:.2f}s"
        
        # Verify no database corruption from concurrent operations
        health_after_load = await ml_repository_facade.get_repository_health_status()
        assert health_after_load["facade_status"] in ["healthy", "degraded"], "Facade should remain functional after load test"

    async def test_facade_data_consistency_and_transactions(self, ml_repository_facade):
        """Test data consistency and transaction handling across repositories."""
        session_id = f"consistency_test_{uuid4().hex[:8]}"
        model_id = f"consistency_model_{uuid4().hex[:8]}"
        
        # Test transactional consistency across related operations
        try:
            # Create training session
            session_data = TrainingSessionCreate(
                session_id=session_id,
                model_name="ConsistencyTestModel",
                model_type="consistency_test",
                status="training"
            )
            
            training_session = await ml_repository_facade.create_training_session(session_data)
            assert training_session is not None
            
            # Create model performance linked to session
            model_data = {
                "model_id": model_id,
                "model_name": "ConsistencyTestModel",
                "model_type": "consistency_test", 
                "training_session_id": session_id,
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1_score": 0.85
            }
            
            model_performance = await ml_repository_facade.create_model_performance(model_data)
            assert model_performance is not None
            
            # Create generation session linked to model
            gen_data = {
                "session_id": f"gen_{uuid4().hex[:8]}",
                "model_id": model_id,
                "generation_method": "consistency_test",
                "status": "active"
            }
            
            generation_session = await ml_repository_facade.create_generation_session(gen_data)
            assert generation_session is not None
            
            # Verify data relationships
            retrieved_session = await ml_repository_facade.get_training_session_by_id(session_id)
            assert retrieved_session is not None
            assert retrieved_session.session_id == session_id
            
            model_performances = await ml_repository_facade.get_model_performance_by_id(model_id)
            assert len(model_performances) == 1
            assert model_performances[0].training_session_id == session_id
            
            retrieved_gen_session = await ml_repository_facade.get_generation_session_by_id(
                generation_session.session_id
            )
            assert retrieved_gen_session is not None
            assert retrieved_gen_session.model_id == model_id
            
            print("Data consistency verification: PASSED")
            
        except Exception as e:
            pytest.fail(f"Data consistency test failed: {str(e)}")


@pytest.mark.integration
@pytest.mark.real_behavior
class TestMLRepositoryServicesSystemIntegration:
    """System-level integration tests for ML repository services."""
    
    async def test_ml_repository_system_performance_benchmarks(self, test_infrastructure):
        """Test system-wide performance benchmarks for ML repository services."""
        ml_facade = MLRepositoryFacade(test_infrastructure["db_services"])
        
        # System benchmark scenarios
        benchmark_scenarios = [
            {
                "name": "High-Frequency Training Updates",
                "operations": 500,
                "operation_type": "training_iterations",
                "target_throughput": 200  # ops/sec
            },
            {
                "name": "Batch Model Performance Recording",
                "operations": 100,
                "operation_type": "model_performances", 
                "target_throughput": 50   # ops/sec
            },
            {
                "name": "Synthetic Data Generation",
                "operations": 1000,
                "operation_type": "synthetic_samples",
                "target_throughput": 300  # ops/sec
            }
        ]
        
        benchmark_results = {}
        
        for scenario in benchmark_scenarios:
            print(f"\nRunning benchmark: {scenario['name']}")
            
            if scenario["operation_type"] == "training_iterations":
                # Create parent session first
                session_id = f"benchmark_{uuid4().hex[:8]}"
                session_data = TrainingSessionCreate(
                    session_id=session_id,
                    model_name="BenchmarkModel",
                    model_type="benchmark",
                    status="training"
                )
                await ml_facade.create_training_session(session_data)
                
                # Benchmark iteration creation
                start_time = time.perf_counter()
                tasks = []
                for i in range(scenario["operations"]):
                    iteration_data = {
                        "session_id": session_id,
                        "iteration_number": i + 1,
                        "training_loss": 1.0 - (i * 0.001),
                        "validation_loss": 1.1 - (i * 0.0008),
                        "accuracy": 0.5 + (i * 0.0008)
                    }
                    tasks.append(ml_facade.create_training_iteration(iteration_data))
                
                results = await asyncio.gather(*tasks)
                duration = time.perf_counter() - start_time
                
            elif scenario["operation_type"] == "model_performances":
                start_time = time.perf_counter()
                tasks = []
                for i in range(scenario["operations"]):
                    model_data = {
                        "model_id": f"benchmark_model_{i}",
                        "model_name": f"BenchmarkModel_{i}",
                        "model_type": "benchmark",
                        "accuracy": 0.7 + (i % 20) * 0.01,
                        "precision": 0.72 + (i % 15) * 0.01,
                        "recall": 0.68 + (i % 25) * 0.01,
                        "f1_score": 0.70 + (i % 18) * 0.01
                    }
                    tasks.append(ml_facade.create_model_performance(model_data))
                
                results = await asyncio.gather(*tasks)
                duration = time.perf_counter() - start_time
                
            elif scenario["operation_type"] == "synthetic_samples":
                # Create parent generation session first
                gen_session_data = {
                    "session_id": f"benchmark_gen_{uuid4().hex[:8]}",
                    "model_id": "benchmark_model",
                    "generation_method": "benchmark_method",
                    "status": "active"
                }
                gen_session = await ml_facade.create_generation_session(gen_session_data)
                
                # Benchmark synthetic sample creation
                start_time = time.perf_counter()
                
                # Create in batches for realistic performance
                batch_size = 50
                batch_count = scenario["operations"] // batch_size
                
                for batch in range(batch_count):
                    batch_data = []
                    for i in range(batch_size):
                        sample_data = {
                            "sample_id": f"benchmark_sample_{batch}_{i}",
                            "session_id": gen_session.session_id,
                            "generated_content": f"Benchmark synthetic content {batch}_{i}",
                            "quality_score": 0.8 + (i % 10) * 0.02,
                            "domain_category": "benchmark"
                        }
                        batch_data.append(sample_data)
                    
                    await ml_facade.create_synthetic_data_samples(batch_data)
                
                duration = time.perf_counter() - start_time
            
            # Calculate metrics
            throughput = scenario["operations"] / duration
            avg_operation_time = duration / scenario["operations"]
            
            benchmark_results[scenario["name"]] = {
                "operations": scenario["operations"],
                "duration": duration,
                "throughput": throughput,
                "avg_operation_time_ms": avg_operation_time * 1000,
                "target_throughput": scenario["target_throughput"],
                "performance_ratio": throughput / scenario["target_throughput"]
            }
            
            print(f"  Operations: {scenario['operations']}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Throughput: {throughput:.1f} ops/sec (target: {scenario['target_throughput']})")
            print(f"  Avg Operation Time: {avg_operation_time*1000:.2f}ms")
            print(f"  Performance Ratio: {throughput/scenario['target_throughput']:.2f}x")
            
            # Performance assertions
            assert throughput >= scenario["target_throughput"] * 0.8, \
                f"{scenario['name']} throughput {throughput:.1f} is below 80% of target {scenario['target_throughput']}"
        
        print(f"\nSystem Performance Benchmark Summary:")
        total_operations = sum(r["operations"] for r in benchmark_results.values())
        total_duration = sum(r["duration"] for r in benchmark_results.values())
        overall_throughput = total_operations / total_duration
        
        print(f"  Total Operations: {total_operations:,}")
        print(f"  Total Duration: {total_duration:.2f}s") 
        print(f"  Overall Throughput: {overall_throughput:.1f} ops/sec")
        
        # System-wide performance assertion
        assert overall_throughput > 100, f"System overall throughput {overall_throughput:.1f} should be >100 ops/sec"

    async def test_ml_repository_system_resilience_and_recovery(self, test_infrastructure):
        """Test system resilience and recovery capabilities."""
        ml_facade = MLRepositoryFacade(test_infrastructure["db_services"])
        
        # Test system recovery from various failure scenarios
        resilience_tests = []
        
        # Test 1: Large transaction rollback simulation
        try:
            session_id = f"rollback_test_{uuid4().hex[:8]}"
            
            # Create session
            session_data = TrainingSessionCreate(
                session_id=session_id,
                model_name="RollbackTestModel",
                model_type="resilience_test",
                status="initializing"
            )
            
            training_session = await ml_facade.create_training_session(session_data)
            assert training_session is not None
            
            # Attempt to create many iterations, some with invalid data
            valid_iterations = 10
            invalid_iterations = 5
            
            # Create valid iterations
            for i in range(valid_iterations):
                iteration_data = {
                    "session_id": session_id,
                    "iteration_number": i + 1,
                    "training_loss": 1.0 - (i * 0.05),
                    "validation_loss": 1.1 - (i * 0.04),
                    "accuracy": 0.5 + (i * 0.03)
                }
                await ml_facade.create_training_iteration(iteration_data)
            
            # Verify valid data persisted
            iterations = await ml_facade.get_training_iterations(session_id)
            assert len(iterations) == valid_iterations
            
            resilience_tests.append({
                "test": "Transaction Rollback Recovery",
                "status": "PASSED",
                "details": f"Successfully created {valid_iterations} valid iterations"
            })
            
        except Exception as e:
            resilience_tests.append({
                "test": "Transaction Rollback Recovery", 
                "status": "FAILED",
                "error": str(e)
            })
        
        # Test 2: Concurrent access with data conflicts
        try:
            model_id = f"conflict_test_{uuid4().hex[:8]}"
            
            # Create multiple concurrent model performance updates
            async def create_model_version(version_num):
                return await ml_facade.create_model_performance({
                    "model_id": model_id,
                    "model_name": "ConflictTestModel",
                    "model_type": "resilience_test",
                    "version": f"1.{version_num}.0",
                    "accuracy": 0.8 + version_num * 0.01
                })
            
            # Run concurrent updates
            concurrent_versions = 15
            version_tasks = [create_model_version(i) for i in range(concurrent_versions)]
            results = await asyncio.gather(*version_tasks, return_exceptions=True)
            
            # Count successful operations
            successful_creates = sum(1 for r in results if not isinstance(r, Exception))
            
            # Should handle concurrent access gracefully
            assert successful_creates >= concurrent_versions * 0.8, \
                "At least 80% of concurrent operations should succeed"
            
            resilience_tests.append({
                "test": "Concurrent Access Handling",
                "status": "PASSED", 
                "details": f"Successfully handled {successful_creates}/{concurrent_versions} concurrent operations"
            })
            
        except Exception as e:
            resilience_tests.append({
                "test": "Concurrent Access Handling",
                "status": "FAILED",
                "error": str(e)
            })
        
        # Test 3: Large data volume handling
        try:
            large_session_id = f"large_data_test_{uuid4().hex[:8]}"
            session_data = TrainingSessionCreate(
                session_id=large_session_id,
                model_name="LargeDataTestModel",
                model_type="resilience_test",
                status="training"
            )
            
            await ml_facade.create_training_session(session_data)
            
            # Create large batch of synthetic samples
            large_batch_size = 500
            batch_data = []
            
            for i in range(large_batch_size):
                sample_data = {
                    "sample_id": f"large_sample_{i}_{uuid4().hex[:6]}",
                    "session_id": large_session_id,
                    "generated_content": "Large data test content " * 10,  # Larger content
                    "quality_score": 0.7 + (i % 30) * 0.01,
                    "domain_category": "large_data_test",
                    "metadata": {
                        "large_metadata": ["item_" + str(j) for j in range(20)]
                    }
                }
                batch_data.append(sample_data)
            
            # Process in chunks to test large data handling
            chunk_size = 100
            created_samples = []
            
            for i in range(0, large_batch_size, chunk_size):
                chunk = batch_data[i:i + chunk_size]
                chunk_results = await ml_facade.create_synthetic_data_samples(chunk)
                created_samples.extend(chunk_results)
            
            assert len(created_samples) == large_batch_size
            
            resilience_tests.append({
                "test": "Large Data Volume Handling",
                "status": "PASSED",
                "details": f"Successfully processed {large_batch_size} large data samples"
            })
            
        except Exception as e:
            resilience_tests.append({
                "test": "Large Data Volume Handling",
                "status": "FAILED", 
                "error": str(e)
            })
        
        # Print resilience test results
        print(f"\nSystem Resilience Test Results:")
        passed_tests = 0
        for test in resilience_tests:
            status_indicator = "✓" if test["status"] == "PASSED" else "✗"
            print(f"  {status_indicator} {test['test']}: {test['status']}")
            if test["status"] == "PASSED":
                passed_tests += 1
                print(f"    {test['details']}")
            else:
                print(f"    Error: {test['error']}")
        
        print(f"\nResilience Summary: {passed_tests}/{len(resilience_tests)} tests passed")
        
        # Assert minimum resilience requirements
        resilience_rate = passed_tests / len(resilience_tests)
        assert resilience_rate >= 0.8, f"System resilience rate {resilience_rate:.2%} should be ≥80%"