"""Integration tests for ML Repository Facade with real ML operations and testcontainers.

Tests the complete ML repository facade with real database operations,
actual ML workflows, and data integrity validation.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict

from prompt_improver.database import DatabaseServices
from prompt_improver.database.models import (
    GenerationSession,
    MLModelPerformance,
    SyntheticDataSample,
    TrainingSession,
    TrainingSessionCreate,
)
from prompt_improver.repositories.impl.ml_repository_service import MLRepositoryFacade
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    TrainingSessionFilter,
    ModelPerformanceFilter,
)
from tests.containers.postgres_container import get_postgres_container


class TestMLRepositoryFacadeIntegration:
    """Integration tests for ML Repository Facade with real ML operations."""

    @pytest.fixture(scope="class")
    def event_loop(self):
        """Create an event loop for the test class."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    async def postgres_container(self):
        """Provide a real PostgreSQL container for testing."""
        container = get_postgres_container()
        try:
            yield container
        finally:
            container.stop()

    @pytest.fixture(scope="class")
    async def db_services(self, postgres_container):
        """Provide DatabaseServices with real PostgreSQL connection."""
        connection_string = postgres_container.get_connection_url()
        db_services = DatabaseServices(connection_string)
        
        try:
            await db_services.initialize()
            yield db_services
        finally:
            await db_services.cleanup()

    @pytest.fixture
    async def ml_repository(self, db_services):
        """Provide ML Repository Facade instance."""
        repository = MLRepositoryFacade(db_services)
        return repository

    async def test_facade_initialization(self, ml_repository):
        """Test that facade initializes all specialized repositories correctly."""
        # Test that all specialized repositories are initialized
        assert ml_repository.model_repository is not None
        assert ml_repository.training_repository is not None
        assert ml_repository.metrics_repository is not None
        assert ml_repository.experiment_repository is not None
        assert ml_repository.inference_repository is not None
        
        # Test facade health check
        health_status = await ml_repository.get_repository_health_status()
        assert health_status["facade_status"] in ["healthy", "degraded"]
        assert health_status["total_repositories"] == 5
        assert "repositories" in health_status

    async def test_training_session_workflow(self, ml_repository):
        """Test complete training session workflow through facade."""
        # Create training session
        session_data = TrainingSessionCreate(
            name="test_training_session",
            status="running",
            continuous_mode=True,
            current_performance=0.75,
            best_performance=0.80,
            efficiency_score=0.85,
            resource_utilization={"cpu_percent": 45.2, "memory_percent": 62.1},
        )
        
        training_session = await ml_repository.create_training_session(session_data)
        assert training_session.id is not None
        assert training_session.name == "test_training_session"
        assert training_session.status == "running"
        
        # Get training session by ID
        retrieved_session = await ml_repository.get_training_session_by_id(training_session.id)
        assert retrieved_session is not None
        assert retrieved_session.id == training_session.id
        
        # Create training iterations
        iteration_data = {
            "session_id": training_session.id,
            "iteration_number": 1,
            "performance_score": 0.72,
            "duration_seconds": 120,
        }
        
        iteration = await ml_repository.create_training_iteration(iteration_data)
        assert iteration.session_id == training_session.id
        assert iteration.iteration_number == 1
        
        # Get training metrics
        metrics = await ml_repository.get_training_session_metrics(training_session.id)
        assert metrics is not None
        assert metrics.session_id == training_session.id
        assert metrics.total_iterations == 1

    async def test_model_performance_tracking(self, ml_repository):
        """Test model performance tracking through facade."""
        # Create model performance record
        performance_data = {
            "model_id": "test_model_001",
            "model_type": "classification",
            "model_version": "v1.0",
            "accuracy": 0.892,
            "precision": 0.878,
            "recall": 0.901,
            "f1_score": 0.889,
            "training_samples": 10000,
            "metadata": {"experiment_id": "exp_001"},
        }
        
        performance = await ml_repository.create_model_performance(performance_data)
        assert performance.model_id == "test_model_001"
        assert performance.accuracy == 0.892
        
        # Get model performance history
        history = await ml_repository.get_model_performance_by_id("test_model_001")
        assert len(history) == 1
        assert history[0].model_id == "test_model_001"
        
        # Get best performing models
        best_models = await ml_repository.get_best_performing_models(metric="accuracy", limit=5)
        assert len(best_models) >= 1
        assert any(model.model_id == "test_model_001" for model in best_models)
        
        # Get model version history
        version_history = await ml_repository.get_model_version_history("test_model_001")
        assert len(version_history) >= 1
        assert version_history[0].model_id == "test_model_001"

    async def test_generation_session_workflow(self, ml_repository):
        """Test generation session and experiment workflow."""
        # Create generation session
        session_data = {
            "session_type": "synthetic_data",
            "generation_method": "gpt_augmentation",
            "status": "running",
            "quality_threshold": 0.8,
            "metadata": {"experiment_type": "data_augmentation"},
        }
        
        generation_session = await ml_repository.create_generation_session(session_data)
        assert generation_session.id is not None
        assert generation_session.session_type == "synthetic_data"
        
        # Create generation batch
        batch_data = {
            "session_id": generation_session.id,
            "batch_size": 100,
            "status": "completed",
            "samples_generated": 95,
            "avg_quality_score": 0.83,
            "generation_time_ms": 5200,
        }
        
        batch = await ml_repository.create_generation_batch(batch_data)
        assert batch.session_id == generation_session.id
        assert batch.samples_generated == 95
        
        # Get generation batches
        batches = await ml_repository.get_generation_batches(generation_session.id)
        assert len(batches) == 1
        assert batches[0].id == batch.id

    async def test_synthetic_data_operations(self, ml_repository):
        """Test synthetic data sample operations through facade."""
        # Create generation session first
        session_data = {
            "session_type": "synthetic_data",
            "generation_method": "transformer_based",
            "status": "completed",
            "quality_threshold": 0.7,
        }
        
        generation_session = await ml_repository.create_generation_session(session_data)
        
        # Create synthetic data samples
        samples_data = [
            {
                "session_id": generation_session.id,
                "synthetic_data": "Generated sample 1",
                "quality_score": 0.85,
                "generation_time_ms": 250,
                "domain_category": "text_classification",
                "status": "active",
            },
            {
                "session_id": generation_session.id,
                "synthetic_data": "Generated sample 2", 
                "quality_score": 0.78,
                "generation_time_ms": 310,
                "domain_category": "text_classification",
                "status": "active",
            }
        ]
        
        samples = await ml_repository.create_synthetic_data_samples(samples_data)
        assert len(samples) == 2
        assert all(sample.session_id == generation_session.id for sample in samples)
        
        # Get synthetic data samples with filters
        retrieved_samples = await ml_repository.get_synthetic_data_samples(
            session_id=generation_session.id,
            min_quality_score=0.8,
        )
        assert len(retrieved_samples) == 1
        assert retrieved_samples[0].quality_score >= 0.8
        
        # Archive samples
        sample_ids = [sample.id for sample in samples]
        archived_count = await ml_repository.archive_synthetic_samples(sample_ids)
        assert archived_count == 2

    async def test_analytics_and_metrics(self, ml_repository):
        """Test analytics and metrics functionality through facade."""
        # Create training data for analytics
        session_data = TrainingSessionCreate(
            name="analytics_test_session",
            status="completed",
            continuous_mode=False,
            current_performance=0.88,
            best_performance=0.91,
        )
        
        training_session = await ml_repository.create_training_session(session_data)
        
        # Get training analytics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        analytics = await ml_repository.get_training_analytics(start_date, end_date)
        assert "training_sessions" in analytics
        assert analytics["training_sessions"]["total"] >= 1

    async def test_cleanup_operations(self, ml_repository):
        """Test cleanup and maintenance operations."""
        # Create training session with iterations
        session_data = TrainingSessionCreate(
            name="cleanup_test_session",
            status="running",
            continuous_mode=True,
        )
        
        training_session = await ml_repository.create_training_session(session_data)
        
        # Create multiple iterations
        for i in range(5):
            iteration_data = {
                "session_id": training_session.id,
                "iteration_number": i + 1,
                "performance_score": 0.7 + i * 0.02,
                "duration_seconds": 100 + i * 10,
            }
            await ml_repository.create_training_iteration(iteration_data)
        
        # Test cleanup - keep only latest 3 iterations
        deleted_count = await ml_repository.cleanup_old_iterations(
            training_session.id, keep_latest=3
        )
        assert deleted_count == 2
        
        # Verify only 3 iterations remain
        remaining_iterations = await ml_repository.get_training_iterations(training_session.id)
        assert len(remaining_iterations) == 3

    async def test_intelligence_processing(self, ml_repository):
        """Test intelligence processing and caching functionality."""
        # Test batch processing capabilities
        batch_data = [
            {
                "rule_id": "test_rule_001",
                "effectiveness_ratio": 0.75,
                "confidence_score": 0.82,
                "usage_count": 150,
            },
            {
                "rule_id": "test_rule_002", 
                "effectiveness_ratio": 0.68,
                "confidence_score": 0.79,
                "usage_count": 89,
            }
        ]
        
        predictions = await ml_repository.process_ml_predictions_batch(batch_data)
        assert len(predictions) == 2
        assert all("predicted_effectiveness" in pred for pred in predictions)
        assert all("prediction_confidence" in pred for pred in predictions)
        
        # Test cache cleanup
        cache_result = await ml_repository.cleanup_expired_cache()
        assert "cache_cleaned" in cache_result
        assert isinstance(cache_result["cache_cleaned"], int)

    async def test_cross_repository_integration(self, ml_repository):
        """Test integration across multiple repository domains."""
        # Create training session
        session_data = TrainingSessionCreate(
            name="integration_test_session",
            status="completed",
            current_performance=0.89,
        )
        training_session = await ml_repository.create_training_session(session_data)
        
        # Create model performance linked to training
        performance_data = {
            "model_id": f"model_from_session_{training_session.id}",
            "model_type": "integration_test",
            "accuracy": 0.89,
            "training_samples": 5000,
            "metadata": {"training_session_id": training_session.id},
        }
        model_performance = await ml_repository.create_model_performance(performance_data)
        
        # Create generation session for synthetic data
        generation_data = {
            "session_type": "model_augmentation",
            "training_session_id": training_session.id,
            "status": "completed",
            "quality_threshold": 0.8,
        }
        generation_session = await ml_repository.create_generation_session(generation_data)
        
        # Verify cross-repository data integrity
        assert training_session.id is not None
        assert model_performance.metadata["training_session_id"] == training_session.id
        assert generation_session.training_session_id == training_session.id
        
        # Test comprehensive metrics that span repositories
        metrics = await ml_repository.get_training_session_metrics(training_session.id)
        synthetic_metrics = await ml_repository.get_synthetic_data_metrics(generation_session.id)
        
        assert metrics is not None
        assert metrics.session_id == training_session.id
        # synthetic_metrics might be None if no samples exist, which is acceptable

    async def test_error_handling_and_resilience(self, ml_repository):
        """Test error handling and resilience of facade operations."""
        # Test with non-existent session ID
        non_existent_session = await ml_repository.get_training_session_by_id("non_existent_id")
        assert non_existent_session is None
        
        # Test with invalid model performance data
        with pytest.raises(Exception):
            await ml_repository.create_model_performance({})  # Missing required fields
        
        # Test metrics for non-existent session
        metrics = await ml_repository.get_training_session_metrics("non_existent_id")
        assert metrics is None
        
        # Test cleanup with non-existent session
        deleted_count = await ml_repository.cleanup_old_iterations("non_existent_id")
        assert deleted_count == 0

    async def test_performance_with_realistic_data_volumes(self, ml_repository):
        """Test performance with realistic ML data volumes."""
        # Create training session
        session_data = TrainingSessionCreate(
            name="performance_test_session",
            status="running",
            current_performance=0.82,
        )
        training_session = await ml_repository.create_training_session(session_data)
        
        # Create multiple training iterations (simulating long training)
        iterations_count = 50
        start_time = datetime.now()
        
        for i in range(iterations_count):
            iteration_data = {
                "session_id": training_session.id,
                "iteration_number": i + 1,
                "performance_score": 0.5 + (i / iterations_count) * 0.4,  # Gradual improvement
                "duration_seconds": 60 + (i % 20),  # Variable duration
            }
            await ml_repository.create_training_iteration(iteration_data)
        
        creation_time = datetime.now() - start_time
        
        # Test retrieval performance
        start_time = datetime.now()
        iterations = await ml_repository.get_training_iterations(training_session.id)
        retrieval_time = datetime.now() - start_time
        
        # Verify all iterations were created and retrieved
        assert len(iterations) == iterations_count
        
        # Performance assertions (reasonable times for 50 iterations)
        assert creation_time.total_seconds() < 30  # Should complete within 30 seconds
        assert retrieval_time.total_seconds() < 5   # Should retrieve within 5 seconds
        
        # Test performance trend calculation
        start_time = datetime.now()
        trend = await ml_repository.get_iteration_performance_trend(training_session.id)
        trend_time = datetime.now() - start_time
        
        assert len(trend) == iterations_count
        assert trend_time.total_seconds() < 5  # Should calculate trend within 5 seconds
        
        # Verify trend shows improvement
        first_score = trend[0]["performance_score"]
        last_score = trend[-1]["performance_score"]
        assert last_score > first_score  # Performance should improve over iterations

    async def test_data_consistency_across_repositories(self, ml_repository):
        """Test data consistency and referential integrity across repositories."""
        # Create a complete ML workflow with cross-repository references
        
        # 1. Training Session
        session_data = TrainingSessionCreate(
            name="consistency_test_session",
            status="completed",
            current_performance=0.87,
            best_performance=0.92,
        )
        training_session = await ml_repository.create_training_session(session_data)
        
        # 2. Training Iterations
        iteration_data = {
            "session_id": training_session.id,
            "iteration_number": 1,
            "performance_score": 0.87,
            "duration_seconds": 150,
        }
        iteration = await ml_repository.create_training_iteration(iteration_data)
        
        # 3. Model Performance (linked to training)
        performance_data = {
            "model_id": f"consistency_model_{training_session.id}",
            "model_type": "consistency_test",
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "training_samples": 8000,
            "metadata": {"source_session": training_session.id},
        }
        model_performance = await ml_repository.create_model_performance(performance_data)
        
        # 4. Generation Session (for synthetic data)
        generation_data = {
            "session_type": "data_augmentation",
            "training_session_id": training_session.id,
            "status": "completed",
            "quality_threshold": 0.8,
            "metadata": {"source_model": model_performance.model_id},
        }
        generation_session = await ml_repository.create_generation_session(generation_data)
        
        # 5. Synthetic Data Samples
        samples_data = [
            {
                "session_id": generation_session.id,
                "synthetic_data": f"Consistency test sample {i}",
                "quality_score": 0.8 + (i * 0.05),
                "generation_time_ms": 200 + (i * 50),
                "domain_category": "consistency_test",
                "status": "active",
            }
            for i in range(5)
        ]
        samples = await ml_repository.create_synthetic_data_samples(samples_data)
        
        # Verify all cross-references are maintained
        
        # Check training session exists and has correct data
        retrieved_session = await ml_repository.get_training_session_by_id(training_session.id)
        assert retrieved_session.id == training_session.id
        assert retrieved_session.current_performance == 0.87
        
        # Check iterations are linked to session
        iterations = await ml_repository.get_training_iterations(training_session.id)
        assert len(iterations) == 1
        assert iterations[0].session_id == training_session.id
        
        # Check model performance references training session
        model_history = await ml_repository.get_model_performance_by_id(model_performance.model_id)
        assert len(model_history) == 1
        assert model_history[0].metadata["source_session"] == training_session.id
        
        # Check generation session references training session
        retrieved_generation = await ml_repository.get_generation_session_by_id(generation_session.id)
        assert retrieved_generation.training_session_id == training_session.id
        assert retrieved_generation.metadata["source_model"] == model_performance.model_id
        
        # Check synthetic samples are linked to generation session
        retrieved_samples = await ml_repository.get_synthetic_data_samples(
            session_id=generation_session.id
        )
        assert len(retrieved_samples) == 5
        assert all(sample.session_id == generation_session.id for sample in retrieved_samples)
        
        # Verify metrics calculation includes all related data
        training_metrics = await ml_repository.get_training_session_metrics(training_session.id)
        assert training_metrics is not None
        assert training_metrics.total_iterations == 1
        assert training_metrics.current_performance == 0.87
        
        synthetic_metrics = await ml_repository.get_synthetic_data_metrics(generation_session.id)
        if synthetic_metrics:  # May be None depending on implementation
            assert synthetic_metrics.session_id == generation_session.id
            assert synthetic_metrics.total_samples == 5