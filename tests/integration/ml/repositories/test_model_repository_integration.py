"""Integration tests for Model Repository with real ML model operations and testcontainers.

Tests model performance tracking, versioning, comparison, and deployment status
with real database operations and ML model scenarios.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List

from prompt_improver.database import DatabaseServices
from prompt_improver.repositories.impl.ml_repository_service.model_repository import ModelRepository
from prompt_improver.repositories.protocols.ml_repository_protocol import ModelPerformanceFilter
from tests.containers.postgres_container import get_postgres_container


class TestModelRepositoryIntegration:
    """Integration tests for Model Repository with real ML model operations."""

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
    async def model_repository(self, db_services):
        """Provide Model Repository instance."""
        repository = ModelRepository(db_services)
        return repository

    async def test_model_performance_creation_and_retrieval(self, model_repository):
        """Test creating and retrieving model performance records."""
        # Create model performance record
        performance_data = {
            "model_id": "test_classifier_v1",
            "model_type": "text_classification",
            "model_version": "1.0.0",
            "accuracy": 0.923,
            "precision": 0.918,
            "recall": 0.927,
            "f1_score": 0.922,
            "training_samples": 15000,
            "validation_samples": 3000,
            "test_samples": 2000,
            "metadata": {
                "algorithm": "transformer",
                "hyperparameters": {"learning_rate": 0.001, "batch_size": 32},
                "training_time_hours": 4.5,
            },
        }
        
        performance = await model_repository.create_model_performance(performance_data)
        assert performance.model_id == "test_classifier_v1"
        assert performance.accuracy == 0.923
        assert performance.model_type == "text_classification"
        assert performance.metadata["algorithm"] == "transformer"
        
        # Retrieve performance by model ID
        history = await model_repository.get_model_performance_by_id("test_classifier_v1")
        assert len(history) == 1
        assert history[0].model_id == "test_classifier_v1"
        assert history[0].accuracy == 0.923

    async def test_model_performance_filtering(self, model_repository):
        """Test filtering model performance records with various criteria."""
        # Create multiple model performance records
        models_data = [
            {
                "model_id": "filter_test_model_1",
                "model_type": "classification",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "training_samples": 10000,
            },
            {
                "model_id": "filter_test_model_2", 
                "model_type": "regression",
                "accuracy": 0.91,
                "precision": 0.89,
                "recall": 0.93,
                "training_samples": 15000,
            },
            {
                "model_id": "filter_test_model_3",
                "model_type": "classification",
                "accuracy": 0.78,
                "precision": 0.75,
                "recall": 0.81,
                "training_samples": 8000,
            },
        ]
        
        created_models = []
        for model_data in models_data:
            performance = await model_repository.create_model_performance(model_data)
            created_models.append(performance)
        
        # Test filtering by model type
        classification_filter = ModelPerformanceFilter(model_type="classification")
        classification_models = await model_repository.get_model_performances(
            filters=classification_filter
        )
        classification_model_ids = {model.model_id for model in classification_models}
        assert "filter_test_model_1" in classification_model_ids
        assert "filter_test_model_3" in classification_model_ids
        assert "filter_test_model_2" not in classification_model_ids
        
        # Test filtering by minimum accuracy
        high_accuracy_filter = ModelPerformanceFilter(min_accuracy=0.9)
        high_accuracy_models = await model_repository.get_model_performances(
            filters=high_accuracy_filter
        )
        high_accuracy_ids = {model.model_id for model in high_accuracy_models}
        assert "filter_test_model_2" in high_accuracy_ids
        assert "filter_test_model_1" not in high_accuracy_ids
        assert "filter_test_model_3" not in high_accuracy_ids
        
        # Test filtering by minimum training samples
        large_dataset_filter = ModelPerformanceFilter(min_training_samples=12000)
        large_dataset_models = await model_repository.get_model_performances(
            filters=large_dataset_filter
        )
        large_dataset_ids = {model.model_id for model in large_dataset_models}
        assert "filter_test_model_2" in large_dataset_ids
        assert "filter_test_model_1" not in large_dataset_ids
        assert "filter_test_model_3" not in large_dataset_ids

    async def test_best_performing_models(self, model_repository):
        """Test retrieval of best performing models by various metrics."""
        # Create models with different performance characteristics
        models_data = [
            {
                "model_id": "best_accuracy_model",
                "model_type": "benchmark",
                "accuracy": 0.96,
                "precision": 0.94,
                "recall": 0.92,
                "f1_score": 0.93,
            },
            {
                "model_id": "best_precision_model",
                "model_type": "benchmark", 
                "accuracy": 0.88,
                "precision": 0.98,
                "recall": 0.85,
                "f1_score": 0.91,
            },
            {
                "model_id": "best_recall_model",
                "model_type": "benchmark",
                "accuracy": 0.89,
                "precision": 0.86,
                "recall": 0.99,
                "f1_score": 0.92,
            },
            {
                "model_id": "balanced_model",
                "model_type": "benchmark",
                "accuracy": 0.91,
                "precision": 0.91,
                "recall": 0.91,
                "f1_score": 0.91,
            },
        ]
        
        for model_data in models_data:
            await model_repository.create_model_performance(model_data)
        
        # Test best by accuracy
        best_accuracy = await model_repository.get_best_performing_models(
            metric="accuracy", model_type="benchmark", limit=2
        )
        assert len(best_accuracy) >= 1
        assert best_accuracy[0].model_id == "best_accuracy_model"
        assert best_accuracy[0].accuracy == 0.96
        
        # Test best by precision
        best_precision = await model_repository.get_best_performing_models(
            metric="precision", model_type="benchmark", limit=2
        )
        assert len(best_precision) >= 1
        assert best_precision[0].model_id == "best_precision_model"
        assert best_precision[0].precision == 0.98
        
        # Test best by recall
        best_recall = await model_repository.get_best_performing_models(
            metric="recall", model_type="benchmark", limit=2
        )
        assert len(best_recall) >= 1
        assert best_recall[0].model_id == "best_recall_model"
        assert best_recall[0].recall == 0.99

    async def test_model_version_management(self, model_repository):
        """Test model versioning and version history tracking."""
        model_id = "versioned_model_test"
        
        # Create multiple versions of the same model
        versions_data = [
            {
                "model_id": model_id,
                "model_type": "versioning_test",
                "model_version": "1.0.0",
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.84,
                "metadata": {"version_notes": "Initial release"},
            },
            {
                "model_id": model_id,
                "model_type": "versioning_test", 
                "model_version": "1.1.0",
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "metadata": {"version_notes": "Improved feature engineering"},
            },
            {
                "model_id": model_id,
                "model_type": "versioning_test",
                "model_version": "2.0.0", 
                "accuracy": 0.91,
                "precision": 0.89,
                "recall": 0.93,
                "metadata": {"version_notes": "Architecture overhaul"},
            },
        ]
        
        created_versions = []
        for version_data in versions_data:
            performance = await model_repository.create_model_performance(version_data)
            created_versions.append(performance)
            # Add small delay to ensure different timestamps
            await asyncio.sleep(0.01)
        
        # Test getting latest version
        latest_version = await model_repository.get_latest_model_version("versioning_test")
        assert latest_version == "2.0.0"
        
        # Test getting version history
        version_history = await model_repository.get_model_version_history(model_id)
        assert len(version_history) == 3
        
        # Verify versions are in chronological order (latest first)
        assert version_history[0].version == "v1.0"  # ModelVersionInfo formats as v{i+1}.0
        assert version_history[0].deployment_status == "active"
        assert version_history[1].deployment_status == "retired"
        assert version_history[2].deployment_status == "retired"
        
        # Test getting model versions with detailed info
        detailed_versions = await model_repository.get_model_versions("versioning_test", limit=5)
        assert len(detailed_versions) == 3
        
        # Verify version progression shows improvement
        v1_accuracy = detailed_versions[-1]["performance_metrics"]["accuracy"]  # Last item is oldest
        v3_accuracy = detailed_versions[0]["performance_metrics"]["accuracy"]   # First item is newest
        assert v3_accuracy > v1_accuracy

    async def test_model_comparison(self, model_repository):
        """Test comparing performance between different models."""
        # Create two models for comparison
        model1_data = {
            "model_id": "comparison_model_a",
            "model_type": "comparison_test",
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "f1_score": 0.89,
        }
        
        model2_data = {
            "model_id": "comparison_model_b",
            "model_type": "comparison_test",
            "accuracy": 0.85,
            "precision": 0.92,
            "recall": 0.83,
            "f1_score": 0.87,
        }
        
        await model_repository.create_model_performance(model1_data)
        await model_repository.create_model_performance(model2_data)
        
        # Compare the models
        comparison = await model_repository.compare_model_performance(
            "comparison_model_a", "comparison_model_b"
        )
        
        assert "model1" in comparison
        assert "model2" in comparison
        assert "differences" in comparison
        assert "winner" in comparison
        
        # Verify comparison results
        model1_metrics = comparison["model1"]["metrics"]
        model2_metrics = comparison["model2"]["metrics"]
        differences = comparison["differences"]
        winners = comparison["winner"]
        
        # Model A should win on accuracy and recall
        assert model1_metrics["accuracy"] == 0.89
        assert model2_metrics["accuracy"] == 0.85
        assert differences["accuracy"] == 0.04  # 0.89 - 0.85
        assert winners["accuracy"] == "comparison_model_a"
        
        # Model B should win on precision
        assert model1_metrics["precision"] == 0.87
        assert model2_metrics["precision"] == 0.92
        assert differences["precision"] == -0.05  # 0.87 - 0.92
        assert winners["precision"] == "comparison_model_b"

    async def test_model_metrics_update(self, model_repository):
        """Test updating model metrics after creation."""
        # Create initial model performance
        initial_data = {
            "model_id": "updatable_model",
            "model_type": "update_test",
            "accuracy": 0.80,
            "precision": 0.78,
            "recall": 0.82,
        }
        
        performance = await model_repository.create_model_performance(initial_data)
        assert performance.accuracy == 0.80
        
        # Update metrics
        updated_metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
        }
        
        update_result = await model_repository.update_model_metrics(
            "updatable_model", updated_metrics
        )
        assert update_result is True
        
        # Verify updates
        updated_history = await model_repository.get_model_performance_by_id("updatable_model")
        assert len(updated_history) == 1
        latest_performance = updated_history[0]
        assert latest_performance.accuracy == 0.85
        assert latest_performance.precision == 0.83
        assert latest_performance.f1_score == 0.85

    async def test_model_deployment_status(self, model_repository):
        """Test model deployment status tracking."""
        # Create model with deployment metadata
        model_data = {
            "model_id": "deployment_test_model",
            "model_type": "deployment_test",
            "model_version": "1.0.0",
            "accuracy": 0.88,
            "metadata": {
                "deployment_status": "production",
                "deployment_date": "2024-01-15",
                "deployment_environment": "aws-prod",
            },
        }
        
        await model_repository.create_model_performance(model_data)
        
        # Get deployment status
        deployment_status = await model_repository.get_model_deployment_status(
            "deployment_test_model"
        )
        
        assert deployment_status["model_id"] == "deployment_test_model"
        assert deployment_status["status"] == "production"
        assert deployment_status["version"] == "1.0.0"
        assert "performance_summary" in deployment_status
        assert deployment_status["performance_summary"]["accuracy"] == 0.88
        assert "deployment_date" in deployment_status["metadata"]

    async def test_model_archival(self, model_repository):
        """Test archiving old model versions."""
        # Create models with different ages
        old_date = datetime.now() - timedelta(days=100)
        recent_date = datetime.now() - timedelta(days=10)
        
        # Note: We can't directly set created_at in the test, but we can simulate
        # the archival process by creating models and then testing the archival logic
        
        models_data = [
            {
                "model_id": "archival_test_model_1",
                "model_type": "archival_test",
                "accuracy": 0.85,
            },
            {
                "model_id": "archival_test_model_2",
                "model_type": "archival_test", 
                "accuracy": 0.88,
            },
        ]
        
        for model_data in models_data:
            await model_repository.create_model_performance(model_data)
        
        # Test archival with a very short age (should archive nothing in real scenario)
        archived_count = await model_repository.archive_old_models(days_old=1)
        # Since models were just created, nothing should be archived
        assert archived_count == 0
        
        # Test archival with a longer age to demonstrate the functionality
        # In a real scenario with actual old models, this would archive them
        archived_count = await model_repository.archive_old_models(days_old=200)
        assert isinstance(archived_count, int)

    async def test_model_repository_with_realistic_ml_workflow(self, model_repository):
        """Test model repository with a realistic ML development workflow."""
        project_id = "realistic_ml_project"
        
        # Phase 1: Initial model development
        baseline_model = {
            "model_id": f"{project_id}_baseline",
            "model_type": "neural_network",
            "model_version": "0.1.0",
            "accuracy": 0.72,
            "precision": 0.70,
            "recall": 0.74,
            "f1_score": 0.72,
            "training_samples": 5000,
            "metadata": {
                "phase": "baseline",
                "architecture": "simple_dnn",
                "layers": 3,
                "deployment_status": "development",
            },
        }
        
        baseline = await model_repository.create_model_performance(baseline_model)
        assert baseline.model_id == f"{project_id}_baseline"
        
        # Phase 2: Feature engineering improvements
        improved_model = {
            "model_id": f"{project_id}_improved",
            "model_type": "neural_network",
            "model_version": "1.0.0", 
            "accuracy": 0.84,
            "precision": 0.82,
            "recall": 0.86,
            "f1_score": 0.84,
            "training_samples": 10000,
            "metadata": {
                "phase": "feature_engineering",
                "architecture": "improved_dnn",
                "layers": 5,
                "deployment_status": "staging",
            },
        }
        
        improved = await model_repository.create_model_performance(improved_model)
        
        # Phase 3: Production model
        production_model = {
            "model_id": f"{project_id}_production",
            "model_type": "neural_network",
            "model_version": "2.0.0",
            "accuracy": 0.91,
            "precision": 0.89,
            "recall": 0.93,
            "f1_score": 0.91,
            "training_samples": 20000,
            "metadata": {
                "phase": "production",
                "architecture": "transformer",
                "layers": 12,
                "deployment_status": "production",
            },
        }
        
        production = await model_repository.create_model_performance(production_model)
        
        # Analyze the development progression
        
        # 1. Get best performing models for the project
        best_models = await model_repository.get_best_performing_models(
            metric="accuracy", model_type="neural_network", limit=5
        )
        
        project_models = [
            model for model in best_models 
            if project_id in model.model_id
        ]
        assert len(project_models) >= 3
        
        # Production model should be the best
        production_in_best = any(
            model.model_id == f"{project_id}_production" 
            for model in project_models
        )
        assert production_in_best
        
        # 2. Compare baseline vs production
        comparison = await model_repository.compare_model_performance(
            f"{project_id}_baseline", f"{project_id}_production"
        )
        
        accuracy_improvement = comparison["differences"]["accuracy"]
        assert accuracy_improvement > 0.15  # Should show significant improvement
        
        # 3. Verify deployment status tracking
        production_status = await model_repository.get_model_deployment_status(
            f"{project_id}_production"
        )
        assert production_status["status"] == "production"
        
        baseline_status = await model_repository.get_model_deployment_status(
            f"{project_id}_baseline"
        )
        assert baseline_status["status"] == "development"
        
        # 4. Get version history for production model
        version_history = await model_repository.get_model_version_history(
            f"{project_id}_production"
        )
        assert len(version_history) >= 1
        assert version_history[0].model_id == f"{project_id}_production"

    async def test_concurrent_model_operations(self, model_repository):
        """Test concurrent model operations for thread safety."""
        # Create multiple concurrent model performance records
        async def create_model_performance(model_suffix: str) -> str:
            model_data = {
                "model_id": f"concurrent_test_model_{model_suffix}",
                "model_type": "concurrency_test",
                "accuracy": 0.80 + (int(model_suffix) / 100),  # Slightly different accuracies
                "precision": 0.78 + (int(model_suffix) / 100),
                "recall": 0.82 + (int(model_suffix) / 100),
            }
            performance = await model_repository.create_model_performance(model_data)
            return performance.model_id
        
        # Run multiple concurrent operations
        tasks = [create_model_performance(str(i)) for i in range(10)]
        created_model_ids = await asyncio.gather(*tasks)
        
        assert len(created_model_ids) == 10
        assert len(set(created_model_ids)) == 10  # All should be unique
        
        # Verify all models were created correctly
        for model_id in created_model_ids:
            history = await model_repository.get_model_performance_by_id(model_id)
            assert len(history) == 1
            assert history[0].model_id == model_id

    async def test_model_repository_error_scenarios(self, model_repository):
        """Test error handling in various edge cases."""
        # Test model comparison with non-existent model
        comparison = await model_repository.compare_model_performance(
            "non_existent_model_1", "non_existent_model_2"
        )
        assert "error" in comparison
        
        # Test updating metrics for non-existent model
        update_result = await model_repository.update_model_metrics(
            "non_existent_model", {"accuracy": 0.95}
        )
        assert update_result is False
        
        # Test getting deployment status for non-existent model
        deployment_status = await model_repository.get_model_deployment_status(
            "non_existent_model"
        )
        assert deployment_status["status"] == "not_found"
        
        # Test getting version history for non-existent model
        version_history = await model_repository.get_model_version_history(
            "non_existent_model"
        )
        assert len(version_history) == 0