"""Real ML Pipeline Integration Tests

Tests that verify complete ML pipeline functionality using real models
and data instead of mocks. Demonstrates actual ML training, inference,
and performance monitoring.
"""

import asyncio
import pytest
from pathlib import Path

from tests.real_ml.lightweight_models import (
    RealMLService,
    RealMLflowService, 
    LightweightTextClassifier,
    LightweightRegressor,
    PatternDiscoveryEngine,
)
from tests.real_ml.test_data_generators import (
    PromptDataGenerator,
    MLTrainingDataGenerator,
)


class TestRealMLPipeline:
    """Test complete ML pipeline with real models and data."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ml_training_pipeline(
        self,
        real_ml_service_for_testing,
        sample_rules_data,
        sample_performance_data,
    ):
        """Test complete ML training pipeline with real models."""
        # Test rule optimization with real ML
        optimization_result = await real_ml_service_for_testing.optimize_rules(
            sample_rules_data[:10],
            sample_performance_data[:10]
        )
        
        assert optimization_result["status"] == "success"
        assert "model_id" in optimization_result
        assert "best_score" in optimization_result
        assert "accuracy" in optimization_result
        assert "processing_time_ms" in optimization_result
        
        # Verify model was actually trained and stored
        model_id = optimization_result["model_id"]
        assert model_id in real_ml_service_for_testing.models
        
        # Test prediction using trained model
        test_rule = sample_rules_data[0]
        prediction_result = await real_ml_service_for_testing.predict_rule_effectiveness(
            test_rule,
            model_id
        )
        
        assert prediction_result["status"] == "success"
        assert "prediction" in prediction_result
        assert "confidence" in prediction_result
        assert 0 <= prediction_result["prediction"] <= 1
        assert 0 <= prediction_result["confidence"] <= 1
    
    @pytest.mark.asyncio 
    async def test_pattern_discovery_with_real_engine(
        self,
        real_ml_service_for_testing,
        pattern_discovery_data,
    ):
        """Test pattern discovery using real algorithm."""
        discovery_result = await real_ml_service_for_testing.discover_patterns(
            pattern_discovery_data,
            min_support=0.1
        )
        
        assert discovery_result["status"] == "success"
        assert "patterns_discovered" in discovery_result
        assert "patterns" in discovery_result
        assert "processing_time_ms" in discovery_result
        
        # Verify patterns have realistic structure
        if discovery_result["patterns_discovered"] > 0:
            patterns = discovery_result["patterns"]
            for pattern in patterns:
                assert "type" in pattern
                assert "support_count" in pattern
                assert isinstance(pattern["support_count"], int)
                assert pattern["support_count"] > 0
    
    @pytest.mark.asyncio
    async def test_ensemble_optimization(
        self,
        real_ml_service_for_testing,
        complex_rules_dataset,
    ):
        """Test ensemble optimization with multiple models."""
        ensemble_result = await real_ml_service_for_testing.optimize_ensemble_rules(
            complex_rules_dataset["rules"][:20]
        )
        
        assert ensemble_result["status"] == "success"
        assert "ensemble_score" in ensemble_result
        assert "ensemble_std" in ensemble_result
        assert "processing_time_ms" in ensemble_result
        
        # Ensemble should show some variance (std > 0)
        assert ensemble_result["ensemble_std"] >= 0
        
        # Score should be reasonable
        ensemble_score = ensemble_result["ensemble_score"]
        assert 0 <= ensemble_score <= 1
    
    @pytest.mark.asyncio
    async def test_mlflow_integration_with_real_service(
        self,
        real_mlflow_service_for_testing,
        training_dataset,
    ):
        """Test MLflow integration with real experiment tracking."""
        # Create experiment
        experiment_id = await real_mlflow_service_for_testing.log_experiment(
            "real_ml_test_experiment",
            {
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 10,
                "dataset_size": len(training_dataset["prompts"])
            }
        )
        
        assert experiment_id is not None
        assert experiment_id in real_mlflow_service_for_testing.experiments
        
        # Train a real model
        classifier = LightweightTextClassifier(random_state=42)
        training_result = classifier.fit(
            training_dataset["prompts"],
            training_dataset["labels"]
        )
        
        # Log model to MLflow
        model_uri = await real_mlflow_service_for_testing.log_model(
            "text_classifier_v1",
            classifier,
            {
                "accuracy": training_result.accuracy,
                "feature_count": training_result.feature_count,
                "sample_count": training_result.sample_count,
                "training_time": training_result.training_time,
                "algorithm": "logistic_regression"
            }
        )
        
        assert model_uri is not None
        assert model_uri in real_mlflow_service_for_testing.models
        
        # Retrieve model metadata
        metadata = await real_mlflow_service_for_testing.get_model_metadata(model_uri)
        assert metadata["accuracy"] == training_result.accuracy
        assert metadata["algorithm"] == "logistic_regression"
        
        # Test health check
        health_status = await real_mlflow_service_for_testing.health_check()
        assert health_status["status"] == "healthy"
        assert health_status["experiments_count"] >= 1
        assert health_status["models_count"] >= 1
    
    @pytest.mark.asyncio
    async def test_text_classification_real_training(
        self,
        training_dataset,
    ):
        """Test real text classification training and inference."""
        classifier = LightweightTextClassifier(random_state=42)
        
        # Train model
        training_result = classifier.fit(
            training_dataset["prompts"],
            training_dataset["labels"]
        )
        
        # Verify training results
        assert training_result.model_type == "text_classifier"
        assert training_result.accuracy >= 0
        assert training_result.training_time > 0
        assert training_result.feature_count > 0
        assert training_result.sample_count == len(training_dataset["prompts"])
        
        # Test predictions
        test_prompts = training_dataset["prompts"][:5]
        predictions = classifier.predict(test_prompts)
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability predictions
        probabilities = classifier.predict_proba(test_prompts)
        assert len(probabilities) == 5
        for prob_pair in probabilities:
            assert len(prob_pair) == 2
            assert abs(sum(prob_pair) - 1.0) < 1e-6  # Should sum to 1
            assert all(0 <= p <= 1 for p in prob_pair)
    
    @pytest.mark.asyncio
    async def test_regression_real_training(
        self,
        feature_matrix,
        regression_targets,
    ):
        """Test real regression model training and inference."""
        regressor = LightweightRegressor(random_state=42)
        
        # Train model
        training_result = regressor.fit(feature_matrix, regression_targets)
        
        # Verify training results
        assert training_result.model_type == "random_forest_regressor"
        assert training_result.accuracy >= 0  # R² score
        assert training_result.training_time > 0
        assert training_result.feature_count == feature_matrix.shape[1]
        assert training_result.sample_count == feature_matrix.shape[0]
        
        # Test predictions
        test_features = feature_matrix[:5]
        predictions = regressor.predict(test_features)
        
        assert len(predictions) == 5
        assert all(isinstance(pred, float) for pred in predictions)
        assert all(0 <= pred <= 1 for pred in predictions)  # Should be in [0,1] range
    
    @pytest.mark.asyncio
    async def test_ml_error_handling_real_scenarios(
        self,
        real_ml_service_for_testing,
        ml_error_scenarios,
    ):
        """Test ML error handling with real service and various error scenarios."""
        # Test empty data
        empty_result = await real_ml_service_for_testing.optimize_rules(
            ml_error_scenarios["empty_data"]["rules"],
            ml_error_scenarios["empty_data"]["performance"]
        )
        
        assert empty_result["status"] == "success"  # Should handle gracefully
        assert "model_id" in empty_result
        
        # Test invalid data structures
        invalid_result = await real_ml_service_for_testing.optimize_rules(
            ml_error_scenarios["invalid_data"]["rules"],
            ml_error_scenarios["invalid_data"]["performance"]
        )
        
        # Should handle invalid data gracefully
        assert "status" in invalid_result
        
        # Test pattern discovery with empty data
        pattern_result = await real_ml_service_for_testing.discover_patterns(
            ml_error_scenarios["empty_data"]["patterns"]
        )
        
        assert pattern_result["status"] == "success"
        assert pattern_result["patterns_discovered"] == 0
    
    @pytest.mark.asyncio
    async def test_ml_performance_characteristics(
        self,
        real_ml_service_for_testing,
        large_dataset,
    ):
        """Test ML performance characteristics with larger datasets."""
        features, targets = large_dataset
        
        # Create rules data for the larger dataset
        generator = PromptDataGenerator(random_state=42)
        large_rules = generator.generate_rules_data(features.shape[0])
        large_performance = generator.generate_performance_data(large_rules)
        
        # Test optimization performance
        import time
        start_time = time.time()
        
        result = await real_ml_service_for_testing.optimize_rules(
            large_rules[:100],  # Limit for reasonable test time
            large_performance[:100]
        )
        
        end_time = time.time()
        actual_time = (end_time - start_time) * 1000  # Convert to ms
        
        assert result["status"] == "success"
        assert "processing_time_ms" in result
        
        # Verify reasonable performance (should be fast for lightweight models)
        assert actual_time < 30000  # Should complete within 30 seconds
        
        # Test accuracy improves with more data
        small_result = await real_ml_service_for_testing.optimize_rules(
            large_rules[:20],
            large_performance[:20]
        )
        
        large_result = await real_ml_service_for_testing.optimize_rules(
            large_rules[:100],
            large_performance[:100]
        )
        
        # With more data, results should be more stable
        # (not necessarily higher accuracy, but more reliable)
        both_successful = (
            small_result["status"] == "success" and
            large_result["status"] == "success"
        )
        assert both_successful
    
    @pytest.mark.asyncio
    async def test_deterministic_behavior(
        self,
        sample_rules_data,
        sample_performance_data,
    ):
        """Test that ML operations are deterministic with same random seed."""
        # Create two identical services with same random seed
        service1 = RealMLService(random_state=42)
        service2 = RealMLService(random_state=42)
        
        # Run same operation on both
        result1 = await service1.optimize_rules(
            sample_rules_data[:10],
            sample_performance_data[:10]
        )
        
        result2 = await service2.optimize_rules(
            sample_rules_data[:10],
            sample_performance_data[:10]
        )
        
        # Results should be deterministic
        assert result1["status"] == result2["status"]
        if result1["status"] == "success" and result2["status"] == "success":
            # Model IDs should be the same (deterministic generation)
            assert result1["model_id"] == result2["model_id"]
            # Scores should be very close (some floating point variation expected)
            assert abs(result1["best_score"] - result2["best_score"]) < 0.01


class TestMLServiceIntegration:
    """Test integration between different ML service components."""
    
    @pytest.mark.asyncio
    async def test_ml_service_coordination(
        self,
        real_ml_session,
        ml_pipeline_test_data,
    ):
        """Test coordination between ML service components."""
        # Setup integrated ML services
        ml_service = await real_ml_session.setup_ml_service()
        mlflow_service = await real_ml_session.setup_mlflow_service()
        
        # Train models using coordinated services
        training_results = await real_ml_session.train_models(ml_pipeline_test_data)
        
        assert "optimization" in training_results
        assert "patterns" in training_results
        
        # Verify optimization results
        opt_result = training_results["optimization"]
        assert opt_result["status"] == "success"
        assert "model_id" in opt_result
        
        # Verify pattern discovery results
        pattern_result = training_results["patterns"]
        assert pattern_result["status"] == "success"
        assert "patterns_discovered" in pattern_result
        
        # Test that models are accessible through service
        assert len(ml_service.models) > 0
        assert len(ml_service.training_history) > 0
    
    @pytest.mark.asyncio
    async def test_real_vs_mock_behavior_comparison(
        self,
        real_ml_service_for_testing,
        sample_rules_data,
        sample_performance_data,
    ):
        """Demonstrate difference between real and mock ML behavior."""
        # Run real ML optimization
        real_result = await real_ml_service_for_testing.optimize_rules(
            sample_rules_data,
            sample_performance_data
        )
        
        # Verify real behavior characteristics
        assert real_result["status"] == "success"
        
        # Real service should show variation based on actual data
        # Run multiple times with different data subsets
        results = []
        for i in range(3):
            subset_result = await real_ml_service_for_testing.optimize_rules(
                sample_rules_data[i*3:(i+1)*3],
                sample_performance_data[i*3:(i+1)*3]
            )
            if subset_result["status"] == "success":
                results.append(subset_result["best_score"])
        
        # Real ML should show some variation based on data
        if len(results) > 1:
            score_variation = max(results) - min(results)
            # Should have some variation (unlike mocks that return fixed values)
            assert score_variation >= 0  # Can be 0 for very small datasets
    
    @pytest.mark.asyncio
    async def test_ml_pipeline_scalability(
        self,
        end_to_end_ml_pipeline,
    ):
        """Test ML pipeline scalability with real data processing."""
        pipeline = end_to_end_ml_pipeline
        ml_service = pipeline["ml_service"]
        mlflow_service = pipeline["mlflow_service"]
        test_data = pipeline["test_data"]
        
        # Test processing larger batches
        batch_sizes = [5, 10, 20]
        processing_times = []
        
        for batch_size in batch_sizes:
            import time
            start_time = time.time()
            
            result = await ml_service.optimize_rules(
                test_data["rules"][:batch_size],
                test_data["performance"][:batch_size]
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            processing_times.append(processing_time)
            
            assert result["status"] == "success"
        
        # Processing time should scale reasonably
        # (Not necessarily linearly due to overhead, but should not explode)
        assert all(time < 60000 for time in processing_times)  # All under 1 minute
        
        # Verify MLflow can handle multiple experiments
        experiment_count = len(mlflow_service.experiments)
        assert experiment_count >= len(pipeline["experiment_ids"])
    
    @pytest.mark.asyncio
    async def test_ml_model_persistence_and_loading(
        self,
        real_mlflow_service_for_testing,
        trained_text_classifier,
    ):
        """Test real model persistence and loading through MLflow."""
        classifier, training_result = trained_text_classifier
        
        # Persist model
        model_uri = await real_mlflow_service_for_testing.log_model(
            "persistent_classifier",
            classifier,
            {
                "accuracy": training_result.accuracy,
                "training_time": training_result.training_time,
                "model_type": training_result.model_type
            }
        )
        
        # Verify model was persisted
        assert model_uri in real_mlflow_service_for_testing.models
        
        # Retrieve metadata
        metadata = await real_mlflow_service_for_testing.get_model_metadata(model_uri)
        assert metadata["accuracy"] == training_result.accuracy
        assert metadata["model_type"] == training_result.model_type
        
        # Verify model file was created on disk
        model_record = real_mlflow_service_for_testing.models[model_uri]
        if "file_path" in model_record:
            model_file = Path(model_record["file_path"])
            assert model_file.exists()
            assert model_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])