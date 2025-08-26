"""Real ML Fixtures for Integration Testing

Provides pytest fixtures that create real ML services and data
instead of mocks, ensuring true ML behavior testing.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.real_ml.lightweight_models import (
    LightweightRegressor,
    LightweightTextClassifier,
    PatternDiscoveryEngine,
    RealMLflowService,
    RealMLService,
)
from tests.real_ml.test_data_generators import (
    MLTrainingDataGenerator,
    PromptDataGenerator,
)


@pytest.fixture
def real_ml_service():
    """Real ML service instance for testing."""
    return RealMLService(random_state=42)


@pytest.fixture
def real_text_classifier():
    """Real text classifier for prompt analysis."""
    return LightweightTextClassifier(random_state=42)


@pytest.fixture
def real_regressor():
    """Real regression model for rule optimization."""
    return LightweightRegressor(random_state=42)


@pytest.fixture
def pattern_discovery_engine():
    """Real pattern discovery engine."""
    return PatternDiscoveryEngine(random_state=42)


@pytest.fixture
def real_mlflow_service():
    """Real MLflow service with temporary storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        service = RealMLflowService(storage_dir=Path(temp_dir))
        yield service


@pytest.fixture
def prompt_data_generator():
    """Deterministic prompt data generator."""
    return PromptDataGenerator(random_state=42)


@pytest.fixture
def ml_training_data_generator():
    """ML training data generator."""
    return MLTrainingDataGenerator(random_state=42)


@pytest.fixture
def sample_prompts(prompt_data_generator):
    """Sample prompt texts for testing."""
    return prompt_data_generator.generate_prompts(20)


@pytest.fixture
def sample_rules_data(prompt_data_generator):
    """Sample rule configuration data."""
    return prompt_data_generator.generate_rules_data(15)


@pytest.fixture
def sample_performance_data(sample_rules_data, prompt_data_generator):
    """Sample performance data correlated with rules."""
    return prompt_data_generator.generate_performance_data(sample_rules_data)


@pytest.fixture
def training_dataset(prompt_data_generator):
    """Complete training dataset with prompts, labels, and metadata."""
    prompts, labels, metadata = prompt_data_generator.generate_training_dataset(50)
    return {
        "prompts": prompts,
        "labels": labels,
        "metadata": metadata
    }


@pytest.fixture
def pattern_discovery_data(prompt_data_generator):
    """Data suitable for pattern discovery testing."""
    return prompt_data_generator.generate_pattern_data(30)


@pytest.fixture
def feature_matrix(ml_training_data_generator):
    """Sample feature matrix for ML training."""
    return ml_training_data_generator.generate_feature_matrix(
        n_samples=100,
        n_features=8,
        noise_level=0.05
    )


@pytest.fixture
def regression_targets(feature_matrix, ml_training_data_generator):
    """Regression targets correlated with features."""
    return ml_training_data_generator.generate_regression_targets(
        feature_matrix,
        target_type="linear"
    )


@pytest.fixture
def classification_targets(feature_matrix, ml_training_data_generator):
    """Classification targets for testing."""
    return ml_training_data_generator.generate_classification_targets(
        feature_matrix,
        n_classes=2
    )


@pytest.fixture
def trained_text_classifier(real_text_classifier, training_dataset):
    """Pre-trained text classifier for testing."""
    classifier = real_text_classifier
    result = classifier.fit(
        training_dataset["prompts"],
        training_dataset["labels"]
    )
    return classifier, result


@pytest.fixture
def trained_regressor(real_regressor, feature_matrix, regression_targets):
    """Pre-trained regressor for testing."""
    regressor = real_regressor
    result = regressor.fit(feature_matrix, regression_targets)
    return regressor, result


@pytest.fixture
async def ml_service_with_models(real_ml_service, sample_rules_data, sample_performance_data):
    """ML service with pre-trained models."""
    service = real_ml_service

    # Train some models
    optimization_result = await service.optimize_rules(
        sample_rules_data[:10],
        sample_performance_data[:10]
    )

    pattern_result = await service.discover_patterns(sample_performance_data)

    return service, {
        "optimization_result": optimization_result,
        "pattern_result": pattern_result
    }


@pytest.fixture
async def mlflow_with_experiments(real_mlflow_service, sample_rules_data):
    """MLflow service with sample experiments and models."""
    service = real_mlflow_service

    # Create sample experiments
    experiments = []
    for i, rule in enumerate(sample_rules_data[:3]):
        run_id = await service.log_experiment(
            f"experiment_{rule['name']}",
            rule.get("default_parameters", {})
        )
        experiments.append(run_id)

        # Log a model for each experiment
        model_uri = await service.log_model(
            f"model_{rule['name']}",
            {"type": "test_model", "rule_id": rule["id"]},
            {
                "accuracy": 0.85 + i * 0.02,
                "algorithm": "test_algorithm",
                "features": 10 + i
            }
        )

    return service, {"experiments": experiments}


@pytest.fixture
def time_series_data(ml_training_data_generator):
    """Time series data for temporal ML testing."""
    return ml_training_data_generator.generate_time_series_data(
        length=100,
        n_features=3,
        trend=True,
        seasonality=True,
        noise_level=0.1
    )


@pytest.fixture
def ml_pipeline_test_data(
    sample_rules_data,
    sample_performance_data,
    pattern_discovery_data,
    feature_matrix,
    regression_targets
):
    """Complete test data package for ML pipeline testing."""
    return {
        "rules": sample_rules_data,
        "performance": sample_performance_data,
        "patterns": pattern_discovery_data,
        "features": feature_matrix,
        "targets": regression_targets,
        "n_samples": len(sample_rules_data),
        "n_features": feature_matrix.shape[1]
    }


@pytest.fixture
def real_ml_components(
    real_ml_service,
    real_text_classifier,
    real_regressor,
    pattern_discovery_engine,
    real_mlflow_service
):
    """Collection of all real ML components for comprehensive testing."""
    return {
        "ml_service": real_ml_service,
        "text_classifier": real_text_classifier,
        "regressor": real_regressor,
        "pattern_engine": pattern_discovery_engine,
        "mlflow_service": real_mlflow_service
    }


class RealMLTestSession:
    """Session manager for real ML testing with proper cleanup."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.services = {}
        self.trained_models = {}

    async def setup_ml_service(self) -> RealMLService:
        """Setup real ML service."""
        service = RealMLService(random_state=42)
        self.services["ml_service"] = service
        return service

    async def setup_mlflow_service(self) -> RealMLflowService:
        """Setup real MLflow service."""
        service = RealMLflowService(storage_dir=self.storage_dir)
        self.services["mlflow_service"] = service
        return service

    async def train_models(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train models with provided data."""
        results = {}

        if "ml_service" in self.services:
            service = self.services["ml_service"]

            # Train optimization model
            if "rules" in training_data and "performance" in training_data:
                opt_result = await service.optimize_rules(
                    training_data["rules"],
                    training_data["performance"]
                )
                results["optimization"] = opt_result

            # Discover patterns
            if "patterns" in training_data:
                pattern_result = await service.discover_patterns(
                    training_data["patterns"]
                )
                results["patterns"] = pattern_result

        self.trained_models.update(results)
        return results

    async def cleanup(self):
        """Cleanup resources."""
        self.services.clear()
        self.trained_models.clear()


@pytest.fixture
async def real_ml_session():
    """Real ML testing session with proper lifecycle management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session = RealMLTestSession(Path(temp_dir))
        yield session
        await session.cleanup()


# Performance testing fixtures

@pytest.fixture
def large_dataset(ml_training_data_generator):
    """Large dataset for performance testing."""
    features = ml_training_data_generator.generate_feature_matrix(
        n_samples=1000,
        n_features=20,
        noise_level=0.05
    )
    targets = ml_training_data_generator.generate_regression_targets(
        features,
        target_type="nonlinear"
    )
    return features, targets


@pytest.fixture
def complex_rules_dataset(prompt_data_generator):
    """Complex rules dataset for stress testing."""
    rules = prompt_data_generator.generate_rules_data(100)
    performance = prompt_data_generator.generate_performance_data(rules)
    patterns = prompt_data_generator.generate_pattern_data(200)

    return {
        "rules": rules,
        "performance": performance,
        "patterns": patterns
    }


# Integration testing fixtures

@pytest.fixture
async def end_to_end_ml_pipeline(
    real_ml_session,
    ml_pipeline_test_data
):
    """Complete end-to-end ML pipeline for integration testing."""

    # Setup services
    ml_service = await real_ml_session.setup_ml_service()
    mlflow_service = await real_ml_session.setup_mlflow_service()

    # Train models
    training_results = await real_ml_session.train_models(ml_pipeline_test_data)

    # Create experiments in MLflow
    experiment_ids = []
    for i in range(3):
        exp_id = await mlflow_service.log_experiment(
            f"pipeline_test_{i}",
            {"test_param": i, "dataset_size": len(ml_pipeline_test_data["rules"])}
        )
        experiment_ids.append(exp_id)

    return {
        "ml_service": ml_service,
        "mlflow_service": mlflow_service,
        "training_results": training_results,
        "experiment_ids": experiment_ids,
        "test_data": ml_pipeline_test_data
    }


# Error simulation fixtures

@pytest.fixture
def ml_error_scenarios():
    """Scenarios for testing ML error handling with real components."""
    return {
        "empty_data": {
            "rules": [],
            "performance": [],
            "patterns": []
        },
        "invalid_data": {
            "rules": [{"invalid": "structure"}],
            "performance": [{"missing": "fields"}],
            "patterns": [{"bad": "format"}]
        },
        "mismatched_data": {
            "rules": [{"id": "rule_1", "name": "test"}],
            "performance": [{"rule_id": "rule_2", "effectiveness": 0.8}],  # Mismatched IDs
            "patterns": []
        }
    }
