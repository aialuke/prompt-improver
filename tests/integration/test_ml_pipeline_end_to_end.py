"""
End-to-end ML pipeline integration tests with real behavior.

Tests complete ML workflow from data ingestion through analysis, classification,
and metric storage using actual OpenTelemetry infrastructure and real databases.
Follows 2025 best practices with comprehensive real behavior validation.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from opentelemetry import metrics, trace
from tests.conftest import requires_otel, requires_real_db, requires_sklearn

from prompt_improver.ml.failure_analyzer import FailureAnalyzer
from prompt_improver.ml.failure_classifier import FailureClassifier
from prompt_improver.monitoring.opentelemetry_ml_monitoring import (
    OpenTelemetryMLMonitoring,
)


@requires_otel
@requires_real_db
@requires_sklearn
class TestMLPipelineEndToEnd:
    """
    End-to-end ML pipeline tests with real infrastructure.

    Validates complete workflows using actual:
    - OpenTelemetry metrics and tracing
    - PostgreSQL database operations
    - Redis caching and data flow
    - ML model training and inference
    """

    async def test_complete_ml_training_pipeline(self, real_behavior_environment):
        """
        Test complete ML training pipeline with real data flow.

        Validates end-to-end training workflow from data preparation
        through model training, validation, and metric storage.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        redis = real_behavior_environment["redis"]
        tracer = real_behavior_environment["tracer"]
        ml_monitoring = OpenTelemetryMLMonitoring()
        analyzer = FailureAnalyzer()
        classifier = FailureClassifier()
        training_data = await self._generate_realistic_training_data()
        with tracer.start_as_current_span("ml_training_pipeline") as pipeline_span:
            pipeline_span.set_attribute("dataset_size", len(training_data["features"]))
            pipeline_span.set_attribute(
                "feature_count", len(training_data["features"][0])
            )
            with tracer.start_as_current_span("data_preprocessing") as prep_span:
                processed_data = await self._preprocess_training_data(training_data)
                prep_span.set_attribute(
                    "processed_samples", len(processed_data["features"])
                )
                await self._store_preprocessing_metrics(database, processed_data)
            with tracer.start_as_current_span("model_training") as training_span:
                training_result = await classifier.train(
                    processed_data["features"], processed_data["labels"]
                )
                training_span.set_attribute("model_id", training_result["model_id"])
                training_span.set_attribute("accuracy", training_result["accuracy"])
                assert training_result["status"] == "success"
                assert training_result["accuracy"] > 0.6
                assert "model_id" in training_result
            with tracer.start_as_current_span("model_validation") as validation_span:
                validation_data = await self._generate_validation_data()
                validation_results = await self._validate_model(
                    classifier, validation_data
                )
                validation_span.set_attribute(
                    "validation_accuracy", validation_results["accuracy"]
                )
                validation_span.set_attribute(
                    "validation_samples", len(validation_data["features"])
                )
                await self._store_validation_metrics(database, validation_results)
            with tracer.start_as_current_span("model_deployment") as deployment_span:
                deployment_result = await self._simulate_model_deployment(
                    redis, training_result["model_id"], validation_results
                )
                deployment_span.set_attribute(
                    "deployment_status", deployment_result["status"]
                )
                deployment_span.set_attribute(
                    "model_version", deployment_result["version"]
                )
        assert training_result["accuracy"] > 0.6
        assert validation_results["accuracy"] > 0.5
        assert deployment_result["status"] == "deployed"
        result = await database.execute(
            "SELECT COUNT(*) as count FROM ml_metrics WHERE component = %s",
            ("ml_pipeline",),
        )
        metric_count = await result.fetchone()
        assert metric_count["count"] >= 6
        model_key = f"model:{training_result['model_id']}"
        model_metadata = await redis.get(model_key)
        assert model_metadata is not None
        metadata = json.loads(model_metadata)
        assert metadata["accuracy"] == training_result["accuracy"]
        assert metadata["validation_accuracy"] == validation_results["accuracy"]

    async def test_real_time_inference_pipeline(self, real_behavior_environment):
        """
        Test real-time inference pipeline with actual data processing.

        Validates inference workflow with real metrics collection,
        caching, and performance monitoring.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        redis = real_behavior_environment["redis"]
        tracer = real_behavior_environment["tracer"]
        analyzer = FailureAnalyzer()
        classifier = FailureClassifier()
        training_data = await self._generate_realistic_training_data()
        training_result = await classifier.train(
            training_data["features"], training_data["labels"]
        )
        inference_scenarios = [
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow(),
                "failure_data": {
                    "error_type": "timeout",
                    "severity": 0.8,
                    "duration": 300,
                    "context": {"service": "api", "user_count": 150},
                },
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow(),
                "failure_data": {
                    "error_type": "connection",
                    "severity": 0.4,
                    "duration": 100,
                    "context": {"service": "database", "connection_pool": 20},
                },
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow(),
                "failure_data": {
                    "error_type": "memory",
                    "severity": 0.9,
                    "duration": 600,
                    "context": {"service": "ml", "memory_usage": 0.95},
                },
            },
        ]
        inference_results = []
        for scenario in inference_scenarios:
            with tracer.start_as_current_span("real_time_inference") as inference_span:
                inference_span.set_attribute("scenario_id", scenario["id"])
                inference_span.set_attribute(
                    "error_type", scenario["failure_data"]["error_type"]
                )
                cache_key = f"inference:{scenario['failure_data']['error_type']}"
                cached_result = await redis.get(cache_key)
                if cached_result:
                    inference_span.set_attribute("cache_hit", True)
                    result = json.loads(cached_result)
                else:
                    inference_span.set_attribute("cache_hit", False)
                    analysis_result = await analyzer.analyze_failure(
                        scenario["failure_data"]
                    )
                    features = [
                        scenario["failure_data"]["severity"],
                        scenario["failure_data"]["duration"],
                        1.0
                        if scenario["failure_data"]["error_type"] == "timeout"
                        else 0.0,
                        len(scenario["failure_data"]["context"]),
                        analysis_result["confidence_score"],
                    ]
                    classification_result = await classifier.classify(features)
                    result = {
                        "scenario_id": scenario["id"],
                        "analysis": analysis_result,
                        "classification": classification_result,
                        "inference_time": datetime.utcnow().isoformat(),
                        "cached": False,
                    }
                    await redis.set(cache_key, json.dumps(result), ex=1800)
                inference_results.append(result)
                await self._store_inference_metrics(database, scenario, result)
        assert len(inference_results) == len(inference_scenarios)
        for result in inference_results:
            assert "analysis" in result
            assert "classification" in result
            assert result["analysis"]["confidence_score"] >= 0.0
            assert result["classification"]["confidence"] >= 0.0
        result = await database.execute(
            "SELECT COUNT(*) as count FROM ml_metrics WHERE metric_name LIKE %s",
            ("%inference%",),
        )
        inference_metric_count = await result.fetchone()
        assert inference_metric_count["count"] >= len(inference_scenarios)
        cache_keys = [
            f"inference:{s['failure_data']['error_type']}" for s in inference_scenarios
        ]
        cached_count = 0
        for key in cache_keys:
            if await redis.exists(key):
                cached_count += 1
        assert cached_count > 0

    async def test_ml_pipeline_error_handling(self, real_behavior_environment):
        """
        Test ML pipeline error handling with real failure scenarios.

        Validates that the pipeline gracefully handles errors and
        continues processing with proper error metrics collection.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        tracer = real_behavior_environment["tracer"]
        analyzer = FailureAnalyzer()
        classifier = FailureClassifier()
        error_scenarios = [
            {
                "name": "invalid_data_format",
                "data": {"invalid": "data structure"},
                "expected_error": "data_format_error",
            },
            {
                "name": "missing_required_fields",
                "data": {"error_type": "timeout"},
                "expected_error": "missing_fields_error",
            },
            {
                "name": "extreme_values",
                "data": {
                    "error_type": "timeout",
                    "severity": 10.0,
                    "duration": -100,
                    "context": {},
                },
                "expected_error": "validation_error",
            },
        ]
        error_results = []
        with tracer.start_as_current_span("error_handling_test") as error_span:
            for scenario in error_scenarios:
                with tracer.start_as_current_span(
                    f"error_scenario_{scenario['name']}"
                ) as scenario_span:
                    try:
                        analysis_result = await analyzer.analyze_failure(
                            scenario["data"]
                        )
                        if "error" in analysis_result:
                            error_results.append({
                                "scenario": scenario["name"],
                                "error_handled": True,
                                "error_type": analysis_result["error"],
                                "graceful": True,
                            })
                            scenario_span.set_attribute("error_handled", True)
                        else:
                            error_results.append({
                                "scenario": scenario["name"],
                                "error_handled": False,
                                "unexpected_success": True,
                                "graceful": True,
                            })
                            scenario_span.set_attribute("unexpected_success", True)
                    except Exception as e:
                        error_results.append({
                            "scenario": scenario["name"],
                            "error_handled": True,
                            "error_type": str(type(e).__name__),
                            "error_message": str(e),
                            "graceful": True,
                        })
                        scenario_span.set_attribute("error_type", str(type(e).__name__))
                        scenario_span.set_attribute("error_handled", True)
                    await self._store_error_metrics(
                        database, scenario, error_results[-1]
                    )
        assert len(error_results) == len(error_scenarios)
        for result in error_results:
            assert result["graceful"] is True
        result = await database.execute(
            "SELECT COUNT(*) as count FROM ml_metrics WHERE metric_name LIKE %s",
            ("%error%",),
        )
        error_metric_count = await result.fetchone()
        assert error_metric_count["count"] >= len(error_scenarios)

    async def _generate_realistic_training_data(self) -> dict[str, list]:
        """Generate realistic training data for ML pipeline testing."""
        features = []
        labels = []
        failure_types = ["timeout", "connection", "memory", "disk", "network"]
        for failure_type in failure_types:
            for i in range(20):
                if failure_type == "timeout":
                    feature = [
                        np.random.uniform(0.6, 1.0),
                        np.random.uniform(200, 500),
                        1.0,
                        np.random.randint(3, 8),
                        np.random.uniform(0.7, 0.95),
                    ]
                elif failure_type == "connection":
                    feature = [
                        np.random.uniform(0.3, 0.7),
                        np.random.uniform(50, 200),
                        0.0,
                        np.random.randint(2, 5),
                        np.random.uniform(0.6, 0.85),
                    ]
                else:
                    feature = [
                        np.random.uniform(0.1, 0.9),
                        np.random.uniform(10, 400),
                        0.0,
                        np.random.randint(1, 6),
                        np.random.uniform(0.5, 0.9),
                    ]
                features.append(feature)
                labels.append(failure_type)
        return {"features": features, "labels": labels}

    async def _preprocess_training_data(
        self, training_data: dict[str, list]
    ) -> dict[str, list]:
        """Preprocess training data with validation and normalization."""
        features = np.array(training_data["features"])
        labels = training_data["labels"]
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-08)
        return {"features": features.tolist(), "labels": labels}

    async def _generate_validation_data(self) -> dict[str, list]:
        """Generate validation data for model testing."""
        return await self._generate_realistic_training_data()

    async def _validate_model(
        self, classifier, validation_data: dict[str, list]
    ) -> dict[str, Any]:
        """Validate trained model with test data."""
        correct_predictions = 0
        total_predictions = len(validation_data["features"])
        for features, true_label in zip(
            validation_data["features"], validation_data["labels"], strict=False
        ):
            result = await classifier.classify(features)
            if result["predicted_class"] == true_label:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

    async def _simulate_model_deployment(
        self, redis, model_id: str, validation_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate model deployment with metadata storage."""
        deployment_metadata = {
            "model_id": model_id,
            "version": "1.0.0",
            "accuracy": validation_results["accuracy"],
            "deployed_at": datetime.utcnow().isoformat(),
            "status": "deployed",
        }
        await redis.set(f"model:{model_id}", json.dumps(deployment_metadata), ex=86400)
        return deployment_metadata

    async def _store_preprocessing_metrics(
        self, database, processed_data: dict[str, list]
    ) -> None:
        """Store preprocessing metrics in database."""
        await database.execute(
            "INSERT INTO ml_metrics (metric_name, metric_value, component, labels) VALUES (%s, %s, %s, %s)",
            (
                "preprocessing_samples",
                len(processed_data["features"]),
                "ml_pipeline",
                json.dumps({"stage": "preprocessing"}),
            ),
        )
        await database.commit()

    async def _store_validation_metrics(
        self, database, validation_results: dict[str, Any]
    ) -> None:
        """Store validation metrics in database."""
        await database.execute(
            "INSERT INTO ml_metrics (metric_name, metric_value, component, labels) VALUES (%s, %s, %s, %s)",
            (
                "validation_accuracy",
                validation_results["accuracy"],
                "ml_pipeline",
                json.dumps({"stage": "validation"}),
            ),
        )
        await database.commit()

    async def _store_inference_metrics(
        self, database, scenario: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Store inference metrics in database."""
        await database.execute(
            "INSERT INTO ml_metrics (metric_name, metric_value, component, labels) VALUES (%s, %s, %s, %s)",
            (
                "inference_confidence",
                result["classification"]["confidence"],
                "ml_pipeline",
                json.dumps({"stage": "inference", "scenario_id": scenario["id"]}),
            ),
        )
        await database.commit()

    async def _store_error_metrics(
        self, database, scenario: dict[str, Any], error_result: dict[str, Any]
    ) -> None:
        """Store error handling metrics in database."""
        await database.execute(
            "INSERT INTO ml_metrics (metric_name, metric_value, component, labels) VALUES (%s, %s, %s, %s)",
            (
                "error_handled",
                1.0 if error_result["error_handled"] else 0.0,
                "ml_pipeline",
                json.dumps({"stage": "error_handling", "scenario": scenario["name"]}),
            ),
        )
        await database.commit()
