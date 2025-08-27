"""
Real behavior integration tests for ML components.

Tests failure_analyzer.py and failure_classifier.py using actual OpenTelemetry
metrics collection, real PostgreSQL database, and genuine data flow.
Follows 2025 best practices with no mocking of core functionality.
"""

import json
import uuid
from datetime import datetime

import pytest
from tests.fixtures.foundation.utils import requires_sklearn

# Test markers
requires_otel = pytest.mark.skipif(
    True,  # OTEL is disabled in tests
    reason="OpenTelemetry disabled in test environment"
)

requires_real_db = pytest.mark.skipif(
    False,  # Real DB is available in tests
    reason="Real database not available"
)

from prompt_improver.ml.failure_analyzer import FailureAnalyzer
from prompt_improver.ml.failure_classifier import FailureClassifier


@requires_otel
@requires_real_db
@requires_sklearn
class TestRealBehaviorMLComponents:
    """
    Real behavior integration tests for ML components.

    Tests actual functionality with real infrastructure:
    - OpenTelemetry metrics collection
    - PostgreSQL database interactions
    - Real data processing and analysis
    """

    async def test_failure_analyzer_real_metrics_collection(
        self, real_behavior_environment
    ):
        """
        Test FailureAnalyzer with real OpenTelemetry metrics collection.

        Validates that the migrated component correctly emits metrics
        to actual OpenTelemetry infrastructure.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        tracer = real_behavior_environment["tracer"]
        analyzer = FailureAnalyzer()
        failure_data = {
            "error_type": "timeout",
            "error_message": "Request timeout after 30 seconds",
            "stack_trace": "TimeoutError: Request timeout\n  at request.py:45",
            "context": {
                "endpoint": "/api/analyze",
                "user_id": "test_user_123",
                "request_size": 1024,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        with tracer.start_as_current_span("test_failure_analysis") as span:
            analysis_result = await analyzer.analyze_failure(failure_data)
            assert analysis_result is not None
            assert "failure_type" in analysis_result
            assert "confidence_score" in analysis_result
            assert "analysis_data" in analysis_result
            assert analysis_result["confidence_score"] >= 0.0
            assert analysis_result["confidence_score"] <= 1.0
            span_attributes = span.get_span_context()
            assert span_attributes is not None
        failure_counter = otel_metrics["failure_analysis_counter"]
        assert failure_counter is not None
        result = await database.execute(
            "SELECT * FROM failure_analysis WHERE failure_type = %s",
            (analysis_result["failure_type"],),
        )
        db_records = await result.fetchall()
        assert len(db_records) > 0
        db_record = db_records[0]
        assert db_record["confidence_score"] == analysis_result["confidence_score"]
        assert db_record["analysis_data"] is not None

    async def test_failure_classifier_real_data_processing(
        self, real_behavior_environment
    ):
        """
        Test FailureClassifier with real data processing and metrics.

        Validates classification accuracy with actual ML processing
        and real metric emission.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        meter = real_behavior_environment["meter"]
        classifier = FailureClassifier()
        training_features = [
            [0.8, 150, 1.0, 5, 0.7],
            [0.3, 50, 0.2, 2, 0.1],
            [0.9, 300, 1.0, 8, 0.9],
            [0.5, 100, 0.6, 3, 0.4],
            [0.2, 25, 0.1, 1, 0.05],
        ] * 10
        training_labels = ["timeout", "connection", "system", "error", "warning"] * 10
        training_result = await classifier.train(training_features, training_labels)
        assert training_result["status"] == "success"
        assert "model_id" in training_result
        assert "accuracy" in training_result
        assert training_result["accuracy"] > 0.5
        test_features = [0.7, 200, 0.8, 6, 0.6]
        classification_result = await classifier.classify(test_features)
        assert classification_result is not None
        assert "predicted_class" in classification_result
        assert "confidence" in classification_result
        assert "probabilities" in classification_result
        assert classification_result["confidence"] >= 0.0
        assert classification_result["confidence"] <= 1.0
        classification_counter = otel_metrics["failure_classification_counter"]
        assert classification_counter is not None
        result = await database.execute(
            "SELECT * FROM ml_metrics WHERE component = %s", ("failure_classifier",)
        )
        ml_metrics = await result.fetchall()
        assert len(ml_metrics) > 0
        metric_names = [record["metric_name"] for record in ml_metrics]
        assert "classification_accuracy" in metric_names
        assert "training_duration" in metric_names

    async def test_end_to_end_ml_pipeline_real_behavior(
        self, real_behavior_environment
    ):
        """
        Test complete ML pipeline with real data flow and infrastructure.

        Validates end-to-end processing from failure analysis through
        classification with actual metrics and database storage.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        redis = real_behavior_environment["redis"]
        tracer = real_behavior_environment["tracer"]
        analyzer = FailureAnalyzer()
        classifier = FailureClassifier()
        failure_scenarios = [
            {
                "id": str(uuid.uuid4()),
                "error_type": "timeout",
                "severity": 0.8,
                "duration": 250,
                "context": {"service": "api", "endpoint": "/analyze"},
            },
            {
                "id": str(uuid.uuid4()),
                "error_type": "connection",
                "severity": 0.3,
                "duration": 50,
                "context": {"service": "database", "operation": "select"},
            },
            {
                "id": str(uuid.uuid4()),
                "error_type": "system",
                "severity": 0.9,
                "duration": 500,
                "context": {"service": "ml", "component": "training"},
            },
        ]
        pipeline_results = []
        with tracer.start_as_current_span("ml_pipeline_test") as pipeline_span:
            for scenario in failure_scenarios:
                with tracer.start_as_current_span("failure_analysis") as analysis_span:
                    analysis_result = await analyzer.analyze_failure(scenario)
                    analysis_span.set_attribute("failure_id", scenario["id"])
                    analysis_span.set_attribute(
                        "failure_type", analysis_result["failure_type"]
                    )
                features = [
                    scenario["severity"],
                    scenario["duration"],
                    1.0 if scenario["error_type"] == "timeout" else 0.0,
                    len(scenario["context"]),
                    analysis_result["confidence_score"],
                ]
                with tracer.start_as_current_span(
                    "failure_classification"
                ) as classification_span:
                    classification_result = await classifier.classify(features)
                    classification_span.set_attribute(
                        "predicted_class", classification_result["predicted_class"]
                    )
                    classification_span.set_attribute(
                        "confidence", classification_result["confidence"]
                    )
                cache_key = f"ml_result:{scenario['id']}"
                pipeline_result = {
                    "scenario_id": scenario["id"],
                    "analysis": analysis_result,
                    "classification": classification_result,
                    "processed_at": datetime.utcnow().isoformat(),
                }
                await redis.set(cache_key, json.dumps(pipeline_result), ex=3600)
                pipeline_results.append(pipeline_result)
        assert len(pipeline_results) == len(failure_scenarios)
        for result in pipeline_results:
            assert "analysis" in result
            assert result["analysis"]["confidence_score"] >= 0.0
            assert "classification" in result
            assert result["classification"]["predicted_class"] in {
                "timeout",
                "connection",
                "system",
                "error",
                "warning",
            }
            cached_result = await redis.get(f"ml_result:{result['scenario_id']}")
            assert cached_result is not None
            cached_data = json.loads(cached_result)
            assert cached_data["scenario_id"] == result["scenario_id"]
        result = await database.execute(
            "SELECT COUNT(*) as count FROM ml_metrics WHERE component IN %s",
            (("failure_analyzer", "failure_classifier"),),
        )
        metric_count = await result.fetchone()
        assert metric_count["count"] >= len(failure_scenarios) * 2
        result = await database.execute(
            "SELECT COUNT(*) as count FROM failure_analysis"
        )
        analysis_count = await result.fetchone()
        assert analysis_count["count"] >= len(failure_scenarios)

    async def test_ml_component_performance_real_metrics(
        self, real_behavior_environment
    ):
        """
        Test ML component performance with real timing metrics.

        Validates that performance metrics are accurately captured
        and stored using actual OpenTelemetry infrastructure.
        """
        otel_metrics = real_behavior_environment["otel_metrics"]
        database = real_behavior_environment["database"]
        analyzer = FailureAnalyzer()
        classifier = FailureClassifier()
        test_data = {
            "error_type": "performance_test",
            "error_message": "Performance testing scenario",
            "context": {"test": True, "iterations": 100},
        }
        start_time = datetime.utcnow()
        for _i in range(10):
            analysis_result = await analyzer.analyze_failure(test_data)
            features = [0.5, 100, 0.5, 3, analysis_result["confidence_score"]]
            classification_result = await classifier.classify(features)
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds() * 1000
        result = await database.execute(
            "SELECT * FROM ml_metrics WHERE metric_name LIKE %s", ("%duration%",)
        )
        duration_metrics = await result.fetchall()
        assert len(duration_metrics) > 0
        assert total_duration < 5000
        for metric in duration_metrics:
            assert metric["metric_value"] > 0
            assert metric["metric_value"] < 1000
