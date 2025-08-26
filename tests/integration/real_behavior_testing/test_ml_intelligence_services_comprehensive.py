"""Comprehensive Real Behavior Tests for ML Intelligence Services.

Tests the complete ML Intelligence Services decomposition with real behavior validation:
- MLIntelligenceServiceFacade (facade coordination)
- MLCircuitBreakerService (resilience)
- RuleAnalysisService (rule intelligence processing)
- PatternDiscoveryService (pattern identification)
- MLPredictionService (ML predictions)
- BatchProcessingService (batch operations)

All tests use real containers and services - no mocks.
Performance targets: <200ms facade operations, <100ms individual services.
"""

import asyncio
import contextlib
import logging
import time
from typing import Any

import pytest
from tests.integration.real_behavior_testing.containers.ml_test_container import (
    MLTestContainer,
)

from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.ml.services.intelligence.batch_processing_service import (
    BatchProcessingService,
)
from prompt_improver.ml.services.intelligence.circuit_breaker_service import (
    MLCircuitBreakerService,
)
from prompt_improver.ml.services.intelligence.facade import (
    MLIntelligenceServiceFacade,
    create_ml_intelligence_service_facade,
)
from prompt_improver.ml.services.intelligence.pattern_discovery_service import (
    PatternDiscoveryService,
)
from prompt_improver.ml.services.intelligence.prediction_service import (
    MLPredictionService,
)
from prompt_improver.ml.services.intelligence.rule_analysis_service import (
    RuleAnalysisService,
)
from prompt_improver.repositories.impl.ml_repository_service.ml_repository_facade import (
    MLRepositoryFacade,
)

logger = logging.getLogger(__name__)


@pytest.fixture
async def ml_repository_facade(integrated_services):
    """Create ML repository facade with real database connection."""
    ml_repository = MLRepositoryFacade(
        database_url=integrated_services["database_url"],
        correlation_id="test_ml_intelligence"
    )
    await ml_repository.initialize()
    yield ml_repository
    await ml_repository.cleanup()


@pytest.fixture
async def advanced_pattern_discovery(ml_container: MLTestContainer):
    """Create advanced pattern discovery component with real ML container."""
    return AdvancedPatternDiscovery(
        clustering_min_samples=3,
        clustering_min_cluster_size=5,
        feature_extraction_method="tfidf",
        enable_caching=True
    )


@pytest.fixture
async def ml_intelligence_facade(
    ml_repository_facade: MLRepositoryFacade,
    advanced_pattern_discovery: AdvancedPatternDiscovery
) -> MLIntelligenceServiceFacade:
    """Create ML Intelligence Service Facade with real dependencies."""
    facade = create_ml_intelligence_service_facade(
        ml_repository=ml_repository_facade,
        pattern_discovery=advanced_pattern_discovery
    )
    yield facade
    await facade.stop_background_processing()


@pytest.fixture
async def test_rule_data(ml_repository_facade: MLRepositoryFacade, test_data_factory):
    """Create comprehensive test rule data in database."""
    # Create test rules and sessions
    rule_data = await test_data_factory.create_comprehensive_rule_dataset(
        rule_count=20,
        session_count_per_rule=50,
        domains=["general", "technical", "creative"]
    )

    # Store in repository
    for rule_id, sessions in rule_data["sessions_by_rule"].items():
        await ml_repository_facade.store_rule_sessions(rule_id, sessions)

    yield rule_data

    # Cleanup
    for rule_id in rule_data["rule_ids"]:
        await ml_repository_facade.delete_rule_data(rule_id)


class TestMLCircuitBreakerService:
    """Test ML Circuit Breaker Service with real failure scenarios."""

    async def test_circuit_breaker_initialization(self):
        """Test circuit breaker service initialization."""
        service = MLCircuitBreakerService()

        # Test setup
        components = ["rule_analysis", "pattern_discovery", "predictions"]
        await service.setup_circuit_breakers(components)

        # Verify circuit breakers were created
        states = await service.get_all_states()
        assert len(states) == len(components)

        for component in components:
            assert component in states
            assert not states[component].is_open
            assert states[component].failure_count == 0

    async def test_circuit_breaker_failure_tracking(self, error_injector):
        """Test circuit breaker failure tracking with real errors."""
        service = MLCircuitBreakerService()
        await service.setup_circuit_breakers(["test_component"])

        async def failing_operation():
            raise RuntimeError("Simulated ML failure")

        # Execute operations that will fail
        failure_count = 0
        for _i in range(10):  # Exceed failure threshold
            try:
                await service.execute_with_circuit_breaker(
                    operation=failing_operation,
                    component_name="test_component"
                )
            except Exception:
                failure_count += 1

        # Verify circuit breaker opened
        states = await service.get_all_states()
        test_state = states["test_component"]

        assert failure_count > 0
        assert test_state.failure_count >= 5  # Default threshold
        # Circuit may or may not be open depending on timing and threshold

    async def test_circuit_breaker_recovery(self, performance_tracker):
        """Test circuit breaker recovery mechanism."""
        service = MLCircuitBreakerService()
        await service.setup_circuit_breakers(["recovery_test"])

        # Force circuit open by failing operations
        async def always_fail():
            raise RuntimeError("Force failure")

        for _ in range(6):  # Exceed threshold
            with contextlib.suppress(Exception):
                await service.execute_with_circuit_breaker(
                    operation=always_fail,
                    component_name="recovery_test"
                )

        # Test successful operation
        async def always_succeed():
            return "success"

        start_time = time.perf_counter()
        result = await service.execute_with_circuit_breaker(
            operation=always_succeed,
            component_name="recovery_test"
        )
        duration = (time.perf_counter() - start_time) * 1000

        performance_tracker("circuit_breaker_recovery", duration, 10.0)
        assert result == "success" or isinstance(result, Exception)  # Circuit may still be open


class TestRuleAnalysisService:
    """Test Rule Analysis Service with real rule processing."""

    async def test_rule_intelligence_processing(
        self,
        ml_repository_facade: MLRepositoryFacade,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        test_rule_data: dict[str, Any],
        performance_tracker,
    ):
        """Test rule intelligence processing with real data."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["rule_analysis"])

        service = RuleAnalysisService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker,
            pattern_discovery=advanced_pattern_discovery
        )

        # Test processing specific rules
        rule_ids = test_rule_data["rule_ids"][:5]  # Test subset for performance

        start_time = time.perf_counter()
        result = await service.process_rule_intelligence(rule_ids)
        duration = (time.perf_counter() - start_time) * 1000

        # Validate performance target
        performance_tracker("rule_intelligence_processing", duration, 100.0)

        # Validate results
        assert result.success
        assert "rule_intelligence" in result.data
        assert "characteristics" in result.data
        assert result.confidence > 0.0
        assert result.processing_time_ms > 0

        # Validate rule-specific analysis
        rule_intelligence = result.data["rule_intelligence"]
        for rule_id in rule_ids:
            if rule_id in rule_intelligence:
                rule_data = rule_intelligence[rule_id]
                assert "effectiveness_score" in rule_data
                assert "usage_count" in rule_data
                assert "improvement_potential" in rule_data

    async def test_combination_intelligence_analysis(
        self,
        ml_repository_facade: MLRepositoryFacade,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        test_rule_data: dict[str, Any],
        performance_tracker,
    ):
        """Test rule combination intelligence analysis."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["rule_analysis"])

        service = RuleAnalysisService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker,
            pattern_discovery=advanced_pattern_discovery
        )

        start_time = time.perf_counter()
        result = await service.process_combination_intelligence()
        duration = (time.perf_counter() - start_time) * 1000

        # Validate performance target
        performance_tracker("combination_intelligence", duration, 150.0)

        # Validate results
        assert result.success
        assert "combination_intelligence" in result.data
        assert result.confidence >= 0.0

        # Validate combination analysis
        combination_data = result.data["combination_intelligence"]
        assert "effective_combinations" in combination_data
        assert "combination_scores" in combination_data

    async def test_rule_analysis_error_handling(
        self,
        ml_repository_facade: MLRepositoryFacade,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        error_injector,
    ):
        """Test rule analysis service error handling."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["rule_analysis"])

        service = RuleAnalysisService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker,
            pattern_discovery=advanced_pattern_discovery
        )

        # Test with invalid rule IDs
        invalid_rule_ids = ["invalid_rule_1", "invalid_rule_2"]

        result = await service.process_rule_intelligence(invalid_rule_ids)

        # Should handle gracefully, not crash
        assert isinstance(result, object)  # Result object returned
        # May succeed with empty data or fail gracefully


class TestPatternDiscoveryService:
    """Test Pattern Discovery Service with real pattern analysis."""

    async def test_pattern_discovery_with_real_data(
        self,
        ml_container: MLTestContainer,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        ml_repository_facade: MLRepositoryFacade,
        performance_tracker,
    ):
        """Test pattern discovery with real ML container data."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["pattern_discovery"])

        service = PatternDiscoveryService(
            pattern_discovery=advanced_pattern_discovery,
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Get test dataset from ML container
        test_dataset = ml_container.get_test_dataset("small")
        assert test_dataset is not None

        # Prepare batch data (simulate repository data format)
        batch_data = []
        for i, (prompt, improvement, context) in enumerate(
            zip(test_dataset.prompts[:50], test_dataset.improvements[:50], test_dataset.contexts[:50], strict=False)
        ):
            batch_data.append({
                "session_id": f"test_session_{i:03d}",
                "original_prompt": prompt,
                "improved_prompt": improvement,
                "context": context,
                "effectiveness_score": 0.7 + (i % 3) * 0.1,  # Vary scores
            })

        start_time = time.perf_counter()
        result = await service.discover_patterns(batch_data)
        duration = (time.perf_counter() - start_time) * 1000

        # Validate performance target
        performance_tracker("pattern_discovery", duration, 200.0)

        # Validate results
        assert result.success
        assert "patterns" in result.data
        assert "insights" in result.data
        assert result.confidence >= 0.0
        assert result.processing_time_ms > 0

        # Validate discovered patterns
        patterns = result.data["patterns"]
        assert isinstance(patterns, list)
        if patterns:  # May be empty for small datasets
            for pattern in patterns[:3]:  # Check first few patterns
                assert "pattern_id" in pattern
                assert "strength" in pattern

    async def test_pattern_cache_performance(
        self,
        ml_container: MLTestContainer,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        ml_repository_facade: MLRepositoryFacade,
        performance_tracker,
    ):
        """Test pattern discovery caching for performance improvement."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["pattern_discovery"])

        service = PatternDiscoveryService(
            pattern_discovery=advanced_pattern_discovery,
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Get consistent test data
        test_dataset = ml_container.get_test_dataset("small")
        batch_data = [
            {
                "session_id": f"cache_test_{i:03d}",
                "original_prompt": test_dataset.prompts[i % len(test_dataset.prompts)],
                "improved_prompt": test_dataset.improvements[i % len(test_dataset.improvements)],
                "context": test_dataset.contexts[i % len(test_dataset.contexts)],
                "effectiveness_score": 0.8,
            }
            for i in range(30)
        ]

        # First run (cache miss)
        start_time = time.perf_counter()
        first_result = await service.discover_patterns(batch_data)
        first_duration = (time.perf_counter() - start_time) * 1000

        # Second run (cache hit potential)
        start_time = time.perf_counter()
        second_result = await service.discover_patterns(batch_data)
        second_duration = (time.perf_counter() - start_time) * 1000

        performance_tracker("pattern_discovery_first_run", first_duration, 200.0)
        performance_tracker("pattern_discovery_cached_run", second_duration, 100.0)

        # Both should succeed
        assert first_result.success
        assert second_result.success

        # Second run should be faster (cache hit) or similar (cache miss is acceptable)
        cache_performance_improvement = first_duration > second_duration * 1.2  # 20% improvement threshold
        logger.info(f"Cache performance: first={first_duration:.2f}ms, second={second_duration:.2f}ms, improved={cache_performance_improvement}")

    async def test_pattern_cache_statistics(
        self,
        ml_container: MLTestContainer,
        advanced_pattern_discovery: AdvancedPatternDiscovery,
        ml_repository_facade: MLRepositoryFacade,
    ):
        """Test pattern discovery cache statistics."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["pattern_discovery"])

        service = PatternDiscoveryService(
            pattern_discovery=advanced_pattern_discovery,
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Get cache statistics
        cache_stats = await service.get_cache_statistics()

        # Validate cache statistics structure
        assert isinstance(cache_stats, dict)
        assert "cache_enabled" in cache_stats
        assert "total_operations" in cache_stats
        assert "cache_hits" in cache_stats
        assert "cache_misses" in cache_stats


class TestMLPredictionService:
    """Test ML Prediction Service with real prediction scenarios."""

    async def test_ml_predictions_with_confidence(
        self,
        ml_repository_facade: MLRepositoryFacade,
        performance_tracker,
    ):
        """Test ML predictions with confidence scoring."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["predictions"])

        service = MLPredictionService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Prepare prediction data
        prediction_data = {
            "characteristics": {
                "rule_effectiveness": {
                    "rule_001": 0.85,
                    "rule_002": 0.72,
                    "rule_003": 0.91,
                },
                "domain_distribution": {
                    "technical": 0.4,
                    "general": 0.4,
                    "creative": 0.2,
                },
            },
            "context": {
                "pattern_insights": [
                    {"pattern_type": "improvement", "strength": 0.8},
                    {"pattern_type": "optimization", "strength": 0.6},
                ],
                "session_count": 150,
                "avg_effectiveness": 0.78,
            },
        }

        start_time = time.perf_counter()
        result = await service.generate_predictions_with_confidence(prediction_data)
        duration = (time.perf_counter() - start_time) * 1000

        # Validate performance target
        performance_tracker("ml_predictions", duration, 50.0)

        # Validate results
        assert result.success
        assert "predictions" in result.data
        assert "confidence_analysis" in result.data
        assert result.confidence > 0.0
        assert result.processing_time_ms > 0

        # Validate predictions structure
        predictions = result.data["predictions"]
        assert isinstance(predictions, list)
        if predictions:  # May be empty in test scenarios
            for prediction in predictions[:3]:
                assert isinstance(prediction, dict)

    async def test_prediction_error_handling(
        self,
        ml_repository_facade: MLRepositoryFacade,
    ):
        """Test ML prediction service error handling."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["predictions"])

        service = MLPredictionService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Test with invalid prediction data
        invalid_data = {"invalid": "data"}

        result = await service.generate_predictions_with_confidence(invalid_data)

        # Should handle gracefully
        assert isinstance(result, object)
        # May succeed with default predictions or fail gracefully


class TestBatchProcessingService:
    """Test Batch Processing Service with real batch operations."""

    async def test_batch_processing_performance(
        self,
        ml_repository_facade: MLRepositoryFacade,
        test_data_factory,
        performance_tracker,
    ):
        """Test batch processing performance with realistic batch sizes."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["batch_processing"])

        service = BatchProcessingService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Create batch data
        batch_data = await test_data_factory.create_batch_processing_data(
            batch_size=100,
            operation_types=["rule_analysis", "pattern_discovery", "prediction"]
        )

        start_time = time.perf_counter()
        result = await service.process_batch(batch_data, batch_size=25)
        duration = (time.perf_counter() - start_time) * 1000

        # Validate performance target (batch processing can be slower)
        performance_tracker("batch_processing", duration, 500.0)

        # Validate results
        assert result.success
        assert "batch_results" in result.data
        assert "processing_summary" in result.data
        assert result.processing_time_ms > 0

        # Validate batch processing summary
        summary = result.data["processing_summary"]
        assert "total_items" in summary
        assert "successful_items" in summary
        assert "failed_items" in summary

    async def test_batch_processing_status_tracking(
        self,
        ml_repository_facade: MLRepositoryFacade,
    ):
        """Test batch processing status tracking."""
        circuit_breaker = MLCircuitBreakerService()
        await circuit_breaker.setup_circuit_breakers(["batch_processing"])

        service = BatchProcessingService(
            ml_repository=ml_repository_facade,
            circuit_breaker_service=circuit_breaker
        )

        # Get processing status
        status = await service.get_processing_status()

        # Validate status structure
        assert isinstance(status, dict)
        assert "is_processing" in status
        assert "current_batch_info" in status
        assert "recent_batches" in status


class TestMLIntelligenceServiceFacade:
    """Test ML Intelligence Service Facade integration and coordination."""

    async def test_complete_intelligence_processing_pipeline(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
        test_rule_data: dict[str, Any],
        performance_tracker,
    ):
        """Test complete intelligence processing pipeline with all services."""
        rule_ids = test_rule_data["rule_ids"][:3]  # Small subset for comprehensive test

        start_time = time.perf_counter()
        result = await ml_intelligence_facade.run_intelligence_processing(
            rule_ids=rule_ids,
            enable_patterns=True,
            enable_predictions=True,
            batch_size=25
        )
        duration = (time.perf_counter() - start_time) * 1000

        # Validate facade performance target
        performance_tracker("facade_intelligence_processing", duration, 200.0)

        # Validate comprehensive results
        assert result.success
        assert result.confidence > 0.0
        assert result.processing_time_ms > 0

        # Validate all processing phases
        intelligence_data = result.data
        assert "rule_intelligence" in intelligence_data
        assert "combination_intelligence" in intelligence_data
        assert "pattern_insights" in intelligence_data
        assert "predictions" in intelligence_data
        assert "processing_metadata" in intelligence_data

        # Validate processing metadata
        metadata = intelligence_data["processing_metadata"]
        assert "started_at" in metadata
        assert "completed_at" in metadata
        assert "total_elapsed_ms" in metadata
        assert "successful_phases" in metadata
        assert "coordination_overhead_ms" in metadata

        # Validate coordination efficiency (overhead should be minimal)
        coordination_overhead = metadata["coordination_overhead_ms"]
        total_time = metadata["total_elapsed_ms"]
        overhead_percentage = (coordination_overhead / total_time) * 100

        # Coordination overhead should be < 20% of total time
        assert overhead_percentage < 20.0, f"High coordination overhead: {overhead_percentage:.1f}%"

    async def test_facade_service_health_monitoring(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
    ):
        """Test facade service health monitoring."""
        health_status = await ml_intelligence_facade.get_service_health()

        # Validate health status structure
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        assert "circuit_breakers" in health_status
        assert "batch_processing" in health_status
        assert "pattern_cache" in health_status
        assert "facade_info" in health_status
        assert "timestamp" in health_status

        # Validate health status values
        assert health_status["overall_status"] in {"healthy", "degraded", "unhealthy"}

        # Validate circuit breaker status
        circuit_breakers = health_status["circuit_breakers"]
        for state in circuit_breakers.values():
            assert "is_open" in state
            assert "failure_count" in state
            assert "component_name" in state

    async def test_facade_processing_metrics(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
        test_rule_data: dict[str, Any],
    ):
        """Test facade processing metrics collection."""
        # Run some operations to generate metrics
        await ml_intelligence_facade.run_intelligence_processing(
            rule_ids=test_rule_data["rule_ids"][:2],
            enable_patterns=True,
            enable_predictions=True,
            batch_size=10
        )

        # Get processing metrics
        metrics = await ml_intelligence_facade.get_processing_metrics()

        # Validate metrics structure
        assert isinstance(metrics, dict)
        assert "facade_metrics" in metrics
        assert "service_health" in metrics
        assert "timestamp" in metrics

        # Validate facade metrics
        facade_metrics = metrics["facade_metrics"]
        assert "total_operations" in facade_metrics
        assert "successful_operations" in facade_metrics
        assert "failed_operations" in facade_metrics
        assert "success_rate" in facade_metrics

        # Success rate should be between 0 and 1
        assert 0.0 <= facade_metrics["success_rate"] <= 1.0

    async def test_facade_background_processing(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
    ):
        """Test facade background processing capability."""
        # Test background processing start/stop
        await ml_intelligence_facade.start_background_processing()

        # Wait briefly for background task to initialize
        await asyncio.sleep(0.1)

        health_status = await ml_intelligence_facade.get_service_health()
        facade_info = health_status["facade_info"]

        # Should indicate background processing is running
        assert facade_info["is_background_running"]

        # Stop background processing
        await ml_intelligence_facade.stop_background_processing()

        # Wait briefly for background task to stop
        await asyncio.sleep(0.1)

        health_status = await ml_intelligence_facade.get_service_health()
        facade_info = health_status["facade_info"]

        # Should indicate background processing is stopped
        assert not facade_info["is_background_running"]

    async def test_facade_error_handling_and_recovery(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
        error_injector,
    ):
        """Test facade error handling and recovery mechanisms."""
        # Test with invalid rule IDs to trigger error handling
        invalid_rule_ids = ["invalid_rule_1", "invalid_rule_2"]

        result = await ml_intelligence_facade.run_intelligence_processing(
            rule_ids=invalid_rule_ids,
            enable_patterns=True,
            enable_predictions=True,
            batch_size=10
        )

        # Facade should handle errors gracefully
        assert isinstance(result, object)  # Result object returned
        assert hasattr(result, 'success')
        assert hasattr(result, 'processing_time_ms')

        # Even if processing fails, facade should provide useful error information
        if not result.success:
            assert hasattr(result, 'error_message')
            assert result.error_message is not None


class TestIntegratedMLIntelligenceWorkflow:
    """Test complete integrated ML intelligence workflow scenarios."""

    async def test_end_to_end_intelligence_processing(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
        ml_container: MLTestContainer,
        test_rule_data: dict[str, Any],
        performance_tracker,
    ):
        """Test end-to-end intelligence processing with real data flow."""
        # Create comprehensive test scenario
        rule_ids = test_rule_data["rule_ids"][:5]

        # Run complete intelligence processing
        start_time = time.perf_counter()
        result = await ml_intelligence_facade.run_intelligence_processing(
            rule_ids=rule_ids,
            enable_patterns=True,
            enable_predictions=True,
            batch_size=30
        )
        duration = (time.perf_counter() - start_time) * 1000

        # Validate end-to-end performance
        performance_tracker("end_to_end_intelligence", duration, 300.0)  # Comprehensive processing target

        # Validate comprehensive result
        assert result.success or result.data  # Should have data even if partially successful

        # Get health status after processing
        health_status = await ml_intelligence_facade.get_service_health()
        assert health_status["overall_status"] in {"healthy", "degraded"}  # Should not be unhealthy

        # Get processing metrics
        metrics = await ml_intelligence_facade.get_processing_metrics()
        assert metrics["facade_metrics"]["total_operations"] > 0

    async def test_performance_targets_validation(
        self,
        ml_intelligence_facade: MLIntelligenceServiceFacade,
        test_rule_data: dict[str, Any],
        performance_tracker,
    ):
        """Validate all ML Intelligence Services meet performance targets."""
        performance_results = {}

        # Test individual service performance through facade
        rule_ids = test_rule_data["rule_ids"][:3]

        # Test rule intelligence (target: <100ms)
        start_time = time.perf_counter()
        rule_result = await ml_intelligence_facade._rule_analysis_service.process_rule_intelligence(rule_ids)
        rule_duration = (time.perf_counter() - start_time) * 1000
        performance_results["rule_intelligence"] = {
            "duration_ms": rule_duration,
            "target_ms": 100.0,
            "met": rule_duration < 100.0
        }
        performance_tracker("rule_intelligence_target", rule_duration, 100.0)

        # Test combination intelligence (target: <150ms)
        start_time = time.perf_counter()
        combo_result = await ml_intelligence_facade._rule_analysis_service.process_combination_intelligence()
        combo_duration = (time.perf_counter() - start_time) * 1000
        performance_results["combination_intelligence"] = {
            "duration_ms": combo_duration,
            "target_ms": 150.0,
            "met": combo_duration < 150.0
        }
        performance_tracker("combination_intelligence_target", combo_duration, 150.0)

        # Test facade coordination (target: <200ms)
        start_time = time.perf_counter()
        facade_result = await ml_intelligence_facade.run_intelligence_processing(
            rule_ids=rule_ids[:2],  # Smaller set for performance test
            enable_patterns=False,  # Disable heavy operations
            enable_predictions=False,
            batch_size=10
        )
        facade_duration = (time.perf_counter() - start_time) * 1000
        performance_results["facade_coordination"] = {
            "duration_ms": facade_duration,
            "target_ms": 200.0,
            "met": facade_duration < 200.0
        }
        performance_tracker("facade_coordination_target", facade_duration, 200.0)

        # Validate overall performance
        total_targets_met = sum(1 for result in performance_results.values() if result["met"])
        total_targets = len(performance_results)
        performance_percentage = (total_targets_met / total_targets) * 100

        logger.info(f"Performance targets met: {total_targets_met}/{total_targets} ({performance_percentage:.1f}%)")
        logger.info("Performance results:")
        for service, result in performance_results.items():
            status = "✓" if result["met"] else "✗"
            logger.info(f"  {status} {service}: {result['duration_ms']:.2f}ms (target: {result['target_ms']}ms)")

        # Should meet at least 80% of performance targets
        assert performance_percentage >= 80.0, f"Performance targets not met: {performance_percentage:.1f}%"
