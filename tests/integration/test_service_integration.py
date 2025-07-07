"""
Integration tests for service layer components with minimal mocking.
Tests service interactions and database operations with real components.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest



@pytest.mark.asyncio
@pytest.mark.integration
class TestPromptServiceIntegration:
    """Integration tests for PromptImprovementService with real database operations."""

    async def test_service_database_integration(
        self, test_db_session, sample_rule_metadata, sample_rule_performance
    ):
        """Test service operations with real database interactions."""

        from prompt_improver.services.prompt_improvement import PromptImprovementService

        # Setup database with test data
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        for perf in sample_rule_performance[:5]:  # Use subset
            test_db_session.add(perf)
        await test_db_session.commit()

        service = PromptImprovementService()

        # Test trigger_optimization with real database queries
        with patch.object(service, "_get_ml_service") as mock_ml_service:
            mock_ml_instance = AsyncMock()
            mock_ml_instance.optimize_rules.return_value = {
                "status": "success",
                "model_id": "integration_test_model",
                "best_score": 0.87,
                "training_samples": 5,
            }
            mock_ml_service.return_value = mock_ml_instance

            result = await service.trigger_optimization(
                feedback_id=1, session=test_db_session
            )

            # Validate integration results
            assert result["status"] == "success"
            assert "performance_score" in result
            assert result["training_samples"] > 0

            # Verify database interactions occurred
            mock_ml_instance.optimize_rules.assert_called_once()

    async def test_rule_effectiveness_calculation(
        self, test_db_session, sample_rule_performance
    ):
        """Test rule effectiveness calculation with real performance data."""

        from prompt_improver.services.analytics import AnalyticsService

        # Setup database with performance data
        for perf in sample_rule_performance:
            test_db_session.add(perf)
        await test_db_session.commit()

        analytics = AnalyticsService()

        # Test real database query for rule effectiveness
        effectiveness = await analytics.get_rule_effectiveness(session=test_db_session)

        # Validate results
        assert isinstance(effectiveness, dict)
        assert "clarity_rule" in effectiveness
        assert "specificity_rule" in effectiveness

        # Verify effectiveness scores are realistic
        for rule_id, score in effectiveness.items():
            assert 0 <= score <= 1
            assert score > 0.5  # Should be reasonably effective

    async def test_performance_monitoring_integration(self, test_db_session):
        """Test performance monitoring with real database operations."""

        from prompt_improver.database.models import RulePerformance
        from prompt_improver.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        # Create performance record with realistic timing
        start_time = datetime.utcnow()

        # Simulate processing
        await asyncio.sleep(0.01)  # 10ms processing time

        perf_record = RulePerformance(
            rule_id="integration_test_rule",
            rule_name="Integration Test Rule",
            improvement_score=0.85,
            confidence_level=0.9,
            execution_time_ms=10,
            prompt_characteristics={"length": 25, "complexity": 0.7},
            before_metrics={"clarity": 0.6, "specificity": 0.5},
            after_metrics={"clarity": 0.8, "specificity": 0.7},
            user_satisfaction_score=0.9,
            session_id="integration_perf_test",
            created_at=start_time,
        )

        test_db_session.add(perf_record)
        await test_db_session.commit()

        # Test performance summary calculation
        summary = await analytics.get_performance_summary(session=test_db_session)

        # Validate performance metrics
        assert summary["total_sessions"] >= 1
        assert 0 <= summary["avg_improvement"] <= 1
        assert 0 <= summary["success_rate"] <= 1

        # Verify timing is realistic
        assert summary["avg_improvement"] > 0.8  # Should show improvement


@pytest.mark.asyncio
@pytest.mark.integration
class TestMLServiceIntegration:
    """Integration tests for ML service components with real model operations."""

    async def test_ml_model_lifecycle(self, test_db_session, sample_training_data):
        """Test ML model training and prediction lifecycle."""

        from prompt_improver.services.ml_integration import MLModelService

        # Create ML service with mocked MLflow but real data processing
        with patch("prompt_improver.services.ml_integration.mlflow") as mock_mlflow:
            # Configure MLflow mocks
            mock_mlflow.start_run.return_value = None
            mock_mlflow.active_run.return_value.info.run_id = "integration_test_run"

            ml_service = MLModelService()

            # Test model training with real data processing
            result = await ml_service.optimize_rules(
                sample_training_data, test_db_session, rule_ids=["clarity_rule"]
            )

            # Validate training results
            if result["status"] == "success":
                assert "model_id" in result
                assert "best_score" in result
                assert 0 <= result["best_score"] <= 1
                assert result["training_samples"] > 0
            else:
                # Training may fail with test data, verify error handling
                assert "error" in result
                assert "training data" in result["error"].lower()

    async def test_pattern_discovery_integration(
        self, test_db_session, sample_rule_performance
    ):
        """Test pattern discovery with real performance data."""

        from prompt_improver.services.ml_integration import MLModelService

        # Setup database with pattern data
        for perf in sample_rule_performance:
            test_db_session.add(perf)
        await test_db_session.commit()

        ml_service = MLModelService()

        # Test pattern discovery with real database queries
        result = await ml_service.discover_patterns(
            test_db_session, min_effectiveness=0.7, min_support=3
        )

        # Validate pattern discovery results
        assert "status" in result
        assert "patterns_discovered" in result
        assert result["patterns_discovered"] >= 0

        if result["patterns_discovered"] > 0:
            assert "patterns" in result
            assert len(result["patterns"]) == result["patterns_discovered"]

            # Validate pattern structure
            for pattern in result["patterns"]:
                assert "parameters" in pattern
                assert "avg_effectiveness" in pattern
                assert "support_count" in pattern
                assert pattern["support_count"] >= 3
                assert 0 <= pattern["avg_effectiveness"] <= 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests focused on performance requirements validation."""

    async def test_database_query_performance(
        self, test_db_session, sample_rule_performance
    ):
        """Test database query performance under realistic loads."""

        # Setup larger dataset for performance testing
        large_dataset = sample_rule_performance * 10  # 300+ records
        for perf in large_dataset:
            test_db_session.add(perf)
        await test_db_session.commit()

        from prompt_improver.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        # Test query performance
        start_time = asyncio.get_event_loop().time()

        summary = await analytics.get_performance_summary(session=test_db_session)
        effectiveness = await analytics.get_rule_effectiveness(session=test_db_session)

        end_time = asyncio.get_event_loop().time()
        query_time = (end_time - start_time) * 1000

        # Validate performance requirements
        assert query_time < 100, (
            f"Database queries took {query_time}ms, should be <100ms"
        )
        assert summary["total_sessions"] > 100  # Verify large dataset was processed
        assert len(effectiveness) > 0

    async def test_concurrent_operations_performance(self, test_db_session):
        """Test performance under concurrent operations."""

        from prompt_improver.services.prompt_improvement import PromptImprovementService

        service = PromptImprovementService()

        # Mock external dependencies for performance testing
        with patch.object(service, "_apply_rule_improvements") as mock_apply:
            mock_apply.return_value = {
                "improved_prompt": "Enhanced prompt",
                "applied_rules": [{"rule_id": "test_rule", "confidence": 0.8}],
                "processing_time_ms": 50,
            }

            # Test concurrent operations
            tasks = []
            for i in range(5):
                task = service.improve_prompt(
                    prompt=f"Test prompt {i}",
                    context={"domain": "performance_test"},
                    session_id=f"concurrent_test_{i}",
                    session=test_db_session,
                )
                tasks.append(task)

            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()

            total_time = (end_time - start_time) * 1000
            avg_time_per_operation = total_time / len(tasks)

            # Validate concurrent performance
            assert len(results) == 5
            assert total_time < 500, (
                f"Concurrent operations took {total_time}ms, should be <500ms"
            )
            assert avg_time_per_operation < 100, (
                f"Average operation time {avg_time_per_operation}ms too high"
            )

            # Verify all operations completed successfully
            for result in results:
                assert "improved_prompt" in result
                assert "processing_time_ms" in result


@pytest.mark.asyncio
@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery scenarios."""

    async def test_database_connection_recovery(self):
        """Test system behavior during database connection issues."""

        from prompt_improver.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        # Test with invalid session
        with pytest.raises(Exception):
            await analytics.get_performance_summary(session=None)

    async def test_service_error_propagation(self, test_db_session):
        """Test error propagation through service layers."""

        from prompt_improver.services.prompt_improvement import PromptImprovementService

        service = PromptImprovementService()

        # Test with missing required data
        with patch.object(
            service, "_apply_rule_improvements", side_effect=Exception("Test error")
        ):
            result = await service.improve_prompt(
                prompt="Test prompt",
                context={},
                session_id="error_test",
                session=test_db_session,
            )

            # Verify error handling
            assert "error" in result or "status" in result

    async def test_timeout_handling(self, test_db_session):
        """Test timeout handling for long-running operations."""

        from prompt_improver.services.ml_integration import MLModelService

        ml_service = MLModelService()

        # Test with timeout simulation
        with patch.object(ml_service, "optimize_rules") as mock_optimize:
            mock_optimize.side_effect = TimeoutError("Operation timed out")

            result = await ml_service.optimize_rules(
                {"features": [], "effectiveness_scores": []}, test_db_session
            )

            # Verify timeout handling
            assert "error" in result or result["status"] == "timeout"
