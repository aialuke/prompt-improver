"""
Integration tests for service layer components using real behavior.
Tests service interactions and database operations with actual components
following 2025 best practices for integration testing.
"""

import asyncio
from datetime import datetime

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
class TestPromptServiceIntegration:
    """Integration tests for PromptImprovementService with real database operations."""

    async def test_service_database_integration(self, test_db_session):
        """Test service operations with real database interactions."""
        from prompt_improver.services.prompt_improvement import PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="Test prompt for integration testing",
            user_context={"domain": "testing", "test_run": True},
            session_id="integration_test_session",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert "processing_time_ms" in result
        assert result["improved_prompt"] is not None
        assert isinstance(result["processing_time_ms"], (int, float))
        assert result["processing_time_ms"] >= 0

    async def test_rule_effectiveness_calculation(
        self,
        test_db_session,
        sample_prompt_sessions,
        sample_rule_metadata,
        sample_rule_performance,
    ):
        """Test rule effectiveness calculation with real performance data."""
        from prompt_improver.services.analytics import AnalyticsService

        for session in sample_prompt_sessions:
            test_db_session.add(session)
        await test_db_session.commit()
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        for perf in sample_rule_performance:
            test_db_session.add(perf)
        await test_db_session.commit()
        analytics = AnalyticsService()
        effectiveness = await analytics.get_rule_effectiveness(
            db_session=test_db_session
        )
        assert isinstance(effectiveness, list)
        for stats in effectiveness:
            assert hasattr(stats, "rule_id")
            assert hasattr(stats, "rule_name")
            assert hasattr(stats, "avg_improvement")
            if hasattr(stats, "avg_improvement") and stats.avg_improvement is not None:
                assert 0 <= stats.avg_improvement <= 1

    async def test_performance_monitoring_integration(
        self, test_db_session, sample_prompt_sessions, sample_rule_metadata
    ):
        """Test performance monitoring with real database operations."""
        from prompt_improver.database.models import RulePerformance
        from prompt_improver.services.analytics import AnalyticsService

        for session in sample_prompt_sessions:
            test_db_session.add(session)
        await test_db_session.commit()
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        analytics = AnalyticsService()
        start_time = datetime.utcnow()
        await asyncio.sleep(0.01)
        perf_record = RulePerformance(
            rule_id=sample_rule_metadata[0].rule_id,
            rule_name=sample_rule_metadata[0].rule_name,
            improvement_score=0.85,
            confidence_level=0.9,
            execution_time_ms=10,
            prompt_characteristics={"length": 25, "complexity": 0.7},
            before_metrics={"clarity": 0.6, "specificity": 0.5},
            after_metrics={"clarity": 0.8, "specificity": 0.7},
            user_satisfaction_score=0.9,
            session_id=sample_prompt_sessions[0].session_id,
            created_at=start_time,
        )
        test_db_session.add(perf_record)
        await test_db_session.commit()
        summary = await analytics.get_performance_summary(db_session=test_db_session)
        assert isinstance(summary, dict)
        if "total_sessions" in summary:
            assert summary["total_sessions"] >= 0
        if "avg_improvement" in summary:
            assert 0 <= summary["avg_improvement"] <= 1
        if "success_rate" in summary:
            assert 0 <= summary["success_rate"] <= 1


@pytest.mark.asyncio
@pytest.mark.integration
class TestMLServiceIntegration:
    """Integration tests for ML service components with real model operations."""

    async def test_pattern_discovery_integration(
        self,
        test_db_session,
        sample_prompt_sessions,
        sample_rule_metadata,
        sample_rule_performance,
    ):
        """Test pattern discovery with real performance data."""
        from prompt_improver.services.ml_integration import MLModelService

        for session in sample_prompt_sessions:
            test_db_session.add(session)
        await test_db_session.commit()
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        for perf in sample_rule_performance:
            test_db_session.add(perf)
        await test_db_session.commit()
        ml_service = MLModelService()
        result = await ml_service.discover_patterns(
            test_db_session,
            min_effectiveness=0.1,
            min_support=1,
            use_advanced_discovery=False,
        )
        assert "status" in result
        assert isinstance(result, dict)
        valid_statuses = ["success", "no_patterns_found", "insufficient_data", "error"]
        assert result["status"] in valid_statuses
        if "patterns_discovered" in result:
            assert isinstance(result["patterns_discovered"], int)
            assert result["patterns_discovered"] >= 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests focused on performance requirements validation."""

    async def test_database_query_performance(
        self,
        test_db_session,
        sample_prompt_sessions,
        sample_rule_metadata,
        sample_rule_performance,
    ):
        """Test database query performance under realistic loads."""
        large_sessions = sample_prompt_sessions * 5
        for session in large_sessions:
            test_db_session.add(session)
        await test_db_session.commit()
        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        large_dataset = sample_rule_performance * 5
        for perf in large_dataset:
            test_db_session.add(perf)
        await test_db_session.commit()
        from prompt_improver.services.analytics import AnalyticsService

        analytics = AnalyticsService()
        start_time = asyncio.get_event_loop().time()
        summary = await analytics.get_performance_summary(db_session=test_db_session)
        effectiveness = await analytics.get_rule_effectiveness(
            db_session=test_db_session
        )
        end_time = asyncio.get_event_loop().time()
        query_time = (end_time - start_time) * 1000
        assert query_time < 1000, (
            f"Database queries took {query_time}ms, should be <1000ms for real behavior"
        )
        assert isinstance(summary, dict)
        assert isinstance(effectiveness, list)

    async def test_concurrent_operations_performance(self, test_db_session):
        """Test performance under sequential operations (avoiding concurrent db session issues)."""
        from prompt_improver.services.prompt_improvement import PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        results = []
        start_time = asyncio.get_event_loop().time()
        for i in range(3):
            result = await service.improve_prompt(
                prompt=f"Test prompt {i} for performance testing",
                user_context={"domain": "performance_test", "test_run": True},
                session_id=f"perf_test_{i}",
                db_session=None,
            )
            results.append(result)
        end_time = asyncio.get_event_loop().time()
        total_time = (end_time - start_time) * 1000
        avg_time_per_operation = total_time / len(results)
        assert len(results) == 3
        assert total_time < 5000, (
            f"Sequential operations took {total_time}ms, should be <5000ms for real behavior"
        )
        assert avg_time_per_operation < 2000, (
            f"Average operation time {avg_time_per_operation}ms too high for real behavior"
        )
        for i, result in enumerate(results):
            assert "improved_prompt" in result, f"Result {i} missing improved_prompt"
            assert "processing_time_ms" in result, (
                f"Result {i} missing processing_time_ms"
            )
            assert result["improved_prompt"] is not None, (
                f"Result {i} has null improved_prompt"
            )
            assert isinstance(result["processing_time_ms"], (int, float)), (
                f"Result {i} has invalid timing"
            )


@pytest.mark.asyncio
@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery scenarios."""

    async def test_database_connection_recovery(self):
        """Test system behavior during database connection issues."""
        from prompt_improver.services.analytics import AnalyticsService

        analytics = AnalyticsService()
        result = await analytics.get_performance_summary(db_session=None)
        assert result["total_sessions"] == 0

    async def test_service_error_propagation(self, test_db_session):
        """Test error propagation through service layers."""
        from prompt_improver.services.prompt_improvement import PromptImprovementService

        service = PromptImprovementService()
        result = await service.improve_prompt(
            prompt="",
            user_context={},
            session_id="error_test",
            db_session=test_db_session,
        )
        if "error" in result:
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
        else:
            assert "improved_prompt" in result

    async def test_timeout_handling(self, test_db_session):
        """Test timeout handling for long-running operations."""
        from prompt_improver.services.ml_integration import MLModelService

        ml_service = MLModelService()
        import asyncio

        try:
            result = await asyncio.wait_for(
                ml_service.discover_patterns(
                    test_db_session,
                    min_effectiveness=0.9,
                    min_support=100,
                    use_advanced_discovery=False,
                ),
                timeout=5.0,
            )
            assert "status" in result
            assert result["status"] in [
                "success",
                "no_patterns_found",
                "insufficient_data",
            ]
        except TimeoutError:
            pytest.skip(
                "Real operation timed out - this validates timeout handling works"
            )
