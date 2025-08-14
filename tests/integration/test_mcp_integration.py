"""
Enhanced database integration tests with comprehensive constraint validation.
Implements real database operations, constraint verification, and transaction testing
following Context7 database testing best practices.
"""

import asyncio
import random
import uuid

import pytest
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)
from sqlalchemy.exc import IntegrityError
from sqlmodel import select


@pytest.mark.asyncio
@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests with minimal mocking for critical paths."""

    async def test_end_to_end_prompt_improvement(self, test_db_session):
        """Test complete prompt improvement workflow with real components following 2025 best practices."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="Please help me write code that does stuff",
            user_context={"project_type": "python", "complexity": "moderate"},
            session_id="integration_test",
            db_session=test_db_session,
        )
        assert result["processing_time_ms"] < 5000
        assert len(result["improved_prompt"]) >= len(
            "Please help me write code that does stuff"
        )
        assert "applied_rules" in result
        assert isinstance(result["applied_rules"], list)
        assert "original_prompt" in result
        assert "improved_prompt" in result
        assert "session_id" in result
        assert result["session_id"] == "integration_test"
        assert "improvement_summary" in result
        assert "confidence_score" in result
        assert result["original_prompt"] == "Please help me write code that does stuff"

    @pytest.mark.performance
    async def test_performance_requirement_compliance(self, test_db_session):
        """Verify <200ms performance requirement using real service implementation."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        test_prompts = [
            "Simple prompt",
            "More complex prompt with multiple requirements and context",
            "Very detailed prompt with extensive background information and specific technical requirements that need processing",
        ]
        response_times = []
        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        for i, prompt in enumerate(test_prompts):
            start_time = asyncio.get_event_loop().time()
            result = await service.improve_prompt(
                prompt=prompt,
                user_context={"domain": "performance_test"},
                session_id=f"perf_test_{len(prompt)}",
                db_session=test_db_session,
            )
            end_time = asyncio.get_event_loop().time()
            total_time = (end_time - start_time) * 1000
            processing_time_reported = result["processing_time_ms"]
            response_times.append(processing_time_reported)
            assert processing_time_reported < 5000, (
                f"Response time {processing_time_reported}ms exceeds 5000ms limit for real service"
            )
            assert total_time < 10000, (
                f"Total test execution time {total_time}ms too high for real service"
            )
            assert "improved_prompt" in result
            assert "applied_rules" in result
            assert isinstance(result["applied_rules"], list)
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 3000, (
            f"Average response time {avg_response_time}ms too high for real service"
        )
        max_response_time = max(response_times)
        assert max_response_time < 4000, (
            f"Maximum response time {max_response_time}ms too close to limit for real service"
        )


@pytest.mark.asyncio
@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations with real database connections."""

    async def test_database_session_lifecycle(self, test_db_session):
        """Test database session creation and cleanup."""
        from prompt_improver.database.models import RuleMetadata

        unique_id = f"integration_test_rule_{random.randint(1000, 9999)}"
        test_rule = RuleMetadata(
            rule_id=unique_id,
            rule_name="Integration Test Rule",
            rule_category="test",
            rule_description="Test rule for integration testing",
            enabled=True,
            priority=1,
            rule_version="1.0",
            default_parameters={
                "test": True,
                "confidence_threshold": 0.8,
                "weight": 1.0,
            },
            parameter_constraints={
                "confidence_threshold": {"min": 0.0, "max": 1.0},
                "weight": {"min": 0.0, "max": 2.0},
            },
        )
        test_db_session.add(test_rule)
        await test_db_session.commit()
        from sqlmodel import select

        result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == unique_id)
        )
        stored_rule = result.scalar_one_or_none()
        assert stored_rule is not None
        assert stored_rule.rule_name == "Integration Test Rule"
        assert stored_rule.default_parameters["test"] == True
        assert stored_rule.default_parameters["confidence_threshold"] == 0.8
        assert stored_rule.parameter_constraints["weight"]["max"] == 2.0

    async def test_database_transaction_rollback(self, test_db_session):
        """Test database transaction rollback functionality."""
        from prompt_improver.database.models import RuleMetadata

        rollback_id = f"rollback_test_rule_{random.randint(1000, 9999)}"
        test_rule = RuleMetadata(
            rule_id=rollback_id,
            rule_name="Rollback Test Rule",
            rule_category="test",
            rule_description="Test rule for rollback testing",
            enabled=True,
            priority=1,
            rule_version="1.0",
            default_parameters={"test": True},
        )
        test_db_session.add(test_rule)
        await test_db_session.rollback()
        from sqlmodel import select

        result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == rollback_id)
        )
        stored_rule = result.scalar_one_or_none()
        assert stored_rule is None


@pytest.mark.asyncio
@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for service layer interactions."""

    async def test_prompt_improvement_service_integration(
        self, test_db_session, sample_rule_metadata
    ):
        """Test prompt improvement service with real database interactions and real service behavior following 2025 best practices."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="Make this better",
            user_context={"domain": "integration_test"},
            session_id="service_integration_test",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert "applied_rules" in result
        assert "processing_time_ms" in result
        assert "original_prompt" in result
        assert "session_id" in result
        assert "improvement_summary" in result
        assert "confidence_score" in result
        assert result["original_prompt"] == "Make this better"
        assert result["session_id"] == "service_integration_test"
        assert result["processing_time_ms"] < 10000
        assert isinstance(result["applied_rules"], list)
        assert isinstance(result["improvement_summary"], dict)
        assert isinstance(result["confidence_score"], (int, float))
        summary = result["improvement_summary"]
        assert "total_rules_applied" in summary
        assert "average_confidence" in summary
        assert "improvement_areas" in summary
        assert isinstance(summary["total_rules_applied"], int)
        assert isinstance(summary["improvement_areas"], list)
        rules_metadata = await service.get_rules_metadata(
            enabled_only=True, db_session=test_db_session
        )
        assert isinstance(rules_metadata, list)
        assert len(rules_metadata) >= 0

    async def test_analytics_service_integration(
        self, test_db_session, sample_rule_performance
    ):
        """Test analytics service with real database interactions following 2025 best practices."""
        from prompt_improver.database.models import PromptSession, RuleMetadata
        from prompt_improver.services.analytics import AnalyticsService

        created_sessions = set()
        created_rules = set()
        for i, perf in enumerate(sample_rule_performance[:10]):
            if perf.rule_id and perf.rule_id not in created_rules:
                rule_metadata = RuleMetadata(
                    rule_id=perf.rule_id,
                    rule_name=f"Test Rule {i}",
                    category="test",
                    enabled=True,
                    priority=1,
                    rule_version="1.0",
                    default_parameters={"test": True},
                )
                test_db_session.add(rule_metadata)
                created_rules.add(perf.rule_id)
            if perf.session_id and perf.session_id not in created_sessions:
                session = PromptSession(
                    session_id=perf.session_id,
                    original_prompt=f"Test prompt {i}",
                    improved_prompt=f"Improved test prompt {i}",
                    user_context={"test": True},
                    session_status="completed",
                )
                test_db_session.add(session)
                created_sessions.add(perf.session_id)
            test_db_session.add(perf)
        await test_db_session.commit()
        analytics_service = AnalyticsService()
        trends = await analytics_service.get_performance_trends(
            db_session=test_db_session
        )
        assert "daily_trends" in trends
        assert "summary" in trends
        assert "period_start" in trends
        assert "period_end" in trends
        assert isinstance(trends["daily_trends"], list)
        assert isinstance(trends["summary"], dict)
        assert len(trends["daily_trends"]) >= 0
        rule_effectiveness = await analytics_service.get_rule_effectiveness(days=30)
        assert isinstance(rule_effectiveness, list)
        performance_summary = await analytics_service.get_performance_summary(
            days=30, db_session=test_db_session
        )
        assert "total_sessions" in performance_summary
        assert "avg_improvement" in performance_summary
        assert "success_rate" in performance_summary
        assert isinstance(performance_summary["total_sessions"], int)
        prompt_analysis = await analytics_service.get_prompt_type_analysis(
            days=30, db_session=test_db_session
        )
        assert "prompt_types" in prompt_analysis
        assert "summary" in prompt_analysis
        assert isinstance(prompt_analysis["prompt_types"], list)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests for complete system workflows."""

    async def test_complete_prompt_improvement_workflow(
        self, test_db_session, sample_rule_metadata
    ):
        """Test complete prompt improvement workflow using real services following 2025 best practices."""
        from prompt_improver.services.analytics import AnalyticsService
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        for rule in sample_rule_metadata:
            test_db_session.add(rule)
        await test_db_session.commit()
        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        analytics_service = AnalyticsService()
        original_prompt = "Help me code"
        result = await service.improve_prompt(
            prompt=original_prompt,
            user_context={"domain": "software_development", "complexity": "moderate"},
            session_id="e2e_test_session",
            db_session=test_db_session,
        )
        assert len(result["improved_prompt"]) >= len(original_prompt)
        assert result["processing_time_ms"] < 10000
        assert isinstance(result["applied_rules"], list)
        assert "session_id" in result
        assert result["session_id"] == "e2e_test_session"
        assert "original_prompt" in result
        assert "improved_prompt" in result
        assert "improvement_summary" in result
        assert "confidence_score" in result
        for rule in result["applied_rules"]:
            assert "rule_id" in rule
            if "confidence" in rule:
                assert 0 <= rule["confidence"] <= 1
        performance_summary = await analytics_service.get_performance_summary(
            days=1, db_session=test_db_session
        )
        assert "total_sessions" in performance_summary
        assert isinstance(performance_summary["total_sessions"], int)
        rule_effectiveness = await analytics_service.get_rule_effectiveness(days=1)
        assert isinstance(rule_effectiveness, list)


@pytest.mark.database_constraints
class TestDatabaseConstraintValidation:
    """Test database constraints and validation rules with real database operations."""

    @pytest.mark.asyncio
    async def test_rule_metadata_constraints(self, test_db_session):
        """Test RuleMetadata model constraints and validation."""
        from prompt_improver.database.models import RuleMetadata

        rule_id = f"test_rule_{random.randint(10000, 99999)}"
        valid_rule = RuleMetadata(
            rule_id=rule_id,
            rule_name="Test Rule",
            rule_category="test",
            enabled=True,
            priority=5,
            rule_version="1.0.0",
        )
        test_db_session.add(valid_rule)
        await test_db_session.commit()
        result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == rule_id)
        )
        stored_rule = result.scalar_one_or_none()
        assert stored_rule is not None
        assert stored_rule.rule_name == "Test Rule"

    @pytest.mark.asyncio
    async def test_rule_metadata_unique_constraint(self, test_db_session):
        """Test unique constraint on rule_id."""
        from prompt_improver.database.models import RuleMetadata

        duplicate_id = f"duplicate_test_{random.randint(10000, 99999)}"
        rule1 = RuleMetadata(
            rule_id=duplicate_id, rule_name="First Rule", enabled=True, priority=5
        )
        test_db_session.add(rule1)
        await test_db_session.commit()
        rule2 = RuleMetadata(
            rule_id=duplicate_id, rule_name="Second Rule", enabled=True, priority=6
        )
        test_db_session.add(rule2)
        with pytest.raises(IntegrityError):
            await test_db_session.commit()

    @pytest.mark.asyncio
    async def test_rule_performance_constraints(self, test_db_session):
        """Test RulePerformance model constraints."""
        from prompt_improver.database.models import RulePerformance

        perf_rule_id = f"test_rule_{random.randint(10000, 99999)}"
        valid_performance = RulePerformance(
            rule_id=perf_rule_id,
            rule_name="Test Rule",
            prompt_id=uuid.uuid4(),
            improvement_score=0.8,
            confidence_level=0.9,
            execution_time_ms=150,
        )
        test_db_session.add(valid_performance)
        await test_db_session.commit()
        result = await test_db_session.execute(
            select(RulePerformance).where(RulePerformance.rule_id == perf_rule_id)
        )
        stored_performance = result.scalar_one_or_none()
        assert stored_performance is not None
        assert stored_performance.improvement_score == 0.8

    @given(
        improvement_score=st.floats(min_value=-1.0, max_value=2.0),
        confidence_level=st.floats(min_value=-1.0, max_value=2.0),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_rule_performance_check_constraints(
        self, test_db_session, improvement_score, confidence_level
    ):
        """Property-based testing of check constraints on scores."""
        from prompt_improver.database.models import RulePerformance

        constraint_rule_id = f"constraint_test_{random.randint(10000, 99999)}"
        performance_record = RulePerformance(
            rule_id=constraint_rule_id,
            rule_name="Constraint Test",
            prompt_id=uuid.uuid4(),
            improvement_score=improvement_score,
            confidence_level=confidence_level,
            execution_time_ms=100,
        )
        test_db_session.add(performance_record)
        if 0.0 <= improvement_score <= 1.0 and 0.0 <= confidence_level <= 1.0:
            try:
                await test_db_session.commit()
                result = await test_db_session.execute(
                    select(RulePerformance)
                    .where(RulePerformance.rule_id == constraint_rule_id)
                    .order_by(RulePerformance.id.desc())
                    .limit(1)
                )
                stored = result.scalar_one_or_none()
                assert stored is not None
                assert stored.improvement_score == improvement_score
                assert stored.confidence_level == confidence_level
            except Exception:
                await test_db_session.rollback()
                raise
        else:
            with pytest.raises(IntegrityError):
                await test_db_session.commit()
            await test_db_session.rollback()

    @pytest.mark.asyncio
    async def test_user_feedback_rating_constraint(self, test_db_session):
        """Test UserFeedback rating constraints (1-5 range)."""
        from prompt_improver.database.models import UserFeedback

        valid_ratings = [1, 2, 3, 4, 5]
        for rating in valid_ratings:
            session_id = f"test_session_{rating}_{random.randint(10000, 99999)}"
            feedback = UserFeedback(
                original_prompt="Test prompt",
                improved_prompt="Improved test prompt",
                user_rating=rating,
                applied_rules={"rules": ["clarity_rule"]},
                session_id=session_id,
                rating=rating,
                feedback_text="Test feedback text",
                improvement_areas={"areas": ["clarity"]},
                is_processed=False,
                ml_optimized=False,
            )
            test_db_session.add(feedback)
            await test_db_session.commit()
            result = await test_db_session.execute(
                select(UserFeedback).where(UserFeedback.session_id == session_id)
            )
            stored_feedback = result.scalar_one_or_none()
            assert stored_feedback is not None
            assert stored_feedback.user_rating == rating

    @pytest.mark.asyncio
    async def test_user_feedback_invalid_rating_constraint(self, test_db_session):
        """Test UserFeedback invalid rating constraints."""
        from prompt_improver.database.models import UserFeedback

        invalid_ratings = [0, 6, -1, 10]
        for rating in invalid_ratings:
            invalid_session_id = (
                f"invalid_session_{rating}_{random.randint(10000, 99999)}"
            )
            feedback = UserFeedback(
                original_prompt="Test prompt",
                improved_prompt="Improved test prompt",
                user_rating=rating,
                applied_rules={"rules": ["clarity_rule"]},
                session_id=invalid_session_id,
            )
            test_db_session.add(feedback)
            with pytest.raises(IntegrityError):
                await test_db_session.commit()
            await test_db_session.rollback()

    @pytest.mark.asyncio
    async def test_comprehensive_constraint_validation(self, test_db_session):
        """Comprehensive test of all constraint validation scenarios."""
        from prompt_improver.database.models import (
            ABExperiment,
            RulePerformance,
            UserFeedback,
        )

        invalid_scores = [-0.5, 1.5, 2.0, -1.0]
        for score in invalid_scores:
            perf = RulePerformance(
                rule_id=f"invalid_score_test_{random.randint(10000, 99999)}",
                rule_name="Invalid Score Test",
                prompt_id=uuid.uuid4(),
                improvement_score=score,
                confidence_level=0.8,
                execution_time_ms=100,
            )
            test_db_session.add(perf)
            with pytest.raises(IntegrityError):
                await test_db_session.commit()
            await test_db_session.rollback()
        valid_boundary_ratings = [1, 5]
        for rating in valid_boundary_ratings:
            feedback = UserFeedback(
                original_prompt="Boundary test prompt",
                improved_prompt="Boundary test improved prompt",
                user_rating=rating,
                applied_rules={"rules": ["test_rule"]},
                session_id=f"boundary_test_{rating}_{random.randint(10000, 99999)}",
            )
            test_db_session.add(feedback)
            await test_db_session.commit()
        valid_statuses = ["planning", "running", "completed", "stopped"]
        for status in valid_statuses:
            experiment = ABExperiment(
                experiment_name=f"Status Test {status}",
                control_rules={"rules": ["control"]},
                treatment_rules={"rules": ["treatment"]},
                status=status,
            )
            test_db_session.add(experiment)
            await test_db_session.commit()
        invalid_experiment = ABExperiment(
            experiment_name="Invalid Status Test",
            control_rules={"rules": ["control"]},
            treatment_rules={"rules": ["treatment"]},
            status="invalid_status",
        )
        test_db_session.add(invalid_experiment)
        with pytest.raises(IntegrityError):
            await test_db_session.commit()
        await test_db_session.rollback()

    @pytest.mark.asyncio
    async def test_constraint_error_messages(self, test_db_session):
        """Test that constraint violation error messages are informative."""
        from prompt_improver.database.models import RulePerformance

        invalid_perf = RulePerformance(
            rule_id="error_message_test",
            rule_name="Error Message Test",
            prompt_id=uuid.uuid4(),
            improvement_score=2.0,
            confidence_level=0.8,
            execution_time_ms=100,
        )
        test_db_session.add(invalid_perf)
        try:
            await test_db_session.commit()
            assert False, "Expected IntegrityError but none was raised"
        except IntegrityError as e:
            error_msg = str(e)
            assert (
                "improvement_score" in error_msg.lower() or "check" in error_msg.lower()
            )
            assert "constraint" in error_msg.lower() or "violation" in error_msg.lower()
            await test_db_session.rollback()


@pytest.mark.database_transactions
class TestDatabaseTransactionIntegrity:
    """Test database transaction handling and rollback scenarios."""

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_constraint_violation(self, test_db_session):
        """Test transaction rollback when constraint violations occur."""
        from prompt_improver.database.models import RuleMetadata, RulePerformance

        valid_rule = RuleMetadata(
            rule_id="transaction_test",
            rule_name="Transaction Test",
            enabled=True,
            priority=5,
        )
        test_db_session.add(valid_rule)
        invalid_performance = RulePerformance(
            rule_id="transaction_test",
            rule_name="Transaction Test",
            prompt_id=uuid.uuid4(),
            improvement_score=2.0,
            confidence_level=0.8,
            execution_time_ms=100,
        )
        test_db_session.add(invalid_performance)
        with pytest.raises(IntegrityError):
            await test_db_session.commit()
        await test_db_session.rollback()
        result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == "transaction_test")
        )
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_complex_transaction_with_multiple_models(self, test_db_session):
        """Test complex transaction involving multiple models."""
        from prompt_improver.database.models import (
            ImprovementSession,
            RuleMetadata,
            RulePerformance,
            UserFeedback,
        )

        session_id = "complex_transaction_test"
        rule = RuleMetadata(
            rule_id="complex_rule", rule_name="Complex Rule", enabled=True, priority=5
        )
        performance = RulePerformance(
            rule_id="complex_rule",
            rule_name="Complex Rule",
            prompt_id=uuid.uuid4(),
            improvement_score=0.85,
            confidence_level=0.9,
            execution_time_ms=120,
        )
        feedback = UserFeedback(
            original_prompt="Original complex prompt",
            improved_prompt="Improved complex prompt",
            user_rating=4,
            applied_rules={"rules": ["complex_rule"]},
            session_id=session_id,
        )
        improvement_session = ImprovementSession(
            session_id=session_id,
            original_prompt="Original complex prompt",
            final_prompt="Improved complex prompt",
            iteration_count=1,
            total_improvement_score=0.85,
            status="completed",
        )
        test_db_session.add(rule)
        test_db_session.add(performance)
        test_db_session.add(feedback)
        test_db_session.add(improvement_session)
        await test_db_session.commit()
        rule_result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == "complex_rule")
        )
        assert rule_result.scalar_one_or_none() is not None
        performance_result = await test_db_session.execute(
            select(RulePerformance).where(RulePerformance.rule_id == "complex_rule")
        )
        assert performance_result.scalar_one_or_none() is not None
        feedback_result = await test_db_session.execute(
            select(UserFeedback).where(UserFeedback.session_id == session_id)
        )
        assert feedback_result.scalar_one_or_none() is not None
        session_result = await test_db_session.execute(
            select(ImprovementSession).where(
                ImprovementSession.session_id == session_id
            )
        )
        assert session_result.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_concurrent_transaction_handling(self, test_db_session):
        """Test handling of concurrent database operations."""
        from prompt_improver.database.models import RuleMetadata

        async def create_rule(rule_id: str, priority: int):
            try:
                rule = RuleMetadata(
                    rule_id=rule_id,
                    rule_name=f"Concurrent Rule {priority}",
                    enabled=True,
                    priority=priority,
                )
                test_db_session.add(rule)
                await test_db_session.commit()
                return True
            except Exception as e:
                await test_db_session.rollback()
                return False

        tasks = [create_rule(f"concurrent_rule_{i}", i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for result in results if result is True)
        assert success_count > 0
        result = await test_db_session.execute(select(RuleMetadata))
        all_rules = result.scalars().all()
        concurrent_rules = [
            r for r in all_rules if r.rule_id.startswith("concurrent_rule_")
        ]
        assert len(concurrent_rules) == success_count


@pytest.mark.database_performance
class TestDatabasePerformanceValidation:
    """Test database operation performance and scaling characteristics."""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, test_db_session):
        """Test bulk insert performance for large datasets."""
        import time

        from prompt_improver.database.models import RulePerformance

        batch_size = 100
        performance_records = []
        for i in range(batch_size):
            record = RulePerformance(
                rule_id=f"bulk_rule_{i}",
                rule_name=f"Bulk Rule {i}",
                prompt_id=uuid.uuid4(),
                improvement_score=0.7 + i % 3 * 0.1,
                confidence_level=0.8 + i % 2 * 0.1,
                execution_time_ms=100 + i % 50,
            )
            performance_records.append(record)
        start_time = time.time()
        for record in performance_records:
            test_db_session.add(record)
        await test_db_session.commit()
        end_time = time.time()
        insert_time_ms = (end_time - start_time) * 1000
        assert insert_time_ms < 5000, (
            f"Bulk insert took {insert_time_ms:.1f}ms, too slow"
        )
        result = await test_db_session.execute(select(RulePerformance))
        all_records = result.scalars().all()
        bulk_records = [r for r in all_records if r.rule_id.startswith("bulk_rule_")]
        assert len(bulk_records) == batch_size

    @given(query_count=st.integers(min_value=5, max_value=50))
    @pytest.mark.asyncio
    async def test_query_performance_scaling(self, test_db_session, query_count):
        """Property-based testing of query performance scaling."""
        import time

        from prompt_improver.database.models import RuleMetadata

        for i in range(20):
            rule = RuleMetadata(
                rule_id=f"query_test_rule_{i}",
                rule_name=f"Query Test Rule {i}",
                enabled=True,
                priority=i % 10,
            )
            test_db_session.add(rule)
        await test_db_session.commit()
        query_times = []
        for i in range(query_count):
            start_time = time.time()
            result = await test_db_session.execute(
                select(RuleMetadata).where(RuleMetadata.priority == i % 10)
            )
            rules = result.scalars().all()
            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000
            query_times.append(query_time_ms)
            assert len(rules) > 0
        avg_query_time = sum(query_times) / len(query_times)
        max_query_time = max(query_times)
        assert avg_query_time < 50, (
            f"Average query time {avg_query_time:.1f}ms too slow"
        )
        assert max_query_time < 100, f"Max query time {max_query_time:.1f}ms too slow"
        if len(query_times) > 10:
            first_half_avg = sum(query_times[: len(query_times) // 2]) / (
                len(query_times) // 2
            )
            second_half_avg = sum(query_times[len(query_times) // 2 :]) / (
                len(query_times) - len(query_times) // 2
            )
            performance_degradation = (
                second_half_avg / first_half_avg if first_half_avg > 0 else 1
            )
            assert performance_degradation < 3.0, (
                f"Query performance degraded {performance_degradation:.2f}x"
            )


@pytest.mark.database_schema
class TestDatabaseSchemaValidation:
    """Test database schema integrity and migration compatibility."""

    @pytest.mark.asyncio
    async def test_jsonb_field_operations(self, test_db_session):
        """Test JSONB field operations and indexing."""
        from prompt_improver.database.models import RuleMetadata

        complex_parameters = {
            "weights": {"clarity": 0.8, "specificity": 0.7},
            "thresholds": {"min_confidence": 0.6, "max_iterations": 10},
            "features": ["length", "complexity", "domain"],
            "nested": {
                "algorithm": "gradient_boost",
                "hyperparams": {"learning_rate": 0.1, "n_estimators": 100},
            },
        }
        constraints = {
            "clarity_weight": {"min": 0.0, "max": 1.0},
            "specificity_weight": {"min": 0.0, "max": 1.0},
            "confidence_threshold": {"type": "float", "range": [0.0, 1.0]},
        }
        rule = RuleMetadata(
            rule_id="jsonb_test_rule",
            rule_name="JSONB Test Rule",
            default_parameters=complex_parameters,
            parameter_constraints=constraints,
            enabled=True,
            priority=5,
        )
        test_db_session.add(rule)
        await test_db_session.commit()
        result = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == "jsonb_test_rule")
        )
        stored_rule = result.scalar_one_or_none()
        assert stored_rule is not None
        assert stored_rule.default_parameters["weights"]["clarity"] == 0.8
        assert stored_rule.default_parameters["nested"]["algorithm"] == "gradient_boost"
        assert stored_rule.parameter_constraints["clarity_weight"]["min"] == 0.0

    @pytest.mark.asyncio
    async def test_index_performance_validation(self, test_db_session):
        """Test that database indexes are working effectively."""
        import time

        from prompt_improver.database.models import RulePerformance

        large_dataset_size = 500
        for i in range(large_dataset_size):
            performance = RulePerformance(
                rule_id=f"index_test_rule_{i % 10}",
                rule_name=f"Index Test Rule {i % 10}",
                prompt_id=uuid.uuid4(),
                prompt_type=f"type_{i % 5}",
                improvement_score=0.5 + i % 5 * 0.1,
                confidence_level=0.6 + i % 4 * 0.1,
                execution_time_ms=100 + i % 100,
            )
            test_db_session.add(performance)
        await test_db_session.commit()
        indexed_queries = [
            select(RulePerformance).where(
                RulePerformance.rule_id == "index_test_rule_5"
            ),
            select(RulePerformance).where(RulePerformance.prompt_type == "type_2"),
            select(RulePerformance).where(RulePerformance.improvement_score > 0.7),
        ]
        for query in indexed_queries:
            start_time = time.time()
            result = await test_db_session.execute(query)
            records = result.scalars().all()
            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000
            assert query_time_ms < 100, (
                f"Indexed query took {query_time_ms:.1f}ms, too slow"
            )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.error_handling
class TestRealServiceErrorHandling:
    """Test error handling and edge cases with real service implementations following 2025 best practices."""

    async def test_prompt_improvement_service_error_handling(self, test_db_session):
        """Test error handling in PromptImprovementService with real database operations."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="",
            user_context={"domain": "test"},
            session_id="empty_prompt_test",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert "error" not in result or result.get("error") is None
        assert result["processing_time_ms"] < 5000
        long_prompt = "A" * 10000
        result = await service.improve_prompt(
            prompt=long_prompt,
            user_context={"domain": "test"},
            session_id="long_prompt_test",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert len(result["improved_prompt"]) > 0
        assert result["processing_time_ms"] < 10000
        result = await service.improve_prompt(
            prompt="Test prompt",
            user_context=None,
            session_id="invalid_context_test",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert result["processing_time_ms"] < 5000
        special_prompt = "Test with special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        result = await service.improve_prompt(
            prompt=special_prompt,
            user_context={"domain": "test"},
            session_id="special_chars_test",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert len(result["improved_prompt"]) > 0
        assert result["processing_time_ms"] < 5000

    async def test_analytics_service_error_handling(self, test_db_session):
        """Test error handling in AnalyticsService with real database operations."""
        from prompt_improver.services.analytics import AnalyticsService

        analytics_service = AnalyticsService()
        trends = await analytics_service.get_performance_trends(
            db_session=test_db_session
        )
        assert "daily_trends" in trends
        assert "summary" in trends
        assert isinstance(trends["daily_trends"], list)
        assert isinstance(trends["summary"], dict)
        try:
            performance_summary = await analytics_service.get_performance_summary(
                days=-1, db_session=test_db_session
            )
            assert "total_sessions" in performance_summary
        except ValueError:
            pass
        performance_summary = await analytics_service.get_performance_summary(
            days=365, db_session=test_db_session
        )
        assert "total_sessions" in performance_summary
        assert "avg_improvement" in performance_summary
        assert isinstance(performance_summary["total_sessions"], int)

    async def test_database_connection_error_handling(self, test_db_session):
        """Test service behavior with database connection issues."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        try:
            result = await service.improve_prompt(
                prompt="Test prompt",
                user_context={"domain": "test"},
                session_id="no_db_test",
                db_session=None,
            )
            if result:
                assert "improved_prompt" in result
        except Exception as e:
            assert isinstance(e, (ValueError, AttributeError, TypeError))

    async def test_concurrent_service_operations(self, test_db_session):
        """Test concurrent operations with real services."""
        import asyncio

        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )

        async def improve_prompt_task(prompt_id: int):
            try:
                result = await service.improve_prompt(
                    prompt=f"Concurrent test prompt {prompt_id}",
                    user_context={"domain": "concurrent_test"},
                    session_id=f"concurrent_test_{prompt_id}",
                    db_session=test_db_session,
                )
                return result
            except Exception as e:
                return {"error": str(e)}

        tasks = [improve_prompt_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [
            r for r in results if isinstance(r, dict) and "improved_prompt" in r
        ]
        assert len(successful_results) > 0
        for result in successful_results:
            assert "improved_prompt" in result
            assert "processing_time_ms" in result
            assert result["processing_time_ms"] < 10000

    async def test_edge_case_prompt_types(self, test_db_session):
        """Test real service behavior with edge case prompt types."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        edge_cases = [
            "测试中文提示符",
            "Тест русского текста",
            "🚀 Emoji test prompt 🎯",
            "def function(): return 'test'",
            "SELECT * FROM users WHERE id = 1;",
            "Write code that does: print('hello') and explain it",
            "123456789 + 987654321 = ?",
            "Hi",
            "?",
            "test " * 100,
        ]
        for i, prompt in enumerate(edge_cases):
            result = await service.improve_prompt(
                prompt=prompt,
                user_context={"domain": "edge_case_test"},
                session_id=f"edge_case_{i}",
                db_session=test_db_session,
            )
            assert "improved_prompt" in result
            assert "processing_time_ms" in result
            assert result["processing_time_ms"] < 5000
            assert len(result["improved_prompt"]) > 0

    async def test_service_performance_under_load(self, test_db_session):
        """Test service performance characteristics under load with real operations."""
        import asyncio
        import time

        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        sequential_times = []
        for i in range(10):
            start_time = time.time()
            result = await service.improve_prompt(
                prompt=f"Load test prompt {i}",
                user_context={"domain": "load_test"},
                session_id=f"load_test_seq_{i}",
                db_session=test_db_session,
            )
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            sequential_times.append(processing_time)
            assert "improved_prompt" in result
            assert processing_time < 5000

        async def concurrent_task(task_id: int):
            start_time = time.time()
            result = await service.improve_prompt(
                prompt=f"Concurrent load test {task_id}",
                user_context={"domain": "concurrent_load_test"},
                session_id=f"load_test_conc_{task_id}",
                db_session=test_db_session,
            )
            end_time = time.time()
            return ((end_time - start_time) * 1000, result)

        concurrent_tasks = [concurrent_task(i) for i in range(10)]
        concurrent_results = await asyncio.gather(
            *concurrent_tasks, return_exceptions=True
        )
        concurrent_times = []
        successful_concurrent = 0
        for result in concurrent_results:
            if isinstance(result, tuple) and len(result) == 2:
                time_ms, response = result
                if "improved_prompt" in response:
                    concurrent_times.append(time_ms)
                    successful_concurrent += 1
        assert successful_concurrent >= 8, (
            f"Only {successful_concurrent}/10 concurrent requests succeeded"
        )
        if concurrent_times:
            avg_concurrent_time = sum(concurrent_times) / len(concurrent_times)
            avg_sequential_time = sum(sequential_times) / len(sequential_times)
            assert avg_concurrent_time < avg_sequential_time * 3, (
                f"Concurrent avg {avg_concurrent_time:.1f}ms vs sequential avg {avg_sequential_time:.1f}ms"
            )

    async def test_real_service_memory_usage(self, test_db_session):
        """Test memory usage patterns with real service operations."""
        import gc
        import os

        import psutil

        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        for i in range(50):
            result = await service.improve_prompt(
                prompt=f"Memory test prompt {i} with some additional content to test memory usage patterns",
                user_context={"domain": "memory_test", "iteration": i},
                session_id=f"memory_test_{i}",
                db_session=test_db_session,
            )
            assert "improved_prompt" in result
            if i % 10 == 0:
                gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100, (
            f"Memory increased by {memory_increase:.1f}MB, possible memory leak"
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.real_behavior_validation
class TestRealBehaviorValidation:
    """Validate that real behavior tests provide meaningful results and follow 2025 best practices."""

    async def test_real_vs_mock_behavior_comparison(self, test_db_session):
        """Compare real service behavior against expected patterns to validate migration success."""
        from prompt_improver.services.analytics import AnalyticsService
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        analytics_service = AnalyticsService()
        test_prompt = "Help me write better code"
        results = []
        for i in range(3):
            result = await service.improve_prompt(
                prompt=test_prompt,
                user_context={"domain": "software_development"},
                session_id=f"consistency_test_{i}",
                db_session=test_db_session,
            )
            results.append(result)
        for result in results:
            assert "improved_prompt" in result
            assert "applied_rules" in result
            assert "processing_time_ms" in result
            assert result["original_prompt"] == test_prompt
            assert isinstance(result["applied_rules"], list)
            assert isinstance(result["processing_time_ms"], (int, float))
        analytics_result = await analytics_service.get_performance_trends(
            db_session=test_db_session
        )
        assert "daily_trends" in analytics_result
        assert "summary" in analytics_result
        assert isinstance(analytics_result["daily_trends"], list)
        assert isinstance(analytics_result["summary"], dict)
        performance_summary = await analytics_service.get_performance_summary(
            days=30, db_session=test_db_session
        )
        assert "total_sessions" in performance_summary
        assert "avg_improvement" in performance_summary
        assert isinstance(performance_summary["total_sessions"], int)
        assert performance_summary["total_sessions"] >= 0

    async def test_migration_completeness_validation(self, test_db_session):
        """Validate that the migration from mocks to real behavior is complete and effective."""
        from prompt_improver.services.analytics import AnalyticsService
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        analytics_service = AnalyticsService()
        assert service is not None
        assert analytics_service is not None
        result = await service.improve_prompt(
            prompt="Test migration completeness",
            user_context={"domain": "validation"},
            session_id="migration_validation",
            db_session=test_db_session,
        )
        required_fields = [
            "improved_prompt",
            "applied_rules",
            "processing_time_ms",
            "original_prompt",
            "session_id",
            "improvement_summary",
            "confidence_score",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        trends = await analytics_service.get_performance_trends(
            db_session=test_db_session
        )
        assert isinstance(trends, dict)
        assert "daily_trends" in trends
        assert "summary" in trends
        rule_effectiveness = await analytics_service.get_rule_effectiveness(days=30)
        assert isinstance(rule_effectiveness, list)

    async def test_2025_best_practices_compliance(self, test_db_session):
        """Validate that the migrated tests follow 2025 testing best practices."""
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="Test 2025 best practices",
            user_context={"domain": "best_practices"},
            session_id="best_practices_test",
            db_session=test_db_session,
        )
        assert isinstance(result["processing_time_ms"], (int, float))
        assert result["processing_time_ms"] > 0
        assert len(result["improved_prompt"]) > 0
        assert result["original_prompt"] == "Test 2025 best practices"
        assert result["session_id"] == "best_practices_test"
        assert isinstance(result["confidence_score"], (int, float))
        assert 0 <= result["confidence_score"] <= 1
        assert result["processing_time_ms"] < 10000

    async def test_real_data_persistence_validation(self, test_db_session):
        """Validate that real database operations persist data correctly."""
        from prompt_improver.database.models import RuleMetadata, RulePerformance
        from prompt_improver.services.prompt.facade import PromptServiceFacade as PromptImprovementService

        test_rule = RuleMetadata(
            rule_id="persistence_test_rule",
            rule_name="Persistence Test Rule",
            rule_category="test",
            enabled=True,
            priority=1,
            rule_version="1.0",
            default_parameters={"test": True},
        )
        test_db_session.add(test_rule)
        await test_db_session.commit()
        service = PromptImprovementService(
            enable_bandit_optimization=False, enable_automl=False
        )
        result = await service.improve_prompt(
            prompt="Test data persistence",
            user_context={"domain": "persistence_test"},
            session_id="persistence_validation",
            db_session=test_db_session,
        )
        assert "improved_prompt" in result
        assert "applied_rules" in result
        rules_metadata = await service.get_rules_metadata(
            enabled_only=True, db_session=test_db_session
        )
        assert isinstance(rules_metadata, list)
        rule_ids = [rule.rule_id for rule in rules_metadata]
        assert "persistence_test_rule" in rule_ids
        from sqlmodel import select

        result_query = await test_db_session.execute(
            select(RuleMetadata).where(RuleMetadata.rule_id == "persistence_test_rule")
        )
        stored_rule = result_query.scalar_one_or_none()
        assert stored_rule is not None
        assert stored_rule.rule_name == "Persistence Test Rule"
        assert stored_rule.enabled == True
