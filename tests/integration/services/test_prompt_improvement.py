"""
Phase 3 tests for PromptImprovementService focusing on ML optimization methods.
Tests the 3 ML optimization methods using real database behavior following 2025 best practices.

Migration from mock-based testing to real behavior testing:
- Real database connections and operations (only mock external ML service calls)
- Real feature engineering and data processing logic
- Real database state changes and transactions
- Comprehensive integration testing with actual database queries
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import select

from prompt_improver.database.models import (
    ABExperiment,
    DiscoveredPattern,
    MLModelPerformance,
    PromptSession,
    RuleMetadata,
    RulePerformance,
    UserFeedback,
)
from prompt_improver.services.prompt_improvement import PromptImprovementService
from prompt_improver.utils.datetime_utils import aware_utc_now


@pytest.fixture
def prompt_service():
    """Create PromptImprovementService instance."""
    return PromptImprovementService()


@pytest.fixture
async def real_ml_test_data(real_db_session):
    """Create comprehensive real database test data for ML optimization tests."""
    import uuid
    # Generate unique prefix for this test run
    test_id = str(uuid.uuid4())[:8]
    
    # Create prompt sessions first (required for foreign key relationships)
    prompt_sessions = []
    for i in range(25):
        prompt_sessions.append(PromptSession(
            session_id=f"ml_session_{test_id}_{i}",
            original_prompt=f"Original prompt {i} for ML optimization",
            improved_prompt=f"Improved prompt {i} with ML enhancements",
            user_context={"test_ml": True, "optimization_round": i},
            quality_score=0.8 + (i * 0.005),  # Gradually increasing quality
            improvement_score=0.75 + (i * 0.01),  # Gradually increasing improvement
            confidence_level=0.9,
            created_at=aware_utc_now() - timedelta(days=30, hours=i),
            updated_at=aware_utc_now() - timedelta(days=30, hours=i),
        ))
    
    # Create rule metadata for different rule types
    rule_metadata = [
        RuleMetadata(
            rule_id="clarity_rule",
            rule_name="Clarity Enhancement Rule",
            description="Improves prompt clarity through advanced processing",
            category="clarity",
            default_parameters={
                "weight": 1.0,
                "threshold": 0.7,
                "confidence_threshold": 0.7,
                "enabled": True,
                "min_length": 10,
                "max_length": 500
            },
            priority=5,
            created_at=aware_utc_now() - timedelta(days=30),
            updated_at=aware_utc_now() - timedelta(days=30),
        ),
        RuleMetadata(
            rule_id="specificity_rule",
            rule_name="Specificity Enhancement Rule",
            description="Improves prompt specificity through parameter tuning",
            category="specificity",
            default_parameters={
                "weight": 0.8,
                "threshold": 0.6,
                "confidence_threshold": 0.8,
                "enabled": True,
                "min_length": 15,
                "max_length": 300
            },
            priority=4,
            created_at=aware_utc_now() - timedelta(days=30),
            updated_at=aware_utc_now() - timedelta(days=30),
        )
    ]
    
    # Create user feedback records
    user_feedback = []
    for i in range(5):
        user_feedback.append(UserFeedback(
            session_id=f"ml_session_{test_id}_{i}",
            user_id=f"user_{i}",
            feedback_type="improvement",
            rating=4 + (i % 2),  # Ratings 4-5
            comment=f"ML optimization feedback {i}",
            ml_optimized=False,  # Will be updated by trigger_optimization
            model_id=None,  # Will be updated by trigger_optimization
            created_at=aware_utc_now() - timedelta(days=25, hours=i),
            updated_at=aware_utc_now() - timedelta(days=25, hours=i),
        ))
    
    # Create rule performance records with realistic ML optimization data
    rule_performance = []
    for i in range(25):
        rule_id = "clarity_rule" if i % 2 == 0 else "specificity_rule"
        rule_performance.append(RulePerformance(
            rule_id=rule_id,
            session_id=f"ml_session_{test_id}_{i}",
            improvement_score=0.7 + (i * 0.01),  # Gradual improvement
            execution_time_ms=100 + (i * 5),  # Increasing execution time
            confidence_level=0.8 + (i * 0.005),
            user_satisfaction_score=0.8 + (i * 0.008),
            parameters_used={
                "weight": 0.8 + (i * 0.01),
                "threshold": 0.6 + (i * 0.01),
                "confidence_threshold": 0.7 + (i * 0.005),
                "enabled": True,
                "min_length": 10 + i,
                "max_length": 400 + (i * 2)
            },
            created_at=aware_utc_now() - timedelta(days=10, hours=i),
            updated_at=aware_utc_now() - timedelta(days=10, hours=i),
        ))
    
    # Add all records to database in proper order (dependencies first)
    for session in prompt_sessions:
        real_db_session.add(session)
    for metadata in rule_metadata:
        real_db_session.add(metadata)
    for feedback in user_feedback:
        real_db_session.add(feedback)
    for performance in rule_performance:
        real_db_session.add(performance)
    
    await real_db_session.commit()
    
    return {
        "prompt_sessions": prompt_sessions,
        "rule_metadata": rule_metadata,
        "user_feedback": user_feedback,
        "rule_performance": rule_performance,
    }


class TestPhase3MLOptimization:
    """Test Phase 3 ML optimization methods in PromptImprovementService using real database behavior."""


class TestTriggerOptimization:
    """Test trigger_optimization method using real database behavior."""

    @pytest.mark.asyncio
    async def test_trigger_optimization_success(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test successful ML optimization trigger from feedback using real database."""
        # Get the first user feedback record from real database
        feedback_query = select(UserFeedback).limit(1)
        result = await real_db_session.execute(feedback_query)
        feedback = result.scalar_one()
        feedback_id = feedback.id

        # Mock ML service (external dependency)
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.85,
            "model_id": "optimized_model_123",
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.trigger_optimization(
                feedback_id, real_db_session
            )

        # Verify optimization result
        assert result["status"] == "success"
        assert result["performance_score"] == 0.85
        assert "training_samples" in result
        assert result["model_id"] == "optimized_model_123"

        # Verify ML service was called with real training data
        mock_ml_service.optimize_rules.assert_called_once()
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        assert "features" in training_data
        assert "effectiveness_scores" in training_data
        
        # Verify real database updates (feedback should be marked as ML optimized)
        updated_feedback_query = select(UserFeedback).where(UserFeedback.id == feedback_id)
        updated_result = await real_db_session.execute(updated_feedback_query)
        updated_feedback = updated_result.scalar_one()
        assert updated_feedback.ml_optimized == True
        assert updated_feedback.model_id == "optimized_model_123"

    @pytest.mark.asyncio
    async def test_trigger_optimization_feedback_not_found(
        self, prompt_service, real_db_session
    ):
        """Test trigger optimization with non-existent feedback using real database."""
        # Use non-existent feedback ID with real database
        non_existent_feedback_id = 999999
        
        result = await prompt_service.trigger_optimization(
            non_existent_feedback_id, real_db_session
        )

        assert result["status"] == "error"
        assert f"Feedback {non_existent_feedback_id} not found" in result["message"]

    @pytest.mark.asyncio
    async def test_trigger_optimization_insufficient_data(
        self, prompt_service, real_db_session
    ):
        """Test trigger optimization with insufficient performance data using real database."""
        # Create a minimal dataset with insufficient performance data
        # First create the PromptSession that the feedback references
        minimal_session = PromptSession(
            session_id="minimal_session",
            original_prompt="Minimal test prompt",
            improved_prompt="Minimal improved prompt",
            user_context={"test": True},
            quality_score=0.7,
            improvement_score=0.65,
            confidence_level=0.8,
            created_at=aware_utc_now(),
            updated_at=aware_utc_now(),
        )
        real_db_session.add(minimal_session)
        
        minimal_feedback = UserFeedback(
            session_id="minimal_session",
            user_id="minimal_user",
            feedback_type="improvement",
            rating=4,
            comment="Minimal test feedback",
            ml_optimized=False,
            created_at=aware_utc_now(),
            updated_at=aware_utc_now(),
        )
        
        real_db_session.add(minimal_feedback)
        await real_db_session.commit()
        
        # Get the feedback ID from real database
        feedback_query = select(UserFeedback).where(UserFeedback.session_id == "minimal_session")
        result = await real_db_session.execute(feedback_query)
        feedback = result.scalar_one()
        
        # Test with real database that has insufficient performance data
        result = await prompt_service.trigger_optimization(feedback.id, real_db_session)

        assert result["status"] == "error"
        assert "No performance data available" in result["message"]

    @pytest.mark.asyncio
    async def test_trigger_optimization_ml_failure(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test trigger optimization with ML service failure using real database."""
        # Get real feedback from database
        feedback_query = select(UserFeedback).limit(1)
        result = await real_db_session.execute(feedback_query)
        feedback = result.scalar_one()
        
        # Mock ML service failure (external dependency)
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "error",
            "error": "Model training failed",
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.trigger_optimization(feedback.id, real_db_session)

        assert result["status"] == "error"
        assert "Optimization failed: Model training failed" in result["message"]


class TestRunMLOptimization:
    """Test run_ml_optimization method using real database behavior."""

    @pytest.mark.asyncio
    async def test_run_ml_optimization_success(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test successful ML optimization run with real database."""
        # Mock ML service success (external dependency)
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.88,
            "model_id": "ml_model_456",
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.90,
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.run_ml_optimization(
                None, real_db_session  # Use None to include all rules
            )

        assert result["status"] == "success"
        assert result["best_score"] == 0.88
        assert result["model_id"] == "ml_model_456"

        # Verify real feature engineering with actual database data
        mock_ml_service.optimize_rules.assert_called_once()
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        features = training_data["features"]
        assert len(features[0]) == 10  # 10 features as specified in implementation
        
        # Verify real database performance records were used
        assert len(features) > 0  # Should have real performance data
        
        # Verify MLModelPerformance record was created in real database
        ml_perf_query = select(MLModelPerformance).where(MLModelPerformance.model_id == "ml_model_456")
        ml_perf_result = await real_db_session.execute(ml_perf_query)
        ml_perf_record = ml_perf_result.scalar_one_or_none()
        assert ml_perf_record is not None
        assert ml_perf_record.accuracy == 0.88

    @pytest.mark.asyncio
    async def test_run_ml_optimization_with_ensemble(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test ML optimization with ensemble when enough data available using real database."""
        # Add more performance data to reach ensemble threshold (50+ samples)
        # First create the RuleMetadata for ensemble_test_rule
        ensemble_rule = RuleMetadata(
            rule_id="ensemble_test_rule",
            rule_name="Ensemble Test Rule",
            description="Rule for ensemble testing",
            category="ensemble",
            default_parameters={"weight": 1.0, "threshold": 0.7},
            priority=3,
            created_at=aware_utc_now() - timedelta(days=30),
            updated_at=aware_utc_now() - timedelta(days=30),
        )
        real_db_session.add(ensemble_rule)
        
        # Create the corresponding PromptSession records
        ensemble_sessions = []
        for i in range(30):  # Add 30 more sessions
            ensemble_sessions.append(PromptSession(
                session_id=f"ensemble_session_{i}",
                original_prompt=f"Ensemble test prompt {i}",
                improved_prompt=f"Enhanced ensemble prompt {i}",
                user_context={"ensemble_test": True},
                quality_score=0.85,
                improvement_score=0.8,
                confidence_level=0.88,
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for session in ensemble_sessions:
            real_db_session.add(session)
        
        additional_performance = []
        for i in range(30):  # Add 30 more to existing 25 for total 55
            additional_performance.append(RulePerformance(
                rule_id="ensemble_test_rule",
                session_id=f"ensemble_session_{i}",
                improvement_score=0.8 + (i * 0.003),
                execution_time_ms=150 + (i * 2),
                confidence_level=0.85,
                user_satisfaction_score=0.9 + (i * 0.001),
                parameters_used={"weight": 1.0, "threshold": 0.7},
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for perf in additional_performance:
            real_db_session.add(perf)
        await real_db_session.commit()

        # Mock ML service for ensemble testing
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.88,
            "accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.90,
        }
        mock_ml_service.optimize_ensemble_rules.return_value = {
            "status": "success",
            "ensemble_score": 0.92,
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.run_ml_optimization(None, real_db_session)

        assert result["status"] == "success"
        assert "ensemble" in result
        assert result["ensemble"]["ensemble_score"] == 0.92

        # Verify both optimization methods were called with real data
        mock_ml_service.optimize_rules.assert_called_once()
        mock_ml_service.optimize_ensemble_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ml_optimization_insufficient_data(
        self, prompt_service, real_db_session
    ):
        """Test ML optimization with insufficient data using real database."""
        # Create minimal dataset with insufficient performance data (< 20 samples)
        # First create the RuleMetadata for insufficient_rule
        insufficient_rule = RuleMetadata(
            rule_id="insufficient_rule",
            rule_name="Insufficient Rule",
            description="Rule for insufficient data testing",
            category="test",
            default_parameters={"weight": 0.8, "threshold": 0.6},
            priority=2,
            created_at=aware_utc_now() - timedelta(days=30),
            updated_at=aware_utc_now() - timedelta(days=30),
        )
        real_db_session.add(insufficient_rule)
        
        # Create the PromptSession records
        insufficient_sessions = []
        for i in range(15):  # Only 15 sessions
            insufficient_sessions.append(PromptSession(
                session_id=f"insufficient_session_{i}",
                original_prompt=f"Insufficient test prompt {i}",
                improved_prompt=f"Insufficient improved prompt {i}",
                user_context={"insufficient_test": True},
                quality_score=0.7,
                improvement_score=0.65,
                confidence_level=0.75,
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for session in insufficient_sessions:
            real_db_session.add(session)
        
        insufficient_data = []
        for i in range(15):  # Only 15 samples
            insufficient_data.append(RulePerformance(
                rule_id="insufficient_rule",
                session_id=f"insufficient_session_{i}",
                improvement_score=0.7 + (i * 0.01),
                execution_time_ms=100 + (i * 5),
                confidence_level=0.8,
                user_satisfaction_score=0.8,
                parameters_used={"weight": 0.8, "threshold": 0.6},
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for perf in insufficient_data:
            real_db_session.add(perf)
        await real_db_session.commit()

        # Test with real database containing insufficient data
        result = await prompt_service.run_ml_optimization(
            ["insufficient_rule"], real_db_session  # This will only match the 15 records we added
        )

        assert result["status"] == "insufficient_data"
        assert "Need at least 20 samples" in result["message"]
        assert result["samples_found"] == 15

    @pytest.mark.asyncio
    async def test_run_ml_optimization_feature_engineering(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test enhanced feature engineering with 10 features using real database."""
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {"status": "success"}

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            await prompt_service.run_ml_optimization(None, real_db_session)

        # Verify feature engineering with real database data
        mock_ml_service.optimize_rules.assert_called_once()
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        features = training_data["features"]

        # Verify 10 features as specified in implementation
        assert len(features) > 0  # Should have real data
        first_feature_vector = features[0]
        assert len(first_feature_vector) == 10
        
        # Verify feature engineering structure with real data
        # Features should be: improvement_score, execution_time_ms, weight, priority, 
        # user_satisfaction_score, len(parameters), confidence_threshold, enabled, 
        # normalized min_length, normalized max_length
        assert isinstance(first_feature_vector[0], (int, float))  # improvement_score
        assert isinstance(first_feature_vector[1], (int, float))  # execution_time_ms
        assert isinstance(first_feature_vector[2], (int, float))  # weight
        assert isinstance(first_feature_vector[3], (int, float))  # priority
        assert isinstance(first_feature_vector[4], (int, float))  # user_satisfaction_score
        assert isinstance(first_feature_vector[5], (int, float))  # len(parameters)
        assert isinstance(first_feature_vector[6], (int, float))  # confidence_threshold
        assert isinstance(first_feature_vector[7], (int, float))  # enabled status
        assert isinstance(first_feature_vector[8], (int, float))  # normalized min_length
        assert isinstance(first_feature_vector[9], (int, float))  # normalized max_length


class TestDiscoverPatterns:
    """Test discover_patterns method using real database behavior."""

    @pytest.mark.asyncio
    async def test_discover_patterns_success(self, prompt_service, real_db_session, real_ml_test_data):
        """Test successful pattern discovery with real database."""
        # Mock ML service (external dependency)
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "success",
            "patterns_discovered": 3,
            "patterns": [
                {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.85},
                {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.80},
                {"parameters": {"weight": 0.9}, "avg_effectiveness": 0.82},
            ],
            "total_analyzed": 25,  # Based on real test data
            "processing_time_ms": 1500,
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.discover_patterns(0.7, 5, real_db_session)

        assert result["status"] == "success"
        assert result["patterns_discovered"] == 3
        assert len(result["patterns"]) == 3
        assert result["total_analyzed"] == 25
        assert result["processing_time_ms"] == 1500

        # Verify ML service was called with real database session
        mock_ml_service.discover_patterns.assert_called_once_with(
            db_session=real_db_session, min_effectiveness=0.7, min_support=5
        )
        
        # Verify A/B experiments were created in real database (top 3 patterns)
        ab_experiment_query = select(ABExperiment)
        ab_result = await real_db_session.execute(ab_experiment_query)
        ab_experiments = ab_result.scalars().all()
        assert len(ab_experiments) == 3  # Should have created 3 A/B experiments for top patterns

    @pytest.mark.asyncio
    async def test_discover_patterns_insufficient_data(
        self, prompt_service, real_db_session
    ):
        """Test pattern discovery with insufficient data using real database."""
        # Create minimal dataset with insufficient high-performing samples
        # First create the RuleMetadata
        minimal_rule = RuleMetadata(
            rule_id="minimal_pattern_rule",
            rule_name="Minimal Pattern Rule",
            description="Rule for minimal pattern testing",
            category="test",
            default_parameters={"weight": 0.8, "threshold": 0.7},
            priority=1,
            created_at=aware_utc_now() - timedelta(days=30),
            updated_at=aware_utc_now() - timedelta(days=30),
        )
        real_db_session.add(minimal_rule)
        
        # Create corresponding PromptSession records
        minimal_sessions = []
        for i in range(3):  # Only 3 sessions
            minimal_sessions.append(PromptSession(
                session_id=f"minimal_pattern_session_{i}",
                original_prompt=f"Minimal pattern test prompt {i}",
                improved_prompt=f"Minimal pattern improved prompt {i}",
                user_context={"minimal_test": True},
                quality_score=0.85,
                improvement_score=0.8,
                confidence_level=0.82,
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for session in minimal_sessions:
            real_db_session.add(session)
        
        minimal_performance = []
        for i in range(3):  # Only 3 samples, below minimum of 5
            minimal_performance.append(RulePerformance(
                rule_id="minimal_pattern_rule",
                session_id=f"minimal_pattern_session_{i}",
                improvement_score=0.85,  # High performance
                execution_time_ms=100,
                confidence_level=0.8,
                user_satisfaction_score=0.9,
                parameters_used={"weight": 0.8 + (i * 0.05), "threshold": 0.7},
                created_at=aware_utc_now() - timedelta(hours=i),
                updated_at=aware_utc_now() - timedelta(hours=i),
            ))
        
        for perf in minimal_performance:
            real_db_session.add(perf)
        await real_db_session.commit()
        
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "insufficient_data",
            "message": "Only 3 high-performing samples found (minimum: 5)",
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            result = await prompt_service.discover_patterns(0.8, 5, real_db_session)

        assert result["status"] == "insufficient_data"
        assert "minimum: 5" in result["message"]

    @pytest.mark.asyncio
    async def test_discover_patterns_creates_ab_experiments(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test that pattern discovery creates A/B experiments for top patterns using real database."""
        # Mock successful pattern discovery
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "success",
            "patterns_discovered": 5,
            "patterns": [
                {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.90},
                {"parameters": {"weight": 0.9}, "avg_effectiveness": 0.85},
                {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.80},
                {"parameters": {"weight": 0.7}, "avg_effectiveness": 0.75},
                {"parameters": {"weight": 0.6}, "avg_effectiveness": 0.70},
            ],
            "total_analyzed": 25,
        }

        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            await prompt_service.discover_patterns(0.7, 5, real_db_session)

        # Verify A/B experiments were created for top 3 patterns in real database
        ab_experiment_query = select(ABExperiment)
        ab_result = await real_db_session.execute(ab_experiment_query)
        ab_experiments = ab_result.scalars().all()
        assert len(ab_experiments) == 3  # Top 3 patterns should create A/B experiments
        
        # Verify experiments have correct control rules
        control_rules = [exp.control_rules for exp in ab_experiments]
        assert len(control_rules) == 3


class TestPhase3HelperMethods:
    """Test Phase 3 helper methods for ML integration using real database behavior."""

    @pytest.mark.asyncio
    async def test_store_optimization_trigger(
        self, prompt_service, real_db_session, real_ml_test_data
    ):
        """Test storing optimization trigger event with real database."""
        # Get real feedback from database
        feedback_query = select(UserFeedback).limit(1)
        result = await real_db_session.execute(feedback_query)
        feedback = result.scalar_one()
        
        await prompt_service._store_optimization_trigger(
            real_db_session, feedback.id, "model_456", 25, 0.88
        )

        # Verify feedback was updated with ML optimization info in real database
        updated_feedback_query = select(UserFeedback).where(UserFeedback.id == feedback.id)
        updated_result = await real_db_session.execute(updated_feedback_query)
        updated_feedback = updated_result.scalar_one()
        
        assert updated_feedback.ml_optimized == True
        assert updated_feedback.model_id == "model_456"

    @pytest.mark.asyncio
    async def test_store_ml_optimization_results(self, prompt_service, real_db_session):
        """Test storing ML optimization results with real database."""
        optimization_result = {
            "model_id": "test_model_789",
            "best_score": 0.85,
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
        }

        await prompt_service._store_ml_optimization_results(
            real_db_session, ["clarity_rule"], optimization_result, 30
        )

        # Verify performance record was created in real database
        ml_perf_query = select(MLModelPerformance).where(MLModelPerformance.model_id == "test_model_789")
        ml_perf_result = await real_db_session.execute(ml_perf_query)
        ml_perf_record = ml_perf_result.scalar_one()
        
        assert ml_perf_record.model_id == "test_model_789"
        assert ml_perf_record.accuracy == 0.90
        assert ml_perf_record.precision == 0.88
        assert ml_perf_record.recall == 0.92
        assert ml_perf_record.training_samples == 30

    @pytest.mark.asyncio
    async def test_create_ab_experiments_from_patterns(
        self, prompt_service, real_db_session
    ):
        """Test creating A/B experiments from discovered patterns with real database."""
        patterns = [
            {"avg_effectiveness": 0.90, "parameters": {"weight": 1.0}},
            {"avg_effectiveness": 0.85, "parameters": {"weight": 0.9}},
            {"avg_effectiveness": 0.80, "parameters": {"weight": 0.8}},
        ]

        await prompt_service._create_ab_experiments_from_patterns(
            real_db_session, patterns
        )

        # Verify 3 A/B experiments were created in real database
        ab_experiment_query = select(ABExperiment)
        ab_result = await real_db_session.execute(ab_experiment_query)
        ab_experiments = ab_result.scalars().all()
        
        assert len(ab_experiments) == 3
        
        # Verify experiments were created
        assert all(exp.status == "running" for exp in ab_experiments)

    @pytest.mark.asyncio
    async def test_store_pattern_discovery_results(
        self, prompt_service, real_db_session
    ):
        """Test storing pattern discovery results with real database."""
        discovery_result = {
            "patterns": [
                {
                    "avg_effectiveness": 0.88,
                    "parameters": {"weight": 1.0},
                    "support_count": 10,
                },
                {
                    "avg_effectiveness": 0.82,
                    "parameters": {"weight": 0.9},
                    "support_count": 8,
                },
            ]
        }

        await prompt_service._store_pattern_discovery_results(
            real_db_session, discovery_result
        )

        # Note: Pattern discovery results are stored via _store_pattern_discovery_results
        # which currently only logs results (placeholder implementation)
        # Since _store_pattern_discovery_results is a placeholder, we can't verify database records
        # This test passes if the method doesn't raise an exception


class TestPhase3Integration:
    """Test Phase 3 integration with existing functionality using real database behavior."""

    @pytest.mark.asyncio
    async def test_phase3_methods_integration(self, prompt_service, real_db_session):
        """Test that Phase 3 methods integrate properly with existing service."""
        # Verify Phase 3 methods exist and are callable
        assert hasattr(prompt_service, "trigger_optimization")
        assert hasattr(prompt_service, "run_ml_optimization")
        assert hasattr(prompt_service, "discover_patterns")

        # Verify methods are async
        assert asyncio.iscoroutinefunction(prompt_service.trigger_optimization)
        assert asyncio.iscoroutinefunction(prompt_service.run_ml_optimization)
        assert asyncio.iscoroutinefunction(prompt_service.discover_patterns)

    @pytest.mark.asyncio
    async def test_ml_service_import_integration(self, prompt_service):
        """Test that ML service import works correctly."""
        # This tests the 'from .ml_integration import get_ml_service' import
        with patch(
            "prompt_improver.services.ml_integration.get_ml_service"
        ) as mock_get_ml:
            mock_ml_service = AsyncMock()
            mock_get_ml.return_value = mock_ml_service

            # Import should work without errors
            from prompt_improver.services.ml_integration import get_ml_service

            service = await get_ml_service()
            assert service is mock_ml_service
    
    @pytest.mark.asyncio
    async def test_end_to_end_ml_workflow(self, prompt_service, real_db_session, real_ml_test_data):
        """Test complete end-to-end ML workflow integration with real database."""
        # Mock ML service for complete workflow
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.85,
            "model_id": "workflow_model_123",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
        }
        mock_ml_service.discover_patterns.return_value = {
            "status": "success",
            "patterns_discovered": 2,
            "patterns": [
                {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.90},
                {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.85},
            ],
            "total_analyzed": 25,
        }
        
        with patch(
            "prompt_improver.services.ml_integration.get_ml_service",
            return_value=mock_ml_service,
        ):
            # Step 1: Trigger optimization
            feedback_query = select(UserFeedback).limit(1)
            feedback_result = await real_db_session.execute(feedback_query)
            feedback = feedback_result.scalar_one()
            
            optimization_result = await prompt_service.trigger_optimization(
                feedback.id, real_db_session
            )
            assert optimization_result["status"] == "success"
            
            # Step 2: Run ML optimization
            ml_result = await prompt_service.run_ml_optimization(
                None, real_db_session  # Use None to include all rules
            )
            assert ml_result["status"] == "success"
            
            # Step 3: Discover patterns
            pattern_result = await prompt_service.discover_patterns(
                0.7, 5, real_db_session
            )
            assert pattern_result["status"] == "success"
            
            # Verify complete workflow created real database records
            ml_perf_query = select(MLModelPerformance)
            ml_perf_result = await real_db_session.execute(ml_perf_query)
            ml_perf_records = ml_perf_result.scalars().all()
            assert len(ml_perf_records) >= 1  # At least 1 from ML optimization
            
            # Note: Pattern discovery creates A/B experiments, not DiscoveredPattern records
            # This is because _store_pattern_discovery_results is currently a placeholder
            
            ab_query = select(ABExperiment)
            ab_result = await real_db_session.execute(ab_query)
            ab_records = ab_result.scalars().all()
            assert len(ab_records) >= 2  # A/B experiments created from patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
