"""
Real Behavior Tests for Adaptive Data Generation - 2025 Best Practices
Tests verify actual behavior using real database interactions and components.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from prompt_improver.database import get_sessionmanager
from prompt_improver.database.models import (
    TrainingSession, RulePerformance, UserFeedback, ImprovementSession
)
from prompt_improver.ml.analysis.performance_gap_analyzer import (
    PerformanceGapAnalyzer, PerformanceGap
)
from prompt_improver.ml.analysis.generation_strategy_analyzer import (
    GenerationStrategyAnalyzer, GenerationStrategy
)
from prompt_improver.ml.analysis.difficulty_distribution_analyzer import (
    DifficultyDistributionAnalyzer
)
from prompt_improver.ml.preprocessing.synthetic_data_generator import (
    ProductionSyntheticDataGenerator
)
from prompt_improver.ml.orchestration.coordinators.adaptive_training_coordinator import (
    AdaptiveTrainingCoordinator
)
from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator
)


@pytest.fixture
async def real_database_session():
    """Provide real database session for integration testing."""
    async with get_sessionmanager().get_session() as session:
        yield session


@pytest.fixture
async def sample_performance_data(real_database_session: AsyncSession):
    """Create real performance data in database for testing."""
    
    # Create sample improvement sessions
    sessions = []
    for i in range(10):
        session = ImprovementSession(
            session_id=f"test_session_{i}",
            original_prompt=f"Original prompt {i}",
            final_prompt=f"Improved prompt {i}",
            rules_applied=["clarity", "specificity"],
            improvement_metrics={
                "clarity_score": 0.6 + (i * 0.02),
                "specificity_score": 0.5 + (i * 0.03),
                "overall_score": 0.55 + (i * 0.025)
            }
        )
        real_database_session.add(session)
        sessions.append(session)
    
    # Create sample rule performance data
    rule_performances = []
    for i in range(20):
        rule_perf = RulePerformance(
            rule_id=f"rule_{i % 5}",  # 5 different rules
            effectiveness_score=0.4 + (i * 0.02),
            consistency_score=0.5 + (i * 0.015),
            coverage_score=0.6 + (i * 0.01),
            confidence_score=0.7 + (i * 0.005)
        )
        real_database_session.add(rule_perf)
        rule_performances.append(rule_perf)
    
    # Create sample user feedback
    feedback_items = []
    for i in range(15):
        feedback = UserFeedback(
            session_id=f"test_session_{i % 10}",
            rating=3 + (i % 3),  # Ratings 3-5
            feedback_text=f"Test feedback {i}",
            improvement_areas=["clarity", "specificity"][i % 2:i % 2 + 1]
        )
        real_database_session.add(feedback)
        feedback_items.append(feedback)
    
    await real_database_session.commit()
    
    return {
        "sessions": sessions,
        "rule_performances": rule_performances,
        "feedback": feedback_items
    }


@pytest.fixture
def real_components():
    """Create real component instances for testing."""
    gap_analyzer = PerformanceGapAnalyzer()
    strategy_analyzer = GenerationStrategyAnalyzer()
    difficulty_analyzer = DifficultyDistributionAnalyzer()
    data_generator = ProductionSyntheticDataGenerator(
        target_samples=200,
        enable_gap_targeting=True,
        difficulty_distribution="adaptive"
    )
    
    return {
        "gap_analyzer": gap_analyzer,
        "strategy_analyzer": strategy_analyzer,
        "difficulty_analyzer": difficulty_analyzer,
        "data_generator": data_generator
    }


class TestRealPerformanceGapAnalysis:
    """Test real behavior of performance gap analysis with actual database data."""
    
    async def test_real_gap_analysis_with_database(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test gap analysis using real database data."""
        
        gap_analyzer = real_components["gap_analyzer"]
        
        # Perform real gap analysis
        gap_result = await gap_analyzer.analyze_performance_gaps(
            session=real_database_session,
            rule_ids=None,  # Analyze all rules
            baseline_window=10
        )
        
        # Verify real behavior
        assert gap_result is not None
        assert gap_result.total_gaps_detected >= 0
        assert isinstance(gap_result.critical_gaps, list)
        assert isinstance(gap_result.improvement_opportunities, list)
        assert isinstance(gap_result.correlation_score, float)
        assert isinstance(gap_result.plateau_detected, bool)
        assert isinstance(gap_result.stopping_criteria_met, bool)
        
        # Verify metadata contains real analysis data
        assert "analysis_duration_ms" in gap_result.metadata
        assert "rules_analyzed" in gap_result.metadata
        assert gap_result.metadata["analysis_duration_ms"] > 0
        
        print(f"Real gap analysis completed: {gap_result.total_gaps_detected} gaps detected")
    
    async def test_enhanced_gap_analysis_for_generation(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test enhanced gap analysis specifically for data generation."""
        
        gap_analyzer = real_components["gap_analyzer"]
        
        # Perform enhanced gap analysis
        enhanced_result = await gap_analyzer.analyze_gaps_for_targeted_generation(
            session=real_database_session,
            rule_ids=None,
            focus_areas=["clarity", "specificity"]
        )
        
        # Verify enhanced analysis structure
        assert "standard_analysis" in enhanced_result
        assert "enhanced_gaps" in enhanced_result
        assert "hardness_analysis" in enhanced_result
        assert "focus_priorities" in enhanced_result
        assert "strategy_recommendations" in enhanced_result
        assert "generation_config" in enhanced_result
        
        # Verify generation config is ready for use
        gen_config = enhanced_result["generation_config"]
        assert "recommended_strategy" in gen_config
        assert "focus_areas" in gen_config
        assert "difficulty_weights" in gen_config
        assert "target_samples" in gen_config
        
        print(f"Enhanced gap analysis: {len(enhanced_result['enhanced_gaps'])} enhanced gaps")


class TestRealStrategyDetermination:
    """Test real behavior of generation strategy determination."""
    
    async def test_real_strategy_analysis(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test strategy analysis with real gap data."""
        
        gap_analyzer = real_components["gap_analyzer"]
        strategy_analyzer = real_components["strategy_analyzer"]
        
        # Get real gap analysis
        gap_result = await gap_analyzer.analyze_performance_gaps(
            session=real_database_session
        )
        
        # Analyze strategy with real data
        strategy_recommendation = await strategy_analyzer.analyze_optimal_strategy(
            gap_analysis=gap_result,
            hardness_analysis={"distribution": {"hard_examples_ratio": 0.3, "std": 0.2}},
            focus_areas=["clarity", "specificity"]
        )
        
        # Verify real strategy recommendation
        assert isinstance(strategy_recommendation.primary_strategy, GenerationStrategy)
        assert isinstance(strategy_recommendation.secondary_strategy, GenerationStrategy)
        assert 0.0 <= strategy_recommendation.confidence <= 1.0
        assert strategy_recommendation.estimated_samples > 0
        assert isinstance(strategy_recommendation.reasoning, str)
        assert len(strategy_recommendation.reasoning) > 0
        
        # Verify difficulty distribution is valid
        difficulty_dist = strategy_recommendation.difficulty_distribution
        assert "easy" in difficulty_dist
        assert "medium" in difficulty_dist
        assert "hard" in difficulty_dist
        assert abs(sum(difficulty_dist.values()) - 1.0) < 0.01  # Should sum to 1.0
        
        print(f"Strategy recommendation: {strategy_recommendation.primary_strategy.value} "
              f"(confidence: {strategy_recommendation.confidence:.2f})")


class TestRealDataGeneration:
    """Test real behavior of adaptive data generation."""
    
    async def test_real_targeted_data_generation(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test targeted data generation with real performance gaps."""
        
        gap_analyzer = real_components["gap_analyzer"]
        data_generator = real_components["data_generator"]
        
        # Get real performance gaps
        gap_result = await gap_analyzer.analyze_performance_gaps(
            session=real_database_session
        )
        
        # Extract performance gaps for targeting
        all_gaps = gap_result.critical_gaps + gap_result.improvement_opportunities
        performance_gaps = {
            f"{gap.gap_type}_{gap.rule_id}": gap.gap_magnitude
            for gap in all_gaps
        }
        
        # Generate targeted data with real gaps
        generated_data = await data_generator.generate_targeted_data(
            performance_gaps=performance_gaps,
            strategy="gap_based",
            batch_size=100,
            focus_areas=["clarity", "specificity"]
        )
        
        # Verify real data generation
        assert "features" in generated_data
        assert "effectiveness" in generated_data
        assert "prompts" in generated_data
        assert "metadata" in generated_data
        
        features = generated_data["features"]
        effectiveness = generated_data["effectiveness"]
        prompts = generated_data["prompts"]
        
        assert len(features) > 0
        assert len(effectiveness) == len(features)
        assert len(prompts) == len(features)
        
        # Verify data quality
        assert all(isinstance(f, list) for f in features)
        assert all(0.0 <= e <= 1.0 for e in effectiveness)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in prompts)
        
        # Verify targeting metadata
        targeting_info = generated_data["metadata"].get("targeting_info", {})
        assert "performance_gaps" in targeting_info
        assert "strategy_used" in targeting_info
        assert "focus_areas" in targeting_info
        
        print(f"Generated {len(features)} targeted samples with strategy: "
              f"{targeting_info.get('strategy_used', 'unknown')}")


class TestRealAdaptiveTrainingIntegration:
    """Test real behavior of adaptive training integration."""
    
    async def test_real_adaptive_training_session_creation(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test creation of real adaptive training session."""
        
        # Create real orchestrator and coordinator
        orchestrator = MLPipelineOrchestrator()
        data_generator = real_components["data_generator"]
        
        coordinator = AdaptiveTrainingCoordinator(
            orchestrator=orchestrator,
            data_generator=data_generator
        )
        
        # Start real adaptive training session
        session_config = {
            "max_iterations": 3,  # Limited for testing
            "improvement_threshold": 0.01,
            "timeout_hours": 1
        }
        
        session_id = await coordinator.start_adaptive_training_session(
            session_config=session_config,
            focus_areas=["clarity", "specificity"]
        )
        
        # Verify session was created
        assert session_id is not None
        assert session_id.startswith("adaptive_training_")
        
        # Wait a moment for session to initialize
        await asyncio.sleep(0.5)
        
        # Verify session status
        status = await coordinator.get_session_status(session_id)
        assert status["session_id"] == session_id
        assert status["status"] in ["active", "running", "initializing"]
        
        # Verify database record was created
        result = await real_database_session.execute(
            select(TrainingSession).where(TrainingSession.session_id == session_id)
        )
        training_session = result.scalar_one_or_none()
        
        assert training_session is not None
        assert training_session.continuous_mode is True
        assert training_session.status in ["running", "initializing"]
        assert training_session.checkpoint_data is not None
        assert training_session.checkpoint_data.get("adaptive_training") is True
        
        print(f"Created adaptive training session: {session_id}")
        
        # Clean up - stop the session
        # Note: In a real implementation, we'd have a stop method
        if session_id in coordinator.active_sessions:
            del coordinator.active_sessions[session_id]


class TestRealEndToEndWorkflow:
    """Test complete end-to-end adaptive generation workflow."""
    
    async def test_complete_adaptive_workflow(
        self,
        real_database_session: AsyncSession,
        sample_performance_data: Dict[str, Any],
        real_components: Dict[str, Any]
    ):
        """Test complete workflow from gap analysis to data generation."""
        
        gap_analyzer = real_components["gap_analyzer"]
        strategy_analyzer = real_components["strategy_analyzer"]
        difficulty_analyzer = real_components["difficulty_analyzer"]
        data_generator = real_components["data_generator"]
        
        # Step 1: Real gap analysis
        print("Step 1: Analyzing performance gaps...")
        enhanced_gap_result = await gap_analyzer.analyze_gaps_for_targeted_generation(
            session=real_database_session,
            focus_areas=["clarity", "specificity"]
        )
        
        gap_result = enhanced_gap_result["standard_analysis"]
        assert gap_result.total_gaps_detected >= 0
        
        # Step 2: Real strategy determination
        print("Step 2: Determining generation strategy...")
        strategy_recommendation = await strategy_analyzer.analyze_optimal_strategy(
            gap_analysis=gap_result,
            hardness_analysis=enhanced_gap_result["hardness_analysis"],
            focus_areas=["clarity", "specificity"]
        )
        
        assert strategy_recommendation.confidence > 0.0
        
        # Step 3: Real difficulty analysis
        print("Step 3: Analyzing difficulty distribution...")
        all_gaps = gap_result.critical_gaps + gap_result.improvement_opportunities
        difficulty_profile = await difficulty_analyzer.analyze_optimal_difficulty_distribution(
            performance_gaps=all_gaps,
            hardness_analysis=enhanced_gap_result["hardness_analysis"],
            focus_areas=["clarity", "specificity"]
        )
        
        assert len(difficulty_profile.distribution_weights) > 0
        
        # Step 4: Real targeted data generation
        print("Step 4: Generating targeted synthetic data...")
        performance_gaps = {
            f"{gap.gap_type}_{gap.rule_id}": gap.gap_magnitude
            for gap in all_gaps
        }
        
        generated_data = await data_generator.generate_targeted_data(
            performance_gaps=performance_gaps,
            strategy=strategy_recommendation.primary_strategy.value,
            batch_size=strategy_recommendation.estimated_samples,
            focus_areas=strategy_recommendation.focus_areas
        )
        
        # Verify complete workflow results
        assert len(generated_data["features"]) > 0
        assert "targeting_info" in generated_data["metadata"]
        
        targeting_info = generated_data["metadata"]["targeting_info"]
        assert targeting_info["strategy_used"] == strategy_recommendation.primary_strategy.value
        assert targeting_info["gap_targeting_enabled"] is True
        
        print(f"Complete workflow successful: Generated {len(generated_data['features'])} "
              f"samples using {targeting_info['strategy_used']} strategy")
        
        # Verify data quality and targeting effectiveness
        features = generated_data["features"]
        effectiveness_scores = generated_data["effectiveness"]
        
        # Check that generated data has appropriate difficulty distribution
        difficulty_dist = difficulty_profile.distribution_weights
        expected_hard_ratio = difficulty_dist.get("hard", 0.33)
        
        # Count hard examples (effectiveness < 0.5 indicates harder examples)
        hard_examples = sum(1 for score in effectiveness_scores if score < 0.5)
        actual_hard_ratio = hard_examples / len(effectiveness_scores)
        
        # Allow some tolerance in difficulty distribution
        assert abs(actual_hard_ratio - expected_hard_ratio) < 0.2
        
        print(f"Difficulty distribution verification: Expected {expected_hard_ratio:.2f}, "
              f"Actual {actual_hard_ratio:.2f} hard examples ratio")


if __name__ == "__main__":
    # Run tests with real behavior verification
    pytest.main([__file__, "-v", "--tb=short"])
