#!/usr/bin/env python3
"""
Test Runner for Adaptive Data Generation - 2025 Best Practices
Executes comprehensive real behavior tests for Week 5 implementation.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.database import get_sessionmanager
from prompt_improver.database.models import (
    TrainingSession, RulePerformance, UserFeedback, ImprovementSession
)


async def setup_test_database():
    """Set up test database with sample data."""
    print("Setting up test database...")
    
    try:
        async with get_sessionmanager().get_session() as session:
            # Create sample improvement sessions
            for i in range(5):
                improvement_session = ImprovementSession(
                    session_id=f"test_setup_session_{i}",
                    original_prompt=f"Test original prompt {i}",
                    final_prompt=f"Test improved prompt {i}",
                    rules_applied=["clarity", "specificity"],
                    improvement_metrics={
                        "clarity_score": 0.5 + (i * 0.1),
                        "specificity_score": 0.4 + (i * 0.12),
                        "overall_score": 0.45 + (i * 0.11)
                    }
                )
                session.add(improvement_session)
            
            # Create sample rule performance data
            rule_ids = ["clarity_rule", "specificity_rule", "effectiveness_rule", "consistency_rule"]
            for i, rule_id in enumerate(rule_ids):
                for j in range(3):
                    rule_perf = RulePerformance(
                        rule_id=rule_id,
                        effectiveness_score=0.3 + (i * 0.1) + (j * 0.05),
                        consistency_score=0.4 + (i * 0.08) + (j * 0.04),
                        coverage_score=0.5 + (i * 0.06) + (j * 0.03),
                        confidence_score=0.6 + (i * 0.05) + (j * 0.02)
                    )
                    session.add(rule_perf)
            
            await session.commit()
            print("Test database setup completed successfully")
            
    except Exception as e:
        print(f"Error setting up test database: {e}")
        raise


async def test_performance_gap_analysis():
    """Test real performance gap analysis."""
    print("\n=== Testing Performance Gap Analysis ===")
    
    from prompt_improver.ml.analysis.performance_gap_analyzer import PerformanceGapAnalyzer
    
    try:
        gap_analyzer = PerformanceGapAnalyzer()
        
        async with get_sessionmanager().get_session() as session:
            # Test standard gap analysis
            gap_result = await gap_analyzer.analyze_performance_gaps(
                session=session,
                rule_ids=None,
                baseline_window=10
            )
            
            print(f"✓ Standard gap analysis: {gap_result.total_gaps_detected} gaps detected")
            print(f"  - Critical gaps: {len(gap_result.critical_gaps)}")
            print(f"  - Improvement opportunities: {len(gap_result.improvement_opportunities)}")
            print(f"  - Correlation score: {gap_result.correlation_score:.3f}")
            print(f"  - Stopping criteria met: {gap_result.stopping_criteria_met}")
            
            # Test enhanced gap analysis
            enhanced_result = await gap_analyzer.analyze_gaps_for_targeted_generation(
                session=session,
                rule_ids=None,
                focus_areas=["clarity", "specificity"]
            )
            
            print(f"✓ Enhanced gap analysis: {len(enhanced_result['enhanced_gaps'])} enhanced gaps")
            print(f"  - Generation config ready: {enhanced_result['metadata']['generation_ready']}")
            print(f"  - Recommended strategy: {enhanced_result['generation_config']['recommended_strategy']}")
            
    except Exception as e:
        print(f"✗ Performance gap analysis failed: {e}")
        raise


async def test_strategy_determination():
    """Test generation strategy determination."""
    print("\n=== Testing Strategy Determination ===")
    
    from prompt_improver.ml.analysis.performance_gap_analyzer import PerformanceGapAnalyzer
    from prompt_improver.ml.analysis.generation_strategy_analyzer import GenerationStrategyAnalyzer
    
    try:
        gap_analyzer = PerformanceGapAnalyzer()
        strategy_analyzer = GenerationStrategyAnalyzer()
        
        async with get_sessionmanager().get_session() as session:
            # Get gap analysis
            gap_result = await gap_analyzer.analyze_performance_gaps(session=session)
            
            # Determine strategy
            strategy_recommendation = await strategy_analyzer.analyze_optimal_strategy(
                gap_analysis=gap_result,
                hardness_analysis={"distribution": {"hard_examples_ratio": 0.3, "std": 0.2}},
                focus_areas=["clarity", "specificity"]
            )
            
            print(f"✓ Strategy determination completed")
            print(f"  - Primary strategy: {strategy_recommendation.primary_strategy.value}")
            print(f"  - Secondary strategy: {strategy_recommendation.secondary_strategy.value}")
            print(f"  - Confidence: {strategy_recommendation.confidence:.3f}")
            print(f"  - Estimated samples: {strategy_recommendation.estimated_samples}")
            print(f"  - Reasoning: {strategy_recommendation.reasoning}")
            
    except Exception as e:
        print(f"✗ Strategy determination failed: {e}")
        raise


async def test_difficulty_distribution():
    """Test difficulty distribution analysis."""
    print("\n=== Testing Difficulty Distribution ===")
    
    from prompt_improver.ml.analysis.performance_gap_analyzer import (
        PerformanceGapAnalyzer, PerformanceGap
    )
    from prompt_improver.ml.analysis.difficulty_distribution_analyzer import (
        DifficultyDistributionAnalyzer
    )
    
    try:
        gap_analyzer = PerformanceGapAnalyzer()
        difficulty_analyzer = DifficultyDistributionAnalyzer()
        
        async with get_sessionmanager().get_session() as session:
            # Get gap analysis
            gap_result = await gap_analyzer.analyze_performance_gaps(session=session)
            all_gaps = gap_result.critical_gaps + gap_result.improvement_opportunities
            
            # Analyze difficulty distribution
            difficulty_profile = await difficulty_analyzer.analyze_optimal_difficulty_distribution(
                performance_gaps=all_gaps,
                hardness_analysis={"distribution": {"hard_examples_ratio": 0.35, "std": 0.22}},
                focus_areas=["clarity", "specificity"]
            )
            
            print(f"✓ Difficulty distribution analysis completed")
            print(f"  - Distribution weights: {difficulty_profile.distribution_weights}")
            print(f"  - Hardness threshold: {difficulty_profile.hardness_threshold:.3f}")
            print(f"  - Focus areas: {difficulty_profile.focus_areas}")
            
            # Test focus area targeting
            focus_targets = await difficulty_analyzer.generate_focus_area_targets(
                performance_gaps=all_gaps,
                focus_areas=["clarity", "specificity", "effectiveness"],
                current_performance={
                    "clarity_effectiveness": 0.6,
                    "specificity_effectiveness": 0.5,
                    "effectiveness_effectiveness": 0.4
                },
                target_samples=500
            )
            
            print(f"✓ Focus area targeting: {len(focus_targets)} targets generated")
            for target in focus_targets:
                print(f"  - {target.area_name}: priority={target.priority:.3f}, "
                      f"allocation={target.sample_allocation:.3f}")
            
    except Exception as e:
        print(f"✗ Difficulty distribution analysis failed: {e}")
        raise


async def test_targeted_data_generation():
    """Test targeted synthetic data generation."""
    print("\n=== Testing Targeted Data Generation ===")
    
    from prompt_improver.ml.analysis.performance_gap_analyzer import PerformanceGapAnalyzer
    from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
    
    try:
        gap_analyzer = PerformanceGapAnalyzer()
        data_generator = ProductionSyntheticDataGenerator(
            target_samples=100,
            enable_gap_targeting=True,
            difficulty_distribution="adaptive"
        )
        
        async with get_sessionmanager().get_session() as session:
            # Get performance gaps
            gap_result = await gap_analyzer.analyze_performance_gaps(session=session)
            all_gaps = gap_result.critical_gaps + gap_result.improvement_opportunities
            
            # Extract performance gaps for targeting
            performance_gaps = {
                f"{gap.gap_type}_{gap.rule_id}": gap.gap_magnitude
                for gap in all_gaps
            }
            
            # Generate targeted data
            generated_data = await data_generator.generate_targeted_data(
                performance_gaps=performance_gaps,
                strategy="gap_based",
                batch_size=50,
                focus_areas=["clarity", "specificity"]
            )
            
            print(f"✓ Targeted data generation completed")
            print(f"  - Generated samples: {len(generated_data['features'])}")
            print(f"  - Strategy used: {generated_data['metadata']['targeting_info']['strategy_used']}")
            print(f"  - Gap targeting enabled: {generated_data['metadata']['targeting_info']['gap_targeting_enabled']}")
            
            # Verify data quality
            features = generated_data["features"]
            effectiveness = generated_data["effectiveness"]
            prompts = generated_data["prompts"]
            
            assert len(features) == len(effectiveness) == len(prompts)
            assert all(isinstance(f, list) for f in features)
            assert all(0.0 <= e <= 1.0 for e in effectiveness)
            assert all(isinstance(p, tuple) and len(p) == 2 for p in prompts)
            
            print(f"✓ Data quality verification passed")
            
    except Exception as e:
        print(f"✗ Targeted data generation failed: {e}")
        raise


async def test_adaptive_training_integration():
    """Test adaptive training integration."""
    print("\n=== Testing Adaptive Training Integration ===")
    
    from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
    from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
    from prompt_improver.ml.orchestration.coordinators.adaptive_training_coordinator import AdaptiveTrainingCoordinator
    
    try:
        # Create components
        orchestrator = MLPipelineOrchestrator()
        data_generator = ProductionSyntheticDataGenerator(
            target_samples=50,
            enable_gap_targeting=True
        )
        
        coordinator = AdaptiveTrainingCoordinator(
            orchestrator=orchestrator,
            data_generator=data_generator
        )
        
        # Start adaptive training session
        session_config = {
            "max_iterations": 2,  # Limited for testing
            "improvement_threshold": 0.01,
            "timeout_hours": 1
        }
        
        session_id = await coordinator.start_adaptive_training_session(
            session_config=session_config,
            focus_areas=["clarity", "specificity"]
        )
        
        print(f"✓ Adaptive training session started: {session_id}")
        
        # Wait for initialization
        await asyncio.sleep(1.0)
        
        # Check session status
        status = await coordinator.get_session_status(session_id)
        print(f"✓ Session status: {status['status']}")
        
        # Verify database record
        async with get_sessionmanager().get_session() as session:
            from sqlalchemy import select
            
            result = await session.execute(
                select(TrainingSession).where(TrainingSession.session_id == session_id)
            )
            training_session = result.scalar_one_or_none()
            
            if training_session:
                print(f"✓ Database record created: continuous_mode={training_session.continuous_mode}")
                print(f"  - Status: {training_session.status}")
                print(f"  - Adaptive training: {training_session.checkpoint_data.get('adaptive_training', False)}")
            else:
                print("✗ Database record not found")
        
        # Clean up
        if session_id in coordinator.active_sessions:
            del coordinator.active_sessions[session_id]
            
    except Exception as e:
        print(f"✗ Adaptive training integration failed: {e}")
        raise


async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Adaptive Data Generation Tests - Week 5 Implementation")
    print("=" * 70)
    
    try:
        # Setup
        await setup_test_database()
        
        # Run individual test suites
        await test_performance_gap_analysis()
        await test_strategy_determination()
        await test_difficulty_distribution()
        await test_targeted_data_generation()
        await test_adaptive_training_integration()
        
        print("\n" + "=" * 70)
        print("🎉 All adaptive data generation tests completed successfully!")
        print("✓ Performance gap analysis working")
        print("✓ Strategy determination working")
        print("✓ Difficulty distribution working")
        print("✓ Targeted data generation working")
        print("✓ Adaptive training integration working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return False


async def main():
    """Main test runner."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose logging during tests
    logging.getLogger("apes").setLevel(logging.WARNING)
    
    success = await run_comprehensive_tests()
    
    if success:
        print("\n🎯 Week 5 Adaptive Data Generation Implementation: VERIFIED")
        sys.exit(0)
    else:
        print("\n💥 Week 5 Implementation: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
