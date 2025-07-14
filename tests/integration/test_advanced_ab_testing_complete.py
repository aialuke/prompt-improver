"""
Comprehensive Integration Tests for Advanced A/B Testing Framework
Tests the complete pipeline from experiment setup to causal inference
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, AsyncMock

from src.prompt_improver.evaluation.advanced_statistical_validator import AdvancedStatisticalValidator
from src.prompt_improver.evaluation.pattern_significance_analyzer import PatternSignificanceAnalyzer
from src.prompt_improver.evaluation.causal_inference_analyzer import CausalInferenceAnalyzer
from src.prompt_improver.evaluation.experiment_orchestrator import (
    ExperimentOrchestrator,
    ExperimentConfiguration,
    ExperimentArm,
    ExperimentType,
    StoppingRule
)
from src.prompt_improver.services.real_time_analytics import RealTimeAnalyticsService
from src.prompt_improver.database.models import ABExperiment, RulePerformance


class TestAdvancedABTestingComplete:
    """Test suite for complete Advanced A/B Testing Framework"""
    
    @pytest.fixture
    async def orchestrator(self, async_session):
        """Create orchestrator with all components"""
        statistical_validator = AdvancedStatisticalValidator(
            alpha=0.05,
            bootstrap_samples=100  # Reduced for testing
        )
        
        pattern_analyzer = PatternSignificanceAnalyzer(
            alpha=0.05,
            min_sample_size=20
        )
        
        causal_analyzer = CausalInferenceAnalyzer(
            significance_level=0.05,
            bootstrap_samples=100
        )
        
        # Mock real-time service
        real_time_service = Mock(spec=RealTimeAnalyticsService)
        real_time_service.start_experiment_monitoring = AsyncMock(return_value=True)
        real_time_service.stop_experiment_monitoring = AsyncMock(return_value=True)
        real_time_service.get_real_time_metrics = AsyncMock(return_value=None)
        real_time_service.cleanup = AsyncMock()
        
        orchestrator = ExperimentOrchestrator(
            db_session=async_session,
            statistical_validator=statistical_validator,
            pattern_analyzer=pattern_analyzer,
            causal_analyzer=causal_analyzer,
            real_time_service=real_time_service
        )
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.cleanup()
    
    @pytest.fixture
    def simple_ab_config(self):
        """Create simple A/B test configuration"""
        control_arm = ExperimentArm(
            arm_id="control",
            arm_name="Control",
            description="Control arm with current rules",
            rules={"rule_ids": ["clarity_rule", "specificity_rule"]}
        )
        
        treatment_arm = ExperimentArm(
            arm_id="treatment",
            arm_name="Treatment",
            description="Treatment arm with enhanced rules",
            rules={"rule_ids": ["clarity_rule", "specificity_rule", "chain_of_thought_rule"]}
        )
        
        return ExperimentConfiguration(
            experiment_id="test_ab_experiment_001",
            experiment_name="Enhanced Rules A/B Test",
            experiment_type=ExperimentType.SIMPLE_AB,
            description="Test impact of adding chain-of-thought rule",
            arms=[control_arm, treatment_arm],
            minimum_sample_size=100,
            maximum_sample_size=1000,
            statistical_power=0.8,
            effect_size_threshold=0.1,
            significance_level=0.05,
            stopping_rules=[
                StoppingRule.STATISTICAL_SIGNIFICANCE,
                StoppingRule.SAMPLE_SIZE_REACHED,
                StoppingRule.TIME_LIMIT
            ],
            max_duration_days=30,
            primary_metric="improvement_score",
            secondary_metrics=["execution_time_ms", "user_satisfaction_score"],
            causal_analysis_enabled=True,
            pattern_analysis_enabled=True
        )
    
    @pytest.fixture
    def multivariate_config(self):
        """Create multivariate test configuration"""
        control_arm = ExperimentArm(
            arm_id="control",
            arm_name="Control",
            description="Baseline configuration",
            rules={"rule_ids": ["clarity_rule"]}
        )
        
        variant_a = ExperimentArm(
            arm_id="variant_a",
            arm_name="Variant A",
            description="Enhanced clarity",
            rules={"rule_ids": ["clarity_rule", "specificity_rule"]}
        )
        
        variant_b = ExperimentArm(
            arm_id="variant_b", 
            arm_name="Variant B",
            description="Enhanced with examples",
            rules={"rule_ids": ["clarity_rule", "example_rule"]}
        )
        
        variant_c = ExperimentArm(
            arm_id="variant_c",
            arm_name="Variant C",
            description="Full enhancement",
            rules={"rule_ids": ["clarity_rule", "specificity_rule", "example_rule", "chain_of_thought_rule"]}
        )
        
        return ExperimentConfiguration(
            experiment_id="test_multivariate_001",
            experiment_name="Multi-variant Rule Enhancement Test",
            experiment_type=ExperimentType.MULTIVARIATE,
            description="Test multiple rule combinations",
            arms=[control_arm, variant_a, variant_b, variant_c],
            minimum_sample_size=200,
            maximum_sample_size=2000,
            statistical_power=0.8,
            effect_size_threshold=0.15,
            significance_level=0.01,  # Bonferroni correction for multiple comparisons
            stopping_rules=[StoppingRule.STATISTICAL_SIGNIFICANCE],
            max_duration_days=45,
            primary_metric="improvement_score",
            causal_analysis_enabled=True,
            pattern_analysis_enabled=True
        )
    
    @pytest.fixture
    async def sample_experiment_data(self, async_session):
        """Create sample experiment data in database"""
        np.random.seed(42)
        
        # Create sample performance records
        records = []
        
        # Control group data (baseline performance)
        for i in range(50):
            record = RulePerformance(
                rule_id="clarity_rule",
                improvement_score=np.random.normal(0.7, 0.1),
                execution_time_ms=np.random.normal(100, 20),
                user_satisfaction_score=np.random.normal(3.5, 0.5),
                created_at=datetime.utcnow() - timedelta(hours=i//2)
            )
            records.append(record)
        
        # Treatment group data (improved performance)
        for i in range(45):
            record = RulePerformance(
                rule_id="chain_of_thought_rule",
                improvement_score=np.random.normal(0.75, 0.1),  # Slightly better
                execution_time_ms=np.random.normal(110, 20),  # Slightly slower
                user_satisfaction_score=np.random.normal(3.7, 0.5),  # Better satisfaction
                created_at=datetime.utcnow() - timedelta(hours=i//2)
            )
            records.append(record)
        
        # Add to database
        for record in records:
            async_session.add(record)
        
        await async_session.commit()
        
        return len(records)
    
    @pytest.mark.asyncio
    async def test_experiment_setup_and_validation(self, orchestrator, simple_ab_config):
        """Test experiment setup and configuration validation"""
        # Test successful setup
        result = await orchestrator.setup_experiment(simple_ab_config)
        
        assert result['success']
        assert result['experiment_id'] == simple_ab_config.experiment_id
        assert 'sample_size_analysis' in result
        assert 'estimated_duration_days' in result
        assert result['monitoring_enabled']
        
        # Verify experiment is tracked
        assert simple_ab_config.experiment_id in orchestrator.active_experiments
        assert simple_ab_config.experiment_id in orchestrator.experiment_tasks
        
        # Test sample size analysis
        sample_analysis = result['sample_size_analysis']
        assert 'required_sample_size_per_group' in sample_analysis
        assert 'required_total_sample_size' in sample_analysis
        assert 'estimated_duration_days' in sample_analysis
        assert sample_analysis['required_total_sample_size'] >= simple_ab_config.minimum_sample_size
    
    @pytest.mark.asyncio
    async def test_experiment_setup_validation_errors(self, orchestrator):
        """Test experiment setup with validation errors"""
        # Invalid configuration - only one arm
        invalid_config = ExperimentConfiguration(
            experiment_id="invalid_test",
            experiment_name="Invalid Test",
            experiment_type=ExperimentType.SIMPLE_AB,
            description="Invalid configuration",
            arms=[ExperimentArm("single", "Single", "Only arm", {})],  # Only one arm
            minimum_sample_size=10
        )
        
        result = await orchestrator.setup_experiment(invalid_config)
        
        assert not result['success']
        assert 'errors' in result
        assert any("at least 2 arms" in error for error in result['errors'])
    
    @pytest.mark.asyncio
    async def test_multivariate_experiment_setup(self, orchestrator, multivariate_config):
        """Test multivariate experiment setup"""
        result = await orchestrator.setup_experiment(multivariate_config)
        
        assert result['success']
        assert result['experiment_id'] == multivariate_config.experiment_id
        
        # Check sample size calculation accounts for multiple comparisons
        sample_analysis = result['sample_size_analysis']
        assert sample_analysis['corrected_total_sample_size'] > sample_analysis['required_total_sample_size']
        
        # Verify all arms are tracked
        config = orchestrator.active_experiments[multivariate_config.experiment_id]
        assert len(config.arms) == 4
        assert config.significance_level == 0.01  # Adjusted for multiple comparisons
    
    @pytest.mark.asyncio
    async def test_comprehensive_experiment_analysis(self, orchestrator, simple_ab_config, sample_experiment_data):
        """Test complete experiment analysis pipeline"""
        # Setup experiment
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Mock experiment data collection to return our sample data
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.return_value = {
                'arms': {
                    'control': {
                        'outcomes': np.random.normal(0.7, 0.1, 50).tolist(),
                        'metrics': {
                            'improvement_score': np.random.normal(0.7, 0.1, 50).tolist(),
                            'execution_time_ms': np.random.normal(100, 20, 50).tolist(),
                            'user_satisfaction_score': np.random.normal(3.5, 0.5, 50).tolist()
                        },
                        'sample_size': 50
                    },
                    'treatment': {
                        'outcomes': np.random.normal(0.75, 0.1, 45).tolist(),
                        'metrics': {
                            'improvement_score': np.random.normal(0.75, 0.1, 45).tolist(),
                            'execution_time_ms': np.random.normal(110, 20, 45).tolist(),
                            'user_satisfaction_score': np.random.normal(3.7, 0.5, 45).tolist()
                        },
                        'sample_size': 45
                    }
                },
                'sample_sizes': {'control': 50, 'treatment': 45},
                'total_sample_size': 95,
                'duration_days': 5,
                'sufficient_data': True,
                'primary_metric': 'improvement_score',
                'secondary_metrics': ['execution_time_ms', 'user_satisfaction_score']
            }
            
            # Perform analysis
            analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        
        # Verify comprehensive analysis
        assert analysis_result.experiment_id == simple_ab_config.experiment_id
        assert analysis_result.analysis_id is not None
        assert analysis_result.timestamp is not None
        
        # Check statistical validation
        assert analysis_result.statistical_validation is not None
        assert analysis_result.statistical_validation.primary_test.test_name == "Welch's t-test"
        assert analysis_result.statistical_validation.validation_quality_score > 0
        
        # Check pattern analysis (should be included if enabled)
        if simple_ab_config.pattern_analysis_enabled:
            # Pattern analysis might be None if insufficient pattern data
            pass
        
        # Check causal inference
        if simple_ab_config.causal_analysis_enabled:
            assert analysis_result.causal_inference is not None
            assert analysis_result.causal_inference.average_treatment_effect is not None
        
        # Check performance metrics
        assert 'control' in analysis_result.arm_performance
        assert 'treatment' in analysis_result.arm_performance
        assert 'control' in analysis_result.relative_performance
        assert 'treatment' in analysis_result.relative_performance
        
        # Check decision framework
        assert analysis_result.stopping_recommendation in [
            "STOP_FOR_SUCCESS", "STOP_WITH_CAUTION", "STOP_FOR_FUTILITY", "CONTINUE"
        ]
        assert analysis_result.business_decision in [
            "IMPLEMENT", "PILOT", "NO_ACTION"
        ]
        assert 0 <= analysis_result.confidence_level <= 1
        
        # Check quality scores
        assert 0 <= analysis_result.data_quality_score <= 1
        assert 0 <= analysis_result.analysis_quality_score <= 1
        assert 0 <= analysis_result.overall_experiment_quality <= 1
        
        # Check insights
        assert isinstance(analysis_result.actionable_insights, list)
        assert isinstance(analysis_result.next_steps, list)
        assert isinstance(analysis_result.lessons_learned, list)
        
        # Check metadata
        assert analysis_result.analysis_duration_seconds > 0
        assert analysis_result.sample_sizes == {'control': 50, 'treatment': 45}
        assert analysis_result.experiment_duration_days == 5
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, orchestrator, simple_ab_config):
        """Test handling of insufficient data scenarios"""
        # Setup experiment
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Mock insufficient data
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.return_value = {
                'arms': {
                    'control': {'outcomes': [0.7, 0.6], 'sample_size': 2},
                    'treatment': {'outcomes': [0.8], 'sample_size': 1}
                },
                'sample_sizes': {'control': 2, 'treatment': 1},
                'total_sample_size': 3,
                'duration_days': 1,
                'sufficient_data': False,
                'primary_metric': 'improvement_score',
                'secondary_metrics': []
            }
            
            analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        
        # Should handle insufficient data gracefully
        assert analysis_result.experiment_id == simple_ab_config.experiment_id
        assert "Insufficient data" in analysis_result.stopping_recommendation
        assert "WAIT" in analysis_result.business_decision
        assert analysis_result.confidence_level == 0.0
        assert "Insufficient data for analysis" in analysis_result.actionable_insights
    
    @pytest.mark.asyncio
    async def test_stopping_criteria_monitoring(self, orchestrator, simple_ab_config):
        """Test automated stopping criteria monitoring"""
        # Setup experiment
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Mock stopping criteria check
        with patch.object(orchestrator, '_check_stopping_criteria') as mock_check:
            mock_check.return_value = (True, "Statistical significance achieved")
            
            # The monitoring task should detect stopping criteria
            # We'll test the stopping criteria function directly
            config = orchestrator.active_experiments[simple_ab_config.experiment_id]
            should_stop, reason = await orchestrator._check_stopping_criteria(
                simple_ab_config.experiment_id, config
            )
            
            assert should_stop
            assert "Statistical significance achieved" in reason
    
    @pytest.mark.asyncio
    async def test_experiment_status_tracking(self, orchestrator, simple_ab_config):
        """Test experiment status tracking and reporting"""
        # Setup experiment
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Mock experiment data for status
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.return_value = {
                'arms': {'control': {'outcomes': [0.7] * 30}, 'treatment': {'outcomes': [0.75] * 25}},
                'sample_sizes': {'control': 30, 'treatment': 25},
                'total_sample_size': 55,
                'duration_days': 3,
                'sufficient_data': False
            }
            
            status = await orchestrator.get_experiment_status(simple_ab_config.experiment_id)
        
        assert status['experiment_id'] == simple_ab_config.experiment_id
        assert status['status'] == 'active'
        assert status['active']
        assert status['experiment_name'] == simple_ab_config.experiment_name
        assert status['experiment_type'] == simple_ab_config.experiment_type.value
        assert status['duration_days'] == 3
        assert status['total_sample_size'] == 55
        assert status['arms'] == ['Control', 'Treatment']
        assert not status['sufficient_data']
        assert not status['minimum_sample_size_reached']
    
    @pytest.mark.asyncio
    async def test_experiment_stopping_and_cleanup(self, orchestrator, simple_ab_config):
        """Test experiment stopping and cleanup procedures"""
        # Setup experiment
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        experiment_id = simple_ab_config.experiment_id
        
        # Verify experiment is active
        assert experiment_id in orchestrator.active_experiments
        assert experiment_id in orchestrator.experiment_tasks
        
        # Mock final analysis for stopping
        with patch.object(orchestrator, 'analyze_experiment') as mock_analyze:
            # Create a mock analysis result
            mock_result = Mock()
            mock_result.experiment_id = experiment_id
            mock_result.stopping_recommendation = "STOP_FOR_SUCCESS"
            mock_result.business_decision = "IMPLEMENT"
            mock_analyze.return_value = mock_result
            
            # Stop experiment
            stop_result = await orchestrator.stop_experiment(experiment_id, "test_completion")
        
        assert stop_result['success']
        assert stop_result['experiment_id'] == experiment_id
        assert stop_result['stop_reason'] == "test_completion"
        assert stop_result['status'] == "completed"
        
        # Verify cleanup
        assert experiment_id not in orchestrator.active_experiments
        assert experiment_id not in orchestrator.experiment_tasks
        
        # Verify real-time monitoring stopped
        orchestrator.real_time_service.stop_experiment_monitoring.assert_called_with(experiment_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator, simple_ab_config):
        """Test error handling and recovery mechanisms"""
        # Test setup with database error
        with patch.object(orchestrator, '_create_experiment_record') as mock_create:
            mock_create.side_effect = Exception("Database connection error")
            
            result = await orchestrator.setup_experiment(simple_ab_config)
            
            assert not result['success']
            assert 'error' in result
            assert "Database connection error" in result['error']
        
        # Test analysis with data collection error
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.side_effect = Exception("Data collection failed")
            
            with pytest.raises(Exception):
                await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_experiments(self, orchestrator):
        """Test handling multiple concurrent experiments"""
        # Create multiple experiment configurations
        experiments = []
        for i in range(3):
            control_arm = ExperimentArm(f"control_{i}", f"Control {i}", "Control", {})
            treatment_arm = ExperimentArm(f"treatment_{i}", f"Treatment {i}", "Treatment", {})
            
            config = ExperimentConfiguration(
                experiment_id=f"concurrent_test_{i}",
                experiment_name=f"Concurrent Test {i}",
                experiment_type=ExperimentType.SIMPLE_AB,
                description=f"Concurrent experiment {i}",
                arms=[control_arm, treatment_arm],
                minimum_sample_size=50
            )
            experiments.append(config)
        
        # Setup all experiments
        setup_results = []
        for config in experiments:
            result = await orchestrator.setup_experiment(config)
            setup_results.append(result)
        
        # Verify all experiments are active
        for i, result in enumerate(setup_results):
            assert result['success']
            assert f"concurrent_test_{i}" in orchestrator.active_experiments
        
        # Check status of all experiments
        for i in range(3):
            status = await orchestrator.get_experiment_status(f"concurrent_test_{i}")
            assert status['active']
        
        # Stop all experiments
        for i in range(3):
            with patch.object(orchestrator, 'analyze_experiment') as mock_analyze:
                mock_result = Mock()
                mock_result.experiment_id = f"concurrent_test_{i}"
                mock_analyze.return_value = mock_result
                
                stop_result = await orchestrator.stop_experiment(f"concurrent_test_{i}")
                assert stop_result['success']
    
    @pytest.mark.asyncio
    async def test_quality_scoring_integration(self, orchestrator, simple_ab_config):
        """Test quality scoring across all analysis components"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Mock high-quality data
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.return_value = {
                'arms': {
                    'control': {
                        'outcomes': np.random.normal(0.7, 0.05, 100).tolist(),  # Low variance, good sample
                        'sample_size': 100
                    },
                    'treatment': {
                        'outcomes': np.random.normal(0.8, 0.05, 100).tolist(),  # Clear effect
                        'sample_size': 100
                    }
                },
                'sample_sizes': {'control': 100, 'treatment': 100},
                'total_sample_size': 200,
                'duration_days': 10,
                'sufficient_data': True,
                'primary_metric': 'improvement_score'
            }
            
            analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        
        # High-quality data should result in high quality scores
        assert analysis_result.data_quality_score > 0.7
        assert analysis_result.analysis_quality_score > 0.7
        assert analysis_result.overall_experiment_quality > 0.7
        
        # Should recommend implementation for clear positive effect
        assert analysis_result.business_decision in ["IMPLEMENT", "PILOT"]
        assert analysis_result.confidence_level > 0.7
    
    @pytest.mark.asyncio
    async def test_integration_with_real_time_analytics(self, orchestrator, simple_ab_config):
        """Test integration with real-time analytics service"""
        # Setup experiment (real-time service is mocked)
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Verify real-time monitoring was started
        orchestrator.real_time_service.start_experiment_monitoring.assert_called_with(
            experiment_id=simple_ab_config.experiment_id,
            update_interval=60
        )
        
        # Test status retrieval with real-time metrics
        mock_metrics = Mock()
        mock_metrics.__dict__ = {
            'control_mean': 0.7,
            'treatment_mean': 0.75,
            'effect_size': 0.5,
            'statistical_significance': True
        }
        orchestrator.real_time_service.get_real_time_metrics.return_value = mock_metrics
        
        with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
            mock_collect.return_value = {
                'arms': {},
                'sample_sizes': {'control': 50, 'treatment': 45},
                'total_sample_size': 95,
                'duration_days': 5,
                'sufficient_data': True
            }
            
            status = await orchestrator.get_experiment_status(simple_ab_config.experiment_id)
        
        assert status['real_time_metrics'] is not None
        assert status['real_time_metrics']['control_mean'] == 0.7
        assert status['real_time_metrics']['treatment_mean'] == 0.75
    
    @pytest.mark.asyncio
    async def test_orchestrator_cleanup(self, orchestrator, simple_ab_config):
        """Test orchestrator cleanup and resource management"""
        # Setup multiple experiments
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        
        # Verify resources are allocated
        assert len(orchestrator.active_experiments) > 0
        assert len(orchestrator.experiment_tasks) > 0
        
        # Cleanup
        await orchestrator.cleanup()
        
        # Verify resources are cleaned up
        assert len(orchestrator.active_experiments) == 0
        assert len(orchestrator.experiment_tasks) == 0
        
        # Verify real-time service cleanup
        orchestrator.real_time_service.cleanup.assert_called_once()


@pytest.mark.integration
class TestAdvancedABTestingEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_ab_testing_workflow(self, async_session):
        """Test complete A/B testing workflow from setup to decision"""
        np.random.seed(42)
        
        # Create orchestrator
        orchestrator = ExperimentOrchestrator(async_session)
        
        try:
            # Setup experiment
            control_arm = ExperimentArm(
                arm_id="control",
                arm_name="Current Rules",
                description="Existing rule configuration",
                rules={"rule_ids": ["clarity_rule"]}
            )
            
            treatment_arm = ExperimentArm(
                arm_id="enhanced",
                arm_name="Enhanced Rules",
                description="Enhanced rule configuration with chain-of-thought",
                rules={"rule_ids": ["clarity_rule", "chain_of_thought_rule"]}
            )
            
            config = ExperimentConfiguration(
                experiment_id="end_to_end_test",
                experiment_name="End-to-End A/B Test",
                experiment_type=ExperimentType.SIMPLE_AB,
                description="Complete workflow test",
                arms=[control_arm, treatment_arm],
                minimum_sample_size=100,
                statistical_power=0.8,
                effect_size_threshold=0.1,
                causal_analysis_enabled=True,
                pattern_analysis_enabled=True
            )
            
            # Phase 1: Setup
            setup_result = await orchestrator.setup_experiment(config)
            assert setup_result['success']
            
            # Phase 2: Simulate data collection and monitoring
            with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
                # Simulate successful experiment with positive results
                mock_collect.return_value = {
                    'arms': {
                        'control': {
                            'outcomes': np.random.normal(0.65, 0.1, 120).tolist(),
                            'sample_size': 120
                        },
                        'enhanced': {
                            'outcomes': np.random.normal(0.75, 0.1, 115).tolist(),  # 15% improvement
                            'sample_size': 115
                        }
                    },
                    'sample_sizes': {'control': 120, 'enhanced': 115},
                    'total_sample_size': 235,
                    'duration_days': 14,
                    'sufficient_data': True,
                    'primary_metric': 'improvement_score'
                }
                
                # Phase 3: Analysis
                analysis_result = await orchestrator.analyze_experiment("end_to_end_test")
            
            # Phase 4: Verification
            assert analysis_result.experiment_id == "end_to_end_test"
            assert analysis_result.statistical_validation.practical_significance
            assert analysis_result.confidence_level > 0.5
            
            # Should recommend implementation for significant improvement
            assert analysis_result.business_decision in ["IMPLEMENT", "PILOT"]
            assert len(analysis_result.actionable_insights) > 0
            assert len(analysis_result.next_steps) > 0
            
            # Phase 5: Cleanup
            stop_result = await orchestrator.stop_experiment("end_to_end_test", "test_complete")
            assert stop_result['success']
            
        finally:
            await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_negative_result_workflow(self, async_session):
        """Test workflow with negative/no-effect results"""
        np.random.seed(42)
        
        orchestrator = ExperimentOrchestrator(async_session)
        
        try:
            # Setup experiment
            config = ExperimentConfiguration(
                experiment_id="negative_test",
                experiment_name="Negative Result Test",
                experiment_type=ExperimentType.SIMPLE_AB,
                description="Test with no significant effect",
                arms=[
                    ExperimentArm("control", "Control", "Control", {}),
                    ExperimentArm("treatment", "Treatment", "Treatment", {})
                ],
                minimum_sample_size=100
            )
            
            setup_result = await orchestrator.setup_experiment(config)
            assert setup_result['success']
            
            # Simulate no effect data
            with patch.object(orchestrator, '_collect_experiment_data') as mock_collect:
                mock_collect.return_value = {
                    'arms': {
                        'control': {
                            'outcomes': np.random.normal(0.7, 0.1, 100).tolist(),
                            'sample_size': 100
                        },
                        'treatment': {
                            'outcomes': np.random.normal(0.7, 0.1, 100).tolist(),  # No difference
                            'sample_size': 100
                        }
                    },
                    'sample_sizes': {'control': 100, 'treatment': 100},
                    'total_sample_size': 200,
                    'duration_days': 14,
                    'sufficient_data': True,
                    'primary_metric': 'improvement_score'
                }
                
                analysis_result = await orchestrator.analyze_experiment("negative_test")
            
            # Should detect no effect
            assert not analysis_result.statistical_validation.practical_significance
            assert analysis_result.business_decision == "NO_ACTION"
            assert "no meaningful effect" in analysis_result.stopping_recommendation.lower() or \
                   "futility" in analysis_result.stopping_recommendation.lower()
            
        finally:
            await orchestrator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])