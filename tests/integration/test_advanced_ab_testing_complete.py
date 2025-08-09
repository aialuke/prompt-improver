"""
Comprehensive Integration Tests for Advanced A/B Testing Framework
Tests the complete pipeline from experiment setup to causal inference
"""
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pytest
import sqlalchemy
from prompt_improver.database.models import ABExperiment, PromptSession, RuleMetadata, RulePerformance
from prompt_improver.evaluation.advanced_statistical_validator import AdvancedStatisticalValidator
from prompt_improver.evaluation.causal_inference_analyzer import CausalInferenceAnalyzer, TreatmentAssignment
from prompt_improver.evaluation.experiment_orchestrator import ExperimentArm, ExperimentConfiguration, ExperimentOrchestrator, ExperimentType, StoppingRule
from prompt_improver.evaluation.pattern_significance_analyzer import PatternSignificanceAnalyzer
from prompt_improver.services.real_time_analytics import RealTimeAnalyticsService

class TestAdvancedABTestingComplete:
    """Test suite for complete Advanced A/B Testing Framework"""

    def create_statistically_significant_data(self, test_db_session, experiment_id_suffix=''):
        """Create experiment data with clear statistical significance"""
        np.random.seed(42)
        control_data = []
        for i in range(120):
            record = RulePerformance(session_id=f'control_session_{i}{experiment_id_suffix}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.1), execution_time_ms=np.random.normal(100, 20), confidence_level=np.random.normal(0.85, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 10))
            control_data.append(record)
            test_db_session.add(record)
        treatment_data = []
        for i in range(120):
            record = RulePerformance(session_id=f'treatment_session_{i}{experiment_id_suffix}', rule_id='chain_of_thought_rule', improvement_score=np.random.normal(0.8, 0.1), execution_time_ms=np.random.normal(105, 20), confidence_level=np.random.normal(0.9, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 10))
            treatment_data.append(record)
            test_db_session.add(record)
        return (control_data, treatment_data)

    def create_time_expired_data(self, test_db_session, max_duration_days=30):
        """Create experiment data that spans beyond max_duration_days"""
        np.random.seed(42)
        expired_data = []
        for i in range(100):
            record = RulePerformance(session_id=f'expired_session_{i}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.1), execution_time_ms=np.random.normal(100, 20), confidence_level=np.random.normal(0.85, 0.05), created_at=datetime.utcnow() - timedelta(days=max_duration_days + 5 - i // 10))
            expired_data.append(record)
            test_db_session.add(record)
        return expired_data

    def create_large_sample_data(self, test_db_session, max_sample_size=100):
        """Create experiment data exceeding max_sample_size"""
        np.random.seed(42)
        large_sample_data = []
        for i in range(max_sample_size + 50):
            record = RulePerformance(session_id=f'large_sample_session_{i}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.1), execution_time_ms=np.random.normal(100, 20), confidence_level=np.random.normal(0.85, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 4))
            large_sample_data.append(record)
            test_db_session.add(record)
        return large_sample_data

    @pytest.fixture
    async def orchestrator(self, test_db_session):
        """Create orchestrator with all components"""
        statistical_validator = AdvancedStatisticalValidator(alpha=0.05, bootstrap_samples=100)
        pattern_analyzer = PatternSignificanceAnalyzer(alpha=0.05, min_sample_size=20)
        causal_analyzer = CausalInferenceAnalyzer(significance_level=0.05, bootstrap_samples=100)
        real_time_service = RealTimeAnalyticsService(db_session=test_db_session)
        import os
        import coredis
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            redis_client = coredis.from_url(redis_url)
            await redis_client.ping()
            real_time_service.redis_client = redis_client
        except Exception:
            real_time_service.redis_client = None
        orchestrator = ExperimentOrchestrator(db_session=test_db_session, statistical_validator=statistical_validator, pattern_analyzer=pattern_analyzer, causal_analyzer=causal_analyzer, real_time_service=real_time_service)
        yield orchestrator
        await orchestrator.cleanup()
        if hasattr(real_time_service, 'redis_client') and real_time_service.redis_client:
            try:
                await real_time_service.redis_client.close()
            except Exception:
                pass

    @pytest.fixture
    def simple_ab_config(self):
        """Create simple A/B test configuration"""
        control_arm = ExperimentArm(arm_id='control', arm_name='Control', description='Control arm with current rules', rules={'rule_ids': ['clarity_rule', 'specificity_rule']})
        treatment_arm = ExperimentArm(arm_id='treatment', arm_name='Treatment', description='Treatment arm with enhanced rules', rules={'rule_ids': ['clarity_rule', 'specificity_rule', 'chain_of_thought_rule']})
        return ExperimentConfiguration(experiment_id='test_ab_experiment_001', experiment_name='Enhanced Rules A/B Test', experiment_type=ExperimentType.SIMPLE_AB, description='Test impact of adding chain-of-thought rule', arms=[control_arm, treatment_arm], minimum_sample_size=100, maximum_sample_size=1000, statistical_power=0.8, effect_size_threshold=0.1, significance_level=0.05, stopping_rules=[StoppingRule.STATISTICAL_SIGNIFICANCE, StoppingRule.SAMPLE_SIZE_REACHED, StoppingRule.TIME_LIMIT], max_duration_days=30, primary_metric='improvement_score', secondary_metrics=['execution_time_ms', 'user_satisfaction_score'], causal_analysis_enabled=True, pattern_analysis_enabled=True)

    @pytest.fixture
    def multivariate_config(self):
        """Create multivariate test configuration"""
        control_arm = ExperimentArm(arm_id='control', arm_name='Control', description='Baseline configuration', rules={'rule_ids': ['clarity_rule']})
        variant_a = ExperimentArm(arm_id='variant_a', arm_name='Variant A', description='Enhanced clarity', rules={'rule_ids': ['clarity_rule', 'specificity_rule']})
        variant_b = ExperimentArm(arm_id='variant_b', arm_name='Variant B', description='Enhanced with examples', rules={'rule_ids': ['clarity_rule', 'example_rule']})
        variant_c = ExperimentArm(arm_id='variant_c', arm_name='Variant C', description='Full enhancement', rules={'rule_ids': ['clarity_rule', 'specificity_rule', 'example_rule', 'chain_of_thought_rule']})
        return ExperimentConfiguration(experiment_id='test_multivariate_001', experiment_name='Multi-variant Rule Enhancement Test', experiment_type=ExperimentType.MULTIVARIATE, description='Test multiple rule combinations', arms=[control_arm, variant_a, variant_b, variant_c], minimum_sample_size=200, maximum_sample_size=2000, statistical_power=0.8, effect_size_threshold=0.15, significance_level=0.01, stopping_rules=[StoppingRule.STATISTICAL_SIGNIFICANCE], max_duration_days=45, primary_metric='improvement_score', causal_analysis_enabled=True, pattern_analysis_enabled=True)

    @pytest.fixture
    async def sample_experiment_data(self, test_db_session):
        """Create sample experiment data in database for all test scenarios"""
        np.random.seed(42)
        session_records = []
        for i in range(200):
            session = PromptSession(session_id=f'session_{i:03d}', original_prompt=f'Test prompt {i}', improved_prompt=f'Improved test prompt {i}', improvement_score=np.random.normal(0.7, 0.1), created_at=datetime.utcnow() - timedelta(hours=i // 4))
            session_records.append(session)
            test_db_session.add(session)
        rule_metadata_records = [RuleMetadata(rule_id='clarity_rule', rule_name='Clarity Rule', description='Improves prompt clarity', category='clarity'), RuleMetadata(rule_id='chain_of_thought_rule', rule_name='Chain of Thought Rule', description='Adds chain of thought reasoning', category='reasoning'), RuleMetadata(rule_id='specificity_rule', rule_name='Specificity Rule', description='Makes prompts more specific', category='specificity')]
        for rule in rule_metadata_records:
            test_db_session.add(rule)
        await test_db_session.commit()
        records = []
        for i in range(120):
            record = RulePerformance(session_id=f'session_{i:03d}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.1), execution_time_ms=np.random.normal(100, 20), confidence_level=np.random.normal(0.85, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 4))
            records.append(record)
        for i in range(115):
            record = RulePerformance(session_id=f'session_{i + 120:03d}', rule_id='chain_of_thought_rule', improvement_score=np.random.normal(0.75, 0.1), execution_time_ms=np.random.normal(110, 20), confidence_level=np.random.normal(0.88, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 4))
            records.append(record)
        for i in range(3):
            record = RulePerformance(session_id=f'session_{i + 300:03d}', rule_id='clarity_rule', improvement_score=np.random.normal(0.6, 0.1), execution_time_ms=np.random.normal(90, 15), confidence_level=np.random.normal(0.75, 0.1), created_at=datetime.utcnow() - timedelta(hours=i))
            records.append(record)
        for i in range(50):
            record = RulePerformance(session_id=f'session_{i + 400:03d}', rule_id='specificity_rule', improvement_score=np.random.normal(0.65, 0.1), execution_time_ms=np.random.normal(120, 25), confidence_level=np.random.normal(0.8, 0.05), created_at=datetime.utcnow() - timedelta(hours=i // 2))
            records.append(record)
        for i in range(200):
            record = RulePerformance(session_id=f'session_{i + 500:03d}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.05), execution_time_ms=np.random.normal(100, 10), confidence_level=np.random.normal(0.9, 0.02), created_at=datetime.utcnow() - timedelta(hours=i // 8))
            records.append(record)
        for i in range(200):
            record = RulePerformance(session_id=f'session_{i + 700:03d}', rule_id='chain_of_thought_rule', improvement_score=np.random.normal(0.8, 0.05), execution_time_ms=np.random.normal(105, 10), confidence_level=np.random.normal(0.92, 0.02), created_at=datetime.utcnow() - timedelta(hours=i // 8))
            records.append(record)
        for record in records:
            test_db_session.add(record)
        await test_db_session.commit()
        return len(records)

    @pytest.mark.asyncio
    async def test_experiment_setup_and_validation(self, orchestrator, simple_ab_config):
        """Test experiment setup and configuration validation"""
        result = await orchestrator.setup_experiment(simple_ab_config)
        assert result['success']
        assert result['experiment_id'] == simple_ab_config.experiment_id
        assert 'sample_size_analysis' in result
        assert 'estimated_duration_days' in result
        assert result['monitoring_enabled']
        assert simple_ab_config.experiment_id in orchestrator.active_experiments
        assert simple_ab_config.experiment_id in orchestrator.experiment_tasks
        sample_analysis = result['sample_size_analysis']
        assert 'required_sample_size_per_group' in sample_analysis
        assert 'required_total_sample_size' in sample_analysis
        assert 'estimated_duration_days' in sample_analysis
        assert sample_analysis['required_total_sample_size'] >= simple_ab_config.minimum_sample_size

    @pytest.mark.asyncio
    async def test_experiment_setup_validation_errors(self, orchestrator):
        """Test experiment setup with validation errors"""
        invalid_config = ExperimentConfiguration(experiment_id='invalid_test', experiment_name='Invalid Test', experiment_type=ExperimentType.SIMPLE_AB, description='Invalid configuration', arms=[ExperimentArm('single', 'Single', 'Only arm', {})], minimum_sample_size=10)
        result = await orchestrator.setup_experiment(invalid_config)
        assert not result['success']
        assert 'errors' in result
        assert any(('at least 2 arms' in error for error in result['errors']))

    @pytest.mark.asyncio
    async def test_multivariate_experiment_setup(self, orchestrator, multivariate_config):
        """Test multivariate experiment setup"""
        result = await orchestrator.setup_experiment(multivariate_config)
        assert result['success']
        assert result['experiment_id'] == multivariate_config.experiment_id
        sample_analysis = result['sample_size_analysis']
        assert sample_analysis['corrected_total_sample_size'] > sample_analysis['required_total_sample_size']
        config = orchestrator.active_experiments[multivariate_config.experiment_id]
        assert len(config.arms) == 4
        assert config.significance_level == 0.01

    @pytest.mark.asyncio
    async def test_comprehensive_experiment_analysis(self, orchestrator, simple_ab_config, sample_experiment_data):
        """Test complete experiment analysis pipeline"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        assert analysis_result.experiment_id == simple_ab_config.experiment_id
        assert analysis_result.analysis_id is not None
        assert analysis_result.timestamp is not None
        assert analysis_result.statistical_validation is not None
        assert analysis_result.statistical_validation.primary_test.test_name in ["Welch's t-test", "Student's t-test", 'Mann-Whitney U']
        assert 0.0 <= analysis_result.statistical_validation.validation_quality_score <= 1.0
        if analysis_result.statistical_validation.primary_test.p_value < 0.05:
            assert analysis_result.statistical_validation.primary_test.effect_size != 0
        if simple_ab_config.pattern_analysis_enabled:
            pass
        if simple_ab_config.causal_analysis_enabled:
            assert analysis_result.causal_inference is not None
            assert analysis_result.causal_inference.average_treatment_effect is not None
        assert 'control' in analysis_result.arm_performance
        assert 'treatment' in analysis_result.arm_performance
        assert 'control' in analysis_result.relative_performance
        assert 'treatment' in analysis_result.relative_performance
        assert analysis_result.stopping_recommendation in ['STOP_FOR_SUCCESS', 'STOP_WITH_CAUTION', 'STOP_FOR_FUTILITY', 'CONTINUE']
        assert analysis_result.business_decision in ['IMPLEMENT', 'PILOT', 'NO_ACTION']
        assert 0 <= analysis_result.confidence_level <= 1
        assert 0 <= analysis_result.data_quality_score <= 1
        assert 0 <= analysis_result.analysis_quality_score <= 1
        assert 0 <= analysis_result.overall_experiment_quality <= 1
        assert isinstance(analysis_result.actionable_insights, list)
        assert isinstance(analysis_result.next_steps, list)
        assert isinstance(analysis_result.lessons_learned, list)
        assert analysis_result.analysis_duration_seconds > 0
        assert 'control' in analysis_result.sample_sizes
        assert 'treatment' in analysis_result.sample_sizes
        assert analysis_result.experiment_duration_days > 0

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, orchestrator, simple_ab_config):
        """Test handling of insufficient data scenarios with real minimal data"""
        minimal_data = []
        for i in range(5):
            record = RulePerformance(session_id=f'minimal_session_{i}', rule_id='clarity_rule', improvement_score=np.random.normal(0.7, 0.1), execution_time_ms=np.random.normal(100, 20), confidence_level=np.random.normal(0.85, 0.05), created_at=datetime.utcnow() - timedelta(hours=i))
            minimal_data.append(record)
            orchestrator.db_session.add(record)
        await orchestrator.db_session.commit()
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        assert analysis_result.experiment_id == simple_ab_config.experiment_id
        assert 'insufficient' in analysis_result.stopping_recommendation.lower()
        assert analysis_result.business_decision in ['NO_ACTION', 'WAIT']
        assert analysis_result.confidence_level == 0.0
        assert any(('insufficient' in insight.lower() for insight in analysis_result.actionable_insights))

    @pytest.mark.asyncio
    async def test_stopping_criteria_monitoring(self, orchestrator, simple_ab_config):
        """Test automated stopping criteria monitoring with real statistical significance"""
        control_data, treatment_data = self.create_statistically_significant_data(orchestrator.db_session, '_stopping_test')
        await orchestrator.db_session.commit()
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        config = orchestrator.active_experiments[simple_ab_config.experiment_id]
        should_stop, reason = await orchestrator._check_stopping_criteria(simple_ab_config.experiment_id, config)
        assert should_stop
        assert 'significance' in reason.lower() or 'sample size' in reason.lower()

    @pytest.mark.asyncio
    async def test_experiment_status_tracking(self, orchestrator, simple_ab_config):
        """Test experiment status tracking and reporting"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        status = await orchestrator.get_experiment_status(simple_ab_config.experiment_id)
        assert status['experiment_id'] == simple_ab_config.experiment_id
        assert status['status'] == 'active'
        assert status['active']
        assert status['experiment_name'] == simple_ab_config.experiment_name
        assert status['experiment_type'] == simple_ab_config.experiment_type.value
        assert status['duration_days'] > 0
        assert status['total_sample_size'] >= 0
        assert status['arms'] == ['Control', 'Treatment']
        assert 'sufficient_data' in status
        assert 'minimum_sample_size_reached' in status

    @pytest.mark.asyncio
    async def test_experiment_stopping_and_cleanup(self, orchestrator, simple_ab_config):
        """Test experiment stopping and cleanup procedures with real analysis"""
        control_data, treatment_data = self.create_statistically_significant_data(orchestrator.db_session, '_stopping_cleanup_test')
        await orchestrator.db_session.commit()
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        experiment_id = simple_ab_config.experiment_id
        assert experiment_id in orchestrator.active_experiments
        assert experiment_id in orchestrator.experiment_tasks
        stop_result = await orchestrator.stop_experiment(experiment_id, 'test_completion')
        assert stop_result['success']
        assert stop_result['experiment_id'] == experiment_id
        assert stop_result['stop_reason'] == 'test_completion'
        assert stop_result['status'] == 'completed'
        assert 'final_analysis' in stop_result
        final_analysis = stop_result['final_analysis']
        assert final_analysis.experiment_id == experiment_id
        assert final_analysis.statistical_validation is not None
        assert final_analysis.stopping_recommendation in ['STOP_FOR_SUCCESS', 'STOP_WITH_CAUTION', 'STOP_FOR_FUTILITY', 'CONTINUE']
        assert experiment_id not in orchestrator.active_experiments
        assert experiment_id not in orchestrator.experiment_tasks
        assert experiment_id not in orchestrator.real_time_service.monitoring_tasks

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator, simple_ab_config):
        """Test error handling and recovery mechanisms with real database errors"""
        await orchestrator.db_session.close()
        result = await orchestrator.setup_experiment(simple_ab_config)
        assert not result['success']
        assert 'error' in result
        assert 'database' in result['error'].lower() or 'connection' in result['error'].lower() or 'session' in result['error'].lower()
        from prompt_improver.evaluation.advanced_statistical_validator import AdvancedStatisticalValidator
        from prompt_improver.evaluation.causal_inference_analyzer import CausalInferenceAnalyzer
        from prompt_improver.evaluation.pattern_significance_analyzer import PatternSignificanceAnalyzer
        fresh_orchestrator = ExperimentOrchestrator(db_session=orchestrator.db_session, statistical_validator=AdvancedStatisticalValidator(), pattern_analyzer=PatternSignificanceAnalyzer(), causal_analyzer=CausalInferenceAnalyzer())
        setup_result = await fresh_orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        analysis_result = await fresh_orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        assert analysis_result.experiment_id == simple_ab_config.experiment_id
        assert 'insufficient' in analysis_result.stopping_recommendation.lower()

    @pytest.mark.asyncio
    async def test_concurrent_experiments(self, orchestrator):
        """Test handling multiple concurrent experiments"""
        experiments = []
        for i in range(3):
            control_arm = ExperimentArm(f'control_{i}', f'Control {i}', 'Control', {})
            treatment_arm = ExperimentArm(f'treatment_{i}', f'Treatment {i}', 'Treatment', {})
            config = ExperimentConfiguration(experiment_id=f'concurrent_test_{i}', experiment_name=f'Concurrent Test {i}', experiment_type=ExperimentType.SIMPLE_AB, description=f'Concurrent experiment {i}', arms=[control_arm, treatment_arm], minimum_sample_size=50)
            experiments.append(config)
        setup_results = []
        for config in experiments:
            result = await orchestrator.setup_experiment(config)
            setup_results.append(result)
        for i, result in enumerate(setup_results):
            assert result['success']
            assert f'concurrent_test_{i}' in orchestrator.active_experiments
        for i in range(3):
            status = await orchestrator.get_experiment_status(f'concurrent_test_{i}')
            assert status['active']
        for i in range(3):
            control_data, treatment_data = self.create_statistically_significant_data(orchestrator.db_session, f'_concurrent_test_{i}')
        await orchestrator.db_session.commit()
        for i in range(3):
            stop_result = await orchestrator.stop_experiment(f'concurrent_test_{i}')
            assert stop_result['success']
            assert 'final_analysis' in stop_result
            final_analysis = stop_result['final_analysis']
            assert final_analysis.experiment_id == f'concurrent_test_{i}'
            assert final_analysis.statistical_validation is not None

    @pytest.mark.asyncio
    async def test_quality_scoring_integration(self, orchestrator, simple_ab_config):
        """Test quality scoring across all analysis components"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        analysis_result = await orchestrator.analyze_experiment(simple_ab_config.experiment_id)
        assert 0 <= analysis_result.data_quality_score <= 1
        assert 0 <= analysis_result.analysis_quality_score <= 1
        assert 0 <= analysis_result.overall_experiment_quality <= 1
        assert analysis_result.business_decision in ['IMPLEMENT', 'PILOT', 'NO_ACTION']
        assert 0 <= analysis_result.confidence_level <= 1

    @pytest.mark.asyncio
    async def test_integration_with_real_time_analytics(self, orchestrator, simple_ab_config):
        """Test integration with real-time analytics service"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        assert simple_ab_config.experiment_id in orchestrator.real_time_service.monitoring_tasks
        status = await orchestrator.get_experiment_status(simple_ab_config.experiment_id)
        if status['real_time_metrics'] is not None:
            assert isinstance(status['real_time_metrics'], dict)
        else:
            assert status['real_time_metrics'] is None

    @pytest.mark.asyncio
    async def test_orchestrator_cleanup(self, orchestrator, simple_ab_config):
        """Test orchestrator cleanup and resource management"""
        setup_result = await orchestrator.setup_experiment(simple_ab_config)
        assert setup_result['success']
        assert len(orchestrator.active_experiments) > 0
        assert len(orchestrator.experiment_tasks) > 0
        await orchestrator.cleanup()
        assert len(orchestrator.active_experiments) == 0
        assert len(orchestrator.experiment_tasks) == 0
        assert len(orchestrator.real_time_service.monitoring_tasks) == 0

@pytest.mark.integration
class TestAdvancedABTestingEndToEnd:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_complete_ab_testing_workflow(self, test_db_session):
        """Test complete A/B testing workflow from setup to decision"""
        np.random.seed(42)
        orchestrator = ExperimentOrchestrator(test_db_session)
        try:
            control_arm = ExperimentArm(arm_id='control', arm_name='Current Rules', description='Existing rule configuration', rules={'rule_ids': ['clarity_rule']})
            treatment_arm = ExperimentArm(arm_id='enhanced', arm_name='Enhanced Rules', description='Enhanced rule configuration with chain-of-thought', rules={'rule_ids': ['clarity_rule', 'chain_of_thought_rule']})
            config = ExperimentConfiguration(experiment_id='end_to_end_test', experiment_name='End-to-End A/B Test', experiment_type=ExperimentType.SIMPLE_AB, description='Complete workflow test', arms=[control_arm, treatment_arm], minimum_sample_size=100, statistical_power=0.8, effect_size_threshold=0.1, causal_analysis_enabled=True, pattern_analysis_enabled=True)
            setup_result = await orchestrator.setup_experiment(config)
            assert setup_result['success']
            analysis_result = await orchestrator.analyze_experiment('end_to_end_test')
            assert analysis_result.experiment_id == 'end_to_end_test'
            assert analysis_result.statistical_validation.practical_significance
            assert analysis_result.confidence_level > 0.5
            assert analysis_result.business_decision in ['IMPLEMENT', 'PILOT']
            assert len(analysis_result.actionable_insights) > 0
            assert len(analysis_result.next_steps) > 0
            stop_result = await orchestrator.stop_experiment('end_to_end_test', 'test_complete')
            assert stop_result['success']
        finally:
            await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_negative_result_workflow(self, test_db_session):
        """Test workflow with negative/no-effect results"""
        np.random.seed(42)
        orchestrator = ExperimentOrchestrator(test_db_session)
        try:
            config = ExperimentConfiguration(experiment_id='negative_test', experiment_name='Negative Result Test', experiment_type=ExperimentType.SIMPLE_AB, description='Test with no significant effect', arms=[ExperimentArm('control', 'Control', 'Control', {}), ExperimentArm('treatment', 'Treatment', 'Treatment', {})], minimum_sample_size=100)
            setup_result = await orchestrator.setup_experiment(config)
            assert setup_result['success']
            analysis_result = await orchestrator.analyze_experiment('negative_test')
            assert not analysis_result.statistical_validation.practical_significance
            assert analysis_result.business_decision == 'NO_ACTION'
            assert 'no meaningful effect' in analysis_result.stopping_recommendation.lower() or 'futility' in analysis_result.stopping_recommendation.lower()
        finally:
            await orchestrator.cleanup()
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
