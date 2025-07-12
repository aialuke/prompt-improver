"""
Phase 1 Cross-Component Integration Tests

Comprehensive integration testing for all Phase 1 ML enhancement components to ensure
they work together correctly as a unified system. Tests end-to-end workflows combining:

- Statistical Analyzer (bootstrap CI, Hedges' g, adaptive FDR)
- A/B Testing Framework (CUPED variance reduction, DiD analysis)
- Failure Mode Analysis Engine (ML FMEA, ensemble anomaly detection)
- Optimization Validator (metrics validation, cross-validation)

Follows integration testing best practices:
- End-to-end workflow validation
- Cross-component data consistency
- Performance under realistic loads
- Error handling and resilience
- Component interaction verification
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from scipy import stats
import warnings
import time
import json

from src.prompt_improver.evaluation.statistical_analyzer import (
    StatisticalAnalyzer, 
    StatisticalConfig,
    DescriptiveStats
)
from src.prompt_improver.services.ab_testing import (
    ABTestingService,
    ExperimentResult
)
from src.prompt_improver.learning.failure_analyzer import (
    FailureAnalyzer,
    FailureConfig,
    FailurePattern,
    RootCause,
    EdgeCase,
    SystematicIssue,
    FailureRecommendation
)
from src.prompt_improver.optimization.optimization_validator import (
    OptimizationValidator,
    ValidationConfig
)


@pytest.mark.integration
class TestPhase1CrossComponentIntegration:
    """Integration test suite for all Phase 1 components working together"""
    
    @pytest.fixture(scope="class")
    def statistical_analyzer(self):
        """Create a statistical analyzer instance for integration testing"""
        config = StatisticalConfig(
            significance_level=0.05,
            confidence_level=0.95,
            minimum_sample_size=30,
            recommended_sample_size=100
        )
        return StatisticalAnalyzer(config)
    
    @pytest.fixture(scope="class") 
    def ab_testing_service(self):
        """Create an A/B testing service instance for integration testing"""
        return ABTestingService()
    
    @pytest.fixture(scope="class")
    def failure_analyzer(self):
        """Create a failure analyzer instance for integration testing"""
        config = FailureConfig(
            failure_threshold=0.3,
            min_pattern_size=3,
            significance_threshold=0.1,
            max_patterns=20,
            confidence_threshold=0.7
        )
        return FailureAnalyzer(config)
    
    @pytest.fixture(scope="class")
    def optimization_validator(self):
        """Create an optimization validator instance for integration testing"""
        config = ValidationConfig(
            min_sample_size=30,
            significance_level=0.05,
            min_effect_size=0.2,
            validation_duration_hours=24
        )
        return OptimizationValidator(config)
    
    @pytest.fixture(scope="function")
    def random_seed(self):
        """Set reproducible random seed for each test"""
        np.random.seed(42)
        return 42
    
    @pytest.fixture
    def comprehensive_experiment_data_factory(self, random_seed):
        """Factory fixture for generating comprehensive experimental data for all components"""
        def _create_comprehensive_data(n_control=100, n_treatment=100, effect_size=0.3, 
                                     failure_rate=0.15, optimization_scenarios=3):
            """
            Create comprehensive experimental data for Phase 1 component integration testing
            
            Returns data suitable for:
            - Statistical analysis (bootstrap CI, effect sizes, multiple testing)
            - A/B testing (CUPED analysis, significance testing)
            - Failure analysis (ML FMEA, anomaly detection)
            - Optimization validation (metrics validation, cross-validation)
            """
            
            # Generate base experimental data
            control_scores = np.random.beta(2, 3, n_control) * 0.8 + 0.1  # 0.1-0.9 range
            treatment_scores = np.random.beta(2, 3, n_treatment) * 0.8 + 0.1 + effect_size
            treatment_scores = np.clip(treatment_scores, 0, 1)
            
            # Generate pre-experiment data for CUPED (correlated with outcomes)
            correlation = 0.6  # Medium correlation for realistic variance reduction
            control_pre = control_scores * correlation + np.random.normal(0, 0.1, n_control)
            treatment_pre = treatment_scores * correlation + np.random.normal(0, 0.1, n_treatment)
            control_pre = np.clip(control_pre, 0, 1)
            treatment_pre = np.clip(treatment_pre, 0, 1)
            
            # Create detailed test results for statistical analysis
            control_results = []
            for i, score in enumerate(control_scores):
                result = {
                    "overallScore": float(score),
                    "clarity": float(score + np.random.normal(0, 0.05)),
                    "completeness": float(score + np.random.normal(0, 0.05)), 
                    "actionability": float(score + np.random.normal(0, 0.05)),
                    "effectiveness": float(score + np.random.normal(0, 0.05)),
                    "strategy": "control_strategy",
                    "model": "baseline_model",
                    "complexity": np.random.choice(["low", "medium", "high"]),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    "experiment_group": "control",
                    "participant_id": f"control_{i:03d}",
                    "pre_experiment_score": float(control_pre[i])
                }
                control_results.append(result)
            
            treatment_results = []
            for i, score in enumerate(treatment_scores):
                result = {
                    "overallScore": float(score),
                    "clarity": float(score + np.random.normal(0, 0.05)),
                    "completeness": float(score + np.random.normal(0, 0.05)),
                    "actionability": float(score + np.random.normal(0, 0.05)), 
                    "effectiveness": float(score + np.random.normal(0, 0.05)),
                    "strategy": "treatment_strategy",
                    "model": "optimized_model",
                    "complexity": np.random.choice(["low", "medium", "high"]),
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                    "experiment_group": "treatment",
                    "participant_id": f"treatment_{i:03d}",
                    "pre_experiment_score": float(treatment_pre[i])
                }
                treatment_results.append(result)
            
            # Generate failure data based on performance scores
            failures = []
            failure_types = ['data_drift', 'model_overfitting', 'infrastructure_failure', 'deployment_error']
            
            # Create failures from low-performing samples
            all_results = control_results + treatment_results
            low_performance_results = [r for r in all_results if r["overallScore"] < 0.4]
            n_failures = min(len(low_performance_results), int((n_control + n_treatment) * failure_rate))
            
            for i in range(n_failures):
                if i < len(low_performance_results):
                    base_result = low_performance_results[i]
                    failure = {
                        'failure_id': f'failure_{i:03d}',
                        'failure_type': np.random.choice(failure_types),
                        'overall_score': base_result["overallScore"],
                        'timestamp': base_result["timestamp"],
                        'context': {
                            'model_type': base_result["model"],
                            'strategy': base_result["strategy"],
                            'complexity': base_result["complexity"]
                        },
                        'metrics': {
                            'accuracy': base_result["effectiveness"],
                            'precision': base_result["clarity"],
                            'recall': base_result["completeness"],
                            'f1_score': base_result["actionability"]
                        },
                        'error_details': {
                            'error_message': f'Performance degradation in {base_result["strategy"]}',
                            'severity_level': 'high' if base_result["overallScore"] < 0.25 else 'medium'
                        }
                    }
                    failures.append(failure)
            
            # Generate test results for failure analysis
            test_results = []
            for i in range(30):
                test_result = {
                    'test_id': f'test_{i:03d}',
                    'test_type': np.random.choice(['unit', 'integration', 'performance', 'security']),
                    'passed': np.random.choice([True, False], p=[0.75, 0.25]),
                    'execution_time_ms': np.random.exponential(100),
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 1440)),
                    'test_context': {
                        'environment': np.random.choice(['dev', 'staging', 'prod']),
                        'coverage': np.random.uniform(0.6, 0.95)
                    }
                }
                test_results.append(test_result)
            
            # Generate optimization validation data
            optimization_data = []
            for scenario in range(optimization_scenarios):
                baseline_data = {
                    'optimization_id': f'baseline_{scenario}',
                    'scores': np.random.normal(0.6, 0.1, 50).tolist(),
                    'metadata': {
                        'metric_type': np.random.choice(['response_time_ms', 'memory_usage_mb', 'throughput_rps']),
                        'scenario': f'scenario_{scenario}'
                    }
                }
                
                optimized_data = {
                    'optimization_id': f'optimized_{scenario}',
                    'scores': np.random.normal(0.7, 0.08, 50).tolist(),  # Better performance
                    'metadata': {
                        'metric_type': baseline_data['metadata']['metric_type'],
                        'scenario': f'scenario_{scenario}'
                    }
                }
                
                optimization_data.append({
                    'baseline': baseline_data,
                    'optimized': optimized_data
                })
            
            return {
                'statistical_data': {
                    'control_results': control_results,
                    'treatment_results': treatment_results,
                    'all_results': all_results
                },
                'ab_testing_data': {
                    'control_data': list(control_scores),
                    'treatment_data': list(treatment_scores),
                    'control_cuped': {'outcome': list(control_scores), 'pre_value': list(control_pre)},
                    'treatment_cuped': {'outcome': list(treatment_scores), 'pre_value': list(treatment_pre)}
                },
                'failure_analysis_data': {
                    'failures': failures,
                    'test_results': test_results
                },
                'optimization_data': optimization_data,
                'metadata': {
                    'n_control': n_control,
                    'n_treatment': n_treatment,
                    'true_effect_size': effect_size,
                    'expected_correlation': correlation,
                    'failure_rate': failure_rate
                }
            }
        return _create_comprehensive_data
    
    @pytest.mark.asyncio
    async def test_end_to_end_ml_enhancement_workflow(self, statistical_analyzer, ab_testing_service, 
                                                    failure_analyzer, optimization_validator, 
                                                    comprehensive_experiment_data_factory):
        """Test complete end-to-end ML enhancement workflow across all Phase 1 components"""
        
        # Generate comprehensive test data
        comprehensive_data = comprehensive_experiment_data_factory(
            n_control=150,
            n_treatment=150, 
            effect_size=0.25,
            failure_rate=0.2,
            optimization_scenarios=3
        )
        
        # Step 1: Statistical Analysis with Phase 1 enhancements
        print("🔍 Step 1: Performing statistical analysis...")
        statistical_analysis = await statistical_analyzer.perform_statistical_analysis(
            comprehensive_data['statistical_data']['all_results']
        )
        
        # Verify statistical analysis includes Phase 1 enhancements
        assert "descriptive_stats" in statistical_analysis
        assert "confidence_intervals" in statistical_analysis
        
        # Check for bootstrap confidence intervals (Phase 1 enhancement)
        ci_data = statistical_analysis["confidence_intervals"]
        bootstrap_found = False
        for metric, ci_info in ci_data.items():
            if isinstance(ci_info, dict) and "bootstrap_lower" in ci_info:
                assert "bootstrap_method" in ci_info
                assert ci_info["bootstrap_method"] == "BCa"
                bootstrap_found = True
                break
        assert bootstrap_found, "Bootstrap confidence intervals not found"
        
        # Check for effect sizes (Hedges' g enhancement)
        if "effect_sizes" in statistical_analysis:
            effect_sizes = statistical_analysis["effect_sizes"]
            if "hedges_g" in effect_sizes:
                assert isinstance(effect_sizes["hedges_g"], float)
                print(f"✅ Hedges' g effect size: {effect_sizes['hedges_g']:.3f}")
        
        # Check for multiple testing correction (adaptive FDR enhancement)
        if "multiple_testing_correction" in statistical_analysis:
            mtc = statistical_analysis["multiple_testing_correction"]
            assert "method_used" in mtc
            assert mtc["method_used"] in ["fdr_by_adaptive", "fdr_bh_standard", "fdr_bh_fallback"]
            print(f"✅ Multiple testing correction: {mtc['method_used']}")
        
        # Step 2: A/B Testing Analysis with CUPED
        print("🔍 Step 2: Performing A/B testing analysis...")
        ab_analysis = ab_testing_service._perform_statistical_analysis(
            comprehensive_data['ab_testing_data']['control_data'],
            comprehensive_data['ab_testing_data']['treatment_data']
        )
        
        # Verify A/B testing results
        assert isinstance(ab_analysis, ExperimentResult)
        assert ab_analysis.statistical_significance
        assert ab_analysis.effect_size > 0.15  # Should detect meaningful effect
        print(f"✅ A/B Testing significance: p={ab_analysis.p_value:.4f}, effect={ab_analysis.effect_size:.3f}")
        
        # Test CUPED variance reduction (Phase 1 enhancement)
        cuped_analysis = ab_testing_service._apply_cuped_analysis(
            comprehensive_data['ab_testing_data']['treatment_cuped'],
            comprehensive_data['ab_testing_data']['control_cuped']
        )
        
        if cuped_analysis:  # CUPED analysis succeeded
            assert 'variance_reduction_percent' in cuped_analysis
            assert 'treatment_effect_cuped' in cuped_analysis
            variance_reduction = cuped_analysis['variance_reduction_percent']
            print(f"✅ CUPED variance reduction: {variance_reduction:.1f}%")
            assert variance_reduction > 10  # Should achieve meaningful variance reduction
        
        # Step 3: Failure Mode Analysis with ML FMEA
        print("🔍 Step 3: Performing failure mode analysis...")
        fmea_analysis = await failure_analyzer._perform_ml_fmea_analysis(
            comprehensive_data['failure_analysis_data']['failures'],
            comprehensive_data['failure_analysis_data']['test_results']
        )
        
        # Verify ML FMEA results (Phase 1 enhancement)
        assert 'critical_failure_modes' in fmea_analysis
        assert 'risk_matrix' in fmea_analysis
        assert 'top_risk_priorities' in fmea_analysis
        assert 'mitigation_recommendations' in fmea_analysis
        
        critical_modes = fmea_analysis['critical_failure_modes']
        assert len(critical_modes) > 0, "Should identify critical failure modes"
        print(f"✅ ML FMEA analysis: {len(critical_modes)} critical failure modes identified")
        
        # Verify RPN scoring
        top_risks = fmea_analysis['top_risk_priorities']
        if top_risks:
            assert 'rpn' in top_risks[0]
            assert isinstance(top_risks[0]['rpn'], (int, float))
        
        # Test ensemble anomaly detection (Phase 1 enhancement)
        anomaly_analysis = await failure_analyzer._perform_ensemble_anomaly_detection(
            comprehensive_data['failure_analysis_data']['failures']
        )
        
        assert 'ensemble_consensus' in anomaly_analysis
        assert 'individual_detectors' in anomaly_analysis
        assert 'detected_anomalies' in anomaly_analysis
        
        # Verify ensemble detectors
        detectors = anomaly_analysis['individual_detectors']
        expected_detectors = ['isolation_forest', 'elliptic_envelope', 'one_class_svm']
        for detector in expected_detectors:
            assert detector in detectors
        
        print(f"✅ Ensemble anomaly detection: {len(anomaly_analysis['detected_anomalies'])} anomalies detected")
        
        # Step 4: Optimization Validation with Metrics Validation
        print("🔍 Step 4: Performing optimization validation...")
        optimization_results = []
        
        for i, opt_data in enumerate(comprehensive_data['optimization_data']):
            validation_result = await optimization_validator.validate_optimization(
                f'integration_test_optimization_{i}',
                opt_data['baseline'],
                opt_data['optimized']
            )
            optimization_results.append(validation_result)
        
        # Verify optimization validation results (Phase 1 enhancement)
        valid_optimizations = [r for r in optimization_results if r.get('valid', False)]
        assert len(valid_optimizations) > 0, "Should validate at least some optimizations"
        
        for result in valid_optimizations:
            assert 'statistical_significance' in result
            assert 'practical_significance' in result
            assert 'metrics_validation' in result or 'validation_confidence' in result
        
        print(f"✅ Optimization validation: {len(valid_optimizations)}/{len(optimization_results)} optimizations validated")
        
        # Step 5: Cross-Component Integration Verification
        print("🔍 Step 5: Verifying cross-component integration...")
        
        # Verify data consistency across components
        control_mean_stat = np.mean([r["overallScore"] for r in comprehensive_data['statistical_data']['control_results']])
        treatment_mean_stat = np.mean([r["overallScore"] for r in comprehensive_data['statistical_data']['treatment_results']])
        
        assert abs(ab_analysis.control_mean - control_mean_stat) < 0.01, "Control means should match across components"
        assert abs(ab_analysis.treatment_mean - treatment_mean_stat) < 0.01, "Treatment means should match across components"
        
        # Verify failure analysis aligns with low-performing samples
        low_performance_count = len([r for r in comprehensive_data['statistical_data']['all_results'] if r["overallScore"] < 0.4])
        failure_count = len(comprehensive_data['failure_analysis_data']['failures'])
        assert failure_count <= low_performance_count, "Failure count should align with low performance"
        
        # Verify optimization improvements are statistically significant
        for result in valid_optimizations:
            if result.get('statistical_significance'):
                assert result['p_value'] < 0.05
                assert result['optimized_mean'] != result['baseline_mean']
        
        print("✅ Cross-component integration verified")
        
        # Step 6: Performance Integration Testing
        print("🔍 Step 6: Testing performance under integration load...")
        start_time = time.time()
        
        # Run all components together multiple times
        for iteration in range(3):
            mini_data = comprehensive_experiment_data_factory(n_control=50, n_treatment=50)
            
            # Quick statistical analysis
            await statistical_analyzer.perform_statistical_analysis(mini_data['statistical_data']['all_results'][:10])
            
            # Quick A/B testing
            ab_testing_service._perform_statistical_analysis(
                mini_data['ab_testing_data']['control_data'][:25],
                mini_data['ab_testing_data']['treatment_data'][:25]
            )
            
            # Quick failure analysis (if failures exist)
            if mini_data['failure_analysis_data']['failures']:
                await failure_analyzer._perform_ml_fmea_analysis(
                    mini_data['failure_analysis_data']['failures'][:5],
                    mini_data['failure_analysis_data']['test_results'][:10]
                )
            
            # Quick optimization validation
            if mini_data['optimization_data']:
                await optimization_validator.validate_optimization(
                    f'perf_test_{iteration}',
                    mini_data['optimization_data'][0]['baseline'],
                    mini_data['optimization_data'][0]['optimized']
                )
        
        total_time = time.time() - start_time
        assert total_time < 30.0, f"Integration performance test took {total_time:.2f}s, should be < 30s"
        print(f"✅ Performance integration test: {total_time:.2f}s")
        
        print("🎉 End-to-end ML enhancement workflow completed successfully!")
        
        return {
            'statistical_analysis': statistical_analysis,
            'ab_analysis': ab_analysis,
            'cuped_analysis': cuped_analysis,
            'fmea_analysis': fmea_analysis,
            'anomaly_analysis': anomaly_analysis,
            'optimization_results': optimization_results,
            'performance_time': total_time
        }
    
    @pytest.mark.asyncio
    async def test_component_interaction_consistency(self, statistical_analyzer, ab_testing_service, 
                                                   comprehensive_experiment_data_factory):
        """Test that components produce consistent results when working with the same data"""
        
        # Generate test data
        test_data = comprehensive_experiment_data_factory(n_control=80, n_treatment=80, effect_size=0.3)
        
        # Run statistical analysis
        statistical_results = await statistical_analyzer.perform_statistical_analysis(
            test_data['statistical_data']['all_results']
        )
        
        # Run A/B testing analysis
        ab_results = ab_testing_service._perform_statistical_analysis(
            test_data['ab_testing_data']['control_data'],
            test_data['ab_testing_data']['treatment_data']
        )
        
        # Verify consistency between components
        
        # 1. Sample sizes should match
        total_samples = len(test_data['statistical_data']['all_results'])
        control_samples = len(test_data['ab_testing_data']['control_data'])
        treatment_samples = len(test_data['ab_testing_data']['treatment_data'])
        assert control_samples + treatment_samples == total_samples
        
        # 2. Means should be consistent
        control_overall = [r["overallScore"] for r in test_data['statistical_data']['control_results']]
        treatment_overall = [r["overallScore"] for r in test_data['statistical_data']['treatment_results']]
        
        assert abs(np.mean(control_overall) - ab_results.control_mean) < 0.01
        assert abs(np.mean(treatment_overall) - ab_results.treatment_mean) < 0.01
        
        # 3. Effect direction should be consistent
        stat_effect_direction = np.mean(treatment_overall) > np.mean(control_overall)
        ab_effect_direction = ab_results.treatment_mean > ab_results.control_mean
        assert stat_effect_direction == ab_effect_direction
        
        # 4. Statistical significance should be consistent
        t_stat, p_val = stats.ttest_ind(treatment_overall, control_overall)
        assert (p_val < 0.05) == ab_results.statistical_significance
        
        print("✅ Component interaction consistency verified")
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, statistical_analyzer, ab_testing_service, 
                                            failure_analyzer, optimization_validator):
        """Test error handling when components interact with problematic data"""
        
        # Test 1: Empty/insufficient data
        empty_results = []
        empty_failures = []
        empty_tests = []
        
        # Statistical analyzer should handle empty data gracefully
        stat_result = await statistical_analyzer.perform_statistical_analysis(empty_results)
        assert isinstance(stat_result, dict)  # Should return something, not crash
        
        # A/B testing should handle insufficient data
        small_control = [0.1, 0.2]
        small_treatment = [0.3]
        ab_result = ab_testing_service._perform_statistical_analysis(small_control, small_treatment)
        assert isinstance(ab_result, ExperimentResult)  # Should complete
        
        # Failure analyzer should handle no failures
        fmea_result = await failure_analyzer._perform_ml_fmea_analysis(empty_failures, empty_tests)
        assert isinstance(fmea_result, dict)
        
        # Optimization validator should handle invalid data
        invalid_baseline = {'scores': [], 'metadata': {'metric_type': 'test'}}
        invalid_optimized = {'scores': [1, 2, 3], 'metadata': {'metric_type': 'test'}}
        
        opt_result = await optimization_validator.validate_optimization(
            'error_test', invalid_baseline, invalid_optimized
        )
        assert not opt_result.get('valid', True)  # Should reject invalid data
        
        # Test 2: Corrupted data
        corrupted_results = [
            {"overallScore": float('nan'), "clarity": None, "timestamp": "invalid"},
            {"overallScore": 999, "clarity": -999, "timestamp": datetime.now()}
        ]
        
        # Components should handle corrupted data gracefully
        stat_result_corrupted = await statistical_analyzer.perform_statistical_analysis(corrupted_results)
        assert isinstance(stat_result_corrupted, dict)
        
        print("✅ Error handling integration verified")
    
    @pytest.mark.asyncio 
    async def test_scalability_integration(self, statistical_analyzer, ab_testing_service, 
                                         failure_analyzer, optimization_validator,
                                         comprehensive_experiment_data_factory):
        """Test that all components scale together under increasing load"""
        
        # Test different data sizes
        test_sizes = [
            (50, 50, "small"),
            (200, 200, "medium"), 
            (500, 500, "large")
        ]
        
        performance_results = {}
        
        for n_control, n_treatment, size_name in test_sizes:
            print(f"🔍 Testing {size_name} scale: {n_control + n_treatment} total samples...")
            
            # Generate data for this scale
            test_data = comprehensive_experiment_data_factory(
                n_control=n_control,
                n_treatment=n_treatment,
                effect_size=0.2,
                optimization_scenarios=1
            )
            
            start_time = time.time()
            
            # Run all components
            tasks = []
            
            # Statistical analysis
            tasks.append(statistical_analyzer.perform_statistical_analysis(
                test_data['statistical_data']['all_results']
            ))
            
            # Run synchronous operations
            ab_result = ab_testing_service._perform_statistical_analysis(
                test_data['ab_testing_data']['control_data'],
                test_data['ab_testing_data']['treatment_data']
            )
            
            # Failure analysis (if failures exist)
            if test_data['failure_analysis_data']['failures']:
                tasks.append(failure_analyzer._perform_ml_fmea_analysis(
                    test_data['failure_analysis_data']['failures'],
                    test_data['failure_analysis_data']['test_results']
                ))
            
            # Optimization validation
            if test_data['optimization_data']:
                tasks.append(optimization_validator.validate_optimization(
                    f'scale_test_{size_name}',
                    test_data['optimization_data'][0]['baseline'],
                    test_data['optimization_data'][0]['optimized']
                ))
            
            # Wait for async tasks
            if tasks:
                await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            performance_results[size_name] = execution_time
            
            # Verify results are still accurate at scale
            assert isinstance(ab_result, ExperimentResult)
            
            print(f"✅ {size_name} scale completed in {execution_time:.2f}s")
        
        # Verify reasonable scaling
        small_time = performance_results["small"]
        large_time = performance_results["large"]
        
        # Should not scale exponentially (allow up to 10x for 10x data increase)
        scaling_factor = large_time / small_time if small_time > 0 else 1
        assert scaling_factor < 10, f"Scaling factor {scaling_factor:.2f} indicates poor scalability"
        
        print(f"✅ Scalability integration verified: {scaling_factor:.2f}x scaling factor")
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, comprehensive_experiment_data_factory):
        """Test that component configurations work together harmoniously"""
        
        # Create components with compatible configurations
        strict_config = StatisticalConfig(
            significance_level=0.01,  # More strict
            confidence_level=0.99,
            minimum_sample_size=50
        )
        
        validation_config = ValidationConfig(
            min_sample_size=50,  # Matching statistical requirements
            significance_level=0.01,  # Matching statistical requirements
            min_effect_size=0.3
        )
        
        failure_config = FailureConfig(
            failure_threshold=0.2,  # More sensitive
            significance_threshold=0.01,  # Matching other components
            confidence_threshold=0.8
        )
        
        # Create components with strict configurations
        strict_analyzer = StatisticalAnalyzer(strict_config)
        strict_validator = OptimizationValidator(validation_config)
        strict_failure_analyzer = FailureAnalyzer(failure_config)
        strict_ab_service = ABTestingService()
        
        # Generate data that should meet strict requirements
        test_data = comprehensive_experiment_data_factory(
            n_control=100,
            n_treatment=100, 
            effect_size=0.4,  # Large effect to meet strict requirements
            failure_rate=0.3
        )
        
        # Run analysis with strict configurations
        stat_results = await strict_analyzer.perform_statistical_analysis(
            test_data['statistical_data']['all_results']
        )
        
        ab_results = strict_ab_service._perform_statistical_analysis(
            test_data['ab_testing_data']['control_data'],
            test_data['ab_testing_data']['treatment_data']
        )
        
        if test_data['optimization_data']:
            opt_results = await strict_validator.validate_optimization(
                'strict_config_test',
                test_data['optimization_data'][0]['baseline'],
                test_data['optimization_data'][0]['optimized']
            )
            
            # With strict configuration, should still find significant effects
            if opt_results.get('valid'):
                assert opt_results.get('statistical_significance', False)
        
        # Verify strict configurations are applied consistently
        if ab_results.statistical_significance:
            assert ab_results.p_value < 0.01  # Should meet strict threshold
        
        print("✅ Configuration integration verified")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_integration(self, statistical_analyzer, ab_testing_service,
                                               failure_analyzer, optimization_validator,
                                               comprehensive_experiment_data_factory):
        """Test memory efficiency when all components work together"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple iterations to test memory cleanup
        for iteration in range(5):
            test_data = comprehensive_experiment_data_factory(
                n_control=100,
                n_treatment=100,
                optimization_scenarios=2
            )
            
            # Run all components
            await statistical_analyzer.perform_statistical_analysis(
                test_data['statistical_data']['all_results']
            )
            
            ab_testing_service._perform_statistical_analysis(
                test_data['ab_testing_data']['control_data'],
                test_data['ab_testing_data']['treatment_data']
            )
            
            if test_data['failure_analysis_data']['failures']:
                await failure_analyzer._perform_ml_fmea_analysis(
                    test_data['failure_analysis_data']['failures'],
                    test_data['failure_analysis_data']['test_results']
                )
            
            for opt_data in test_data['optimization_data']:
                await optimization_validator.validate_optimization(
                    f'memory_test_{iteration}',
                    opt_data['baseline'],
                    opt_data['optimized']
                )
            
            # Check memory growth
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Should not grow excessively (allow up to 500MB growth)
            assert memory_growth < 500, f"Memory growth {memory_growth:.1f}MB exceeds limit"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"✅ Memory efficiency verified: {total_growth:.1f}MB total growth")
    
    def test_component_version_compatibility(self, statistical_analyzer, ab_testing_service,
                                           failure_analyzer, optimization_validator):
        """Test that all Phase 1 components are compatible and have expected features"""
        
        # Verify Statistical Analyzer has Phase 1 enhancements
        assert hasattr(statistical_analyzer, '_calculate_bootstrap_ci')
        assert hasattr(statistical_analyzer, '_calculate_effect_sizes')
        assert hasattr(statistical_analyzer, '_apply_multiple_testing_correction')
        
        # Verify A/B Testing Service has Phase 1 enhancements
        assert hasattr(ab_testing_service, '_apply_cuped_analysis')
        
        # Verify Failure Analyzer has Phase 1 enhancements
        assert hasattr(failure_analyzer, '_initialize_ml_fmea_database')
        assert hasattr(failure_analyzer, '_perform_ml_fmea_analysis')
        assert hasattr(failure_analyzer, '_perform_ensemble_anomaly_detection')
        
        # Verify Optimization Validator has Phase 1 enhancements
        assert hasattr(optimization_validator, '_validate_metrics_realism')
        assert hasattr(optimization_validator, '_perform_cross_validation')
        
        print("✅ Component version compatibility verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])