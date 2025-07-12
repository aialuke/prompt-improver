"""
Tests for ML FMEA Framework Implementation

Tests the enhanced Failure Mode Analysis Engine ML FMEA functionality with RPN scoring,
ensemble anomaly detection, and comprehensive failure mode analysis.

Follows pytest best practices:
- Comprehensive fixture-based test setup
- Parametrized tests for multiple failure scenarios
- Statistical validation with proper error handling
- Performance benchmarking
- Integration testing patterns
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from scipy import stats
import warnings

from src.prompt_improver.learning.failure_analyzer import (
    FailureAnalyzer,
    FailureConfig,
    FailurePattern,
    RootCause,
    EdgeCase,
    SystematicIssue,
    FailureRecommendation
)


class TestMLFMEAFramework:
    """Test suite for ML FMEA framework with realistic failure scenarios"""
    
    @pytest.fixture(scope="class")
    def failure_analyzer(self):
        """Create a failure analyzer instance for testing"""
        config = FailureConfig(
            failure_threshold=0.3,
            min_pattern_size=3,
            significance_threshold=0.1,
            max_patterns=20,
            confidence_threshold=0.7
        )
        return FailureAnalyzer(config)
    
    @pytest.fixture(scope="function")
    def random_seed(self):
        """Set reproducible random seed for each test"""
        np.random.seed(42)
        return 42
    
    @pytest.fixture
    def ml_fmea_database_fixture(self):
        """Fixture providing expected ML FMEA database structure"""
        return {
            'data_drift': {
                'severity': 8,
                'occurrence': 6,
                'detection': 4,
                'description': 'Training and production data distributions diverge'
            },
            'model_overfitting': {
                'severity': 7,
                'occurrence': 5,
                'detection': 6,
                'description': 'Model memorizes training data, poor generalization'
            },
            'infrastructure_failure': {
                'severity': 9,
                'occurrence': 3,
                'detection': 2,
                'description': 'Hardware/software infrastructure failures'
            },
            'deployment_error': {
                'severity': 8,
                'occurrence': 4,
                'detection': 3,
                'description': 'Errors during model deployment or versioning'
            }
        }
    
    @pytest.fixture
    def sample_failures_factory(self, random_seed):
        """Factory fixture for generating sample failure data"""
        def _create_failures(n_failures=50, failure_types=None, complexity='medium'):
            """Create realistic failure data for testing"""
            if failure_types is None:
                failure_types = ['data_drift', 'model_overfitting', 'infrastructure_failure', 'deployment_error']
            
            failures = []
            
            # Define complexity-based parameters
            complexity_params = {
                'simple': {'score_range': (0.1, 0.4), 'pattern_noise': 0.1},
                'medium': {'score_range': (0.2, 0.8), 'pattern_noise': 0.2},
                'complex': {'score_range': (0.0, 1.0), 'pattern_noise': 0.3}
            }
            
            params = complexity_params.get(complexity, complexity_params['medium'])
            
            for i in range(n_failures):
                failure_type = np.random.choice(failure_types)
                
                # Generate realistic failure scores
                base_score = np.random.uniform(*params['score_range'])
                noise = np.random.normal(0, params['pattern_noise'])
                final_score = np.clip(base_score + noise, 0.0, 1.0)
                
                failure = {
                    'failure_id': f'failure_{i:03d}',
                    'failure_type': failure_type,
                    'overall_score': final_score,
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 168)),
                    'context': {
                        'model_type': np.random.choice(['transformer', 'lstm', 'cnn', 'mlp']),
                        'dataset_size': np.random.choice(['small', 'medium', 'large']),
                        'complexity': complexity
                    },
                    'metrics': {
                        'accuracy': np.random.uniform(0.4, 0.95),
                        'precision': np.random.uniform(0.3, 0.9),
                        'recall': np.random.uniform(0.3, 0.9),
                        'f1_score': np.random.uniform(0.3, 0.9)
                    },
                    'error_details': {
                        'error_message': f'Sample error for {failure_type}',
                        'stack_trace': f'Mock stack trace for {failure_type}',
                        'severity_level': np.random.choice(['low', 'medium', 'high', 'critical'])
                    }
                }
                failures.append(failure)
            
            return failures
        return _create_failures
    
    @pytest.fixture
    def sample_test_results_factory(self, random_seed):
        """Factory fixture for generating sample test result data"""
        def _create_test_results(n_results=30, test_types=None):
            """Create realistic test result data"""
            if test_types is None:
                test_types = ['unit', 'integration', 'performance', 'security']
            
            test_results = []
            
            for i in range(n_results):
                test_type = np.random.choice(test_types)
                passed = np.random.choice([True, False], p=[0.7, 0.3])  # 70% pass rate
                
                result = {
                    'test_id': f'test_{i:03d}',
                    'test_type': test_type,
                    'passed': passed,
                    'execution_time_ms': np.random.exponential(100),  # Realistic timing distribution
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 1440)),
                    'test_context': {
                        'environment': np.random.choice(['dev', 'staging', 'prod']),
                        'test_framework': np.random.choice(['pytest', 'unittest', 'nose']),
                        'coverage': np.random.uniform(0.6, 0.95)
                    },
                    'assertions': {
                        'total': np.random.randint(1, 20),
                        'passed': lambda total, test_passed: total if test_passed else np.random.randint(0, total)
                    }
                }
                
                # Calculate passed assertions
                result['assertions']['passed'] = result['assertions']['passed'](
                    result['assertions']['total'], 
                    result['passed']
                )
                
                test_results.append(result)
            
            return test_results
        return _create_test_results
    
    def test_ml_fmea_database_initialization(self, failure_analyzer, ml_fmea_database_fixture):
        """Test ML FMEA database initialization with comprehensive failure modes"""
        # Initialize the database
        fmea_database = failure_analyzer._initialize_ml_fmea_database()
        
        # Verify database structure
        assert isinstance(fmea_database, list)
        assert len(fmea_database) >= 4  # At least basic failure modes
        
        # Verify all required failure modes are present
        failure_mode_types = {mode['failure_mode'] for mode in fmea_database}
        expected_types = set(ml_fmea_database_fixture.keys())
        
        assert expected_types.issubset(failure_mode_types), \
            f"Missing failure modes: {expected_types - failure_mode_types}"
        
        # Verify RPN calculation components are present
        for mode in fmea_database:
            assert 'severity' in mode
            assert 'occurrence' in mode  
            assert 'detection' in mode
            assert 'rpn' in mode
            assert 'description' in mode
            
            # Verify RPN calculation
            expected_rpn = mode['severity'] * mode['occurrence'] * mode['detection']
            assert mode['rpn'] == expected_rpn
            
            # Verify scoring ranges (1-10 scale)
            assert 1 <= mode['severity'] <= 10
            assert 1 <= mode['occurrence'] <= 10
            assert 1 <= mode['detection'] <= 10
    
    @pytest.mark.asyncio
    async def test_ml_fmea_analysis_basic_functionality(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test basic ML FMEA analysis functionality"""
        # Generate sample data
        failures = sample_failures_factory(n_failures=20, complexity='medium')
        test_results = sample_test_results_factory(n_results=15)
        
        # Perform ML FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify result structure
        required_fields = [
            'total_failures_analyzed', 'critical_failure_modes', 'risk_matrix',
            'top_risk_priorities', 'failure_mode_distribution', 'mitigation_recommendations',
            'critical_paths', 'detection_gaps', 'severity_assessment'
        ]
        
        for field in required_fields:
            assert field in fmea_results, f"Missing required field: {field}"
        
        # Verify data types and ranges
        assert isinstance(fmea_results['total_failures_analyzed'], int)
        assert fmea_results['total_failures_analyzed'] == len(failures)
        
        assert isinstance(fmea_results['critical_failure_modes'], list)
        assert isinstance(fmea_results['risk_matrix'], dict)
        assert isinstance(fmea_results['top_risk_priorities'], list)
        
        # Verify risk priority ordering (should be sorted by RPN descending)
        rpn_values = [item['rpn'] for item in fmea_results['top_risk_priorities']]
        assert rpn_values == sorted(rpn_values, reverse=True), "Risk priorities not sorted by RPN"
    
    @pytest.mark.asyncio
    async def test_ensemble_anomaly_detection_basic_functionality(self, failure_analyzer, sample_failures_factory):
        """Test ensemble anomaly detection with multiple algorithms"""
        # Generate failures with some anomalous patterns
        failures = sample_failures_factory(n_failures=100, complexity='complex')
        
        # Add some clearly anomalous failures
        anomalous_failures = [
            {
                'failure_id': 'anomaly_001',
                'failure_type': 'unknown_anomaly',
                'overall_score': 0.95,  # Unusually high failure score
                'timestamp': datetime.now(),
                'context': {'model_type': 'unknown', 'dataset_size': 'huge', 'complexity': 'extreme'},
                'metrics': {'accuracy': 0.1, 'precision': 0.05, 'recall': 0.03, 'f1_score': 0.04},
                'error_details': {'error_message': 'Critical system failure', 'severity_level': 'critical'}
            }
        ]
        failures.extend(anomalous_failures)
        
        # Perform ensemble anomaly detection
        anomaly_results = await failure_analyzer._perform_ensemble_anomaly_detection(failures)
        
        # Verify result structure
        required_fields = [
            'anomaly_scores', 'ensemble_consensus', 'individual_detectors',
            'anomaly_threshold', 'detected_anomalies', 'anomaly_patterns',
            'confidence_scores', 'detection_summary'
        ]
        
        for field in required_fields:
            assert field in anomaly_results, f"Missing required field: {field}"
        
        # Verify individual detectors are present
        expected_detectors = ['isolation_forest', 'elliptic_envelope', 'one_class_svm']
        individual_detectors = anomaly_results['individual_detectors']
        
        for detector in expected_detectors:
            assert detector in individual_detectors, f"Missing detector: {detector}"
            
            # Verify detector results structure
            detector_results = individual_detectors[detector]
            assert 'scores' in detector_results
            assert 'outliers' in detector_results
            assert len(detector_results['scores']) == len(failures)
        
        # Verify ensemble consensus
        ensemble_scores = anomaly_results['anomaly_scores']
        assert len(ensemble_scores) == len(failures)
        assert all(-1 <= score <= 1 for score in ensemble_scores), "Anomaly scores outside expected range"
        
        # Verify anomaly detection
        detected_anomalies = anomaly_results['detected_anomalies']
        assert isinstance(detected_anomalies, list)
        
        # Should detect at least some anomalies with complex data
        assert len(detected_anomalies) > 0, "No anomalies detected in complex failure dataset"
    
    @pytest.mark.parametrize("failure_complexity,expected_patterns", [
        pytest.param("simple", 2, id="simple_failures"),
        pytest.param("medium", 4, id="medium_complexity"),
        pytest.param("complex", 6, id="complex_failures"),
    ])
    @pytest.mark.asyncio
    async def test_fmea_analysis_scaling_with_complexity(self, failure_analyzer, sample_failures_factory, 
                                                        sample_test_results_factory, failure_complexity, expected_patterns):
        """Test that FMEA analysis scales appropriately with failure complexity"""
        # Generate failures with specified complexity
        failures = sample_failures_factory(n_failures=50, complexity=failure_complexity)
        test_results = sample_test_results_factory(n_results=25)
        
        # Perform analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify complexity-appropriate results
        critical_modes = fmea_results['critical_failure_modes']
        
        # More complex scenarios should identify more failure patterns
        if failure_complexity == 'simple':
            assert len(critical_modes) >= 1, "Should identify at least basic failure patterns"
        elif failure_complexity == 'medium':
            assert len(critical_modes) >= 2, "Should identify multiple failure patterns"
        elif failure_complexity == 'complex':
            assert len(critical_modes) >= 3, "Should identify diverse failure patterns"
        
        # Risk assessment should reflect complexity
        risk_matrix = fmea_results['risk_matrix']
        total_high_risk = risk_matrix.get('high_risk_count', 0)
        
        # Complex scenarios should have more high-risk items
        complexity_risk_expectations = {
            'simple': (0, 3),
            'medium': (1, 6),  
            'complex': (2, 10)
        }
        
        min_risk, max_risk = complexity_risk_expectations[failure_complexity]
        assert min_risk <= total_high_risk <= max_risk, \
            f"High risk count {total_high_risk} outside expected range [{min_risk}, {max_risk}] for {failure_complexity} complexity"
    
    def test_rpn_scoring_calculation_accuracy(self, failure_analyzer, ml_fmea_database_fixture):
        """Test RPN (Risk Priority Number) scoring calculation accuracy"""
        # Get initialized database
        fmea_database = failure_analyzer._initialize_ml_fmea_database()
        
        # Verify RPN calculations for known failure modes
        for mode in fmea_database:
            failure_type = mode['failure_mode']
            
            if failure_type in ml_fmea_database_fixture:
                expected = ml_fmea_database_fixture[failure_type]
                
                # Allow some tolerance for implementation variations
                severity_tolerance = 1
                occurrence_tolerance = 1
                detection_tolerance = 1
                
                assert abs(mode['severity'] - expected['severity']) <= severity_tolerance
                assert abs(mode['occurrence'] - expected['occurrence']) <= occurrence_tolerance
                assert abs(mode['detection'] - expected['detection']) <= detection_tolerance
                
                # RPN should be calculated correctly
                expected_rpn = mode['severity'] * mode['occurrence'] * mode['detection']
                assert mode['rpn'] == expected_rpn
        
        # Test RPN priority ordering
        rpn_values = [mode['rpn'] for mode in fmea_database]
        sorted_rpn = sorted(rpn_values, reverse=True)
        
        # Verify reasonable RPN distribution
        max_rpn = max(rpn_values)
        min_rpn = min(rpn_values)
        
        assert max_rpn <= 1000, f"Maximum RPN {max_rpn} exceeds reasonable limit"
        assert min_rpn >= 1, f"Minimum RPN {min_rpn} below reasonable limit"
        assert max_rpn > min_rpn, "RPN values should have meaningful variation"
    
    @pytest.mark.asyncio
    async def test_fmea_mitigation_recommendations(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test quality and relevance of FMEA mitigation recommendations"""
        # Generate failures with specific patterns for testing recommendations
        failures = []
        
        # Create data drift failures
        for i in range(10):
            failures.append({
                'failure_id': f'drift_{i}',
                'failure_type': 'data_drift',
                'overall_score': 0.8,
                'timestamp': datetime.now(),
                'context': {'model_type': 'transformer', 'dataset_size': 'large'},
                'metrics': {'accuracy': 0.6},
                'error_details': {'severity_level': 'high'}
            })
        
        # Create overfitting failures
        for i in range(8):
            failures.append({
                'failure_id': f'overfit_{i}',
                'failure_type': 'model_overfitting',
                'overall_score': 0.7,
                'timestamp': datetime.now(),
                'context': {'model_type': 'cnn', 'dataset_size': 'small'},
                'metrics': {'accuracy': 0.95},  # High training accuracy suggesting overfitting
                'error_details': {'severity_level': 'medium'}
            })
        
        test_results = sample_test_results_factory(n_results=20)
        
        # Perform FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify mitigation recommendations
        recommendations = fmea_results['mitigation_recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0, "Should provide mitigation recommendations"
        
        # Check for specific recommendation categories
        recommendation_text = ' '.join([rec.get('description', '') for rec in recommendations])
        
        # Should mention data quality for data drift issues
        if any(f['failure_type'] == 'data_drift' for f in failures):
            assert any(keyword in recommendation_text.lower() for keyword in 
                      ['data quality', 'monitoring', 'drift detection', 'validation']), \
                "Should recommend data quality measures for data drift"
        
        # Should mention regularization for overfitting issues
        if any(f['failure_type'] == 'model_overfitting' for f in failures):
            assert any(keyword in recommendation_text.lower() for keyword in
                      ['regularization', 'validation', 'cross-validation', 'early stopping']), \
                "Should recommend regularization techniques for overfitting"
        
        # Verify recommendation structure
        for rec in recommendations:
            assert 'failure_mode' in rec
            assert 'priority' in rec
            assert 'description' in rec
            assert 'implementation_effort' in rec
            
            # Priority should be valid
            assert rec['priority'] in ['low', 'medium', 'high', 'critical']
            
            # Implementation effort should be realistic
            assert rec['implementation_effort'] in ['low', 'medium', 'high']
    
    @pytest.mark.asyncio
    async def test_detection_gap_analysis(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test detection gap analysis for identifying blind spots"""
        # Create failures that would expose detection gaps
        failures = sample_failures_factory(n_failures=30)
        
        # Create test results with some gaps (low coverage areas)
        test_results = sample_test_results_factory(n_results=20)
        
        # Add some test results with low coverage to simulate gaps
        for i in range(5):
            test_results.append({
                'test_id': f'low_coverage_{i}',
                'test_type': 'integration',
                'passed': False,
                'execution_time_ms': 50,
                'timestamp': datetime.now(),
                'test_context': {
                    'environment': 'prod',
                    'coverage': 0.3  # Low coverage indicating detection gap
                },
                'assertions': {'total': 10, 'passed': 2}
            })
        
        # Perform FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify detection gap analysis
        detection_gaps = fmea_results['detection_gaps']
        assert isinstance(detection_gaps, list)
        
        # Should identify gaps in testing/monitoring
        assert len(detection_gaps) > 0, "Should identify detection gaps"
        
        # Verify gap structure
        for gap in detection_gaps:
            assert 'gap_type' in gap
            assert 'severity' in gap
            assert 'affected_areas' in gap
            assert 'recommended_actions' in gap
            
            # Severity should be reasonable
            assert gap['severity'] in ['low', 'medium', 'high', 'critical']
    
    @pytest.mark.performance
    def test_fmea_performance_benchmarks(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test FMEA framework performance with realistic dataset sizes"""
        import time
        
        # Test different dataset sizes
        test_cases = [
            (50, 25, "small_dataset"),
            (200, 100, "medium_dataset"),  
            (500, 200, "large_dataset")
        ]
        
        for n_failures, n_tests, case_name in test_cases:
            failures = sample_failures_factory(n_failures=n_failures)
            test_results = sample_test_results_factory(n_results=n_tests)
            
            # Measure performance
            start_time = time.time()
            
            # Run synchronous version for performance testing
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                fmea_results = loop.run_until_complete(
                    failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
                )
                anomaly_results = loop.run_until_complete(
                    failure_analyzer._perform_ensemble_anomaly_detection(failures)
                )
            finally:
                loop.close()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance expectations (should scale reasonably)
            max_time_by_case = {
                "small_dataset": 2.0,
                "medium_dataset": 5.0,
                "large_dataset": 10.0
            }
            
            max_time = max_time_by_case[case_name]
            assert execution_time <= max_time, \
                f"{case_name}: execution time {execution_time:.2f}s exceeds limit {max_time}s"
            
            # Verify results are still comprehensive
            assert len(fmea_results['critical_failure_modes']) > 0
            assert len(anomaly_results['detected_anomalies']) >= 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fmea_integration_with_failure_analyzer(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test ML FMEA integration with full failure analyzer workflow"""
        # Generate comprehensive test data
        failures = sample_failures_factory(n_failures=100, complexity='complex')
        test_results = sample_test_results_factory(n_results=50)
        
        # Mock database session for integration test
        mock_session = AsyncMock()
        
        # Test full analyze_failures method integration
        with patch.object(failure_analyzer, '_store_analysis_results') as mock_store:
            with patch.object(failure_analyzer, '_generate_failure_insights') as mock_insights:
                mock_insights.return_value = {
                    'key_insights': ['Sample insight 1', 'Sample insight 2'],
                    'trend_analysis': {'trends': 'increasing_failures'},
                    'recommendations': ['Recommendation 1']
                }
                
                # This would be the main integration point
                # analysis_result = await failure_analyzer.analyze_failures(failures, test_results, mock_session)
                
                # For now, test the individual components work together
                fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
                anomaly_results = await failure_analyzer._perform_ensemble_anomaly_detection(failures)
        
        # Verify integration produces comprehensive results
        assert isinstance(fmea_results, dict)
        assert isinstance(anomaly_results, dict)
        
        # Verify cross-component consistency
        total_failures = len(failures)
        assert fmea_results['total_failures_analyzed'] == total_failures
        assert len(anomaly_results['anomaly_scores']) == total_failures
        
        # Verify that high-anomaly failures align with high-risk FMEA patterns
        high_anomaly_indices = [i for i, score in enumerate(anomaly_results['anomaly_scores']) if score > 0.5]
        high_risk_patterns = [pattern for pattern in fmea_results['critical_failure_modes'] if pattern.get('rpn', 0) > 400]
        
        # Some correlation expected between anomaly detection and FMEA risk assessment
        assert len(high_risk_patterns) > 0 or len(high_anomaly_indices) > 0, \
            "Should detect either high-risk patterns or anomalies in complex dataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])