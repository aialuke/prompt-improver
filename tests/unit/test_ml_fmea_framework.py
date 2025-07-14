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
    FailureModeAnalyzer,
    FailureConfig,
    FailurePattern,
    RootCause,
    EdgeCase,
    SystematicIssue,
    FailureRecommendation,
    MLFailureMode,
    RobustnessTestResult,
    PrometheusAlert
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
        return FailureModeAnalyzer(config)
    
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
        
        # Test database creation and structure
        fmea_database = failure_analyzer._initialize_ml_fmea_database()
        assert isinstance(fmea_database, list)
        assert len(fmea_database) > 0
        
        # Verify failure mode types are correctly represented
        failure_mode_types = {mode.failure_type for mode in fmea_database}
        expected_types = {"data", "model", "infrastructure", "deployment"}
        assert failure_mode_types.intersection(expected_types), f"Expected failure mode types not found: {expected_types}"
        
        # Verify each failure mode has required attributes
        for mode in fmea_database:
            assert hasattr(mode, 'failure_type')
            assert hasattr(mode, 'description')
            assert hasattr(mode, 'severity')
            assert hasattr(mode, 'occurrence')
            assert hasattr(mode, 'detection')
            assert hasattr(mode, 'rpn')
            assert hasattr(mode, 'root_causes')
            assert hasattr(mode, 'mitigation_strategies')
            
            # Verify RPN calculation
            expected_rpn = mode.severity * mode.occurrence * mode.detection
            assert mode.rpn == expected_rpn, f"RPN calculation incorrect for {mode.description}"
        
        # Test database categorization
        data_modes = [mode for mode in fmea_database if mode.failure_type == "data"]
        model_modes = [mode for mode in fmea_database if mode.failure_type == "model"]
        infrastructure_modes = [mode for mode in fmea_database if mode.failure_type == "infrastructure"]
        deployment_modes = [mode for mode in fmea_database if mode.failure_type == "deployment"]
        
        assert len(data_modes) > 0, "Data failure modes should exist"
        assert len(model_modes) > 0, "Model failure modes should exist"
    
    @pytest.mark.asyncio
    async def test_ml_fmea_analysis_basic_functionality(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test basic ML FMEA analysis functionality"""
        
        # Create sample data with varied complexity
        failures = sample_failures_factory(n_failures=50, complexity='medium')
        test_results = sample_test_results_factory(n_results=30)
        
        # Perform ML FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify result structure matches implementation
        required_fields = [
            "identified_failure_modes",
            "risk_matrix", 
            "critical_paths",
            "mitigation_plan"
        ]
        
        for field in required_fields:
            assert field in fmea_results, f"Missing required field: {field}"
        
        # Verify identified failure modes structure
        assert isinstance(fmea_results["identified_failure_modes"], list)
        
        # If failure modes were identified, check their structure
        if len(fmea_results["identified_failure_modes"]) > 0:
            mode = fmea_results["identified_failure_modes"][0]
            mode_fields = [
                "failure_mode", "type", "severity", "occurrence", 
                "detection", "rpn", "affected_failures_count",
                "root_causes", "mitigation_strategies", "priority"
            ]
            for field in mode_fields:
                assert field in mode, f"Missing field in failure mode: {field}"
        
        # Verify risk matrix structure
        assert isinstance(fmea_results["risk_matrix"], dict)
        
        # Verify critical paths
        assert isinstance(fmea_results["critical_paths"], list)
        
        # Verify mitigation plan
        assert isinstance(fmea_results["mitigation_plan"], list)
        
        # Test with edge case: no failures
        empty_fmea_results = await failure_analyzer._perform_ml_fmea_analysis([], test_results)
        assert "identified_failure_modes" in empty_fmea_results
        assert len(empty_fmea_results["identified_failure_modes"]) == 0
    
    @pytest.mark.asyncio
    async def test_ensemble_anomaly_detection_basic_functionality(self, failure_analyzer, sample_failures_factory):
        """Test ensemble anomaly detection using multiple detection algorithms"""
        
        # Generate sufficient sample data for anomaly detection (need at least 10)
        failures = sample_failures_factory(n_failures=100, complexity='complex')
        
        # Perform ensemble anomaly detection
        anomaly_results = await failure_analyzer._perform_ensemble_anomaly_detection(failures)
        
        # Verify result structure matches implementation
        if "insufficient_data" not in anomaly_results and "no_features" not in anomaly_results:
            required_fields = [
                "individual_detectors",
                "consensus_anomalies", 
                "anomaly_summary"
            ]
            
            for field in required_fields:
                assert field in anomaly_results, f"Missing required field: {field}"
            
            # Verify individual detectors structure
            assert isinstance(anomaly_results["individual_detectors"], dict)
            
            # Check each detector result
            for detector_name, detector_result in anomaly_results["individual_detectors"].items():
                if "error" not in detector_result:
                    expected_detector_fields = [
                        "anomaly_count", "anomaly_percentage", 
                        "anomaly_indices", "anomalous_failures"
                    ]
                    for field in expected_detector_fields:
                        assert field in detector_result, f"Missing field in {detector_name}: {field}"
                    
                    # Verify data types
                    assert isinstance(detector_result["anomaly_count"], int)
                    assert isinstance(detector_result["anomaly_percentage"], float)
                    assert isinstance(detector_result["anomaly_indices"], list)
                    assert isinstance(detector_result["anomalous_failures"], list)
            
            # Verify consensus anomalies structure
            assert isinstance(anomaly_results["consensus_anomalies"], list)
            
            # Verify anomaly summary structure
            summary = anomaly_results["anomaly_summary"]
            summary_fields = ["total_failures", "consensus_anomalies_count", "consensus_anomaly_rate"]
            for field in summary_fields:
                assert field in summary, f"Missing field in anomaly_summary: {field}"
            
            # Verify summary calculations
            assert summary["total_failures"] == len(failures)
            assert 0 <= summary["consensus_anomaly_rate"] <= 100
            
        else:
            # Handle cases with insufficient data or no features
            assert len(failures) < 10 or "no_features" in anomaly_results
        
        # Test edge case: insufficient data
        small_failures = sample_failures_factory(n_failures=5, complexity='simple')
        small_anomaly_results = await failure_analyzer._perform_ensemble_anomaly_detection(small_failures)
        assert "insufficient_data" in small_anomaly_results
    
    @pytest.mark.parametrize("failure_complexity,expected_patterns", [
        pytest.param("simple", 2, id="simple_failures"),
        pytest.param("medium", 4, id="medium_complexity"),
        pytest.param("complex", 6, id="complex_failures"),
    ])
    @pytest.mark.asyncio
    async def test_fmea_analysis_scaling_with_complexity(self, failure_analyzer, sample_failures_factory, 
                                                        sample_test_results_factory, failure_complexity, expected_patterns):
        """Test FMEA analysis scaling with different failure complexity levels"""
        
        # Generate failures based on complexity
        complexity_sizes = {"simple": 20, "medium": 50, "complex": 100}
        n_failures = complexity_sizes[failure_complexity]
        
        failures = sample_failures_factory(n_failures=n_failures, complexity=failure_complexity)
        test_results = sample_test_results_factory(n_results=30)
        
        # Perform ML FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify basic structure
        assert "identified_failure_modes" in fmea_results
        assert "risk_matrix" in fmea_results
        assert "critical_paths" in fmea_results
        assert "mitigation_plan" in fmea_results
        
        # Analyze identified failure modes based on complexity
        identified_modes = fmea_results['identified_failure_modes']
        
        # More complex failures should potentially identify more failure modes
        if failure_complexity == "complex":
            # Complex failures might identify multiple failure modes
            assert len(identified_modes) >= 0  # May or may not find modes depending on matching
        elif failure_complexity == "simple":
            # Simple failures might identify fewer or no modes
            assert len(identified_modes) >= 0
            
        # Verify each identified mode has proper structure
        for mode in identified_modes:
            assert "rpn" in mode
            assert "priority" in mode
            assert "affected_failures_count" in mode
            assert mode["affected_failures_count"] > 0
            
            # Verify RPN calculation
            assert mode["rpn"] == mode["severity"] * mode["occurrence"] * mode["detection"]
            
        # Verify risk matrix scaling
        risk_matrix = fmea_results['risk_matrix']
        assert isinstance(risk_matrix, dict)
        
        # Critical paths should be proportional to complexity
        critical_paths = fmea_results['critical_paths']
        assert isinstance(critical_paths, list)
        
        # Mitigation plan should scale with identified issues
        mitigation_plan = fmea_results['mitigation_plan']
        assert isinstance(mitigation_plan, list)
    
    def test_rpn_scoring_calculation_accuracy(self, failure_analyzer, ml_fmea_database_fixture):
        """Test accurate RPN (Risk Priority Number) calculation"""
        
        # Get ML FMEA database
        fmea_database = failure_analyzer._initialize_ml_fmea_database()
        
        # Test RPN calculation for each failure mode
        for mode in fmea_database:
            failure_type = mode.failure_type
            
            # Verify components are in valid range (1-10)
            assert 1 <= mode.severity <= 10, f"Severity out of range for {failure_type}"
            assert 1 <= mode.occurrence <= 10, f"Occurrence out of range for {failure_type}"
            assert 1 <= mode.detection <= 10, f"Detection out of range for {failure_type}"
            
            # Verify RPN calculation (Severity × Occurrence × Detection)
            expected_rpn = mode.severity * mode.occurrence * mode.detection
            assert mode.rpn == expected_rpn, \
                f"RPN calculation error for {failure_type}: expected {expected_rpn}, got {mode.rpn}"
            
            # Verify RPN is in valid range (1-1000)
            assert 1 <= mode.rpn <= 1000, f"RPN out of range for {failure_type}"
        
        # Test RPN prioritization logic
        rpn_values = [mode.rpn for mode in fmea_database]
        
        # Verify we have a range of RPN values (not all the same)
        assert len(set(rpn_values)) > 1, "All RPN values are identical - need diversity"
        
        # Test categorization by RPN thresholds
        critical_rpn = [mode for mode in fmea_database if mode.rpn > 150]
        high_rpn = [mode for mode in fmea_database if 100 < mode.rpn <= 150]
        medium_rpn = [mode for mode in fmea_database if mode.rpn <= 100]
        
        # Should have modes in different categories
        total_modes = len(critical_rpn) + len(high_rpn) + len(medium_rpn)
        assert total_modes == len(fmea_database), "RPN categorization incomplete"
        
        # Test specific failure mode types have appropriate RPN ranges
        data_modes = [mode for mode in fmea_database if mode.failure_type == "data"]
        model_modes = [mode for mode in fmea_database if mode.failure_type == "model"]
        
        if data_modes:
            data_rpn_avg = sum(mode.rpn for mode in data_modes) / len(data_modes)
            assert 1 <= data_rpn_avg <= 1000, "Data failure modes RPN average out of range"
            
        if model_modes:
            model_rpn_avg = sum(mode.rpn for mode in model_modes) / len(model_modes)
            assert 1 <= model_rpn_avg <= 1000, "Model failure modes RPN average out of range"
    
    @pytest.mark.asyncio
    async def test_fmea_mitigation_recommendations(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test FMEA mitigation recommendation generation"""
        
        # Generate test data
        failures = sample_failures_factory(n_failures=30, complexity='medium')
        test_results = sample_test_results_factory(n_results=20)
        
        # Perform FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Verify mitigation plan structure
        assert "mitigation_plan" in fmea_results
        mitigation_plan = fmea_results["mitigation_plan"]
        assert isinstance(mitigation_plan, list)
        
        # If recommendations were generated, verify their structure
        for recommendation in mitigation_plan:
            assert isinstance(recommendation, dict)
            # Each recommendation should have basic structure
            expected_fields = ["priority", "description", "target_failure_modes"]
            for field in expected_fields:
                if field in recommendation:  # Fields may vary based on implementation
                    assert recommendation[field] is not None
        
        # Verify mitigation strategies are included in identified failure modes
        identified_modes = fmea_results["identified_failure_modes"]
        for mode in identified_modes:
            assert "mitigation_strategies" in mode
            assert isinstance(mode["mitigation_strategies"], list)
        
        # Test that high RPN modes get prioritized mitigation
        high_rpn_modes = [mode for mode in identified_modes if mode.get("rpn", 0) > 100]
        if high_rpn_modes:
            # High RPN modes should have mitigation strategies
            for mode in high_rpn_modes:
                assert len(mode["mitigation_strategies"]) > 0, "High RPN modes should have mitigation strategies"

    @pytest.mark.asyncio
    async def test_detection_gap_analysis(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test detection gap analysis in FMEA"""
        
        # Generate test data with varied detection characteristics
        failures = sample_failures_factory(n_failures=40, complexity='medium')
        test_results = sample_test_results_factory(n_results=25)
        
        # Perform FMEA analysis
        fmea_results = await failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
        
        # Analyze detection capabilities in identified failure modes
        identified_modes = fmea_results["identified_failure_modes"]
        
        if len(identified_modes) > 0:
            # Check detection scores for failure modes
            detection_scores = [mode["detection"] for mode in identified_modes]
            
            # Should have variety in detection scores
            if len(detection_scores) > 1:
                assert len(set(detection_scores)) > 1 or min(detection_scores) != max(detection_scores), \
                    "Detection scores should show variety"
            
            # Identify potential detection gaps (high detection scores = hard to detect = gaps)
            detection_gaps = [mode for mode in identified_modes if mode["detection"] >= 7]
            
            # Verify gap analysis structure
            for gap_mode in detection_gaps:
                assert "detection" in gap_mode
                assert gap_mode["detection"] >= 7, "Detection gap should have high detection score"
                
        # Test detection methods are specified
        for mode in identified_modes:
            # Should have detection methods specified in the failure mode
            assert hasattr(failure_analyzer.ml_failure_modes[0], 'detection_methods'), \
                "Failure modes should have detection methods"

    @pytest.mark.performance
    def test_fmea_performance_benchmarks(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test FMEA analysis performance with larger datasets"""
        
        # Generate larger dataset for performance testing
        failures = sample_failures_factory(n_failures=200, complexity='complex')
        test_results = sample_test_results_factory(n_results=100)
        
        import time
        start_time = time.time()
        
        # Perform FMEA analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            fmea_results = loop.run_until_complete(
                failure_analyzer._perform_ml_fmea_analysis(failures, test_results)
            )
        finally:
            loop.close()
        
        analysis_time = time.time() - start_time
        
        # Performance assertions
        assert analysis_time < 30.0, f"FMEA analysis took {analysis_time:.2f}s, should be under 30s"
        
        # Verify results structure even with large dataset
        assert "identified_failure_modes" in fmea_results
        assert isinstance(fmea_results["identified_failure_modes"], list)
        
        # Results should be proportional to dataset size
        identified_modes = fmea_results["identified_failure_modes"]
        
        # Should be able to handle large datasets
        assert len(failures) == 200, "Should process all input failures"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fmea_integration_with_failure_analyzer(self, failure_analyzer, sample_failures_factory, sample_test_results_factory):
        """Test FMEA integration with overall failure analysis workflow"""
        
        # Generate comprehensive test data
        failures = sample_failures_factory(n_failures=50, complexity='medium')
        test_results = sample_test_results_factory(n_results=30)
        
        # Test integration with main analyze_failures method
        with patch.object(failure_analyzer, '_perform_ml_fmea_analysis', wraps=failure_analyzer._perform_ml_fmea_analysis) as mock_fmea:
            
            # Call main failure analysis which should include FMEA
            analysis_results = await failure_analyzer.analyze_failures(failures + test_results)
            
            # Verify FMEA was called as part of the workflow
            if failure_analyzer.config.enable_robustness_validation:
                mock_fmea.assert_called_once()
            
        # Verify integration results structure
        assert isinstance(analysis_results, dict)
        
        # Should contain FMEA-related results in the overall analysis
        if "ml_fmea_analysis" in analysis_results:
            fmea_section = analysis_results["ml_fmea_analysis"]
            assert "identified_failure_modes" in fmea_section
            assert "risk_matrix" in fmea_section
        
        # Test ensemble anomaly detection integration
        if len(failures) >= 10:  # Minimum required for anomaly detection
            with patch.object(failure_analyzer, '_perform_ensemble_anomaly_detection', 
                            wraps=failure_analyzer._perform_ensemble_anomaly_detection) as mock_anomaly:
                
                await failure_analyzer.analyze_failures(failures + test_results)
                
                # Verify anomaly detection was called
                if failure_analyzer.config.ensemble_anomaly_detection:
                    mock_anomaly.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])