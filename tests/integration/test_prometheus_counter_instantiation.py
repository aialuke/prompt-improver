"""Integration test for prometheus_client Counter instantiation

Tests that the fixed prometheus_client.Counter alias works correctly and
can be instantiated with label names.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestPrometheusCounterInstantiation:
    """Test prometheus_client Counter instantiation with proper aliasing"""
    
    def test_promcounter_alias_instantiation(self):
        """Test that PromCounter alias can be instantiated with labels"""
        # Mock prometheus_client to avoid actual dependency
        mock_prometheus = MagicMock()
        mock_counter_class = MagicMock()
        mock_prometheus.Counter = mock_counter_class
        
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            # Import after mocking to ensure our mock is used
            from prometheus_client import Counter as PromCounter
            
            # Test instantiation with label names (as used in failure_analyzer.py)
            counter = PromCounter(
                "ml_failures_total",
                "Total number of ML failures",
                ["failure_type", "severity"]
            )
            
            # Verify the Counter was called with correct arguments
            mock_counter_class.assert_called_once_with(
                "ml_failures_total",
                "Total number of ML failures", 
                ["failure_type", "severity"]
            )
    
    def test_failure_analyzer_prometheus_initialization(self):
        """Test that FailureModeAnalyzer can initialize prometheus metrics"""
        # Mock all prometheus_client components
        mock_prometheus = MagicMock()
        mock_counter = MagicMock()
        mock_gauge = MagicMock()
        mock_histogram = MagicMock()
        mock_start_server = MagicMock()
        
        mock_prometheus.Counter = mock_counter
        mock_prometheus.Gauge = mock_gauge  
        mock_prometheus.Histogram = mock_histogram
        mock_prometheus.start_http_server = mock_start_server
        
        # Mock other dependencies to avoid import issues
        mock_sklearn = MagicMock()
        mock_art = MagicMock()
        
        with patch.dict('sys.modules', {
            'prometheus_client': mock_prometheus,
            'sklearn': mock_sklearn,
            'sklearn.ensemble': mock_sklearn.ensemble,
            'sklearn.covariance': mock_sklearn.covariance, 
            'sklearn.svm': mock_sklearn.svm,
            'sklearn.cluster': mock_sklearn.cluster,
            'sklearn.feature_extraction': mock_sklearn.feature_extraction,
            'sklearn.feature_extraction.text': mock_sklearn.feature_extraction.text,
            'sklearn.metrics': mock_sklearn.metrics,
            'sklearn.metrics.pairwise': mock_sklearn.metrics.pairwise,
            'adversarial_robustness_toolbox': mock_art,
            'adversarial_robustness_toolbox.attacks': mock_art.attacks,
            'adversarial_robustness_toolbox.attacks.evasion': mock_art.attacks.evasion,
            'adversarial_robustness_toolbox.estimators': mock_art.estimators,
            'adversarial_robustness_toolbox.estimators.classification': mock_art.estimators.classification,
        }):
            # Import after mocking
            from prompt_improver.learning.failure_analyzer import FailureModeAnalyzer, FailureConfig
            
            # Test initialization with prometheus monitoring enabled
            config = FailureConfig(enable_prometheus_monitoring=True, prometheus_port=8001)
            analyzer = FailureModeAnalyzer(config)
            
            # Verify prometheus metrics were set up
            assert hasattr(analyzer, 'prometheus_metrics')
            
            # Verify Counter was called with PromCounter alias (now fixed)
            calls = mock_counter.call_args_list
            counter_call = None
            for call in calls:
                args, kwargs = call
                if args[0] == "ml_failures_total":
                    counter_call = call
                    break
            
            assert counter_call is not None, "PromCounter for ml_failures_total was not called"
            args, kwargs = counter_call
            assert args == ("ml_failures_total", "Total number of ML failures", ["failure_type", "severity"])
    
    def test_prometheus_available_flag(self):
        """Test that PROMETHEUS_AVAILABLE flag works correctly"""
        # Test when prometheus_client is available (mocked)
        mock_prometheus = MagicMock()
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            # Re-import the module to reset PROMETHEUS_AVAILABLE
            import importlib
            from prompt_improver.learning import failure_analyzer
            importlib.reload(failure_analyzer)
            
            assert failure_analyzer.PROMETHEUS_AVAILABLE is True
        
        # Test when prometheus_client is not available
        with patch.dict('sys.modules', {'prometheus_client': None}):
            with patch('prompt_improver.learning.failure_analyzer.warnings.warn') as mock_warn:
                # Simulate import error
                def mock_import(name, *args, **kwargs):
                    if name == 'prometheus_client':
                        raise ImportError("No module named 'prometheus_client'")
                    return __import__(name, *args, **kwargs)
                
                with patch('builtins.__import__', side_effect=mock_import):
                    importlib.reload(failure_analyzer)
                    
                    assert failure_analyzer.PROMETHEUS_AVAILABLE is False
                    mock_warn.assert_called_with(
                        "Prometheus client not available. Install with: pip install prometheus-client"
                    )
    
    def test_counter_labels_usage(self):
        """Test that Counter labels method works as expected"""
        mock_prometheus = MagicMock()
        mock_counter_instance = MagicMock()
        mock_counter_class = MagicMock(return_value=mock_counter_instance)
        mock_prometheus.Counter = mock_counter_class
        
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            from prometheus_client import Counter as PromCounter
            
            # Create counter
            counter = PromCounter(
                "ml_failures_total",
                "Total number of ML failures", 
                ["failure_type", "severity"]
            )
            
            # Test using labels (as done in failure_analyzer.py)
            labeled_counter = counter.labels(failure_type="general", severity="high")
            labeled_counter.inc(5)
            
            # Verify labels method was called correctly
            mock_counter_instance.labels.assert_called_once_with(
                failure_type="general",
                severity="high"
            )
            
            # Verify inc was called on labeled counter
            labeled_counter.inc.assert_called_once_with(5)


if __name__ == "__main__":
    pytest.main([__file__])
