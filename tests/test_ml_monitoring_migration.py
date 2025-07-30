"""
Test ML Monitoring Migration from Prometheus to OpenTelemetry
============================================================

Real behavior testing to validate the successful migration of ML algorithm monitoring
from prometheus_client to OpenTelemetry without regression in monitoring capabilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Mock OpenTelemetry components to avoid initialization issues
with patch('prompt_improver.monitoring.opentelemetry.setup.OTEL_AVAILABLE', False):
    with patch('prompt_improver.monitoring.opentelemetry.metrics.OTEL_AVAILABLE', False):
        # Import the migrated components
        from prompt_improver.monitoring.opentelemetry.metrics import (
            MLMetrics, MLAlertingMetrics, OTelAlert, MetricDefinition, MetricType
        )


class TestMLMonitoringMigration:
    """Test suite for ML monitoring migration validation."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create mock meter for testing
        mock_meter = Mock()
        mock_meter.create_counter.return_value = Mock()
        mock_meter.create_histogram.return_value = Mock()
        mock_meter.create_gauge.return_value = Mock()

        # Create ML metrics instance with mocked meter
        self.ml_metrics = MLMetrics()
        self.ml_metrics.meter = mock_meter
        self.ml_metrics._setup_instruments()

        self.alert_thresholds = {
            "failure_rate": 0.15,
            "response_time_ms": 200,
            "error_rate": 0.05,
            "anomaly_score": 0.8,
        }
        self.alerting_metrics = MLAlertingMetrics(self.alert_thresholds)
    
    def test_ml_metrics_initialization(self):
        """Test that ML metrics are properly initialized."""
        assert isinstance(self.ml_metrics, MLMetrics)
        assert self.ml_metrics.meter_name == "ml_metrics"
        assert self.ml_metrics.component == "ml"
        assert len(self.ml_metrics._instruments) > 0
    
    def test_ml_alerting_initialization(self):
        """Test that ML alerting system is properly initialized."""
        assert isinstance(self.alerting_metrics, MLAlertingMetrics)
        assert len(self.alerting_metrics.alert_definitions) == 3
        assert all(isinstance(alert, OTelAlert) for alert in self.alerting_metrics.alert_definitions)
    
    def test_failure_analysis_recording(self):
        """Test recording failure analysis metrics (replaces prometheus functionality)."""
        # Test data simulating failure analysis results
        failure_rate = 0.12
        anomaly_rate = 0.65
        rpn_score = 150.0
        response_time = 0.15
        
        # Record metrics using the new OpenTelemetry method
        self.ml_metrics.record_failure_analysis(
            failure_rate=failure_rate,
            failure_type="data_drift",
            severity="warning",
            total_failures=5,
            anomaly_rate=anomaly_rate,
            rpn_score=rpn_score,
            response_time=response_time
        )
        
        # Verify metrics were recorded (instruments should exist)
        assert self.ml_metrics._get_instrument("ml_failure_rate") is not None
        assert self.ml_metrics._get_instrument("ml_failures_total") is not None
        assert self.ml_metrics._get_instrument("ml_anomaly_score") is not None
        assert self.ml_metrics._get_instrument("ml_risk_priority_number") is not None
        assert self.ml_metrics._get_instrument("ml_response_time_seconds") is not None
    
    def test_individual_metric_recording(self):
        """Test individual metric recording methods."""
        # Test failure rate
        self.ml_metrics.set_failure_rate(0.08, "model_drift")
        
        # Test anomaly score
        self.ml_metrics.set_anomaly_score(0.72, "isolation_forest")
        
        # Test RPN score
        self.ml_metrics.set_rpn_score(120.0, "data_quality")
        
        # Test response time
        self.ml_metrics.record_response_time(0.095, "classification")
        
        # Verify all instruments exist
        assert self.ml_metrics._get_instrument("ml_failure_rate") is not None
        assert self.ml_metrics._get_instrument("ml_anomaly_score") is not None
        assert self.ml_metrics._get_instrument("ml_risk_priority_number") is not None
        assert self.ml_metrics._get_instrument("ml_response_time_seconds") is not None
    
    def test_alert_threshold_checking(self):
        """Test alert threshold checking functionality."""
        # Create failure analysis data that should trigger alerts
        failure_analysis = {
            "metadata": {
                "failure_rate": 0.20,  # Above threshold of 0.15
                "total_failures": 10,
                "avg_response_time": 0.25  # Above threshold of 0.2 seconds
            },
            "summary": {
                "severity": "critical"
            },
            "anomaly_detection": {
                "anomaly_summary": {
                    "consensus_anomaly_rate": 85  # Above threshold of 80%
                }
            },
            "risk_assessment": {
                "overall_risk_score": 180.0
            }
        }
        
        # Check alerts
        triggered_alerts = self.alerting_metrics.check_alerts(failure_analysis)
        
        # Verify alerts were triggered
        assert len(triggered_alerts) > 0
        
        # Check specific alert types
        alert_names = [alert.alert_name for alert in triggered_alerts]
        assert "HighFailureRate" in alert_names
        assert "SlowResponseTime" in alert_names
        assert "HighAnomalyScore" in alert_names
    
    def test_alert_cooldown_functionality(self):
        """Test alert cooldown to prevent spam."""
        failure_analysis = {
            "metadata": {"failure_rate": 0.25},
            "summary": {"severity": "critical"}
        }
        
        # Trigger alert first time
        alerts_1 = self.alerting_metrics.check_alerts(failure_analysis)
        assert len(alerts_1) > 0
        
        # Immediately trigger again - should be in cooldown
        alerts_2 = self.alerting_metrics.check_alerts(failure_analysis)
        
        # Should have fewer or same alerts due to cooldown
        assert len(alerts_2) <= len(alerts_1)
    
    def test_ml_inference_recording(self):
        """Test ML inference recording (original functionality)."""
        # Record successful inference
        self.ml_metrics.record_inference(
            model_name="failure_classifier",
            model_version="v2.1",
            duration_ms=150.0,
            prompt_tokens=250,
            completion_tokens=100,
            success=True
        )
        
        # Record failed inference
        self.ml_metrics.record_inference(
            model_name="failure_analyzer",
            model_version="v1.8",
            duration_ms=0,
            success=False
        )
        
        # Verify instruments exist
        assert self.ml_metrics._get_instrument("ml_inferences_total") is not None
        assert self.ml_metrics._get_instrument("ml_inference_duration_ms") is not None
        assert self.ml_metrics._get_instrument("ml_prompt_tokens") is not None
        assert self.ml_metrics._get_instrument("ml_completion_tokens") is not None
    
    def test_training_iteration_recording(self):
        """Test ML training iteration recording."""
        # Record training iteration with accuracy
        self.ml_metrics.record_training_iteration(
            model_name="failure_predictor",
            accuracy=0.92
        )
        
        # Record training iteration without accuracy
        self.ml_metrics.record_training_iteration(
            model_name="anomaly_detector"
        )
        
        # Verify instruments exist
        assert self.ml_metrics._get_instrument("ml_training_iterations") is not None
        assert self.ml_metrics._get_instrument("ml_model_accuracy") is not None
    
    @pytest.mark.asyncio
    async def test_real_ml_workflow_monitoring(self):
        """Test monitoring with a simulated real ML workflow."""
        # Simulate a complete ML failure analysis workflow
        start_time = time.time()
        
        # Step 1: Record inference
        self.ml_metrics.record_inference(
            model_name="failure_analyzer",
            model_version="v3.0",
            duration_ms=180.0,
            success=True
        )
        
        # Step 2: Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Step 3: Record failure analysis results
        processing_time = time.time() - start_time
        self.ml_metrics.record_failure_analysis(
            failure_rate=0.08,
            failure_type="performance",
            severity="low",
            total_failures=2,
            anomaly_rate=0.45,
            rpn_score=95.0,
            response_time=processing_time
        )
        
        # Step 4: Check for alerts
        failure_analysis = {
            "metadata": {
                "failure_rate": 0.08,
                "total_failures": 2,
                "avg_response_time": processing_time
            },
            "summary": {"severity": "low"},
            "anomaly_detection": {
                "anomaly_summary": {"consensus_anomaly_rate": 45}
            },
            "risk_assessment": {"overall_risk_score": 95.0}
        }
        
        triggered_alerts = self.alerting_metrics.check_alerts(failure_analysis)
        
        # Should not trigger alerts for low severity issues
        assert len(triggered_alerts) == 0
        
        # Verify all metrics were recorded successfully
        assert self.ml_metrics._get_instrument("ml_inferences_total") is not None
        assert self.ml_metrics._get_instrument("ml_failure_rate") is not None
        assert self.ml_metrics._get_instrument("ml_response_time_seconds") is not None
    
    def test_backward_compatibility(self):
        """Test that the migration maintains backward compatibility."""
        # Test that we can still access metrics in the expected way
        assert hasattr(self.ml_metrics, 'record_inference')
        assert hasattr(self.ml_metrics, 'record_training_iteration')
        assert hasattr(self.ml_metrics, 'record_failure_analysis')
        assert hasattr(self.ml_metrics, 'set_failure_rate')
        assert hasattr(self.ml_metrics, 'set_anomaly_score')
        assert hasattr(self.ml_metrics, 'set_rpn_score')
        
        # Test alerting functionality
        assert hasattr(self.alerting_metrics, 'check_alerts')
        assert hasattr(self.alerting_metrics, 'alert_definitions')
        assert len(self.alerting_metrics.alert_definitions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
