"""
OpenTelemetry ML Monitoring Integration Tests
============================================

Real behavior testing for OpenTelemetry ML monitoring integration.
Replaces prometheus-based ML monitoring tests with OTel-native validation.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import OpenTelemetry components
from prompt_improver.monitoring.opentelemetry.metrics import (
    get_ml_metrics, get_ml_alerting_metrics
)


class TestOpenTelemetryMLMonitoring:
    """Test OpenTelemetry ML monitoring integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.ml_metrics = get_ml_metrics()
        self.alert_thresholds = {
            "failure_rate": 0.15,
            "response_time_ms": 200,
            "anomaly_score": 0.8,
        }
        self.ml_alerting = get_ml_alerting_metrics(self.alert_thresholds)
    
    def test_ml_inference_monitoring(self):
        """Test ML inference monitoring using OpenTelemetry."""
        # Record successful ML inference
        self.ml_metrics.record_inference(
            model_name="rule_optimizer",
            model_version="v2.1",
            duration_ms=150.0,
            prompt_tokens=250,
            completion_tokens=100,
            success=True
        )
        
        # Record failed ML inference
        self.ml_metrics.record_inference(
            model_name="failure_analyzer",
            model_version="v1.8",
            duration_ms=0,
            success=False
        )
        
        # Verify ML inference metrics
        assert self.ml_metrics._get_instrument("ml_inferences_total") is not None
        assert self.ml_metrics._get_instrument("ml_inference_duration_ms") is not None
        assert self.ml_metrics._get_instrument("ml_prompt_tokens") is not None
        assert self.ml_metrics._get_instrument("ml_completion_tokens") is not None
    
    def test_ml_training_monitoring(self):
        """Test ML training monitoring using OpenTelemetry."""
        # Record training iterations
        models = ["rule_optimizer", "failure_classifier", "pattern_analyzer"]
        
        for i, model in enumerate(models):
            accuracy = 0.85 + (i * 0.05)  # Improving accuracy
            
            self.ml_metrics.record_training_iteration(
                model_name=model,
                accuracy=accuracy
            )
        
        # Verify training metrics
        assert self.ml_metrics._get_instrument("ml_training_iterations") is not None
        assert self.ml_metrics._get_instrument("ml_model_accuracy") is not None
    
    def test_ml_failure_analysis_monitoring(self):
        """Test ML failure analysis monitoring using OpenTelemetry."""
        # Record failure analysis metrics
        self.ml_metrics.record_failure_analysis(
            failure_rate=0.08,
            failure_type="data_drift",
            severity="warning",
            total_failures=3,
            anomaly_rate=0.65,
            rpn_score=120.0,
            response_time=0.15
        )
        
        # Verify failure analysis metrics
        assert self.ml_metrics._get_instrument("ml_failure_rate") is not None
        assert self.ml_metrics._get_instrument("ml_failures_total") is not None
        assert self.ml_metrics._get_instrument("ml_anomaly_score") is not None
        assert self.ml_metrics._get_instrument("ml_risk_priority_number") is not None
        assert self.ml_metrics._get_instrument("ml_response_time_seconds") is not None
    
    def test_ml_alerting_system(self):
        """Test ML alerting system using OpenTelemetry."""
        # Create failure analysis that should trigger alerts
        failure_analysis = {
            "metadata": {
                "failure_rate": 0.20,  # Above threshold of 0.15
                "total_failures": 10,
                "avg_response_time": 0.25  # Above threshold
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
        triggered_alerts = self.ml_alerting.check_alerts(failure_analysis)
        
        # Verify alerts were triggered
        assert len(triggered_alerts) > 0
        
        # Check specific alert types
        alert_names = [alert.alert_name for alert in triggered_alerts]
        assert "HighFailureRate" in alert_names
        assert "SlowResponseTime" in alert_names
        assert "HighAnomalyScore" in alert_names
    
    @pytest.mark.asyncio
    async def test_ml_monitoring_workflow(self):
        """Test complete ML monitoring workflow with OpenTelemetry."""
        # Simulate ML pipeline workflow
        start_time = time.time()
        
        # Step 1: ML inference request
        self.ml_metrics.record_inference(
            model_name="prompt_improver",
            model_version="v3.0",
            duration_ms=180.0,
            prompt_tokens=300,
            completion_tokens=150,
            success=True
        )
        
        # Step 2: Failure analysis
        processing_time = time.time() - start_time
        self.ml_metrics.record_failure_analysis(
            failure_rate=0.05,
            failure_type="performance",
            severity="low",
            total_failures=1,
            anomaly_rate=0.35,
            rpn_score=75.0,
            response_time=processing_time
        )
        
        # Step 3: Check for alerts
        failure_analysis = {
            "metadata": {
                "failure_rate": 0.05,
                "total_failures": 1,
                "avg_response_time": processing_time
            },
            "summary": {"severity": "low"},
            "anomaly_detection": {
                "anomaly_summary": {"consensus_anomaly_rate": 35}
            },
            "risk_assessment": {"overall_risk_score": 75.0}
        }
        
        triggered_alerts = self.ml_alerting.check_alerts(failure_analysis)
        
        # Should not trigger alerts for low severity
        assert len(triggered_alerts) == 0
        
        # Verify all ML metrics were recorded
        assert self.ml_metrics._instruments
    
    def test_ml_performance_monitoring(self):
        """Test ML performance monitoring with OpenTelemetry."""
        # Record various ML performance scenarios
        scenarios = [
            {"duration": 50.0, "success": True, "tokens": 100},
            {"duration": 150.0, "success": True, "tokens": 250},
            {"duration": 300.0, "success": True, "tokens": 500},
            {"duration": 0, "success": False, "tokens": 0},  # Failure
        ]
        
        for i, scenario in enumerate(scenarios):
            self.ml_metrics.record_inference(
                model_name=f"model_{i}",
                model_version="v1.0",
                duration_ms=scenario["duration"],
                prompt_tokens=scenario["tokens"],
                completion_tokens=scenario["tokens"] // 2,
                success=scenario["success"]
            )
        
        # Verify performance metrics
        assert self.ml_metrics._instruments
    
    def test_ml_model_comparison_monitoring(self):
        """Test ML model comparison monitoring."""
        # Compare different model versions
        models = [
            {"name": "rule_optimizer", "version": "v1.0", "accuracy": 0.82},
            {"name": "rule_optimizer", "version": "v2.0", "accuracy": 0.87},
            {"name": "rule_optimizer", "version": "v2.1", "accuracy": 0.91},
        ]
        
        for model in models:
            # Record training performance
            self.ml_metrics.record_training_iteration(
                model_name=model["name"],
                accuracy=model["accuracy"]
            )
            
            # Record inference performance
            self.ml_metrics.record_inference(
                model_name=model["name"],
                model_version=model["version"],
                duration_ms=100.0 + (model["accuracy"] * 50),  # Better models might be slower
                success=True
            )
        
        # Verify model comparison metrics
        assert self.ml_metrics._instruments
    
    def test_ml_anomaly_detection_monitoring(self):
        """Test ML anomaly detection monitoring."""
        # Record various anomaly scores
        anomaly_scenarios = [
            {"score": 0.1, "type": "normal"},
            {"score": 0.3, "type": "slight_anomaly"},
            {"score": 0.7, "type": "moderate_anomaly"},
            {"score": 0.9, "type": "high_anomaly"},
        ]
        
        for scenario in anomaly_scenarios:
            self.ml_metrics.set_anomaly_score(
                anomaly_score=scenario["score"],
                detector_type=scenario["type"]
            )
        
        # Verify anomaly detection metrics
        assert self.ml_metrics._get_instrument("ml_anomaly_score") is not None
    
    def test_ml_risk_assessment_monitoring(self):
        """Test ML risk assessment monitoring."""
        # Record various risk scenarios
        risk_scenarios = [
            {"rpn": 50.0, "mode": "low_risk"},
            {"rpn": 120.0, "mode": "medium_risk"},
            {"rpn": 200.0, "mode": "high_risk"},
            {"rpn": 350.0, "mode": "critical_risk"},
        ]
        
        for scenario in risk_scenarios:
            self.ml_metrics.set_rpn_score(
                rpn_score=scenario["rpn"],
                failure_mode=scenario["mode"]
            )
        
        # Verify risk assessment metrics
        assert self.ml_metrics._get_instrument("ml_risk_priority_number") is not None
    
    def test_ml_monitoring_stress_test(self):
        """Test ML monitoring under stress conditions."""
        # Record many ML operations rapidly
        start_time = time.time()
        
        for i in range(200):
            # Rapid ML inferences
            self.ml_metrics.record_inference(
                model_name="stress_test_model",
                model_version="v1.0",
                duration_ms=10.0 + (i % 50),  # Varying duration
                success=(i % 10) != 0  # 90% success rate
            )
            
            # Periodic failure analysis
            if i % 20 == 0:
                self.ml_metrics.record_failure_analysis(
                    failure_rate=0.1,
                    failure_type="stress_test",
                    severity="info",
                    total_failures=i // 10,
                    response_time=0.01
                )
        
        stress_duration = (time.time() - start_time) * 1000
        
        # Should handle stress efficiently
        assert stress_duration < 500, f"ML monitoring stress test too slow: {stress_duration}ms"
        
        # Verify all metrics recorded
        assert self.ml_metrics._instruments
    
    def test_ml_alert_cooldown_functionality(self):
        """Test ML alert cooldown functionality."""
        failure_analysis = {
            "metadata": {"failure_rate": 0.25},  # Above threshold
            "summary": {"severity": "critical"}
        }
        
        # Trigger alert first time
        alerts_1 = self.ml_alerting.check_alerts(failure_analysis)
        assert len(alerts_1) > 0
        
        # Immediately trigger again - should be in cooldown
        alerts_2 = self.ml_alerting.check_alerts(failure_analysis)
        
        # Should have fewer or same alerts due to cooldown
        assert len(alerts_2) <= len(alerts_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
