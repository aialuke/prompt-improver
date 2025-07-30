"""
OpenTelemetry System Metrics Integration Tests
==============================================

Real behavior testing for OpenTelemetry system metrics collection.
Replaces prometheus-based system metrics tests with OTel-native validation.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import OpenTelemetry components
from prompt_improver.monitoring.opentelemetry.metrics import (
    get_http_metrics, get_database_metrics, get_ml_metrics, get_business_metrics
)


class TestOpenTelemetrySystemMetrics:
    """Test OpenTelemetry system metrics integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.http_metrics = get_http_metrics()
        self.database_metrics = get_database_metrics()
        self.ml_metrics = get_ml_metrics()
        self.business_metrics = get_business_metrics()
    
    def test_http_metrics_collection(self):
        """Test HTTP metrics collection using OpenTelemetry."""
        # Record HTTP request metrics
        self.http_metrics.record_request(
            method="GET",
            endpoint="/api/v1/health",
            status_code=200,
            duration_ms=45.2,
            request_size=1024,
            response_size=512
        )
        
        # Verify metrics instruments exist
        assert self.http_metrics._get_instrument("http_requests_total") is not None
        assert self.http_metrics._get_instrument("http_request_duration_ms") is not None
        assert self.http_metrics._get_instrument("http_request_size_bytes") is not None
        assert self.http_metrics._get_instrument("http_response_size_bytes") is not None
    
    def test_database_metrics_collection(self):
        """Test database metrics collection using OpenTelemetry."""
        # Record database query metrics
        self.database_metrics.record_query(
            operation="SELECT",
            table="rule_metadata",
            duration_ms=12.5,
            success=True
        )
        
        # Record connection metrics
        self.database_metrics.set_connection_metrics(
            active_connections=5,
            pool_size=10,
            pool_name="main"
        )
        
        # Verify metrics instruments exist
        assert self.database_metrics._get_instrument("db_queries_total") is not None
        assert self.database_metrics._get_instrument("db_query_duration_ms") is not None
        assert self.database_metrics._get_instrument("db_connections_active") is not None
        assert self.database_metrics._get_instrument("db_pool_size") is not None
    
    def test_ml_metrics_collection(self):
        """Test ML metrics collection using OpenTelemetry."""
        # Record ML inference metrics
        self.ml_metrics.record_inference(
            model_name="rule_optimizer",
            model_version="v2.1",
            duration_ms=150.0,
            prompt_tokens=250,
            completion_tokens=100,
            success=True
        )
        
        # Record training iteration
        self.ml_metrics.record_training_iteration(
            model_name="failure_analyzer",
            accuracy=0.92
        )
        
        # Verify metrics instruments exist
        assert self.ml_metrics._get_instrument("ml_inferences_total") is not None
        assert self.ml_metrics._get_instrument("ml_inference_duration_ms") is not None
        assert self.ml_metrics._get_instrument("ml_training_iterations") is not None
        assert self.ml_metrics._get_instrument("ml_model_accuracy") is not None
    
    def test_business_metrics_collection(self):
        """Test business metrics collection using OpenTelemetry."""
        # Record business metrics
        self.business_metrics.record_feature_flag_evaluation(
            flag_name="enhanced_rule_engine",
            enabled=True
        )
        
        self.business_metrics.update_active_sessions(
            change=1,
            session_type="improvement"
        )
        
        # Verify metrics instruments exist
        assert self.business_metrics._get_instrument("feature_flags_evaluated_total") is not None
        assert self.business_metrics._get_instrument("active_sessions") is not None
    
    @pytest.mark.asyncio
    async def test_real_behavior_metrics_workflow(self):
        """Test real behavior metrics workflow with OpenTelemetry."""
        # Simulate a complete request workflow
        start_time = time.time()
        
        # Step 1: HTTP request received
        self.http_metrics.record_request(
            method="POST",
            endpoint="/api/v1/improve-prompt",
            status_code=200,
            duration_ms=250.0,
            request_size=2048,
            response_size=1536
        )
        
        # Step 2: Database query executed
        self.database_metrics.record_query(
            operation="INSERT",
            table="prompt_sessions",
            duration_ms=8.2,
            success=True
        )
        
        # Step 3: ML inference performed
        self.ml_metrics.record_inference(
            model_name="prompt_improver",
            model_version="v3.0",
            duration_ms=180.0,
            prompt_tokens=300,
            completion_tokens=150,
            success=True
        )
        
        # Step 4: Business metrics updated
        self.business_metrics.record_feature_flag_evaluation(
            flag_name="ml_optimization",
            enabled=True
        )
        
        # Verify workflow completed successfully
        workflow_duration = (time.time() - start_time) * 1000
        assert workflow_duration < 1000  # Should complete quickly
        
        # All metrics should have been recorded
        assert self.http_metrics._instruments
        assert self.database_metrics._instruments
        assert self.ml_metrics._instruments
        assert self.business_metrics._instruments
    
    def test_metrics_error_handling(self):
        """Test metrics error handling and resilience."""
        # Test with invalid data - should not crash
        try:
            self.http_metrics.record_request(
                method="INVALID",
                endpoint="",
                status_code=-1,
                duration_ms=-100.0
            )
            # Should handle gracefully
        except Exception as e:
            pytest.fail(f"Metrics should handle invalid data gracefully: {e}")
        
        # Test with None values
        try:
            self.database_metrics.record_query(
                operation=None,
                table=None,
                duration_ms=None,
                success=None
            )
            # Should handle gracefully
        except Exception as e:
            pytest.fail(f"Metrics should handle None values gracefully: {e}")
    
    def test_metrics_performance(self):
        """Test metrics collection performance."""
        # Measure metrics recording performance
        start_time = time.time()
        
        # Record multiple metrics rapidly
        for i in range(100):
            self.http_metrics.record_request(
                method="GET",
                endpoint=f"/api/test/{i}",
                status_code=200,
                duration_ms=10.0
            )
        
        recording_duration = (time.time() - start_time) * 1000
        
        # Should be very fast (< 100ms for 100 recordings)
        assert recording_duration < 100, f"Metrics recording too slow: {recording_duration}ms"
    
    def test_metrics_thread_safety(self):
        """Test metrics collection thread safety."""
        import threading
        import concurrent.futures
        
        def record_metrics(thread_id: int):
            """Record metrics from a thread."""
            for i in range(10):
                self.http_metrics.record_request(
                    method="POST",
                    endpoint=f"/thread/{thread_id}/request/{i}",
                    status_code=200,
                    duration_ms=15.0
                )
        
        # Run metrics recording from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(5)]
            
            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pytest.fail(f"Thread safety test failed: {e}")
    
    def test_metrics_labels_and_attributes(self):
        """Test metrics labels and attributes functionality."""
        # Test HTTP metrics with various labels
        self.http_metrics.record_request(
            method="GET",
            endpoint="/api/v1/rules",
            status_code=200,
            duration_ms=25.0
        )
        
        self.http_metrics.record_request(
            method="POST", 
            endpoint="/api/v1/rules",
            status_code=201,
            duration_ms=45.0
        )
        
        # Test database metrics with different operations
        self.database_metrics.record_query(
            operation="SELECT",
            table="rules",
            duration_ms=5.0,
            success=True
        )
        
        self.database_metrics.record_query(
            operation="UPDATE",
            table="rules",
            duration_ms=12.0,
            success=True
        )
        
        # Verify instruments handle different label combinations
        assert self.http_metrics._instruments
        assert self.database_metrics._instruments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
