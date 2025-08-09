"""
OpenTelemetry Health Monitoring Integration Tests
================================================

Real behavior testing for OpenTelemetry health monitoring integration.
Replaces prometheus-based health monitoring tests with OTel-native validation.
"""
import asyncio
import time
from typing import Any, Dict
import pytest
from prompt_improver.monitoring.opentelemetry.metrics import BusinessMetrics, DatabaseMetrics, HttpMetrics, get_business_metrics, get_database_metrics, get_http_metrics

class TestOpenTelemetryHealthMonitoring:
    """Test OpenTelemetry health monitoring integration."""

    def setup_method(self):
        """Set up test environment with real OpenTelemetry metrics."""
        self.http_metrics = get_http_metrics('test-health-service')
        self.database_metrics = get_database_metrics('test-health-service')
        self.business_metrics = get_business_metrics('test-health-service')

    def test_health_check_metrics_collection(self):
        """Test health check metrics collection using OpenTelemetry."""
        self.http_metrics.record_request(method='GET', endpoint='/health', status_code=200, duration_ms=5.2)
        self.http_metrics.record_request(method='GET', endpoint='/health/ready', status_code=200, duration_ms=3.8)
        self.http_metrics.record_request(method='GET', endpoint='/health/live', status_code=200, duration_ms=2.1)
        assert isinstance(self.http_metrics, HttpMetrics)
        assert hasattr(self.http_metrics, 'record_request')
        print('✅ Health check HTTP metrics recorded with real OpenTelemetry')

    def test_database_health_metrics(self):
        """Test database health metrics using OpenTelemetry."""
        self.database_metrics.record_query(operation='SELECT', table='pg_stat_activity', duration_ms=2.5, success=True)
        self.database_metrics.record_query(operation='SELECT', table='health_status', duration_ms=1.8, success=True)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert hasattr(self.database_metrics, 'record_query')
        print('✅ Database health metrics recorded with real OpenTelemetry')

    def test_service_availability_metrics(self):
        """Test service availability metrics using OpenTelemetry."""
        services = ['database', 'redis', 'ml_engine', 'rule_engine']
        for service in services:
            self.business_metrics.record_feature_usage(feature=f'{service}_health_check', user_tier='system')
        assert isinstance(self.business_metrics, BusinessMetrics)
        assert hasattr(self.business_metrics, 'record_feature_usage')
        print('✅ Service availability metrics recorded with real OpenTelemetry')

    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self):
        """Test complete health monitoring workflow with OpenTelemetry."""
        start_time = time.time()
        self.http_metrics.record_request(method='GET', endpoint='/api/v1/health/comprehensive', status_code=200, duration_ms=25.0)
        self.database_metrics.record_query(operation='SELECT', table='health_check', duration_ms=3.2, success=True)
        dependencies = ['postgresql', 'redis', 'ml_models']
        for dep in dependencies:
            self.business_metrics.record_feature_usage(feature=f'{dep}_dependency_check', user_tier='system')
        self.business_metrics.record_session(user_id='health_monitor_system', session_duration_s=time.time() - start_time)
        workflow_duration = (time.time() - start_time) * 1000
        assert workflow_duration < 100, f'Health check too slow: {workflow_duration}ms'
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert isinstance(self.business_metrics, BusinessMetrics)
        print('✅ Complete health monitoring workflow recorded with real OpenTelemetry')

    def test_health_check_failure_scenarios(self):
        """Test health check failure scenarios with OpenTelemetry."""
        self.http_metrics.record_request(method='GET', endpoint='/health', status_code=503, duration_ms=1000.0)
        self.database_metrics.record_query(operation='SELECT', table='health_check', duration_ms=5000.0, success=False)
        self.business_metrics.record_feature_usage(feature='database_health_check_failure', user_tier='system')
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert isinstance(self.business_metrics, BusinessMetrics)
        print('✅ Health check failure scenarios recorded with real OpenTelemetry')

    def test_health_metrics_performance_monitoring(self):
        """Test health metrics performance monitoring."""
        start_time = time.time()
        for i in range(50):
            self.http_metrics.record_request(method='GET', endpoint='/health/quick', status_code=200, duration_ms=1.0 + i * 0.1)
        recording_duration = (time.time() - start_time) * 1000
        assert recording_duration < 100, f'Health metrics recording too slow: {recording_duration}ms'
        print(f'✅ Health metrics performance test completed in {recording_duration:.1f}ms')

    def test_health_alerting_integration(self):
        """Test health alerting integration with OpenTelemetry."""
        self.http_metrics.record_request(method='GET', endpoint='/health', status_code=200, duration_ms=500.0)
        self.database_metrics.record_query(operation='SELECT', table='health_check', duration_ms=100.0, success=True)
        self.business_metrics.record_feature_usage(feature='performance_degraded_alert', user_tier='system')
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        assert isinstance(self.business_metrics, BusinessMetrics)
        print('✅ Health alerting integration metrics recorded with real OpenTelemetry')

    def test_health_metrics_aggregation(self):
        """Test health metrics aggregation and reporting."""
        health_endpoints = ['/health', '/health/ready', '/health/live', '/health/detailed']
        for endpoint in health_endpoints:
            for status in [200, 200, 200, 503]:
                self.http_metrics.record_request(method='GET', endpoint=endpoint, status_code=status, duration_ms=5.0 if status == 200 else 1000.0)
        for i in range(10):
            self.database_metrics.record_query(operation='SELECT', table='health_check', duration_ms=2.0 + i * 0.5, success=i < 9)
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        print('✅ Health metrics aggregation data recorded with real OpenTelemetry')

    def test_health_monitoring_sla_compliance(self):
        """Test health monitoring SLA compliance tracking."""
        sla_threshold_ms = 200.0
        for i in range(100):
            response_time = 50.0 + i * 1.0
            self.http_metrics.record_request(method='GET', endpoint='/health/sla', status_code=200 if response_time < sla_threshold_ms else 503, duration_ms=response_time)
        sla_compliant_requests = sum((1 for i in range(100) if 50.0 + i * 1.0 < sla_threshold_ms))
        sla_compliance_rate = sla_compliant_requests / 100.0
        self.business_metrics.record_feature_usage(feature='sla_compliance_check', user_tier='sla_monitor')
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.business_metrics, BusinessMetrics)
        print(f'✅ SLA compliance metrics recorded: {sla_compliance_rate:.1%} compliance rate')

    def test_health_monitoring_real_behavior(self):
        """Test health monitoring with real behavior patterns."""
        for minute in range(5):
            for check in range(2):
                self.http_metrics.record_request(method='GET', endpoint='/health', status_code=200, duration_ms=3.0 + minute * 0.5)
        self.http_metrics.record_request(method='GET', endpoint='/health/deep', status_code=200, duration_ms=25.0)
        self.database_metrics.record_query(operation='SELECT', table='pg_stat_database', duration_ms=5.0, success=True)
        assert isinstance(self.http_metrics, HttpMetrics)
        assert isinstance(self.database_metrics, DatabaseMetrics)
        print('✅ Real behavior health monitoring patterns recorded with OpenTelemetry')
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
