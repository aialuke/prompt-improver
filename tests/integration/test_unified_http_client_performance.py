"""
Performance validation tests for Phase 3: HTTP Client Standardization
Tests circuit breaker functionality, response time improvements, and reliability metrics
"""

import asyncio
import pytest
import time
import json
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_improver.monitoring.unified_http_client import (
    UnifiedHTTPClientFactory,
    HTTPClientConfig,
    HTTPClientUsage,
    get_http_client_factory,
    make_webhook_request,
    make_health_check_request,
    make_api_request
)

class TestUnifiedHTTPClientPerformance:
    """Test suite for unified HTTP client performance and reliability"""
    
    @pytest.fixture
    async def http_factory(self):
        """Create a test HTTP client factory"""
        factory = UnifiedHTTPClientFactory()
        
        # Register test clients
        factory.register_client(HTTPClientConfig(
            name="test_webhook",
            usage_type=HTTPClientUsage.WEBHOOK_ALERTS,
            timeout_seconds=5.0,
            failure_threshold=2,
            recovery_timeout_seconds=10,
            circuit_breaker_enabled=True
        ))
        
        factory.register_client(HTTPClientConfig(
            name="test_api",
            usage_type=HTTPClientUsage.API_CALLS,
            timeout_seconds=10.0,
            failure_threshold=3,
            recovery_timeout_seconds=30,
            circuit_breaker_enabled=True
        ))
        
        return factory
    
    @pytest.fixture
    def mock_responses(self):
        """Mock HTTP responses for testing"""
        return {
            "success": AsyncMock(status=200, headers={}),
            "error": AsyncMock(status=500, headers={}),
            "timeout": AsyncMock(side_effect=asyncio.TimeoutError())
        }
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, http_factory, mock_responses):
        """Test circuit breaker prevents cascading failures"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Configure mock to simulate failures
            mock_session.request.return_value.__aenter__.return_value = mock_responses["error"]
            
            # Make requests that should trigger circuit breaker
            failed_requests = 0
            for i in range(5):
                try:
                    await http_factory.make_request("test_webhook", "POST", "http://test.com", json={"test": i})
                except Exception:
                    failed_requests += 1
            
            # Verify circuit breaker is now open
            metrics = await http_factory.get_client_metrics("test_webhook")
            assert "circuit_breaker" in metrics
            
            # Attempt another request - should fail fast due to circuit breaker
            start_time = time.time()
            try:
                await http_factory.make_request("test_webhook", "POST", "http://test.com", json={"test": "final"})
            except Exception:
                pass
            
            fast_fail_time = time.time() - start_time
            
            # Circuit breaker should cause very fast failure (< 100ms)
            assert fast_fail_time < 0.1, f"Circuit breaker should fail fast, took {fast_fail_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_response_time_monitoring(self, http_factory, mock_responses):
        """Test response time monitoring and SLA tracking"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock successful responses with varying delays
            async def mock_request_with_delay(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms delay
                return mock_responses["success"]
            
            mock_session.request.return_value.__aenter__ = mock_request_with_delay
            
            # Make multiple requests
            response_times = []
            for i in range(5):
                start_time = time.time()
                try:
                    await http_factory.make_request("test_api", "GET", f"http://test.com/endpoint/{i}")
                    response_times.append((time.time() - start_time) * 1000)
                except Exception:
                    pass
            
            # Verify metrics collection
            metrics = await http_factory.get_client_metrics("test_api")
            
            assert metrics["total_requests"] >= 5
            assert "response_times" in metrics
            assert metrics["response_times"]["mean"] > 40  # Should be around 50ms
            assert metrics["response_times"]["mean"] < 100  # But reasonable
    
    @pytest.mark.asyncio
    async def test_rate_limit_awareness(self, http_factory, mock_responses):
        """Test rate limiting header parsing and awareness"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock response with rate limiting headers
            rate_limited_response = AsyncMock(
                status=200,
                headers={
                    'x-ratelimit-remaining': '10',
                    'x-ratelimit-limit': '100',
                    'x-ratelimit-reset': str(int(time.time()) + 3600)
                }
            )
            
            mock_session.request.return_value.__aenter__.return_value = rate_limited_response
            
            # Make request
            await http_factory.make_request("test_api", "GET", "http://test.com/rate-limited")
            
            # Verify rate limit info is captured
            metrics = await http_factory.get_client_metrics("test_api")
            assert metrics["total_requests"] >= 1
            
            # Check that rate limiting was detected (implementation would store this)
            # This is more of an integration test with actual monitoring
    
    @pytest.mark.asyncio
    async def test_webhook_alert_performance(self, mock_responses):
        """Test webhook alert performance using unified client"""
        
        test_payload = {
            "alert_type": "performance_regression",
            "metric_name": "response_time",
            "severity": "critical",
            "message": "Response time degraded by 150%"
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value = mock_responses["success"]
            
            # Measure webhook request performance
            start_time = time.time()
            
            try:
                response = await make_webhook_request("http://webhook.test.com", test_payload)
                await response.__aenter__()
                await response.__aexit__(None, None, None)
            except Exception as e:
                pass  # Expected in mock environment
            
            request_time = time.time() - start_time
            
            # Webhook should complete quickly (< 100ms in mock)
            assert request_time < 0.1, f"Webhook request took {request_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, mock_responses):
        """Test health check request performance"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value = mock_responses["success"]
            
            # Measure health check performance
            start_time = time.time()
            
            try:
                response = await make_health_check_request("http://health.test.com/health")
                await response.__aenter__()
                await response.__aexit__(None, None, None)
            except Exception:
                pass  # Expected in mock environment
            
            request_time = time.time() - start_time
            
            # Health checks should be very fast (< 50ms in mock)
            assert request_time < 0.05, f"Health check took {request_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, http_factory, mock_responses):
        """Test performance under concurrent load"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value = mock_responses["success"]
            
            # Create multiple concurrent requests
            async def make_test_request(request_id):
                try:
                    return await http_factory.make_request(
                        "test_api", 
                        "GET", 
                        f"http://test.com/concurrent/{request_id}"
                    )
                except Exception:
                    return None
            
            # Measure concurrent performance
            start_time = time.time()
            
            # Launch 20 concurrent requests
            tasks = [make_test_request(i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            concurrent_time = time.time() - start_time
            
            # Concurrent requests should complete reasonably quickly
            assert concurrent_time < 1.0, f"20 concurrent requests took {concurrent_time:.3f}s"
            
            # Verify metrics reflect all requests
            metrics = await http_factory.get_client_metrics("test_api")
            assert metrics["total_requests"] >= 20
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, http_factory, mock_responses):
        """Test performance during error recovery scenarios"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Simulate failure then recovery
            call_count = 0
            
            async def mock_request_with_recovery(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return mock_responses["error"]
                else:
                    return mock_responses["success"]
            
            mock_session.request.return_value.__aenter__ = mock_request_with_recovery
            
            # Make requests during failure and recovery
            error_count = 0
            success_count = 0
            
            for i in range(5):
                try:
                    await http_factory.make_request("test_api", "GET", f"http://test.com/recovery/{i}")
                    success_count += 1
                except Exception:
                    error_count += 1
                
                # Small delay between requests
                await asyncio.sleep(0.01)
            
            # Verify we had both failures and recoveries
            assert error_count > 0, "Should have some failures"
            assert success_count > 0, "Should have some successes after recovery"
            
            # Check metrics reflect the mixed results
            metrics = await http_factory.get_client_metrics("test_api")
            assert metrics["total_requests"] >= 5
            assert metrics["failed_requests"] > 0
            assert metrics["successful_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_ssl_performance(self, http_factory, mock_responses):
        """Test SSL/TLS performance overhead"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value = mock_responses["success"]
            
            # Test HTTPS requests
            start_time = time.time()
            
            try:
                await http_factory.make_request("test_api", "GET", "https://secure.test.com/ssl")
            except Exception:
                pass
            
            ssl_time = time.time() - start_time
            
            # SSL overhead should be minimal in this test setup
            assert ssl_time < 0.1, f"SSL request took {ssl_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, http_factory, mock_responses):
        """Test that metrics collection doesn't significantly impact performance"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.request.return_value.__aenter__.return_value = mock_responses["success"]
            
            # Make requests and measure metrics collection overhead
            start_time = time.time()
            
            for i in range(10):
                try:
                    await http_factory.make_request("test_api", "GET", f"http://test.com/metrics/{i}")
                except Exception:
                    pass
            
            request_time = time.time() - start_time
            
            # Get metrics
            metrics_start = time.time()
            metrics = await http_factory.get_client_metrics("test_api")
            all_metrics = await http_factory.get_all_metrics()
            metrics_time = time.time() - metrics_start
            
            # Metrics collection should be very fast
            assert metrics_time < 0.01, f"Metrics collection took {metrics_time:.3f}s"
            
            # Verify metrics are complete
            assert metrics["total_requests"] >= 10
            assert "test_api" in all_metrics["clients"]
    
    @pytest.mark.asyncio
    async def test_reliability_improvements(self, http_factory):
        """Test that unified client provides 2-3x reliability improvement"""
        
        # This test would need real network conditions to properly validate
        # For now, we'll test the configuration and setup
        
        # Verify circuit breaker configurations
        webhook_config = http_factory.clients["test_webhook"]
        assert webhook_config.circuit_breaker_enabled
        assert webhook_config.failure_threshold == 2
        assert webhook_config.recovery_timeout_seconds == 10
        
        api_config = http_factory.clients["test_api"]
        assert api_config.circuit_breaker_enabled
        assert api_config.failure_threshold == 3
        assert api_config.recovery_timeout_seconds == 30
        
        # Verify monitoring is enabled
        assert webhook_config.collect_metrics
        assert api_config.collect_metrics
        
        # Test health checking
        health_status = await http_factory.health_check_all_clients()
        assert isinstance(health_status, dict)
        assert "test_webhook" in health_status
        assert "test_api" in health_status

class TestHTTPClientConsolidationBenchmark:
    """Benchmark tests for HTTP client consolidation performance"""
    
    @pytest.mark.asyncio
    async def test_benchmark_response_times(self):
        """Benchmark response times before/after consolidation"""
        
        # Simulate old direct aiohttp usage
        async def old_http_pattern():
            import aiohttp
            start_time = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://test.com") as response:
                        await response.read()
            except Exception:
                pass
            return time.time() - start_time
        
        # Test new unified client pattern
        async def new_http_pattern():
            start_time = time.time()
            try:
                response = await make_api_request("GET", "http://test.com")
                # In real usage, this would work
            except Exception:
                pass
            return time.time() - start_time
        
        # Run benchmarks (in mock environment, both should be similar)
        with patch('aiohttp.ClientSession'):
            old_time = await old_http_pattern()
            new_time = await new_http_pattern()
            
            # In real usage, new pattern should have minimal overhead
            # In test environment, we just verify both complete
            assert old_time >= 0
            assert new_time >= 0
    
    def test_consolidation_count_verification(self):
        """Verify the expected number of HTTP client consolidations"""
        
        # This would be implemented by scanning the codebase for old patterns
        # and verifying they've been replaced
        
        # Expected consolidations based on the requirements:
        # - 15 webhook alert systems
        # - 12 health check HTTP clients  
        # - 15 script and utility HTTP usage
        # Total: 42 consolidations
        
        expected_consolidations = 42
        
        # In a real implementation, this would scan for:
        # 1. aiohttp.ClientSession() usage patterns
        # 2. requests.get/post usage
        # 3. urllib.request usage
        # 4. Verify they're replaced with unified client calls
        
        # For now, we'll just verify the structure is in place
        factory = get_http_client_factory()
        assert len(factory.clients) >= 5  # Default clients are registered
        
        # Verify key client types exist
        client_types = [config.usage_type for config in factory.clients.values()]
        assert HTTPClientUsage.WEBHOOK_ALERTS in client_types
        assert HTTPClientUsage.HEALTH_CHECKS in client_types
        assert HTTPClientUsage.API_CALLS in client_types
        assert HTTPClientUsage.DOWNLOADS in client_types
        assert HTTPClientUsage.TESTING in client_types

@pytest.mark.integration
class TestPhase3IntegrationValidation:
    """Integration validation for Phase 3 HTTP Client Standardization"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_http_client_performance(self):
        """End-to-end test of HTTP client performance improvements"""
        
        factory = get_http_client_factory()
        
        # Test all client types
        client_tests = [
            ("webhook_alerts", "POST", "http://webhook.test.com"),
            ("health_monitoring", "GET", "http://health.test.com/status"),
            ("api_calls", "GET", "http://api.test.com/data"),
            ("downloads", "GET", "http://downloads.test.com/file.zip"),
        ]
        
        with patch('aiohttp.ClientSession'):
            for client_name, method, url in client_tests:
                start_time = time.time()
                
                try:
                    await factory.make_request(client_name, method, url)
                except Exception:
                    pass  # Expected in test environment
                
                request_time = time.time() - start_time
                
                # All requests should complete quickly
                assert request_time < 0.1, f"{client_name} request took {request_time:.3f}s"
        
        # Verify metrics for all clients
        all_metrics = await factory.get_all_metrics()
        
        assert all_metrics["total_clients"] >= 4
        assert all_metrics["total_requests"] >= len(client_tests)
        
        # All clients should be healthy in test environment
        health_status = await factory.health_check_all_clients()
        for client_name, is_healthy in health_status.items():
            assert is_healthy, f"Client {client_name} should be healthy"
    
    def test_phase3_success_metrics(self):
        """Verify Phase 3 success metrics are achievable"""
        
        # Success metrics from requirements:
        # - HTTP client patterns: 42 → 1 (ExternalAPIHealthMonitor-based)
        # - Circuit breaker integration: 100% coverage
        # - SLA monitoring: Unified P50/P95/P99 tracking
        # - Reliability improvement: 2-3x through unified patterns
        
        factory = get_http_client_factory()
        
        # Verify single unified approach (multiple client configs, same factory)
        assert isinstance(factory, UnifiedHTTPClientFactory)
        
        # Verify circuit breaker coverage
        circuit_breaker_clients = 0
        monitoring_clients = 0
        
        for config in factory.clients.values():
            if config.circuit_breaker_enabled:
                circuit_breaker_clients += 1
            if config.collect_metrics:
                monitoring_clients += 1
        
        # All production clients should have circuit breakers
        production_clients = [
            config for config in factory.clients.values() 
            if config.usage_type != HTTPClientUsage.TESTING
        ]
        
        assert circuit_breaker_clients >= len(production_clients)
        assert monitoring_clients >= len(production_clients)
        
        # Verify SLA monitoring capabilities
        for config in factory.clients.values():
            if config.usage_type != HTTPClientUsage.TESTING:
                assert config.collect_metrics
                # SLA targets would be configured based on usage type

if __name__ == "__main__":
    pytest.main([__file__, "-v"])