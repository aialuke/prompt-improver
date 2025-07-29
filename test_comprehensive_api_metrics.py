#!/usr/bin/env python3
"""
Comprehensive testing strategy for API metrics module.
Tests integration, performance, and APES system compliance.
"""

import asyncio
import pytest
import time
import json
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the fixed API metrics module
from src.prompt_improver.metrics.api_metrics import (
    APIMetricsCollector,
    APIUsageMetric,
    UserJourneyMetric,
    RateLimitMetric,
    AuthenticationMetric,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod,
    get_api_metrics_collector,
    record_api_request,
    record_user_journey_event
)

# Import database utilities for PostgreSQL testing
try:
    from src.prompt_improver.database.connection_manager import get_connection_manager
    from src.prompt_improver.database.models import Base
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("PostgreSQL integration not available for testing")


class TestResults:
    """Track test results and performance metrics."""

    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
        self.errors = []

    def record_test(self, test_name: str, passed: bool, duration: float, details: str = ""):
        """Record test result."""
        self.results[test_name] = {
            "passed": passed,
            "duration_ms": duration * 1000,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def record_performance(self, metric_name: str, value: float, unit: str = "ms"):
        """Record performance metric."""
        self.performance_metrics[metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def add_error(self, test_name: str, error: str):
        """Record error."""
        self.errors.append({
            "test": test_name,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["passed"])

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_errors": len(self.errors),
            "performance_metrics": self.performance_metrics,
            "detailed_results": self.results,
            "errors": self.errors
        }


class ComprehensiveAPIMetricsTest:
    """Comprehensive test suite for API metrics."""

    def __init__(self):
        self.results = TestResults()
        self.test_collector = None

    async def setup_test_environment(self):
        """Set up test environment."""
        print("🔧 Setting up test environment...")

        # Create test collector with optimized config
        config = {
            "max_api_metrics": 10000,
            "max_journey_metrics": 5000,
            "max_rate_limit_metrics": 2000,
            "max_auth_metrics": 3000,
            "aggregation_window_minutes": 1,
            "retention_hours": 24,
            "journey_timeout_minutes": 30
        }

        self.test_collector = APIMetricsCollector(config)
        await self.test_collector.start_collection()
        print("✓ Test collector initialized and started")

    async def teardown_test_environment(self):
        """Clean up test environment."""
        if self.test_collector:
            await self.test_collector.stop_collection()
        print("✓ Test environment cleaned up")

    async def test_type_annotations_and_runtime(self):
        """Test 1.1: Validate type annotations and runtime behavior."""
        test_name = "type_annotations_runtime"
        start_time = time.time()

        try:
            # Test that all metric types can be created without errors
            api_metric = APIUsageMetric(
                endpoint="/api/v1/test",
                method=HTTPMethod.POST,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200,
                response_time_ms=150.5,
                request_size_bytes=1024,
                response_size_bytes=2048,
                user_id="test_user",
                session_id="test_session",
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=3,
                payload_type="application/json",
                rate_limited=False,
                cache_hit=True,
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                api_version="v1"
            )

            journey_metric = UserJourneyMetric(
                user_id="test_user",
                session_id="test_session",
                journey_stage=UserJourneyStage.FIRST_USE,
                event_type="test_event",
                endpoint="/api/v1/test",
                success=True,
                conversion_value=10.0,
                time_to_action_seconds=30.0,
                previous_stage=UserJourneyStage.ONBOARDING,
                feature_flags_active=["test_flag"],
                cohort_id="test_cohort",
                timestamp=datetime.now(timezone.utc),
                metadata={"test": "data"}
            )

            # Test recording without errors
            await self.test_collector.record_api_usage(api_metric)
            await self.test_collector.record_user_journey(journey_metric)

            # Verify collections are properly typed
            assert len(self.test_collector.api_usage_metrics) == 1
            assert len(self.test_collector.journey_metrics) == 1

            duration = time.time() - start_time
            self.results.record_test(test_name, True, duration, "All type annotations resolved correctly")
            print(f"✓ {test_name}: PASSED ({duration*1000:.2f}ms)")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")

    async def test_metrics_collection_and_aggregation(self):
        """Test 1.2: Validate metrics collection and aggregation with real data."""
        test_name = "metrics_collection_aggregation"
        start_time = time.time()

        try:
            # Record multiple metrics
            for i in range(10):
                api_metric = APIUsageMetric(
                    endpoint=f"/api/v1/endpoint{i % 3}",
                    method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200 if i < 8 else 500,
                    response_time_ms=100.0 + i * 10,
                    request_size_bytes=500 + i * 50,
                    response_size_bytes=1000 + i * 100,
                    user_id=f"user_{i}",
                    session_id=f"session_{i}",
                    ip_address="192.168.1.1",
                    user_agent="TestAgent/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=i,
                    payload_type="application/json",
                    rate_limited=i == 9,
                    cache_hit=i % 2 == 0,
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)

            # Test aggregation
            await self.test_collector._aggregate_metrics()

            # Test analytics generation
            analytics = await self.test_collector.get_endpoint_analytics(hours=1)

            # Validate analytics structure
            assert "total_requests" in analytics
            assert "endpoint_analytics" in analytics
            assert analytics["total_requests"] == 10
            assert len(analytics["endpoint_analytics"]) == 3  # 3 unique endpoints

            duration = time.time() - start_time
            self.results.record_test(test_name, True, duration, f"Processed {analytics['total_requests']} requests")
            print(f"✓ {test_name}: PASSED ({duration*1000:.2f}ms)")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")

    async def test_prometheus_fallback(self):
        """Test 1.3: Verify graceful degradation when Prometheus is unavailable."""
        test_name = "prometheus_fallback"
        start_time = time.time()

        try:
            # Test that metrics work even if Prometheus is not available
            # The MetricsMixin should handle this gracefully

            # Check if metrics are available
            metrics_available = self.test_collector.metrics_available
            print(f"Metrics available: {metrics_available}")

            # Record a metric regardless of Prometheus availability
            api_metric = APIUsageMetric(
                endpoint="/api/v1/fallback_test",
                method=HTTPMethod.GET,
                category=EndpointCategory.HEALTH_CHECK,
                status_code=200,
                response_time_ms=50.0,
                request_size_bytes=100,
                response_size_bytes=200,
                user_id="fallback_user",
                session_id="fallback_session",
                ip_address="127.0.0.1",
                user_agent="FallbackTest/1.0",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=0,
                payload_type="application/json",
                rate_limited=False,
                cache_hit=False,
                authentication_method=AuthenticationMethod.ANONYMOUS,
                api_version="v1"
            )

            await self.test_collector.record_api_usage(api_metric)

            # Verify the metric was recorded in our collections
            assert len(self.test_collector.api_usage_metrics) > 0

            duration = time.time() - start_time
            self.results.record_test(test_name, True, duration, f"Graceful fallback working, metrics_available={metrics_available}")
            print(f"✓ {test_name}: PASSED ({duration*1000:.2f}ms)")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")

    async def test_performance_benchmarks(self):
        """Test 3.1: Performance benchmarks for <200ms SLA compliance."""
        test_name = "performance_benchmarks"
        start_time = time.time()

        try:
            # Test single metric recording performance
            single_metric_times = []
            for i in range(100):
                metric_start = time.time()

                api_metric = APIUsageMetric(
                    endpoint=f"/api/v1/perf_test_{i}",
                    method=HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200,
                    response_time_ms=100.0,
                    request_size_bytes=1024,
                    response_size_bytes=2048,
                    user_id=f"perf_user_{i}",
                    session_id=f"perf_session_{i}",
                    ip_address="192.168.1.1",
                    user_agent="PerfTest/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=1,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=False,
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )

                await self.test_collector.record_api_usage(api_metric)

                metric_end = time.time()
                single_metric_times.append((metric_end - metric_start) * 1000)  # Convert to ms

            # Calculate performance metrics
            avg_recording_time = sum(single_metric_times) / len(single_metric_times)
            max_recording_time = max(single_metric_times)
            min_recording_time = min(single_metric_times)

            # Test analytics generation performance
            analytics_start = time.time()
            analytics = await self.test_collector.get_endpoint_analytics(hours=1)
            analytics_time = (time.time() - analytics_start) * 1000

            # Record performance metrics
            self.results.record_performance("avg_metric_recording_time", avg_recording_time, "ms")
            self.results.record_performance("max_metric_recording_time", max_recording_time, "ms")
            self.results.record_performance("min_metric_recording_time", min_recording_time, "ms")
            self.results.record_performance("analytics_generation_time", analytics_time, "ms")

            # Check SLA compliance (should be well under 200ms)
            sla_compliant = max_recording_time < 200 and analytics_time < 200

            duration = time.time() - start_time
            details = f"Avg: {avg_recording_time:.2f}ms, Max: {max_recording_time:.2f}ms, Analytics: {analytics_time:.2f}ms"
            self.results.record_test(test_name, sla_compliant, duration, details)

            if sla_compliant:
                print(f"✓ {test_name}: PASSED - {details}")
            else:
                print(f"✗ {test_name}: FAILED - SLA violation: {details}")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")

    async def test_concurrent_operations(self):
        """Test 3.2: Concurrent metric collection under load."""
        test_name = "concurrent_operations"
        start_time = time.time()

        try:
            # Test concurrent metric recording
            async def record_metrics_batch(batch_id: int, count: int):
                for i in range(count):
                    api_metric = APIUsageMetric(
                        endpoint=f"/api/v1/concurrent_test_{batch_id}_{i}",
                        method=HTTPMethod.GET,
                        category=EndpointCategory.PROMPT_IMPROVEMENT,
                        status_code=200,
                        response_time_ms=50.0,
                        request_size_bytes=512,
                        response_size_bytes=1024,
                        user_id=f"concurrent_user_{batch_id}_{i}",
                        session_id=f"concurrent_session_{batch_id}_{i}",
                        ip_address="192.168.1.1",
                        user_agent="ConcurrentTest/1.0",
                        timestamp=datetime.now(timezone.utc),
                        query_parameters_count=0,
                        payload_type="application/json",
                        rate_limited=False,
                        cache_hit=False,
                        authentication_method=AuthenticationMethod.JWT_TOKEN,
                        api_version="v1"
                    )
                    await self.test_collector.record_api_usage(api_metric)

            # Run concurrent batches
            concurrent_start = time.time()
            tasks = [record_metrics_batch(i, 20) for i in range(10)]  # 10 batches of 20 metrics each
            await asyncio.gather(*tasks)
            concurrent_time = (time.time() - concurrent_start) * 1000

            # Verify all metrics were recorded
            expected_metrics = 200  # 10 batches * 20 metrics
            actual_metrics = len(self.test_collector.api_usage_metrics)

            # Record performance
            self.results.record_performance("concurrent_recording_time", concurrent_time, "ms")
            self.results.record_performance("concurrent_throughput", expected_metrics / (concurrent_time / 1000), "metrics/sec")

            success = actual_metrics >= expected_metrics and concurrent_time < 5000  # Should complete in under 5 seconds

            duration = time.time() - start_time
            details = f"Recorded {actual_metrics}/{expected_metrics} metrics in {concurrent_time:.2f}ms"
            self.results.record_test(test_name, success, duration, details)

            if success:
                print(f"✓ {test_name}: PASSED - {details}")
            else:
                print(f"✗ {test_name}: FAILED - {details}")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")

    async def test_memory_usage_and_cleanup(self):
        """Test 3.3: Memory usage and cleanup efficiency."""
        test_name = "memory_usage_cleanup"
        start_time = time.time()

        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Record a large number of metrics
            for i in range(1000):
                api_metric = APIUsageMetric(
                    endpoint=f"/api/v1/memory_test_{i}",
                    method=HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200,
                    response_time_ms=100.0,
                    request_size_bytes=1024,
                    response_size_bytes=2048,
                    user_id=f"memory_user_{i}",
                    session_id=f"memory_session_{i}",
                    ip_address="192.168.1.1",
                    user_agent="MemoryTest/1.0",
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=25),  # Old timestamp for cleanup
                    query_parameters_count=1,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=False,
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)

            # Check memory after recording
            after_recording_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Trigger cleanup
            await self.test_collector._cleanup_old_metrics(datetime.now(timezone.utc))

            # Check memory after cleanup
            after_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Record memory metrics
            self.results.record_performance("initial_memory_mb", initial_memory, "MB")
            self.results.record_performance("after_recording_memory_mb", after_recording_memory, "MB")
            self.results.record_performance("after_cleanup_memory_mb", after_cleanup_memory, "MB")
            self.results.record_performance("memory_growth_mb", after_recording_memory - initial_memory, "MB")
            self.results.record_performance("memory_freed_mb", after_recording_memory - after_cleanup_memory, "MB")

            # Check if cleanup was effective
            cleanup_effective = after_cleanup_memory < after_recording_memory
            memory_reasonable = (after_recording_memory - initial_memory) < 100  # Less than 100MB growth

            success = cleanup_effective and memory_reasonable

            duration = time.time() - start_time
            details = f"Memory: {initial_memory:.1f}MB → {after_recording_memory:.1f}MB → {after_cleanup_memory:.1f}MB"
            self.results.record_test(test_name, success, duration, details)

            if success:
                print(f"✓ {test_name}: PASSED - {details}")
            else:
                print(f"✗ {test_name}: FAILED - {details}")

        except Exception as e:
            duration = time.time() - start_time
            self.results.record_test(test_name, False, duration, str(e))
            self.results.add_error(test_name, str(e))
            print(f"✗ {test_name}: FAILED - {e}")


async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive API Metrics Testing")
    print("=" * 60)

    test_suite = ComprehensiveAPIMetricsTest()

    try:
        # Setup
        await test_suite.setup_test_environment()

        # Run all tests
        print("\n📋 1. INTEGRATION TESTS")
        print("-" * 30)
        await test_suite.test_type_annotations_and_runtime()
        await test_suite.test_metrics_collection_and_aggregation()
        await test_suite.test_prometheus_fallback()

        print("\n⚡ 3. PERFORMANCE TESTS")
        print("-" * 30)
        await test_suite.test_performance_benchmarks()
        await test_suite.test_concurrent_operations()
        await test_suite.test_memory_usage_and_cleanup()

        # Generate summary
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        summary = test_suite.results.get_summary()

        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Errors: {summary['total_errors']}")

        print("\n🔍 PERFORMANCE METRICS")
        print("-" * 30)
        for metric_name, metric_data in summary['performance_metrics'].items():
            print(f"{metric_name}: {metric_data['value']:.2f} {metric_data['unit']}")

        if summary['errors']:
            print("\n❌ ERRORS")
            print("-" * 30)
            for error in summary['errors']:
                print(f"Test: {error['test']}")
                print(f"Error: {error['error']}")
                print()

        # Save detailed results
        with open('api_metrics_test_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: api_metrics_test_results.json")

        return summary['success_rate'] == 100.0

    finally:
        await test_suite.teardown_test_environment()


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    exit(0 if success else 1)
