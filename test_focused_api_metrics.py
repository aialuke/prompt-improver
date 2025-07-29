#!/usr/bin/env python3
"""
Focused testing for API metrics core functionality.
"""

import asyncio
import time
import json
from datetime import datetime, timezone, timedelta

from src.prompt_improver.metrics.api_metrics import (
    APIMetricsCollector,
    APIUsageMetric,
    UserJourneyMetric,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod,
    get_api_metrics_collector,
    record_api_request,
    record_user_journey_event
)


async def test_basic_functionality():
    """Test 1: Basic functionality and type safety."""
    print("🧪 Test 1: Basic Functionality")
    
    collector = APIMetricsCollector({
        "max_api_metrics": 1000,
        "aggregation_window_minutes": 1
    })
    
    await collector.start_collection()
    
    try:
        # Test API metric recording
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
        
        await collector.record_api_usage(api_metric)
        
        # Test user journey recording
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
        
        await collector.record_user_journey(journey_metric)
        
        # Verify collections
        assert len(collector.api_usage_metrics) == 1
        assert len(collector.journey_metrics) == 1
        
        # Test stats
        stats = collector.get_collection_stats()
        assert stats["api_calls_tracked"] == 1
        assert stats["journey_events_tracked"] == 1
        
        print("✓ Basic functionality: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality: FAILED - {e}")
        return False
    finally:
        await collector.stop_collection()


async def test_analytics_generation():
    """Test 2: Analytics generation with real data."""
    print("🧪 Test 2: Analytics Generation")
    
    collector = APIMetricsCollector()
    await collector.start_collection()
    
    try:
        # Record multiple metrics for analytics
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
            await collector.record_api_usage(api_metric)
        
        # Generate analytics
        analytics = await collector.get_endpoint_analytics(hours=1)
        
        # Validate analytics
        assert "total_requests" in analytics
        assert "endpoint_analytics" in analytics
        assert analytics["total_requests"] == 10
        assert len(analytics["endpoint_analytics"]) == 3
        
        # Test journey analytics
        journey_analytics = await collector.get_user_journey_analytics(hours=1)
        assert "status" in journey_analytics  # Should be "no_data" since no journey metrics
        
        print("✓ Analytics generation: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Analytics generation: FAILED - {e}")
        return False
    finally:
        await collector.stop_collection()


async def test_performance_sla():
    """Test 3: Performance SLA compliance (<200ms)."""
    print("🧪 Test 3: Performance SLA Compliance")
    
    collector = APIMetricsCollector()
    await collector.start_collection()
    
    try:
        # Test single metric recording time
        times = []
        for i in range(50):
            start_time = time.time()
            
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
            
            await collector.record_api_usage(api_metric)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate performance metrics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Test analytics generation time
        analytics_start = time.time()
        analytics = await collector.get_endpoint_analytics(hours=1)
        analytics_time = (time.time() - analytics_start) * 1000
        
        print(f"  Average recording time: {avg_time:.2f}ms")
        print(f"  Maximum recording time: {max_time:.2f}ms")
        print(f"  Analytics generation time: {analytics_time:.2f}ms")
        
        # Check SLA compliance
        sla_compliant = max_time < 200 and analytics_time < 200
        
        if sla_compliant:
            print("✓ Performance SLA: PASSED")
        else:
            print(f"✗ Performance SLA: FAILED - Max time: {max_time:.2f}ms, Analytics: {analytics_time:.2f}ms")
        
        return sla_compliant
        
    except Exception as e:
        print(f"✗ Performance SLA: FAILED - {e}")
        return False
    finally:
        await collector.stop_collection()


async def test_concurrent_safety():
    """Test 4: Concurrent operations safety."""
    print("🧪 Test 4: Concurrent Operations Safety")
    
    collector = APIMetricsCollector()
    await collector.start_collection()
    
    try:
        async def record_batch(batch_id: int, count: int):
            for i in range(count):
                api_metric = APIUsageMetric(
                    endpoint=f"/api/v1/concurrent_{batch_id}_{i}",
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
                await collector.record_api_usage(api_metric)
        
        # Run concurrent batches
        start_time = time.time()
        tasks = [record_batch(i, 10) for i in range(5)]  # 5 batches of 10 metrics each
        await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000
        
        # Verify all metrics were recorded
        expected_metrics = 50
        actual_metrics = len(collector.api_usage_metrics)
        
        print(f"  Expected metrics: {expected_metrics}")
        print(f"  Actual metrics: {actual_metrics}")
        print(f"  Total time: {total_time:.2f}ms")
        
        success = actual_metrics >= expected_metrics
        
        if success:
            print("✓ Concurrent operations: PASSED")
        else:
            print(f"✗ Concurrent operations: FAILED - Missing metrics")
        
        return success
        
    except Exception as e:
        print(f"✗ Concurrent operations: FAILED - {e}")
        return False
    finally:
        await collector.stop_collection()


async def test_convenience_functions():
    """Test 5: Convenience functions."""
    print("🧪 Test 5: Convenience Functions")
    
    try:
        # Test convenience functions
        await record_api_request(
            endpoint="/api/v1/convenience_test",
            method=HTTPMethod.GET,
            category=EndpointCategory.HEALTH_CHECK,
            status_code=200,
            response_time_ms=100.0,
            user_id="convenience_user"
        )
        
        await record_user_journey_event(
            user_id="convenience_user",
            session_id="convenience_session",
            journey_stage=UserJourneyStage.REGULAR_USE,
            event_type="api_call",
            endpoint="/api/v1/convenience_test",
            success=True
        )
        
        # Verify global collector was used
        global_collector = get_api_metrics_collector()
        stats = global_collector.get_collection_stats()
        
        success = stats["api_calls_tracked"] >= 1 and stats["journey_events_tracked"] >= 1
        
        if success:
            print("✓ Convenience functions: PASSED")
        else:
            print(f"✗ Convenience functions: FAILED - Stats: {stats}")
        
        return success
        
    except Exception as e:
        print(f"✗ Convenience functions: FAILED - {e}")
        return False


async def main():
    """Run all focused tests."""
    print("🚀 Focused API Metrics Testing")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_analytics_generation,
        test_performance_sla,
        test_concurrent_safety,
        test_convenience_functions
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
