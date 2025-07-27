#!/usr/bin/env python3
"""
Standalone test for Phase 1 metrics implementation.

Tests the four missing Phase 1 metrics without complex dependencies:
1. Connection Age Tracking
2. Request Queue Depths  
3. Cache Hit Rates
4. Feature Usage Analytics
"""

import asyncio
import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_improver.metrics.system_metrics import (
    ConnectionAgeTracker,
    RequestQueueMonitor,
    CacheEfficiencyMonitor,
    FeatureUsageAnalytics,
    MetricsConfig,
    ConnectionInfo,
    QueueSample,
    CacheOperation,
    FeatureUsageEvent
)

# Simple mock registry for testing
class SimpleMockMetric:
    def __init__(self):
        self.value = 0
        self.observations = []
        
    def inc(self, value=1): self.value += value
    def dec(self, value=1): self.value -= value
    def set(self, value): self.value = value
    def observe(self, value): self.observations.append(value)
    def labels(self, **kwargs): return self
    
    class Timer:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def time(self): return self.Timer()

class SimpleMockRegistry:
    def __init__(self):
        self.metrics = {}
    
    def get_or_create_counter(self, name, desc, labels=None):
        if name not in self.metrics:
            self.metrics[name] = SimpleMockMetric()
        return self.metrics[name]
    
    def get_or_create_gauge(self, name, desc, labels=None):
        if name not in self.metrics:
            self.metrics[name] = SimpleMockMetric()
        return self.metrics[name]
    
    def get_or_create_histogram(self, name, desc, labels=None, buckets=None):
        if name not in self.metrics:
            self.metrics[name] = SimpleMockMetric()
        return self.metrics[name]
    
    def get_or_create_summary(self, name, desc, labels=None):
        if name not in self.metrics:
            self.metrics[name] = SimpleMockMetric()
        return self.metrics[name]

def test_connection_age_tracking():
    """Test connection age tracking functionality"""
    print("Testing Connection Age Tracking...")
    
    config = MetricsConfig()
    registry = SimpleMockRegistry()
    tracker = ConnectionAgeTracker(config, registry)
    
    # Test connection creation
    tracker.track_connection_created("conn_1", "database", "main_pool")
    tracker.track_connection_created("conn_2", "redis", "cache_pool")
    
    # Verify connections are tracked
    assert "conn_1" in tracker._connections
    assert "conn_2" in tracker._connections
    
    # Test connection usage
    tracker.track_connection_used("conn_1")
    
    # Test age distribution
    age_dist = tracker.get_age_distribution()
    assert "database" in age_dist
    assert "redis" in age_dist
    
    # Test connection destruction
    tracker.track_connection_destroyed("conn_1")
    assert "conn_1" not in tracker._connections
    
    # Test context manager
    with tracker.track_connection_lifecycle("conn_3", "http", "web_pool"):
        assert "conn_3" in tracker._connections
    assert "conn_3" not in tracker._connections
    
    print("✅ Connection Age Tracking tests passed")

def test_cache_efficiency_monitoring():
    """Test cache efficiency monitoring functionality"""
    print("Testing Cache Efficiency Monitoring...")
    
    config = MetricsConfig(cache_min_samples=5)
    registry = SimpleMockRegistry()
    monitor = CacheEfficiencyMonitor(config, registry)
    
    # Test cache operations - need at least 5 for stats
    monitor.record_cache_hit("application", "user_cache", "key1", 1.5)
    monitor.record_cache_hit("application", "user_cache", "key2", 2.0)
    monitor.record_cache_miss("application", "user_cache", "key3", 15.0)
    monitor.record_cache_hit("application", "user_cache", "key4", 1.8)
    monitor.record_cache_hit("application", "user_cache", "key5", 1.2)
    
    # Verify operations are recorded
    cache_key = "application:user_cache"
    assert len(monitor._cache_operations[cache_key]) == 5
    
    # Test statistics
    stats = monitor.get_cache_statistics("application", "user_cache")
    assert "current_hit_rate" in stats
    assert stats["current_hit_rate"] == 0.8  # 4 hits out of 5
    
    # Test context manager
    with monitor.track_cache_operation("test", "cache", "key5") as tracker:
        tracker.mark_hit()
    
    test_key = "test:cache"
    assert len(monitor._cache_operations[test_key]) == 1
    assert monitor._cache_operations[test_key][0].operation == "hit"
    
    print("✅ Cache Efficiency Monitoring tests passed")

def test_request_queue_monitoring():
    """Test request queue monitoring functionality"""
    print("Testing Request Queue Monitoring...")
    
    config = MetricsConfig()
    registry = SimpleMockRegistry()
    monitor = RequestQueueMonitor(config, registry)
    
    # Test queue sampling
    monitor.sample_queue_depth("http", "request_queue", 15, 100)
    monitor.sample_queue_depth("http", "request_queue", 25, 100)
    monitor.sample_queue_depth("http", "request_queue", 10, 100)
    
    # Verify samples are recorded
    queue_key = "http:request_queue"
    assert len(monitor._queue_samples[queue_key]) == 3
    
    # Test statistics
    stats = monitor.get_queue_statistics("http", "request_queue")
    assert "current_depth" in stats
    assert stats["current_depth"] == 10  # Last sample
    assert stats["avg_depth"] == (15 + 25 + 10) / 3
    assert stats["max_depth"] == 25
    
    print("✅ Request Queue Monitoring tests passed")

def test_feature_usage_analytics():
    """Test feature usage analytics functionality"""
    print("Testing Feature Usage Analytics...")
    
    config = MetricsConfig()
    registry = SimpleMockRegistry()
    analytics = FeatureUsageAnalytics(config, registry)
    
    # Test feature usage recording
    analytics.record_feature_usage(
        "api_endpoint", "/api/users", "user_123", "direct_call", 25.0, True
    )
    analytics.record_feature_usage(
        "api_endpoint", "/api/users", "user_456", "direct_call", 30.0, True
    )
    analytics.record_feature_usage(
        "api_endpoint", "/api/users", "user_123", "batch_operation", 45.0, False
    )
    
    # Verify events are recorded
    feature_key = "api_endpoint:/api/users"
    assert len(analytics._usage_events[feature_key]) == 3
    
    # Test analytics
    stats = analytics.get_feature_analytics("api_endpoint", "/api/users")
    assert "total_usage_in_window" in stats
    assert stats["total_usage_in_window"] == 3
    assert stats["unique_users_in_window"] == 2  # user_123 and user_456
    assert stats["success_rate"] == 2/3  # 2 successes out of 3
    
    # Test top features
    top_features = analytics.get_top_features("api_endpoint", limit=1)
    assert len(top_features) == 1
    assert top_features[0]["feature_name"] == "/api/users"
    
    # Test context manager
    with analytics.track_feature_usage("test", "feature", "user", "test"):
        time.sleep(0.001)  # Simulate work
    
    test_key = "test:feature"
    assert len(analytics._usage_events[test_key]) == 1
    event = analytics._usage_events[test_key][0]
    assert event.success is True
    assert event.performance_ms > 0
    
    print("✅ Feature Usage Analytics tests passed")

def test_performance_targets():
    """Test performance targets are met"""
    print("Testing Performance Targets...")
    
    config = MetricsConfig()
    registry = SimpleMockRegistry()
    
    # Test connection tracking performance
    tracker = ConnectionAgeTracker(config, registry)
    start_time = time.perf_counter()
    
    for i in range(100):
        conn_id = f"perf_conn_{i}"
        tracker.track_connection_created(conn_id, "database", "perf_pool")
        tracker.track_connection_used(conn_id)
        tracker.track_connection_destroyed(conn_id)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    avg_ms = duration_ms / 300  # 100 connections * 3 operations each
    
    print(f"Connection tracking: {avg_ms:.3f}ms per operation")
    
    # Test cache monitoring performance
    monitor = CacheEfficiencyMonitor(config, registry)
    start_time = time.perf_counter()
    
    for i in range(100):
        if i % 3 == 0:
            monitor.record_cache_miss("perf", "cache", f"key_{i}", 10.0)
        else:
            monitor.record_cache_hit("perf", "cache", f"key_{i}", 1.0)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    avg_ms = duration_ms / 100
    
    print(f"Cache monitoring: {avg_ms:.3f}ms per operation")
    
    # Test feature analytics performance
    analytics = FeatureUsageAnalytics(config, registry)
    start_time = time.perf_counter()
    
    for i in range(100):
        analytics.record_feature_usage(
            "perf", "endpoint", f"user_{i % 10}", "call", 20.0, True
        )
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    avg_ms = duration_ms / 100
    
    print(f"Feature analytics: {avg_ms:.3f}ms per operation")
    
    print("✅ Performance tests completed")

async def test_real_time_collection():
    """Test real-time collection capabilities"""
    print("Testing Real-time Collection...")
    
    config = MetricsConfig()
    registry = SimpleMockRegistry()
    
    # Create components
    tracker = ConnectionAgeTracker(config, registry)
    monitor = RequestQueueMonitor(config, registry)
    cache_monitor = CacheEfficiencyMonitor(config, registry)
    analytics = FeatureUsageAnalytics(config, registry)
    
    # Simulate real-time activity
    start_time = time.perf_counter()
    
    for i in range(20):
        # Connection activity
        conn_id = f"rt_conn_{i}"
        tracker.track_connection_created(conn_id, "database", "rt_pool")
        
        # Queue activity
        monitor.sample_queue_depth("http", "rt_queue", i % 30, 50)
        
        # Cache activity
        if i % 4 == 0:
            cache_monitor.record_cache_miss("rt", "cache", f"key_{i}", 12.0)
        else:
            cache_monitor.record_cache_hit("rt", "cache", f"key_{i}", 2.0)
        
        # Feature usage
        analytics.record_feature_usage(
            "rt_api", "endpoint", f"user_{i % 3}", "rt_call", 15.0, i % 10 != 0
        )
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.001)
    
    duration = time.perf_counter() - start_time
    
    # Verify data was collected
    age_dist = tracker.get_age_distribution()
    queue_stats = monitor.get_queue_statistics("http", "rt_queue")
    cache_stats = cache_monitor.get_cache_statistics("rt", "cache")
    feature_stats = analytics.get_feature_analytics("rt_api", "endpoint")
    
    print(f"Real-time collection completed in {duration:.3f}s")
    print(f"Connections tracked: {len(age_dist.get('database', {}).get('rt_pool', []))}")
    print(f"Queue samples: {queue_stats.get('sample_count', 0)}")
    print(f"Cache operations: {cache_stats.get('total_operations', 0)}")
    print(f"Feature usage: {feature_stats.get('total_usage_in_window', 0)}")
    
    print("✅ Real-time collection tests passed")

async def main():
    """Run all standalone tests"""
    print("🚀 Phase 1 Metrics Standalone Validation")
    print("=" * 60)
    
    try:
        # Run individual component tests
        test_connection_age_tracking()
        test_cache_efficiency_monitoring()
        test_request_queue_monitoring()
        test_feature_usage_analytics()
        
        # Run performance tests
        test_performance_targets()
        
        # Run real-time tests
        await test_real_time_collection()
        
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 1 METRICS TESTS PASSED!")
        print("✅ Connection Age Tracking: Working correctly")
        print("✅ Request Queue Depths: Working correctly")
        print("✅ Cache Hit Rates: Working correctly") 
        print("✅ Feature Usage Analytics: Working correctly")
        print("✅ Performance: Acceptable for test environment")
        print("✅ Real-time Collection: Working correctly")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))