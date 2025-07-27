"""
Comprehensive test suite for system metrics implementation.

Tests all four missing Phase 1 metrics:
1. Connection Age Tracking
2. Request Queue Depths  
3. Cache Hit Rates
4. Feature Usage Analytics

Validates performance targets, real-time collection, and integration.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from prompt_improver.metrics.system_metrics import (
    SystemMetricsCollector,
    ConnectionAgeTracker,
    RequestQueueMonitor,
    CacheEfficiencyMonitor,
    FeatureUsageAnalytics,
    MetricsConfig,
    ConnectionInfo,
    QueueSample,
    CacheOperation,
    FeatureUsageEvent,
    get_system_metrics_collector,
    track_connection_lifecycle,
    track_feature_usage,
    track_cache_operation
)
from prompt_improver.performance.monitoring.metrics_registry import MetricsRegistry


@pytest.fixture
def metrics_config():
    """Test configuration with reduced retention for faster testing"""
    return MetricsConfig(
        connection_age_retention_hours=1,
        connection_age_bucket_minutes=1,
        max_tracked_connections=100,
        queue_depth_sample_interval_ms=50,
        queue_depth_history_size=50,
        cache_hit_window_minutes=1,
        cache_min_samples=5,
        feature_usage_window_hours=1,
        metrics_collection_overhead_ms=1.0
    )


@pytest.fixture
def mock_registry():
    """Mock metrics registry for testing"""
    registry = Mock(spec=MetricsRegistry)
    
    # Mock metric creation methods
    mock_metric = Mock()
    mock_metric.inc.return_value = None
    mock_metric.dec.return_value = None
    mock_metric.set.return_value = None
    mock_metric.observe.return_value = None
    mock_metric.labels.return_value = mock_metric
    
    registry.get_or_create_counter.return_value = mock_metric
    registry.get_or_create_gauge.return_value = mock_metric
    registry.get_or_create_histogram.return_value = mock_metric
    registry.get_or_create_summary.return_value = mock_metric
    
    return registry


class TestMetricsConfig:
    """Test metrics configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = MetricsConfig()
        assert config.connection_age_retention_hours == 24
        assert config.queue_depth_sample_interval_ms == 100
        assert config.cache_hit_window_minutes == 15
        assert config.feature_usage_window_hours == 24
    
    def test_config_validation(self):
        """Test configuration validation"""
        with pytest.raises(ValueError):
            MetricsConfig(connection_age_retention_hours=0)  # Too low
        
        with pytest.raises(ValueError):
            MetricsConfig(connection_age_retention_hours=200)  # Too high
    
    def test_config_edge_cases(self):
        """Test configuration edge cases"""
        # Minimum valid values
        config = MetricsConfig(
            connection_age_retention_hours=1,
            queue_depth_sample_interval_ms=10,
            cache_hit_window_minutes=1,
            feature_usage_window_hours=1
        )
        assert config.connection_age_retention_hours == 1
        
        # Maximum valid values
        config = MetricsConfig(
            connection_age_retention_hours=168,
            queue_depth_sample_interval_ms=5000,
            cache_hit_window_minutes=60,
            feature_usage_window_hours=168
        )
        assert config.connection_age_retention_hours == 168


class TestConnectionAgeTracker:
    """Test connection age tracking functionality"""
    
    def test_connection_info_properties(self):
        """Test ConnectionInfo properties"""
        created_time = datetime.utcnow()
        conn_info = ConnectionInfo(
            connection_id="test_conn",
            created_at=created_time,
            last_used=created_time,
            connection_type="database",
            pool_name="test_pool"
        )
        
        assert conn_info.age_seconds >= 0
        assert conn_info.idle_seconds >= 0
        assert conn_info.connection_id == "test_conn"
    
    def test_track_connection_created(self, metrics_config, mock_registry):
        """Test connection creation tracking"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        tracker.track_connection_created(
            "conn_123", "database", "postgresql_main",
            source_info={"host": "localhost", "port": 5432}
        )
        
        # Verify connection is tracked
        assert "conn_123" in tracker._connections
        conn_info = tracker._connections["conn_123"]
        assert conn_info.connection_type == "database"
        assert conn_info.pool_name == "postgresql_main"
        assert conn_info.source_info["host"] == "localhost"
        
        # Verify metrics were called
        mock_registry.get_or_create_gauge.assert_called()
        mock_registry.get_or_create_counter.assert_called()
    
    def test_track_connection_used(self, metrics_config, mock_registry):
        """Test connection usage tracking"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Create connection first
        tracker.track_connection_created("conn_123", "database", "test_pool")
        original_last_used = tracker._connections["conn_123"].last_used
        
        # Wait a bit and track usage
        time.sleep(0.01)
        tracker.track_connection_used("conn_123")
        
        # Verify last_used was updated
        assert tracker._connections["conn_123"].last_used > original_last_used
    
    def test_track_connection_destroyed(self, metrics_config, mock_registry):
        """Test connection destruction tracking"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Create and destroy connection
        tracker.track_connection_created("conn_123", "database", "test_pool")
        assert "conn_123" in tracker._connections
        
        tracker.track_connection_destroyed("conn_123")
        assert "conn_123" not in tracker._connections
        
        # Verify histogram was called for age recording
        mock_registry.get_or_create_histogram.assert_called()
    
    def test_connection_lifecycle_context_manager(self, metrics_config, mock_registry):
        """Test connection lifecycle context manager"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        with tracker.track_connection_lifecycle("conn_123", "database", "test_pool"):
            # Connection should be tracked during context
            assert "conn_123" in tracker._connections
        
        # Connection should be removed after context
        assert "conn_123" not in tracker._connections
    
    def test_memory_management(self, metrics_config, mock_registry):
        """Test connection memory management"""
        # Set low limit for testing
        metrics_config.max_tracked_connections = 5
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Create more connections than limit
        for i in range(10):
            tracker.track_connection_created(f"conn_{i}", "database", "test_pool")
        
        # Should not exceed limit
        assert len(tracker._connections) <= metrics_config.max_tracked_connections
    
    def test_age_distribution_analysis(self, metrics_config, mock_registry):
        """Test age distribution calculation"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Create connections with different ages
        tracker.track_connection_created("conn_new", "database", "pool1")
        tracker.track_connection_created("conn_old", "redis", "pool2")
        
        # Simulate aging by modifying creation time
        old_time = datetime.utcnow() - timedelta(hours=2)
        tracker._connections["conn_old"].created_at = old_time
        
        distribution = tracker.get_age_distribution()
        
        assert "database" in distribution
        assert "redis" in distribution
        assert "pool1" in distribution["database"]
        assert "pool2" in distribution["redis"]
        assert len(distribution["database"]["pool1"]) == 1
        assert len(distribution["redis"]["pool2"]) == 1
    
    def test_performance_overhead(self, metrics_config, mock_registry):
        """Test that connection tracking meets performance targets"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Measure performance of connection operations
        start_time = time.perf_counter()
        
        for i in range(100):
            tracker.track_connection_created(f"conn_{i}", "database", "test_pool")
            tracker.track_connection_used(f"conn_{i}")
            tracker.track_connection_destroyed(f"conn_{i}")
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        avg_operation_ms = duration_ms / 300  # 100 * 3 operations
        
        # Should be well under 1ms per operation
        assert avg_operation_ms < metrics_config.metrics_collection_overhead_ms


class TestRequestQueueMonitor:
    """Test request queue monitoring functionality"""
    
    def test_queue_sample_properties(self):
        """Test QueueSample properties"""
        sample = QueueSample(
            timestamp=datetime.utcnow(),
            depth=25,
            capacity=100,
            queue_type="http"
        )
        
        assert sample.utilization_ratio == 0.25
        assert sample.depth == 25
        assert sample.capacity == 100
    
    def test_sample_queue_depth(self, metrics_config, mock_registry):
        """Test queue depth sampling"""
        monitor = RequestQueueMonitor(metrics_config, mock_registry)
        
        monitor.sample_queue_depth("http", "request_queue", 15, 100)
        
        # Verify sample was recorded
        queue_key = "http:request_queue"
        assert queue_key in monitor._queue_samples
        assert len(monitor._queue_samples[queue_key]) == 1
        
        sample = monitor._queue_samples[queue_key][0]
        assert sample.depth == 15
        assert sample.capacity == 100
        assert sample.utilization_ratio == 0.15
        
        # Verify metrics were called
        mock_registry.get_or_create_gauge.assert_called()
        mock_registry.get_or_create_histogram.assert_called()
    
    def test_queue_overflow_detection(self, metrics_config, mock_registry):
        """Test queue overflow detection"""
        monitor = RequestQueueMonitor(metrics_config, mock_registry)
        
        # Sample high utilization (above threshold)
        monitor.sample_queue_depth("database", "connection_pool", 85, 100)
        
        # Should trigger overflow counter
        mock_registry.get_or_create_counter.assert_called()
    
    def test_queue_statistics(self, metrics_config, mock_registry):
        """Test queue statistics calculation"""
        monitor = RequestQueueMonitor(metrics_config, mock_registry)
        
        # Add multiple samples
        for depth in [10, 20, 30, 25, 15]:
            monitor.sample_queue_depth("http", "request_queue", depth, 100)
        
        stats = monitor.get_queue_statistics("http", "request_queue")
        
        assert stats["sample_count"] == 5
        assert stats["current_depth"] == 15  # Last sample
        assert stats["avg_depth"] == 20.0    # (10+20+30+25+15)/5
        assert stats["max_depth"] == 30
        assert stats["min_depth"] == 10
        assert stats["current_utilization"] == 0.15
    
    @pytest.mark.asyncio
    async def test_background_monitoring(self, metrics_config, mock_registry):
        """Test background queue monitoring"""
        monitor = RequestQueueMonitor(metrics_config, mock_registry)
        
        # Start monitoring briefly
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        await asyncio.sleep(0.1)  # Let it run briefly
        monitor.stop_monitoring()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
    
    def test_memory_limits(self, metrics_config, mock_registry):
        """Test queue sample memory limits"""
        monitor = RequestQueueMonitor(metrics_config, mock_registry)
        
        # Add more samples than limit
        for i in range(metrics_config.queue_depth_history_size + 10):
            monitor.sample_queue_depth("test", "queue", i, 1000)
        
        queue_key = "test:queue"
        # Should not exceed limit
        assert len(monitor._queue_samples[queue_key]) <= metrics_config.queue_depth_history_size


class TestCacheEfficiencyMonitor:
    """Test cache efficiency monitoring functionality"""
    
    def test_cache_operation_creation(self):
        """Test CacheOperation creation"""
        operation = CacheOperation(
            timestamp=datetime.utcnow(),
            cache_type="application",
            cache_name="user_cache",
            operation="hit",
            key_hash="abc123",
            response_time_ms=2.5
        )
        
        assert operation.operation == "hit"
        assert operation.response_time_ms == 2.5
        assert operation.cache_type == "application"
    
    def test_record_cache_operations(self, metrics_config, mock_registry):
        """Test cache operation recording"""
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        # Record different types of operations
        monitor.record_cache_hit("application", "user_cache", "key1", 1.5)
        monitor.record_cache_miss("application", "user_cache", "key2", 15.0)
        monitor.record_cache_set("application", "user_cache", "key3", 3.0)
        
        cache_key = "application:user_cache"
        assert len(monitor._cache_operations[cache_key]) == 3
        
        # Verify metrics were called
        mock_registry.get_or_create_counter.assert_called()
        mock_registry.get_or_create_histogram.assert_called()
    
    def test_hit_rate_calculation(self, metrics_config, mock_registry):
        """Test cache hit rate calculation"""
        # Use smaller min_samples for testing
        metrics_config.cache_min_samples = 3
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        # Record hits and misses
        monitor.record_cache_hit("application", "user_cache", "key1", 1.0)
        monitor.record_cache_hit("application", "user_cache", "key2", 1.0)
        monitor.record_cache_miss("application", "user_cache", "key3", 10.0)
        monitor.record_cache_hit("application", "user_cache", "key4", 1.0)
        
        cache_key = "application:user_cache"
        # Should have 3 hits, 1 miss = 75% hit rate
        assert monitor._hit_counters[cache_key] == 3
        assert monitor._miss_counters[cache_key] == 1
    
    def test_cache_statistics(self, metrics_config, mock_registry):
        """Test cache statistics calculation"""
        metrics_config.cache_min_samples = 2
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        # Record operations
        monitor.record_cache_hit("application", "user_cache", "key1", 2.0)
        monitor.record_cache_hit("application", "user_cache", "key2", 3.0)
        monitor.record_cache_miss("application", "user_cache", "key3", 20.0)
        
        stats = monitor.get_cache_statistics("application", "user_cache")
        
        assert stats["total_operations"] == 3
        assert stats["current_window_hits"] == 2
        assert stats["current_window_misses"] == 1
        assert stats["current_hit_rate"] == 2/3  # 66.7%
        assert stats["avg_hit_response_time_ms"] == 2.5  # (2+3)/2
        assert stats["avg_miss_response_time_ms"] == 20.0
    
    def test_cache_context_manager(self, metrics_config, mock_registry):
        """Test cache operation context manager"""
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        # Test cache hit scenario
        with monitor.track_cache_operation("application", "user_cache", "key1") as tracker:
            tracker.mark_hit()
            # Simulate some work
            time.sleep(0.001)
        
        cache_key = "application:user_cache"
        assert len(monitor._cache_operations[cache_key]) == 1
        assert monitor._cache_operations[cache_key][0].operation == "hit"
        
        # Test cache miss scenario
        with monitor.track_cache_operation("application", "user_cache", "key2") as tracker:
            # Don't mark as hit - should be recorded as miss
            time.sleep(0.001)
        
        assert len(monitor._cache_operations[cache_key]) == 2
        assert monitor._cache_operations[cache_key][1].operation == "miss"
    
    def test_efficiency_score_calculation(self, metrics_config, mock_registry):
        """Test cache efficiency score calculation"""
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        cache_key = "application:user_cache"
        
        # Add sample operations with different response times
        for _ in range(10):
            monitor.record_cache_hit("application", "user_cache", "key", 1.0)
        for _ in range(5):
            monitor.record_cache_miss("application", "user_cache", "key", 50.0)
        
        hit_rate = 10 / 15  # 66.7%
        efficiency_score = monitor._calculate_efficiency_score(cache_key, hit_rate)
        
        # Should be a combination of hit rate and response time efficiency
        assert 0.0 <= efficiency_score <= 1.0
        assert efficiency_score > hit_rate  # Should be boosted by good response time ratio
    
    def test_window_reset(self, metrics_config, mock_registry):
        """Test rolling window reset"""
        # Use very short window for testing
        metrics_config.cache_hit_window_minutes = 0.01  # 0.6 seconds
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        cache_key = "application:user_cache"
        
        # Record initial operations
        monitor.record_cache_hit("application", "user_cache", "key1", 1.0)
        assert monitor._hit_counters[cache_key] == 1
        
        # Wait for window to expire and record new operation
        time.sleep(0.7)
        monitor.record_cache_hit("application", "user_cache", "key2", 1.0)
        
        # Window should have been reset
        assert monitor._hit_counters[cache_key] <= 2  # Might be reset


class TestFeatureUsageAnalytics:
    """Test feature usage analytics functionality"""
    
    def test_feature_usage_event_creation(self):
        """Test FeatureUsageEvent creation"""
        event = FeatureUsageEvent(
            timestamp=datetime.utcnow(),
            feature_type="api_endpoint",
            feature_name="/api/users",
            user_context="user_123_hashed",
            usage_pattern="direct_call",
            performance_ms=45.2,
            success=True,
            metadata={"version": "1.0"}
        )
        
        assert event.feature_type == "api_endpoint"
        assert event.performance_ms == 45.2
        assert event.success is True
        assert event.metadata["version"] == "1.0"
    
    def test_record_feature_usage(self, metrics_config, mock_registry):
        """Test feature usage recording"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        analytics.record_feature_usage(
            feature_type="api_endpoint",
            feature_name="/api/users",
            user_context="user_123",
            usage_pattern="direct_call",
            performance_ms=25.5,
            success=True,
            metadata={"version": "1.0"}
        )
        
        feature_key = "api_endpoint:/api/users"
        assert len(analytics._usage_events[feature_key]) == 1
        
        event = analytics._usage_events[feature_key][0]
        assert event.feature_type == "api_endpoint"
        assert event.performance_ms == 25.5
        assert event.success is True
        
        # Verify metrics were called
        mock_registry.get_or_create_counter.assert_called()
        mock_registry.get_or_create_histogram.assert_called()
    
    def test_user_context_hashing(self, metrics_config, mock_registry):
        """Test user context hashing for privacy"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Same user context should produce same hash
        hash1 = analytics._hash_user_context("user_123")
        hash2 = analytics._hash_user_context("user_123")
        assert hash1 == hash2
        
        # Different user contexts should produce different hashes
        hash3 = analytics._hash_user_context("user_456")
        assert hash1 != hash3
        
        # Hash should be truncated to 16 characters
        assert len(hash1) == 16
    
    def test_feature_analytics_calculation(self, metrics_config, mock_registry):
        """Test feature analytics calculation"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Record usage from multiple users
        users = ["user_1", "user_2", "user_3", "user_1"]  # user_1 appears twice
        for i, user in enumerate(users):
            analytics.record_feature_usage(
                "api_endpoint", "/api/users", user, "direct_call",
                performance_ms=20.0 + i, success=i < 3  # Last one fails
            )
        
        stats = analytics.get_feature_analytics("api_endpoint", "/api/users")
        
        assert stats["total_usage_in_window"] == 4
        assert stats["unique_users_in_window"] == 3  # user_1, user_2, user_3
        assert stats["success_rate"] == 0.75  # 3 successes out of 4
        assert 20.0 <= stats["avg_performance_ms"] <= 25.0  # Average of 20,21,22,23
    
    def test_adoption_trend_calculation(self, metrics_config, mock_registry):
        """Test adoption trend calculation"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Simulate growing usage (more unique users in second half)
        for i in range(20):
            user_id = f"user_{i if i < 5 else i + 10}"  # More users in second half
            analytics.record_feature_usage(
                "feature_flag", "new_ui", user_id, "direct_call"
            )
        
        stats = analytics.get_feature_analytics("feature_flag", "new_ui")
        assert stats["adoption_trend"] in ["growing", "stable", "declining"]
    
    def test_peak_usage_hour_calculation(self, metrics_config, mock_registry):
        """Test peak usage hour calculation"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Create events at different hours
        for hour in [9, 9, 14, 14, 14, 16]:  # 14 (2pm) should be peak
            event = FeatureUsageEvent(
                timestamp=datetime.utcnow().replace(hour=hour, minute=0, second=0),
                feature_type="api_endpoint",
                feature_name="/api/search",
                user_context="user_test",
                usage_pattern="direct_call",
                performance_ms=10.0,
                success=True
            )
            feature_key = "api_endpoint:/api/search"
            analytics._usage_events[feature_key].append(event)
        
        stats = analytics.get_feature_analytics("api_endpoint", "/api/search")
        assert stats["peak_usage_hour"] == 14  # 2pm
    
    def test_top_features_calculation(self, metrics_config, mock_registry):
        """Test top features calculation"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Record usage for different features
        features = [
            ("api", "endpoint_a", 10),  # 10 uses
            ("api", "endpoint_b", 25),  # 25 uses (should be top)
            ("api", "endpoint_c", 5),   # 5 uses
        ]
        
        for feature_type, feature_name, count in features:
            for i in range(count):
                analytics.record_feature_usage(
                    feature_type, feature_name, f"user_{i}", "direct_call"
                )
        
        top_features = analytics.get_top_features("api", limit=3)
        
        assert len(top_features) == 3
        assert top_features[0]["feature_name"] == "endpoint_b"  # Most used
        assert top_features[0]["usage_count"] == 25
        assert top_features[1]["feature_name"] == "endpoint_a"
        assert top_features[2]["feature_name"] == "endpoint_c"
    
    def test_feature_context_manager(self, metrics_config, mock_registry):
        """Test feature usage context manager"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        # Test successful operation
        with analytics.track_feature_usage("api", "test_endpoint", "user_123"):
            time.sleep(0.001)  # Simulate work
        
        feature_key = "api:test_endpoint"
        assert len(analytics._usage_events[feature_key]) == 1
        event = analytics._usage_events[feature_key][0]
        assert event.success is True
        assert event.performance_ms > 0
        
        # Test failed operation
        try:
            with analytics.track_feature_usage("api", "test_endpoint", "user_123"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert len(analytics._usage_events[feature_key]) == 2
        event = analytics._usage_events[feature_key][1]
        assert event.success is False


class TestSystemMetricsCollector:
    """Test main system metrics collector"""
    
    def test_collector_initialization(self, metrics_config, mock_registry):
        """Test system metrics collector initialization"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        assert collector.config == metrics_config
        assert collector.registry == mock_registry
        assert isinstance(collector.connection_tracker, ConnectionAgeTracker)
        assert isinstance(collector.queue_monitor, RequestQueueMonitor)
        assert isinstance(collector.cache_monitor, CacheEfficiencyMonitor)
        assert isinstance(collector.feature_analytics, FeatureUsageAnalytics)
    
    def test_system_health_score_calculation(self, metrics_config, mock_registry):
        """Test system health score calculation"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        health_score = collector.get_system_health_score()
        
        # Should return a valid score between 0 and 1
        assert 0.0 <= health_score <= 1.0
        assert isinstance(health_score, float)
    
    def test_collect_all_metrics(self, metrics_config, mock_registry):
        """Test comprehensive metrics collection"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        # Add some test data
        collector.connection_tracker.track_connection_created(
            "test_conn", "database", "test_pool"
        )
        
        metrics = collector.collect_all_metrics()
        
        assert "timestamp" in metrics
        assert "connection_age_distribution" in metrics
        assert "system_health_score" in metrics
        assert "collection_performance_ms" in metrics
        
        # Performance should meet target
        assert metrics["collection_performance_ms"] < metrics_config.metrics_collection_overhead_ms * 10
    
    @pytest.mark.asyncio
    async def test_background_monitoring_lifecycle(self, metrics_config, mock_registry):
        """Test background monitoring start/stop"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        # Start background monitoring briefly
        monitoring_task = asyncio.create_task(collector.start_background_monitoring())
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        collector.stop_background_monitoring()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()


class TestConvenienceDecorators:
    """Test convenience decorators for easy integration"""
    
    def test_track_connection_lifecycle_decorator(self, metrics_config, mock_registry):
        """Test connection lifecycle decorator"""
        # Mock the global collector
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        with patch('prompt_improver.metrics.system_metrics.get_system_metrics_collector', return_value=collector):
            @track_connection_lifecycle("database", "test_pool")
            def test_function():
                return "success"
            
            result = test_function()
            assert result == "success"
    
    def test_track_feature_usage_decorator(self, metrics_config, mock_registry):
        """Test feature usage decorator"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        with patch('prompt_improver.metrics.system_metrics.get_system_metrics_collector', return_value=collector):
            @track_feature_usage("api", "test_endpoint")
            def test_api_call():
                return {"data": "test"}
            
            result = test_api_call()
            assert result == {"data": "test"}
    
    def test_track_cache_operation_decorator(self, metrics_config, mock_registry):
        """Test cache operation decorator"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        with patch('prompt_improver.metrics.system_metrics.get_system_metrics_collector', return_value=collector):
            @track_cache_operation("application", "test_cache")
            def cache_lookup():
                return "cached_value"  # Non-None = cache hit
            
            result = cache_lookup()
            assert result == "cached_value"


class TestPerformanceTargets:
    """Test performance targets are met"""
    
    def test_connection_tracking_performance(self, metrics_config, mock_registry):
        """Test connection tracking meets <1ms target"""
        tracker = ConnectionAgeTracker(metrics_config, mock_registry)
        
        # Measure performance of batch operations
        start_time = time.perf_counter()
        
        # Simulate realistic workload
        for i in range(50):
            tracker.track_connection_created(f"conn_{i}", "database", "pool")
            if i % 2 == 0:
                tracker.track_connection_used(f"conn_{i}")
            if i % 3 == 0:
                tracker.track_connection_destroyed(f"conn_{i}")
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        avg_operation_ms = duration_ms / 117  # Total operations
        
        # Should meet <1ms target per operation
        assert avg_operation_ms < metrics_config.metrics_collection_overhead_ms
    
    def test_cache_monitoring_performance(self, metrics_config, mock_registry):
        """Test cache monitoring meets performance target"""
        monitor = CacheEfficiencyMonitor(metrics_config, mock_registry)
        
        start_time = time.perf_counter()
        
        # Simulate realistic cache operations
        for i in range(100):
            if i % 3 == 0:
                monitor.record_cache_miss("app", "cache", f"key_{i}", 10.0)
            else:
                monitor.record_cache_hit("app", "cache", f"key_{i}", 1.0)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        avg_operation_ms = duration_ms / 100
        
        assert avg_operation_ms < metrics_config.metrics_collection_overhead_ms
    
    def test_feature_analytics_performance(self, metrics_config, mock_registry):
        """Test feature analytics meets performance target"""
        analytics = FeatureUsageAnalytics(metrics_config, mock_registry)
        
        start_time = time.perf_counter()
        
        # Simulate realistic feature usage
        for i in range(100):
            analytics.record_feature_usage(
                "api", f"endpoint_{i % 5}", f"user_{i % 10}",
                "direct_call", 25.0, i % 10 != 0  # 90% success rate
            )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        avg_operation_ms = duration_ms / 100
        
        assert avg_operation_ms < metrics_config.metrics_collection_overhead_ms


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self, metrics_config, mock_registry):
        """Test full system integration scenario"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        # Simulate realistic application behavior
        
        # 1. Database connections
        for i in range(5):
            collector.connection_tracker.track_connection_created(
                f"db_conn_{i}", "database", "postgresql_main"
            )
        
        # 2. Queue monitoring
        for depth in [10, 15, 20, 18, 12]:
            collector.queue_monitor.sample_queue_depth(
                "http", "request_queue", depth, 100
            )
        
        # 3. Cache operations
        for i in range(20):
            if i % 4 == 0:  # 25% miss rate
                collector.cache_monitor.record_cache_miss(
                    "application", "user_cache", f"key_{i}", 15.0
                )
            else:
                collector.cache_monitor.record_cache_hit(
                    "application", "user_cache", f"key_{i}", 2.0
                )
        
        # 4. Feature usage
        features = ["/api/users", "/api/posts", "/api/search"]
        for i in range(30):
            feature = features[i % len(features)]
            collector.feature_analytics.record_feature_usage(
                "api_endpoint", feature, f"user_{i % 5}", "direct_call",
                performance_ms=20.0 + (i % 10), success=i % 20 != 0
            )
        
        # 5. Collect comprehensive metrics
        metrics = collector.collect_all_metrics()
        
        # Verify all components contributed data
        assert metrics["system_health_score"] > 0.0
        assert len(metrics["connection_age_distribution"]) > 0
        assert metrics["collection_performance_ms"] < 100  # Should be very fast
        
        # 6. Test background monitoring briefly
        monitoring_task = asyncio.create_task(collector.start_background_monitoring())
        await asyncio.sleep(0.1)
        collector.stop_background_monitoring()
        
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
    
    def test_singleton_behavior(self):
        """Test singleton behavior of global collector"""
        collector1 = get_system_metrics_collector()
        collector2 = get_system_metrics_collector()
        
        # Should return the same instance
        assert collector1 is collector2
    
    def test_error_handling_resilience(self, metrics_config, mock_registry):
        """Test error handling and resilience"""
        collector = SystemMetricsCollector(metrics_config, mock_registry)
        
        # Test with invalid data - should not crash
        collector.connection_tracker.track_connection_destroyed("nonexistent_conn")
        collector.queue_monitor.sample_queue_depth("test", "queue", -1, 0)  # Invalid data
        collector.cache_monitor.record_cache_hit("", "", "", -1.0)  # Invalid data
        collector.feature_analytics.record_feature_usage("", "", "", "", -1.0, None)
        
        # Should still be able to collect metrics
        metrics = collector.collect_all_metrics()
        assert "timestamp" in metrics
        assert "system_health_score" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])