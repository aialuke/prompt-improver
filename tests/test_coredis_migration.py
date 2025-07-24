"""
Comprehensive tests for Redis to Coredis migration

This test suite validates:
1. Analytics components initialize correctly with coredis
2. Redis integration for metrics storage works
3. Real-time analytics data collection functions
4. Performance monitoring Redis integration
5. Canary testing Redis operations work with coredis
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

import coredis
import numpy as np

# Import the components we're testing
from src.prompt_improver.performance.analytics.real_time_analytics import (
    EnhancedRealTimeAnalyticsService,
    RealTimeAnalyticsService,
    AnalyticsEvent,
    EventType,
    RealTimeMetrics,
    get_real_time_analytics_service
)
from src.prompt_improver.performance.optimization.async_optimizer import (
    IntelligentCache,
    CacheConfig,
    AsyncOptimizer,
    get_async_optimizer
)
from src.prompt_improver.performance.testing.canary_testing import (
    CanaryTestingService,
    CanaryMetrics
)


class TestCoredisAnalyticsMigration:
    """Test migration of analytics components to coredis"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_db_session = AsyncMock()
        self.mock_redis_client = AsyncMock(spec=coredis.Redis)
        
        # Configure mock redis responses
        self.mock_redis_client.ping.return_value = True
        self.mock_redis_client.get.return_value = None
        self.mock_redis_client.set.return_value = True
        self.mock_redis_client.setex.return_value = True
        self.mock_redis_client.delete.return_value = 1

    @pytest.mark.asyncio
    async def test_enhanced_analytics_service_initialization(self):
        """Test EnhancedRealTimeAnalyticsService initializes with coredis client"""
        service = EnhancedRealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        assert service.redis_client is self.mock_redis_client
        assert isinstance(service.redis_client, (AsyncMock, coredis.Redis))
        
    @pytest.mark.asyncio
    async def test_analytics_service_initialization(self):
        """Test backward compatible RealTimeAnalyticsService with coredis"""
        service = RealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        assert service.redis_client is self.mock_redis_client
        assert isinstance(service.redis_client, (AsyncMock, coredis.Redis))

    @pytest.mark.asyncio
    async def test_analytics_event_emission_with_redis_storage(self):
        """Test analytics event emission stores correctly in Redis"""
        service = EnhancedRealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        # Create test event
        event = AnalyticsEvent(
            event_id="test_event_001",
            event_type=EventType.CONVERSION_EVENT,
            experiment_id="exp_001",
            user_id="user_123",
            session_id="session_456",
            timestamp=datetime.utcnow(),
            data={"conversion_value": 50.0}
        )
        
        # Emit event
        await service.emit_event(event)
        
        # Verify event is in buffer
        assert len(service.event_buffer) > 0
        assert service.event_buffer[-1] == event

    @pytest.mark.asyncio
    async def test_window_results_storage_uses_coredis(self):
        """Test window results are stored using coredis operations"""
        service = EnhancedRealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        # Create mock window
        from src.prompt_improver.performance.analytics.real_time_analytics import StreamWindow
        window = StreamWindow(
            window_id="window_test_001",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=30),
            experiment_id="exp_001",
            events_count=10,
            metrics={"conversion_rate": 0.15}
        )
        
        # Store window results
        await service._store_window_results(window)
        
        # Verify Redis setex was called with correct parameters
        self.mock_redis_client.setex.assert_called_once()
        call_args = self.mock_redis_client.setex.call_args
        
        # Check the key format
        key = call_args[0][0]
        assert key.startswith(f"analytics_window:{window.experiment_id}:")
        
        # Check TTL (7 days)
        ttl = call_args[0][1]
        assert ttl == 86400 * 7
        
        # Check data is JSON serialized
        data = call_args[0][2]
        parsed_data = json.loads(data)
        assert parsed_data["window_id"] == window.window_id
        assert parsed_data["events_count"] == 10

    @pytest.mark.asyncio
    async def test_websocket_broadcast_with_redis(self):
        """Test WebSocket broadcast functionality with Redis"""
        with patch('src.prompt_improver.performance.analytics.real_time_analytics.publish_experiment_update') as mock_publish:
            service = EnhancedRealTimeAnalyticsService(
                db_session=self.mock_db_session,
                redis_client=self.mock_redis_client
            )
            
            from src.prompt_improver.performance.analytics.real_time_analytics import StreamWindow
            window = StreamWindow(
                window_id="window_test_002",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(seconds=30),
                experiment_id="exp_002",
                events_count=5,
                metrics={"conversion_rate": 0.12}
            )
            
            await service._broadcast_window_results(window)
            
            # Verify publish_experiment_update was called with correct parameters
            mock_publish.assert_called_once()
            call_args = mock_publish.call_args[0]
            
            assert call_args[0] == "exp_002"  # experiment_id
            assert call_args[1]["type"] == "window_update"
            assert call_args[2] is self.mock_redis_client  # redis client

    @pytest.mark.asyncio
    async def test_get_singleton_service_with_coredis(self):
        """Test singleton service getter with coredis client"""
        # Clear any existing singleton
        from src.prompt_improver.performance.analytics import real_time_analytics
        real_time_analytics._real_time_service = None
        
        service = await get_real_time_analytics_service(
            self.mock_db_session,
            self.mock_redis_client
        )
        
        assert isinstance(service, RealTimeAnalyticsService)
        assert service.redis_client is self.mock_redis_client


class TestCoredisOptimizerMigration:
    """Test migration of async optimizer components to coredis"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_redis_client = AsyncMock(spec=coredis.Redis)
        self.mock_redis_client.ping = AsyncMock(return_value=True)
        self.mock_redis_client.get = AsyncMock(return_value=None)
        self.mock_redis_client.setex = AsyncMock(return_value=True)
        self.mock_redis_client.delete = AsyncMock(return_value=1)

    @pytest.mark.asyncio
    async def test_intelligent_cache_initialization(self):
        """Test IntelligentCache initializes with coredis"""
        config = CacheConfig(max_size=100, ttl_seconds=300)
        cache = IntelligentCache(config)
        
        # Mock the Redis initialization
        with patch('src.prompt_improver.performance.optimization.async_optimizer.coredis.Redis') as mock_redis_constructor:
            mock_redis_constructor.return_value = self.mock_redis_client
            # Ensure ping returns a proper awaitable
            self.mock_redis_client.ping = AsyncMock(return_value=True)
            
            await cache.ensure_redis_connection()
            
            # Verify Redis was initialized with correct parameters
            mock_redis_constructor.assert_called_once_with(
                host="localhost", 
                port=6379, 
                decode_responses=True
            )
            assert cache.redis_client is self.mock_redis_client

    @pytest.mark.asyncio
    async def test_cache_get_operations_use_coredis(self):
        """Test cache get operations use coredis methods"""
        config = CacheConfig(max_size=100, ttl_seconds=300)
        cache = IntelligentCache(config)
        cache.redis_client = self.mock_redis_client
        
        # Test cache miss in local cache, hit in Redis
        self.mock_redis_client.get = AsyncMock(return_value=json.dumps({"test": "value"}))
        
        result = await cache.get("test_key")
        
        # Verify Redis get was called
        self.mock_redis_client.get.assert_called_once_with("test_key")
        assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_cache_set_operations_use_coredis(self):
        """Test cache set operations use coredis methods"""
        config = CacheConfig(max_size=100, ttl_seconds=300)
        cache = IntelligentCache(config)
        cache.redis_client = self.mock_redis_client
        
        test_value = {"data": "test_data"}
        await cache.set("test_key", test_value, ttl=600)
        
        # Verify Redis setex was called with correct parameters
        self.mock_redis_client.setex.assert_called_once_with(
            "test_key", 
            600, 
            json.dumps(test_value)
        )

    @pytest.mark.asyncio
    async def test_cache_delete_operations_use_coredis(self):
        """Test cache delete operations use coredis methods"""
        config = CacheConfig(max_size=100, ttl_seconds=300)
        cache = IntelligentCache(config)
        cache.redis_client = self.mock_redis_client
        
        # Set up local cache entry
        cache.cache["test_key"] = "test_value"
        cache.access_times["test_key"] = datetime.utcnow()
        cache.access_counts["test_key"] = 1
        
        await cache.delete("test_key")
        
        # Verify Redis delete was called
        self.mock_redis_client.delete.assert_called_once_with("test_key")
        
        # Verify local cache was cleared
        assert "test_key" not in cache.cache
        assert "test_key" not in cache.access_times
        assert "test_key" not in cache.access_counts

    @pytest.mark.asyncio 
    async def test_async_optimizer_with_coredis_cache(self):
        """Test AsyncOptimizer works with coredis-based intelligent cache"""
        from src.prompt_improver.performance.optimization.async_optimizer import AsyncOperationConfig
        
        config = AsyncOperationConfig(enable_intelligent_caching=True)
        optimizer = AsyncOptimizer(config)
        
        try:
            await optimizer.initialize()
            
            # Verify connection manager is initialized
            assert optimizer.connection_manager is not None
            assert optimizer.connection_manager._is_started
            
        finally:
            await optimizer.shutdown()


class TestCoredisCanaryTestingMigration:
    """Test migration of canary testing components to coredis"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_redis_client = AsyncMock(spec=coredis.Redis)
        self.mock_redis_client.get = AsyncMock(return_value=None)
        self.mock_redis_client.set = AsyncMock(return_value=True)
        self.mock_redis_client.setex = AsyncMock(return_value=True)

    @pytest.mark.asyncio
    async def test_canary_service_record_metrics_async(self):
        """Test canary service records metrics using async Redis operations"""
        service = CanaryTestingService()
        service.redis_client = self.mock_redis_client
        
        # Mock the should_enable_cache method
        with patch.object(service, 'should_enable_cache', return_value=True):
            await service.record_request_metrics(
                user_id="user_123",
                session_id="session_456", 
                response_time_ms=150.0,
                success=True,
                cache_hit=True
            )
        
        # Verify Redis operations were called
        self.mock_redis_client.get.assert_called()
        self.mock_redis_client.setex.assert_called()
        
        # Check the setex call parameters
        call_args = self.mock_redis_client.setex.call_args[0]
        assert call_args[1] == 86400 * 2  # 48 hours TTL
        
        # Verify metrics data structure
        metrics_data = json.loads(call_args[2])
        assert "total_requests" in metrics_data
        assert "successful_requests" in metrics_data
        assert "response_times" in metrics_data
        assert "cache_hits" in metrics_data

    @pytest.mark.asyncio
    async def test_canary_service_get_metrics_async(self):
        """Test canary service retrieves metrics using async Redis operations"""
        service = CanaryTestingService()
        service.redis_client = self.mock_redis_client
        
        # Mock Redis get to return sample metrics
        sample_metrics = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "response_times": [100, 150, 200],
            "cache_hits": 80,
            "cache_misses": 20
        }
        self.mock_redis_client.get = AsyncMock(return_value=json.dumps(sample_metrics))
        
        metrics = await service.get_canary_metrics(hours=1)
        
        # Verify Redis get was called for both control and treatment groups
        assert self.mock_redis_client.get.call_count >= 2
        
        # Verify returned structure
        assert "control" in metrics
        assert "treatment" in metrics
        assert isinstance(metrics["control"], CanaryMetrics)
        assert isinstance(metrics["treatment"], CanaryMetrics)

    @pytest.mark.asyncio
    async def test_canary_service_update_rollout_percentage_async(self):
        """Test canary service updates rollout percentage using async Redis operations"""
        service = CanaryTestingService()
        service.redis_client = self.mock_redis_client
        service.config = {}
        
        await service._update_rollout_percentage(25)
        
        # Verify Redis set was called with correct parameters
        self.mock_redis_client.set.assert_called_once_with(
            "canary_config:rollout_percentage", 
            "25"
        )
        
        # Verify local config was updated
        assert service.config["ab_testing"]["canary"]["initial_percentage"] == 25

    @pytest.mark.asyncio
    async def test_canary_service_get_status_async(self):
        """Test canary service gets status using async Redis operations"""
        service = CanaryTestingService()
        service.redis_client = self.mock_redis_client
        service.config = {
            "ab_testing": {
                "canary": {
                    "enabled": True,
                    "initial_percentage": 10
                }
            }
        }
        
        # Mock Redis get to return stored percentage
        self.mock_redis_client.get = AsyncMock(return_value="15")
        
        # Mock other methods to avoid complex setup
        mock_metrics = CanaryMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_response_time_ms=150.0,
            cache_hit_ratio=0.8,
            error_rate=0.05,
            p95_response_time_ms=200.0,
            p99_response_time_ms=250.0,
            timestamp=datetime.utcnow()
        )
        with patch.object(service, 'get_canary_metrics', return_value={"control": mock_metrics, "treatment": mock_metrics}):
            with patch.object(service, 'evaluate_canary_success', return_value={"success_criteria_met": True}):
                status = await service.get_canary_status()
        
        # Verify Redis get was called for percentage
        self.mock_redis_client.get.assert_called_with("canary_config:rollout_percentage")
        
        # Verify status structure
        assert "enabled" in status
        assert "current_percentage" in status
        assert status["current_percentage"] == 15  # From Redis, not config


class TestCoredisIntegrationScenarios:
    """Test comprehensive integration scenarios with coredis"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_redis_client = AsyncMock(spec=coredis.Redis)
        self.mock_db_session = AsyncMock()

    @pytest.mark.asyncio
    async def test_end_to_end_analytics_pipeline_with_coredis(self):
        """Test complete analytics pipeline using coredis"""
        # Initialize service
        service = EnhancedRealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        # Configure Redis responses
        self.mock_redis_client.ping.return_value = True
        self.mock_redis_client.setex.return_value = True
        
        # Simulate real analytics workflow
        events = []
        for i in range(10):
            event = AnalyticsEvent(
                event_id=f"event_{i:03d}",
                event_type=EventType.CONVERSION_EVENT if i % 2 == 0 else EventType.USER_ASSIGNMENT,
                experiment_id="exp_integration_test",
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                timestamp=datetime.utcnow(),
                data={"conversion_value": 25.0 * (i + 1)} if i % 2 == 0 else {"variant": "treatment"}
            )
            events.append(event)
            await service.emit_event(event)
        
        # Verify events were processed
        assert len(service.event_buffer) == 10
        
        # Simulate window processing
        from src.prompt_improver.performance.analytics.real_time_analytics import StreamWindow
        window = StreamWindow(
            window_id="integration_window_001",
            start_time=datetime.utcnow() - timedelta(seconds=30),
            end_time=datetime.utcnow(),
            experiment_id="exp_integration_test",
            events_count=len(events),
            metrics={
                "total_conversions": 5,
                "total_revenue": 375.0,
                "assignments_treatment": 5,
                "conversion_rate": 1.0
            }
        )
        
        await service._store_window_results(window)
        
        # Verify Redis operations
        self.mock_redis_client.setex.assert_called()
        call_args = self.mock_redis_client.setex.call_args[0]
        
        # Verify key format and data integrity
        key = call_args[0]
        assert "analytics_window:exp_integration_test:" in key
        
        ttl = call_args[1]
        assert ttl == 86400 * 7  # 7 days
        
        stored_data = json.loads(call_args[2])
        assert stored_data["events_count"] == 10
        assert stored_data["metrics"]["total_conversions"] == 5
        
    @pytest.mark.asyncio
    async def test_cache_performance_with_coredis(self):
        """Test cache performance characteristics with coredis"""
        config = CacheConfig(max_size=1000, ttl_seconds=300)
        cache = IntelligentCache(config)
        cache.redis_client = self.mock_redis_client
        
        # Configure Redis responses for performance test
        self.mock_redis_client.get = AsyncMock(return_value=None)  # Cache miss
        self.mock_redis_client.setex = AsyncMock(return_value=True)
        
        # Test cache operations under load
        start_time = datetime.utcnow()
        
        # Perform multiple cache operations
        tasks = []
        for i in range(100):
            key = f"perf_test_key_{i}"
            value = {"data": f"test_data_{i}", "timestamp": datetime.utcnow().isoformat()}
            tasks.append(cache.set(key, value))
        
        await asyncio.gather(*tasks)
        
        # Verify operations completed quickly
        duration = (datetime.utcnow() - start_time).total_seconds()
        assert duration < 5.0, f"Cache operations took too long: {duration}s"
        
        # Verify Redis was called for each operation
        assert self.mock_redis_client.setex.call_count == 100
        
        # Test hit rate calculation
        cache.cache_stats["hits"] = 75
        cache.cache_stats["misses"] = 25
        hit_rate = cache.get_hit_rate()
        assert hit_rate == 0.75

    @pytest.mark.asyncio
    async def test_error_handling_with_coredis_failures(self):
        """Test error handling when coredis operations fail"""
        service = EnhancedRealTimeAnalyticsService(
            db_session=self.mock_db_session,
            redis_client=self.mock_redis_client
        )
        
        # Configure Redis to fail
        self.mock_redis_client.setex.side_effect = coredis.exceptions.ConnectionError("Redis connection failed")
        
        # Test window storage with Redis failure
        from src.prompt_improver.performance.analytics.real_time_analytics import StreamWindow
        window = StreamWindow(
            window_id="error_test_window",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=30),
            experiment_id="exp_error_test",
            events_count=1,
            metrics={"test_metric": 1.0}
        )
        
        # Should not raise exception, just log error
        await service._store_window_results(window)
        
        # Verify Redis operation was attempted
        self.mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_management_with_coredis(self):
        """Test connection lifecycle management with coredis"""
        config = CacheConfig()
        cache = IntelligentCache(config)
        
        # Mock coredis.Redis to return our mock client
        with patch('src.prompt_improver.performance.optimization.async_optimizer.coredis.Redis') as mock_redis_constructor:
            mock_redis_constructor.return_value = self.mock_redis_client
            self.mock_redis_client.ping = AsyncMock(return_value=True)
            
            # Test initialization
            await cache.ensure_redis_connection()
            
            # Verify connection was established
            mock_redis_constructor.assert_called_once()
            self.mock_redis_client.ping.assert_called_once()
            assert cache.redis_client is self.mock_redis_client
            
            # Test connection failure handling
            self.mock_redis_client.ping = AsyncMock(side_effect=coredis.exceptions.ConnectionError("Connection failed"))
            
            # Reinitialize - should handle error gracefully  
            cache._redis_initialization_task = None  # Reset task
            await cache.ensure_redis_connection()
            # The cache client will still exist but connection error should be logged
            # In real usage, operations would fail but the client object remains


if __name__ == "__main__":
    # Run a quick smoke test
    async def smoke_test():
        """Quick smoke test to verify imports and basic functionality work"""
        print("Running coredis migration smoke test...")
        
        # Test 1: Import verification
        try:
            import coredis
            from src.prompt_improver.performance.analytics.real_time_analytics import RealTimeAnalyticsService
            from src.prompt_improver.performance.optimization.async_optimizer import IntelligentCache
            print("✅ All imports successful")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        
        # Test 2: Basic service instantiation
        try:
            mock_db = AsyncMock()
            mock_redis = AsyncMock(spec=coredis.Redis)
            
            service = RealTimeAnalyticsService(mock_db, mock_redis)
            assert service.redis_client is mock_redis
            print("✅ Service instantiation successful")
        except Exception as e:
            print(f"❌ Service instantiation error: {e}")
            return False
        
        # Test 3: Cache initialization
        try:
            from src.prompt_improver.performance.optimization.async_optimizer import CacheConfig
            config = CacheConfig()
            cache = IntelligentCache(config)
            print("✅ Cache initialization successful")
        except Exception as e:
            print(f"❌ Cache initialization error: {e}")
            return False
        
        print("🎉 All smoke tests passed!")
        return True
    
    # Run smoke test if executed directly
    asyncio.run(smoke_test())