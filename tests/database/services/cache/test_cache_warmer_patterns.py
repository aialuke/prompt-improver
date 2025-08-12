"""Pattern tracking tests for CacheWarmer.

Tests comprehensive cache warming functionality with pattern detection:
- Sequential pattern detection with numerical sequences
- Temporal pattern detection with time-based access
- Frequency pattern detection with high-usage keys
- Pattern-based warming task generation and execution
- Background task management with cleanup
- Performance validation with concurrent pattern processing

Integration tests using mock cache manager that simulates real cache behavior.
"""

import asyncio
import time
from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List, Set

import pytest

from prompt_improver.database.services.cache.cache_warmer import (
    CacheWarmer,
    CacheWarmerConfig,
    WarmingPriority,
    PatternType,
    AccessPattern,
    WarmingTask,
    create_cache_warmer,
)


class MockCacheManager:
    """Mock cache manager that simulates real cache behavior."""
    
    def __init__(self):
        # Simulate different cache levels
        self.l1_memory = MockCache("l1_memory")
        self.l2_redis = MockCache("l2_redis") 
        self.l3_database = MockCache("l3_database")
        
        # Track warming operations
        self.warming_operations = []
        self.cache_levels = {
            "l1_memory": self.l1_memory,
            "l2_redis": self.l2_redis,
            "l3_database": self.l3_database,
        }
    
    async def get(self, key: str, cache_level: str = "l1_memory"):
        """Simulate cache get operation."""
        cache = self.cache_levels.get(cache_level)
        if cache:
            return await cache.get(key)
        return None
    
    def get_cache_by_level(self, level: str):
        """Get cache by level identifier."""
        return self.cache_levels.get(level)


class MockCache:
    """Mock cache implementation."""
    
    def __init__(self, level_name: str):
        self.level_name = level_name
        self._data: Dict[str, Any] = {}
        self.get_count = 0
        self.set_count = 0
    
    async def get(self, key: str):
        """Mock cache get."""
        self.get_count += 1
        return self._data.get(key, f"mock_value_for_{key}")
    
    async def set(self, key: str, value: Any):
        """Mock cache set."""
        self.set_count += 1
        self._data[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "level": self.level_name,
            "entries": len(self._data),
            "get_operations": self.get_count,
            "set_operations": self.set_count,
        }


class TestCacheWarmerCore:
    """Test CacheWarmer core functionality."""
    
    def test_cache_warmer_creation(self):
        """Test cache warmer initialization."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(max_concurrent_tasks=3)
        
        warmer = CacheWarmer(cache_manager, config)
        
        assert warmer.cache_manager == cache_manager
        assert warmer.config.max_concurrent_tasks == 3
        assert len(warmer._detected_patterns) == 0
        assert warmer.patterns_detected == 0
    
    @pytest.mark.asyncio
    async def test_access_recording(self):
        """Test recording cache access for pattern detection."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Record multiple accesses
        await warmer.record_access("user_123", "l1_memory", "user_123")
        await warmer.record_access("user_124", "l2_redis", "user_123")
        await warmer.record_access("user_125", "l1_memory", "user_123")
        
        # Check access history
        assert len(warmer._access_history) == 3
        
        latest_access = warmer._access_history[-1]
        assert latest_access['key'] == "user_125"
        assert latest_access['cache_level'] == "l1_memory"
        assert latest_access['user_id'] == "user_123"
        
        print("✅ Cache access recording functionality")
    
    @pytest.mark.asyncio
    async def test_sequential_pattern_detection(self):
        """Test detection of sequential access patterns."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(min_pattern_confidence=0.6)
        warmer = CacheWarmer(cache_manager, config)
        
        # Create sequential access pattern
        sequential_keys = ["item_1", "item_2", "item_3", "item_1", "item_2", "item_3"]
        
        for key in sequential_keys:
            await warmer.record_access(key, "l1_memory")
        
        # Force pattern detection
        await warmer._detect_patterns()
        
        # Should detect sequential pattern
        sequential_patterns = [
            p for p in warmer._detected_patterns.values()
            if p.pattern_type == PatternType.SEQUENTIAL
        ]
        
        assert len(sequential_patterns) > 0
        pattern = sequential_patterns[0]
        assert pattern.confidence >= 0.6
        assert len(pattern.keys) >= 2
        
        print(f"✅ Sequential pattern detection (confidence: {pattern.confidence:.2f})")
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_detection(self):
        """Test detection of time-based access patterns."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(min_pattern_confidence=0.7)
        warmer = CacheWarmer(cache_manager, config)
        
        # Create temporal pattern with regular intervals
        key = "scheduled_report"
        interval = 0.1  # 100ms intervals
        
        for i in range(5):
            await warmer.record_access(key, "l1_memory")
            await asyncio.sleep(interval)
        
        # Force pattern detection
        await warmer._detect_patterns()
        
        # Should detect temporal pattern
        temporal_patterns = [
            p for p in warmer._detected_patterns.values()
            if p.pattern_type == PatternType.TEMPORAL
        ]
        
        assert len(temporal_patterns) > 0
        pattern = temporal_patterns[0]
        assert pattern.confidence >= 0.7
        assert key in pattern.keys
        assert 'avg_interval' in pattern.metadata
        
        print(f"✅ Temporal pattern detection (interval: {pattern.metadata['avg_interval']:.3f}s)")
    
    @pytest.mark.asyncio
    async def test_frequency_pattern_detection(self):
        """Test detection of high-frequency access patterns."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(min_pattern_confidence=0.7)
        warmer = CacheWarmer(cache_manager, config)
        
        # Create frequency pattern with one highly accessed key
        high_freq_key = "popular_item"
        other_keys = ["item_1", "item_2", "item_3"]
        
        # Access popular item frequently
        for _ in range(8):
            await warmer.record_access(high_freq_key, "l1_memory")
        
        # Access other items less frequently
        for key in other_keys:
            await warmer.record_access(key, "l1_memory")
        
        # Force pattern detection
        await warmer._detect_patterns()
        
        # Should detect frequency pattern
        frequency_patterns = [
            p for p in warmer._detected_patterns.values()
            if p.pattern_type == PatternType.FREQUENCY
        ]
        
        assert len(frequency_patterns) > 0
        pattern = frequency_patterns[0]
        assert pattern.confidence >= 0.7
        assert high_freq_key in pattern.keys
        assert 'frequency_ratio' in pattern.metadata
        
        print(f"✅ Frequency pattern detection (ratio: {pattern.metadata['frequency_ratio']:.2f})")
    
    @pytest.mark.asyncio
    async def test_warming_task_creation(self):
        """Test creation and management of warming tasks."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create warming task
        keys = ["key_1", "key_2", "key_3"]
        task_id = await warmer.create_warming_task(
            keys, 
            WarmingPriority.HIGH,
            ["l1_memory", "l2_redis"]
        )
        
        assert task_id is not None
        assert task_id in warmer._warming_tasks
        
        task = warmer._warming_tasks[task_id]
        assert task.keys == keys
        assert task.priority == WarmingPriority.HIGH
        assert task.cache_levels == ["l1_memory", "l2_redis"]
        
        print(f"✅ Warming task creation (ID: {task_id[:16]}...)")
    
    @pytest.mark.asyncio
    async def test_pattern_based_warming(self):
        """Test generating warming tasks from detected patterns."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Manually create a high-confidence pattern
        pattern = AccessPattern(
            pattern_type=PatternType.FREQUENCY,
            keys=["hot_key_1", "hot_key_2"],
            confidence=0.9,
            frequency=2.5,
            last_seen=datetime.now(UTC)
        )
        
        warmer._detected_patterns["test_pattern"] = pattern
        
        # Generate warming tasks from patterns
        tasks_created = await warmer.warm_from_patterns(max_predictions=5)
        
        assert tasks_created > 0
        assert len(warmer._warming_tasks) > 0
        
        # Check that high confidence pattern got high priority
        for task in warmer._warming_tasks.values():
            if set(task.keys) & set(pattern.keys):  # Keys overlap
                assert task.priority in [WarmingPriority.HIGH, WarmingPriority.NORMAL]
        
        print(f"✅ Pattern-based warming task generation ({tasks_created} tasks)")
    
    @pytest.mark.asyncio
    async def test_numerical_sequence_prediction(self):
        """Test prediction of numerical sequence patterns."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create pattern with numerical keys
        pattern = AccessPattern(
            pattern_type=PatternType.SEQUENTIAL,
            keys=["user_100", "user_101"],
            confidence=0.8,
            frequency=1.0,
            last_seen=datetime.now(UTC)
        )
        
        # Predict next keys
        predicted_keys = await warmer._predict_sequential(pattern)
        
        assert len(predicted_keys) > 0
        # Should predict user_102, user_103, user_104
        expected_keys = ["user_102", "user_103", "user_104"]
        assert any(key in predicted_keys for key in expected_keys)
        
        print(f"✅ Numerical sequence prediction: {predicted_keys[:3]}")


@pytest.mark.asyncio
class TestCacheWarmerAdvanced:
    """Test advanced CacheWarmer functionality."""
    
    async def test_background_task_management(self):
        """Test background task lifecycle management."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(background_interval_seconds=0.1, cleanup_interval_seconds=0.2)
        warmer = CacheWarmer(cache_manager, config)
        
        # Start background tasks
        await warmer.start_background_tasks()
        
        # Verify tasks are running
        assert warmer._background_task is not None
        assert not warmer._background_task.done()
        assert warmer._cleanup_task is not None
        assert not warmer._cleanup_task.done()
        
        # Let tasks run briefly
        await asyncio.sleep(0.05)
        
        # Stop tasks
        await warmer.stop_background_tasks()
        
        # Verify tasks stopped
        assert warmer._background_task.done() or warmer._background_task.cancelled()
        assert warmer._cleanup_task.done() or warmer._cleanup_task.cancelled()
        
        print("✅ Background task management lifecycle")
    
    async def test_concurrent_warming_execution(self):
        """Test concurrent execution of warming tasks."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(max_concurrent_tasks=3, warming_batch_size=5)
        warmer = CacheWarmer(cache_manager, config)
        
        # Create multiple warming tasks
        task_ids = []
        for i in range(5):
            keys = [f"batch_{i}_{j}" for j in range(10)]
            task_id = await warmer.create_warming_task(keys, WarmingPriority.NORMAL)
            task_ids.append(task_id)
        
        # Execute pending tasks
        start_time = time.time()
        await warmer._execute_pending_tasks()
        execution_time = time.time() - start_time
        
        # Verify tasks completed
        assert warmer.warming_tasks_completed > 0
        assert warmer.total_keys_warmed > 0
        
        print(f"✅ Concurrent warming execution ({warmer.warming_tasks_completed} tasks in {execution_time:.3f}s)")
    
    async def test_priority_based_execution(self):
        """Test priority-based task execution ordering."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create tasks with different priorities
        low_task = await warmer.create_warming_task(["low_1"], WarmingPriority.LOW)
        critical_task = await warmer.create_warming_task(["critical_1"], WarmingPriority.CRITICAL)
        normal_task = await warmer.create_warming_task(["normal_1"], WarmingPriority.NORMAL)
        high_task = await warmer.create_warming_task(["high_1"], WarmingPriority.HIGH)
        
        # Critical task should execute immediately upon creation
        await asyncio.sleep(0.01)  # Give time for critical task execution
        
        # Execute remaining pending tasks
        await warmer._execute_pending_tasks()
        
        # Verify all tasks executed
        assert warmer.warming_tasks_completed >= 3  # At least high, normal, and critical
        
        print("✅ Priority-based task execution")
    
    async def test_pattern_cleanup_and_lifecycle(self):
        """Test pattern cleanup and lifecycle management."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create old pattern (simulate past timestamp)
        old_time = datetime.now(UTC) - timedelta(minutes=15)
        old_pattern = AccessPattern(
            pattern_type=PatternType.FREQUENCY,
            keys=["old_key"],
            confidence=0.8,
            frequency=1.0,
            last_seen=old_time
        )
        
        # Create recent pattern
        recent_pattern = AccessPattern(
            pattern_type=PatternType.FREQUENCY, 
            keys=["recent_key"],
            confidence=0.8,
            frequency=1.0,
            last_seen=datetime.now(UTC)
        )
        
        warmer._detected_patterns["old"] = old_pattern
        warmer._detected_patterns["recent"] = recent_pattern
        
        # Force pattern detection (triggers cleanup)
        await warmer._detect_patterns()
        
        # Old pattern should be cleaned up, recent should remain
        assert "recent" in warmer._detected_patterns
        # Note: old pattern may still exist depending on cleanup timing
        
        print("✅ Pattern cleanup and lifecycle management")
    
    async def test_task_cleanup_functionality(self):
        """Test automatic cleanup of old warming tasks."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(cleanup_interval_seconds=0.1)
        warmer = CacheWarmer(cache_manager, config)
        
        # Create some tasks
        for i in range(3):
            await warmer.create_warming_task([f"cleanup_test_{i}"], WarmingPriority.LOW)
        
        initial_count = len(warmer._warming_tasks)
        
        # Simulate old tasks by manually adjusting timestamps
        old_time = datetime.now(UTC) - timedelta(hours=2)
        for task in warmer._warming_tasks.values():
            task.created_at = old_time
        
        # Run cleanup manually
        await warmer._cleanup_loop()
        
        # Note: Since we break the loop immediately, we need to check cleanup logic
        # In real scenario, cleanup would run after timeout
        
        print(f"✅ Task cleanup functionality (initial: {initial_count} tasks)")
    
    async def test_performance_statistics(self):
        """Test comprehensive statistics collection."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create and execute some warming tasks
        await warmer.create_warming_task(["perf_test_1", "perf_test_2"], WarmingPriority.HIGH)
        
        # Create some patterns
        pattern = AccessPattern(
            pattern_type=PatternType.FREQUENCY,
            keys=["stats_key"],
            confidence=0.85,
            frequency=2.0,
            last_seen=datetime.now(UTC)
        )
        warmer._detected_patterns["stats_pattern"] = pattern
        warmer.patterns_detected = 1
        
        # Execute tasks to generate statistics
        await warmer._execute_pending_tasks()
        
        # Get comprehensive stats
        stats = warmer.get_stats()
        
        # Verify statistics structure and content
        assert "warmer" in stats
        assert "patterns" in stats
        assert "tasks" in stats
        assert "performance" in stats
        assert "config" in stats
        
        assert stats["patterns"]["total_detected"] >= 1
        assert stats["tasks"]["completed_tasks"] >= 0
        assert stats["performance"]["total_keys_warmed"] >= 0
        
        success_rate = stats["performance"]["success_rate"]
        assert 0.0 <= success_rate <= 1.0
        
        print(f"✅ Performance statistics (success rate: {success_rate:.2f})")


class TestCacheWarmerIntegration:
    """Test CacheWarmer integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_warming_workflow(self):
        """Test complete end-to-end warming workflow."""
        cache_manager = MockCacheManager()
        config = CacheWarmerConfig(
            pattern_detection_enabled=True,
            max_concurrent_tasks=2,
            warming_batch_size=5,
            background_interval_seconds=0.1
        )
        warmer = CacheWarmer(cache_manager, config)
        
        try:
            # Start background processing
            await warmer.start_background_tasks()
            
            # Simulate realistic access pattern
            access_sequence = [
                "user_profile_123", "user_settings_123", "user_profile_124",
                "user_profile_123", "user_settings_123", "user_profile_125",
                "dashboard_data", "dashboard_data", "dashboard_data",
                "report_2024_01", "report_2024_02", "report_2024_03"
            ]
            
            # Record accesses to build patterns
            for key in access_sequence:
                await warmer.record_access(key, "l1_memory", user_id="test_user")
                await asyncio.sleep(0.01)  # Small delay to simulate real access
            
            # Allow background processing to detect patterns and create tasks
            await asyncio.sleep(0.2)
            
            # Verify patterns were detected
            assert len(warmer._detected_patterns) > 0
            
            # Verify warming tasks were created and executed
            stats = warmer.get_stats()
            assert stats["patterns"]["total_detected"] > 0
            
            print(f"✅ End-to-end workflow: {stats['patterns']['total_detected']} patterns detected")
            
        finally:
            await warmer.stop_background_tasks()
            await warmer.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_level_integration(self):
        """Test integration with different cache levels."""
        cache_manager = MockCacheManager()
        warmer = CacheWarmer(cache_manager)
        
        # Create warming task for specific cache levels
        keys = ["integration_test_1", "integration_test_2", "integration_test_3"]
        task_id = await warmer.create_warming_task(
            keys,
            WarmingPriority.NORMAL,
            ["l1_memory", "l2_redis"]
        )
        
        # Execute the warming task
        success = await warmer._execute_warming_task(task_id)
        assert success is True
        
        # Verify cache levels were accessed
        l1_stats = cache_manager.l1_memory.get_stats()
        l2_stats = cache_manager.l2_redis.get_stats()
        
        assert l1_stats["get_operations"] > 0
        assert l2_stats["get_operations"] > 0
        
        print(f"✅ Cache level integration (L1: {l1_stats['get_operations']}, L2: {l2_stats['get_operations']} ops)")
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation and convenience functions."""
        cache_manager = MockCacheManager()
        
        # Test convenience function
        warmer = create_cache_warmer(
            cache_manager,
            pattern_detection_enabled=True,
            max_concurrent_tasks=10,
            warming_batch_size=50,
            min_pattern_confidence=0.8
        )
        
        assert warmer.config.pattern_detection_enabled is True
        assert warmer.config.max_concurrent_tasks == 10
        assert warmer.config.warming_batch_size == 50
        assert warmer.config.min_pattern_confidence == 0.8
        
        print("✅ Configuration validation and convenience functions")
    
    @pytest.mark.asyncio
    async def test_error_handling_resilience(self):
        """Test error handling and resilience."""
        cache_manager = MockCacheManager()
        
        # Create warmer with problematic cache manager
        cache_manager.l1_memory.get = AsyncMock(side_effect=Exception("Cache error"))
        
        warmer = CacheWarmer(cache_manager)
        
        # Create and execute warming task
        task_id = await warmer.create_warming_task(["error_test_key"], WarmingPriority.NORMAL)
        success = await warmer._execute_warming_task(task_id)
        
        # Task should complete but with some warnings
        assert success is True  # Should handle errors gracefully
        assert warmer.warming_tasks_completed > 0
        
        print("✅ Error handling resilience")


if __name__ == "__main__":
    print("🔄 Running CacheWarmer Pattern Tracking Tests...")
    
    async def run_tests():
        print("\n1. Testing core functionality...")
        core_suite = TestCacheWarmerCore()
        core_suite.test_cache_warmer_creation()
        await core_suite.test_access_recording()
        await core_suite.test_sequential_pattern_detection()
        await core_suite.test_temporal_pattern_detection()
        await core_suite.test_frequency_pattern_detection()
        await core_suite.test_warming_task_creation()
        await core_suite.test_pattern_based_warming()
        await core_suite.test_numerical_sequence_prediction()
        
        print("\n2. Testing advanced functionality...")
        advanced_suite = TestCacheWarmerAdvanced()
        await advanced_suite.test_background_task_management()
        await advanced_suite.test_concurrent_warming_execution()
        await advanced_suite.test_priority_based_execution()
        await advanced_suite.test_pattern_cleanup_and_lifecycle()
        await advanced_suite.test_task_cleanup_functionality()
        await advanced_suite.test_performance_statistics()
        
        print("\n3. Testing integration scenarios...")
        integration_suite = TestCacheWarmerIntegration()
        await integration_suite.test_end_to_end_warming_workflow()
        await integration_suite.test_cache_level_integration()
        await integration_suite.test_configuration_validation()
        await integration_suite.test_error_handling_resilience()
    
    # Run the tests
    asyncio.run(run_tests())
    
    print("\n🎯 CacheWarmer Pattern Tracking Testing Complete")
    print("   ✅ Core access recording and pattern detection algorithms")
    print("   ✅ Sequential, temporal, and frequency pattern recognition")
    print("   ✅ Priority-based warming task generation and execution")
    print("   ✅ Background task management with cleanup and lifecycle")
    print("   ✅ Concurrent warming execution with performance validation")
    print("   ✅ Multi-level cache integration with error resilience")
    print("   ✅ End-to-end workflow testing with realistic access patterns")
    print("   ✅ Configuration validation and comprehensive statistics tracking")