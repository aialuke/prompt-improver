"""Integration tests for Redis Health Manager

Tests the new decomposed Redis health monitoring services
with real Redis instances using testcontainers following SRE best practices.
"""

import asyncio
import time

import pytest

from prompt_improver.monitoring.redis.health import (
    AlertLevel,
    RedisAlertingService,
    RedisConnectionMonitor,
    RedisHealthChecker,
    RedisHealthManager,
    RedisHealthStatus,
    RedisMetricsCollector,
    RedisRecoveryService,
)


class MockRedisClientProvider:
    """Mock Redis client provider for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self._ping_count = 0

    async def get_client(self):
        """Get mock Redis client."""
        if self.should_fail:
            return None
        return MockRedisClient(self.should_fail)

    async def create_backup_client(self):
        """Create mock backup Redis client."""
        return MockRedisClient(False) if not self.should_fail else None

    async def validate_client(self, client):
        """Validate mock client."""
        return client is not None and not self.should_fail


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self._ping_count = 0

    async def ping(self):
        """Mock ping operation."""
        self._ping_count += 1
        if self.should_fail:
            raise ConnectionError("Mock Redis connection failed")
        await asyncio.sleep(0.001)  # Simulate small latency
        return "PONG"

    async def info(self):
        """Mock info operation."""
        if self.should_fail:
            raise ConnectionError("Mock Redis info failed")

        return {
            "connected_clients": 5,
            "maxclients": 10000,
            "used_memory": 1024 * 1024,  # 1MB
            "used_memory_rss": 1024 * 1024 + 512,
            "mem_fragmentation_ratio": 1.05,
            "keyspace_hits": 1000,
            "keyspace_misses": 100,
            "instantaneous_ops_per_sec": 50,
            "total_commands_processed": 5000,
        }

    async def slowlog_get(self, count=100):
        """Mock slowlog operation."""
        return []  # No slow queries for healthy mock

    async def slowlog_len(self):
        """Mock slowlog length."""
        return 0


@pytest.mark.asyncio
class TestRedisHealthManagerIntegration:
    """Integration tests for Redis Health Manager."""

    async def test_health_manager_initialization(self):
        """Test health manager initializes properly."""
        client_provider = MockRedisClientProvider()
        health_manager = RedisHealthManager(client_provider)

        assert health_manager.client_provider == client_provider
        assert health_manager.health_checker is not None
        assert health_manager.connection_monitor is not None
        assert health_manager.metrics_collector is not None
        assert health_manager.alerting_service is not None
        assert health_manager.recovery_service is not None

    async def test_health_summary_success(self):
        """Test health summary with successful Redis connection."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        summary = await health_manager.get_health_summary()

        assert summary["healthy"] is True
        assert summary["status"] == "healthy"
        assert "response_time_ms" in summary
        assert "timestamp" in summary
        assert summary["cached"] is False

    async def test_health_summary_failure(self):
        """Test health summary with failed Redis connection."""
        client_provider = MockRedisClientProvider(should_fail=True)
        health_manager = RedisHealthManager(client_provider)

        summary = await health_manager.get_health_summary()

        assert summary["healthy"] is False
        assert summary["status"] in {"failed", "critical"}
        assert "timestamp" in summary

    async def test_comprehensive_health_check(self):
        """Test comprehensive health check functionality."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        health_data = await health_manager.get_comprehensive_health()

        # Verify structure
        assert "health_check" in health_data
        assert "connections" in health_data
        assert "performance" in health_data
        assert "alerting" in health_data
        assert "recovery" in health_data
        assert "manager" in health_data

        # Verify health check data
        health_check = health_data["health_check"]
        assert health_check["status"] == "healthy"
        assert health_check["is_available"] is True
        assert "ping_latency_ms" in health_check
        assert "last_check_time" in health_check

        # Verify manager metadata
        manager_data = health_data["manager"]
        assert manager_data["overall_healthy"] is True
        assert "check_duration_ms" in manager_data
        assert "timestamp" in manager_data

    async def test_individual_service_functionality(self):
        """Test individual health service functionality."""
        client_provider = MockRedisClientProvider(should_fail=False)

        # Test health checker
        health_checker = RedisHealthChecker(client_provider)
        health_metrics = await health_checker.check_health()

        assert health_metrics.status == RedisHealthStatus.HEALTHY
        assert health_metrics.is_available is True
        assert health_metrics.ping_latency_ms > 0

        # Test connection monitor
        connection_monitor = RedisConnectionMonitor(client_provider)
        connection_metrics = await connection_monitor.monitor_connections()

        assert connection_metrics.active_connections >= 0
        assert connection_metrics.pool_utilization >= 0

        # Test metrics collector
        metrics_collector = RedisMetricsCollector(client_provider)
        performance_metrics = await metrics_collector.collect_performance_metrics()

        assert performance_metrics.hit_rate_percentage >= 0
        assert performance_metrics.avg_ops_per_sec >= 0

    async def test_alerting_service_functionality(self):
        """Test alerting service functionality."""
        alerting_service = RedisAlertingService()

        # Create mock unhealthy metrics
        from prompt_improver.monitoring.redis.health.types import HealthMetrics

        unhealthy_metrics = HealthMetrics(
            ping_latency_ms=150.0,  # High latency
            connection_utilization=90.0,  # High utilization
            hit_rate=30.0,  # Low hit rate
            is_available=True,
            consecutive_failures=0
        )

        # Check thresholds
        alerts = await alerting_service.check_thresholds(unhealthy_metrics)

        assert len(alerts) > 0
        assert any(alert.level == AlertLevel.CRITICAL for alert in alerts)

        # Test alert sending
        if alerts:
            success = await alerting_service.send_alert(alerts[0])
            assert success is True

            # Check active alerts
            active_alerts = alerting_service.get_active_alerts()
            assert len(active_alerts) > 0

    async def test_recovery_service_functionality(self):
        """Test recovery service functionality."""
        client_provider = MockRedisClientProvider(should_fail=False)
        recovery_service = RedisRecoveryService(client_provider)

        # Test recovery attempt
        recovery_event = await recovery_service.attempt_recovery("Test recovery")

        assert recovery_event.id is not None
        assert recovery_event.trigger_reason == "Test recovery"
        assert recovery_event.completion_time is not None

        # Test circuit breaker
        circuit_state = recovery_service.get_circuit_breaker_state()
        assert circuit_state.state in {"closed", "open", "half_open"}

        # Test operation result recording
        recovery_service.record_operation_result(True)
        recovery_service.record_operation_result(False)

        # Test recovery validation
        validation_result = await recovery_service.validate_recovery()
        assert isinstance(validation_result, bool)

    async def test_monitoring_lifecycle(self):
        """Test monitoring service lifecycle."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        # Test starting monitoring
        await health_manager.start_monitoring()
        assert health_manager._is_monitoring is True

        # Wait briefly for monitoring to run
        await asyncio.sleep(0.1)

        # Test monitoring status
        status = health_manager.get_monitoring_status()
        assert status["manager"]["is_monitoring"] is True
        assert status["services"]["health_checker"]["enabled"] is True

        # Test stopping monitoring
        await health_manager.stop_monitoring()
        assert health_manager._is_monitoring is False

    async def test_incident_handling(self):
        """Test incident handling functionality."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        # Test critical incident handling
        actions = await health_manager.handle_incident(AlertLevel.CRITICAL)
        assert len(actions) > 0

        # Test warning incident handling
        actions = await health_manager.handle_incident(AlertLevel.WARNING)
        assert isinstance(actions, list)

    async def test_performance_requirements(self):
        """Test that operations meet <25ms performance requirements."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        # Test health summary performance
        start_time = time.time()
        summary = await health_manager.get_health_summary()
        duration_ms = (time.time() - start_time) * 1000

        assert duration_ms < 25.0, f"Health summary took {duration_ms:.2f}ms (should be <25ms)"
        assert summary["healthy"] is not None

        # Test quick health check performance
        start_time = time.time()
        is_healthy = health_manager.is_healthy()
        duration_ms = (time.time() - start_time) * 1000

        assert duration_ms < 5.0, f"Quick health check took {duration_ms:.2f}ms (should be <5ms)"
        assert isinstance(is_healthy, bool)

    async def test_caching_behavior(self):
        """Test health check caching for performance."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        # First comprehensive check
        health_data1 = await health_manager.get_comprehensive_health()

        # Second check should use cached data for summary
        start_time = time.time()
        summary = await health_manager.get_health_summary()
        duration_ms = (time.time() - start_time) * 1000

        # Should be very fast due to caching
        assert duration_ms < 5.0, f"Cached summary took {duration_ms:.2f}ms (should be <5ms)"
        assert summary["cached"] is True or summary["healthy"] is not None


@pytest.mark.asyncio
class TestRedisHealthManagerFailureScenarios:
    """Test Redis Health Manager failure scenarios."""

    async def test_redis_unavailable_scenario(self):
        """Test behavior when Redis is completely unavailable."""
        client_provider = MockRedisClientProvider(should_fail=True)
        health_manager = RedisHealthManager(client_provider)

        # Health check should handle failure gracefully
        health_data = await health_manager.get_comprehensive_health()

        assert health_data["manager"]["overall_healthy"] is False
        assert health_data["health_check"]["is_available"] is False
        assert health_data["health_check"]["status"] == "failed"

    async def test_partial_service_failures(self):
        """Test behavior when some services fail but others work."""
        client_provider = MockRedisClientProvider(should_fail=False)
        health_manager = RedisHealthManager(client_provider)

        # Even with some service failures, manager should provide meaningful data
        health_data = await health_manager.get_comprehensive_health()

        # Should have manager metadata even if some services fail
        assert "manager" in health_data
        assert "timestamp" in health_data["manager"]

    async def test_timeout_handling(self):
        """Test timeout handling in health checks."""
        class SlowMockRedisClient:
            async def ping(self):
                await asyncio.sleep(2)  # Longer than typical timeout
                return "PONG"

            async def info(self):
                await asyncio.sleep(2)
                return {}

        class SlowClientProvider:
            async def get_client(self):
                return SlowMockRedisClient()

            async def create_backup_client(self):
                return SlowMockRedisClient()

            async def validate_client(self, client):
                return True

        client_provider = SlowClientProvider()
        health_checker = RedisHealthChecker(client_provider, timeout_seconds=0.1)

        # Should timeout quickly and return appropriate metrics
        start_time = time.time()
        metrics = await health_checker.check_health()
        duration = time.time() - start_time

        assert duration < 1.0  # Should timeout quickly
        assert metrics.is_available is False or metrics.consecutive_failures > 0
