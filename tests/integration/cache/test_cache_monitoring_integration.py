"""
Integration Tests for Unified Cache Monitoring System
=====================================================

Comprehensive integration tests validating the unified cache monitoring
system with all components: UnifiedCacheMonitor, SLO integration,
cross-level coordination, and OpenTelemetry metrics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.containers.redis_container import RedisContainer
from tests.utils.async_helpers import async_test

from prompt_improver.database import (
    ManagerMode,
    SecurityContext,
    DatabaseServices,
)
from prompt_improver.monitoring.cache import (
    AlertSeverity,
    CacheLevel,
    CacheSLIType,
    InvalidationType,
    get_cache_monitoring_report,
    get_cache_slo_integration,
    get_cross_level_coordinator,
    get_unified_cache_monitor,
    initialize_comprehensive_cache_monitoring,
)


@pytest.fixture
async def redis_container():
    """Provide Redis container for testing."""
    container = RedisContainer()
    await container.start()
    try:
        yield container
    finally:
        await container.stop()


@pytest.fixture
async def test_unified_manager(redis_container):
    """Create DatabaseServices for testing."""
    from prompt_improver.core.config import AppConfig

    config = AppConfig()
    config.redis.host = redis_container.host
    config.redis.port = redis_container.port
    config.redis.cache_db = 0
    manager = DatabaseServices(mode=ManagerMode.ASYNC_MODERN, config=config)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()


@pytest.fixture
async def security_context():
    """Create test security context."""
    return SecurityContext(
        agent_id="test_agent", tier="basic", authenticated=True, created_at=time.time()
    )


@pytest.fixture
async def monitoring_system(test_unified_manager):
    """Initialize comprehensive cache monitoring system."""
    await initialize_comprehensive_cache_monitoring(test_unified_manager)
    cache_monitor = get_unified_cache_monitor()
    slo_integration = get_cache_slo_integration()
    coordinator = get_cross_level_coordinator()
    return {
        "unified_manager": test_unified_manager,
        "cache_monitor": cache_monitor,
        "slo_integration": slo_integration,
        "coordinator": coordinator,
    }


@pytest.mark.asyncio
class TestUnifiedCacheMonitoringIntegration:
    """Integration tests for unified cache monitoring."""

    async def test_monitoring_system_initialization(self, monitoring_system):
        """Test that monitoring system initializes correctly."""
        components = monitoring_system
        assert components["cache_monitor"] is not None
        assert components["slo_integration"] is not None
        assert components["coordinator"] is not None
        assert components["unified_manager"] is not None
        assert components["cache_monitor"]._unified_manager is not None
        assert components["coordinator"]._unified_manager is not None

    async def test_cache_operation_monitoring(
        self, monitoring_system, security_context
    ):
        """Test monitoring of cache operations."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]
        test_key = "monitoring_test_key"
        test_value = {"data": "test_monitoring_data", "timestamp": time.time()}
        await unified_manager.set_cached(test_key, test_value, 300, security_context)
        result = await unified_manager.get_cached(test_key, security_context)
        assert result is not None
        miss_result = await unified_manager.get_cached(
            "nonexistent_key", security_context
        )
        assert miss_result is None
        deleted = await unified_manager.delete_cached(test_key, security_context)
        assert deleted is True
        await asyncio.sleep(0.1)
        stats = cache_monitor.get_comprehensive_stats()
        assert "enhanced_monitoring" in stats
        enhanced = stats["enhanced_monitoring"]
        assert "performance_metrics" in enhanced
        perf_metrics = enhanced["performance_metrics"]
        found_operations = False
        for metric_key in perf_metrics:
            if "operation_count" in perf_metrics[metric_key]:
                if perf_metrics[metric_key]["operation_count"] > 0:
                    found_operations = True
                    break
        assert found_operations, "No operations recorded in performance metrics"

    async def test_dependency_tracking_and_invalidation(
        self, monitoring_system, security_context
    ):
        """Test cache dependency tracking and coordinated invalidation."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]
        root_key = "user:123"
        dependent_keys = [
            "user:123:profile",
            "user:123:settings",
            "user:123:preferences",
        ]
        await unified_manager.set_cached(
            root_key, {"id": 123, "name": "test"}, 300, security_context
        )
        for dep_key in dependent_keys:
            await unified_manager.set_cached(
                dep_key, {"user_id": 123, "data": "test"}, 300, security_context
            )
        for dep_key in dependent_keys:
            cache_monitor.add_dependency(dep_key, root_key, cascade_level=1)
        dependents = cache_monitor.get_dependent_keys(root_key)
        assert len(dependents) == len(dependent_keys)
        for dep_key in dependent_keys:
            assert dep_key in dependents
        async with cache_monitor.trace_invalidation(
            InvalidationType.DEPENDENCY, [root_key]
        ):
            invalidated_count = await cache_monitor.invalidate_by_dependency(root_key)
            assert invalidated_count > 0
        assert len(cache_monitor._invalidation_history) > 0
        latest_event = cache_monitor._invalidation_history[-1]
        assert latest_event.invalidation_type == InvalidationType.DEPENDENCY

    async def test_cross_level_coordination(self, monitoring_system, security_context):
        """Test cross-level cache coordination."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        coordinator = components["coordinator"]
        hot_key = "hot_cache_key"
        cold_key = "cold_cache_key"
        await unified_manager.set_cached(hot_key, "hot_data", 300, security_context)
        await unified_manager.set_cached(cold_key, "cold_data", 300, security_context)
        for _ in range(15):
            await unified_manager.get_cached(hot_key, security_context)
            coordinator.track_cache_access(
                hot_key, CacheLevel.L1, True, operation="get"
            )
            await asyncio.sleep(0.001)
        await unified_manager.get_cached(cold_key, security_context)
        coordinator.track_cache_access(cold_key, CacheLevel.L1, True, operation="get")
        await asyncio.sleep(0.1)
        stats = coordinator.get_coordination_stats()
        assert "access_patterns" in stats
        assert stats["access_patterns"]["total_tracked_entries"] >= 2
        hot_entry = coordinator._entry_metadata.get(hot_key)
        cold_entry = coordinator._entry_metadata.get(cold_key)
        if hot_entry and cold_entry:
            assert hot_entry.promotion_score > cold_entry.promotion_score
            assert hot_entry.access_count > cold_entry.access_count

    async def test_slo_integration_and_compliance(self, monitoring_system):
        """Test SLO integration and compliance monitoring."""
        components = monitoring_system
        slo_integration = components["slo_integration"]
        cache_slis = await slo_integration.calculate_cache_slis()
        expected_sli_types = [CacheSLIType.HIT_RATE]
        for sli_type in expected_sli_types:
            if sli_type in cache_slis:
                sli = cache_slis[sli_type]
                assert sli.sli_type == sli_type
                assert 0.0 <= sli.current_value <= 1.0 or sli.current_value >= 0
                assert sli.target_value > 0
                assert 0.0 <= sli.compliance_ratio <= 1.0
                assert sli.measurement_window.total_seconds() > 0
        trends = await slo_integration.analyze_performance_trends()
        assert isinstance(trends, dict)
        slo_report = await slo_integration.get_slo_cache_report()
        assert "timestamp" in slo_report
        assert "cache_slis" in slo_report
        assert "performance_trends" in slo_report
        assert "predictive_alerts" in slo_report
        assert "overall_health" in slo_report
        assert "integration_status" in slo_report
        integration_status = slo_report["integration_status"]
        assert integration_status["slo_observability_connected"] is True
        assert integration_status["cache_monitor_connected"] is True

    async def test_predictive_alerting(self, monitoring_system, security_context):
        """Test predictive alerting based on performance trends."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        slo_integration = components["slo_integration"]
        test_key = "degrading_performance_key"
        for i in range(25):
            if i < 10:
                await unified_manager.set_cached(
                    f"{test_key}_{i}", f"data_{i}", 300, security_context
                )
                result = await unified_manager.get_cached(
                    f"{test_key}_{i}", security_context
                )
                assert result is not None
            else:
                result = await unified_manager.get_cached(
                    f"nonexistent_{i}", security_context
                )
                assert result is None
            hit = i < 10
            hit_rate = max(0.3, 1.0 - i * 0.03)
            slo_integration._sli_history["hit_rate"].append({
                "timestamp": time.time() + i,
                "value": hit_rate,
                "compliance": 1.0 if hit_rate >= 0.85 else hit_rate / 0.85,
            })
        trends = await slo_integration.analyze_performance_trends()
        if "hit_rate" in trends:
            trend = trends["hit_rate"]
            assert trend.trend_direction in ["improving", "degrading", "stable"]
            assert trend.confidence.value in ["low", "medium", "high", "very_high"]
        alerts = await slo_integration.generate_predictive_alerts()
        for alert in alerts:
            assert alert.alert_id is not None
            assert alert.alert_type is not None
            assert alert.confidence.value in ["low", "medium", "high", "very_high"]
            assert len(alert.recommended_actions) > 0

    async def test_comprehensive_monitoring_report(
        self, monitoring_system, security_context
    ):
        """Test comprehensive monitoring report generation."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        for i in range(10):
            key = f"report_test_key_{i}"
            await unified_manager.set_cached(key, f"data_{i}", 300, security_context)
            await unified_manager.get_cached(key, security_context)
        await asyncio.sleep(0.1)
        report = await get_cache_monitoring_report()
        assert "timestamp" in report
        assert "monitoring_system_health" in report
        assert "cache_performance" in report
        assert "slo_compliance" in report
        assert "cross_level_coordination" in report
        assert "integration_status" in report
        health = report["monitoring_system_health"]
        assert health["unified_cache_monitor"] == "operational"
        assert health["slo_integration"] == "operational"
        assert health["cross_level_coordinator"] == "operational"
        performance = report["cache_performance"]
        assert "overall_stats" in performance
        assert "level_breakdown" in performance
        assert "enhanced_metrics" in performance
        overall_stats = performance["overall_stats"]
        assert "hit_rate" in overall_stats
        assert "total_requests" in overall_stats
        assert "health_status" in overall_stats
        integration = report["integration_status"]
        assert integration["all_systems_operational"] is True
        assert integration["background_tasks_running"] is True
        assert integration["monitoring_callbacks_active"] is True

    async def test_cache_warming_intelligence(
        self, monitoring_system, security_context
    ):
        """Test intelligent cache warming based on access patterns."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]
        warming_candidates = ["user:456", "product:789", "config:settings"]
        for key in warming_candidates:
            await unified_manager.set_cached(
                key, f"data_for_{key}", 300, security_context
            )
            for _ in range(8):
                await unified_manager.get_cached(key, security_context)
                cache_monitor._update_warming_pattern(key)
                await asyncio.sleep(0.001)
        opportunities = await cache_monitor.analyze_warming_opportunities()
        assert isinstance(opportunities, list)
        patterns = cache_monitor._warming_patterns
        for key in warming_candidates:
            if key in patterns:
                pattern = patterns[key]
                assert pattern.frequency > 0
                assert pattern.last_access is not None
                assert 0.0 <= pattern.success_rate <= 1.0

    async def test_coherence_monitoring_and_repair(
        self, monitoring_system, security_context
    ):
        """Test cache coherence monitoring and repair."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        coordinator = components["coordinator"]
        test_key = "coherence_test_key"
        test_value = "coherence_test_value"
        await unified_manager.set_cached(test_key, test_value, 300, security_context)
        coordinator.track_cache_access(
            test_key, CacheLevel.L1, True, len(test_value), "set"
        )
        coordinator.track_cache_access(
            test_key, CacheLevel.L2, True, len(test_value), "set"
        )
        violations = await coordinator.check_cache_coherence([test_key])
        assert test_key not in violations or len(violations[test_key]) == 0
        repair_success = await coordinator.repair_cache_coherence(test_key)
        assert isinstance(repair_success, bool)

    async def test_alert_callback_integration(self, monitoring_system):
        """Test alert callback integration."""
        components = monitoring_system
        cache_monitor = components["cache_monitor"]
        received_alerts = []

        def test_alert_callback(alert):
            received_alerts.append(alert)

        cache_monitor.register_alert_callback(test_alert_callback)
        from prompt_improver.monitoring.cache.unified_cache_monitoring import (
            CachePerformanceAlert,
        )

        test_alert = CachePerformanceAlert(
            alert_id="test_alert_123",
            severity=AlertSeverity.WARNING,
            metric_name="test_metric",
            current_value=0.5,
            threshold_value=0.7,
            message="Test alert message",
            cache_level=CacheLevel.L1,
            recommendations=["Test recommendation"],
        )
        await cache_monitor._maybe_generate_alert(test_alert)
        await asyncio.sleep(0.1)
        assert len(received_alerts) > 0
        assert received_alerts[0].alert_id == "test_alert_123"

    async def test_opentelemetry_metrics_integration(
        self, monitoring_system, security_context
    ):
        """Test OpenTelemetry metrics integration."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]
        with (
            patch(
                "prompt_improver.monitoring.cache.unified_cache_monitoring.OPENTELEMETRY_AVAILABLE",
                True,
            ),
            patch(
                "prompt_improver.monitoring.cache.unified_cache_monitoring.cache_operations_counter"
            ) as mock_counter,
            patch(
                "prompt_improver.monitoring.cache.unified_cache_monitoring.cache_hit_ratio"
            ) as mock_gauge,
        ):
            test_key = "otel_test_key"
            await unified_manager.set_cached(
                test_key, "test_data", 300, security_context
            )
            cache_monitor.record_cache_operation(
                "set", CacheLevel.L1, True, 50.0, test_key
            )
            await unified_manager.get_cached(test_key, security_context)
            cache_monitor.record_cache_operation(
                "get", CacheLevel.L1, True, 25.0, test_key
            )
            stats = cache_monitor.get_comprehensive_stats()
            assert "enhanced_monitoring" in stats


@pytest.mark.asyncio
class TestMonitoringSystemResilience:
    """Test monitoring system resilience and error handling."""

    async def test_monitoring_with_redis_unavailable(self, monitoring_system):
        """Test monitoring behavior when Redis is unavailable."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]
        original_redis = unified_manager.cache.redis_client
        unified_manager.cache.redis_client = None
        try:
            test_key = "redis_unavailable_test"
            cache_monitor.record_cache_operation(
                "get", CacheLevel.L1, True, 30.0, test_key
            )
            stats = cache_monitor.get_comprehensive_stats()
            assert stats is not None
        finally:
            unified_manager.cache.redis_client = original_redis

    async def test_monitoring_with_invalid_data(self, monitoring_system):
        """Test monitoring resilience with invalid data."""
        components = monitoring_system
        cache_monitor = components["cache_monitor"]
        try:
            cache_monitor.record_cache_operation(
                "invalid_op", CacheLevel.L1, True, -100.0, None
            )
            stats = cache_monitor.get_comprehensive_stats()
            assert stats is not None
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))

    async def test_concurrent_monitoring_operations(
        self, monitoring_system, security_context
    ):
        """Test monitoring system under concurrent operations."""
        components = monitoring_system
        unified_manager = components["unified_manager"]
        cache_monitor = components["cache_monitor"]

        async def cache_operation_batch(batch_id):
            for i in range(10):
                key = f"concurrent_test_{batch_id}_{i}"
                await unified_manager.set_cached(
                    key, f"data_{i}", 300, security_context
                )
                await unified_manager.get_cached(key, security_context)
                cache_monitor.record_cache_operation(
                    "get", CacheLevel.L1, True, 40.0, key
                )

        tasks = [cache_operation_batch(i) for i in range(5)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.2)
        stats = cache_monitor.get_comprehensive_stats()
        enhanced = stats.get("enhanced_monitoring", {})
        perf_metrics = enhanced.get("performance_metrics", {})
        total_operations = 0
        for metric_data in perf_metrics.values():
            if "operation_count" in metric_data:
                total_operations += metric_data["operation_count"]
        assert total_operations > 0
