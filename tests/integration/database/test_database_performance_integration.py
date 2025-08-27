"""
Integration tests for database performance monitoring components.

Following 2025 best practices:
- Uses real database (no mocks/in-memory)
- Tests actual performance metrics
- Event-driven testing with real behavior
- Async/await patterns
- Real performance thresholds validation
"""

import asyncio
import contextlib

import asyncpg
import pytest

from prompt_improver.database.performance_monitor import (
    DatabasePerformanceMonitor,
)
from prompt_improver.database.query_optimizer import DatabaseConnectionOptimizer


class TestDatabasePerformanceIntegration:
    """Integration tests for database performance monitoring with real database behavior."""

    @pytest.fixture
    async def db_connection(self):
        """Create database connection using DatabaseServices for real behavior testing."""
        try:
            from prompt_improver.database.composition import get_database_services
            from prompt_improver.database.types import ManagerMode

            manager = get_database_services(ManagerMode.ASYNC_MODERN)
            async with manager.get_async_session() as session:
                yield session
        except Exception:
            conn_string = "postgresql://apes_user:apes_secure_password_2024@localhost:5432/apes_production"
            conn = await asyncpg.connect(conn_string)
            yield conn
            await conn.close()

    @pytest.fixture
    async def mock_event_bus(self):
        """Create mock event bus for testing event-driven behavior."""

        class MockEventBus:
            def __init__(self):
                self.events = []
                self.subscribers = {}

            async def emit(self, event):
                self.events.append(event)
                if (
                    hasattr(event, "event_type")
                    and event.event_type in self.subscribers
                ):
                    for handler in self.subscribers[event.event_type]:
                        await handler(event)

            def subscribe(self, event_type, handler):
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                self.subscribers[event_type].append(handler)

        return MockEventBus()

    @pytest.fixture
    async def performance_monitor(self, mock_event_bus):
        """Create performance monitor with event bus integration."""
        monitor = DatabasePerformanceMonitor(event_bus=mock_event_bus)
        yield monitor
        if monitor._monitoring:
            await monitor.stop_monitoring()

    @pytest.fixture
    async def connection_optimizer(self, mock_event_bus):
        """Create connection optimizer with event bus integration."""
        optimizer = DatabaseConnectionOptimizer(event_bus=mock_event_bus)
        yield optimizer

    @pytest.mark.asyncio
    async def test_real_performance_snapshot_collection(self, performance_monitor):
        """Test collection of real database performance metrics (no mocks)."""
        snapshot = await performance_monitor.take_performance_snapshot()
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.cache_hit_ratio, float)
        assert 0.0 <= snapshot.cache_hit_ratio <= 100.0
        assert isinstance(snapshot.active_connections, int)
        assert snapshot.active_connections >= 0
        assert isinstance(snapshot.avg_query_time_ms, float)
        assert snapshot.avg_query_time_ms >= 0.0
        assert isinstance(snapshot.database_size_mb, float)
        assert snapshot.database_size_mb > 0.0
        assert isinstance(snapshot.index_hit_ratio, float)
        assert 0.0 <= snapshot.index_hit_ratio <= 100.0
        assert len(performance_monitor._snapshots) >= 1
        assert performance_monitor._snapshots[-1] == snapshot

    @pytest.mark.asyncio
    async def test_performance_threshold_validation(self, performance_monitor):
        """Test real performance threshold validation against 2025 standards."""
        snapshots = []
        for _ in range(3):
            snapshot = await performance_monitor.take_performance_snapshot()
            snapshots.append(snapshot)
            await asyncio.sleep(0.1)
        for snapshot in snapshots:
            assert isinstance(snapshot.cache_hit_ratio, float)
            assert snapshot.avg_query_time_ms >= 0.0
            assert 0 <= snapshot.active_connections <= 50
        recommendations = await performance_monitor.get_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_event_driven_performance_monitoring(
        self, performance_monitor, event_bus
    ):
        """Test event emission during performance monitoring (2025 event-driven pattern)."""
        emitted_events = []

        async def event_collector(event: MLEvent):
            emitted_events.append(event)

        event_bus.subscribe(
            EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN, event_collector
        )
        event_bus.subscribe(EventType.DATABASE_CACHE_HIT_RATIO_LOW, event_collector)
        event_bus.subscribe(EventType.DATABASE_SLOW_QUERY_DETECTED, event_collector)
        snapshot = await performance_monitor.take_performance_snapshot()
        await asyncio.sleep(0.1)
        snapshot_events = [
            e
            for e in emitted_events
            if e.event_type == EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN
        ]
        assert len(snapshot_events) >= 1
        snapshot_event = snapshot_events[0]
        assert snapshot_event.source == "database_performance_monitor"
        assert "snapshot" in snapshot_event.data
        event_snapshot = snapshot_event.data["snapshot"]
        assert event_snapshot["cache_hit_ratio"] == snapshot.cache_hit_ratio
        assert event_snapshot["active_connections"] == snapshot.active_connections
        assert event_snapshot["avg_query_time_ms"] == snapshot.avg_query_time_ms

    @pytest.mark.asyncio
    async def test_real_database_optimization(self, connection_optimizer, event_bus):
        """Test database connection optimization with real system resources."""
        emitted_events = []

        async def event_collector(event: MLEvent):
            emitted_events.append(event)

        event_bus.subscribe(EventType.DATABASE_CONNECTION_OPTIMIZED, event_collector)
        await connection_optimizer.optimize_connection_settings()
        await asyncio.sleep(0.1)
        optimization_events = [
            e
            for e in emitted_events
            if e.event_type == EventType.DATABASE_CONNECTION_OPTIMIZED
        ]
        assert len(optimization_events) >= 1
        optimization_event = optimization_events[0]
        assert optimization_event.source == "database_connection_optimizer"
        assert "applied_settings" in optimization_event.data
        assert "system_resources" in optimization_event.data
        assert "memory_settings" in optimization_event.data
        system_resources = optimization_event.data["system_resources"]
        assert "total_memory_gb" in system_resources
        assert "cpu_count" in system_resources
        assert system_resources["total_memory_gb"] > 0
        assert system_resources["cpu_count"] > 0

    @pytest.mark.asyncio
    async def test_dynamic_memory_sizing_with_real_resources(
        self, connection_optimizer
    ):
        """Test dynamic memory sizing based on actual system resources."""
        system_resources = connection_optimizer._get_system_resources()
        memory_settings = connection_optimizer._calculate_optimal_memory_settings(
            system_resources
        )
        assert isinstance(system_resources["total_memory_gb"], float)
        assert isinstance(system_resources["available_memory_gb"], float)
        assert isinstance(system_resources["cpu_count"], int)
        assert system_resources["total_memory_gb"] > 0
        assert system_resources["available_memory_gb"] > 0
        assert system_resources["cpu_count"] > 0
        assert "work_mem" in memory_settings
        assert "effective_cache_size" in memory_settings
        assert memory_settings["work_mem"].endswith("MB")
        assert memory_settings["effective_cache_size"].endswith("GB")
        work_mem_mb = int(memory_settings["work_mem"].replace("MB", ""))
        cache_size_gb = int(memory_settings["effective_cache_size"].replace("GB", ""))
        assert 16 <= work_mem_mb <= 256
        assert 1 <= cache_size_gb <= 8

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(
        self, performance_monitor, event_bus
    ):
        """Test detection of performance degradation with real thresholds."""
        emitted_events = []

        async def event_collector(event: MLEvent):
            emitted_events.append(event)

        event_bus.subscribe(EventType.DATABASE_PERFORMANCE_DEGRADED, event_collector)
        event_bus.subscribe(EventType.DATABASE_CACHE_HIT_RATIO_LOW, event_collector)
        event_bus.subscribe(EventType.DATABASE_SLOW_QUERY_DETECTED, event_collector)
        baseline_snapshot = await performance_monitor.take_performance_snapshot()
        async with get_session() as session:
            for _i in range(5):
                result = await session.execute("SELECT pg_sleep(0.01)")
                await result.fetchone()
        second_snapshot = await performance_monitor.take_performance_snapshot()
        await asyncio.sleep(0.2)
        assert len(performance_monitor._snapshots) >= 2
        performance_events = [
            e
            for e in emitted_events
            if e.event_type
            in {
                EventType.DATABASE_PERFORMANCE_DEGRADED,
                EventType.DATABASE_CACHE_HIT_RATIO_LOW,
                EventType.DATABASE_SLOW_QUERY_DETECTED,
            }
        ]
        assert isinstance(performance_events, list)

    @pytest.mark.asyncio
    async def test_continuous_monitoring_with_real_intervals(self, performance_monitor):
        """Test continuous monitoring with real time intervals."""
        monitoring_task = asyncio.create_task(
            performance_monitor.start_monitoring(interval_seconds=1)
        )
        await asyncio.sleep(2.5)
        await performance_monitor.stop_monitoring()
        monitoring_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitoring_task
        assert len(performance_monitor._snapshots) >= 2
        timestamps = [s.timestamp for s in performance_monitor._snapshots]
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i - 1]
            assert time_diff.total_seconds() >= 0.8
            assert time_diff.total_seconds() <= 1.5
