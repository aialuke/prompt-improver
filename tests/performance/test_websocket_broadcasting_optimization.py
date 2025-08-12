"""
WebSocket Broadcasting Optimization Performance Tests
Validates the 40-60% efficiency improvement from targeted group broadcasting
"""

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocket

from prompt_improver.database import (
    ManagerMode,
    get_unified_manager,
)


class TestWebSocketBroadcastingOptimization:
    """Test WebSocket broadcasting performance optimizations"""

    @pytest.fixture
    def connection_manager(self):
        """Create database services for testing"""
        manager = get_database_services(ManagerMode.ASYNC_MODERN)
        manager.init_websocket_management()
        return manager

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection"""
        mock_ws = Mock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    async def test_group_broadcasting_efficiency(
        self, connection_manager, mock_websocket
    ):
        """Test that group broadcasting is more efficient than broadcast_to_all"""
        dashboard_connections = [Mock(spec=WebSocket) for _ in range(50)]
        experiment_connections = [Mock(spec=WebSocket) for _ in range(100)]
        for ws in dashboard_connections + experiment_connections:
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.close = AsyncMock()
        for i, ws in enumerate(dashboard_connections):
            await connection_manager.websocket_connect_to_group(
                ws, "analytics_dashboard", f"user_{i}"
            )
        for i, ws in enumerate(experiment_connections):
            await connection_manager.websocket_connect(
                ws, "test_experiment", f"user_{i}"
            )
        message = {"type": "dashboard_update", "data": "test"}
        start_time = time.time()
        await connection_manager.websocket_broadcast_to_group(
            "analytics_dashboard", message
        )
        group_broadcast_time = time.time() - start_time
        for ws in dashboard_connections:
            ws.send_text.assert_called_once()
        for ws in experiment_connections:
            ws.send_text.assert_not_called()
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.reset_mock()
        start_time = time.time()
        # Note: broadcast_to_all not available in unified manager - using targeted broadcasting
        await connection_manager.websocket_broadcast_to_group(
            "analytics_dashboard", message
        )
        await connection_manager.websocket_broadcast_to_experiment(
            "test_experiment", message
        )
        all_broadcast_time = time.time() - start_time
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.assert_called_once()
        efficiency_improvement = (
            (all_broadcast_time - group_broadcast_time) / all_broadcast_time * 100
        )
        assert efficiency_improvement > 40, (
            f"Expected >40% efficiency improvement, got {efficiency_improvement:.1f}%"
        )
        assert group_broadcast_time < all_broadcast_time, (
            "Group broadcasting should be faster than broadcast_to_all"
        )
        assert (
            connection_manager.get_websocket_connection_count(
                group_id="analytics_dashboard"
            )
            == 50
        )
        assert (
            connection_manager.get_websocket_connection_count(
                experiment_id="test_experiment"
            )
            == 100
        )
        assert connection_manager.get_websocket_connection_count() == 150

    async def test_connection_limits_enforcement(self, connection_manager):
        """Test connection limit enforcement"""
        connection_manager.MAX_CONNECTIONS_PER_GROUP = 10
        mock_connections = []
        for i in range(15):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.close = AsyncMock()
            mock_connections.append(ws)
        for i in range(10):
            await connection_manager.connect_to_group(
                mock_connections[i], "test_group", f"user_{i}"
            )
        assert connection_manager.get_connection_count(group_id="test_group") == 10
        for i in range(10, 15):
            await connection_manager.connect_to_group(
                mock_connections[i], "test_group", f"user_{i}"
            )
            mock_connections[i].close.assert_called_once()
        assert connection_manager.get_connection_count(group_id="test_group") == 10

    async def test_rate_limiting(self, connection_manager):
        """Test message rate limiting"""
        connection_manager.MAX_MESSAGES_PER_SECOND = 5
        ws = Mock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        await connection_manager.connect_to_group(ws, "rate_test_group", "user1")
        message = {"type": "test", "data": "rate_limit_test"}
        messages_sent = 0
        for i in range(10):
            await connection_manager.broadcast_to_group("rate_test_group", message)
            messages_sent += 1
        assert ws.send_text.call_count <= connection_manager.MAX_MESSAGES_PER_SECOND + 2

    async def test_connection_stats_reporting(self, connection_manager):
        """Test comprehensive connection statistics"""
        dashboard_ws = Mock(spec=WebSocket)
        dashboard_ws.accept = AsyncMock()
        dashboard_ws.send_text = AsyncMock()
        experiment_ws = Mock(spec=WebSocket)
        experiment_ws.accept = AsyncMock()
        experiment_ws.send_text = AsyncMock()
        session_ws = Mock(spec=WebSocket)
        session_ws.accept = AsyncMock()
        session_ws.send_text = AsyncMock()
        await connection_manager.connect_to_group(
            dashboard_ws, "analytics_dashboard", "user1"
        )
        await connection_manager.connect(experiment_ws, "test_experiment", "user2")
        await connection_manager.connect_to_group(session_ws, "session_123", "user3")
        stats = connection_manager.get_connection_stats()
        assert stats["total_connections"] == 3
        assert stats["experiment_connections"] == 1
        assert stats["group_connections"] == 2
        assert stats["active_experiments"] == 1
        assert stats["active_groups"] == 2
        assert "analytics_dashboard" in stats["group_details"]
        assert "session_123" in stats["group_details"]
        assert "test_experiment" in stats["experiment_details"]
        assert stats["group_details"]["analytics_dashboard"] == 1
        assert stats["group_details"]["session_123"] == 1
        assert stats["experiment_details"]["test_experiment"] == 1

    async def test_message_targeting_accuracy(self, connection_manager):
        """Test that messages are sent only to intended recipients"""
        dashboard_connections = []
        session_connections = []
        experiment_connections = []
        for i in range(3):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            dashboard_connections.append(ws)
            await connection_manager.connect_to_group(
                ws, "analytics_dashboard", f"dashboard_user_{i}"
            )
        for i in range(2):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            session_connections.append(ws)
            await connection_manager.connect_to_group(
                ws, "session_abc", f"session_user_{i}"
            )
        for i in range(4):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            experiment_connections.append(ws)
            await connection_manager.connect(ws, "experiment_123", f"exp_user_{i}")
        dashboard_message = {"type": "dashboard_update", "data": "dashboard_data"}
        await connection_manager.broadcast_to_group(
            "analytics_dashboard", dashboard_message
        )
        for ws in dashboard_connections:
            ws.send_text.assert_called_once()
        for ws in session_connections + experiment_connections:
            ws.send_text.assert_not_called()
        for ws in dashboard_connections + session_connections + experiment_connections:
            ws.send_text.reset_mock()
        session_message = {"type": "session_update", "session_id": "abc"}
        await connection_manager.broadcast_to_group("session_abc", session_message)
        for ws in session_connections:
            ws.send_text.assert_called_once()
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.assert_not_called()
        for ws in dashboard_connections + session_connections + experiment_connections:
            ws.send_text.reset_mock()
        experiment_message = {"type": "experiment_update", "experiment_id": "123"}
        await connection_manager.broadcast_to_experiment(
            "experiment_123", experiment_message
        )
        for ws in experiment_connections:
            ws.send_text.assert_called_once()
        for ws in dashboard_connections + session_connections:
            ws.send_text.assert_not_called()

    def test_performance_targets(self, connection_manager):
        """Test that performance targets are met"""
        assert connection_manager.MAX_CONNECTIONS_PER_GROUP >= 1000, (
            "Should support 1000+ connections per group"
        )
        assert connection_manager.MAX_MESSAGES_PER_SECOND >= 100, (
            "Should support 100+ messages per second"
        )
        assert hasattr(connection_manager, "group_connections"), (
            "Should have group_connections for targeted broadcasting"
        )
        assert hasattr(connection_manager, "broadcast_to_group"), (
            "Should have broadcast_to_group method"
        )
        assert hasattr(connection_manager, "_check_rate_limit"), (
            "Should have rate limiting"
        )
        assert hasattr(connection_manager, "_enforce_connection_limits"), (
            "Should have connection limits"
        )

    async def test_cleanup_all_connection_types(self, connection_manager):
        """Test that cleanup handles both experiment and group connections"""
        dashboard_ws = Mock(spec=WebSocket)
        dashboard_ws.accept = AsyncMock()
        dashboard_ws.send_text = AsyncMock()
        dashboard_ws.close = AsyncMock()
        experiment_ws = Mock(spec=WebSocket)
        experiment_ws.accept = AsyncMock()
        experiment_ws.send_text = AsyncMock()
        experiment_ws.close = AsyncMock()
        await connection_manager.connect_to_group(
            dashboard_ws, "analytics_dashboard", "user1"
        )
        await connection_manager.connect(experiment_ws, "test_experiment", "user2")
        assert connection_manager.get_connection_count() == 2
        await connection_manager.cleanup()
        dashboard_ws.close.assert_called_once()
        experiment_ws.close.assert_called_once()
        assert len(connection_manager.group_connections) == 0
        assert len(connection_manager.experiment_connections) == 0
        assert len(connection_manager.connection_metadata) == 0
        assert len(connection_manager._message_counters) == 0


@pytest.mark.asyncio
class TestPerformanceBenchmark:
    """Integration performance tests"""

    async def test_optimization_effectiveness_simulation(self):
        """Simulate the 40-60% optimization effectiveness"""
        connection_manager = ConnectionManager()
        experiment_connections = []
        dashboard_connections = []
        for i in range(100):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            experiment_connections.append(ws)
            await connection_manager.connect(ws, "busy_experiment", f"exp_user_{i}")
        for i in range(50):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            dashboard_connections.append(ws)
            await connection_manager.connect_to_group(
                ws, "analytics_dashboard", f"dash_user_{i}"
            )
        dashboard_message = {"type": "dashboard_update", "data": "efficiency_test"}
        start_time = time.time()
        await connection_manager.broadcast_to_group(
            "analytics_dashboard", dashboard_message
        )
        efficient_time = time.time() - start_time
        efficient_messages = sum(
            1 for ws in dashboard_connections if ws.send_text.called
        )
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.reset_mock()
        start_time = time.time()
        await connection_manager.broadcast_to_all(dashboard_message)
        inefficient_time = time.time() - start_time
        inefficient_messages = sum(
            1
            for ws in dashboard_connections + experiment_connections
            if ws.send_text.called
        )
        message_efficiency = (
            (inefficient_messages - efficient_messages) / inefficient_messages * 100
        )
        time_efficiency = max(
            0, (inefficient_time - efficient_time) / inefficient_time * 100
        )
        assert efficient_messages == 50, "Should send to 50 dashboard connections only"
        assert inefficient_messages == 150, "Should send to all 150 connections"
        assert message_efficiency >= 66.7, (
            f"Should achieve >66% message efficiency, got {message_efficiency:.1f}%"
        )
        total_efficiency = (message_efficiency + time_efficiency) / 2
        assert total_efficiency >= 40, (
            f"Should achieve >40% total efficiency improvement, got {total_efficiency:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
