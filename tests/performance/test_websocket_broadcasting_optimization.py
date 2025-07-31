"""
WebSocket Broadcasting Optimization Performance Tests
Validates the 40-60% efficiency improvement from targeted group broadcasting
"""

import asyncio
import json
import pytest
import time
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from fastapi import WebSocket
from prompt_improver.utils.websocket_manager import ConnectionManager


class TestWebSocketBroadcastingOptimization:
    """Test WebSocket broadcasting performance optimizations"""

    @pytest.fixture
    def connection_manager(self):
        """Create connection manager for testing"""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection"""
        mock_ws = Mock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    async def test_group_broadcasting_efficiency(self, connection_manager, mock_websocket):
        """Test that group broadcasting is more efficient than broadcast_to_all"""
        
        # Create multiple mock WebSocket connections
        dashboard_connections = [Mock(spec=WebSocket) for _ in range(50)]
        experiment_connections = [Mock(spec=WebSocket) for _ in range(100)]
        
        # Setup mock send_text for all connections
        for ws in dashboard_connections + experiment_connections:
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.close = AsyncMock()
        
        # Connect dashboard clients to dashboard group
        for i, ws in enumerate(dashboard_connections):
            await connection_manager.connect_to_group(ws, "analytics_dashboard", f"user_{i}")
        
        # Connect experiment clients to experiment
        for i, ws in enumerate(experiment_connections):
            await connection_manager.connect(ws, "test_experiment", f"user_{i}")
        
        # Test 1: Measure broadcast_to_group performance (dashboard only)
        message = {"type": "dashboard_update", "data": "test"}
        
        start_time = time.time()
        await connection_manager.broadcast_to_group("analytics_dashboard", message)
        group_broadcast_time = time.time() - start_time
        
        # Verify only dashboard connections received the message
        for ws in dashboard_connections:
            ws.send_text.assert_called_once()
        for ws in experiment_connections:
            ws.send_text.assert_not_called()
            
        # Reset mocks
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.reset_mock()
        
        # Test 2: Measure broadcast_to_all performance (all connections)
        start_time = time.time()
        await connection_manager.broadcast_to_all(message)
        all_broadcast_time = time.time() - start_time
        
        # Verify all connections received the message
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.assert_called_once()
        
        # Performance Analysis
        efficiency_improvement = (all_broadcast_time - group_broadcast_time) / all_broadcast_time * 100
        
        # The group broadcast should be significantly more efficient
        assert efficiency_improvement > 40, f"Expected >40% efficiency improvement, got {efficiency_improvement:.1f}%"
        assert group_broadcast_time < all_broadcast_time, "Group broadcasting should be faster than broadcast_to_all"
        
        # Connection count validation
        assert connection_manager.get_connection_count(group_id="analytics_dashboard") == 50
        assert connection_manager.get_connection_count(experiment_id="test_experiment") == 100
        assert connection_manager.get_connection_count() == 150  # Total

    async def test_connection_limits_enforcement(self, connection_manager):
        """Test connection limit enforcement"""
        
        # Set low connection limit for testing
        connection_manager.MAX_CONNECTIONS_PER_GROUP = 10
        
        mock_connections = []
        for i in range(15):  # Try to connect more than the limit
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.close = AsyncMock()
            mock_connections.append(ws)
        
        # Connect up to the limit
        for i in range(10):
            await connection_manager.connect_to_group(mock_connections[i], "test_group", f"user_{i}")
        
        # Verify we have 10 connections
        assert connection_manager.get_connection_count(group_id="test_group") == 10
        
        # Try to connect beyond the limit - these should be rejected
        for i in range(10, 15):
            await connection_manager.connect_to_group(mock_connections[i], "test_group", f"user_{i}")
            # Connection should be closed due to limit
            mock_connections[i].close.assert_called_once()
        
        # Should still have only 10 connections
        assert connection_manager.get_connection_count(group_id="test_group") == 10

    async def test_rate_limiting(self, connection_manager):
        """Test message rate limiting"""
        
        # Set low rate limit for testing
        connection_manager.MAX_MESSAGES_PER_SECOND = 5
        
        # Create mock connection
        ws = Mock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        
        await connection_manager.connect_to_group(ws, "rate_test_group", "user1")
        
        # Send messages rapidly
        message = {"type": "test", "data": "rate_limit_test"}
        messages_sent = 0
        
        for i in range(10):  # Try to send 10 messages quickly
            await connection_manager.broadcast_to_group("rate_test_group", message)
            messages_sent += 1
        
        # Due to rate limiting, not all messages should be sent
        # The exact count depends on timing, but should be limited
        assert ws.send_text.call_count <= connection_manager.MAX_MESSAGES_PER_SECOND + 2  # Allow small variance

    async def test_connection_stats_reporting(self, connection_manager):
        """Test comprehensive connection statistics"""
        
        # Create various connection types
        dashboard_ws = Mock(spec=WebSocket)
        dashboard_ws.accept = AsyncMock()
        dashboard_ws.send_text = AsyncMock()
        
        experiment_ws = Mock(spec=WebSocket)
        experiment_ws.accept = AsyncMock()
        experiment_ws.send_text = AsyncMock()
        
        session_ws = Mock(spec=WebSocket)
        session_ws.accept = AsyncMock()
        session_ws.send_text = AsyncMock()
        
        # Connect to different groups
        await connection_manager.connect_to_group(dashboard_ws, "analytics_dashboard", "user1")
        await connection_manager.connect(experiment_ws, "test_experiment", "user2")
        await connection_manager.connect_to_group(session_ws, "session_123", "user3")
        
        # Get connection statistics
        stats = connection_manager.get_connection_stats()
        
        # Validate statistics
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
        
        # Create connections for different groups
        dashboard_connections = []
        session_connections = []
        experiment_connections = []
        
        # Dashboard group
        for i in range(3):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            dashboard_connections.append(ws)
            await connection_manager.connect_to_group(ws, "analytics_dashboard", f"dashboard_user_{i}")
        
        # Session group
        for i in range(2):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            session_connections.append(ws)
            await connection_manager.connect_to_group(ws, "session_abc", f"session_user_{i}")
        
        # Experiment group
        for i in range(4):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            experiment_connections.append(ws)
            await connection_manager.connect(ws, "experiment_123", f"exp_user_{i}")
        
        # Test dashboard broadcast targeting
        dashboard_message = {"type": "dashboard_update", "data": "dashboard_data"}
        await connection_manager.broadcast_to_group("analytics_dashboard", dashboard_message)
        
        # Verify only dashboard connections received the message
        for ws in dashboard_connections:
            ws.send_text.assert_called_once()
        for ws in session_connections + experiment_connections:
            ws.send_text.assert_not_called()
        
        # Reset mocks
        for ws in dashboard_connections + session_connections + experiment_connections:
            ws.send_text.reset_mock()
        
        # Test session broadcast targeting
        session_message = {"type": "session_update", "session_id": "abc"}
        await connection_manager.broadcast_to_group("session_abc", session_message)
        
        # Verify only session connections received the message
        for ws in session_connections:
            ws.send_text.assert_called_once()
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.assert_not_called()
        
        # Reset mocks
        for ws in dashboard_connections + session_connections + experiment_connections:
            ws.send_text.reset_mock()
        
        # Test experiment broadcast targeting
        experiment_message = {"type": "experiment_update", "experiment_id": "123"}
        await connection_manager.broadcast_to_experiment("experiment_123", experiment_message)
        
        # Verify only experiment connections received the message
        for ws in experiment_connections:
            ws.send_text.assert_called_once()
        for ws in dashboard_connections + session_connections:
            ws.send_text.assert_not_called()

    def test_performance_targets(self, connection_manager):
        """Test that performance targets are met"""
        
        # Performance targets from the specification
        assert connection_manager.MAX_CONNECTIONS_PER_GROUP >= 1000, "Should support 1000+ connections per group"
        assert connection_manager.MAX_MESSAGES_PER_SECOND >= 100, "Should support 100+ messages per second"
        
        # Validate that connection manager has the required optimization features
        assert hasattr(connection_manager, 'group_connections'), "Should have group_connections for targeted broadcasting"
        assert hasattr(connection_manager, 'broadcast_to_group'), "Should have broadcast_to_group method"
        assert hasattr(connection_manager, '_check_rate_limit'), "Should have rate limiting"
        assert hasattr(connection_manager, '_enforce_connection_limits'), "Should have connection limits"

    async def test_cleanup_all_connection_types(self, connection_manager):
        """Test that cleanup handles both experiment and group connections"""
        
        # Create various connection types
        dashboard_ws = Mock(spec=WebSocket)
        dashboard_ws.accept = AsyncMock()
        dashboard_ws.send_text = AsyncMock()
        dashboard_ws.close = AsyncMock()
        
        experiment_ws = Mock(spec=WebSocket)
        experiment_ws.accept = AsyncMock()
        experiment_ws.send_text = AsyncMock()
        experiment_ws.close = AsyncMock()
        
        # Connect to different groups
        await connection_manager.connect_to_group(dashboard_ws, "analytics_dashboard", "user1")
        await connection_manager.connect(experiment_ws, "test_experiment", "user2")
        
        # Verify connections exist
        assert connection_manager.get_connection_count() == 2
        
        # Cleanup all connections
        await connection_manager.cleanup()
        
        # Verify all connections were closed
        dashboard_ws.close.assert_called_once()
        experiment_ws.close.assert_called_once()
        
        # Verify all data structures are cleared
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
        
        # Simulate scenario: 100 experiment connections, 50 dashboard users
        # Dashboard update should only go to dashboard users, not all connections
        
        experiment_connections = []
        dashboard_connections = []
        
        # Create experiment connections
        for i in range(100):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            experiment_connections.append(ws)
            await connection_manager.connect(ws, "busy_experiment", f"exp_user_{i}")
        
        # Create dashboard connections
        for i in range(50):
            ws = Mock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            dashboard_connections.append(ws)
            await connection_manager.connect_to_group(ws, "analytics_dashboard", f"dash_user_{i}")
        
        # Test efficient broadcasting (dashboard update to dashboard group only)
        dashboard_message = {"type": "dashboard_update", "data": "efficiency_test"}
        
        start_time = time.time()
        await connection_manager.broadcast_to_group("analytics_dashboard", dashboard_message)
        efficient_time = time.time() - start_time
        
        # Count messages sent efficiently (only to dashboard users)
        efficient_messages = sum(1 for ws in dashboard_connections if ws.send_text.called)
        
        # Reset mocks
        for ws in dashboard_connections + experiment_connections:
            ws.send_text.reset_mock()
        
        # Test inefficient broadcasting (broadcast_to_all)
        start_time = time.time()
        await connection_manager.broadcast_to_all(dashboard_message)
        inefficient_time = time.time() - start_time
        
        # Count total messages sent inefficiently (to all connections)
        inefficient_messages = sum(1 for ws in dashboard_connections + experiment_connections if ws.send_text.called)
        
        # Calculate efficiency metrics
        message_efficiency = (inefficient_messages - efficient_messages) / inefficient_messages * 100
        time_efficiency = max(0, (inefficient_time - efficient_time) / inefficient_time * 100)
        
        # Validate optimization effectiveness
        assert efficient_messages == 50, "Should send to 50 dashboard connections only"
        assert inefficient_messages == 150, "Should send to all 150 connections"
        assert message_efficiency >= 66.7, f"Should achieve >66% message efficiency, got {message_efficiency:.1f}%"
        
        # The optimization should result in significant efficiency gains
        total_efficiency = (message_efficiency + time_efficiency) / 2
        assert total_efficiency >= 40, f"Should achieve >40% total efficiency improvement, got {total_efficiency:.1f}%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])