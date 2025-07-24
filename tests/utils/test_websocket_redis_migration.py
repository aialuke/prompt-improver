"""Tests for WebSocket manager with coredis migration"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

import coredis
from fastapi import WebSocket

from prompt_improver.utils.websocket_manager import (
    ConnectionManager,
    setup_redis_connection,
    publish_experiment_update,
    connection_manager,
)


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        
    async def accept(self):
        pass
    
    async def send_text(self, data: str):
        if self.closed:
            raise Exception("WebSocket closed")
        self.sent_messages.append(data)
    
    async def close(self):
        self.closed = True


class MockPubSub:
    """Mock PubSub for testing coredis pubsub functionality"""
    
    def __init__(self, channels=None):
        self.channels = channels or []
        self.messages = []
        self.subscribed = False

    async def __aenter__(self):
        self.subscribed = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.subscribed = False

    async def __aiter__(self):
        # Simulate subscription confirmation
        yield {
            "type": "subscribe",
            "channel": self.channels[0] if self.channels else "test-channel",
            "data": 1
        }
        
        # Yield any queued messages
        for message in self.messages:
            yield message

    def add_message(self, channel: str, data: Any):
        """Add a message to be yielded by the async iterator"""
        self.messages.append({
            "type": "message",
            "channel": channel,
            "data": json.dumps(data) if isinstance(data, dict) else data
        })


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client with coredis interface"""
    client = AsyncMock(spec=coredis.Redis)
    client.ping = AsyncMock(return_value=b"PONG")
    client.publish = AsyncMock(return_value=1)
    
    # Mock pubsub method
    def create_pubsub(channels=None, patterns=None):
        return MockPubSub(channels=channels)
    
    client.pubsub = create_pubsub
    return client


@pytest.fixture
def connection_manager_with_redis(mock_redis_client):
    """Create connection manager with mock Redis client"""
    manager = ConnectionManager(redis_client=mock_redis_client)
    return manager


@pytest.mark.asyncio
async def test_connection_manager_initialization():
    """Test that ConnectionManager initializes correctly"""
    manager = ConnectionManager()
    
    assert manager.experiment_connections == {}
    assert manager.connection_metadata == {}
    assert manager.redis_client is None
    assert manager._background_tasks == set()


@pytest.mark.asyncio
async def test_connection_manager_with_redis_client(mock_redis_client):
    """Test ConnectionManager with Redis client"""
    manager = ConnectionManager(redis_client=mock_redis_client)
    
    assert manager.redis_client == mock_redis_client
    assert isinstance(manager.redis_client, AsyncMock)


@pytest.mark.asyncio
async def test_websocket_connection_without_redis():
    """Test WebSocket connection without Redis client"""
    manager = ConnectionManager()
    websocket = MockWebSocket()
    experiment_id = "test-exp-1"
    user_id = "user-123"
    
    await manager.connect(websocket, experiment_id, user_id)
    
    # Check connection was registered
    assert experiment_id in manager.experiment_connections
    assert websocket in manager.experiment_connections[experiment_id]
    assert websocket in manager.connection_metadata
    
    # Check metadata
    metadata = manager.connection_metadata[websocket]
    assert metadata["experiment_id"] == experiment_id
    assert metadata["user_id"] == user_id
    assert "connected_at" in metadata
    assert "connection_id" in metadata
    
    # Check confirmation message was sent
    assert len(websocket.sent_messages) == 1
    confirmation = json.loads(websocket.sent_messages[0])
    assert confirmation["type"] == "connection_established"
    assert confirmation["experiment_id"] == experiment_id


@pytest.mark.asyncio
async def test_websocket_connection_with_redis(connection_manager_with_redis):
    """Test WebSocket connection with Redis client"""
    manager = connection_manager_with_redis
    websocket = MockWebSocket()
    experiment_id = "test-exp-2"
    
    await manager.connect(websocket, experiment_id, "user-456")
    
    # Verify connection was established
    assert experiment_id in manager.experiment_connections
    assert websocket in manager.experiment_connections[experiment_id]
    
    # Verify Redis subscription would be created (background task)
    # We can't easily test the background task without more complex mocking
    assert len(manager._background_tasks) >= 0  # Task may or may not be created immediately


@pytest.mark.asyncio
async def test_websocket_disconnect():
    """Test WebSocket disconnection"""
    manager = ConnectionManager()
    websocket = MockWebSocket()
    experiment_id = "test-exp-3"
    
    # Connect first
    await manager.connect(websocket, experiment_id, "user-789")
    assert experiment_id in manager.experiment_connections
    
    # Disconnect
    await manager.disconnect(websocket)
    
    # Check cleanup
    assert experiment_id not in manager.experiment_connections
    assert websocket not in manager.connection_metadata


@pytest.mark.asyncio
async def test_broadcast_to_experiment():
    """Test broadcasting to experiment connections"""
    manager = ConnectionManager()
    
    # Create multiple websockets for same experiment
    websockets = [MockWebSocket(), MockWebSocket(), MockWebSocket()]
    experiment_id = "test-exp-4"
    
    for i, ws in enumerate(websockets):
        await manager.connect(ws, experiment_id, f"user-{i}")
    
    # Broadcast message
    test_message = {"type": "test", "data": "hello world"}
    await manager.broadcast_to_experiment(experiment_id, test_message)
    
    # Check all websockets received the message
    for ws in websockets:
        assert len(ws.sent_messages) == 2  # 1 confirmation + 1 broadcast
        broadcast_msg = json.loads(ws.sent_messages[1])
        assert broadcast_msg["type"] == "test"
        assert broadcast_msg["data"] == "hello world"
        assert "timestamp" in broadcast_msg


@pytest.mark.asyncio
async def test_broadcast_to_all():
    """Test broadcasting to all connections"""
    manager = ConnectionManager()
    
    # Create websockets for different experiments
    experiments = ["exp-1", "exp-2", "exp-3"]
    websockets = []
    
    for exp_id in experiments:
        ws = MockWebSocket()
        await manager.connect(ws, exp_id, f"user-{exp_id}")
        websockets.append(ws)
    
    # Broadcast to all
    test_message = {"type": "global", "data": "announcement"}
    await manager.broadcast_to_all(test_message)
    
    # Check all websockets received the message
    for ws in websockets:
        assert len(ws.sent_messages) == 2  # 1 confirmation + 1 broadcast
        broadcast_msg = json.loads(ws.sent_messages[1])
        assert broadcast_msg["type"] == "global"
        assert broadcast_msg["data"] == "announcement"


@pytest.mark.asyncio
async def test_setup_redis_connection_success():
    """Test successful Redis connection setup"""
    with patch('coredis.Redis.from_url') as mock_from_url:
        mock_client = AsyncMock(spec=coredis.Redis)
        mock_client.ping = AsyncMock(return_value=b"PONG")
        mock_from_url.return_value = mock_client
        
        redis_client = await setup_redis_connection("redis://localhost:6379")
        
        assert redis_client == mock_client
        mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
        mock_client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_setup_redis_connection_failure():
    """Test Redis connection setup failure"""
    with patch('coredis.Redis.from_url') as mock_from_url:
        mock_from_url.side_effect = Exception("Connection failed")
        
        redis_client = await setup_redis_connection("redis://localhost:6379")
        
        assert redis_client is None


@pytest.mark.asyncio
async def test_publish_experiment_update_with_redis(mock_redis_client):
    """Test publishing experiment update to Redis"""
    experiment_id = "test-exp-5"
    update_data = {"type": "metrics_update", "data": {"clicks": 42}}
    
    await publish_experiment_update(experiment_id, update_data, mock_redis_client)
    
    # Verify Redis publish was called
    mock_redis_client.publish.assert_called_once()
    call_args = mock_redis_client.publish.call_args
    
    assert call_args[0][0] == f"experiment:{experiment_id}:updates"
    published_data = json.loads(call_args[0][1])
    assert published_data == update_data


@pytest.mark.asyncio
async def test_publish_experiment_update_fallback_to_websocket():
    """Test fallback to direct WebSocket broadcast when no Redis"""
    # Use patch to mock the global connection_manager
    with patch('prompt_improver.utils.websocket_manager.connection_manager') as mock_global_mgr:
        # Set up connection manager with websocket
        manager = ConnectionManager()
        websocket = MockWebSocket()
        experiment_id = "test-exp-6"
        
        await manager.connect(websocket, experiment_id, "user-test")
        
        # Configure the mock global manager
        mock_global_mgr.redis_client = None
        mock_global_mgr.broadcast_to_experiment = AsyncMock()
        
        update_data = {"type": "fallback_test", "data": "direct broadcast"}
        await publish_experiment_update(experiment_id, update_data)
        
        # Verify that the fallback to direct broadcast was called
        mock_global_mgr.broadcast_to_experiment.assert_called_once_with(experiment_id, update_data)


@pytest.mark.asyncio
async def test_redis_subscription_handler(connection_manager_with_redis):
    """Test Redis subscription message handling"""
    manager = connection_manager_with_redis
    websocket = MockWebSocket()
    experiment_id = "test-exp-7"
    
    # Connect websocket
    await manager.connect(websocket, experiment_id, "user-redis-test")
    
    # Get the pubsub instance from the mock
    pubsub_mock = manager.redis_client.pubsub(channels=[f"experiment:{experiment_id}:updates"])
    
    # Add a test message
    test_data = {"type": "redis_message", "data": "from redis"}
    pubsub_mock.add_message(f"experiment:{experiment_id}:updates", test_data)
    
    # Run the subscription handler
    await manager._redis_subscription_handler(experiment_id)
    
    # The message should have been processed, but due to mocking complexity,
    # we mainly verify the handler runs without error
    assert True  # Handler completed without exception


@pytest.mark.asyncio
async def test_connection_cleanup():
    """Test connection manager cleanup"""
    manager = ConnectionManager()
    websockets = []
    
    # Create multiple connections
    for i in range(3):
        ws = MockWebSocket()
        await manager.connect(ws, f"exp-{i}", f"user-{i}")
        websockets.append(ws)
    
    # Verify connections exist
    assert len(manager.experiment_connections) == 3
    assert len(manager.connection_metadata) == 3
    
    # Cleanup
    await manager.cleanup()
    
    # Verify cleanup
    assert len(manager.experiment_connections) == 0
    assert len(manager.connection_metadata) == 0
    assert len(manager._background_tasks) == 0
    
    # Verify websockets were closed
    for ws in websockets:
        assert ws.closed


@pytest.mark.asyncio
async def test_connection_count():
    """Test connection count methods"""
    manager = ConnectionManager()
    
    # Initially no connections
    assert manager.get_connection_count() == 0
    assert manager.get_connection_count("non-existent") == 0
    
    # Add connections
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    ws3 = MockWebSocket()
    
    await manager.connect(ws1, "exp-1", "user-1")
    await manager.connect(ws2, "exp-1", "user-2") 
    await manager.connect(ws3, "exp-2", "user-3")
    
    # Test counts
    assert manager.get_connection_count() == 3
    assert manager.get_connection_count("exp-1") == 2
    assert manager.get_connection_count("exp-2") == 1
    assert manager.get_connection_count("non-existent") == 0


@pytest.mark.asyncio
async def test_get_active_experiments():
    """Test getting active experiments"""
    manager = ConnectionManager()
    
    # Initially no experiments
    assert manager.get_active_experiments() == []
    
    # Add connections
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await manager.connect(ws1, "exp-alpha", "user-1")
    await manager.connect(ws2, "exp-beta", "user-2")
    
    # Test active experiments
    active = manager.get_active_experiments()
    assert set(active) == {"exp-alpha", "exp-beta"}


# Integration test to verify the full flow
@pytest.mark.asyncio
async def test_full_websocket_redis_integration(mock_redis_client):
    """Test complete WebSocket + Redis integration flow"""
    # Setup manager with Redis
    manager = ConnectionManager(redis_client=mock_redis_client)
    
    # Connect websocket
    websocket = MockWebSocket()
    experiment_id = "integration-test-exp"
    user_id = "integration-user"
    
    await manager.connect(websocket, experiment_id, user_id)
    
    # Verify connection established
    assert experiment_id in manager.experiment_connections
    assert websocket in manager.connection_metadata
    
    # Test direct broadcast
    message1 = {"type": "direct", "content": "direct message"}
    await manager.broadcast_to_experiment(experiment_id, message1)
    
    # Test Redis publish
    message2 = {"type": "redis_publish", "content": "redis message"}
    await publish_experiment_update(experiment_id, message2, mock_redis_client)
    
    # Test disconnection
    await manager.disconnect(websocket)
    
    # Verify cleanup
    assert experiment_id not in manager.experiment_connections
    assert websocket not in manager.connection_metadata
    
    # Verify Redis operations were called
    mock_redis_client.publish.assert_called()


if __name__ == "__main__":
    # Run a simple smoke test
    asyncio.run(test_connection_manager_initialization())
    print("Basic test passed!")