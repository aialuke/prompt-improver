"""Integration test for WebSocket manager with coredis Redis client

This test demonstrates the complete integration with a real Redis instance
for end-to-end validation of the migration from redis.asyncio to coredis.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import coredis
import pytest

from prompt_improver.services.cache.l2_redis_service import setup_redis_connection


class MockWebSocket:
    """Mock WebSocket for integration testing"""

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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_websocket_coredis_integration():
    """Test WebSocket manager with coredis client"""
    redis_client = coredis.Redis(host="127.0.0.1", port=6379, db=0)
    redis_client.ping = AsyncMock(return_value=b"PONG")
    redis_client.publish = AsyncMock(return_value=1)
    manager = ConnectionManager(redis_client=redis_client)
    assert isinstance(manager.redis_client, coredis.Redis)
    websocket = MockWebSocket()
    experiment_id = "integration-test-exp"
    user_id = "integration-user"
    await manager.connect(websocket, experiment_id, user_id)
    assert experiment_id in manager.experiment_connections
    assert websocket in manager.experiment_connections[experiment_id]
    test_message = {"type": "integration_test", "data": {"value": 42}}
    await manager.broadcast_to_experiment(experiment_id, test_message)
    assert len(websocket.sent_messages) == 2
    broadcast_msg = json.loads(websocket.sent_messages[1])
    assert broadcast_msg["type"] == "integration_test"
    assert broadcast_msg["data"]["value"] == 42
    assert "timestamp" in broadcast_msg
    redis_message = {"type": "redis_integration", "data": "published via Redis"}
    await publish_experiment_update(experiment_id, redis_message, redis_client)
    redis_client.publish.assert_called_once()
    publish_args = redis_client.publish.call_args
    assert publish_args[0][0] == f"experiment:{experiment_id}:updates"
    await manager.disconnect(websocket)
    assert experiment_id not in manager.experiment_connections
    assert websocket not in manager.connection_metadata
    print("✅ WebSocket + coredis integration test passed!")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_redis_connection_setup():
    """Test Redis connection setup with coredis"""
    with pytest.MonkeyPatch().context() as m:
        mock_client = AsyncMock(spec=coredis.Redis)
        mock_client.ping = AsyncMock(return_value=b"PONG")

        def mock_from_url(url, **kwargs):
            assert url == "redis://localhost:6379"
            assert kwargs.get("decode_responses") is True
            return mock_client

        m.setattr("coredis.Redis.from_url", mock_from_url)
        redis_client = await setup_redis_connection("redis://localhost:6379")
        assert redis_client == mock_client
        mock_client.ping.assert_called_once()
    print("✅ Redis connection setup test passed!")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pubsub_context_manager():
    """Test coredis pubsub async context manager usage"""
    redis_client = coredis.Redis()

    class MockPubSub:
        def __init__(self, channels=None):
            self.channels = channels or []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def __aiter__(self):
            yield {
                "type": "subscribe",
                "channel": self.channels[0] if self.channels else "test",
                "data": 1,
            }
            yield {
                "type": "message",
                "channel": self.channels[0] if self.channels else "test",
                "data": '{"test": "data"}',
            }

    def mock_pubsub(channels=None, patterns=None):
        return MockPubSub(channels=channels)

    redis_client.pubsub = mock_pubsub
    channel_name = "test:channel"
    messages_received = []
    async with redis_client.pubsub(channels=[channel_name]) as pubsub:
        async for message in pubsub:
            messages_received.append(message)
            if message["type"] == "message":
                break
    assert len(messages_received) == 2
    assert messages_received[0]["type"] == "subscribe"
    assert messages_received[1]["type"] == "message"
    assert json.loads(messages_received[1]["data"])["test"] == "data"
    print("✅ PubSub context manager test passed!")


if __name__ == "__main__":

    async def run_tests():
        print("Running WebSocket + coredis integration tests...")
        await test_websocket_coredis_integration()
        await test_redis_connection_setup()
        await test_pubsub_context_manager()
        print("\n🎉 All integration tests passed!")

    asyncio.run(run_tests())
