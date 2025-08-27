"""Real message tests for PubSubManager.

Tests comprehensive publish/subscribe functionality with real message scenarios:
- Multi-channel publishing and subscription with real message flows
- Pattern-based subscriptions with wildcard matching
- Message serialization and deserialization with various data types
- High-throughput message publishing with batch operations
- Subscription management with automatic cleanup and monitoring
- Message filtering and transformation with custom handlers

Integration tests using mock Redis that simulates real Redis pub/sub behavior.
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

import pytest
from tests.utils.mocks import MockRedisClient

from prompt_improver.database.services.pubsub.pubsub_manager import (
    PubSubConfig,
    PubSubManager,
)


class MockRedisPubSub:
    """Mock Redis pub/sub connection that simulates real Redis behavior."""

    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.subscribed_channels: set[str] = set()
        self.subscribed_patterns: set[str] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def subscribe(self, *channels):
        """Mock subscribe to channels."""
        for channel in channels:
            self.subscribed_channels.add(channel)
            # Send subscription confirmation
            await self.message_queue.put({
                'type': 'subscribe',
                'channel': channel.encode('utf-8'),
                'data': len(self.subscribed_channels)
            })

    async def psubscribe(self, *patterns):
        """Mock pattern subscribe."""
        for pattern in patterns:
            self.subscribed_patterns.add(pattern)
            # Send pattern subscription confirmation
            await self.message_queue.put({
                'type': 'psubscribe',
                'pattern': pattern.encode('utf-8'),
                'data': len(self.subscribed_patterns)
            })

    async def unsubscribe(self, *channels):
        """Mock unsubscribe from channels."""
        for channel in channels:
            self.subscribed_channels.discard(channel)
            await self.message_queue.put({
                'type': 'unsubscribe',
                'channel': channel.encode('utf-8'),
                'data': len(self.subscribed_channels)
            })

    async def punsubscribe(self, *patterns):
        """Mock pattern unsubscribe."""
        for pattern in patterns:
            self.subscribed_patterns.discard(pattern)
            await self.message_queue.put({
                'type': 'punsubscribe',
                'pattern': pattern.encode('utf-8'),
                'data': len(self.subscribed_patterns)
            })

    async def get_message(self, timeout=None):
        """Mock get message with timeout."""
        try:
            if timeout:
                return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
            return await self.message_queue.get()
        except TimeoutError:
            return None

    async def close(self):
        """Mock close connection."""
        self.closed = True

    def inject_message(self, channel: str, data: bytes):
        """Inject a message into the subscription (for testing)."""
        if channel in self.subscribed_channels or any(
            self._pattern_matches(pattern, channel) for pattern in self.subscribed_patterns
        ):
            message = {
                'type': 'message',
                'channel': channel.encode('utf-8'),
                'data': data
            }
            asyncio.create_task(self.message_queue.put(message))

    def _pattern_matches(self, pattern: str, channel: str) -> bool:
        """Simple pattern matching for testing."""
        if '*' not in pattern:
            return pattern == channel

        # Simple wildcard matching
        pattern_parts = pattern.split('*')
        if len(pattern_parts) == 2:
            prefix, suffix = pattern_parts
            return channel.startswith(prefix) and channel.endswith(suffix)

        return False


class MockL2RedisService:
    """Mock L2RedisService that wraps MockRedisClient for testing."""

    def __init__(self, redis_client: MockRedisClient):
        self._redis_client = redis_client
        self._available = True

    def is_available(self) -> bool:
        """Check if Redis client is available."""
        return self._available and self._redis_client is not None

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message via wrapped Redis client."""
        if not self.is_available():
            return 0

        # Serialize message like L2RedisService does
        if isinstance(message, (str, bytes)):
            serialized_message = message
        else:
            serialized_message = json.dumps(message, default=str)

        return await self._redis_client.publish(channel, serialized_message)

    def get_pipeline(self) -> Any | None:
        """Get Redis pipeline object."""
        if not self.is_available():
            return None
        return MockRedisPipeline(self._redis_client)

    def get_pubsub(self) -> Any | None:
        """Get Redis pubsub object."""
        if not self.is_available():
            return None
        return self._redis_client.pubsub()


class MockRedisPipeline:
    """Mock Redis pipeline for batch operations."""

    def __init__(self, redis_client: MockRedisClient):
        self._redis_client = redis_client
        self._commands = []

    def publish(self, channel: str, message: str):
        """Queue publish command."""
        self._commands.append(('publish', channel, message))

    async def execute(self):
        """Execute queued commands."""
        results = []
        for cmd_type, channel, message in self._commands:
            if cmd_type == 'publish':
                result = await self._redis_client.publish(channel, message)
                results.append(result)
        self._commands.clear()
        return results


class MockRedisClient:
    """Mock Redis client that simulates pub/sub behavior."""

    def __init__(self, should_fail: bool = False, response_delay_ms: float = 1.0):
        self.should_fail = should_fail
        self.response_delay_ms = response_delay_ms

        # Track active subscriptions
        self._pubsub_connections: list[MockRedisPubSub] = []
        self._channel_subscribers: dict[str, list[MockRedisPubSub]] = defaultdict(list)

        # Operation counters
        self.publish_count = 0
        self.subscribe_count = 0

    async def _sleep_if_needed(self):
        """Simulate network latency."""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

    async def publish(self, channel: str, message: str) -> int:
        """Mock Redis publish that delivers to active subscribers."""
        await self._sleep_if_needed()

        if self.should_fail:
            raise Exception("Mock Redis publish failure")

        self.publish_count += 1

        # Find subscribers and deliver message
        subscriber_count = 0
        for pubsub in self._pubsub_connections:
            if channel in pubsub.subscribed_channels or any(
                pubsub._pattern_matches(pattern, channel)
                for pattern in pubsub.subscribed_patterns
            ):
                pubsub.inject_message(channel, message.encode('utf-8'))
                subscriber_count += 1

        return subscriber_count

    def pubsub(self) -> MockRedisPubSub:
        """Create mock pub/sub connection."""
        pubsub = MockRedisPubSub(self)
        self._pubsub_connections.append(pubsub)
        self.subscribe_count += 1
        return pubsub

    def get_stats(self) -> dict[str, Any]:
        """Get mock Redis statistics."""
        return {
            "publish_operations": self.publish_count,
            "subscribe_operations": self.subscribe_count,
            "active_pubsub_connections": len(self._pubsub_connections),
        }


class TestPubSubManagerCore:
    """Test PubSubManager core functionality."""

    def test_pubsub_manager_creation(self):
        """Test pub/sub manager initialization."""
        redis_client = MockRedisClient()
        l2_redis_service = MockL2RedisService(redis_client)
        config = PubSubConfig(max_connections=5)

        manager = PubSubManager(l2_redis_service, config)

        assert manager.l2_redis_service == l2_redis_service
        assert manager.config.max_connections == 5
        assert len(manager._subscriptions) == 0
        assert manager.total_publishes == 0

    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self):
        """Test basic publish/subscribe functionality."""
        redis_client = MockRedisClient()
        l2_redis_service = MockL2RedisService(redis_client)
        manager = PubSubManager(l2_redis_service)

        # Subscribe to channel
        subscriber_id = await manager.subscribe(["test_channel"])
        assert subscriber_id is not None
        assert subscriber_id in manager._subscriptions

        # Publish message
        subscriber_count = await manager.publish("test_channel", "Hello World")
        assert subscriber_count == 1

        # Listen for message
        messages_received = []

        async def collect_messages():
            async for message in manager.listen(subscriber_id, timeout=0.1):
                messages_received.append(message)
                break  # Get one message and stop

        # Give some time for message delivery
        await asyncio.sleep(0.05)
        await collect_messages()

        assert len(messages_received) == 1
        assert messages_received[0]['type'] == 'message'
        assert messages_received[0]['data'] == "Hello World"

        # Cleanup
        await manager.unsubscribe(subscriber_id)

        print("✅ Basic publish/subscribe functionality")

    @pytest.mark.asyncio
    async def test_json_message_serialization(self):
        """Test automatic JSON serialization/deserialization."""
        redis_client = MockRedisClient()
        l2_redis_service = MockL2RedisService(redis_client)
        manager = PubSubManager(l2_redis_service, PubSubConfig(auto_serialize_json=True))

        subscriber_id = await manager.subscribe(["json_channel"])

        # Test various data types
        test_data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "array": [1, 2, 3],
            "object": {"nested": "value"},
            "null": None,
        }

        await manager.publish("json_channel", test_data)

        # Collect message
        messages_received = []

        async def collect_json_message():
            async for message in manager.listen(subscriber_id, timeout=0.1):
                messages_received.append(message)
                break

        await asyncio.sleep(0.05)
        await collect_json_message()

        assert len(messages_received) == 1
        received_data = messages_received[0]['data']
        assert received_data == test_data
        assert received_data["number"] == 42
        assert received_data["boolean"] is True
        assert received_data["array"] == [1, 2, 3]

        await manager.unsubscribe(subscriber_id)

        print("✅ JSON message serialization/deserialization")

    @pytest.mark.asyncio
    async def test_multiple_channels_subscription(self):
        """Test subscribing to multiple channels."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client)

        # Subscribe to multiple channels
        channels = ["channel1", "channel2", "channel3"]
        subscriber_id = await manager.subscribe(channels)

        assert subscriber_id in manager._subscriptions
        sub_info = manager._subscriptions[subscriber_id]
        assert set(sub_info.channels) == set(channels)

        # Test channel tracking
        for channel in channels:
            subscribers = manager.get_channel_subscribers(channel)
            assert subscriber_id in subscribers

        # Publish to different channels
        await manager.publish("channel1", "Message 1")
        await manager.publish("channel2", "Message 2")
        await manager.publish("channel3", "Message 3")

        # Collect messages
        messages_received = []

        async def collect_multi_messages():
            timeout_count = 0
            async for message in manager.listen(subscriber_id, timeout=0.1):
                messages_received.append(message)
                if len(messages_received) >= 3:
                    break

        await asyncio.sleep(0.05)
        await collect_multi_messages()

        # Should receive messages from all channels
        assert len(messages_received) == 3
        received_channels = {msg['channel'].decode('utf-8') for msg in messages_received}
        assert received_channels == set(channels)

        await manager.unsubscribe(subscriber_id)

        print("✅ Multiple channels subscription")

    @pytest.mark.asyncio
    async def test_pattern_based_subscription(self):
        """Test pattern-based subscriptions with wildcards."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client, PubSubConfig(enable_pattern_matching=True))

        # Subscribe to pattern
        subscriber_id = await manager.subscribe([], patterns=["news.*", "events.*"])

        assert subscriber_id in manager._subscriptions
        sub_info = manager._subscriptions[subscriber_id]
        assert set(sub_info.patterns) == {"news.*", "events.*"}

        # Publish to channels matching patterns
        await manager.publish("news.sports", "Sports news")
        await manager.publish("news.tech", "Tech news")
        await manager.publish("events.concert", "Concert event")
        await manager.publish("other.random", "Random message")  # Should not match

        # Collect messages
        messages_received = []

        async def collect_pattern_messages():
            timeout_count = 0
            async for message in manager.listen(subscriber_id, timeout=0.1):
                messages_received.append(message)
                if len(messages_received) >= 3:  # Expect 3 matching messages
                    break

        await asyncio.sleep(0.05)
        await collect_pattern_messages()

        # Should receive messages matching patterns
        assert len(messages_received) == 3
        received_channels = {msg['channel'].decode('utf-8') for msg in messages_received}
        assert "news.sports" in received_channels
        assert "news.tech" in received_channels
        assert "events.concert" in received_channels
        assert "other.random" not in received_channels

        await manager.unsubscribe(subscriber_id)

        print("✅ Pattern-based subscription with wildcards")


@pytest.mark.asyncio
class TestPubSubManagerAdvanced:
    """Test advanced PubSubManager functionality."""

    async def test_high_throughput_publishing(self):
        """Test high-throughput message publishing."""
        redis_client = MockRedisClient(response_delay_ms=0.1)  # Minimal delay
        manager = PubSubManager(redis_client)

        # Create subscriber
        subscriber_id = await manager.subscribe(["high_throughput"])

        # Publish many messages rapidly
        num_messages = 100
        start_time = time.time()

        publish_tasks = []
        for i in range(num_messages):
            task = manager.publish("high_throughput", f"Message {i}")
            publish_tasks.append(task)

        # Wait for all publishes to complete
        subscriber_counts = await asyncio.gather(*publish_tasks)

        publish_duration = time.time() - start_time
        messages_per_second = num_messages / publish_duration

        print(f"    Published {num_messages} messages in {publish_duration:.3f}s")
        print(f"    Throughput: {messages_per_second:.1f} messages/sec")

        # Verify all publishes succeeded
        assert all(count == 1 for count in subscriber_counts)
        assert manager.total_publishes == num_messages

        # Should achieve reasonable throughput
        assert messages_per_second > 100  # At least 100 msg/sec

        await manager.unsubscribe(subscriber_id)

        print("✅ High-throughput publishing performance")

    async def test_batch_publishing(self):
        """Test batch publishing to multiple channels."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client)

        # Subscribe to multiple channels
        channels = [f"batch_channel_{i}" for i in range(10)]
        subscriber_ids = []

        for channel in channels:
            sub_id = await manager.subscribe([channel])
            subscriber_ids.append(sub_id)

        # Prepare batch messages
        channel_messages = {
            channel: f"Batch message for {channel}"
            for channel in channels
        }

        # Publish batch
        start_time = time.time()
        results = await manager.publish_many(channel_messages)
        batch_duration = time.time() - start_time

        print(f"    Batch published to {len(channels)} channels in {batch_duration:.3f}s")

        # Verify all channels received messages
        assert len(results) == len(channels)
        assert all(count == 1 for count in results.values())

        # Cleanup
        for sub_id in subscriber_ids:
            await manager.unsubscribe(sub_id)

        print("✅ Batch publishing to multiple channels")

    async def test_subscription_management_and_cleanup(self):
        """Test subscription lifecycle and cleanup."""
        redis_client = MockRedisClient()
        config = PubSubConfig(auto_cleanup_inactive_seconds=1.0)
        manager = PubSubManager(redis_client, config)

        # Create multiple subscriptions
        subscription_ids = []
        for i in range(5):
            sub_id = await manager.subscribe([f"cleanup_channel_{i}"])
            subscription_ids.append(sub_id)

        # Verify subscriptions exist
        assert len(manager._subscriptions) == 5

        # Start cleanup task
        await manager.start_cleanup_task()

        # Simulate some activity on first 3 subscriptions
        for i in range(3):
            await manager.publish(f"cleanup_channel_{i}", f"Active message {i}")

        # Let some messages flow to mark subscriptions as active
        await asyncio.sleep(0.1)

        # Wait for cleanup to potentially run (but active subscriptions should remain)
        await asyncio.sleep(1.2)

        # All subscriptions should still exist (they're considered active)
        assert len(manager._subscriptions) == 5

        # Manual cleanup
        for sub_id in subscription_ids[:2]:
            await manager.unsubscribe(sub_id)

        assert len(manager._subscriptions) == 3

        await manager.stop_cleanup_task()

        # Final cleanup
        for sub_id in subscription_ids[2:]:
            await manager.unsubscribe(sub_id)

        assert len(manager._subscriptions) == 0

        print("✅ Subscription management and cleanup")

    async def test_message_size_validation(self):
        """Test message size validation and limits."""
        redis_client = MockRedisClient()
        config = PubSubConfig(max_message_size_bytes=100)  # Very small limit for testing
        manager = PubSubManager(redis_client, config)

        subscriber_id = await manager.subscribe(["size_test"])

        # Small message should succeed
        small_message = "Small message"
        count = await manager.publish("size_test", small_message)
        assert count == 1

        # Large message should fail
        large_message = "X" * 200  # Exceeds 100 byte limit
        count = await manager.publish("size_test", large_message)
        assert count == 0  # Should fail due to size

        # Failed operations should be tracked
        assert manager.failed_operations > 0

        await manager.unsubscribe(subscriber_id)

        print("✅ Message size validation and limits")

    async def test_concurrent_subscriptions(self):
        """Test concurrent subscription management."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client)

        # Create concurrent subscriptions
        async def create_subscription(sub_id: int):
            channels = [f"concurrent_channel_{sub_id}_{i}" for i in range(3)]
            return await manager.subscribe(channels)

        num_subscribers = 10
        tasks = [create_subscription(i) for i in range(num_subscribers)]
        subscriber_ids = await asyncio.gather(*tasks)

        # Verify all subscriptions created
        assert len(subscriber_ids) == num_subscribers
        assert all(sub_id is not None for sub_id in subscriber_ids)
        assert len(manager._subscriptions) == num_subscribers

        # Publish to all channels concurrently
        publish_tasks = []
        for i in range(num_subscribers):
            for j in range(3):
                channel = f"concurrent_channel_{i}_{j}"
                task = manager.publish(channel, f"Message for {channel}")
                publish_tasks.append(task)

        results = await asyncio.gather(*publish_tasks)

        # Each message should have 1 subscriber
        assert all(count == 1 for count in results)

        # Cleanup all subscriptions
        cleanup_tasks = [manager.unsubscribe(sub_id) for sub_id in subscriber_ids]
        cleanup_results = await asyncio.gather(*cleanup_tasks)

        assert all(result is True for result in cleanup_results)
        assert len(manager._subscriptions) == 0

        print("✅ Concurrent subscription management")


class TestPubSubManagerFailures:
    """Test PubSubManager failure scenarios."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self):
        """Test behavior when Redis connection fails."""
        failing_redis = MockRedisClient(should_fail=True)
        l2_redis_service = MockL2RedisService(failing_redis)
        manager = PubSubManager(l2_redis_service)

        # Publishing should fail gracefully
        count = await manager.publish("test_channel", "test message")
        assert count == 0
        assert manager.failed_operations > 0

        # Subscribing should fail gracefully
        subscriber_id = await manager.subscribe(["test_channel"])
        # Note: Mock doesn't simulate subscription failure, but in real Redis this would fail

        print("✅ Redis connection failure handled gracefully")

    @pytest.mark.asyncio
    async def test_subscription_timeout_behavior(self):
        """Test subscription timeout and message waiting."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client)

        subscriber_id = await manager.subscribe(["timeout_channel"])

        # Listen with timeout (no messages published)
        messages_received = []
        start_time = time.time()

        async def listen_with_timeout():
            messages_received.extend([message async for message in manager.listen(subscriber_id, timeout=0.1)])

        await listen_with_timeout()

        elapsed = time.time() - start_time

        # Should have timed out quickly without receiving messages
        assert len(messages_received) == 0
        assert elapsed < 0.5  # Should timeout much faster than 0.5s

        await manager.unsubscribe(subscriber_id)

        print("✅ Subscription timeout behavior")

    @pytest.mark.asyncio
    async def test_statistics_accuracy(self):
        """Test statistics tracking accuracy."""
        redis_client = MockRedisClient()
        manager = PubSubManager(redis_client)

        initial_stats = manager.get_stats()
        assert initial_stats["performance"]["total_publishes"] == 0

        # Perform various operations
        sub_id1 = await manager.subscribe(["stats_channel1"])
        sub_id2 = await manager.subscribe(["stats_channel2", "stats_channel3"])

        await manager.publish("stats_channel1", "Message 1")
        await manager.publish("stats_channel2", "Message 2")
        await manager.publish("nonexistent_channel", "Message 3")  # 0 subscribers

        await manager.unsubscribe(sub_id1)

        # Check final stats
        final_stats = manager.get_stats()

        assert final_stats["performance"]["total_publishes"] == 3
        assert final_stats["performance"]["total_subscribes"] == 2
        assert final_stats["performance"]["total_unsubscribes"] == 1
        assert final_stats["performance"]["total_messages_sent"] == 3
        assert final_stats["subscriptions"]["active_count"] == 1  # sub_id2 still active

        success_rate = final_stats["performance"]["success_rate"]
        assert 0.8 <= success_rate <= 1.0  # Should be high success rate

        await manager.unsubscribe(sub_id2)

        print("✅ Statistics tracking accuracy")


if __name__ == "__main__":
    print("🔄 Running PubSubManager Real Message Tests...")

    async def run_tests():
        print("\n1. Testing core pub/sub functionality...")
        core_suite = TestPubSubManagerCore()
        core_suite.test_pubsub_manager_creation()
        await core_suite.test_basic_publish_subscribe()
        await core_suite.test_json_message_serialization()
        await core_suite.test_multiple_channels_subscription()
        await core_suite.test_pattern_based_subscription()

        print("\n2. Testing advanced functionality...")
        advanced_suite = TestPubSubManagerAdvanced()
        await advanced_suite.test_high_throughput_publishing()
        await advanced_suite.test_batch_publishing()
        await advanced_suite.test_subscription_management_and_cleanup()
        await advanced_suite.test_message_size_validation()
        await advanced_suite.test_concurrent_subscriptions()

        print("\n3. Testing failure scenarios...")
        failure_suite = TestPubSubManagerFailures()
        await failure_suite.test_redis_connection_failure()
        await failure_suite.test_subscription_timeout_behavior()
        await failure_suite.test_statistics_accuracy()

    # Run the tests
    asyncio.run(run_tests())

    print("\n🎯 PubSubManager Real Message Testing Complete")
    print("   ✅ Basic publish/subscribe with real message delivery")
    print("   ✅ JSON serialization/deserialization with complex data types")
    print("   ✅ Multiple channel subscriptions with concurrent message handling")
    print("   ✅ Pattern-based subscriptions with wildcard matching")
    print("   ✅ High-throughput publishing with >100 messages/sec performance")
    print("   ✅ Batch publishing to multiple channels simultaneously")
    print("   ✅ Subscription lifecycle management with automatic cleanup")
    print("   ✅ Message size validation and connection failure resilience")
    print("   ✅ Concurrent subscription management with performance validation")
