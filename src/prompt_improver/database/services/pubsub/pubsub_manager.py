"""Redis-based publish/subscribe messaging service.

This module provides publish/subscribe functionality extracted from
unified_connection_manager.py, implementing:

- PubSubManager: Redis-based messaging with channel management
- PubSubConfig: Configurable connection pools, message serialization settings
- SubscriptionManager: Active subscription tracking and cleanup
- Message filtering and transformation with custom handlers
- Pattern-based subscriptions with wildcard matching

Designed for production messaging with high-throughput pub/sub operations.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Union

from prompt_improver.core.protocols.cache_protocol import CacheSubscriptionProtocol

logger = logging.getLogger(__name__)

# Import OpenTelemetry metrics if available
try:
    from opentelemetry import metrics

    OPENTELEMETRY_AVAILABLE = True
    meter = metrics.get_meter(__name__)
    pubsub_operations_counter = meter.create_counter(
        "pubsub_operations_total", description="Total pub/sub operations"
    )
    pubsub_messages_histogram = meter.create_histogram(
        "pubsub_message_size_bytes", description="Size of pub/sub messages"
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    pubsub_operations_counter = None
    pubsub_messages_histogram = None


# Type aliases
MessageHandler = Callable[[str, Any], None]
AsyncMessageHandler = Callable[[str, Any], Any]
MessageFilter = Callable[[str, Any], bool]


@dataclass
class PubSubConfig:
    """Configuration for publish/subscribe operations."""

    # Connection settings
    max_connections: int = 10
    connection_timeout_seconds: float = 5.0

    # Message settings
    auto_serialize_json: bool = True
    max_message_size_bytes: int = 1024 * 1024  # 1MB default

    # Subscription management
    subscription_timeout_seconds: float = 30.0
    auto_cleanup_inactive_seconds: float = 300.0  # 5 minutes

    # Pattern matching
    enable_pattern_matching: bool = True

    # Performance settings
    batch_publish_size: int = 100
    message_buffer_size: int = 1000

    # Monitoring
    enable_metrics: bool = True
    log_message_stats: bool = False

    def __post_init__(self):
        if self.max_connections <= 0:
            raise ValueError("max_connections must be greater than 0")
        if self.max_message_size_bytes <= 0:
            raise ValueError("max_message_size_bytes must be greater than 0")


@dataclass
class SubscriptionInfo:
    """Information about an active subscription."""

    channels: List[str]
    patterns: List[str]
    connection: Any
    created_at: float
    last_message_at: Optional[float] = None
    message_count: int = 0
    subscriber_id: str = ""

    @property
    def is_active(self) -> bool:
        """Check if subscription has recent activity."""
        if self.last_message_at is None:
            return True  # Newly created, consider active
        return (time.time() - self.last_message_at) < 300  # 5 minutes

    @property
    def age_seconds(self) -> float:
        """Get age of subscription in seconds."""
        return time.time() - self.created_at


class PubSubManager:
    """Redis-based publish/subscribe manager with advanced features.

    Provides comprehensive pub/sub functionality with:
    - Multi-channel subscription management
    - Pattern-based subscriptions with wildcard matching
    - Message filtering and transformation
    - Connection pooling for high throughput
    - Automatic cleanup of inactive subscriptions
    - OpenTelemetry metrics integration
    """

    def __init__(
        self, redis_client, config: Optional[PubSubConfig] = None, security_context=None
    ):
        self.redis_client = redis_client
        self.config = config or PubSubConfig()
        self.security_context = security_context

        # Connection management
        self._pubsub_connections: Dict[str, Any] = {}
        self._connection_pool: List[Any] = []

        # Subscription tracking
        self._subscriptions: Dict[str, SubscriptionInfo] = {}  # subscriber_id -> info
        self._channel_subscribers: Dict[str, Set[str]] = defaultdict(
            set
        )  # channel -> subscriber_ids
        self._pattern_subscribers: Dict[str, Set[str]] = defaultdict(
            set
        )  # pattern -> subscriber_ids

        # Message handlers
        self._message_handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._async_message_handlers: Dict[str, List[AsyncMessageHandler]] = (
            defaultdict(list)
        )
        self._message_filters: Dict[str, List[MessageFilter]] = defaultdict(list)

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.total_publishes = 0
        self.total_subscribes = 0
        self.total_unsubscribes = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.failed_operations = 0

        logger.info(
            f"PubSubManager initialized with max_connections={self.config.max_connections}"
        )

    async def start_cleanup_task(self) -> None:
        """Start background task for cleaning up inactive subscriptions."""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return

        self._shutdown_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started pub/sub cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        self._shutdown_event.set()

        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Pub/sub cleanup task did not stop gracefully")
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

        logger.info("Stopped pub/sub cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up inactive subscriptions."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_inactive_subscriptions()

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.auto_cleanup_inactive_seconds,
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue cleanup

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pub/sub cleanup loop: {e}")
                await asyncio.sleep(30.0)  # Back off on error

    async def _cleanup_inactive_subscriptions(self) -> None:
        """Clean up inactive subscriptions."""
        current_time = time.time()
        inactive_ids = []

        for sub_id, sub_info in self._subscriptions.items():
            if not sub_info.is_active:
                inactive_ids.append(sub_id)

        for sub_id in inactive_ids:
            await self._remove_subscription(sub_id)

        if inactive_ids:
            logger.info(f"Cleaned up {len(inactive_ids)} inactive subscriptions")

    def _serialize_message(self, message: Any) -> Union[str, bytes]:
        """Serialize message for publishing."""
        if isinstance(message, (str, bytes)):
            return message

        if self.config.auto_serialize_json:
            try:
                serialized = json.dumps(message, default=str)
                return serialized
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize message as JSON: {e}")
                return str(message)

        return str(message)

    def _deserialize_message(self, message: Union[str, bytes]) -> Any:
        """Deserialize received message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        if self.config.auto_serialize_json and isinstance(message, str):
            try:
                return json.loads(message)
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON, return as string

        return message

    def _validate_message_size(self, message: Union[str, bytes]) -> bool:
        """Validate message size against limits."""
        size = len(message.encode("utf-8") if isinstance(message, str) else message)
        return size <= self.config.max_message_size_bytes

    def _generate_subscriber_id(self) -> str:
        """Generate unique subscriber ID."""
        import uuid

        return f"sub_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel.

        Args:
            channel: Channel name
            message: Message to publish (will be serialized if needed)

        Returns:
            Number of subscribers that received the message
        """
        if not self.redis_client:
            logger.error("Redis client not available for publishing")
            return 0

        try:
            # Serialize message
            serialized_message = self._serialize_message(message)

            # Validate message size
            if not self._validate_message_size(serialized_message):
                logger.error(
                    f"Message too large for channel {channel}: {len(serialized_message)} bytes"
                )
                self.failed_operations += 1
                return 0

            # Publish to Redis
            subscriber_count = await self.redis_client.publish(
                channel, serialized_message
            )

            self.total_publishes += 1
            self.total_messages_sent += 1

            # Record metrics
            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "publish",
                        "channel": channel,
                        "status": "success",
                    },
                )

                if pubsub_messages_histogram:
                    message_size = len(
                        serialized_message.encode("utf-8")
                        if isinstance(serialized_message, str)
                        else serialized_message
                    )
                    pubsub_messages_histogram.record(
                        message_size,
                        {
                            "operation": "publish",
                            "channel": channel,
                        },
                    )

            if self.config.log_message_stats:
                logger.debug(
                    f"Published to {channel}: {subscriber_count} subscribers, {len(serialized_message)} bytes"
                )

            return subscriber_count

        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            self.failed_operations += 1

            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "publish",
                        "channel": channel,
                        "status": "error",
                    },
                )

            return 0

    async def publish_many(self, channel_messages: Dict[str, Any]) -> Dict[str, int]:
        """Publish messages to multiple channels efficiently.

        Args:
            channel_messages: Dictionary of channel -> message

        Returns:
            Dictionary of channel -> subscriber count
        """
        results = {}

        # Use Redis pipeline for efficiency if available
        try:
            if hasattr(self.redis_client, "pipeline"):
                pipe = self.redis_client.pipeline()

                for channel, message in channel_messages.items():
                    serialized_message = self._serialize_message(message)
                    if self._validate_message_size(serialized_message):
                        pipe.publish(channel, serialized_message)
                    else:
                        results[channel] = 0

                pipeline_results = await pipe.execute()

                # Map results back to channels
                result_idx = 0
                for channel, message in channel_messages.items():
                    if channel not in results:  # Not already marked as too large
                        results[channel] = (
                            pipeline_results[result_idx]
                            if result_idx < len(pipeline_results)
                            else 0
                        )
                        result_idx += 1

                self.total_publishes += len(channel_messages)
                self.total_messages_sent += len(channel_messages)

            else:
                # Fallback to individual publishes
                for channel, message in channel_messages.items():
                    results[channel] = await self.publish(channel, message)

            return results

        except Exception as e:
            logger.error(f"Failed to publish batch messages: {e}")
            self.failed_operations += 1

            # Return zero counts for all channels
            return {channel: 0 for channel in channel_messages.keys()}

    async def subscribe(
        self, channels: List[str], patterns: Optional[List[str]] = None
    ) -> Optional[str]:
        """Subscribe to channels and/or patterns.

        Args:
            channels: List of channel names to subscribe to
            patterns: Optional list of patterns to subscribe to

        Returns:
            Subscriber ID if successful, None if failed
        """
        if not self.redis_client:
            logger.error("Redis client not available for subscription")
            return None

        patterns = patterns or []

        try:
            # Create new pubsub connection
            pubsub = self.redis_client.pubsub()

            # Subscribe to channels
            if channels:
                await pubsub.subscribe(*channels)

            # Subscribe to patterns
            if patterns and self.config.enable_pattern_matching:
                await pubsub.psubscribe(*patterns)

            # Generate subscriber ID and create subscription info
            subscriber_id = self._generate_subscriber_id()

            self._subscriptions[subscriber_id] = SubscriptionInfo(
                channels=list(channels),
                patterns=list(patterns),
                connection=pubsub,
                created_at=time.time(),
                subscriber_id=subscriber_id,
            )

            # Update channel and pattern tracking
            for channel in channels:
                self._channel_subscribers[channel].add(subscriber_id)

            for pattern in patterns:
                self._pattern_subscribers[pattern].add(subscriber_id)

            self.total_subscribes += 1

            # Record metrics
            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "subscribe",
                        "channel_count": len(channels),
                        "pattern_count": len(patterns),
                        "status": "success",
                    },
                )

            logger.debug(
                f"Created subscription {subscriber_id} for channels={channels}, patterns={patterns}"
            )
            return subscriber_id

        except Exception as e:
            logger.error(
                f"Failed to subscribe to channels={channels}, patterns={patterns}: {e}"
            )
            self.failed_operations += 1

            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "subscribe",
                        "channel_count": len(channels),
                        "pattern_count": len(patterns),
                        "status": "error",
                    },
                )

            return None

    async def unsubscribe(
        self, subscriber_id: str, channels: Optional[List[str]] = None
    ) -> bool:
        """Unsubscribe from channels.

        Args:
            subscriber_id: Subscriber ID from subscribe()
            channels: Optional list of specific channels to unsubscribe from.
                     If None, unsubscribes from all channels for this subscriber.

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if subscriber_id not in self._subscriptions:
            logger.warning(f"Subscriber {subscriber_id} not found")
            return False

        try:
            sub_info = self._subscriptions[subscriber_id]

            if channels is None:
                # Unsubscribe from all channels and patterns
                if sub_info.channels:
                    await sub_info.connection.unsubscribe(*sub_info.channels)
                if sub_info.patterns:
                    await sub_info.connection.punsubscribe(*sub_info.patterns)

                # Remove subscription entirely
                await self._remove_subscription(subscriber_id)

            else:
                # Unsubscribe from specific channels only
                valid_channels = [ch for ch in channels if ch in sub_info.channels]
                if valid_channels:
                    await sub_info.connection.unsubscribe(*valid_channels)

                    # Update subscription info
                    for channel in valid_channels:
                        if channel in sub_info.channels:
                            sub_info.channels.remove(channel)
                        self._channel_subscribers[channel].discard(subscriber_id)

                    # If no channels left, remove subscription
                    if not sub_info.channels and not sub_info.patterns:
                        await self._remove_subscription(subscriber_id)

            self.total_unsubscribes += 1

            # Record metrics
            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "unsubscribe",
                        "status": "success",
                    },
                )

            logger.debug(f"Unsubscribed {subscriber_id} from channels={channels}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to unsubscribe {subscriber_id} from channels={channels}: {e}"
            )
            self.failed_operations += 1

            if OPENTELEMETRY_AVAILABLE and pubsub_operations_counter:
                pubsub_operations_counter.add(
                    1,
                    {
                        "operation": "unsubscribe",
                        "status": "error",
                    },
                )

            return False

    async def _remove_subscription(self, subscriber_id: str) -> None:
        """Remove subscription and clean up tracking."""
        if subscriber_id not in self._subscriptions:
            return

        sub_info = self._subscriptions[subscriber_id]

        try:
            # Close connection
            await sub_info.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection for {subscriber_id}: {e}")

        # Clean up tracking
        for channel in sub_info.channels:
            self._channel_subscribers[channel].discard(subscriber_id)
            if not self._channel_subscribers[channel]:
                del self._channel_subscribers[channel]

        for pattern in sub_info.patterns:
            self._pattern_subscribers[pattern].discard(subscriber_id)
            if not self._pattern_subscribers[pattern]:
                del self._pattern_subscribers[pattern]

        # Remove subscription
        del self._subscriptions[subscriber_id]

    async def listen(
        self, subscriber_id: str, timeout: Optional[float] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Listen for messages on a subscription.

        Args:
            subscriber_id: Subscriber ID from subscribe()
            timeout: Optional timeout for each message wait

        Yields:
            Message dictionaries with keys: channel, data, type
        """
        if subscriber_id not in self._subscriptions:
            logger.error(f"Subscriber {subscriber_id} not found")
            return

        sub_info = self._subscriptions[subscriber_id]

        try:
            while True:
                try:
                    # Get message with timeout
                    if timeout:
                        message = await asyncio.wait_for(
                            sub_info.connection.get_message(), timeout=timeout
                        )
                    else:
                        message = await sub_info.connection.get_message()

                    if message is None:
                        continue

                    # Skip subscription confirmation messages
                    if message["type"] in [
                        "subscribe",
                        "psubscribe",
                        "unsubscribe",
                        "punsubscribe",
                    ]:
                        continue

                    # Update subscription stats
                    sub_info.last_message_at = time.time()
                    sub_info.message_count += 1
                    self.total_messages_received += 1

                    # Deserialize data
                    if message["data"]:
                        message["data"] = self._deserialize_message(message["data"])

                    yield message

                except asyncio.TimeoutError:
                    # Timeout is expected, just continue
                    continue

                except Exception as e:
                    logger.error(f"Error receiving message for {subscriber_id}: {e}")
                    break

        except asyncio.CancelledError:
            pass  # Expected when cleaning up
        except Exception as e:
            logger.error(f"Unexpected error in listen loop for {subscriber_id}: {e}")

    def add_message_handler(
        self, channel: str, handler: Union[MessageHandler, AsyncMessageHandler]
    ) -> None:
        """Add message handler for a channel.

        Args:
            channel: Channel name
            handler: Synchronous or asynchronous message handler function
        """
        if asyncio.iscoroutinefunction(handler):
            self._async_message_handlers[channel].append(handler)
        else:
            self._message_handlers[channel].append(handler)

        logger.debug(f"Added message handler for channel {channel}")

    def remove_message_handler(
        self, channel: str, handler: Union[MessageHandler, AsyncMessageHandler]
    ) -> bool:
        """Remove message handler for a channel.

        Args:
            channel: Channel name
            handler: Handler function to remove

        Returns:
            True if handler was found and removed
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                self._async_message_handlers[channel].remove(handler)
            else:
                self._message_handlers[channel].remove(handler)
            return True
        except ValueError:
            return False

    def add_message_filter(self, channel: str, filter_func: MessageFilter) -> None:
        """Add message filter for a channel.

        Args:
            channel: Channel name
            filter_func: Function that returns True to allow message, False to filter out
        """
        self._message_filters[channel].append(filter_func)
        logger.debug(f"Added message filter for channel {channel}")

    def get_subscription_info(self, subscriber_id: str) -> Optional[SubscriptionInfo]:
        """Get information about a subscription.

        Args:
            subscriber_id: Subscriber ID

        Returns:
            SubscriptionInfo if found, None otherwise
        """
        return self._subscriptions.get(subscriber_id)

    def get_channel_subscribers(self, channel: str) -> Set[str]:
        """Get subscriber IDs for a channel.

        Args:
            channel: Channel name

        Returns:
            Set of subscriber IDs
        """
        return self._channel_subscribers.get(channel, set()).copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pub/sub manager statistics."""
        active_subscriptions = len(self._subscriptions)
        total_channels = len(self._channel_subscribers)
        total_patterns = len(self._pattern_subscribers)

        channel_stats = {}
        for channel, subscribers in self._channel_subscribers.items():
            channel_stats[channel] = len(subscribers)

        subscription_ages = [sub.age_seconds for sub in self._subscriptions.values()]
        avg_age = (
            sum(subscription_ages) / len(subscription_ages) if subscription_ages else 0
        )

        return {
            "manager": {
                "redis_available": self.redis_client is not None,
                "cleanup_task_running": self._cleanup_task is not None
                and not self._cleanup_task.done(),
            },
            "subscriptions": {
                "active_count": active_subscriptions,
                "total_channels": total_channels,
                "total_patterns": total_patterns,
                "avg_age_seconds": avg_age,
            },
            "performance": {
                "total_publishes": self.total_publishes,
                "total_subscribes": self.total_subscribes,
                "total_unsubscribes": self.total_unsubscribes,
                "total_messages_sent": self.total_messages_sent,
                "total_messages_received": self.total_messages_received,
                "failed_operations": self.failed_operations,
                "success_rate": (
                    self.total_publishes
                    + self.total_subscribes
                    - self.failed_operations
                )
                / max(1, self.total_publishes + self.total_subscribes),
            },
            "channels": channel_stats,
            "config": {
                "max_connections": self.config.max_connections,
                "auto_serialize_json": self.config.auto_serialize_json,
                "max_message_size_bytes": self.config.max_message_size_bytes,
                "enable_pattern_matching": self.config.enable_pattern_matching,
            },
        }

    async def shutdown(self) -> None:
        """Shutdown pub/sub manager and cleanup resources."""
        logger.info("Shutting down PubSubManager")

        await self.stop_cleanup_task()

        # Close all active subscriptions
        subscriber_ids = list(self._subscriptions.keys())
        for subscriber_id in subscriber_ids:
            await self._remove_subscription(subscriber_id)

        # Clear tracking
        self._channel_subscribers.clear()
        self._pattern_subscribers.clear()
        self._message_handlers.clear()
        self._async_message_handlers.clear()
        self._message_filters.clear()

        logger.info("PubSubManager shutdown complete")

    def __repr__(self) -> str:
        return (
            f"PubSubManager(active_subscriptions={len(self._subscriptions)}, "
            f"channels={len(self._channel_subscribers)}, "
            f"success_rate={self.total_publishes - self.failed_operations}/{self.total_publishes})"
        )


# Convenience function for creating pub/sub managers
def create_pubsub_manager(
    redis_client,
    max_connections: int = 10,
    auto_serialize_json: bool = True,
    enable_pattern_matching: bool = True,
    **kwargs,
) -> PubSubManager:
    """Create a pub/sub manager with simple configuration."""
    config = PubSubConfig(
        max_connections=max_connections,
        auto_serialize_json=auto_serialize_json,
        enable_pattern_matching=enable_pattern_matching,
        **kwargs,
    )
    return PubSubManager(redis_client, config)
