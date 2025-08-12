"""Publish/subscribe messaging services for database connections.

This package provides pub/sub messaging functionality extracted from
unified_connection_manager.py, implementing:

- PubSubManager: Redis-based messaging with channel management and pattern matching
- PubSubConfig: Configurable connection pools, message serialization, and performance settings
- SubscriptionInfo: Active subscription tracking with automatic cleanup
- Message filtering and transformation with custom handlers
- Pattern-based subscriptions with wildcard matching support

Designed for production messaging with high-throughput pub/sub operations.
"""

from .pubsub_manager import (
    AsyncMessageHandler,
    MessageFilter,
    MessageHandler,
    PubSubConfig,
    PubSubManager,
    SubscriptionInfo,
    create_pubsub_manager,
)

__all__ = [
    "PubSubManager",
    "PubSubConfig",
    "SubscriptionInfo",
    "create_pubsub_manager",
    "MessageHandler",
    "AsyncMessageHandler",
    "MessageFilter",
]
