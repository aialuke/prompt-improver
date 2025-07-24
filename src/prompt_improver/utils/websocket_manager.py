"""WebSocket Manager for Real-time A/B Testing Analytics
Provides WebSocket connection management and real-time data broadcasting
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time experiment analytics"""

    def __init__(self, redis_client: redis.Redis | None = None):
        # Active WebSocket connections by experiment_id
        self.experiment_connections: dict[str, set[WebSocket]] = defaultdict(set)
        # Connection metadata (user info, connection time, etc.)
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}
        # Redis client for pub/sub messaging
        self.redis_client = redis_client
        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()

    async def connect(
        self, websocket: WebSocket, experiment_id: str, user_id: str | None = None
    ):
        """Accept new WebSocket connection and register it for experiment updates"""
        await websocket.accept()

        # Add to experiment connections
        if experiment_id not in self.experiment_connections:
            self.experiment_connections[experiment_id] = set()
        self.experiment_connections[experiment_id].add(websocket)

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "connected_at": aware_utc_now(),
            "connection_id": str(uuid4()),
        }

        logger.info(
            f"WebSocket connected for experiment {experiment_id}, user {user_id}"
        )

        # Send initial connection confirmation
        await self._send_to_websocket(
            websocket,
            {
                "type": "connection_established",
                "experiment_id": experiment_id,
                "timestamp": aware_utc_now().isoformat(),
                "connection_id": self.connection_metadata[websocket]["connection_id"],
            },
        )

        # Start Redis subscription for this experiment if not already active
        if self.redis_client:
            await self._ensure_redis_subscription(experiment_id)

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection and clean up"""
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            experiment_id = metadata["experiment_id"]

            # Remove from experiment connections
            if experiment_id in self.experiment_connections:
                self.experiment_connections[experiment_id].discard(websocket)

                # Clean up empty experiment groups
                if not self.experiment_connections[experiment_id]:
                    del self.experiment_connections[experiment_id]
                    # Could also unsubscribe from Redis here if no connections remain

            # Remove metadata
            del self.connection_metadata[websocket]

            logger.info(f"WebSocket disconnected for experiment {experiment_id}")

    async def broadcast_to_experiment(
        self, experiment_id: str, message: dict[str, Any]
    ):
        """Broadcast message to all WebSocket connections for specific experiment"""
        if experiment_id not in self.experiment_connections:
            return

        # Add timestamp to message
        message["timestamp"] = aware_utc_now().isoformat()

        # Send to all connections for this experiment
        disconnected_connections = []
        for websocket in self.experiment_connections[experiment_id]:
            try:
                await self._send_to_websocket(websocket, message)
            except WebSocketDisconnect:
                disconnected_connections.append(websocket)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected_connections.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)

    async def broadcast_to_all(self, message: dict[str, Any]):
        """Broadcast message to all active WebSocket connections"""
        for experiment_id in list(self.experiment_connections.keys()):
            await self.broadcast_to_experiment(experiment_id, message)

    async def send_to_connection(self, websocket: WebSocket, message: dict[str, Any]):
        """Send message to specific WebSocket connection"""
        try:
            await self._send_to_websocket(websocket, message)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            await self.disconnect(websocket)

    async def _send_to_websocket(self, websocket: WebSocket, message: dict[str, Any]):
        """Internal method to send JSON message to WebSocket"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise

    async def _ensure_redis_subscription(self, experiment_id: str):
        """Ensure Redis subscription exists for experiment updates"""
        if not self.redis_client:
            return

        # Create background task for Redis subscription if not exists
        subscription_key = f"experiment_updates_{experiment_id}"

        # Check if we already have a subscription task for this experiment
        existing_tasks = [
            task
            for task in self._background_tasks
            if not task.done() and task.get_name() == subscription_key
        ]

        if not existing_tasks:
            task = asyncio.create_task(
                self._redis_subscription_handler(experiment_id), name=subscription_key
            )
            self._background_tasks.add(task)

            # Clean up completed tasks
            self._background_tasks = {
                task for task in self._background_tasks if not task.done()
            }

    async def _redis_subscription_handler(self, experiment_id: str):
        """Handle Redis pub/sub messages for experiment updates"""
        try:
            pubsub = self.redis_client.pubsub()
            channel_name = f"experiment:{experiment_id}:updates"
            await pubsub.subscribe(channel_name)

            logger.info(f"Started Redis subscription for experiment {experiment_id}")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse Redis message and broadcast to WebSocket connections
                        data = json.loads(message["data"])
                        await self.broadcast_to_experiment(experiment_id, data)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Invalid JSON in Redis message: {message['data']}"
                        )
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")

        except Exception as e:
            logger.error(
                f"Redis subscription error for experiment {experiment_id}: {e}"
            )
        finally:
            try:
                await pubsub.unsubscribe()
            except (ConnectionError, redis.RedisError, Exception) as e:
                logger.warning(f"Failed to unsubscribe from Redis pubsub: {e}")
                pass

    def get_connection_count(self, experiment_id: str = None) -> int:
        """Get number of active connections for experiment or total"""
        if experiment_id:
            return len(self.experiment_connections.get(experiment_id, set()))
        return sum(
            len(connections) for connections in self.experiment_connections.values()
        )

    def get_active_experiments(self) -> list[str]:
        """Get list of experiment IDs with active connections"""
        return list(self.experiment_connections.keys())

    async def cleanup(self):
        """Clean up all connections and background tasks"""
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close all WebSocket connections
        for experiment_connections in self.experiment_connections.values():
            for websocket in list(experiment_connections):
                try:
                    await websocket.close()
                except (ConnectionError, Exception) as e:
                    logger.warning(f"Failed to close websocket connection: {e}")
                    pass

        # Clear all data structures
        self.experiment_connections.clear()
        self.connection_metadata.clear()
        self._background_tasks.clear()

# Global connection manager instance
connection_manager = ConnectionManager()

async def setup_redis_connection(
    redis_url: str = "redis://localhost:6379",
) -> redis.Redis:
    """Setup Redis connection for WebSocket manager"""
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()

        # Set Redis client on connection manager
        connection_manager.redis_client = redis_client

        logger.info("Redis connection established for WebSocket manager")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

async def publish_experiment_update(
    experiment_id: str, update_data: dict[str, Any], redis_client: redis.Redis = None
):
    """Publish experiment update to Redis for real-time broadcasting"""
    if not redis_client and connection_manager.redis_client:
        redis_client = connection_manager.redis_client

    if redis_client:
        try:
            channel_name = f"experiment:{experiment_id}:updates"
            message = json.dumps(update_data, default=str)
            await redis_client.publish(channel_name, message)
            logger.debug(f"Published update to Redis channel {channel_name}")
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
    else:
        # Fallback to direct WebSocket broadcast if no Redis
        await connection_manager.broadcast_to_experiment(experiment_id, update_data)
