"""WebSocket Connection Manager for API endpoints.

Centralized WebSocket connection management following clean architecture principles.
"""

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Centralized WebSocket connection manager for all API endpoints.

    Manages WebSocket connections, groups, channels, and broadcasting
    following clean architecture patterns.
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: dict[str, WebSocket] = {}
        self.groups: dict[str, set[str]] = {}
        self.channels: dict[str, set[str]] = {}
        self.connection_metadata: dict[str, dict] = {}

    async def connect_to_group(self, websocket: WebSocket, group: str, user_id: str):
        """Connect WebSocket to a group."""
        try:
            await websocket.accept()
            connection_id = f"{user_id}_{group}"
            self.active_connections[connection_id] = websocket

            if group not in self.groups:
                self.groups[group] = set()
            self.groups[group].add(connection_id)

            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "group": group,
                "type": "group"
            }

            logger.info(f"WebSocket connected to group {group} for user {user_id}")

        except Exception as e:
            logger.exception(f"Failed to connect to group {group}: {e}")
            await self.disconnect(websocket)

    async def connect(self, websocket: WebSocket, channel: str, user_id: str):
        """Connect WebSocket to a channel."""
        try:
            await websocket.accept()
            connection_id = f"{user_id}_{channel}"
            self.active_connections[connection_id] = websocket

            if channel not in self.channels:
                self.channels[channel] = set()
            self.channels[channel].add(connection_id)

            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "channel": channel,
                "type": "channel"
            }

            logger.info(f"WebSocket connected to channel {channel} for user {user_id}")

        except Exception as e:
            logger.exception(f"Failed to connect to channel {channel}: {e}")
            await self.disconnect(websocket)

    async def send_to_connection(self, websocket: WebSocket, message: dict):
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.exception(f"Failed to send message to connection: {e}")
            await self.disconnect(websocket)

    async def broadcast_to_group(self, group: str, message: dict):
        """Broadcast message to all connections in a group."""
        if group not in self.groups:
            logger.warning(f"Group {group} not found")
            return

        disconnected_connections = []
        for connection_id in self.groups[group].copy():
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.exception(f"Failed to broadcast to {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
            else:
                disconnected_connections.append(connection_id)

        # Clean up disconnected connections
        for conn_id in disconnected_connections:
            self.groups[group].discard(conn_id)
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
            if conn_id in self.connection_metadata:
                del self.connection_metadata[conn_id]

    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast message to all connections in a channel."""
        if channel not in self.channels:
            logger.warning(f"Channel {channel} not found")
            return

        disconnected_connections = []
        for connection_id in self.channels[channel].copy():
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.exception(f"Failed to broadcast to {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
            else:
                disconnected_connections.append(connection_id)

        # Clean up disconnected connections
        for conn_id in disconnected_connections:
            self.channels[channel].discard(conn_id)
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
            if conn_id in self.connection_metadata:
                del self.connection_metadata[conn_id]

    def get_connection_count(self, channel: str | None = None, group: str | None = None) -> int:
        """Get connection count for channel, group, or total."""
        if channel:
            return len(self.channels.get(channel, set()))
        if group:
            return len(self.groups.get(group, set()))
        return len(self.active_connections)

    def get_active_experiments(self) -> list[str]:
        """Get list of active experiment groups/channels."""
        active_experiments = []

        # Get groups that have active connections
        for group, connections in self.groups.items():
            if connections and group.startswith('experiment_'):
                active_experiments.append(group)

        # Get channels that have active connections
        for channel, connections in self.channels.items():
            if connections and channel.startswith('experiment_'):
                active_experiments.append(channel)

        return list(set(active_experiments))

    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket and clean up."""
        try:
            await websocket.close()
        except:
            pass  # Connection might already be closed

        # Find and remove connection from all data structures
        connection_to_remove = None
        for conn_id, ws in self.active_connections.items():
            if ws == websocket:
                connection_to_remove = conn_id
                break

        if connection_to_remove:
            # Remove from active connections
            del self.active_connections[connection_to_remove]

            # Remove from groups
            for connections in self.groups.values():
                connections.discard(connection_to_remove)

            # Remove from channels
            for connections in self.channels.values():
                connections.discard(connection_to_remove)

            # Remove metadata
            if connection_to_remove in self.connection_metadata:
                del self.connection_metadata[connection_to_remove]

            logger.info(f"WebSocket connection {connection_to_remove} disconnected")

    def get_connection_info(self) -> dict:
        """Get detailed connection information for monitoring."""
        return {
            "total_connections": len(self.active_connections),
            "groups": {group: len(connections) for group, connections in self.groups.items()},
            "channels": {channel: len(connections) for channel, connections in self.channels.items()},
            "active_experiments": self.get_active_experiments()
        }


# Global singleton instance
websocket_manager = WebSocketConnectionManager()
