"""Session management tools for MCP server.

Handles session CRUD operations and session-related functionality.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlmodel import Field

if TYPE_CHECKING:
    from prompt_improver.shared.interfaces.protocols.mcp import (
        MCPServerProtocol as APESMCPServer,
    )

logger = logging.getLogger(__name__)


def setup_session_tools(server: "APESMCPServer") -> None:
    """Register session management tools with the server."""

    @server.mcp.tool()
    async def get_session(
        session_id: str = Field(..., description="Session ID to retrieve"),
    ) -> dict[str, Any]:
        """Retrieve session data from the session store."""
        return await _get_session_impl(server, session_id)

    @server.mcp.tool()
    async def set_session(
        session_id: str = Field(..., description="Session ID to set"),
        data: dict[str, Any] = Field(..., description="Session data to store"),
    ) -> dict[str, Any]:
        """Store session data in the session store."""
        return await _set_session_impl(server, session_id, data)

    @server.mcp.tool()
    async def touch_session(
        session_id: str = Field(..., description="Session ID to touch"),
    ) -> dict[str, Any]:
        """Update session last access time."""
        return await _touch_session_impl(server, session_id)

    @server.mcp.tool()
    async def delete_session(
        session_id: str = Field(..., description="Session ID to delete"),
    ) -> dict[str, Any]:
        """Delete session data from the session store."""
        return await _delete_session_impl(server, session_id)


# Implementation functions


async def _get_session_impl(server: "APESMCPServer", session_id: str) -> dict[str, Any]:
    """Implementation of get_session tool using unified session management."""
    try:
        await server._ensure_unified_session_manager()
        data = await server._unified_session_manager.get_mcp_session(session_id)

        if data is None:
            return {
                "session_id": session_id,
                "exists": False,
                "message": "Session not found",
                "timestamp": time.time(),
                "source": "unified_session_manager",
            }

        return {
            "session_id": session_id,
            "exists": True,
            "data": data,
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "exists": False,
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }


async def _set_session_impl(
    server: "APESMCPServer", session_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Implementation of set_session tool using unified session management."""
    try:
        await server._ensure_unified_session_manager()
        success = await server.services.session_store.set_session(session_id, data, ttl=3600)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session data stored successfully"
            if success
            else "Failed to store session data",
            "data_keys": list(data.keys()),
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
            "source": "unified_session_manager",
        }


async def _touch_session_impl(
    server: "APESMCPServer", session_id: str
) -> dict[str, Any]:
    """Implementation of touch_session tool."""
    try:
        success = await server.services.session_store.touch_session(session_id, ttl=3600)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session touched successfully"
            if success
            else "Session not found",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


async def _delete_session_impl(
    server: "APESMCPServer", session_id: str
) -> dict[str, Any]:
    """Implementation of delete_session tool."""
    try:
        success = await server.services.session_store.delete_session(session_id)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Session deleted successfully"
            if success
            else "Session not found",
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }
