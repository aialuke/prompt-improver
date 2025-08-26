"""MCP Server Protocol Definitions.

This module contains protocol interfaces for MCP server components,
extracted to eliminate circular import dependencies between MCP modules.

Following the existing pattern of 113+ protocol files in the codebase,
this provides clean separation of concerns and dependency injection patterns.
"""

from prompt_improver.mcp_server.protocols.server_protocols import (
    MCPServerProtocol,
    ServerConfigProtocol,
    ServerServicesProtocol,
)

__all__ = [
    "MCPServerProtocol",
    "ServerConfigProtocol",
    "ServerServicesProtocol",
]
