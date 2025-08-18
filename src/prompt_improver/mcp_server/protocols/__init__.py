"""MCP Server Protocol Definitions.

This module contains protocol interfaces for MCP server components,
extracted to eliminate circular import dependencies between MCP modules.

Following the existing pattern of 113+ protocol files in the codebase,
this provides clean separation of concerns and dependency injection patterns.
"""

from .server_protocols import (
    MCPServerProtocol,
    ServerServicesProtocol,
    ServerConfigProtocol,
)

__all__ = [
    "MCPServerProtocol",
    "ServerServicesProtocol", 
    "ServerConfigProtocol",
]