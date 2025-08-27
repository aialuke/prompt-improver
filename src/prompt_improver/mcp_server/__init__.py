"""MCP Server module for APES - Modernized Architecture."""

from prompt_improver.mcp_server.lifecycle import main
from prompt_improver.mcp_server.server import APESMCPServer

__all__ = ["APESMCPServer", "main"]
