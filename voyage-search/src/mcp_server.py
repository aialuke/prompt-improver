#!/usr/bin/env python3
"""
MCP Server for Claude Code CLI Integration with Enhanced Voyage AI Search
Provides semantic code search as an MCP tool that Claude Code can use directly.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel
    )
except ImportError:
    print("❌ MCP library not available. Install with: pip install mcp")
    sys.exit(1)

# Import from the same directory
try:
    from .search_integration import ClaudeCodeSearchCLI, ClaudeCodeSearchRequest
except ImportError:
    # Fallback for direct execution
    from search_integration import ClaudeCodeSearchCLI, ClaudeCodeSearchRequest


class VoyageAISearchMCPServer:
    """MCP Server for Voyage AI semantic code search."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server("voyage-ai-search")
        self.search_cli = None
        
        # Register tools
        self._register_tools()
        
        # Register handlers
        self._register_handlers()
    
    def _register_tools(self):
        """Register MCP tools for semantic search."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="semantic_search",
                    description="Semantic code search with Voyage AI embeddings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for semantic code search"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 3)",
                                "default": 3
                            },
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity threshold 0.0-1.0 (default: 0.3)",
                                "default": 0.3
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["explain", "optimize", "review", "refactor"],
                                "description": "Analysis type for Claude Code integration"
                            },
                            "search_method": {
                                "type": "string",
                                "enum": ["semantic", "lexical", "hybrid"],
                                "description": "Search method (default: hybrid)",
                                "default": "hybrid"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_explain",
                    description="Search code and generate explanation prompt",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_optimize",
                    description="Search code and generate optimization suggestions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            
            # Initialize search CLI if needed
            if self.search_cli is None:
                try:
                    self.search_cli = ClaudeCodeSearchCLI()
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"❌ Failed to initialize search system: {e}\n\nEnsure VOYAGE_API_KEY is set and embeddings.pkl exists."
                    )]
            
            try:
                if name == "semantic_search":
                    return await self._handle_semantic_search(arguments)
                elif name == "search_explain":
                    return await self._handle_search_explain(arguments)
                elif name == "search_optimize":
                    return await self._handle_search_optimize(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"❌ Tool execution failed: {e}"
                )]
    
    def _register_handlers(self):
        """Register MCP handlers."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="voyage://search/status",
                    name="Search System Status",
                    description="Status of the Voyage AI search system",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content."""
            if uri == "voyage://search/status":
                status = {
                    "system": "Voyage AI Enhanced Search",
                    "version": "2025.1",
                    "features": [
                        "Contextualized embeddings (voyage-context-3)",
                        "Binary rescoring for performance",
                        "Voyage AI reranking (rerank-2.5-lite)",
                        "Hybrid search (semantic + lexical)",
                        "Claude Code integration"
                    ],
                    "api_key_configured": bool(os.getenv("VOYAGE_API_KEY")),
                    "embeddings_available": os.path.exists("embeddings.pkl")
                }
                return json.dumps(status, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def _handle_semantic_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle semantic search tool call."""
        
        request = ClaudeCodeSearchRequest(
            query=arguments["query"],
            max_results=arguments.get("max_results", 3),
            min_similarity=arguments.get("min_similarity", 0.3),
            search_method=arguments.get("search_method", "hybrid"),
            analysis_type=arguments.get("analysis_type")
        )
        
        result = self.search_cli.search(request)
        
        if not result.results:
            return [TextContent(
                type="text",
                text=f"No relevant code found for query: '{request.query}'"
            )]
        
        # Format results for Claude Code
        formatted_output = self.search_cli.search_and_format_for_claude(
            query=request.query,
            analysis_type=request.analysis_type or "explain",
            max_results=request.max_results,
            min_similarity=request.min_similarity
        )
        
        return [TextContent(
            type="text",
            text=formatted_output
        )]
    
    async def _handle_search_explain(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle search and explain tool call."""
        
        formatted_output = self.search_cli.search_and_format_for_claude(
            query=arguments["query"],
            analysis_type="explain",
            max_results=arguments.get("max_results", 3),
            min_similarity=0.3
        )
        
        return [TextContent(
            type="text",
            text=formatted_output
        )]
    
    async def _handle_search_optimize(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle search and optimize tool call."""
        
        formatted_output = self.search_cli.search_and_format_for_claude(
            query=arguments["query"],
            analysis_type="optimize",
            max_results=arguments.get("max_results", 3),
            min_similarity=0.3
        )
        
        return [TextContent(
            type="text",
            text=formatted_output
        )]
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="voyage-ai-search",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None
                    )
                )
            )


async def main():
    """Main entry point."""

    # Check requirements - FIXED: Set hardcoded API key if not in environment
    if not os.getenv("VOYAGE_API_KEY"):
        os.environ["VOYAGE_API_KEY"] = 'pa-fhFUHl_VQ2SjvA7TNQ0gF8v021efH3R82qbigC3pnnB'
        print("🔧 Using hardcoded API key for MCP server", file=sys.stderr)

    if not os.path.exists("embeddings.pkl"):
        print("❌ embeddings.pkl not found. Run generate_embeddings.py first", file=sys.stderr)
        sys.exit(1)
    
    # Create and run server
    server = VoyageAISearchMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
