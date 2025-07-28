"""
Refactored MCP Server implementation using Service Facade pattern.

This version demonstrates how to reduce coupling by using the MCPServiceFacade
instead of directly importing and managing all dependencies.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Single import for all service dependencies
from prompt_improver.mcp_server.services import MCPServiceFacade, create_mcp_service_facade

# Configure logging to stderr for MCP protocol compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class PromptRequest(BaseModel):
    """Request model for prompt improvement"""
    prompt: str = Field(..., description="The prompt to improve")
    context: Optional[dict] = Field(default_factory=dict, description="Additional context")
    client_id: Optional[str] = Field(default="anonymous", description="Client identifier")

class BatchRequest(BaseModel):
    """Request model for batch prompt improvement"""
    items: list = Field(..., description="List of prompts to improve")
    client_id: Optional[str] = Field(default="anonymous", description="Client identifier")

class RefactoredMCPServer:
    """Refactored MCP Server with reduced coupling"""

    def __init__(self):
        self.mcp = FastMCP(
            name="APES - Adaptive Prompt Enhancement System (Refactored)",
            description="AI-powered prompt optimization service using clean architecture",
        )

        # Single service facade instead of multiple dependencies
        self.service_facade: Optional[MCPServiceFacade] = None

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP request handlers"""

        @self.mcp.tool()
        async def improve_prompt(request: PromptRequest) -> dict:
            """Improve a single prompt using ML-driven rules"""
            if not self.service_facade:
                return {"error": "Service not initialized", "status": "error"}

            return await self.service_facade.process_prompt_request(
                prompt=request.prompt,
                context=request.context,
                client_id=request.client_id
            )

        # Batch processing removed per architectural separation requirements
        # Batch processing should be handled by separate ML training system

        @self.mcp.tool()
        async def get_server_status() -> dict:
            """Get comprehensive server status and health metrics"""
            if not self.service_facade:
                return {"error": "Service not initialized", "status": "error"}

            return await self.service_facade.get_server_status()

        @self.mcp.tool()
        async def health_check() -> dict:
            """Simple health check endpoint"""
            if not self.service_facade:
                return {"status": "unhealthy", "message": "Service not initialized"}

            try:
                status = await self.service_facade.get_server_status()
                return {
                    "status": status.get("status", "unknown"),
                    "timestamp": status.get("timestamp"),
                    "message": "Service operational"
                }
            except Exception as e:
                return {"status": "unhealthy", "message": str(e)}

    async def initialize(self) -> bool:
        """Initialize the server and all services"""
        try:
            logger.info("Initializing MCP Server with Service Facade...")

            # Create service facade
            self.service_facade = create_mcp_service_facade()

            # Initialize all services through facade
            if not await self.service_facade.initialize():
                logger.error("Failed to initialize service facade")
                return False

            logger.info("MCP Server initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP Server: {e}")
            return False

    async def shutdown(self):
        """Gracefully shutdown the server"""
        try:
            logger.info("Shutting down MCP Server...")

            if self.service_facade:
                await self.service_facade.shutdown()

            logger.info("MCP Server shutdown completed")

        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")

    def run(self):
        """Run the MCP server"""
        async def main():
            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()

            def signal_handler():
                logger.info("Received shutdown signal")
                asyncio.create_task(self.shutdown())

            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(sig, signal_handler)

            # Initialize server
            if not await self.initialize():
                logger.error("Server initialization failed")
                sys.exit(1)

            try:
                # Run the MCP server
                await self.mcp.run()
            finally:
                await self.shutdown()

        # Run the server
        asyncio.run(main())

# Global server instance
server = RefactoredMCPServer()

# Entry point for the refactored server
def main():
    """Main entry point for the refactored MCP server"""
    logger.info("Starting Refactored MCP Server...")
    server.run()

if __name__ == "__main__":
    main()
