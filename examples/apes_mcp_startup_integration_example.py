"""Example: Integration of APESMCPServer with startup service session store.

This example demonstrates how to integrate APESMCPServer's session store
with the startup service to avoid creating competing session store instances.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.mcp_server.server import APESMCPServer
from prompt_improver.core.services.startup import init_startup_tasks, shutdown_startup_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedAPESApplication:
    """Example showing APESMCPServer and startup service integration."""

    def __init__(self):
        self.mcp_server = None
        self.startup_components = {}

    async def start(self):
        """Start the integrated application."""
        logger.info("🚀 Starting Integrated APES Application...")

        try:
            # Step 1: Initialize APESMCPServer (which creates its own session store)
            self.mcp_server = APESMCPServer()
            await self.mcp_server.initialize()
            
            # Extract the session store from APESMCPServer
            session_store = self.mcp_server.services.session_store
            logger.info(f"📦 Using APESMCPServer session store: {type(session_store).__name__}")

            # Step 2: Initialize startup service with the MCP server's session store
            # This prevents creating a competing session store instance
            startup_result = await init_startup_tasks(
                max_concurrent_tasks=10,
                session_ttl=3600,
                cleanup_interval=300,
                session_store=session_store  # Inject the APESMCPServer's session store
            )

            if startup_result["status"] != "success":
                logger.error(f"Startup failed: {startup_result.get('error', 'Unknown error')}")
                return False

            self.startup_components = startup_result["component_refs"]
            
            # Verify that both services are using the same session store instance
            startup_session_store = self.startup_components["session_store"]
            mcp_session_store = self.mcp_server.services.session_store
            
            if startup_session_store is mcp_session_store:
                logger.info("✅ Session store integration successful - single instance shared")
            else:
                logger.warning("⚠️ Session store integration issue - different instances detected")

            logger.info("✅ Integrated APES Application started successfully!")
            return True

        except Exception as e:
            logger.error(f"💥 Failed to start integrated application: {e}")
            return False

    async def demonstrate_session_sharing(self):
        """Demonstrate that both services share the same session store."""
        logger.info("🧪 Demonstrating session store sharing...")

        try:
            # Set data through MCP server session store
            await self.mcp_server.services.session_store.set("shared_key", {"source": "mcp_server"})
            
            # Retrieve data through startup service session store
            startup_session_store = self.startup_components["session_store"]
            data = await startup_session_store.get("shared_key")
            
            if data and data.get("source") == "mcp_server":
                logger.info("✅ Session data successfully shared between services")
                logger.info(f"   Retrieved data: {data}")
            else:
                logger.warning("⚠️ Session sharing not working correctly")

        except Exception as e:
            logger.error(f"❌ Error during session sharing test: {e}")

    async def stop(self):
        """Stop the integrated application."""
        logger.info("🛑 Stopping Integrated APES Application...")

        try:
            # Shutdown startup service first (this will NOT stop the session store
            # because it detects it's managed externally)
            shutdown_result = await shutdown_startup_tasks(timeout=30.0)
            
            if shutdown_result["status"] == "success":
                logger.info("✅ Startup service shutdown completed")
            else:
                logger.warning(f"⚠️ Startup service shutdown had issues: {shutdown_result.get('error')}")

            # Shutdown MCP server (this will handle the session store cleanup)
            if self.mcp_server:
                await self.mcp_server.shutdown()
                logger.info("✅ MCP server shutdown completed")

            logger.info("✅ Integrated application stopped successfully")

        except Exception as e:
            logger.error(f"💥 Error during integrated shutdown: {e}")

    async def run(self):
        """Run the complete integrated application lifecycle."""
        if not await self.start():
            return 1

        try:
            # Demonstrate the integration
            await self.demonstrate_session_sharing()
            
            # Simulate some application work
            logger.info("🏃 Simulating application work for 10 seconds...")
            await asyncio.sleep(10)
            
            return 0

        except KeyboardInterrupt:
            logger.info("📱 Received keyboard interrupt")
            return 0
        except Exception as e:
            logger.error(f"💥 Application error: {e}")
            return 1
        finally:
            await self.stop()


async def main():
    """Main entry point for the integration example."""
    logger.info("🎬 APES-MCP Startup Integration Example Starting...")
    
    app = IntegratedAPESApplication()
    return await app.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("👋 Integration example terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal integration example error: {e}")
        sys.exit(1)