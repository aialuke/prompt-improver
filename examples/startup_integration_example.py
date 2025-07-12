"""Example: How to integrate startup task orchestration into APES application.

This example demonstrates how to use the init_startup_tasks() function
to properly start all system components in the correct order.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.services.startup import (
    init_startup_tasks,
    shutdown_startup_tasks,
    startup_context,
    is_startup_complete
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APESApplication:
    """Example APES application with proper startup/shutdown orchestration."""
    
    def __init__(self):
        self.components = {}
        self.shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the APES application with all components."""
        logger.info("🚀 Starting APES Application...")
        
        # Configuration for startup
        startup_config = {
            "max_concurrent_tasks": 15,
            "session_ttl": 7200,  # 2 hours
            "cleanup_interval": 600,  # 10 minutes
            "batch_config": {
                "batch_size": 20,
                "batch_timeout": 60,
                "max_attempts": 3,
                "concurrency": 5,
                "dry_run": False  # Enable real processing
            }
        }
        
        try:
            # Initialize all startup tasks
            startup_result = await init_startup_tasks(**startup_config)
            
            if startup_result["status"] != "success":
                logger.error(f"Startup failed: {startup_result.get('error', 'Unknown error')}")
                return False
            
            # Store component references for use in application
            self.components = startup_result["component_refs"]
            
            logger.info(f"✅ APES Application started successfully!")
            logger.info(f"   - Startup time: {startup_result['startup_time_ms']:.2f}ms")
            logger.info(f"   - Active tasks: {startup_result['active_tasks']}")
            logger.info(f"   - Components: {list(startup_result['components'].keys())}")
            
            return True
            
        except Exception as e:\n            logger.error(f"💥 Failed to start APES application: {e}")\n            return False\n    \n    async def stop(self):\n        \"\"\"Stop the APES application gracefully.\"\"\"\n        logger.info(\"🛑 Stopping APES Application...\")\n        \n        try:\n            # Signal shutdown to application logic\n            self.shutdown_event.set()\n            \n            # Shutdown all startup tasks\n            shutdown_result = await shutdown_startup_tasks(timeout=30.0)\n            \n            if shutdown_result[\"status\"] == \"success\":\n                logger.info(f\"✅ APES Application stopped successfully in {shutdown_result['shutdown_time_ms']:.2f}ms\")\n            else:\n                logger.warning(f\"⚠️ APES Application shutdown had issues: {shutdown_result.get('error', 'Unknown')}\")\n                \n        except Exception as e:\n            logger.error(f\"💥 Error during APES shutdown: {e}\")\n    \n    async def run_application_logic(self):\n        \"\"\"Main application logic that runs after startup.\"\"\"\n        logger.info(\"🏃 Running main application logic...\")\n        \n        # Example: Access components initialized by startup\n        session_store = self.components[\"session_store\"]\n        batch_processor = self.components[\"batch_processor\"]\n        health_service = self.components[\"health_service\"]\n        \n        # Simulate application work\n        counter = 0\n        while not self.shutdown_event.is_set():\n            try:\n                # Example: Store some session data\n                await session_store.set(f\"session_{counter}\", {\n                    \"timestamp\": asyncio.get_event_loop().time(),\n                    \"counter\": counter\n                })\n                \n                # Example: Add work to batch processor\n                await batch_processor.enqueue({\n                    \"original\": f\"Process task {counter}\",\n                    \"enhanced\": f\"Enhanced task {counter} with improvements\",\n                    \"priority\": 50,\n                    \"session_id\": f\"session_{counter}\"\n                })\n                \n                # Periodic health check reporting\n                if counter % 10 == 0:\n                    health_result = await health_service.run_health_check()\n                    logger.info(f\"🏥 Health status: {health_result.overall_status.value}\")\n                \n                counter += 1\n                await asyncio.sleep(5)  # Simulate work every 5 seconds\n                \n            except asyncio.CancelledError:\n                logger.info(\"📱 Application logic cancelled\")\n                break\n            except Exception as e:\n                logger.error(f\"❌ Error in application logic: {e}\")\n                await asyncio.sleep(1)  # Brief pause before retry\n        \n        logger.info(\"🏁 Application logic stopped\")\n    \n    async def run(self):\n        \"\"\"Run the complete application lifecycle.\"\"\"\n        # Start all components\n        if not await self.start():\n            return 1\n        \n        try:\n            # Run main application logic\n            await self.run_application_logic()\n            return 0\n            \n        except KeyboardInterrupt:\n            logger.info(\"📱 Received keyboard interrupt\")\n            return 0\n        except Exception as e:\n            logger.error(f\"💥 Application error: {e}\")\n            return 1\n        finally:\n            # Always ensure clean shutdown\n            await self.stop()\n\n\nasync def main_with_context_manager():\n    \"\"\"Alternative main function using the startup context manager.\"\"\"\n    logger.info(\"🎯 Starting APES with context manager...\")\n    \n    try:\n        # Use context manager for automatic cleanup\n        async with startup_context(\n            max_concurrent_tasks=10,\n            session_ttl=3600,\n            cleanup_interval=300\n        ) as components:\n            logger.info(\"✅ APES components started successfully!\")\n            logger.info(f\"Available components: {list(components.keys())}\")\n            \n            # Simulate some application work\n            session_store = components[\"session_store\"]\n            \n            # Example session operations\n            await session_store.set(\"demo_session\", {\"demo\": \"data\"})\n            data = await session_store.get(\"demo_session\")\n            logger.info(f\"📦 Session data: {data}\")\n            \n            # Simulate running for a short time\n            logger.info(\"🏃 Simulating application work for 30 seconds...\")\n            await asyncio.sleep(30)\n            \n            logger.info(\"🏁 Application work completed\")\n            \n        # Components automatically cleaned up here\n        logger.info(\"✅ APES components automatically shut down\")\n        \n    except Exception as e:\n        logger.error(f\"💥 Error in context manager example: {e}\")\n        return 1\n    \n    return 0\n\n\ndef setup_signal_handlers(app: APESApplication):\n    \"\"\"Setup signal handlers for graceful shutdown.\"\"\"\n    def signal_handler(signum, frame):\n        logger.info(f\"📱 Received signal {signum}, initiating graceful shutdown...\")\n        # Set the shutdown event to trigger application shutdown\n        if not app.shutdown_event.is_set():\n            app.shutdown_event.set()\n    \n    # Register signal handlers\n    signal.signal(signal.SIGINT, signal_handler)\n    signal.signal(signal.SIGTERM, signal_handler)\n\n\nasync def main():\n    \"\"\"Main entry point for the application.\"\"\"\n    logger.info(\"🎬 APES Application Example Starting...\")\n    \n    # Choose which example to run\n    use_context_manager = len(sys.argv) > 1 and sys.argv[1] == \"--context\"\n    \n    if use_context_manager:\n        return await main_with_context_manager()\n    else:\n        # Use full application class\n        app = APESApplication()\n        setup_signal_handlers(app)\n        return await app.run()\n\n\nif __name__ == \"__main__\":\n    try:\n        # Run the application\n        exit_code = asyncio.run(main())\n        sys.exit(exit_code)\n    except KeyboardInterrupt:\n        logger.info(\"👋 Application terminated by user\")\n        sys.exit(0)\n    except Exception as e:\n        logger.error(f\"💥 Fatal application error: {e}\")\n        sys.exit(1)\n"
