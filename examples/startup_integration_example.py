"""Example: How to integrate startup task orchestration into APES application.

This example demonstrates how to use the init_startup_tasks() function
to properly start all system components in the correct order.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from prompt_improver.core.services.startup import (
    init_startup_tasks,
    is_startup_complete,
    shutdown_startup_tasks,
    startup_context,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APESApplication:
    """Example APES application with proper startup/shutdown orchestration."""

    def __init__(self) -> None:
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
                "dry_run": False,  # Enable real processing
            },
        }

        try:
            # Initialize all startup tasks
            startup_result = await init_startup_tasks(**startup_config)

            if startup_result["status"] != "success":
                logger.error(
                    f"Startup failed: {startup_result.get('error', 'Unknown error')}"
                )
                return False

            # Store component references for use in application
            self.components = startup_result["component_refs"]

            logger.info("✅ APES Application started successfully!")
            logger.info(f"   - Startup time: {startup_result['startup_time_ms']:.2f}ms")
            logger.info(f"   - Active tasks: {startup_result['active_tasks']}")
            logger.info(f"   - Components: {list(startup_result['components'].keys())}")

            return True

        except Exception as e:
            logger.error(f"💥 Failed to start APES application: {e}")
            return False

    async def stop(self):
        """Stop the APES application gracefully."""
        logger.info("🛑 Stopping APES Application...")

        try:
            # Signal shutdown to application logic
            self.shutdown_event.set()

            # Shutdown all startup tasks
            shutdown_result = await shutdown_startup_tasks(timeout=30.0)

            if shutdown_result["status"] == "success":
                logger.info(
                    f"✅ APES Application stopped successfully in {shutdown_result['shutdown_time_ms']:.2f}ms"
                )
            else:
                logger.warning(
                    f"⚠️ APES Application shutdown had issues: {shutdown_result.get('error', 'Unknown')}"
                )

        except Exception as e:
            logger.error(f"💥 Error during APES shutdown: {e}")

    async def run_application_logic(self):
        """Main application logic that runs after startup."""
        logger.info("🏃 Running main application logic...")

        # Example: Access components initialized by startup
        session_store = self.components["session_store"]
        batch_processor = self.components["batch_processor"]
        health_service = self.components["health_service"]

        # Simulate application work
        counter = 0
        while not self.shutdown_event.is_set():
            try:
                # Example: Store some session data
                await session_store.set(
                    f"session_{counter}",
                    {
                        "timestamp": asyncio.get_event_loop().time(),
                        "counter": counter,
                    },
                )

                # Example: Add work to batch processor
                await batch_processor.enqueue({
                    "original": f"Process task {counter}",
                    "enhanced": f"Enhanced task {counter} with improvements",
                    "priority": 50,
                    "session_id": f"session_{counter}",
                })

                # Periodic health check reporting
                if counter % 10 == 0:
                    health_result = await health_service.run_health_check()
                    logger.info(
                        f"🏥 Health status: {health_result.overall_status.value}"
                    )

                counter += 1
                await asyncio.sleep(5)  # Simulate work every 5 seconds

            except asyncio.CancelledError:
                logger.info("📱 Application logic cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in application logic: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

        logger.info("🏁 Application logic stopped")

    async def run(self):
        """Run the complete application lifecycle."""
        # Start all components
        if not await self.start():
            return 1

        try:
            # Run main application logic
            await self.run_application_logic()
            return 0

        except KeyboardInterrupt:
            logger.info("📱 Received keyboard interrupt")
            return 0
        except Exception as e:
            logger.error(f"💥 Application error: {e}")
            return 1
        finally:
            # Always ensure clean shutdown
            await self.stop()


async def main_with_context_manager():
    """Alternative main function using the startup context manager."""
    logger.info("🎯 Starting APES with context manager...")

    try:
        # Use context manager for automatic cleanup
        async with startup_context(
            max_concurrent_tasks=10, session_ttl=3600, cleanup_interval=300
        ) as components:
            logger.info("✅ APES components started successfully!")
            logger.info(f"Available components: {list(components.keys())}")

            # Simulate some application work
            session_store = components["session_store"]

            # Example session operations
            await session_store.set("demo_session", {"demo": "data"})
            data = await session_store.get("demo_session")
            logger.info(f"📦 Session data: {data}")

            # Simulate running for a short time
            logger.info("🏃 Simulating application work for 30 seconds...")
            await asyncio.sleep(30)

            logger.info("🏁 Application work completed")

        # Components automatically cleaned up here
        logger.info("✅ APES components automatically shut down")

    except Exception as e:
        logger.error(f"💥 Error in context manager example: {e}")
        return 1

    return 0


def setup_signal_handlers(app: APESApplication):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame) -> None:
        logger.info(f"📱 Received signal {signum}, initiating graceful shutdown...")
        # Set the shutdown event to trigger application shutdown
        if not app.shutdown_event.is_set():
            app.shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the application."""
    logger.info("🎬 APES Application Example Starting...")

    # Choose which example to run
    use_context_manager = len(sys.argv) > 1 and sys.argv[1] == "--context"

    if use_context_manager:
        return await main_with_context_manager()
    
    # Use full application class
    app = APESApplication()
    setup_signal_handlers(app)
    return await app.run()


if __name__ == "__main__":
    try:
        # Run the application
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal application error: {e}")
        sys.exit(1)
