"""Example: How to integrate startup task orchestration into APES application.

This example demonstrates how to use the init_startup_tasks() function
to properly start all system components in the correct order.
"""
import asyncio
import logging
from pathlib import Path
import signal
import sys
from prompt_improver.core.services.startup import init_startup_tasks, is_startup_complete, shutdown_startup_tasks, startup_context
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APESApplication:
    """Example APES application with proper startup/shutdown orchestration."""

    def __init__(self) -> None:
        self.components = {}
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start the APES application with all components."""
        logger.info('🚀 Starting APES Application...')
        startup_config = {'max_concurrent_tasks': 15, 'session_ttl': 7200, 'cleanup_interval': 600, 'batch_config': {'batch_size': 20, 'batch_timeout': 60, 'max_attempts': 3, 'concurrency': 5, 'dry_run': False}}
        try:
            startup_result = await init_startup_tasks(**startup_config)
            if startup_result['status'] != 'success':
                logger.error('Startup failed: %s', startup_result.get('error', 'Unknown error'))
                return False
            self.components = startup_result['component_refs']
            logger.info('✅ APES Application started successfully!')
            logger.info('   - Startup time: %sms', format(startup_result['startup_time_ms'], '.2f'))
            logger.info('   - Active tasks: %s', startup_result['active_tasks'])
            logger.info('   - Components: %s', list(startup_result['components'].keys()))
            return True
        except Exception as e:
            logger.error('💥 Failed to start APES application: %s', e)
            return False

    async def stop(self):
        """Stop the APES application gracefully."""
        logger.info('🛑 Stopping APES Application...')
        try:
            self.shutdown_event.set()
            shutdown_result = await shutdown_startup_tasks(timeout=30.0)
            if shutdown_result['status'] == 'success':
                logger.info('✅ APES Application stopped successfully in %sms', format(shutdown_result['shutdown_time_ms'], '.2f'))
            else:
                logger.warning('⚠️ APES Application shutdown had issues: %s', shutdown_result.get('error', 'Unknown'))
        except Exception as e:
            logger.error('💥 Error during APES shutdown: %s', e)

    async def run_application_logic(self):
        """Main application logic that runs after startup."""
        logger.info('🏃 Running main application logic...')
        session_store = self.components['session_store']
        batch_processor = self.components['batch_processor']
        health_monitor = self.components['health_monitor']
        counter = 0
        while not self.shutdown_event.is_set():
            try:
                await session_store.set(f'session_{counter}', {'timestamp': asyncio.get_event_loop().time(), 'counter': counter})
                await batch_processor.enqueue({'original': f'Process task {counter}', 'enhanced': f'Enhanced task {counter} with improvements', 'priority': 50, 'session_id': f'session_{counter}'})
                if counter % 10 == 0:
                    health_result = await health_monitor.run_health_check()
                    logger.info('🏥 Health status: %s', health_result.overall_status.value)
                counter += 1
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                logger.info('📱 Application logic cancelled')
                break
            except Exception as e:
                logger.error('❌ Error in application logic: %s', e)
                await asyncio.sleep(1)
        logger.info('🏁 Application logic stopped')

    async def run(self):
        """Run the complete application lifecycle."""
        if not await self.start():
            return 1
        try:
            await self.run_application_logic()
            return 0
        except KeyboardInterrupt:
            logger.info('📱 Received keyboard interrupt')
            return 0
        except Exception as e:
            logger.error('💥 Application error: %s', e)
            return 1
        finally:
            await self.stop()

async def main_with_context_manager():
    """Alternative main function using the startup context manager."""
    logger.info('🎯 Starting APES with context manager...')
    try:
        async with startup_context(max_concurrent_tasks=10, session_ttl=3600, cleanup_interval=300) as components:
            logger.info('✅ APES components started successfully!')
            logger.info('Available components: %s', list(components.keys()))
            session_store = components['session_store']
            await session_store.set('demo_session', {'demo': 'data'})
            data = await session_store.get('demo_session')
            logger.info('📦 Session data: %s', data)
            logger.info('🏃 Simulating application work for 30 seconds...')
            await asyncio.sleep(30)
            logger.info('🏁 Application work completed')
        logger.info('✅ APES components automatically shut down')
    except Exception as e:
        logger.error('💥 Error in context manager example: %s', e)
        return 1
    return 0

def setup_signal_handlers(app: APESApplication):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame) -> None:
        logger.info('📱 Received signal %s, initiating graceful shutdown...', signum)
        if not app.shutdown_event.is_set():
            app.shutdown_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point for the application."""
    logger.info('🎬 APES Application Example Starting...')
    use_context_manager = len(sys.argv) > 1 and sys.argv[1] == '--context'
    if use_context_manager:
        return await main_with_context_manager()
    app = APESApplication()
    setup_signal_handlers(app)
    return await app.run()
if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info('👋 Application terminated by user')
        sys.exit(0)
    except Exception as e:
        logger.error('💥 Fatal application error: %s', e)
        sys.exit(1)
