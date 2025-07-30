"""Startup Task Orchestration for the Adaptive Prompt Enhancement System (APES).

Implements robust startup task management with proper error handling,
graceful shutdown, and component health monitoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Set

from ...ml.optimization.batch import (
    UnifiedBatchProcessor as BatchProcessor,
    UnifiedBatchConfig as BatchProcessorConfig,
)
from ...performance.monitoring.health.background_manager import (
    EnhancedBackgroundTaskManager,
    init_background_task_manager,
    shutdown_background_task_manager,
)
# Lazy import to avoid circular dependency - imported in functions where needed
from ...utils.session_store import SessionStore

# Module logger
logger = logging.getLogger(__name__)

# Global startup state tracking
_startup_tasks: set[asyncio.Task] = set()
_startup_complete = False
_shutdown_event = asyncio.Event()

async def init_startup_tasks(
    max_concurrent_tasks: int = 10,
    session_ttl: int = 3600,
    cleanup_interval: int = 300,
    batch_config: BatchProcessorConfig | None = None,
    session_store: SessionStore | None = None,
) -> dict[str, any]:
    """Initialize and start all core system components.

    This function orchestrates the startup of:
    1. EnhancedEnhancedBackgroundTaskManager - For managing async background tasks
    2. SessionStore cleanup - For automatic session cleanup
    3. Periodic batch processor - For training data processing
    4. Health monitor - For system health monitoring

    Args:
        max_concurrent_tasks: Maximum concurrent background tasks
        session_ttl: Session time-to-live in seconds
        cleanup_interval: Session cleanup interval in seconds
        batch_config: Optional batch processor configuration (BatchProcessorConfig)
        session_store: Optional pre-configured SessionStore instance (if None, creates new one)

    Returns:
        Dictionary with startup status and component references

    Raises:
        RuntimeError: If startup fails or components cannot be initialized
    """
    global _startup_complete, _startup_tasks

    if _startup_complete:
        logger.warning("Startup tasks already initialized")
        return {"status": "already_initialized", "components": {}}

    logger.info("🚀 Starting APES system components...")
    startup_start_time = time.time()

    components = {}
    startup_errors = []

    try:
        # Step 1: Initialize EnhancedBackgroundTaskManager
        logger.info("📋 Initializing EnhancedBackgroundTaskManager...")
        try:
            background_manager = await init_background_task_manager(
                max_concurrent_tasks=max_concurrent_tasks
            )
            components["background_manager"] = background_manager
            logger.info(
                f"✅ EnhancedBackgroundTaskManager started (max_tasks: {max_concurrent_tasks})"
            )
        except Exception as e:
            startup_errors.append(f"EnhancedBackgroundTaskManager failed: {e}")
            logger.error(f"❌ EnhancedBackgroundTaskManager startup failed: {e}")
            raise

        # Step 2: Initialize SessionStore with cleanup
        logger.info("💾 Initializing SessionStore with automatic cleanup...")
        try:
            if session_store is None:
                # Create new session store with provided configuration
                session_store = SessionStore(
                    maxsize=1000, ttl=session_ttl, cleanup_interval=cleanup_interval
                )
                await session_store.start_cleanup_task()
                logger.info(
                    f"✅ SessionStore created and started (TTL: {session_ttl}s, cleanup: {cleanup_interval}s)"
                )
            else:
                # Use injected session store (already configured and potentially started)
                logger.info(
                    f"✅ SessionStore injected from external source (pre-configured)"
                )
            
            components["session_store"] = session_store
        except Exception as e:
            startup_errors.append(f"SessionStore failed: {e}")
            logger.error(f"❌ SessionStore startup failed: {e}")
            raise

        # Step 3: Initialize Batch Processor
        logger.info("⚙️ Initializing Batch Processor...")
        try:
            if batch_config is None:
                batch_config = BatchProcessorConfig(
                    batch_size=10,
                    batch_timeout=30,
                    max_attempts=3,
                    concurrency=3,
                )

            batch_processor = BatchProcessor(batch_config)
            components["batch_processor"] = batch_processor
            logger.info(
                f"✅ Batch Processor initialized (size: {batch_config.batch_size})"
            )
        except Exception as e:
            startup_errors.append(f"Batch Processor failed: {e}")
            logger.error(f"❌ Batch Processor startup failed: {e}")
            raise

        # Step 4: Start Periodic Batch Processing Task
        logger.info("🔄 Starting periodic batch processing...")
        try:
            task_id = await background_manager.submit_task(
                "periodic_batch_processor",
                periodic_batch_processor_coroutine,
                batch_processor=batch_processor,
            )
            # Get the actual asyncio task from the background manager
            bg_task = background_manager.tasks.get(task_id)
            if bg_task and bg_task.asyncio_task:
                _startup_tasks.add(bg_task.asyncio_task)
            logger.info("✅ Periodic batch processing started")
        except Exception as e:
            startup_errors.append(f"Periodic batch processing failed: {e}")
            logger.error(f"❌ Periodic batch processing startup failed: {e}")
            raise

        # Step 5: Initialize Health Monitor
        logger.info("🏥 Initializing Health Monitor...")
        try:
            from ...performance.monitoring.health.unified_health_system import get_unified_health_monitor
            health_monitor = get_unified_health_monitor()

            # Start health monitoring task
            task_id = await background_manager.submit_task(
                "health_monitor",
                health_monitor_coroutine,
                health_monitor=health_monitor,
            )
            # Get the actual asyncio task from the background manager
            bg_task = background_manager.tasks.get(task_id)
            if bg_task and bg_task.asyncio_task:
                _startup_tasks.add(bg_task.asyncio_task)
            components["health_monitor"] = health_monitor
            logger.info("✅ Health Monitor started")
        except Exception as e:
            startup_errors.append(f"Health Monitor failed: {e}")
            logger.error(f"❌ Health Monitor startup failed: {e}")
            raise

        # Step 6: Verify all components are healthy
        logger.info("🔍 Running initial health check...")
        try:
            health_result = await health_monitor.run_health_check(parallel=True)
            if health_result.overall_status.value in ["healthy", "warning"]:
                logger.info(
                    f"✅ Initial health check passed ({health_result.overall_status.value})"
                )
            else:
                logger.warning(
                    f"⚠️ Initial health check returned: {health_result.overall_status.value}"
                )
        except Exception as e:
            startup_errors.append(f"Health check failed: {e}")
            logger.warning(f"⚠️ Initial health check failed: {e}")
            # Don't fail startup for health check issues

        # Mark startup as complete
        _startup_complete = True
        startup_time = (time.time() - startup_start_time) * 1000

        logger.info(f"🎉 APES startup completed successfully in {startup_time:.2f}ms")

        return {
            "status": "success",
            "startup_time_ms": startup_time,
            "components": {
                "background_manager": type(components["background_manager"]).__name__,
                "session_store": type(components["session_store"]).__name__,
                "batch_processor": type(components["batch_processor"]).__name__,
                "health_monitor": type(components["health_monitor"]).__name__,
            },
            "component_refs": components,
            "active_tasks": len(_startup_tasks),
            "errors": startup_errors,
        }

    except Exception as e:
        logger.error(f"💥 APES startup failed: {e}")

        # Attempt graceful cleanup of partially initialized components
        await cleanup_partial_startup(components)

        return {
            "status": "failed",
            "error": str(e),
            "startup_time_ms": (time.time() - startup_start_time) * 1000,
            "components": {},
            "errors": startup_errors + [str(e)],
        }

async def health_monitor_coroutine(health_monitor) -> None:
    """Continuous health monitoring coroutine.

    Monitors system health and logs warnings for degraded components.
    """
    logger.info("🏥 Health monitor started")

    while not _shutdown_event.is_set():
        try:
            # Run health check every 60 seconds
            await asyncio.sleep(60)

            if _shutdown_event.is_set():
                break

            health_result = await health_monitor.run_health_check(parallel=True)

            # Log health status
            if health_result.overall_status.value == "healthy":
                logger.debug("💚 System health: All components healthy")
            elif health_result.overall_status.value == "warning":
                warning_components = health_result.warning_checks or []
                logger.warning(
                    f"⚠️ System health: Warnings in {len(warning_components)} components"
                )
            else:
                failed_components = health_result.failed_checks or []
                logger.error(
                    f"❌ System health: {len(failed_components)} components failed"
                )

                # Log details about failed components
                for component in failed_components:
                    if component in health_result.checks:
                        error = health_result.checks[component].error
                        logger.error(f"   - {component}: {error}")

        except asyncio.CancelledError:
            logger.info("🏥 Health monitor cancelled")
            break
        except Exception as e:
            logger.error(f"🏥 Health monitor error: {e}")
            await asyncio.sleep(30)  # Retry after error

    logger.info("🏥 Health monitor stopped")

async def cleanup_partial_startup(components: dict) -> None:
    """Clean up partially initialized components during startup failure.

    Args:
        components: Dictionary of partially initialized components
    """
    logger.info("🧹 Cleaning up partially initialized components...")

    # Stop session store cleanup if initialized and we created it
    if "session_store" in components:
        try:
            # Only stop cleanup if this was a locally created session store
            # External session stores (like from APESMCPServer) manage their own lifecycle
            session_store_instance = components["session_store"]
            if hasattr(session_store_instance, '_cleanup_task') and session_store_instance._cleanup_task:
                await session_store_instance.stop_cleanup_task()
                logger.debug("✅ SessionStore cleanup stopped")
            else:
                logger.debug("✅ SessionStore cleanup managed externally")
        except Exception as e:
            logger.error(f"❌ Error stopping SessionStore: {e}")

    # Cancel any started background tasks
    for task in _startup_tasks:
        if not task.done():
            task.cancel()

    if _startup_tasks:
        try:
            await asyncio.gather(*_startup_tasks, return_exceptions=True)
            logger.debug(f"✅ Cancelled {len(_startup_tasks)} background tasks")
        except Exception as e:
            logger.error(f"❌ Error cancelling background tasks: {e}")

    # Shutdown background manager if initialized
    try:
        await shutdown_background_task_manager(timeout=10.0)
        logger.debug("✅ EnhancedBackgroundTaskManager shutdown")
    except Exception as e:
        logger.error(f"❌ Error shutting down EnhancedBackgroundTaskManager: {e}")

async def shutdown_startup_tasks(timeout: float = 30.0) -> dict[str, any]:
    """Gracefully shutdown all startup tasks and components.

    Args:
        timeout: Maximum time to wait for shutdown

    Returns:
        Dictionary with shutdown status and timing
    """
    global _startup_complete, _startup_tasks

    if not _startup_complete:
        logger.warning("Startup tasks not initialized, nothing to shutdown")
        return {"status": "not_initialized"}

    logger.info("🛑 Shutting down APES system components...")
    shutdown_start_time = time.time()

    # Signal shutdown to all monitoring coroutines
    _shutdown_event.set()

    shutdown_errors = []

    try:
        # Cancel all startup tasks
        logger.info(f"🔄 Cancelling {len(_startup_tasks)} background tasks...")
        tasks_to_cancel = []
        for task in _startup_tasks:
            if not task.done():
                task.cancel()
                tasks_to_cancel.append(task)

        # Wait for tasks to complete cancellation (best practice: await after cancel)
        if tasks_to_cancel:
            try:
                # Await all cancelled tasks to ensure they handle CancelledError properly
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=timeout / 2,
                )
                logger.info("✅ Background tasks cancelled")
            except TimeoutError:
                logger.warning("⚠️ Some background tasks did not cancel within timeout")
                shutdown_errors.append("Background task cancellation timeout")

        # Shutdown EnhancedBackgroundTaskManager
        try:
            await shutdown_background_task_manager(timeout=timeout / 2)
            logger.info("✅ EnhancedBackgroundTaskManager shutdown")
        except Exception as e:
            shutdown_errors.append(f"EnhancedBackgroundTaskManager shutdown failed: {e}")
            logger.error(f"❌ EnhancedBackgroundTaskManager shutdown error: {e}")

        shutdown_time = (time.time() - shutdown_start_time) * 1000
        _startup_complete = False
        _startup_tasks.clear()

        logger.info(f"✅ APES shutdown completed in {shutdown_time:.2f}ms")

        return {
            "status": "success",
            "shutdown_time_ms": shutdown_time,
            "errors": shutdown_errors,
        }

    except Exception as e:
        shutdown_time = (time.time() - shutdown_start_time) * 1000
        logger.error(f"💥 APES shutdown failed: {e}")

        return {
            "status": "failed",
            "error": str(e),
            "shutdown_time_ms": shutdown_time,
            "errors": shutdown_errors + [str(e)],
        }

@asynccontextmanager
async def startup_context(
    max_concurrent_tasks: int = 10,
    session_ttl: int = 3600,
    cleanup_interval: int = 300,
    batch_config: BatchProcessorConfig | None = None,
    session_store: SessionStore | None = None,
):
    """Async context manager for APES startup/shutdown lifecycle.

    Usage:
        async with startup_context() as components:
            # Use APES system components
            pass
        # Components automatically shut down on exit
    """
    startup_result = await init_startup_tasks(
        max_concurrent_tasks=max_concurrent_tasks,
        session_ttl=session_ttl,
        cleanup_interval=cleanup_interval,
        batch_config=batch_config,
        session_store=session_store,
    )

    if startup_result["status"] != "success":
        raise RuntimeError(
            f"Startup failed: {startup_result.get('error', 'Unknown error')}"
        )

    try:
        yield startup_result["component_refs"]
    finally:
        await shutdown_startup_tasks()

def is_startup_complete() -> bool:
    """Check if startup tasks have been completed successfully.

    Returns:
        True if startup is complete, False otherwise
    """
    return _startup_complete

def get_startup_task_count() -> int:
    """Get the number of active startup tasks.

    Returns:
        Number of active background tasks
    """
    return len(_startup_tasks)

class StartupOrchestrator:
    """
    Modern startup orchestrator following 2025 best practices.

    Implements:
    - Asynchronous dependency injection pattern
    - Lifecycle management with proper setup/teardown
    - Lazy loading with singleton pattern
    - Service initialization with callbacks
    - Graceful error handling and partial startup recovery
    """

    def __init__(self):
        """Initialize the orchestrator with empty state."""
        self._initialized_components: Dict[str, Any] = {}
        self._startup_tasks: Set[asyncio.Task] = set()
        self._shutdown_callbacks: List[Callable] = []
        self._is_running = False
        self._startup_complete = False

    async def __aenter__(self):
        """Async context manager entry - starts all services."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - graceful shutdown."""
        await self.shutdown()

    async def startup(self,
                     batch_processor: Optional[BatchProcessor] = None,
                     health_service: Optional[Any] = None,
                     session_store: Optional[SessionStore] = None,
                     startup_delay: float = 2.0) -> Dict[str, Any]:
        """
        Initialize all services with proper dependency injection.

        Args:
            batch_processor: Optional batch processor instance
            health_service: Optional health monitoring service
            session_store: Optional session storage service
            startup_delay: Delay before starting services

        Returns:
            Dictionary of initialized components
        """
        if self._is_running:
            logger.warning("Startup already in progress")
            return self._initialized_components

        self._is_running = True

        try:
            # Use existing init_startup_tasks function with session_store injection
            components = await init_startup_tasks(
                session_store=session_store
            )

            self._initialized_components = components
            self._startup_complete = True

            # Register shutdown callbacks
            if batch_processor:
                self._shutdown_callbacks.append(
                    lambda: logger.info("Batch processor shutdown registered")
                )

            return components

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self._cleanup_partial_startup()
            raise

    async def shutdown(self, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Gracefully shutdown all services.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            Shutdown results for each component
        """
        if not self._is_running:
            return {}

        try:
            # Use existing shutdown_startup_tasks function
            results = await shutdown_startup_tasks(timeout=timeout)

            # Run registered callbacks
            for callback in self._shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Shutdown callback error: {e}")

            self._is_running = False
            self._startup_complete = False
            self._initialized_components.clear()

            return results

        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            raise

    async def _cleanup_partial_startup(self):
        """Clean up after partial startup failure."""
        if self._initialized_components:
            await cleanup_partial_startup(self._initialized_components)
        self._initialized_components.clear()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if services are running."""
        return self._is_running and is_startup_complete()

    @property
    def components(self) -> Dict[str, Any]:
        """Get initialized components."""
        return self._initialized_components.copy()

    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a specific initialized component.

        Args:
            name: Component name

        Returns:
            Component instance or None
        """
        return self._initialized_components.get(name)
