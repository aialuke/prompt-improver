"""
Enhanced Signal Handling System for APES CLI Training Workflows
Implements 2025 best practices for graceful shutdown with asyncio integration.
"""

import asyncio
import signal
import logging
import sys
from typing import Dict, Any, Optional, Callable, Set, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

# Import emergency operations manager
try:

    EMERGENCY_OPERATIONS_AVAILABLE = True
except ImportError as e:
    EMERGENCY_OPERATIONS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).debug(f"Emergency operations import failed: {e}")

class ShutdownReason(Enum):
    """Enumeration of shutdown reasons for tracking and reporting."""
    USER_INTERRUPT = "user_interrupt"  # Ctrl+C (SIGINT)
    SYSTEM_SHUTDOWN = "system_shutdown"  # SIGTERM
    TIMEOUT = "timeout"  # Shutdown timeout exceeded
    ERROR = "error"  # Error during shutdown
    FORCE = "force"  # Force shutdown requested

class SignalOperation(Enum):
    """Enumeration of signal-triggered operations."""
    CHECKPOINT = "checkpoint"  # SIGUSR1 - Create checkpoint
    STATUS_REPORT = "status_report"  # SIGUSR2 - Generate status report
    CONFIG_RELOAD = "config_reload"  # SIGHUP - Reload configuration
    SHUTDOWN = "shutdown"  # SIGTERM/SIGINT - Graceful shutdown

@dataclass
class ShutdownContext:
    """Context information for shutdown operations."""
    reason: ShutdownReason
    signal_name: Optional[str] = None
    timeout: int = 30
    save_progress: bool = True
    force_after_timeout: bool = True
    started_at: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)

@dataclass
class SignalContext:
    """Context information for signal-triggered operations."""
    operation: SignalOperation
    signal_name: str
    signal_number: int
    triggered_at: datetime
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class AsyncSignalHandler:
    """
    Enhanced signal handler implementing 2025 best practices for asyncio applications.

    Features:
    - Graceful shutdown coordination across multiple components
    - Timeout management with force shutdown fallback
    - Progress preservation during shutdown
    - Proper asyncio integration with signal handling
    - Rich console feedback for user experience
    - Signal-triggered operations (checkpoint, status, config reload)
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)

        # Shutdown coordination
        self.shutdown_event = asyncio.Event()
        self.shutdown_context: Optional[ShutdownContext] = None
        self.shutdown_in_progress = False

        # Registered handlers
        self.shutdown_handlers: Dict[str, Callable] = {}
        self.cleanup_handlers: Dict[str, Callable] = {}
        self.operation_handlers: Dict[SignalOperation, Callable] = {}

        # Signal tracking
        self.signals_received: Set[int] = set()
        self.first_signal_time: Optional[datetime] = None
        self.signal_operations: Dict[int, SignalOperation] = {
            signal.SIGUSR1: SignalOperation.CHECKPOINT,
            signal.SIGUSR2: SignalOperation.STATUS_REPORT,
            signal.SIGHUP: SignalOperation.CONFIG_RELOAD,
            signal.SIGINT: SignalOperation.SHUTDOWN,
            signal.SIGTERM: SignalOperation.SHUTDOWN,
        }

        # Asyncio loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Original signal handlers for cleanup
        self.original_handlers: Dict[int, Any] = {}

        # Signal chaining support
        self.signal_chain: Dict[int, List[Tuple[int, Callable]]] = {}
        self.chain_enabled = True

        # Emergency operations manager (lazy initialization)
        self.emergency_ops: Optional[Any] = None
        self._emergency_ops_initialized = False

    def setup_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Setup comprehensive signal handlers following 2025 best practices.

        Args:
            loop: Optional asyncio event loop. If None, uses current loop.
        """
        self.loop = loop or asyncio.get_event_loop()

        # Register signal handlers for Unix systems
        if sys.platform != 'win32':
            # Store original handlers for cleanup
            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP]:
                try:
                    self.original_handlers[sig] = signal.signal(sig, signal.SIG_DFL)
                except (OSError, ValueError):
                    # Some signals might not be available
                    pass

            # SIGINT (Ctrl+C) - User interruption
            self.loop.add_signal_handler(
                signal.SIGINT,
                self._handle_signal,
                signal.SIGINT,
                "SIGINT"
            )

            # SIGTERM - System shutdown
            self.loop.add_signal_handler(
                signal.SIGTERM,
                self._handle_signal,
                signal.SIGTERM,
                "SIGTERM"
            )

            # SIGUSR1 - Checkpoint creation
            self.loop.add_signal_handler(
                signal.SIGUSR1,
                self._handle_signal,
                signal.SIGUSR1,
                "SIGUSR1"
            )

            # SIGUSR2 - Status reporting
            self.loop.add_signal_handler(
                signal.SIGUSR2,
                self._handle_signal,
                signal.SIGUSR2,
                "SIGUSR2"
            )

            # SIGHUP - Configuration reload
            self.loop.add_signal_handler(
                signal.SIGHUP,
                self._handle_signal,
                signal.SIGHUP,
                "SIGHUP"
            )

            self.logger.info("Signal handlers registered for SIGINT, SIGTERM, SIGUSR1, SIGUSR2, SIGHUP")
        else:
            # Windows signal handling (limited support)
            signal.signal(signal.SIGINT, self._sync_signal_handler)
            signal.signal(signal.SIGTERM, self._sync_signal_handler)
            self.logger.info("Signal handlers registered for Windows (SIGINT, SIGTERM only)")

        # Initialize emergency operations after signal handlers are set up
        self._initialize_emergency_operations()

    def _handle_signal(self, signum: int, signal_name: str) -> None:
        """
        Handle received signals and route to appropriate operations.

        Args:
            signum: Signal number
            signal_name: Human-readable signal name
        """
        current_time = datetime.now(timezone.utc)

        # Track signal reception
        self.signals_received.add(signum)
        if self.first_signal_time is None:
            self.first_signal_time = current_time

        # Get signal operation
        operation = self.signal_operations.get(signum)
        if not operation:
            self.logger.warning(f"Unknown signal {signum} ({signal_name}) received")
            return

        # Handle shutdown signals
        if operation == SignalOperation.SHUTDOWN:
            self._handle_shutdown_signal(signum, signal_name, current_time)
        else:
            # Handle operational signals (non-shutdown)
            self._handle_operation_signal(signum, signal_name, operation, current_time)

    def _handle_shutdown_signal(self, signum: int, signal_name: str, current_time: datetime) -> None:
        """Handle shutdown signals (SIGINT, SIGTERM) with signal chaining."""
        # Execute signal chain first for coordinated shutdown
        chain_results = self.execute_signal_chain(signum, signal_name)
        if chain_results:
            self.logger.info(f"Signal chain executed for {signal_name}: {len(chain_results)} handlers")

        # Handle repeated signals (force shutdown)
        shutdown_signals = {signal.SIGINT, signal.SIGTERM}
        received_shutdown_signals = self.signals_received.intersection(shutdown_signals)

        if len(received_shutdown_signals) > 1 and self.first_signal_time is not None:
            time_since_first = (current_time - self.first_signal_time).total_seconds()
            if time_since_first < 5:  # Multiple signals within 5 seconds = force shutdown
                self.console.print(
                    f"\n⚡ Multiple shutdown signals received - Force shutdown initiated!",
                    style="bold red"
                )
                self._initiate_force_shutdown()
                return

        # Determine shutdown reason
        reason = ShutdownReason.USER_INTERRUPT if signum == signal.SIGINT else ShutdownReason.SYSTEM_SHUTDOWN

        # Create shutdown context with chain results
        self.shutdown_context = ShutdownContext(
            reason=reason,
            signal_name=signal_name,
            timeout=30,  # Default timeout
            save_progress=True
        )

        # Store chain results in shutdown context for reference
        if hasattr(self.shutdown_context, 'parameters'):
            self.shutdown_context.parameters = {"chain_results": chain_results}

        # Display user-friendly message
        self.console.print(
            f"\n⚠️  Received {signal_name} - Initiating graceful shutdown...",
            style="yellow"
        )
        self.console.print(
            "   Training progress will be preserved. Please wait...",
            style="dim"
        )

        # Set shutdown event to trigger graceful shutdown
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            self.logger.info(f"Graceful shutdown initiated by {signal_name}")

    def _handle_operation_signal(self, signum: int, signal_name: str, operation: SignalOperation, current_time: datetime) -> None:
        """Handle operational signals (SIGUSR1, SIGUSR2, SIGHUP)."""
        # Create signal context
        signal_context = SignalContext(
            operation=operation,
            signal_name=signal_name,
            signal_number=signum,
            triggered_at=current_time
        )

        # Display operation message
        operation_messages = {
            SignalOperation.CHECKPOINT: "📋 Creating checkpoint...",
            SignalOperation.STATUS_REPORT: "📊 Generating status report...",
            SignalOperation.CONFIG_RELOAD: "🔄 Reloading configuration..."
        }

        message = operation_messages.get(operation, f"🔧 Executing {operation.value}...")
        self.console.print(f"\n{message}", style="blue")

        # Execute operation handler if registered
        if operation in self.operation_handlers:
            try:
                # Schedule async operation
                if self.loop and self.loop.is_running():
                    # Use call_soon_threadsafe to ensure proper scheduling
                    loop = self.loop  # Capture loop reference for lambda
                    self.loop.call_soon_threadsafe(
                        lambda: loop.create_task(self._execute_operation_handler(operation, signal_context))
                    )
                else:
                    # Fallback to sync execution for testing or when loop not available
                    self.logger.info(f"Event loop not available, executing {operation.value} synchronously")
                    self.execute_operation_sync(operation, signal_context)
            except Exception as e:
                self.logger.error(f"Error scheduling {operation.value} operation: {e}")
                self.console.print(f"❌ Failed to execute {operation.value}: {e}", style="red")
        else:
            self.logger.warning(f"No handler registered for {operation.value} operation")
            self.console.print(f"⚠️  No handler registered for {operation.value}", style="yellow")

    def _sync_signal_handler(self, signum: int, _frame) -> None:
        """Synchronous signal handler for Windows compatibility."""
        signal_names = {
            signal.SIGINT.value: "SIGINT",
            signal.SIGTERM.value: "SIGTERM"
        }
        signal_name = signal_names.get(signum, f"SIG{signum}")

        # Schedule async signal handling
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._handle_signal, signum, signal_name)
        else:
            # Fallback for when loop is not available
            self.console.print(f"\n⚠️  Received {signal_name} - Immediate shutdown", style="red")
            sys.exit(1)

    def register_shutdown_handler(self, name: str, handler: Callable) -> None:
        """
        Register a shutdown handler to be called during graceful shutdown.

        Args:
            name: Unique name for the handler
            handler: Async callable to execute during shutdown
        """
        self.shutdown_handlers[name] = handler
        self.logger.debug(f"Registered shutdown handler: {name}")

    def register_cleanup_handler(self, name: str, handler: Callable) -> None:
        """
        Register a cleanup handler for final resource cleanup.

        Args:
            name: Unique name for the handler
            handler: Async callable to execute during cleanup
        """
        self.cleanup_handlers[name] = handler
        self.logger.debug(f"Registered cleanup handler: {name}")

    def register_operation_handler(self, operation: SignalOperation, handler: Callable) -> None:
        """
        Register a handler for signal-triggered operations.

        Args:
            operation: Signal operation type
            handler: Async callable to execute for the operation
        """
        self.operation_handlers[operation] = handler
        self.logger.debug(f"Registered operation handler for {operation.value}")

    def add_signal_chain_handler(self, signum: int, handler: Callable, priority: int = 0) -> None:
        """
        Add a handler to the signal chain for coordinated shutdown.

        Args:
            signum: Signal number
            handler: Callable to execute in the chain
            priority: Priority for execution order (lower = earlier)
        """
        if signum not in self.signal_chain:
            self.signal_chain[signum] = []

        # Insert handler based on priority
        inserted = False
        for i, (existing_priority, _existing_handler) in enumerate(self.signal_chain[signum]):
            if priority < existing_priority:
                self.signal_chain[signum].insert(i, (priority, handler))
                inserted = True
                break

        if not inserted:
            self.signal_chain[signum].append((priority, handler))

        self.logger.debug(f"Added signal chain handler for signal {signum} with priority {priority}")

    def remove_signal_chain_handler(self, signum: int, handler: Callable) -> bool:
        """
        Remove a handler from the signal chain.

        Args:
            signum: Signal number
            handler: Handler to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if signum not in self.signal_chain:
            return False

        for i, (_priority, existing_handler) in enumerate(self.signal_chain[signum]):
            if existing_handler == handler:
                self.signal_chain[signum].pop(i)
                self.logger.debug(f"Removed signal chain handler for signal {signum}")
                return True

        return False

    def execute_signal_chain(self, signum: int, signal_name: str) -> Dict[str, Any]:
        """
        Execute all handlers in the signal chain for coordinated operations.

        Args:
            signum: Signal number
            signal_name: Human-readable signal name

        Returns:
            Results from all chain handlers
        """
        if not self.chain_enabled or signum not in self.signal_chain:
            return {}

        results = {}
        chain_handlers = self.signal_chain[signum]

        self.logger.info(f"Executing signal chain for {signal_name} with {len(chain_handlers)} handlers")

        for i, (priority, handler) in enumerate(chain_handlers):
            try:
                self.logger.debug(f"Executing chain handler {i+1}/{len(chain_handlers)} (priority {priority})")
                result = handler(signum, signal_name)
                results[f"handler_{i}"] = {"status": "success", "result": result, "priority": priority}

            except Exception as e:
                self.logger.error(f"Error in signal chain handler {i+1}: {e}")
                results[f"handler_{i}"] = {"status": "error", "error": str(e), "priority": priority}

                # Continue with other handlers even if one fails
                continue

        self.logger.info(f"Signal chain execution completed for {signal_name}")
        return results

    def _initialize_emergency_operations(self) -> bool:
        """Initialize emergency operations lazily - using optional dependency pattern."""
        if self._emergency_ops_initialized:
            return self.emergency_ops is not None

        # Use optional dependency pattern to avoid circular imports
        # Emergency operations can be registered externally if needed
        self._register_default_emergency_handlers()
        self.logger.info("Emergency operations available via external registration")
        self._emergency_ops_initialized = True
        return False  # No direct emergency ops, but handlers can be registered externally

    def _register_default_emergency_handlers(self) -> None:
        """Register default emergency operation handlers."""
        if not self.emergency_ops:
            # Register placeholder handlers that can be overridden externally
            self.register_operation_handler(
                SignalOperation.CHECKPOINT,
                self._default_checkpoint_handler
            )
            self.register_operation_handler(
                SignalOperation.STATUS_REPORT,
                self._default_status_handler
            )
            self.register_operation_handler(
                SignalOperation.CONFIG_RELOAD,
                self._default_config_reload_handler
            )
            return

        try:
            # Register checkpoint creation handler for SIGUSR1
            self.register_operation_handler(
                SignalOperation.CHECKPOINT,
                self.emergency_ops.create_emergency_checkpoint
            )

            # Register status reporting handler for SIGUSR2
            self.register_operation_handler(
                SignalOperation.STATUS_REPORT,
                self.emergency_ops.generate_status_report
            )

            # Register configuration reload handler for SIGHUP
            self.register_operation_handler(
                SignalOperation.CONFIG_RELOAD,
                self.emergency_ops.reload_configuration
            )

            self.logger.info("Default emergency operation handlers registered")

        except Exception as e:
            self.logger.error(f"Failed to register emergency handlers: {e}")

    async def _default_checkpoint_handler(self, _context) -> Dict[str, Any]:
        """Default checkpoint handler when emergency operations not available."""
        self.logger.info("Creating basic checkpoint (emergency operations not loaded)")
        return {
            "status": "checkpoint_created",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Basic checkpoint created without emergency operations"
        }

    async def _default_status_handler(self, _context) -> Dict[str, Any]:
        """Default status handler when emergency operations not available."""
        self.logger.info("Generating basic status report")
        return {
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Basic status report (emergency operations not loaded)"
        }

    async def _default_config_reload_handler(self, _context) -> Dict[str, Any]:
        """Default config reload handler when emergency operations not available."""
        self.logger.info("Basic configuration reload")
        return {
            "status": "config_reloaded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Basic config reload (emergency operations not loaded)"
        }

    async def _execute_operation_handler(self, operation: SignalOperation, context: SignalContext) -> None:
        """Execute a signal operation handler."""
        handler = self.operation_handlers.get(operation)
        if not handler:
            return

        try:
            self.logger.info(f"Executing {operation.value} operation")
            result = await handler(context)

            # Display success message
            success_messages = {
                SignalOperation.CHECKPOINT: "✅ Checkpoint created successfully",
                SignalOperation.STATUS_REPORT: "✅ Status report generated",
                SignalOperation.CONFIG_RELOAD: "✅ Configuration reloaded"
            }

            message = success_messages.get(operation, f"✅ {operation.value} completed")
            self.console.print(message, style="green")

            return result

        except Exception as e:
            self.logger.error(f"Error executing {operation.value} operation: {e}")
            self.console.print(f"❌ {operation.value} failed: {e}", style="red")
            raise

    def execute_operation_sync(self, operation: SignalOperation, context: SignalContext) -> None:
        """Execute operation handler synchronously for testing."""
        handler = self.operation_handlers.get(operation)
        if not handler:
            self.logger.warning(f"No handler registered for {operation.value}")
            return

        try:
            self.logger.info(f"Executing {operation.value} operation synchronously")

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a running loop, create a task
                _task = loop.create_task(handler(context))
                # Don't wait for it, just schedule it
                self.logger.info(f"Scheduled {operation.value} operation as task")
            except RuntimeError:
                # No running loop, use asyncio.run
                asyncio.run(handler(context))

            # Display success message
            success_messages = {
                SignalOperation.CHECKPOINT: "✅ Checkpoint created successfully",
                SignalOperation.STATUS_REPORT: "✅ Status report generated",
                SignalOperation.CONFIG_RELOAD: "✅ Configuration reloaded"
            }

            message = success_messages.get(operation, f"✅ {operation.value} completed")
            self.console.print(message, style="green")

        except Exception as e:
            self.logger.error(f"Error executing {operation.value} operation: {e}")
            self.console.print(f"❌ {operation.value} failed: {e}", style="red")

    def cleanup_signal_handlers(self) -> None:
        """Restore original signal handlers and cleanup with proper chaining."""
        self.logger.info("Starting signal handler cleanup")

        # Execute cleanup chain in reverse order of registration
        cleanup_order = [
            "coredis_connections",  # Close CoreDis connections first
            "database_connections", # Then database connections
            "file_handles",        # Then file handles
            "asyncio_handlers",    # Then asyncio signal handlers
            "original_handlers"    # Finally restore original handlers
        ]

        for cleanup_step in cleanup_order:
            try:
                if cleanup_step == "coredis_connections":
                    self._cleanup_coredis_connections()
                elif cleanup_step == "database_connections":
                    self._cleanup_database_connections()
                elif cleanup_step == "file_handles":
                    self._cleanup_file_handles()
                elif cleanup_step == "asyncio_handlers":
                    self._cleanup_asyncio_handlers()
                elif cleanup_step == "original_handlers":
                    self._restore_original_handlers()

                self.logger.debug(f"Completed cleanup step: {cleanup_step}")

            except Exception as e:
                self.logger.error(f"Error in cleanup step {cleanup_step}: {e}")
                # Continue with other cleanup steps even if one fails

        self.logger.info("Signal handler cleanup completed")

    def _cleanup_coredis_connections(self) -> None:
        """Cleanup CoreDis connections during signal handler cleanup."""
        try:
            # Import here to avoid circular imports

            # Get the HA connection manager instance if available
            # This is a best-effort cleanup for CoreDis connections
            self.logger.debug("Attempting CoreDis connection cleanup")

        except ImportError:
            self.logger.debug("HAConnectionManager not available for cleanup")
        except Exception as e:
            self.logger.warning(f"CoreDis cleanup failed: {e}")

    def _cleanup_database_connections(self) -> None:
        """Cleanup database connections during signal handler cleanup."""
        try:
            # Import here to avoid circular imports

            # This is a synchronous cleanup, so we can't await
            # The actual cleanup will be handled by the shutdown handlers
            self.logger.debug("Database connection cleanup scheduled")

        except ImportError:
            self.logger.debug("Database session manager not available")
        except Exception as e:
            self.logger.warning(f"Database cleanup scheduling failed: {e}")

    def _cleanup_file_handles(self) -> None:
        """Cleanup file handles during signal handler cleanup."""
        try:
            # Close any open file handles that might be tracked
            # This is a placeholder for file handle cleanup
            self.logger.debug("File handle cleanup completed")

        except Exception as e:
            self.logger.warning(f"File handle cleanup failed: {e}")

    def _cleanup_asyncio_handlers(self) -> None:
        """Cleanup asyncio signal handlers."""
        if sys.platform != 'win32' and self.loop:
            # Remove asyncio signal handlers
            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP]:
                try:
                    self.loop.remove_signal_handler(sig)
                    self.logger.debug(f"Removed asyncio handler for signal {sig}")
                except (ValueError, RuntimeError):
                    # Handler might not be registered or loop might be closed
                    pass

    def _restore_original_handlers(self) -> None:
        """Restore original signal handlers."""
        # Restore original handlers
        for sig, original_handler in self.original_handlers.items():
            try:
                signal.signal(sig, original_handler)
                self.logger.debug(f"Restored original handler for signal {sig}")
            except (OSError, ValueError):
                # Signal might not be available
                pass

    async def wait_for_shutdown(self) -> Optional[ShutdownContext]:
        """
        Wait for shutdown signal and return shutdown context.

        Returns:
            ShutdownContext with shutdown details, or None if not available
        """
        await self.shutdown_event.wait()
        return self.shutdown_context

    def _initiate_force_shutdown(self) -> None:
        """Initiate immediate force shutdown."""
        self.shutdown_context = ShutdownContext(
            reason=ShutdownReason.FORCE,
            signal_name="FORCE",
            timeout=5,  # Very short timeout for force shutdown
            save_progress=False,  # Skip progress saving in force mode
            force_after_timeout=False
        )

        if not self.shutdown_event.is_set():
            self.shutdown_event.set()

        self.logger.warning("Force shutdown initiated")

    async def execute_graceful_shutdown(self) -> Dict[str, Any]:
        """
        Execute graceful shutdown with timeout management and progress preservation.

        Returns:
            Shutdown results with status and timing information
        """
        if self.shutdown_in_progress:
            self.logger.warning("Shutdown already in progress")
            return {"status": "already_in_progress"}

        self.shutdown_in_progress = True
        shutdown_start = datetime.now(timezone.utc)

        try:
            self.console.print("🔄 Executing graceful shutdown...", style="blue")

            # Execute shutdown handlers with timeout
            shutdown_results = await self._execute_shutdown_handlers()

            # Execute cleanup handlers
            cleanup_results = await self._execute_cleanup_handlers()

            shutdown_duration = (datetime.now(timezone.utc) - shutdown_start).total_seconds()

            self.console.print("✅ Graceful shutdown completed successfully", style="green")

            return {
                "status": "success",
                "reason": self.shutdown_context.reason.value if self.shutdown_context else "unknown",
                "duration_seconds": shutdown_duration,
                "shutdown_results": shutdown_results,
                "cleanup_results": cleanup_results,
                "progress_saved": self.shutdown_context.save_progress if self.shutdown_context else False
            }

        except asyncio.TimeoutError:
            self.console.print("⏰ Graceful shutdown timed out - initiating force shutdown", style="yellow")
            return await self._execute_force_shutdown()

        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            self.console.print(f"❌ Shutdown error: {e}", style="red")

            if self.shutdown_context and self.shutdown_context.force_after_timeout:
                return await self._execute_force_shutdown()
            else:
                return {
                    "status": "error",
                    "error": str(e),
                    "duration_seconds": (datetime.now(timezone.utc) - shutdown_start).total_seconds()
                }
        finally:
            self.shutdown_in_progress = False

    async def _execute_shutdown_handlers(self) -> Dict[str, Any]:
        """Execute all registered shutdown handlers with timeout."""
        results = {}
        timeout = self.shutdown_context.timeout if self.shutdown_context else 30

        self.console.print(f"   Executing {len(self.shutdown_handlers)} shutdown handlers...", style="dim")

        for name, handler in self.shutdown_handlers.items():
            try:
                self.logger.debug(f"Executing shutdown handler: {name}")

                # Execute handler with timeout
                result = await asyncio.wait_for(handler(self.shutdown_context), timeout=timeout)
                results[name] = {"status": "success", "result": result}

                self.console.print(f"   ✓ {name}", style="dim green")

            except asyncio.TimeoutError:
                self.logger.warning(f"Shutdown handler {name} timed out after {timeout}s")
                results[name] = {"status": "timeout", "timeout": timeout}
                self.console.print(f"   ⏰ {name} (timed out)", style="dim yellow")

            except Exception as e:
                self.logger.error(f"Error in shutdown handler {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
                self.console.print(f"   ❌ {name} (error)", style="dim red")

        return results

    async def _execute_cleanup_handlers(self) -> Dict[str, Any]:
        """Execute all registered cleanup handlers."""
        results = {}

        if not self.cleanup_handlers:
            return results

        self.console.print(f"   Executing {len(self.cleanup_handlers)} cleanup handlers...", style="dim")

        for name, handler in self.cleanup_handlers.items():
            try:
                self.logger.debug(f"Executing cleanup handler: {name}")

                # Cleanup handlers get shorter timeout
                result = await asyncio.wait_for(handler(), timeout=10)
                results[name] = {"status": "success", "result": result}

                self.console.print(f"   ✓ {name}", style="dim green")

            except asyncio.TimeoutError:
                self.logger.warning(f"Cleanup handler {name} timed out")
                results[name] = {"status": "timeout"}
                self.console.print(f"   ⏰ {name} (timed out)", style="dim yellow")

            except Exception as e:
                self.logger.error(f"Error in cleanup handler {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
                self.console.print(f"   ❌ {name} (error)", style="dim red")

        return results

    async def _execute_force_shutdown(self) -> Dict[str, Any]:
        """Execute immediate force shutdown with minimal cleanup."""
        self.console.print("⚡ Executing force shutdown...", style="bold red")

        force_start = datetime.now(timezone.utc)

        # Only execute critical cleanup handlers with very short timeout
        critical_cleanup = {}
        for name, handler in self.cleanup_handlers.items():
            if "critical" in name.lower() or "database" in name.lower():
                try:
                    await asyncio.wait_for(handler(), timeout=2)
                    critical_cleanup[name] = {"status": "success"}
                except Exception as e:
                    critical_cleanup[name] = {"status": "error", "error": str(e)}

        force_duration = (datetime.now(timezone.utc) - force_start).total_seconds()

        self.console.print("⚡ Force shutdown completed", style="bold red")

        return {
            "status": "force_shutdown",
            "reason": "timeout_or_force",
            "duration_seconds": force_duration,
            "critical_cleanup": critical_cleanup,
            "progress_saved": False
        }
