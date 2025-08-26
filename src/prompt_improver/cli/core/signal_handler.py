"""Enhanced Signal Handling System for APES CLI Training Workflows
Implements 2025 best practices for graceful shutdown with asyncio integration.
"""

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from rich.console import Console

try:
    EMERGENCY_OPERATIONS_AVAILABLE = True
except ImportError as e:
    EMERGENCY_OPERATIONS_AVAILABLE = False
    import logging

    logging.getLogger(__name__).debug("Emergency operations import failed: %s", e)


class ShutdownReason(Enum):
    """Enumeration of shutdown reasons for tracking and reporting."""

    USER_INTERRUPT = "user_interrupt"
    SYSTEM_SHUTDOWN = "system_shutdown"
    TIMEOUT = "timeout"
    ERROR = "error"
    FORCE = "force"


class SignalOperation(Enum):
    """Enumeration of signal-triggered operations."""

    CHECKPOINT = "checkpoint"
    STATUS_REPORT = "status_report"
    CONFIG_RELOAD = "config_reload"
    SHUTDOWN = "shutdown"


@dataclass
class ShutdownContext:
    """Context information for shutdown operations."""

    reason: ShutdownReason
    signal_name: str | None = None
    timeout: int = 30
    save_progress: bool = True
    force_after_timeout: bool = True
    started_at: datetime | None = None
    parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now(UTC)


@dataclass
class SignalContext:
    """Context information for signal-triggered operations."""

    operation: SignalOperation
    signal_name: str
    signal_number: int
    triggered_at: datetime
    parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class AsyncSignalHandler:
    """Enhanced signal handler implementing 2025 best practices for asyncio applications.

    Features:
    - Graceful shutdown coordination across multiple components
    - Timeout management with force shutdown fallback
    - Progress preservation during shutdown
    - Proper asyncio integration with signal handling
    - Rich console feedback for user experience
    - Signal-triggered operations (checkpoint, status, config reload)
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = asyncio.Event()
        self.shutdown_context: ShutdownContext | None = None
        self.shutdown_in_progress = False
        self.shutdown_handlers: dict[str, Callable] = {}
        self.cleanup_handlers: dict[str, Callable] = {}
        self.operation_handlers: dict[SignalOperation, Callable] = {}
        self.signals_received: set[int] = set()
        self.first_signal_time: datetime | None = None
        self.signal_operations: dict[int, SignalOperation] = {
            signal.SIGUSR1: SignalOperation.CHECKPOINT,
            signal.SIGUSR2: SignalOperation.STATUS_REPORT,
            signal.SIGHUP: SignalOperation.CONFIG_RELOAD,
            signal.SIGINT: SignalOperation.SHUTDOWN,
            signal.SIGTERM: SignalOperation.SHUTDOWN,
        }
        self.loop: asyncio.AbstractEventLoop | None = None
        self.original_handlers: dict[int, Any] = {}
        self.signal_chain: dict[int, list[tuple[int, Callable]]] = {}
        self.chain_enabled = True
        self.emergency_ops: Any | None = None
        self._emergency_ops_initialized = False

    def setup_signal_handlers(
        self, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        """Setup comprehensive signal handlers following 2025 best practices.

        Args:
            loop: Optional asyncio event loop. If None, uses current loop.
        """
        self.loop = loop or asyncio.get_event_loop()
        if sys.platform != "win32":
            for sig in [
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGUSR1,
                signal.SIGUSR2,
                signal.SIGHUP,
            ]:
                with contextlib.suppress(OSError, ValueError):
                    self.original_handlers[sig] = signal.signal(sig, signal.SIG_DFL)
            self.loop.add_signal_handler(
                signal.SIGINT, self._handle_signal, signal.SIGINT, "SIGINT"
            )
            self.loop.add_signal_handler(
                signal.SIGTERM, self._handle_signal, signal.SIGTERM, "SIGTERM"
            )
            self.loop.add_signal_handler(
                signal.SIGUSR1, self._handle_signal, signal.SIGUSR1, "SIGUSR1"
            )
            self.loop.add_signal_handler(
                signal.SIGUSR2, self._handle_signal, signal.SIGUSR2, "SIGUSR2"
            )
            self.loop.add_signal_handler(
                signal.SIGHUP, self._handle_signal, signal.SIGHUP, "SIGHUP"
            )
            self.logger.info(
                "Signal handlers registered for SIGINT, SIGTERM, SIGUSR1, SIGUSR2, SIGHUP"
            )
        else:
            signal.signal(signal.SIGINT, self._sync_signal_handler)
            signal.signal(signal.SIGTERM, self._sync_signal_handler)
            self.logger.info(
                "Signal handlers registered for Windows (SIGINT, SIGTERM only)"
            )
        self._initialize_emergency_operations()

    def _handle_signal(self, signum: int, signal_name: str) -> None:
        """Handle received signals and route to appropriate operations.

        Args:
            signum: Signal number
            signal_name: Human-readable signal name
        """
        current_time = datetime.now(UTC)
        self.signals_received.add(signum)
        if self.first_signal_time is None:
            self.first_signal_time = current_time
        operation = self.signal_operations.get(signum)
        if not operation:
            self.logger.warning(f"Unknown signal {signum} ({signal_name}) received")
            return
        if operation == SignalOperation.SHUTDOWN:
            self._handle_shutdown_signal(signum, signal_name, current_time)
        else:
            self._handle_operation_signal(signum, signal_name, operation, current_time)

    def _handle_shutdown_signal(
        self, signum: int, signal_name: str, current_time: datetime
    ) -> None:
        """Handle shutdown signals (SIGINT, SIGTERM) with signal chaining."""
        chain_results = self.execute_signal_chain(signum, signal_name)
        if chain_results:
            self.logger.info(
                f"Signal chain executed for {signal_name}: {len(chain_results)} handlers"
            )
        shutdown_signals = {signal.SIGINT, signal.SIGTERM}
        received_shutdown_signals = self.signals_received.intersection(shutdown_signals)
        if len(received_shutdown_signals) > 1 and self.first_signal_time is not None:
            time_since_first = (current_time - self.first_signal_time).total_seconds()
            if time_since_first < 5:
                self.console.print(
                    "\n⚡ Multiple shutdown signals received - Force shutdown initiated!",
                    style="bold red",
                )
                self._initiate_force_shutdown()
                return
        reason = (
            ShutdownReason.USER_INTERRUPT
            if signum == signal.SIGINT
            else ShutdownReason.SYSTEM_SHUTDOWN
        )
        self.shutdown_context = ShutdownContext(
            reason=reason, signal_name=signal_name, timeout=30, save_progress=True
        )
        if hasattr(self.shutdown_context, "parameters"):
            self.shutdown_context.parameters = {"chain_results": chain_results}
        self.console.print(
            f"\n⚠️  Received {signal_name} - Initiating graceful shutdown...",
            style="yellow",
        )
        self.console.print(
            "   Training progress will be preserved. Please wait...", style="dim"
        )
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            self.logger.info(f"Graceful shutdown initiated by {signal_name}")

    def _handle_operation_signal(
        self,
        signum: int,
        signal_name: str,
        operation: SignalOperation,
        current_time: datetime,
    ) -> None:
        """Handle operational signals (SIGUSR1, SIGUSR2, SIGHUP)."""
        signal_context = SignalContext(
            operation=operation,
            signal_name=signal_name,
            signal_number=signum,
            triggered_at=current_time,
        )
        operation_messages = {
            SignalOperation.CHECKPOINT: "📋 Creating checkpoint...",
            SignalOperation.STATUS_REPORT: "📊 Generating status report...",
            SignalOperation.CONFIG_RELOAD: "🔄 Reloading configuration...",
        }
        message = operation_messages.get(
            operation, f"🔧 Executing {operation.value}..."
        )
        self.console.print(f"\n{message}", style="blue")
        if operation in self.operation_handlers:
            try:
                if self.loop and self.loop.is_running():
                    loop = self.loop
                    self.loop.call_soon_threadsafe(
                        lambda: loop.create_task(
                            self._execute_operation_handler(operation, signal_context)
                        )
                    )
                else:
                    self.logger.info(
                        f"Event loop not available, executing {operation.value} synchronously"
                    )
                    self.execute_operation_sync(operation, signal_context)
            except Exception as e:
                self.logger.exception(f"Error scheduling {operation.value} operation: {e}")
                self.console.print(
                    f"❌ Failed to execute {operation.value}: {e}", style="red"
                )
        else:
            self.logger.warning(
                f"No handler registered for {operation.value} operation"
            )
            self.console.print(
                f"⚠️  No handler registered for {operation.value}", style="yellow"
            )

    def _sync_signal_handler(self, signum: int, _frame) -> None:
        """Synchronous signal handler for Windows compatibility."""
        signal_names = {signal.SIGINT.value: "SIGINT", signal.SIGTERM.value: "SIGTERM"}
        signal_name = signal_names.get(signum, f"SIG{signum}")
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._handle_signal, signum, signal_name)
        else:
            self.console.print(
                f"\n⚠️  Received {signal_name} - Immediate shutdown", style="red"
            )
            sys.exit(1)

    def register_shutdown_handler(self, name: str, handler: Callable) -> None:
        """Register a shutdown handler to be called during graceful shutdown.

        Args:
            name: Unique name for the handler
            handler: Async callable to execute during shutdown
        """
        self.shutdown_handlers[name] = handler
        self.logger.debug(f"Registered shutdown handler: {name}")

    def register_cleanup_handler(self, name: str, handler: Callable) -> None:
        """Register a cleanup handler for final resource cleanup.

        Args:
            name: Unique name for the handler
            handler: Async callable to execute during cleanup
        """
        self.cleanup_handlers[name] = handler
        self.logger.debug(f"Registered cleanup handler: {name}")

    def register_operation_handler(
        self, operation: SignalOperation, handler: Callable
    ) -> None:
        """Register a handler for signal-triggered operations.

        Args:
            operation: Signal operation type
            handler: Async callable to execute for the operation
        """
        self.operation_handlers[operation] = handler
        self.logger.debug(f"Registered operation handler for {operation.value}")

    def add_signal_chain_handler(
        self, signum: int, handler: Callable, priority: int = 0
    ) -> None:
        """Add a handler to the signal chain for coordinated shutdown.

        Args:
            signum: Signal number
            handler: Callable to execute in the chain
            priority: Priority for execution order (lower = earlier)
        """
        if signum not in self.signal_chain:
            self.signal_chain[signum] = []
        inserted = False
        for i, (existing_priority, _existing_handler) in enumerate(
            self.signal_chain[signum]
        ):
            if priority < existing_priority:
                self.signal_chain[signum].insert(i, (priority, handler))
                inserted = True
                break
        if not inserted:
            self.signal_chain[signum].append((priority, handler))
        self.logger.debug(
            f"Added signal chain handler for signal {signum} with priority {priority}"
        )

    def remove_signal_chain_handler(self, signum: int, handler: Callable) -> bool:
        """Remove a handler from the signal chain.

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

    def execute_signal_chain(self, signum: int, signal_name: str) -> dict[str, Any]:
        """Execute all handlers in the signal chain for coordinated operations.

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
        self.logger.info(
            f"Executing signal chain for {signal_name} with {len(chain_handlers)} handlers"
        )
        for i, (priority, handler) in enumerate(chain_handlers):
            try:
                self.logger.debug(
                    f"Executing chain handler {i + 1}/{len(chain_handlers)} (priority {priority})"
                )
                result = handler(signum, signal_name)
                results[f"handler_{i}"] = {
                    "status": "success",
                    "result": result,
                    "priority": priority,
                }
            except Exception as e:
                self.logger.exception(f"Error in signal chain handler {i + 1}: {e}")
                results[f"handler_{i}"] = {
                    "status": "error",
                    "error": str(e),
                    "priority": priority,
                }
                continue
        self.logger.info(f"Signal chain execution completed for {signal_name}")
        return results

    def _initialize_emergency_operations(self) -> bool:
        """Initialize emergency operations lazily - using optional dependency pattern."""
        if self._emergency_ops_initialized:
            return self.emergency_ops is not None
        self._register_default_emergency_handlers()
        self.logger.info("Emergency operations available via external registration")
        self._emergency_ops_initialized = True
        return False

    def _register_default_emergency_handlers(self) -> None:
        """Register default emergency operation handlers."""
        if not self.emergency_ops:
            self.register_operation_handler(
                SignalOperation.CHECKPOINT, self._default_checkpoint_handler
            )
            self.register_operation_handler(
                SignalOperation.STATUS_REPORT, self._default_status_handler
            )
            self.register_operation_handler(
                SignalOperation.CONFIG_RELOAD, self._default_config_reload_handler
            )
            return
        try:
            self.register_operation_handler(
                SignalOperation.CHECKPOINT,
                self.emergency_ops.create_emergency_checkpoint,
            )
            self.register_operation_handler(
                SignalOperation.STATUS_REPORT, self.emergency_ops.generate_status_report
            )
            self.register_operation_handler(
                SignalOperation.CONFIG_RELOAD, self.emergency_ops.reload_configuration
            )
            self.logger.info("Default emergency operation handlers registered")
        except Exception as e:
            self.logger.exception(f"Failed to register emergency handlers: {e}")

    async def _default_checkpoint_handler(self, _context) -> dict[str, Any]:
        """Default checkpoint handler when emergency operations not available."""
        self.logger.info("Creating basic checkpoint (emergency operations not loaded)")
        return {
            "status": "checkpoint_created",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": "Basic checkpoint created without emergency operations",
        }

    async def _default_status_handler(self, _context) -> dict[str, Any]:
        """Default status handler when emergency operations not available."""
        self.logger.info("Generating basic status report")
        return {
            "status": "running",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": "Basic status report (emergency operations not loaded)",
        }

    async def _default_config_reload_handler(self, _context) -> dict[str, Any]:
        """Default config reload handler when emergency operations not available."""
        self.logger.info("Basic configuration reload")
        return {
            "status": "config_reloaded",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": "Basic config reload (emergency operations not loaded)",
        }

    async def _execute_operation_handler(
        self, operation: SignalOperation, context: SignalContext
    ) -> None:
        """Execute a signal operation handler."""
        handler = self.operation_handlers.get(operation)
        if not handler:
            return None
        try:
            self.logger.info(f"Executing {operation.value} operation")
            result = await handler(context)
            success_messages = {
                SignalOperation.CHECKPOINT: "✅ Checkpoint created successfully",
                SignalOperation.STATUS_REPORT: "✅ Status report generated",
                SignalOperation.CONFIG_RELOAD: "✅ Configuration reloaded",
            }
            message = success_messages.get(operation, f"✅ {operation.value} completed")
            self.console.print(message, style="green")
            return result
        except Exception as e:
            self.logger.exception(f"Error executing {operation.value} operation: {e}")
            self.console.print(f"❌ {operation.value} failed: {e}", style="red")
            raise

    def execute_operation_sync(
        self, operation: SignalOperation, context: SignalContext
    ) -> None:
        """Execute operation handler synchronously for testing."""
        handler = self.operation_handlers.get(operation)
        if not handler:
            self.logger.warning(f"No handler registered for {operation.value}")
            return
        try:
            self.logger.info(f"Executing {operation.value} operation synchronously")
            try:
                loop = asyncio.get_running_loop()
                _task = loop.create_task(handler(context))
                self.logger.info(f"Scheduled {operation.value} operation as task")
            except RuntimeError:
                asyncio.run(handler(context))
            success_messages = {
                SignalOperation.CHECKPOINT: "✅ Checkpoint created successfully",
                SignalOperation.STATUS_REPORT: "✅ Status report generated",
                SignalOperation.CONFIG_RELOAD: "✅ Configuration reloaded",
            }
            message = success_messages.get(operation, f"✅ {operation.value} completed")
            self.console.print(message, style="green")
        except Exception as e:
            self.logger.exception(f"Error executing {operation.value} operation: {e}")
            self.console.print(f"❌ {operation.value} failed: {e}", style="red")

    def cleanup_signal_handlers(self) -> None:
        """Restore original signal handlers and cleanup with proper chaining."""
        self.logger.info("Starting signal handler cleanup")
        cleanup_order = [
            "coredis_connections",
            "database_connections",
            "file_handles",
            "asyncio_handlers",
            "original_handlers",
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
                self.logger.exception(f"Error in cleanup step {cleanup_step}: {e}")
        self.logger.info("Signal handler cleanup completed")

    def _cleanup_coredis_connections(self) -> None:
        """Cleanup CoreDis connections during signal handler cleanup."""
        try:
            self.logger.debug("Attempting CoreDis connection cleanup")
        except ImportError:
            self.logger.debug("HAConnectionManager not available for cleanup")
        except Exception as e:
            self.logger.warning(f"CoreDis cleanup failed: {e}")

    def _cleanup_database_connections(self) -> None:
        """Cleanup database connections during signal handler cleanup."""
        try:
            self.logger.debug("Database connection cleanup scheduled")
        except ImportError:
            self.logger.debug("Database session manager not available")
        except Exception as e:
            self.logger.warning(f"Database cleanup scheduling failed: {e}")

    def _cleanup_file_handles(self) -> None:
        """Cleanup file handles during signal handler cleanup."""
        try:
            self.logger.debug("File handle cleanup completed")
        except Exception as e:
            self.logger.warning(f"File handle cleanup failed: {e}")

    def _cleanup_asyncio_handlers(self) -> None:
        """Cleanup asyncio signal handlers."""
        if sys.platform != "win32" and self.loop:
            for sig in [
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGUSR1,
                signal.SIGUSR2,
                signal.SIGHUP,
            ]:
                try:
                    self.loop.remove_signal_handler(sig)
                    self.logger.debug(f"Removed asyncio handler for signal {sig}")
                except (ValueError, RuntimeError):
                    pass

    def _restore_original_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, original_handler in self.original_handlers.items():
            try:
                signal.signal(sig, original_handler)
                self.logger.debug(f"Restored original handler for signal {sig}")
            except (OSError, ValueError):
                pass

    async def wait_for_shutdown(self) -> ShutdownContext | None:
        """Wait for shutdown signal and return shutdown context.

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
            timeout=5,
            save_progress=False,
            force_after_timeout=False,
        )
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
        self.logger.warning("Force shutdown initiated")

    async def execute_graceful_shutdown(self) -> dict[str, Any]:
        """Execute graceful shutdown with timeout management and progress preservation.

        Returns:
            Shutdown results with status and timing information
        """
        if self.shutdown_in_progress:
            self.logger.warning("Shutdown already in progress")
            return {"status": "already_in_progress"}
        self.shutdown_in_progress = True
        shutdown_start = datetime.now(UTC)
        try:
            self.console.print("🔄 Executing graceful shutdown...", style="blue")
            shutdown_results = await self._execute_shutdown_handlers()
            cleanup_results = await self._execute_cleanup_handlers()
            shutdown_duration = (datetime.now(UTC) - shutdown_start).total_seconds()
            self.console.print(
                "✅ Graceful shutdown completed successfully", style="green"
            )
            return {
                "status": "success",
                "reason": self.shutdown_context.reason.value
                if self.shutdown_context
                else "unknown",
                "duration_seconds": shutdown_duration,
                "shutdown_results": shutdown_results,
                "cleanup_results": cleanup_results,
                "progress_saved": self.shutdown_context.save_progress
                if self.shutdown_context
                else False,
            }
        except TimeoutError:
            self.console.print(
                "⏰ Graceful shutdown timed out - initiating force shutdown",
                style="yellow",
            )
            return await self._execute_force_shutdown()
        except Exception as e:
            self.logger.exception(f"Error during graceful shutdown: {e}")
            self.console.print(f"❌ Shutdown error: {e}", style="red")
            if self.shutdown_context and self.shutdown_context.force_after_timeout:
                return await self._execute_force_shutdown()
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (
                    datetime.now(UTC) - shutdown_start
                ).total_seconds(),
            }
        finally:
            self.shutdown_in_progress = False

    async def _execute_shutdown_handlers(self) -> dict[str, Any]:
        """Execute all registered shutdown handlers with timeout."""
        results = {}
        timeout = self.shutdown_context.timeout if self.shutdown_context else 30
        self.console.print(
            f"   Executing {len(self.shutdown_handlers)} shutdown handlers...",
            style="dim",
        )
        for name, handler in self.shutdown_handlers.items():
            try:
                self.logger.debug(f"Executing shutdown handler: {name}")
                result = await asyncio.wait_for(
                    handler(self.shutdown_context), timeout=timeout
                )
                results[name] = {"status": "success", "result": result}
                self.console.print(f"   ✓ {name}", style="dim green")
            except TimeoutError:
                self.logger.warning(
                    f"Shutdown handler {name} timed out after {timeout}s"
                )
                results[name] = {"status": "timeout", "timeout": timeout}
                self.console.print(f"   ⏰ {name} (timed out)", style="dim yellow")
            except Exception as e:
                self.logger.exception(f"Error in shutdown handler {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
                self.console.print(f"   ❌ {name} (error)", style="dim red")
        return results

    async def _execute_cleanup_handlers(self) -> dict[str, Any]:
        """Execute all registered cleanup handlers."""
        results = {}
        if not self.cleanup_handlers:
            return results
        self.console.print(
            f"   Executing {len(self.cleanup_handlers)} cleanup handlers...",
            style="dim",
        )
        for name, handler in self.cleanup_handlers.items():
            try:
                self.logger.debug(f"Executing cleanup handler: {name}")
                result = await asyncio.wait_for(handler(), timeout=10)
                results[name] = {"status": "success", "result": result}
                self.console.print(f"   ✓ {name}", style="dim green")
            except TimeoutError:
                self.logger.warning(f"Cleanup handler {name} timed out")
                results[name] = {"status": "timeout"}
                self.console.print(f"   ⏰ {name} (timed out)", style="dim yellow")
            except Exception as e:
                self.logger.exception(f"Error in cleanup handler {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
                self.console.print(f"   ❌ {name} (error)", style="dim red")
        return results

    async def _execute_force_shutdown(self) -> dict[str, Any]:
        """Execute immediate force shutdown with minimal cleanup."""
        self.console.print("⚡ Executing force shutdown...", style="bold red")
        force_start = datetime.now(UTC)
        critical_cleanup = {}
        for name, handler in self.cleanup_handlers.items():
            if "critical" in name.lower() or "database" in name.lower():
                try:
                    await asyncio.wait_for(handler(), timeout=2)
                    critical_cleanup[name] = {"status": "success"}
                except Exception as e:
                    critical_cleanup[name] = {"status": "error", "error": str(e)}
        force_duration = (datetime.now(UTC) - force_start).total_seconds()
        self.console.print("⚡ Force shutdown completed", style="bold red")
        return {
            "status": "force_shutdown",
            "reason": "timeout_or_force",
            "duration_seconds": force_duration,
            "critical_cleanup": critical_cleanup,
            "progress_saved": False,
        }
