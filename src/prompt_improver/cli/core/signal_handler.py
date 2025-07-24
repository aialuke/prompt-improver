"""
Enhanced Signal Handling System for APES CLI Training Workflows
Implements 2025 best practices for graceful shutdown with asyncio integration.
"""

import asyncio
import signal
import logging
import sys
from typing import Dict, Any, Optional, Callable, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from rich.console import Console


class ShutdownReason(Enum):
    """Enumeration of shutdown reasons for tracking and reporting."""
    USER_INTERRUPT = "user_interrupt"  # Ctrl+C (SIGINT)
    SYSTEM_SHUTDOWN = "system_shutdown"  # SIGTERM
    TIMEOUT = "timeout"  # Shutdown timeout exceeded
    ERROR = "error"  # Error during shutdown
    FORCE = "force"  # Force shutdown requested


@dataclass
class ShutdownContext:
    """Context information for shutdown operations."""
    reason: ShutdownReason
    signal_name: Optional[str] = None
    timeout: int = 30
    save_progress: bool = True
    force_after_timeout: bool = True
    started_at: datetime = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)


class AsyncSignalHandler:
    """
    Enhanced signal handler implementing 2025 best practices for asyncio applications.

    Features:
    - Graceful shutdown coordination across multiple components
    - Timeout management with force shutdown fallback
    - Progress preservation during shutdown
    - Proper asyncio integration with signal handling
    - Rich console feedback for user experience
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)

        # Shutdown coordination
        self.shutdown_event = asyncio.Event()
        self.shutdown_context: Optional[ShutdownContext] = None
        self.shutdown_in_progress = False

        # Registered shutdown handlers
        self.shutdown_handlers: Dict[str, Callable] = {}
        self.cleanup_handlers: Dict[str, Callable] = {}

        # Signal tracking
        self.signals_received: Set[int] = set()
        self.first_signal_time: Optional[datetime] = None

        # Asyncio loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def setup_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Setup signal handlers for graceful shutdown following 2025 best practices.

        Args:
            loop: Optional asyncio event loop. If None, uses current loop.
        """
        self.loop = loop or asyncio.get_event_loop()

        # Register signal handlers for Unix systems
        if sys.platform != 'win32':
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

            self.logger.info("Signal handlers registered for SIGINT and SIGTERM")
        else:
            # Windows signal handling
            signal.signal(signal.SIGINT, self._sync_signal_handler)
            signal.signal(signal.SIGTERM, self._sync_signal_handler)
            self.logger.info("Signal handlers registered for Windows")

    def _handle_signal(self, signum: int, signal_name: str) -> None:
        """
        Handle received signals and initiate graceful shutdown.

        Args:
            signum: Signal number
            signal_name: Human-readable signal name
        """
        current_time = datetime.now(timezone.utc)

        # Track signal reception
        self.signals_received.add(signum)
        if self.first_signal_time is None:
            self.first_signal_time = current_time

        # Handle repeated signals (force shutdown)
        if len(self.signals_received) > 1 or signum in self.signals_received:
            time_since_first = (current_time - self.first_signal_time).total_seconds()
            if time_since_first < 5:  # Multiple signals within 5 seconds = force shutdown
                self.console.print(
                    f"\n⚡ Multiple {signal_name} signals received - Force shutdown initiated!",
                    style="bold red"
                )
                self._initiate_force_shutdown()
                return

        # Determine shutdown reason
        reason = ShutdownReason.USER_INTERRUPT if signum == signal.SIGINT else ShutdownReason.SYSTEM_SHUTDOWN

        # Create shutdown context
        self.shutdown_context = ShutdownContext(
            reason=reason,
            signal_name=signal_name,
            timeout=30,  # Default timeout
            save_progress=True
        )

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

    def _sync_signal_handler(self, signum: int, frame) -> None:
        """Synchronous signal handler for Windows compatibility."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"

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

    async def wait_for_shutdown(self) -> ShutdownContext:
        """
        Wait for shutdown signal and return shutdown context.

        Returns:
            ShutdownContext with shutdown details
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
                "reason": self.shutdown_context.reason.value,
                "duration_seconds": shutdown_duration,
                "shutdown_results": shutdown_results,
                "cleanup_results": cleanup_results,
                "progress_saved": self.shutdown_context.save_progress
            }

        except asyncio.TimeoutError:
            self.console.print("⏰ Graceful shutdown timed out - initiating force shutdown", style="yellow")
            return await self._execute_force_shutdown()

        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            self.console.print(f"❌ Shutdown error: {e}", style="red")

            if self.shutdown_context.force_after_timeout:
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
        timeout = self.shutdown_context.timeout

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
