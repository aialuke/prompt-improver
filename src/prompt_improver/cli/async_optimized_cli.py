"""Async-optimized CLI operations to eliminate blocking calls.

This module provides async versions of CLI commands that eliminate blocking operations
such as subprocess.run(), time.sleep(), and synchronous file I/O.
"""

import asyncio
import aiofiles
import os
import sys
import time
from pathlib import Path
import logging
from typing import Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class AsyncCLIOptimizer:
    """Async-optimized CLI operations with performance monitoring"""

    def __init__(self):
        self.console = console
        self.performance_metrics = {
            "start_times": [],
            "stop_times": [],
            "total_operations": 0
        }

    async def start_async(
        self,
        mcp_port: int = 3000,
        background: bool = False,
        verbose: bool = False,
    ) -> Optional[int]:
        """Async version of start command with non-blocking operations"""
        start_time = time.time()

        self.console.print("🚀 Starting APES MCP server...", style="green")

        try:
            if background:
                result = await self._start_background_async(mcp_port, verbose)
            else:
                result = await self._start_foreground_async(mcp_port)

            # Record performance metrics
            duration = time.time() - start_time
            self.performance_metrics["start_times"].append(duration)
            self.performance_metrics["total_operations"] += 1

            if duration > 1.0:  # Alert on slow operations
                logger.warning(f"Async start operation took {duration:.3f}s")
            else:
                logger.debug(f"Async start completed in {duration:.3f}s")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Async start failed after {duration:.3f}s: {e}")
            raise

    async def _start_background_async(self, mcp_port: int, verbose: bool) -> int:
        """Start server in background using async subprocess"""
        try:
            mcp_server_path = Path(__file__).parent.parent / "mcp_server" / "mcp_server.py"

            # Async file existence check - non-blocking
            if not await asyncio.to_thread(mcp_server_path.exists):
                raise FileNotFoundError(f"MCP server script not found: {mcp_server_path}")

            # Async subprocess creation - non-blocking, no 300s timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(mcp_server_path),
                stdout=asyncio.subprocess.PIPE if not verbose else None,
                stderr=asyncio.subprocess.PIPE if not verbose else None,
            )

            # Async PID file operations
            pid_file = Path.home() / ".local" / "share" / "apes" / "mcp.pid"
            await asyncio.to_thread(pid_file.parent.mkdir, parents=True, exist_ok=True)

            # Non-blocking file write using aiofiles
            async with aiofiles.open(pid_file, 'w') as f:
                await f.write(str(process.pid))

            self.console.print(f"✅ MCP server started in background (PID: {process.pid})", style="green")
            self.console.print(f"📍 PID file: {pid_file}", style="dim")

            return process.pid

        except Exception as e:
            self.console.print(f"❌ Failed to start MCP server: {e}", style="red")
            raise

    async def _start_foreground_async(self, mcp_port: int) -> int:
        """Start server in foreground using async subprocess"""
        try:
            mcp_server_path = Path(__file__).parent.parent / "mcp_server" / "mcp_server.py"

            # Async file existence check
            if not await asyncio.to_thread(mcp_server_path.exists):
                raise FileNotFoundError(f"MCP server script not found: {mcp_server_path}")

            # Create async subprocess - eliminates 300s blocking timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(mcp_server_path),
                stdout=None,  # Inherit stdout for foreground
                stderr=None,  # Inherit stderr for foreground
            )

            # Wait for process completion asynchronously
            returncode = await process.wait()

            if returncode == 0:
                self.console.print("✅ MCP server completed successfully", style="green")
            else:
                self.console.print(f"❌ MCP server exited with code {returncode}", style="red")

            return returncode

        except KeyboardInterrupt:
            self.console.print("\\n👋 MCP server stopped", style="yellow")
            return 0
        except Exception as e:
            self.console.print(f"❌ Error running MCP server: {e}", style="red")
            raise

    async def stop_async(self, graceful: bool = True, timeout: int = 5) -> bool:
        """Async version of stop command with non-blocking operations"""
        start_time = time.time()

        self.console.print("🔄 Stopping APES MCP server...", style="yellow")

        pid_file = Path.home() / ".local" / "share" / "apes" / "mcp.pid"

        try:
            # Async file existence check
            if not await asyncio.to_thread(pid_file.exists):
                self.console.print("⚠️  No running MCP server found", style="yellow")
                return True

            # Async file reading using aiofiles
            async with aiofiles.open(pid_file, 'r') as f:
                pid_text = await f.read()
            pid = int(pid_text.strip())

            if graceful:
                # Async graceful shutdown
                success = await self._graceful_shutdown_async(pid, timeout)
            else:
                # Async force kill
                await asyncio.to_thread(os.kill, pid, 9)
                success = True

            if success:
                # Async file removal
                await asyncio.to_thread(pid_file.unlink)
                self.console.print("✅ MCP server stopped", style="green")

            # Record performance metrics
            duration = time.time() - start_time
            self.performance_metrics["stop_times"].append(duration)
            self.performance_metrics["total_operations"] += 1

            logger.debug(f"Async stop completed in {duration:.3f}s")
            return success

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Async stop failed after {duration:.3f}s: {e}")
            self.console.print(f"❌ Failed to stop MCP server: {e}", style="red")
            return False

    async def _graceful_shutdown_async(self, pid: int, timeout: int) -> bool:
        """Async graceful shutdown with non-blocking waits"""
        try:
            # Import here to avoid circular imports
            from prompt_improver.utils import ensure_running

            # Send SIGTERM - non-blocking
            await asyncio.to_thread(os.kill, pid, 15)
            self.console.print(f"📤 Sent SIGTERM to process {pid}", style="dim")

            # Async wait with timeout - replaces time.sleep(1) blocking call
            for i in range(timeout):
                try:
                    is_running = await asyncio.to_thread(ensure_running, pid)
                    if not is_running:
                        self.console.print("✅ Process terminated gracefully", style="green")
                        return True

                    # Non-blocking async sleep instead of time.sleep(1)
                    await asyncio.sleep(1)

                except ProcessLookupError:
                    self.console.print("✅ Process terminated", style="green")
                    return True
            else:
                # Force kill if still running after timeout
                try:
                    await asyncio.to_thread(os.kill, pid, 9)
                    self.console.print(f"⚡ Force killed process {pid}", style="yellow")
                    return True
                except ProcessLookupError:
                    self.console.print("✅ Process already terminated", style="green")
                    return True

        except ProcessLookupError:
            self.console.print("⚡ Process already terminated", style="dim")
            return True
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
            return False

    def get_performance_metrics(self) -> dict:
        """Get performance metrics for the async CLI operations"""
        start_times = self.performance_metrics["start_times"]
        stop_times = self.performance_metrics["stop_times"]

        return {
            "total_operations": self.performance_metrics["total_operations"],
            "start_operations": len(start_times),
            "stop_operations": len(stop_times),
            "avg_start_time": sum(start_times) / len(start_times) if start_times else 0,
            "avg_stop_time": sum(stop_times) / len(stop_times) if stop_times else 0,
            "max_start_time": max(start_times) if start_times else 0,
            "max_stop_time": max(stop_times) if stop_times else 0,
            "performance_target_met": all(t < 1.0 for t in start_times + stop_times)
        }


# Global instance for CLI commands
_cli_optimizer = AsyncCLIOptimizer()


def start_async_optimized(
    mcp_port: int = typer.Option(3000, "--mcp-port", "-p", help="MCP server port"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start APES MCP server with async optimizations (eliminates 300s timeout)."""
    try:
        result = asyncio.run(_cli_optimizer.start_async(mcp_port, background, verbose))
        if result is not None and (not background or result > 0):
            return 0  # Success
        else:
            return 1  # Failure
    except KeyboardInterrupt:
        console.print("\\n👋 Operation cancelled", style="yellow")
        return 0
    except Exception as e:
        console.print(f"❌ Async start operation failed: {e}", style="red")
        return 1


def stop_async_optimized(
    graceful: bool = typer.Option(True, "--graceful/--force", help="Graceful shutdown"),
    timeout: int = typer.Option(5, "--timeout", "-t", help="Shutdown timeout in seconds"),
):
    """Stop APES MCP server with async optimizations (eliminates blocking sleep)."""
    try:
        success = asyncio.run(_cli_optimizer.stop_async(graceful, timeout))
        return 0 if success else 1
    except KeyboardInterrupt:
        console.print("\\n👋 Operation cancelled", style="yellow")
        return 0
    except Exception as e:
        console.print(f"❌ Async stop operation failed: {e}", style="red")
        return 1


def get_async_cli_metrics():
    """Get performance metrics for async CLI operations."""
    return _cli_optimizer.get_performance_metrics()