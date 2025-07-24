"""
System Overview Widget - displays system health and basic metrics.

Enhanced with 2025 best practices for ML orchestrator integration:
- Async event-driven updates with real-time monitoring
- Comprehensive error handling and recovery patterns
- Performance-optimized health status tracking
- Event bus integration for orchestrator coordination
- Reactive data patterns with proper state management
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from rich.bar import Bar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

class SystemOverviewWidget(Static):
    """
    Widget displaying system overview and health metrics with 2025 best practices.

    features:
    - Real-time system monitoring with async updates
    - Event-driven orchestrator integration
    - Comprehensive error handling and recovery
    - Performance-optimized health status tracking
    - Health monitoring with visual indicators
    - Reactive data patterns for real-time updates
    """

    system_data = reactive({})
    health_status = reactive("unknown")
    last_update = reactive(datetime.now())
    update_interval = 5.0  # seconds

    def __init__(self, **kwargs):
        """
        Initialize SystemOverviewWidget with 2025 best practices.

        Args:
            **kwargs: Keyword arguments passed to Static widget parent class.
                     Supports all standard Textual widget parameters.
        """
        # Following 2025 best practices: Let parent handle all kwargs
        try:
            super().__init__(**kwargs)
        except TypeError as e:
            # Graceful fallback for invalid kwargs
            super().__init__()

        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.data_provider: Optional[Any] = None
        self._update_task: Optional[asyncio.Task] = None
        self._is_updating = False
        self._pending_content: Optional[Any] = None

    def compose(self):
        """Compose the widget layout with enhanced structure."""
        with Vertical():
            yield Static("", id="system-overview-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted with error handling."""
        try:
            # Apply any pending content
            if self._pending_content:
                content_widget = self.query_one("#system-overview-content", Static)
                content_widget.update(self._pending_content)
                self._pending_content = None
            else:
                self.update_display()
        except Exception as e:
            self.logger.error(f"Failed to mount SystemOverviewWidget: {e}")
            # Set error state
            self.system_data = {"error": f"Mount error: {str(e)}"}
            self.health_status = "error"

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider with enhanced error handling."""
        if self._is_updating:
            return  # Prevent concurrent updates

        self._is_updating = True
        self.data_provider = data_provider

        try:
            # Get system overview data
            self.system_data = await data_provider.get_system_overview()
            self.last_update = datetime.now()

            # Update health status based on data
            self._update_health_status()

            # Update display
            self.update_display()

        except Exception as e:
            self.logger.error(f"Failed to update system overview data: {e}")
            self.system_data = {"error": str(e)}
            self.health_status = "error"
            self.update_display()
        finally:
            self._is_updating = False

    def update_display(self) -> None:
        """Update the display with current system data with enhanced error handling."""
        try:
            if not self.system_data:
                # Show loading state
                content = Panel("Loading system overview...", title="System Overview")
                self._safe_update_content(content)
                return

            # Create system overview table with enhanced styling
            table = Table(title="System Overview", show_header=False, box=None)
            table.add_column("Metric", style="cyan", width=15)
            table.add_column("Value", style="white")

            # Add system information with enhanced status display
            status_color = self._get_status_color(self.system_data.get("status", "unknown"))
            status_value = self.system_data.get("status", "unknown").upper()
            table.add_row("Status", f"[{status_color}]{status_value}[/{status_color}]")

            table.add_row("Version", self.system_data.get("version", "unknown"))
            table.add_row("Uptime", self.system_data.get("uptime", "unknown"))

            # Add service information with enhanced display
            active_services = self.system_data.get("active_services", 0)
            total_services = self.system_data.get("total_services", 0)
            service_status = f"{active_services}/{total_services}"
            service_color = "green" if active_services == total_services else "yellow"
            table.add_row("Services", f"[{service_color}]{service_status}[/{service_color}]")

            # Add resource usage with enhanced formatting
            memory_usage = self.system_data.get("memory_usage", 0)
            cpu_usage = self.system_data.get("cpu_usage", 0)
            disk_usage = self.system_data.get("disk_usage", 0)

            table.add_row("Memory", self._format_usage(memory_usage))
            table.add_row("CPU", self._format_usage(cpu_usage))
            table.add_row("Disk", self._format_usage(disk_usage))

            # Add last restart info
            last_restart = self.system_data.get("last_restart")
            if last_restart:
                if isinstance(last_restart, datetime):
                    restart_str = last_restart.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    restart_str = str(last_restart)
                table.add_row("Last Restart", restart_str)

            # Add last update timestamp
            update_str = self.last_update.strftime("%H:%M:%S")
            table.add_row("Last Update", f"[dim]{update_str}[/dim]")

            # Show error if present
            if "error" in self.system_data:
                table.add_row("Error", f"[red]{self.system_data['error']}[/red]")

            # Create panel with health status indicator
            health_indicator = self._get_health_indicator()
            content = Panel(table, title=f"System Overview {health_indicator}")

            # Update the display safely
            self._safe_update_content(content)

        except Exception as e:
            self.logger.error(f"Failed to update system overview display: {e}")
            error_content = Panel(f"[red]Display Error: {str(e)}[/red]", title="System Overview")
            self._safe_update_content(error_content)

    def _safe_update_content(self, content: Any) -> None:
        """Safely update widget content with error handling."""
        try:
            content_widget = self.query_one("#system-overview-content", Static)
            content_widget.update(content)
        except Exception as e:
            # If widget not mounted yet, store content for later
            self._pending_content = content
            self.logger.debug(f"Widget not mounted yet, storing content: {e}")

    def _update_health_status(self) -> None:
        """Update health status based on system data."""
        if "error" in self.system_data:
            self.health_status = "error"
        elif self.system_data.get("status") == "online":
            # Check service health
            active = self.system_data.get("active_services", 0)
            total = self.system_data.get("total_services", 1)
            if active == total:
                self.health_status = "healthy"
            else:
                self.health_status = "warning"
        else:
            self.health_status = "unknown"

    def _get_health_indicator(self) -> str:
        """Get health status indicator for display."""
        indicators = {
            "healthy": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "unknown": "⚪"
        }
        return indicators.get(self.health_status, "⚪")

    def _get_status_color(self, status: str) -> str:
        """Get color for system status."""
        status_colors = {
            "online": "green",
            "warning": "yellow",
            "error": "red",
            "offline": "red",
            "initializing": "blue",
        }
        return status_colors.get(status.lower(), "white")

    def _format_usage(self, usage: float) -> str:
        """Format resource usage percentage with enhanced styling."""
        if usage == 0:
            return "[dim]N/A[/dim]"

        percentage = f"{usage:.1f}%"
        if usage < 50:
            return f"[green]{percentage}[/green]"
        if usage < 80:
            return f"[yellow]{percentage}[/yellow]"
        return f"[red]{percentage}[/red]"

    async def start_auto_refresh(self, interval: float = None) -> None:
        """Start auto-refresh with configurable interval."""
        if interval:
            self.update_interval = interval

        if self._update_task and not self._update_task.done():
            return  # Already running

        self._update_task = asyncio.create_task(self._auto_refresh_loop())

    async def stop_auto_refresh(self) -> None:
        """Stop auto-refresh gracefully."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

    async def _auto_refresh_loop(self) -> None:
        """Auto-refresh loop with error handling and graceful shutdown."""
        try:
            while True:
                await asyncio.sleep(self.update_interval)
                if not self.data_provider:
                    break
                await self.update_data(self.data_provider)
        except asyncio.CancelledError:
            # Graceful shutdown
            pass
        except Exception as e:
            # Log error but don't crash the widget
            self.logger.error(f"Auto-refresh error: {e}")
            self.system_data = {"error": f"Auto-refresh error: {str(e)}"}
            self.health_status = "error"
            self.update_display()

    async def on_unmount(self) -> None:
        """Clean up resources when widget is unmounted."""
        await self.stop_auto_refresh()
