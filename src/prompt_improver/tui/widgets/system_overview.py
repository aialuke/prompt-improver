"""System Overview Widget - displays system health and basic metrics."""

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
    """Widget displaying system overview and health metrics."""

    system_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.console = Console()

    def compose(self):
        """Compose the widget layout."""
        yield Static(id="system-overview-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider."""
        try:
            self.system_data = await data_provider.get_system_overview()
            self.update_display()
        except Exception as e:
            self.system_data = {"error": str(e)}
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current system data."""
        if not self.system_data:
            return

        # Create system overview table
        table = Table(title="System Overview", show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # Add system information
        status_color = self._get_status_color(self.system_data.get("status", "unknown"))
        table.add_row("Status", f"[{status_color}]{self.system_data.get('status', 'unknown').upper()}[/{status_color}]")
        table.add_row("Version", self.system_data.get("version", "unknown"))
        table.add_row("Uptime", self.system_data.get("uptime", "unknown"))

        # Add service information
        active_services = self.system_data.get("active_services", 0)
        total_services = self.system_data.get("total_services", 0)
        service_status = f"{active_services}/{total_services}"
        service_color = "green" if active_services == total_services else "yellow"
        table.add_row("Services", f"[{service_color}]{service_status}[/{service_color}]")

        # Add resource usage
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

        # Show error if present
        if "error" in self.system_data:
            table.add_row("Error", f"[red]{self.system_data['error']}[/red]")

        # Update the display
        content_widget = self.query_one("#system-overview-content", Static)
        content_widget.update(table)

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
        """Format resource usage percentage."""
        if usage == 0:
            return "N/A"

        percentage = f"{usage:.1f}%"
        if usage < 50:
            return f"[green]{percentage}[/green]"
        if usage < 80:
            return f"[yellow]{percentage}[/yellow]"
        return f"[red]{percentage}[/red]"
