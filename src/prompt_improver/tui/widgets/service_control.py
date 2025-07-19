"""Service Control Widget - provides service management interface."""

from datetime import datetime
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static


class ServiceControlWidget(Static):
    """Widget for controlling system services."""

    service_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.console = Console()
        self.data_provider = None

    def compose(self):
        """Compose the widget layout."""
        with Vertical():
            yield Static(id="service-control-content")
            with Horizontal(id="service-control-buttons"):
                yield Button("Refresh", id="refresh-services", variant="primary")
                yield Button("Start All", id="start-all-services", variant="success")
                yield Button("Stop All", id="stop-all-services", variant="error")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider."""
        self.data_provider = data_provider
        try:
            self.service_data = await data_provider.get_service_status()
            self.update_display()
        except Exception as e:
            self.service_data = {"error": str(e)}
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current service data."""
        if not self.service_data:
            return

        content = []

        # Service summary
        summary_table = Table(title="Service Status", show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        total_services = self.service_data.get("total_services", 0)
        running_services = self.service_data.get("running_services", 0)
        failed_services = self.service_data.get("failed_services", 0)
        system_load = self.service_data.get("system_load", 0.0)
        auto_restart = self.service_data.get("auto_restart_enabled", False)

        # Service status overview
        status_text = f"{running_services}/{total_services}"
        status_color = "green" if running_services == total_services else "yellow"
        if failed_services > 0:
            status_color = "red"

        summary_table.add_row("Services", f"[{status_color}]{status_text}[/{status_color}]")
        summary_table.add_row("Failed", f"[red]{failed_services}[/red]" if failed_services > 0 else "0")
        summary_table.add_row("System Load", f"{system_load:.2f}")
        summary_table.add_row("Auto Restart", "✓" if auto_restart else "✗")

        content.append(summary_table)

        # Individual service status
        services = self.service_data.get("services", {})
        if services:
            service_table = Table(title="Service Details", show_header=True, box=None)
            service_table.add_column("Service", style="cyan")
            service_table.add_column("Status", style="white")
            service_table.add_column("PID", style="white")
            service_table.add_column("Memory", style="white")
            service_table.add_column("CPU", style="white")
            service_table.add_column("Uptime", style="white")

            for service_name, service_info in services.items():
                status = service_info.get("status", "unknown")
                pid = service_info.get("pid", "N/A")
                memory = service_info.get("memory_usage", 0)
                cpu = service_info.get("cpu_usage", 0)
                uptime = service_info.get("uptime", "N/A")

                # Color coding for status
                status_color = self._get_service_status_color(status)
                status_display = f"[{status_color}]{status.upper()}[/{status_color}]"

                # Format resource usage
                memory_display = f"{memory:.1f}%" if isinstance(memory, (int, float)) else str(memory)
                cpu_display = f"{cpu:.1f}%" if isinstance(cpu, (int, float)) else str(cpu)

                service_table.add_row(
                    service_name,
                    status_display,
                    str(pid),
                    memory_display,
                    cpu_display,
                    str(uptime)
                )

            content.append(service_table)

        # Service health indicators
        health_indicators = self._create_health_indicators()
        if health_indicators:
            content.append(Panel(health_indicators, title="Health Indicators"))

        # Error handling
        if "error" in self.service_data:
            error_panel = Panel(
                f"[red]{self.service_data['error']}[/red]",
                title="Error",
                border_style="red"
            )
            content.append(error_panel)

        # Combine all content
        if len(content) == 1:
            final_content = content[0]
        else:
            from rich.console import Group
            final_content = Group(*content)

        # Update the display
        content_widget = self.query_one("#service-control-content", Static)
        content_widget.update(final_content)

    @on(Button.Pressed, "#refresh-services")
    async def refresh_services(self) -> None:
        """Refresh service status."""
        if self.data_provider:
            await self.update_data(self.data_provider)
            self.app.notify("Services refreshed")

    @on(Button.Pressed, "#start-all-services")
    async def start_all_services(self) -> None:
        """Start all services."""
        if not self.data_provider:
            self.app.notify("No data provider available", severity="error")
            return

        services = self.service_data.get("services", {})
        failed_services = []

        for service_name, service_info in services.items():
            if service_info.get("status") != "running":
                try:
                    success = await self.data_provider.start_service(service_name)
                    if not success:
                        failed_services.append(service_name)
                except Exception:
                    failed_services.append(service_name)

        if failed_services:
            self.app.notify(f"Failed to start: {', '.join(failed_services)}", severity="warning")
        else:
            self.app.notify("All services started successfully", severity="information")

        # Refresh display
        await self.update_data(self.data_provider)

    @on(Button.Pressed, "#stop-all-services")
    async def stop_all_services(self) -> None:
        """Stop all services."""
        if not self.data_provider:
            self.app.notify("No data provider available", severity="error")
            return

        services = self.service_data.get("services", {})
        failed_services = []

        for service_name, service_info in services.items():
            if service_info.get("status") == "running":
                try:
                    success = await self.data_provider.stop_service(service_name)
                    if not success:
                        failed_services.append(service_name)
                except Exception:
                    failed_services.append(service_name)

        if failed_services:
            self.app.notify(f"Failed to stop: {', '.join(failed_services)}", severity="warning")
        else:
            self.app.notify("All services stopped successfully", severity="information")

        # Refresh display
        await self.update_data(self.data_provider)

    def _get_service_status_color(self, status: str) -> str:
        """Get color for service status."""
        status_colors = {
            "running": "green",
            "stopped": "red",
            "failed": "red",
            "starting": "yellow",
            "stopping": "yellow",
            "unknown": "gray",
        }
        return status_colors.get(status.lower(), "white")

    def _create_health_indicators(self) -> str:
        """Create health indicators display."""
        indicators = []

        # Overall system health
        total_services = self.service_data.get("total_services", 0)
        running_services = self.service_data.get("running_services", 0)
        failed_services = self.service_data.get("failed_services", 0)

        if total_services == 0:
            indicators.append("⚪ No services configured")
        elif running_services == total_services:
            indicators.append("🟢 All services healthy")
        elif failed_services > 0:
            indicators.append(f"🔴 {failed_services} service(s) failed")
        else:
            indicators.append(f"🟡 {total_services - running_services} service(s) not running")

        # System load indicator
        system_load = self.service_data.get("system_load", 0.0)
        if system_load < 0.5:
            indicators.append("🟢 System load: Low")
        elif system_load < 0.8:
            indicators.append("🟡 System load: Medium")
        else:
            indicators.append("🔴 System load: High")

        # Auto-restart indicator
        auto_restart = self.service_data.get("auto_restart_enabled", False)
        if auto_restart:
            indicators.append("🟢 Auto-restart: Enabled")
        else:
            indicators.append("🟡 Auto-restart: Disabled")

        return "\n".join(indicators)
