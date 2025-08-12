"""Service Control Widget - provides service management interface.

Enhanced with 2025 best practices for ML orchestrator integration:
- Async event-driven updates with real-time monitoring
- Comprehensive error handling and recovery patterns
- Performance-optimized service lifecycle management
- Event bus integration for orchestrator coordination
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual import on
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Static

from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)


class ServiceControlWidget(Static):
    """Widget for controlling system services with 2025 best practices.

    features:
    - Real-time service monitoring with async updates
    - Event-driven orchestrator integration
    - Comprehensive error handling and recovery
    - Performance-optimized service lifecycle management
    - Health monitoring with visual indicators
    """

    service_data = reactive({})
    last_update = reactive(datetime.now())
    update_interval = 5.0

    def __init__(self, **kwargs):
        """Initialize ServiceControlWidget with 2025 best practices.

        Args:
            **kwargs: Keyword arguments passed to Static widget parent class.
                     Supports all standard Textual widget parameters.
        """
        try:
            super().__init__(**kwargs)
        except TypeError as e:
            super().__init__()
        self.console = Console()
        self.data_provider: Any | None = None
        self._task_id: str | None = None
        self._is_updating = False
        self._pending_content: Any | None = None

    def compose(self):
        """Compose the widget layout with enhanced controls."""
        with Vertical():
            yield Static(id="service-control-content")
            with Horizontal(id="service-control-buttons"):
                yield Button("Refresh", id="refresh-services", variant="primary")
                yield Button("Start All", id="start-all-services", variant="success")
                yield Button("Stop All", id="stop-all-services", variant="error")
                yield Button(
                    "Auto-Refresh", id="toggle-auto-refresh", variant="default"
                )

    async def on_mount(self) -> None:
        """Initialize the widget when mounted with error handling."""
        try:
            if self._pending_content:
                content_widget = self.query_one("#service-control-content", Static)
                content_widget.update(self._pending_content)
                self._pending_content = None
            else:
                self.update_display()
            if self.data_provider:
                await self._start_auto_refresh()
        except Exception as e:
            self.service_data = {"error": f"Mount error: {e!s}"}
            self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider with enhanced error handling.

        Args:
            data_provider: Data provider instance with service management methods
        """
        self.data_provider = data_provider
        await self._fetch_service_data()

    async def _fetch_service_data(self) -> None:
        """Fetch service data with comprehensive error handling."""
        if not self.data_provider:
            self.service_data = {"error": "No data provider available"}
            self.update_display()
            return
        if self._is_updating:
            return
        self._is_updating = True
        try:
            service_data = await asyncio.wait_for(
                self.data_provider.get_service_status(), timeout=10.0
            )
            self.service_data = service_data
            self.last_update = datetime.now()
            self.update_display()
        except TimeoutError:
            self.service_data = {"error": "Service status request timed out"}
            self.update_display()
        except Exception as e:
            self.service_data = {"error": f"Failed to fetch service data: {e!s}"}
            self.update_display()
        finally:
            self._is_updating = False

    def update_display(self) -> None:
        """Update the display with current service data and enhanced metrics."""
        if not self.service_data:
            return
        content = []
        summary_table = Table(title="Service Status", show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        total_services = self.service_data.get("total_services", 0)
        running_services = self.service_data.get("running_services", 0)
        failed_services = self.service_data.get("failed_services", 0)
        system_load = self.service_data.get("system_load", 0.0)
        auto_restart = self.service_data.get("auto_restart_enabled", False)
        status_text = f"{running_services}/{total_services}"
        if total_services == 0:
            status_color = "dim"
        elif running_services == total_services:
            status_color = "green"
        elif failed_services > 0:
            status_color = "red"
        else:
            status_color = "yellow"
        summary_table.add_row(
            "Services", f"[{status_color}]{status_text}[/{status_color}]"
        )
        summary_table.add_row(
            "Failed", f"[red]{failed_services}[/red]" if failed_services > 0 else "0"
        )
        summary_table.add_row("System Load", f"{system_load:.2f}")
        summary_table.add_row("Auto Restart", "✓" if auto_restart else "✗")
        content.append(summary_table)
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
                status_color = self._get_service_status_color(status)
                status_display = f"[{status_color}]{status.upper()}[/{status_color}]"
                memory_display = (
                    f"{memory:.1f}%"
                    if isinstance(memory, (int, float))
                    else str(memory)
                )
                cpu_display = (
                    f"{cpu:.1f}%" if isinstance(cpu, (int, float)) else str(cpu)
                )
                service_table.add_row(
                    service_name,
                    status_display,
                    str(pid),
                    memory_display,
                    cpu_display,
                    str(uptime),
                )
            content.append(service_table)
        health_indicators = self._create_health_indicators()
        if health_indicators:
            content.append(Panel(health_indicators, title="Health Indicators"))
        if "error" in self.service_data:
            error_panel = Panel(
                f"[red]{self.service_data['error']}[/red]",
                title="Error",
                border_style="red",
            )
            content.append(error_panel)
        if len(content) == 1:
            final_content = content[0]
        else:
            from rich.console import Group

            final_content = Group(*content)
        try:
            content_widget = self.query_one("#service-control-content", Static)
            content_widget.update(final_content)
        except Exception:
            self._pending_content = final_content

    @on(Button.Pressed, "#refresh-services")
    async def refresh_services(self) -> None:
        """Refresh service status with enhanced feedback."""
        if not self.data_provider:
            self.app.notify("No data provider available", severity="error")
            return
        try:
            await self._fetch_service_data()
            self.app.notify("Services refreshed successfully", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to refresh services: {e!s}", severity="error")

    @on(Button.Pressed, "#toggle-auto-refresh")
    async def toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh functionality."""
        if self._task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._task_id)
            self._task_id = None
            self.app.notify("Auto-refresh disabled", severity="information")
            button = self.query_one("#toggle-auto-refresh", Button)
            button.label = "Auto-Refresh"
        else:
            await self._start_auto_refresh()
            self.app.notify("Auto-refresh enabled", severity="information")
            button = self.query_one("#toggle-auto-refresh", Button)
            button.label = "Stop Auto-Refresh"

    @on(Button.Pressed, "#start-all-services")
    async def start_all_services(self) -> None:
        """Start all services with enhanced error handling and progress tracking."""
        if not self.data_provider:
            self.app.notify("No data provider available", severity="error")
            return
        services = self.service_data.get("services", {})
        if not services:
            self.app.notify("No services found to start", severity="warning")
            return
        failed_services = []
        started_services = []
        for service_name, service_info in services.items():
            if service_info.get("status") != "running":
                try:
                    success = await asyncio.wait_for(
                        self.data_provider.start_service(service_name), timeout=30.0
                    )
                    if success:
                        started_services.append(service_name)
                    else:
                        failed_services.append(service_name)
                except TimeoutError:
                    failed_services.append(f"{service_name} (timeout)")
                except Exception as e:
                    failed_services.append(f"{service_name} ({e!s})")
        if started_services:
            self.app.notify(
                f"Started: {', '.join(started_services)}", severity="information"
            )
        if failed_services:
            self.app.notify(
                f"Failed to start: {', '.join(failed_services)}", severity="warning"
            )
        if not started_services and (not failed_services):
            self.app.notify("All services already running", severity="information")
        await self._fetch_service_data()
        await self.update_data(self.data_provider)

    @on(Button.Pressed, "#stop-all-services")
    async def stop_all_services(self) -> None:
        """Stop all services with enhanced error handling and progress tracking."""
        if not self.data_provider:
            self.app.notify("No data provider available", severity="error")
            return
        services = self.service_data.get("services", {})
        if not services:
            self.app.notify("No services found to stop", severity="warning")
            return
        failed_services = []
        stopped_services = []
        for service_name, service_info in services.items():
            if service_info.get("status") == "running":
                try:
                    success = await asyncio.wait_for(
                        self.data_provider.stop_service(service_name), timeout=30.0
                    )
                    if success:
                        stopped_services.append(service_name)
                    else:
                        failed_services.append(service_name)
                except TimeoutError:
                    failed_services.append(f"{service_name} (timeout)")
                except Exception as e:
                    failed_services.append(f"{service_name} ({e!s})")
        if stopped_services:
            self.app.notify(
                f"Stopped: {', '.join(stopped_services)}", severity="information"
            )
        if failed_services:
            self.app.notify(
                f"Failed to stop: {', '.join(failed_services)}", severity="warning"
            )
        if not stopped_services and (not failed_services):
            self.app.notify("No running services to stop", severity="information")
        await self._fetch_service_data()

    async def _start_auto_refresh(self) -> None:
        """Start auto-refresh task with proper error handling."""
        if self._task_id:
            return
        if not self.data_provider:
            return
        task_manager = get_background_task_manager()
        widget_id = f"{self.__class__.__name__}_{id(self)}"
        task_id = await task_manager.submit_enhanced_task(
            task_id=f"tui_refresh_{widget_id}",
            coroutine=self._auto_refresh_loop,
            priority=TaskPriority.LOW,
            tags={
                "service": "tui_widgets",
                "type": "auto_refresh",
                "widget": "service_control",
            },
        )
        self._task_id = task_id

    async def _auto_refresh_loop(self) -> None:
        """Auto-refresh loop with error handling and graceful shutdown."""
        try:
            while True:
                await asyncio.sleep(self.update_interval)
                if not self.data_provider:
                    break
                await self._fetch_service_data()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.service_data = {"error": f"Auto-refresh error: {e!s}"}
            self.update_display()

    async def on_unmount(self) -> None:
        """Clean up resources when widget is unmounted."""
        if self._task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._task_id)
            self._task_id = None

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
            indicators.append(
                f"🟡 {total_services - running_services} service(s) not running"
            )
        system_load = self.service_data.get("system_load", 0.0)
        if system_load < 0.5:
            indicators.append("🟢 System load: Low")
        elif system_load < 0.8:
            indicators.append("🟡 System load: Medium")
        else:
            indicators.append("🔴 System load: High")
        auto_restart = self.service_data.get("auto_restart_enabled", False)
        if auto_restart:
            indicators.append("🟢 Auto-restart: Enabled")
        else:
            indicators.append("🟡 Auto-restart: Disabled")
        return "\n".join(indicators)
