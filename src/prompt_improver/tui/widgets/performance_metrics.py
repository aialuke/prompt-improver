"""Performance Metrics Widget - displays real-time system performance data.

Enhanced with 2025 best practices:
- SLI/SLO integration and monitoring
- Error budget tracking and burn rate analysis
- Adaptive thresholds and anomaly detection
- Multi-dimensional metrics visualization
- Real-time alerting and health status
"""

import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

try:
    from prompt_improver.performance.monitoring.health.unified_health_system import (
        UnifiedHealthMonitor,
        get_unified_health_monitor,
    )

    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False


class PerformanceMetricsWidget(Static):
    """Enhanced Performance Metrics Widget with 2025 best practices.

    features:
    - SLI/SLO monitoring and error budget tracking
    - Burn rate analysis and adaptive thresholds
    - Multi-dimensional metrics visualization
    - Real-time alerting and health status
    - Orchestrator integration support
    """

    performance_data = reactive({})
    slo_data = reactive({})
    health_status = reactive("unknown")

    def __init__(
        self,
        enable_slo_monitoring: bool = True,
        enable_adaptive_thresholds: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.enable_slo_monitoring = (
            enable_slo_monitoring and HEALTH_MONITORING_AVAILABLE
        )
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.adaptive_thresholds = {
            "response_time": {"warning": 150.0, "critical": 300.0},
            "error_rate": {"warning": 0.01, "critical": 0.05},
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "queue_length": {"warning": 20, "critical": 50},
            "cache_hit_rate": {"warning": 0.8, "critical": 0.6},
        }
        self.performance_monitor = None
        if self.enable_slo_monitoring and HEALTH_MONITORING_AVAILABLE:
            try:
                self.performance_monitor = get_unified_health_monitor()
            except Exception as e:
                self.logger.warning(f"Failed to initialize health monitoring: {e}")
                self.enable_slo_monitoring = False

    def compose(self):
        """Compose the widget layout with enhanced structure."""
        with Vertical():
            yield Static("", id="performance-metrics-content")
            if self.enable_slo_monitoring:
                yield Static("", id="slo-metrics-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()
        if self.enable_slo_monitoring:
            self.update_slo_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider with enhanced error handling."""
        try:
            self.performance_data = await data_provider.get_performance_metrics()
            if self.enable_slo_monitoring and self.performance_monitor:
                await self._update_slo_data()
            if self.enable_adaptive_thresholds:
                self._update_adaptive_thresholds()
            self._update_health_status()
            self.update_display()
            if self.enable_slo_monitoring:
                self.update_slo_display()
        except Exception as e:
            self.logger.error(f"Failed to update performance data: {e}")
            self.performance_data = {"error": str(e)}
            self.health_status = "error"
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current performance data."""
        if not self.performance_data:
            return
        content = []
        metrics_table = Table(title="Performance Metrics", show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        response_time = self.performance_data.get("response_time", 0)
        response_color = self._get_response_time_color(response_time)
        metrics_table.add_row(
            "Response Time",
            f"[{response_color}]{response_time:.1f}ms[/{response_color}]",
        )
        throughput = self.performance_data.get("throughput", 0)
        metrics_table.add_row("Throughput", f"{throughput:.1f} req/s")
        error_rate = self.performance_data.get("error_rate", 0.0)
        error_color = self._get_error_rate_color(error_rate)
        metrics_table.add_row(
            "Error Rate", f"[{error_color}]{error_rate:.2%}[/{error_color}]"
        )
        active_connections = self.performance_data.get("active_connections", 0)
        metrics_table.add_row("Active Connections", str(active_connections))
        queue_length = self.performance_data.get("queue_length", 0)
        queue_color = self._get_queue_color(queue_length)
        metrics_table.add_row(
            "Queue Length", f"[{queue_color}]{queue_length}[/{queue_color}]"
        )
        memory_usage = self.performance_data.get("memory_usage", 0)
        cpu_usage = self.performance_data.get("cpu_usage", 0)
        metrics_table.add_row("Memory Usage", self._format_usage(memory_usage))
        metrics_table.add_row("CPU Usage", self._format_usage(cpu_usage))
        cache_hit_rate = self.performance_data.get("cache_hit_rate", 0.0)
        cache_color = self._get_cache_color(cache_hit_rate)
        metrics_table.add_row(
            "Cache Hit Rate", f"[{cache_color}]{cache_hit_rate:.1%}[/{cache_color}]"
        )
        content.append(metrics_table)
        recent_response_times = self.performance_data.get("recent_response_times", [])
        if recent_response_times:
            response_chart = self._create_response_time_chart(recent_response_times)
            content.append(Panel(response_chart, title="Response Time Trend"))
        recent_throughput = self.performance_data.get("recent_throughput", [])
        if recent_throughput:
            throughput_chart = self._create_throughput_chart(recent_throughput)
            content.append(Panel(throughput_chart, title="Throughput Trend"))
        alerts = self._generate_performance_alerts()
        if alerts:
            alert_panel = Panel(
                "\n".join(alerts), title="Performance Alerts", border_style="yellow"
            )
            content.append(alert_panel)
        if "error" in self.performance_data:
            error_panel = Panel(
                f"[red]{self.performance_data['error']}[/red]",
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
            content_widget = self.query_one("#performance-metrics-content", Static)
            content_widget.update(final_content)
        except Exception:
            self.update(final_content)

    def _get_response_time_color(self, response_time: float) -> str:
        """Get color for response time based on adaptive thresholds."""
        warning_threshold = self.adaptive_thresholds["response_time"]["warning"]
        critical_threshold = self.adaptive_thresholds["response_time"]["critical"]
        if response_time < warning_threshold:
            return "green"
        if response_time < critical_threshold:
            return "yellow"
        return "red"

    def _get_error_rate_color(self, error_rate: float) -> str:
        """Get color for error rate based on adaptive thresholds."""
        warning_threshold = self.adaptive_thresholds["error_rate"]["warning"]
        critical_threshold = self.adaptive_thresholds["error_rate"]["critical"]
        if error_rate < warning_threshold:
            return "green"
        if error_rate < critical_threshold:
            return "yellow"
        return "red"

    def _get_queue_color(self, queue_length: int) -> str:
        """Get color for queue length."""
        if queue_length < 10:
            return "green"
        if queue_length < 50:
            return "yellow"
        return "red"

    def _get_cache_color(self, cache_hit_rate: float) -> str:
        """Get color for cache hit rate."""
        if cache_hit_rate > 0.9:
            return "green"
        if cache_hit_rate > 0.7:
            return "yellow"
        return "red"

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

    def _create_response_time_chart(self, response_times: list[float]) -> str:
        """Create a simple text-based response time chart."""
        if not response_times:
            return "No data"
        min_time = min(response_times)
        max_time = max(response_times)
        if max_time == min_time:
            return "─" * len(response_times)
        lines = []
        for time in response_times:
            normalized = (time - min_time) / (max_time - min_time)
            height = int(normalized * 8)
            line_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            lines.append(line_chars[min(height, 7)])
        chart_text = "".join(lines)
        range_info = f"Range: {min_time:.1f}ms - {max_time:.1f}ms"
        return f"{chart_text}\n{range_info}"

    def _create_throughput_chart(self, throughput_data: list[float]) -> str:
        """Create a simple text-based throughput chart."""
        if not throughput_data:
            return "No data"
        min_throughput = min(throughput_data)
        max_throughput = max(throughput_data)
        if max_throughput == min_throughput:
            return "─" * len(throughput_data)
        lines = []
        for throughput in throughput_data:
            normalized = (throughput - min_throughput) / (
                max_throughput - min_throughput
            )
            height = int(normalized * 8)
            line_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            lines.append(line_chars[min(height, 7)])
        chart_text = "".join(lines)
        range_info = f"Range: {min_throughput:.1f} - {max_throughput:.1f} req/s"
        return f"{chart_text}\n{range_info}"

    def _generate_performance_alerts(self) -> list[str]:
        """Generate performance alerts based on current metrics."""
        alerts = []
        response_time = self.performance_data.get("response_time", 0)
        if response_time > 1000:
            alerts.append(f"[red]⚠ High response time: {response_time:.1f}ms[/red]")
        error_rate = self.performance_data.get("error_rate", 0.0)
        if error_rate > 0.05:
            alerts.append(f"[red]⚠ High error rate: {error_rate:.2%}[/red]")
        queue_length = self.performance_data.get("queue_length", 0)
        if queue_length > 50:
            alerts.append(f"[yellow]⚠ High queue length: {queue_length}[/yellow]")
        memory_usage = self.performance_data.get("memory_usage", 0)
        if memory_usage > 90:
            alerts.append(f"[red]⚠ High memory usage: {memory_usage:.1f}%[/red]")
        cpu_usage = self.performance_data.get("cpu_usage", 0)
        if cpu_usage > 90:
            alerts.append(f"[red]⚠ High CPU usage: {cpu_usage:.1f}%[/red]")
        cache_hit_rate = self.performance_data.get("cache_hit_rate", 0.0)
        if cache_hit_rate < 0.7:
            alerts.append(
                f"[yellow]⚠ Low cache hit rate: {cache_hit_rate:.1%}[/yellow]"
            )
        return alerts

    async def _update_slo_data(self) -> None:
        """Update SLO data from performance monitor."""
        if not self.performance_monitor:
            return
        try:
            response_time = self.performance_data.get("response_time", 0)
            error_rate = self.performance_data.get("error_rate", 0.0)
            await self.performance_monitor.record_performance_measurement(
                operation_name="widget_metrics",
                response_time_ms=response_time,
                is_error=error_rate > 0.01,
                business_value=1.0,
            )
            self.slo_data = self.performance_monitor.get_slo_dashboard()
        except Exception as e:
            self.logger.warning(f"Failed to update SLO data: {e}")

    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on historical data."""
        if not self.enable_adaptive_thresholds:
            return
        try:
            response_time = self.performance_data.get("response_time", 0)
            error_rate = self.performance_data.get("error_rate", 0.0)
            if response_time > 0:
                current_warning = self.adaptive_thresholds["response_time"]["warning"]
                current_critical = self.adaptive_thresholds["response_time"]["critical"]
                adjustment_factor = 0.05
                if response_time > current_critical:
                    self.adaptive_thresholds["response_time"]["warning"] *= (
                        1 + adjustment_factor
                    )
                    self.adaptive_thresholds["response_time"]["critical"] *= (
                        1 + adjustment_factor
                    )
                elif response_time < current_warning * 0.8:
                    self.adaptive_thresholds["response_time"]["warning"] *= (
                        1 - adjustment_factor
                    )
                    self.adaptive_thresholds["response_time"]["critical"] *= (
                        1 - adjustment_factor
                    )
            if error_rate > 0:
                current_warning = self.adaptive_thresholds["error_rate"]["warning"]
                current_critical = self.adaptive_thresholds["error_rate"]["critical"]
                adjustment_factor = 0.1
                if error_rate > current_critical:
                    self.adaptive_thresholds["error_rate"]["warning"] *= (
                        1 + adjustment_factor
                    )
                    self.adaptive_thresholds["error_rate"]["critical"] *= (
                        1 + adjustment_factor
                    )
                elif error_rate < current_warning * 0.5:
                    self.adaptive_thresholds["error_rate"]["warning"] *= (
                        1 - adjustment_factor
                    )
                    self.adaptive_thresholds["error_rate"]["critical"] *= (
                        1 - adjustment_factor
                    )
        except Exception as e:
            self.logger.warning(f"Failed to update adaptive thresholds: {e}")

    def _update_health_status(self) -> None:
        """Update overall health status based on current metrics."""
        try:
            if "error" in self.performance_data:
                self.health_status = "error"
                return
            response_time = self.performance_data.get("response_time", 0)
            error_rate = self.performance_data.get("error_rate", 0.0)
            cpu_usage = self.performance_data.get("cpu_usage", 0)
            memory_usage = self.performance_data.get("memory_usage", 0)
            critical_issues = []
            warning_issues = []
            if response_time > self.adaptive_thresholds["response_time"]["critical"]:
                critical_issues.append("response_time")
            elif response_time > self.adaptive_thresholds["response_time"]["warning"]:
                warning_issues.append("response_time")
            if error_rate > self.adaptive_thresholds["error_rate"]["critical"]:
                critical_issues.append("error_rate")
            elif error_rate > self.adaptive_thresholds["error_rate"]["warning"]:
                warning_issues.append("error_rate")
            if cpu_usage > self.adaptive_thresholds["cpu_usage"]["critical"]:
                critical_issues.append("cpu_usage")
            elif cpu_usage > self.adaptive_thresholds["cpu_usage"]["warning"]:
                warning_issues.append("cpu_usage")
            if memory_usage > self.adaptive_thresholds["memory_usage"]["critical"]:
                critical_issues.append("memory_usage")
            elif memory_usage > self.adaptive_thresholds["memory_usage"]["warning"]:
                warning_issues.append("memory_usage")
            if critical_issues:
                self.health_status = "critical"
            elif warning_issues:
                self.health_status = "warning"
            else:
                self.health_status = "healthy"
        except Exception as e:
            self.logger.warning(f"Failed to update health status: {e}")
            self.health_status = "unknown"

    def update_slo_display(self) -> None:
        """Update SLO metrics display."""
        if not self.enable_slo_monitoring or not self.slo_data:
            return
        try:
            content = []
            health_table = Table(title="SLO Health Status", show_header=False, box=None)
            health_table.add_column("Metric", style="cyan")
            health_table.add_column("Status", style="white")
            overall_health = self.slo_data.get("overall_health", "unknown")
            health_color = self._get_health_color(overall_health)
            health_table.add_row(
                "Overall Health",
                f"[{health_color}]{overall_health.upper()}[/{health_color}]",
            )
            active_violations = self.slo_data.get("active_violations", 0)
            violation_color = "red" if active_violations > 0 else "green"
            health_table.add_row(
                "Active Violations",
                f"[{violation_color}]{active_violations}[/{violation_color}]",
            )
            content.append(health_table)
            error_budgets = self.slo_data.get("error_budgets", {})
            if error_budgets:
                budget_table = Table(
                    title="Error Budget Status", show_header=True, box=None
                )
                budget_table.add_column("SLO", style="cyan")
                budget_table.add_column("Remaining", style="white")
                budget_table.add_column("Burn Rate", style="white")
                for slo_name, budget_info in error_budgets.items():
                    remaining_percent = budget_info.get("remaining_percent", 0)
                    burn_rate = budget_info.get("current_burn_rate", 0)
                    remaining_color = self._get_budget_color(remaining_percent)
                    burn_rate_color = self._get_burn_rate_color(burn_rate)
                    budget_table.add_row(
                        slo_name,
                        f"[{remaining_color}]{remaining_percent:.1f}%[/{remaining_color}]",
                        f"[{burn_rate_color}]{burn_rate:.2f}x[/{burn_rate_color}]",
                    )
                content.append(budget_table)
            if len(content) == 1:
                final_content = content[0]
            else:
                from rich.console import Group

                final_content = Group(*content)
            try:
                slo_widget = self.query_one("#slo-metrics-content", Static)
                slo_widget.update(final_content)
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(f"Failed to update SLO display: {e}")

    def _get_health_color(self, health_status: str) -> str:
        """Get color for health status."""
        health_colors = {
            "healthy": "green",
            "warning": "yellow",
            "critical": "red",
            "unknown": "gray",
        }
        return health_colors.get(health_status.lower(), "gray")

    def _get_budget_color(self, remaining_percent: float) -> str:
        """Get color for error budget remaining percentage."""
        if remaining_percent > 50:
            return "green"
        if remaining_percent > 20:
            return "yellow"
        return "red"

    def _get_burn_rate_color(self, burn_rate: float) -> str:
        """Get color for burn rate."""
        if burn_rate < 1.0:
            return "green"
        if burn_rate < 5.0:
            return "yellow"
        return "red"
