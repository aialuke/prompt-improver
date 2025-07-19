"""Performance Metrics Widget - displays real-time system performance data."""

from datetime import datetime
from typing import Any, Dict, List

from rich.bar import Bar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static


class PerformanceMetricsWidget(Static):
    """Widget displaying real-time performance metrics."""

    performance_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.console = Console()

    def compose(self):
        """Compose the widget layout."""
        yield Static(id="performance-metrics-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider."""
        try:
            self.performance_data = await data_provider.get_performance_metrics()
            self.update_display()
        except Exception as e:
            self.performance_data = {"error": str(e)}
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current performance data."""
        if not self.performance_data:
            return

        content = []

        # Main metrics table
        metrics_table = Table(title="Performance Metrics", show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        # Response time
        response_time = self.performance_data.get("response_time", 0)
        response_color = self._get_response_time_color(response_time)
        metrics_table.add_row(
            "Response Time",
            f"[{response_color}]{response_time:.1f}ms[/{response_color}]"
        )

        # Throughput
        throughput = self.performance_data.get("throughput", 0)
        metrics_table.add_row("Throughput", f"{throughput:.1f} req/s")

        # Error rate
        error_rate = self.performance_data.get("error_rate", 0.0)
        error_color = self._get_error_rate_color(error_rate)
        metrics_table.add_row(
            "Error Rate",
            f"[{error_color}]{error_rate:.2%}[/{error_color}]"
        )

        # Active connections
        active_connections = self.performance_data.get("active_connections", 0)
        metrics_table.add_row("Active Connections", str(active_connections))

        # Queue length
        queue_length = self.performance_data.get("queue_length", 0)
        queue_color = self._get_queue_color(queue_length)
        metrics_table.add_row(
            "Queue Length",
            f"[{queue_color}]{queue_length}[/{queue_color}]"
        )

        # Resource usage
        memory_usage = self.performance_data.get("memory_usage", 0)
        cpu_usage = self.performance_data.get("cpu_usage", 0)

        metrics_table.add_row("Memory Usage", self._format_usage(memory_usage))
        metrics_table.add_row("CPU Usage", self._format_usage(cpu_usage))

        # Cache hit rate
        cache_hit_rate = self.performance_data.get("cache_hit_rate", 0.0)
        cache_color = self._get_cache_color(cache_hit_rate)
        metrics_table.add_row(
            "Cache Hit Rate",
            f"[{cache_color}]{cache_hit_rate:.1%}[/{cache_color}]"
        )

        content.append(metrics_table)

        # Recent response times visualization
        recent_response_times = self.performance_data.get("recent_response_times", [])
        if recent_response_times:
            response_chart = self._create_response_time_chart(recent_response_times)
            content.append(Panel(response_chart, title="Response Time Trend"))

        # Throughput visualization
        recent_throughput = self.performance_data.get("recent_throughput", [])
        if recent_throughput:
            throughput_chart = self._create_throughput_chart(recent_throughput)
            content.append(Panel(throughput_chart, title="Throughput Trend"))

        # Performance alerts
        alerts = self._generate_performance_alerts()
        if alerts:
            alert_panel = Panel(
                "\n".join(alerts),
                title="Performance Alerts",
                border_style="yellow"
            )
            content.append(alert_panel)

        # Error handling
        if "error" in self.performance_data:
            error_panel = Panel(
                f"[red]{self.performance_data['error']}[/red]",
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
        content_widget = self.query_one("#performance-metrics-content", Static)
        content_widget.update(final_content)

    def _get_response_time_color(self, response_time: float) -> str:
        """Get color for response time based on performance thresholds."""
        if response_time < 100:
            return "green"
        if response_time < 500:
            return "yellow"
        return "red"

    def _get_error_rate_color(self, error_rate: float) -> str:
        """Get color for error rate."""
        if error_rate < 0.01:  # < 1%
            return "green"
        if error_rate < 0.05:  # < 5%
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
        if cache_hit_rate > 0.9:  # > 90%
            return "green"
        if cache_hit_rate > 0.7:  # > 70%
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

        # Create simple sparkline
        min_time = min(response_times)
        max_time = max(response_times)

        if max_time == min_time:
            return "─" * len(response_times)

        # Create simple line chart
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

        # Create simple sparkline
        min_throughput = min(throughput_data)
        max_throughput = max(throughput_data)

        if max_throughput == min_throughput:
            return "─" * len(throughput_data)

        # Create simple line chart
        lines = []
        for throughput in throughput_data:
            normalized = (throughput - min_throughput) / (max_throughput - min_throughput)
            height = int(normalized * 8)
            line_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            lines.append(line_chars[min(height, 7)])

        chart_text = "".join(lines)
        range_info = f"Range: {min_throughput:.1f} - {max_throughput:.1f} req/s"

        return f"{chart_text}\n{range_info}"

    def _generate_performance_alerts(self) -> list[str]:
        """Generate performance alerts based on current metrics."""
        alerts = []

        # Check response time
        response_time = self.performance_data.get("response_time", 0)
        if response_time > 1000:
            alerts.append(f"[red]⚠ High response time: {response_time:.1f}ms[/red]")

        # Check error rate
        error_rate = self.performance_data.get("error_rate", 0.0)
        if error_rate > 0.05:
            alerts.append(f"[red]⚠ High error rate: {error_rate:.2%}[/red]")

        # Check queue length
        queue_length = self.performance_data.get("queue_length", 0)
        if queue_length > 50:
            alerts.append(f"[yellow]⚠ High queue length: {queue_length}[/yellow]")

        # Check memory usage
        memory_usage = self.performance_data.get("memory_usage", 0)
        if memory_usage > 90:
            alerts.append(f"[red]⚠ High memory usage: {memory_usage:.1f}%[/red]")

        # Check CPU usage
        cpu_usage = self.performance_data.get("cpu_usage", 0)
        if cpu_usage > 90:
            alerts.append(f"[red]⚠ High CPU usage: {cpu_usage:.1f}%[/red]")

        # Check cache hit rate
        cache_hit_rate = self.performance_data.get("cache_hit_rate", 0.0)
        if cache_hit_rate < 0.7:
            alerts.append(f"[yellow]⚠ Low cache hit rate: {cache_hit_rate:.1%}[/yellow]")

        return alerts
