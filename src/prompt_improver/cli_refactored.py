#!/usr/bin/env python3
"""Refactored CLI functions for Phase 4 - High-Impact Refactors.
This module contains the refactored versions of complex CLI functions with:
- Reduced cyclomatic complexity
- Functional extraction
- Strategy pattern implementation
- Better separation of concerns
"""
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


# Configuration constants (extracted magic numbers)
class LogConfig:
    DEFAULT_LINES = 50
    DEFAULT_LEVEL = "INFO"
    FOLLOW_SLEEP_INTERVAL = 0.1
    MAX_TAIL_WAIT_TIME = 5.0


class LogLevel(Enum):
    """Log level enumeration for type safety."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogDisplayOptions:
    """Configuration for log display."""
    lines: int
    level: Optional[str]
    component: Optional[str]
    follow: bool


class LogStyler(Protocol):
    """Protocol for log line styling strategies."""

    def style_line(self, line: str, console: Console) -> None:
        """Style and print a log line."""
        ...


class StandardLogStyler:
    """Standard log styling implementation."""

    def style_line(self, line: str, console: Console) -> None:
        """Apply color coding based on log level."""
        if "ERROR" in line:
            console.print(line.rstrip(), style="red")
        elif "WARNING" in line:
            console.print(line.rstrip(), style="yellow")
        elif "INFO" in line:
            console.print(line.rstrip(), style="green")
        elif "DEBUG" in line:
            console.print(line.rstrip(), style="dim")
        else:
            console.print(line.rstrip())


class LogFilter:
    """Handles log filtering logic."""

    def __init__(self, level_filter: Optional[str] = None):
        self.level_filter = level_filter

    def should_include_line(self, line: str) -> bool:
        """Determine if a log line should be included based on filters."""
        if self.level_filter and self.level_filter.upper() not in line.upper():
            return False
        return True


class LogReader(ABC):
    """Abstract base class for log reading strategies."""

    @abstractmethod
    def read_logs(self, log_file: Path, options: LogDisplayOptions) -> None:
        """Read and display logs according to options."""


class StaticLogReader(LogReader):
    """Reads a fixed number of log lines from the end of file."""

    def __init__(self, console: Console, styler: LogStyler, log_filter: LogFilter):
        self.console = console
        self.styler = styler
        self.log_filter = log_filter

    def read_logs(self, log_file: Path, options: LogDisplayOptions) -> None:
        """Read last N lines from log file."""
        self.console.print(f"📋 Last {options.lines} lines:", style="blue")

        try:
            with open(log_file, encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = (
                    all_lines[-options.lines:] if len(all_lines) > options.lines else all_lines
                )

                for line in recent_lines:
                    if self.log_filter.should_include_line(line):
                        self.styler.style_line(line, self.console)

        except Exception as e:
            self.console.print(f"❌ Failed to read logs: {e}", style="red")
            raise typer.Exit(1)


class FollowLogReader(LogReader):
    """Follows log file for real-time updates."""

    def __init__(self, console: Console, styler: LogStyler, log_filter: LogFilter):
        self.console = console
        self.styler = styler
        self.log_filter = log_filter

    def read_logs(self, log_file: Path, options: LogDisplayOptions) -> None:
        """Follow log file for real-time updates."""
        self.console.print(
            "👁️  Following log output (Press Ctrl+C to stop)...", style="green"
        )

        try:
            self._try_tail_command(log_file, options)
        except FileNotFoundError:
            self._fallback_python_follow(log_file, options)
        except Exception as e:
            self.console.print(f"❌ Failed to follow logs: {e}", style="red")
            raise typer.Exit(1)

    def _try_tail_command(self, log_file: Path, options: LogDisplayOptions) -> None:
        """Try using system tail command for following logs."""
        # Use absolute path for security
        import shutil
        tail_path = shutil.which("tail")
        if not tail_path:
            raise FileNotFoundError("tail command not found in PATH")
        
        # Security: subprocess call with validated executable path and secure parameters
        # - tail_path resolved via shutil.which() to prevent PATH injection
        # - shell=False prevents shell injection attacks
        # - log_file path is validated as existing file before use
        # - Arguments are controlled and validated
        process = subprocess.Popen(  # noqa: S603
            [tail_path, "-f", str(log_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )

        try:
            for line in process.stdout:
                if self.log_filter.should_include_line(line):
                    self.styler.style_line(line, self.console)
        except KeyboardInterrupt:
            self.console.print("\\n🔄 Stopping log viewer...", style="yellow")
            process.terminate()
            process.wait()
            self.console.print("✅ Log viewer stopped", style="green")

    def _fallback_python_follow(self, log_file: Path, options: LogDisplayOptions) -> None:
        """Fallback to Python-based log following."""
        self.console.print(
            "❌ 'tail' command not found. Using Python implementation...",
            style="yellow",
        )

        try:
            with open(log_file, encoding='utf-8') as f:
                # Go to end of file
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if line:
                        if self.log_filter.should_include_line(line):
                            self.styler.style_line(line, self.console)
                    else:
                        time.sleep(LogConfig.FOLLOW_SLEEP_INTERVAL)
        except KeyboardInterrupt:
            self.console.print("\\n✅ Log viewer stopped", style="green")


class LogFileValidator:
    """Validates log file paths and availability."""

    def __init__(self, console: Console):
        self.console = console

    def validate_log_directory(self, log_dir: Path) -> bool:
        """Check if log directory exists."""
        if not log_dir.exists():
            self.console.print(f"❌ Log directory not found: {log_dir}", style="red")
            self.console.print("💡 Run 'apes init' to create the log directory", style="dim")
            return False
        return True

    def determine_log_file(self, log_dir: Path, component: Optional[str]) -> Path:
        """Determine which log file to use based on component."""
        if component:
            return log_dir / f"{component}.log"
        return log_dir / "apes.log"

    def validate_log_file(self, log_file: Path, log_dir: Path) -> bool:
        """Check if log file exists and show available alternatives."""
        if not log_file.exists():
            self.console.print(f"❌ Log file not found: {log_file}", style="red")
            self.console.print("💡 Available log files:", style="dim")
            for log in log_dir.glob("*.log"):
                self.console.print(f"   • {log.name}")
            return False
        return True


class LogViewerService:
    """Main service for log viewing operations."""

    def __init__(self, console: Console):
        self.console = console
        self.validator = LogFileValidator(console)
        self.styler = StandardLogStyler()

    def view_logs(self, options: LogDisplayOptions) -> None:
        """Main entry point for log viewing."""
        # Find log files
        log_dir = Path.home() / ".local" / "share" / "apes" / "data" / "logs"

        if not self.validator.validate_log_directory(log_dir):
            raise typer.Exit(1)

        # Determine log file to view
        log_file = self.validator.determine_log_file(log_dir, options.component)

        if not self.validator.validate_log_file(log_file, log_dir):
            raise typer.Exit(1)

        self.console.print(f"📄 Viewing logs: {log_file}", style="blue")

        # Create appropriate reader strategy
        log_filter = LogFilter(options.level)

        if options.follow:
            reader = FollowLogReader(self.console, self.styler, log_filter)
        else:
            reader = StaticLogReader(self.console, self.styler, log_filter)

        # Execute log reading
        reader.read_logs(log_file, options)


# Refactored logs function with reduced complexity
def logs_refactored(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow log output in real-time"
    ),
    lines: int = typer.Option(LogConfig.DEFAULT_LINES, "--lines", "-n", help="Number of lines to show"),
    level: str = typer.Option(
        LogConfig.DEFAULT_LEVEL, "--level", help="Log level filter (DEBUG, INFO, WARNING, ERROR)"
    ),
    component: Optional[str] = typer.Option(
        None, "--component", help="Filter by component (mcp, database, training)"
    ),
):
    """View APES system logs (Phase 2) - Refactored version."""
    console = Console()

    options = LogDisplayOptions(
        lines=lines,
        level=level,
        component=component,
        follow=follow
    )

    log_service = LogViewerService(console)
    log_service.view_logs(options)


# Health check refactoring with similar patterns
@dataclass
class HealthCheckOptions:
    """Configuration for health check operations."""
    json_output: bool
    detailed: bool


class HealthDisplayFormatter:
    """Handles health check result formatting."""

    def __init__(self, console: Console):
        self.console = console

    def format_json_output(self, results: dict[str, Any]) -> None:
        """Format health results as JSON."""
        self.console.print_json(data=results)

    def format_table_output(self, results: dict[str, Any], detailed: bool) -> None:
        """Format health results as formatted tables."""
        self._show_overall_status(results)
        self._show_component_status(results)
        self._show_issues(results)

        if detailed:
            self._show_system_resources(results)

    def _show_overall_status(self, results: dict[str, Any]) -> None:
        """Display overall health status."""
        overall_status = results.get("overall_status", "unknown")
        status_icon = self._get_status_icon(overall_status)

        self.console.print(
            f"\\n{status_icon} Overall Health: {overall_status.upper()}",
            style="bold",
        )

    def _show_component_status(self, results: dict[str, Any]) -> None:
        """Display component health status table."""
        table = Table(title="Component Health Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="")
        table.add_column("Response Time", style="magenta")
        table.add_column("Details", style="dim")

        checks = results.get("checks", {})
        for component, check_result in checks.items():
            self._add_component_row(table, component, check_result)

        self.console.print(table)

    def _add_component_row(self, table: Table, component: str, check_result: dict[str, Any]) -> None:
        """Add a single component row to the health table."""
        status = check_result.get("status", "unknown")
        status_icon = self._get_status_icon(status)

        response_time = check_result.get("response_time_ms")
        response_str = f"{response_time:.1f}ms" if response_time else "-"

        message = check_result.get("message", "")
        if check_result.get("error"):
            message = f"Error: {check_result['error']}"

        table.add_row(
            component.replace("_", " ").title(),
            f"{status_icon} {status.capitalize()}",
            response_str,
            message,
        )

    def _show_issues(self, results: dict[str, Any]) -> None:
        """Display warnings and failures."""
        if "warning_checks" in results or "failed_checks" in results:
            self.console.print("\\n[bold]Issues Found:[/bold]")

            checks = results.get("checks", {})

            for warning in results.get("warning_checks", []):
                self.console.print(
                    f"⚠️  {warning}: {checks[warning].get('message', '')}",
                    style="yellow",
                )

            for failure in results.get("failed_checks", []):
                self.console.print(
                    f"❌ {failure}: {checks[failure].get('message', '')}",
                    style="red",
                )

    def _show_system_resources(self, results: dict[str, Any]) -> None:
        """Display detailed system resource information."""
        self.console.print("\\n[bold]System Resources:[/bold]")
        system_check = results.get("checks", {}).get("system_resources", {})

        if system_check:
            resource_table = Table()
            resource_table.add_column("Resource", style="cyan")
            resource_table.add_column("Usage", style="yellow")

            resources = [
                ("Memory", "memory_usage_percent"),
                ("CPU", "cpu_usage_percent"),
                ("Disk", "disk_usage_percent")
            ]

            for resource_name, key in resources:
                if key in system_check:
                    resource_table.add_row(resource_name, f"{system_check[key]:.1f}%")

            self.console.print(resource_table)

    def _get_status_icon(self, status: str) -> str:
        """Get appropriate icon for status."""
        if status == "healthy":
            return "✅"
        if status == "warning":
            return "⚠️"
        return "❌"


class HealthCheckService:
    """Service for executing health checks."""

    def __init__(self, console: Console):
        self.console = console
        self.formatter = HealthDisplayFormatter(console)

    async def run_health_check_with_progress(self) -> dict[str, Any]:
        """Run health check with progress indicator."""
        from prompt_improver.services.monitoring import HealthMonitor

        health_monitor = HealthMonitor()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running health diagnostics...", total=None)
            results = await health_monitor.run_health_check()
            progress.update(task, completed=True)

        return results

    def display_results(self, results: dict[str, Any], options: HealthCheckOptions) -> None:
        """Display health check results according to options."""
        if options.json_output:
            self.formatter.format_json_output(results)
        else:
            self.formatter.format_table_output(results, options.detailed)


def health_refactored(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed diagnostics"
    ),
):
    """Run comprehensive system health check (Phase 3B) - Refactored version."""
    import asyncio

    console = Console()
    console.print("🏥 Running APES Health Check...", style="blue")

    options = HealthCheckOptions(json_output=json_output, detailed=detailed)
    health_service = HealthCheckService(console)

    async def run_health_check():
        try:
            results = await health_service.run_health_check_with_progress()
            health_service.display_results(results, options)
        except Exception as e:
            console.print(f"❌ Health check failed: {e}", style="red")
            raise typer.Exit(1)

    try:
        asyncio.run(run_health_check())
    except Exception as e:
        console.print(f"❌ Health check failed: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    # Example usage and testing
    app = typer.Typer()
    app.command()(logs_refactored)
    app.command()(health_refactored)
    app()
