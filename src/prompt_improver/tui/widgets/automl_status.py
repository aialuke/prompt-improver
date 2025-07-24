"""AutoML Status Widget - displays optimization progress and results."""

from datetime import datetime

from rich.bar import Bar
from rich.console import Console
from rich.panel import Panel

from rich.table import Table

from textual.reactive import reactive
from textual.widgets import Static

class AutoMLStatusWidget(Static):
    """Widget displaying AutoML optimization status and progress."""

    automl_data = reactive({})

    def __init__(self, **kwargs):
        """Initialize the AutoML Status Widget.

        Args:
            **kwargs: Keyword arguments passed to the Static widget parent class.
                     Common parameters include 'id', 'classes', 'disabled', etc.
        """
        # Following 2025 best practices: Let the parent class handle all kwargs
        # This ensures proper widget initialization and DOM node creation
        try:
            super().__init__(**kwargs)
        except TypeError as e:
            # Fallback for invalid kwargs - use minimal initialization
            # This handles cases where test frameworks pass invalid parameters
            super().__init__()

        # Initialize console for rich rendering
        self.console = Console()

    def compose(self):
        """Compose the widget layout."""
        yield Static(id="automl-status-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        try:
            self.update_display()
        except Exception as e:
            # Graceful error handling during mount
            self.automl_data = {"error": f"Mount error: {str(e)}"}
            self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider."""
        try:
            self.automl_data = await data_provider.get_automl_status()
            self.update_display()
        except Exception as e:
            self.automl_data = {"error": str(e)}
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current AutoML data."""
        try:
            if not self.automl_data:
                # Show default state when no data is available
                self._show_default_state()
                return

            # Create main content panel
            content = []

            # Status and progress section
            status = self.automl_data.get("status", "unknown")
            status_color = self._get_status_color(status)

            # Create progress table
            progress_table = Table(title="AutoML Optimization", show_header=False, box=None)
            progress_table.add_column("Metric", style="cyan")
            progress_table.add_column("Value", style="white")

            progress_table.add_row("Status", f"[{status_color}]{status.upper()}[/{status_color}]")

            # Progress information
            current_trial = self.automl_data.get("current_trial", 0)
            total_trials = self.automl_data.get("total_trials", 0)

            if total_trials > 0:
                progress_percent = (current_trial / total_trials) * 100
                progress_bar = Bar(
                    size=20,
                    begin=0,
                    end=total_trials,
                    width=current_trial
                )
                progress_table.add_row("Progress", f"{progress_percent:.1f}% ({current_trial}/{total_trials})")
            else:
                progress_table.add_row("Progress", "N/A")

            # Completion stats
            trials_completed = self.automl_data.get("trials_completed", 0)
            trials_failed = self.automl_data.get("trials_failed", 0)
            success_rate = (trials_completed / max(trials_completed + trials_failed, 1)) * 100

            progress_table.add_row("Completed", str(trials_completed))
            progress_table.add_row("Failed", str(trials_failed))
            progress_table.add_row("Success Rate", f"{success_rate:.1f}%")

            # Best results section
            best_score = self.automl_data.get("best_score", 0.0)
            current_objective = self.automl_data.get("current_objective", "accuracy")

            progress_table.add_row("Best Score", f"{best_score:.4f}")
            progress_table.add_row("Objective", current_objective)

            # Time information
            optimization_time = self.automl_data.get("optimization_time", 0)
            eta_completion = self.automl_data.get("eta_completion")

            if optimization_time > 0:
                time_str = self._format_duration(optimization_time)
                progress_table.add_row("Runtime", time_str)

            if eta_completion:
                eta_str = self._format_eta(eta_completion)
                progress_table.add_row("ETA", eta_str)

            content.append(progress_table)

            # Best parameters section
            best_params = self.automl_data.get("best_params", {})
            if best_params:
                params_table = Table(title="Best Parameters", show_header=True, box=None)
                params_table.add_column("Parameter", style="cyan")
                params_table.add_column("Value", style="white")

                for param, value in best_params.items():
                    # Format value based on type
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    elif isinstance(value, bool):
                        formatted_value = "✓" if value else "✗"
                    else:
                        formatted_value = str(value)

                    params_table.add_row(param, formatted_value)

                content.append(params_table)

            # Recent scores visualization
            recent_scores = self.automl_data.get("recent_scores", [])
            if recent_scores:
                # Create simple score trend
                score_trend = self._create_score_trend(recent_scores)
                content.append(Panel(score_trend, title="Score Trend"))

            # Error handling
            if "error" in self.automl_data:
                error_panel = Panel(
                    f"[red]{self.automl_data['error']}[/red]",
                    title="Error",
                    border_style="red"
                )
                content.append(error_panel)

            # Combine all content
            if len(content) == 1:
                final_content = content[0]
            else:
                # Create a group of renderables
                from rich.console import Group
                final_content = Group(*content)

            # Update the display
            content_widget = self.query_one("#automl-status-content", Static)
            content_widget.update(final_content)

        except Exception as e:
            # Graceful error handling during display update
            self._show_error_state(f"Display update error: {str(e)}")

    def _show_default_state(self) -> None:
        """Show default state when no data is available."""
        try:
            default_content = Panel(
                "AutoML Status: No data available",
                title="AutoML Optimization",
                border_style="dim"
            )
            content_widget = self.query_one("#automl-status-content", Static)
            content_widget.update(default_content)
        except Exception:
            # If even default state fails, just pass silently
            pass

    def _show_error_state(self, error_message: str) -> None:
        """Show error state with the given message."""
        try:
            error_content = Panel(
                f"[red]{error_message}[/red]",
                title="AutoML Status Error",
                border_style="red"
            )
            content_widget = self.query_one("#automl-status-content", Static)
            content_widget.update(error_content)
        except Exception:
            # If even error state fails, just pass silently
            pass

    def _get_status_color(self, status: str) -> str:
        """Get color for AutoML status."""
        status_colors = {
            "running": "green",
            "optimizing": "green",
            "completed": "blue",
            "failed": "red",
            "idle": "yellow",
            "paused": "yellow",
            "error": "red",
        }
        return status_colors.get(status.lower(), "white")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        hours = seconds / 3600
        return f"{hours:.1f}h"

    def _format_eta(self, eta: str) -> str:
        """Format ETA for display."""
        if isinstance(eta, datetime):
            now = datetime.now()
            if eta > now:
                delta = eta - now
                return self._format_duration(delta.total_seconds())
            return "Complete"
        return str(eta)

    def _create_score_trend(self, scores: list[float]) -> str:
        """Create a simple text-based score trend visualization."""
        if not scores:
            return "No data"

        # Normalize scores for display
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return "█" * len(scores)

        # Create simple bar chart
        bars = []
        for score in scores:
            normalized = (score - min_score) / (max_score - min_score)
            height = int(normalized * 8)  # 8 levels of bars
            bar_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            bars.append(bar_chars[min(height, 7)])

        trend_text = "".join(bars)

        # Add score range info
        range_info = f"Range: {min_score:.3f} - {max_score:.3f}"

        return f"{trend_text}\n{range_info}"
