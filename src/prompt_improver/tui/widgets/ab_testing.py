"""A/B Testing Widget - displays experiment results and statistics."""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual.reactive import reactive
from textual.widgets import Static


class ABTestingWidget(Static):
    """Widget displaying A/B testing experiments and results."""

    ab_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.console = Console()

    def compose(self):
        """Compose the widget layout."""
        yield Static(id="ab-testing-content")

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()

    async def update_data(self, data_provider) -> None:
        """Update widget data from data provider."""
        try:
            self.ab_data = await data_provider.get_ab_testing_results()
            self.update_display()
        except Exception as e:
            self.ab_data = {"error": str(e)}
            self.update_display()

    def update_display(self) -> None:
        """Update the display with current A/B testing data."""
        if not self.ab_data:
            return
        content = []
        summary_table = Table(title="A/B Testing Summary", show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        active_experiments = self.ab_data.get("active_experiments", 0)
        total_experiments = self.ab_data.get("total_experiments", 0)
        success_rate = self.ab_data.get("success_rate", 0.0)
        avg_improvement = self.ab_data.get("avg_improvement", 0.0)
        significant_results = self.ab_data.get("significant_results", 0)
        summary_table.add_row("Active Experiments", str(active_experiments))
        summary_table.add_row("Total Experiments", str(total_experiments))
        summary_table.add_row("Success Rate", f"{success_rate:.1%}")
        summary_table.add_row("Avg Improvement", f"{avg_improvement:.1%}")
        summary_table.add_row("Significant Results", str(significant_results))
        content.append(summary_table)
        experiments = self.ab_data.get("experiments", [])
        if experiments:
            exp_table = Table(title="Active Experiments", show_header=True, box=None)
            exp_table.add_column("Name", style="cyan")
            exp_table.add_column("Status", style="white")
            exp_table.add_column("Participants", style="white")
            exp_table.add_column("Conversion", style="white")
            exp_table.add_column("Significance", style="white")
            for exp in experiments:
                name = exp.get("name", "Unknown")
                status = exp.get("status", "unknown")
                participants = exp.get("participants", 0)
                conversion_rate = exp.get("conversion_rate", 0.0)
                is_significant = exp.get("statistical_significance", False)
                status_color = self._get_status_color(status)
                status_display = f"[{status_color}]{status.upper()}[/{status_color}]"
                conversion_display = f"{conversion_rate:.2%}"
                significance_display = "✓" if is_significant else "✗"
                significance_color = "green" if is_significant else "red"
                significance_display = f"[{significance_color}]{significance_display}[/{significance_color}]"
                exp_table.add_row(
                    name,
                    status_display,
                    str(participants),
                    conversion_display,
                    significance_display,
                )
            content.append(exp_table)
        if experiments:
            detail_content = []
            for exp in experiments[:3]:
                exp_detail = self._create_experiment_detail(exp)
                detail_content.append(exp_detail)
            if detail_content:
                from rich.console import Group

                content.append(Group(*detail_content))
        if "error" in self.ab_data:
            error_panel = Panel(
                f"[red]{self.ab_data['error']}[/red]", title="Error", border_style="red"
            )
            content.append(error_panel)
        if len(content) == 1:
            final_content = content[0]
        else:
            from rich.console import Group

            final_content = Group(*content)
        content_widget = self.query_one("#ab-testing-content", Static)
        content_widget.update(final_content)

    def _get_status_color(self, status: str) -> str:
        """Get color for experiment status."""
        status_colors = {
            "running": "green",
            "completed": "blue",
            "paused": "yellow",
            "stopped": "red",
            "draft": "gray",
            "analyzing": "cyan",
        }
        return status_colors.get(status.lower(), "white")

    def _create_experiment_detail(self, exp: dict[str, Any]) -> Panel:
        """Create detailed view for an experiment."""
        name = exp.get("name", "Unknown")
        detail_table = Table(show_header=False, box=None)
        detail_table.add_column("Metric", style="cyan")
        detail_table.add_column("Value", style="white")
        start_date = exp.get("start_date")
        if start_date:
            if isinstance(start_date, datetime):
                start_str = format_date_only(start_date)
            else:
                start_str = str(start_date)
            detail_table.add_row("Start Date", start_str)
        participants = exp.get("participants", 0)
        detail_table.add_row("Participants", str(participants))
        conversion_rate = exp.get("conversion_rate", 0.0)
        detail_table.add_row("Conversion Rate", f"{conversion_rate:.2%}")
        is_significant = exp.get("statistical_significance", False)
        significance_text = "Yes" if is_significant else "No"
        significance_color = "green" if is_significant else "red"
        detail_table.add_row(
            "Significant",
            f"[{significance_color}]{significance_text}[/{significance_color}]",
        )
        confidence_interval = exp.get("confidence_interval", [0, 0])
        if confidence_interval and len(confidence_interval) == 2:
            ci_text = f"[{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"
            detail_table.add_row("95% CI", ci_text)
        effect_size = exp.get("effect_size", 0.0)
        effect_color = "green" if effect_size > 0 else "red"
        detail_table.add_row(
            "Effect Size", f"[{effect_color}]{effect_size:.3f}[/{effect_color}]"
        )
        return Panel(detail_table, title=name, border_style="blue")
