"""APES Interactive Dashboard - Rich TUI interface for system monitoring and management.
Provides real-time monitoring of AutoML optimization, A/B testing, and system health.
"""
import asyncio
from datetime import datetime
from rich.console import Console
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Header, TabbedContent, TabPane
from prompt_improver.tui.data_provider import APESDataProvider
from prompt_improver.tui.widgets.ab_testing import ABTestingWidget
from prompt_improver.tui.widgets.automl_status import AutoMLStatusWidget
from prompt_improver.tui.widgets.performance_metrics import PerformanceMetricsWidget
from prompt_improver.tui.widgets.service_control import ServiceControlWidget
from prompt_improver.tui.widgets.system_overview import SystemOverviewWidget

class APESDashboard(App):
    """APES Interactive Dashboard - Rich TUI application for system monitoring.

    features:
    - Real-time system monitoring
    - AutoML optimization tracking
    - A/B testing results
    - Performance metrics
    - Service management controls
    """
    CSS_PATH = 'dashboard.tcss'
    system_status = reactive('initializing')
    last_update = reactive(datetime.now())

    def __init__(self, console: Console | None=None):
        super().__init__()
        self.console = console or Console()
        self.data_provider = APESDataProvider()
        self.update_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header()
        with Container(id='main-container'), TabbedContent(id='main-tabs'):
            with TabPane('Overview', id='overview-tab'):
                with Horizontal(id='overview-layout'):
                    with Vertical(id='left-panel'):
                        yield SystemOverviewWidget(id='system-overview')
                        yield ServiceControlWidget(id='service-control')
                    with Vertical(id='right-panel'):
                        yield PerformanceMetricsWidget(id='performance-metrics')
                        yield AutoMLStatusWidget(id='automl-status')
            with TabPane('A/B Testing', id='ab-testing-tab'):
                yield ABTestingWidget(id='ab-testing')
            with TabPane('System Health', id='health-tab'):
                yield SystemOverviewWidget(id='system-health-detail')
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard when mounted."""
        self.title = 'APES Dashboard'
        self.sub_title = 'Adaptive Prompt Enhancement System'
        self.update_timer = self.set_interval(2.0, self.update_dashboard)
        self.call_after_refresh(self.initial_load)

    async def initial_load(self) -> None:
        """Load initial data for all widgets."""
        try:
            await self.data_provider.initialize()
            await self.refresh_all_widgets()
            self.system_status = 'online'
            self.notify('Dashboard initialized successfully', severity='information')
        except Exception as e:
            self.system_status = 'error'
            self.notify(f'Failed to initialize dashboard: {e!s}', severity='error')

    async def update_dashboard(self) -> None:
        """Update dashboard data periodically."""
        try:
            await self.refresh_all_widgets()
            self.last_update = datetime.now()
        except Exception as e:
            self.notify(f'Update failed: {e!s}', severity='warning')

    async def refresh_all_widgets(self) -> None:
        """Refresh data for all dashboard widgets."""
        tasks = []
        widgets = [self.query_one('#system-overview', SystemOverviewWidget), self.query_one('#automl-status', AutoMLStatusWidget), self.query_one('#ab-testing', ABTestingWidget), self.query_one('#performance-metrics', PerformanceMetricsWidget), self.query_one('#service-control', ServiceControlWidget)]
        for widget in widgets:
            if hasattr(widget, 'update_data'):
                tasks.append(widget.update_data(self.data_provider))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark
        self.notify('Dark mode ' + ('enabled' if self.dark else 'disabled'))

    def action_refresh(self) -> None:
        """Manually refresh dashboard data."""
        self.call_after_refresh(self.refresh_all_widgets)
        self.notify('Dashboard refreshed')

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def on_unmount(self) -> None:
        """Cleanup when dashboard is unmounted."""
        if self.update_timer:
            self.update_timer.stop()

def run_dashboard(console: Console | None=None) -> None:
    """Run the APES dashboard application."""
    app = APESDashboard(console)
    app.run()
if __name__ == '__main__':
    run_dashboard()
