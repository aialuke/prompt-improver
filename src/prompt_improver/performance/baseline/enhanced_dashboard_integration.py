"""Enhanced Dashboard Integration for Performance Baseline System.

Provides real-time dashboard integration with advanced visualizations,
alert management, and interactive performance monitoring.
"""
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from prompt_improver.performance.baseline.baseline_collector import BaselineCollector
from prompt_improver.performance.baseline.models import BaselineMetrics, RegressionAlert
from prompt_improver.performance.baseline.regression_detector import RegressionDetector
from prompt_improver.performance.baseline.statistical_analyzer import StatisticalAnalyzer
from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for dashboard integration."""
    refresh_interval_seconds: int = 5
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    alert_sound: bool = False
    theme: str = 'dark'

class PerformanceDashboard:
    """Enhanced performance dashboard with real-time monitoring."""

    def __init__(self, config: DashboardConfig | None=None, collector: BaselineCollector | None=None, analyzer: StatisticalAnalyzer | None=None, detector: RegressionDetector | None=None):
        self.config = config or DashboardConfig()
        self.collector = collector or BaselineCollector()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.detector = detector or RegressionDetector()
        self.real_time_data: dict[str, list] = {'timestamps': [], 'response_times': [], 'cpu_utilization': [], 'memory_utilization': [], 'error_rates': [], 'throughput': []}
        self.websocket_connections: list = []
        self.running = False
        self.active_alerts: list[RegressionAlert] = []
        self.alert_history: list[dict] = []

    async def start_dashboard_server(self, port: int=8501) -> None:
        """Start the dashboard server."""
        if not STREAMLIT_AVAILABLE:
            logger.error('Streamlit not available for dashboard')
            return
        self.running = True
        task_manager = get_background_task_manager()
        data_task_id = await task_manager.submit_enhanced_task(task_id=f'dashboard_data_collection_{str(uuid.uuid4())[:8]}', coroutine=self._collect_real_time_data(), priority=TaskPriority.NORMAL, tags={'service': 'performance', 'type': 'dashboard', 'component': 'data_collection'})
        websocket_task_id = None
        if WEBSOCKETS_AVAILABLE:
            websocket_task_id = await task_manager.submit_enhanced_task(task_id=f'dashboard_websocket_{port + 1}_{str(uuid.uuid4())[:8]}', coroutine=self._start_websocket_server(port + 1), priority=TaskPriority.NORMAL, tags={'service': 'performance', 'type': 'dashboard', 'component': 'websocket', 'port': str(port + 1)})
        logger.info('Dashboard server started on port %s', port)
        try:
            await self._run_dashboard_loop()
        except Exception as e:
            logger.error('Dashboard server error: %s', e)
        finally:
            await task_manager.cancel_task(data_task_id)
            if websocket_task_id:
                await task_manager.cancel_task(websocket_task_id)

    async def _collect_real_time_data(self) -> None:
        """Collect real-time performance data for dashboard."""
        while self.running:
            try:
                baseline = await self.collector.collect_baseline()
                timestamp = baseline.collection_timestamp
                self.real_time_data['timestamps'].append(timestamp)
                if baseline.response_times:
                    avg_response = sum(baseline.response_times) / len(baseline.response_times)
                    self.real_time_data['response_times'].append(avg_response)
                else:
                    self.real_time_data['response_times'].append(0)
                if baseline.cpu_utilization:
                    avg_cpu = sum(baseline.cpu_utilization) / len(baseline.cpu_utilization)
                    self.real_time_data['cpu_utilization'].append(avg_cpu)
                else:
                    self.real_time_data['cpu_utilization'].append(0)
                if baseline.memory_utilization:
                    avg_memory = sum(baseline.memory_utilization) / len(baseline.memory_utilization)
                    self.real_time_data['memory_utilization'].append(avg_memory)
                else:
                    self.real_time_data['memory_utilization'].append(0)
                if baseline.error_rates:
                    avg_error = sum(baseline.error_rates) / len(baseline.error_rates)
                    self.real_time_data['error_rates'].append(avg_error)
                else:
                    self.real_time_data['error_rates'].append(0)
                if baseline.throughput_values:
                    avg_throughput = sum(baseline.throughput_values) / len(baseline.throughput_values)
                    self.real_time_data['throughput'].append(avg_throughput)
                else:
                    self.real_time_data['throughput'].append(0)
                for key in self.real_time_data:
                    if len(self.real_time_data[key]) > self.config.max_data_points:
                        self.real_time_data[key] = self.real_time_data[key][-self.config.max_data_points:]
                if self.config.enable_alerts:
                    await self._check_and_process_alerts(baseline)
                await self._broadcast_real_time_update()
                await asyncio.sleep(self.config.refresh_interval_seconds)
            except Exception as e:
                logger.error('Error collecting real-time data: %s', e)
                await asyncio.sleep(self.config.refresh_interval_seconds)

    async def _check_and_process_alerts(self, baseline: BaselineMetrics) -> None:
        """Check for alerts and process them."""
        try:
            recent_baselines = await self.collector.load_recent_baselines(2)
            if len(recent_baselines) >= 2:
                alerts = await self.detector.check_for_regressions(baseline, recent_baselines[:-1])
                for alert in alerts:
                    if alert.alert_id not in [a.alert_id for a in self.active_alerts]:
                        self.active_alerts.append(alert)
                        self.alert_history.append({'alert': alert, 'timestamp': datetime.now(UTC), 'status': 'new'})
                        logger.warning('New alert: %s', alert.message)
        except Exception as e:
            logger.error('Error checking alerts: %s', e)

    async def _start_websocket_server(self, port: int) -> None:
        """Start WebSocket server for real-time updates."""
        if not WEBSOCKETS_AVAILABLE:
            return

        async def handle_websocket(websocket, path):
            self.websocket_connections.append(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_connections.remove(websocket)
        try:
            await websockets.serve(handle_websocket, 'localhost', port)
            logger.info('WebSocket server started on port %s', port)
        except Exception as e:
            logger.error('Failed to start WebSocket server: %s', e)

    async def _broadcast_real_time_update(self) -> None:
        """Broadcast real-time updates to WebSocket clients."""
        if not self.websocket_connections:
            return
        update_data = {'type': 'performance_update', 'timestamp': datetime.now(UTC).isoformat(), 'data': {'latest_metrics': {'response_time': self.real_time_data['response_times'][-1] if self.real_time_data['response_times'] else 0, 'cpu_utilization': self.real_time_data['cpu_utilization'][-1] if self.real_time_data['cpu_utilization'] else 0, 'memory_utilization': self.real_time_data['memory_utilization'][-1] if self.real_time_data['memory_utilization'] else 0, 'error_rate': self.real_time_data['error_rates'][-1] if self.real_time_data['error_rates'] else 0, 'throughput': self.real_time_data['throughput'][-1] if self.real_time_data['throughput'] else 0}, 'active_alerts': len(self.active_alerts), 'trend_data': self._get_trend_data()}}
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send(json.dumps(update_data))
            except Exception:
                disconnected.append(websocket)
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)

    def _get_trend_data(self) -> dict[str, str]:
        """Get current trend indicators."""
        trends = {}
        for metric in ['response_times', 'cpu_utilization', 'memory_utilization', 'error_rates']:
            data = self.real_time_data[metric]
            if len(data) >= 10:
                recent = data[-10:]
                earlier = data[-20:-10] if len(data) >= 20 else data[:-10]
                if earlier:
                    recent_avg = sum(recent) / len(recent)
                    earlier_avg = sum(earlier) / len(earlier)
                    if recent_avg > earlier_avg * 1.1:
                        trends[metric] = 'increasing'
                    elif recent_avg < earlier_avg * 0.9:
                        trends[metric] = 'decreasing'
                    else:
                        trends[metric] = 'stable'
                else:
                    trends[metric] = 'stable'
            else:
                trends[metric] = 'insufficient_data'
        return trends

    async def _run_dashboard_loop(self) -> None:
        """Main dashboard loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.refresh_interval_seconds)
            except Exception as e:
                logger.error('Dashboard loop error: %s', e)
                await asyncio.sleep(1)

    def create_performance_charts(self) -> dict[str, Any]:
        """Create performance charts for dashboard."""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        charts = {}
        if self.real_time_data['timestamps'] and self.real_time_data['response_times']:
            fig_response = go.Figure()
            fig_response.add_trace(go.Scatter(x=self.real_time_data['timestamps'], y=self.real_time_data['response_times'], mode='lines+markers', name='Response Time', line=dict(color='#00D4AA', width=2)))
            fig_response.add_hline(y=200, line_dash='dash', line_color='red', annotation_text='200ms Target')
            fig_response.update_layout(title='Response Time Trend', xaxis_title='Time', yaxis_title='Response Time (ms)', template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white')
            charts['response_time'] = fig_response
        if self.real_time_data['timestamps']:
            fig_resources = make_subplots(rows=2, cols=1, subplot_titles=('CPU Utilization', 'Memory Utilization'), vertical_spacing=0.15)
            fig_resources.add_trace(go.Scatter(x=self.real_time_data['timestamps'], y=self.real_time_data['cpu_utilization'], mode='lines', name='CPU %', line=dict(color='#FF6B6B')), row=1, col=1)
            fig_resources.add_trace(go.Scatter(x=self.real_time_data['timestamps'], y=self.real_time_data['memory_utilization'], mode='lines', name='Memory %', line=dict(color='#4ECDC4')), row=2, col=1)
            fig_resources.update_layout(title='System Resource Utilization', template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white', height=600)
            charts['resources'] = fig_resources
        if self.real_time_data['timestamps']:
            fig_errors_throughput = make_subplots(rows=2, cols=1, subplot_titles=('Error Rate', 'Throughput'), vertical_spacing=0.15)
            fig_errors_throughput.add_trace(go.Scatter(x=self.real_time_data['timestamps'], y=self.real_time_data['error_rates'], mode='lines+markers', name='Error Rate %', line=dict(color='#FF4757')), row=1, col=1)
            fig_errors_throughput.add_trace(go.Scatter(x=self.real_time_data['timestamps'], y=self.real_time_data['throughput'], mode='lines', name='Requests/sec', line=dict(color='#5352ED')), row=2, col=1)
            fig_errors_throughput.update_layout(title='Error Rate and Throughput', template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white', height=600)
            charts['errors_throughput'] = fig_errors_throughput
        return charts

    def get_current_status(self) -> dict[str, Any]:
        """Get current system status for dashboard."""
        latest_data = {}
        for metric in ['response_times', 'cpu_utilization', 'memory_utilization', 'error_rates', 'throughput']:
            if self.real_time_data[metric]:
                latest_data[metric] = self.real_time_data[metric][-1]
            else:
                latest_data[metric] = 0
        health_status = 'healthy'
        health_issues = []
        if latest_data['response_times'] > 200:
            health_status = 'warning'
            health_issues.append('Response time exceeds 200ms target')
        if latest_data['cpu_utilization'] > 80:
            health_status = 'warning'
            health_issues.append('High CPU utilization')
        if latest_data['memory_utilization'] > 85:
            health_status = 'warning'
            health_issues.append('High memory utilization')
        if latest_data['error_rates'] > 5:
            health_status = 'critical'
            health_issues.append('High error rate')
        if len(self.active_alerts) > 0:
            health_status = 'critical' if any((alert.severity.value == 'critical' for alert in self.active_alerts)) else 'warning'
        return {'health_status': health_status, 'health_issues': health_issues, 'latest_metrics': latest_data, 'active_alerts': len(self.active_alerts), 'data_collection_status': 'running' if self.running else 'stopped', 'last_update': datetime.now(UTC).isoformat(), 'trends': self._get_trend_data()}

    def get_performance_summary(self, hours: int=24) -> dict[str, Any]:
        """Get performance summary for specified time period."""
        if not self.real_time_data['timestamps']:
            return {'error': 'No data available'}
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        summary = {'time_period_hours': hours, 'total_data_points': len(self.real_time_data['timestamps']), 'metrics_summary': {}}
        for metric in ['response_times', 'cpu_utilization', 'memory_utilization', 'error_rates', 'throughput']:
            data = self.real_time_data[metric]
            if data:
                summary['metrics_summary'][metric] = {'current': data[-1], 'average': sum(data) / len(data), 'maximum': max(data), 'minimum': min(data)}
        return summary

    async def stop_dashboard(self) -> None:
        """Stop the dashboard and cleanup resources."""
        self.running = False
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except Exception:
                pass
        self.websocket_connections.clear()
        logger.info('Dashboard stopped')

def create_streamlit_dashboard_app():
    """Create Streamlit dashboard application."""
    if not STREAMLIT_AVAILABLE:
        return None
    dashboard_code = '\nimport streamlit as st\nimport plotly.graph_objects as go\nfrom datetime import datetime, timedelta\nimport asyncio\nfrom enhanced_dashboard_integration import PerformanceDashboard\n\n# Configure Streamlit page\nst.set_page_config(\n    page_title="Performance Baseline Dashboard",\n    page_icon="📊",\n    layout="wide",\n    initial_sidebar_state="expanded"\n)\n\n# Initialize dashboard\n@st.cache_resource\ndef get_dashboard():\n    return PerformanceDashboard()\n\ndashboard = get_dashboard()\n\n# Sidebar controls\nst.sidebar.title("Dashboard Controls")\nrefresh_rate = st.sidebar.selectbox("Refresh Rate", [5, 10, 30, 60], index=0)\nauto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)\n\nif st.sidebar.button("Manual Refresh"):\n    st.rerun()\n\n# Main dashboard\nst.title("🚀 Performance Baseline Dashboard")\n\n# Status indicators\nstatus = dashboard.get_current_status()\n\ncol1, col2, col3, col4 = st.columns(4)\n\nwith col1:\n    health_color = {\n        "healthy": "green",\n        "warning": "orange", \n        "critical": "red"\n    }.get(status["health_status"], "gray")\n    \n    st.metric(\n        "System Health",\n        status["health_status"].title(),\n        delta_color="inverse"\n    )\n\nwith col2:\n    st.metric(\n        "Response Time",\n        f"{status[\'latest_metrics\'][\'response_times\']:.1f}ms",\n        delta=f"Target: 200ms"\n    )\n\nwith col3:\n    st.metric(\n        "CPU Usage",\n        f"{status[\'latest_metrics\'][\'cpu_utilization\']:.1f}%",\n        delta_color="inverse"\n    )\n\nwith col4:\n    st.metric(\n        "Active Alerts",\n        status["active_alerts"],\n        delta_color="inverse"\n    )\n\n# Charts\ncharts = dashboard.create_performance_charts()\n\nif "response_time" in charts:\n    st.plotly_chart(charts["response_time"], use_container_width=True)\n\nif "resources" in charts:\n    st.plotly_chart(charts["resources"], use_container_width=True)\n\nif "errors_throughput" in charts:\n    st.plotly_chart(charts["errors_throughput"], use_container_width=True)\n\n# Performance Summary\nst.subheader("Performance Summary (Last 24 Hours)")\nsummary = dashboard.get_performance_summary(24)\nif "metrics_summary" in summary:\n    for metric, data in summary["metrics_summary"].items():\n        with st.expander(f"{metric.replace(\'_\', \' \').title()}"):\n            col1, col2, col3, col4 = st.columns(4)\n            col1.metric("Current", f"{data[\'current\']:.2f}")\n            col2.metric("Average", f"{data[\'average\']:.2f}")\n            col3.metric("Maximum", f"{data[\'maximum\']:.2f}")\n            col4.metric("Minimum", f"{data[\'minimum\']:.2f}")\n\n# Auto-refresh\nif auto_refresh:\n    time.sleep(refresh_rate)\n    st.rerun()\n'
    return dashboard_code
_performance_dashboard: PerformanceDashboard | None = None

def get_performance_dashboard() -> PerformanceDashboard:
    """Get global performance dashboard instance."""
    global _performance_dashboard
    if _performance_dashboard is None:
        _performance_dashboard = PerformanceDashboard()
    return _performance_dashboard

async def start_performance_dashboard(port: int=8501) -> None:
    """Start the performance dashboard."""
    dashboard = get_performance_dashboard()
    await dashboard.start_dashboard_server(port)

async def get_dashboard_data() -> dict[str, Any]:
    """Get current dashboard data."""
    dashboard = get_performance_dashboard()
    return {'status': dashboard.get_current_status(), 'charts': dashboard.create_performance_charts(), 'summary': dashboard.get_performance_summary()}
