"""Enhanced Dashboard Integration for Performance Baseline System.

Provides real-time dashboard integration with advanced visualizations,
alert management, and interactive performance monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Dashboard frameworks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .baseline_collector import BaselineCollector
from .statistical_analyzer import StatisticalAnalyzer
from .regression_detector import RegressionDetector
from .models import BaselineMetrics, RegressionAlert

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for dashboard integration."""
    refresh_interval_seconds: int = 5
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    alert_sound: bool = False
    theme: str = "dark"  # light, dark, auto
    
class PerformanceDashboard:
    """Enhanced performance dashboard with real-time monitoring."""
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        collector: Optional[BaselineCollector] = None,
        analyzer: Optional[StatisticalAnalyzer] = None,
        detector: Optional[RegressionDetector] = None
    ):
        self.config = config or DashboardConfig()
        self.collector = collector or BaselineCollector()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.detector = detector or RegressionDetector()
        
        # Real-time data storage
        self.real_time_data: Dict[str, List] = {
            'timestamps': [],
            'response_times': [],
            'cpu_utilization': [],
            'memory_utilization': [],
            'error_rates': [],
            'throughput': []
        }
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List = []
        self.running = False
        
        # Alert management
        self.active_alerts: List[RegressionAlert] = []
        self.alert_history: List[Dict] = []
        
    async def start_dashboard_server(self, port: int = 8501) -> None:
        """Start the dashboard server."""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit not available for dashboard")
            return
        
        self.running = True
        
        # Start real-time data collection
        data_task = asyncio.create_task(self._collect_real_time_data())
        
        # Start WebSocket server for real-time updates
        if WEBSOCKETS_AVAILABLE:
            websocket_task = asyncio.create_task(
                self._start_websocket_server(port + 1)
            )
        
        logger.info(f"Dashboard server started on port {port}")
        
        try:
            # This would normally start Streamlit app
            # In practice, you'd run: streamlit run dashboard_app.py
            await self._run_dashboard_loop()
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
        finally:
            data_task.cancel()
            if WEBSOCKETS_AVAILABLE:
                websocket_task.cancel()
    
    async def _collect_real_time_data(self) -> None:
        """Collect real-time performance data for dashboard."""
        while self.running:
            try:
                # Collect current metrics
                baseline = await self.collector.collect_baseline()
                timestamp = baseline.collection_timestamp
                
                # Update real-time data
                self.real_time_data['timestamps'].append(timestamp)
                
                # Response times
                if baseline.response_times:
                    avg_response = sum(baseline.response_times) / len(baseline.response_times)
                    self.real_time_data['response_times'].append(avg_response)
                else:
                    self.real_time_data['response_times'].append(0)
                
                # CPU utilization
                if baseline.cpu_utilization:
                    avg_cpu = sum(baseline.cpu_utilization) / len(baseline.cpu_utilization)
                    self.real_time_data['cpu_utilization'].append(avg_cpu)
                else:
                    self.real_time_data['cpu_utilization'].append(0)
                
                # Memory utilization
                if baseline.memory_utilization:
                    avg_memory = sum(baseline.memory_utilization) / len(baseline.memory_utilization)
                    self.real_time_data['memory_utilization'].append(avg_memory)
                else:
                    self.real_time_data['memory_utilization'].append(0)
                
                # Error rates
                if baseline.error_rates:
                    avg_error = sum(baseline.error_rates) / len(baseline.error_rates)
                    self.real_time_data['error_rates'].append(avg_error)
                else:
                    self.real_time_data['error_rates'].append(0)
                
                # Throughput
                if baseline.throughput_values:
                    avg_throughput = sum(baseline.throughput_values) / len(baseline.throughput_values)
                    self.real_time_data['throughput'].append(avg_throughput)
                else:
                    self.real_time_data['throughput'].append(0)
                
                # Limit data points
                for key in self.real_time_data:
                    if len(self.real_time_data[key]) > self.config.max_data_points:
                        self.real_time_data[key] = self.real_time_data[key][-self.config.max_data_points:]
                
                # Check for alerts
                if self.config.enable_alerts:
                    await self._check_and_process_alerts(baseline)
                
                # Broadcast to WebSocket clients
                await self._broadcast_real_time_update()
                
                await asyncio.sleep(self.config.refresh_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error collecting real-time data: {e}")
                await asyncio.sleep(self.config.refresh_interval_seconds)
    
    async def _check_and_process_alerts(self, baseline: BaselineMetrics) -> None:
        """Check for alerts and process them."""
        try:
            # Check for new alerts
            recent_baselines = await self.collector.load_recent_baselines(2)
            if len(recent_baselines) >= 2:
                alerts = await self.detector.check_for_regressions(
                    baseline, recent_baselines[:-1]
                )
                
                for alert in alerts:
                    if alert.alert_id not in [a.alert_id for a in self.active_alerts]:
                        self.active_alerts.append(alert)
                        self.alert_history.append({
                            'alert': alert,
                            'timestamp': datetime.now(timezone.utc),
                            'status': 'new'
                        })
                        
                        logger.warning(f"New alert: {alert.message}")
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
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
            await websockets.serve(handle_websocket, "localhost", port)
            logger.info(f"WebSocket server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _broadcast_real_time_update(self) -> None:
        """Broadcast real-time updates to WebSocket clients."""
        if not self.websocket_connections:
            return
        
        # Prepare update data
        update_data = {
            'type': 'performance_update',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': {
                'latest_metrics': {
                    'response_time': self.real_time_data['response_times'][-1] if self.real_time_data['response_times'] else 0,
                    'cpu_utilization': self.real_time_data['cpu_utilization'][-1] if self.real_time_data['cpu_utilization'] else 0,
                    'memory_utilization': self.real_time_data['memory_utilization'][-1] if self.real_time_data['memory_utilization'] else 0,
                    'error_rate': self.real_time_data['error_rates'][-1] if self.real_time_data['error_rates'] else 0,
                    'throughput': self.real_time_data['throughput'][-1] if self.real_time_data['throughput'] else 0
                },
                'active_alerts': len(self.active_alerts),
                'trend_data': self._get_trend_data()
            }
        }
        
        # Broadcast to all connected clients
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send(json.dumps(update_data))
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def _get_trend_data(self) -> Dict[str, str]:
        """Get current trend indicators."""
        trends = {}
        
        # Simple trend calculation for last 10 data points
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
                # Dashboard would update here
                # In practice, this would be handled by Streamlit's auto-refresh
                await asyncio.sleep(self.config.refresh_interval_seconds)
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                await asyncio.sleep(1)
    
    def create_performance_charts(self) -> Dict[str, Any]:
        """Create performance charts for dashboard."""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        charts = {}
        
        # Response Time Chart
        if self.real_time_data['timestamps'] and self.real_time_data['response_times']:
            fig_response = go.Figure()
            fig_response.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['response_times'],
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#00D4AA', width=2)
            ))
            
            # Add target line at 200ms
            fig_response.add_hline(
                y=200, 
                line_dash="dash", 
                line_color="red",
                annotation_text="200ms Target"
            )
            
            fig_response.update_layout(
                title='Response Time Trend',
                xaxis_title='Time',
                yaxis_title='Response Time (ms)',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
            )
            
            charts['response_time'] = fig_response
        
        # Resource Utilization Chart
        if self.real_time_data['timestamps']:
            fig_resources = make_subplots(
                rows=2, cols=1,
                subplot_titles=('CPU Utilization', 'Memory Utilization'),
                vertical_spacing=0.15
            )
            
            # CPU
            fig_resources.add_trace(
                go.Scatter(
                    x=self.real_time_data['timestamps'],
                    y=self.real_time_data['cpu_utilization'],
                    mode='lines',
                    name='CPU %',
                    line=dict(color='#FF6B6B')
                ),
                row=1, col=1
            )
            
            # Memory
            fig_resources.add_trace(
                go.Scatter(
                    x=self.real_time_data['timestamps'],
                    y=self.real_time_data['memory_utilization'],
                    mode='lines',
                    name='Memory %',
                    line=dict(color='#4ECDC4')
                ),
                row=2, col=1
            )
            
            fig_resources.update_layout(
                title='System Resource Utilization',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                height=600
            )
            
            charts['resources'] = fig_resources
        
        # Error Rate and Throughput
        if self.real_time_data['timestamps']:
            fig_errors_throughput = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Error Rate', 'Throughput'),
                vertical_spacing=0.15
            )
            
            # Error Rate
            fig_errors_throughput.add_trace(
                go.Scatter(
                    x=self.real_time_data['timestamps'],
                    y=self.real_time_data['error_rates'],
                    mode='lines+markers',
                    name='Error Rate %',
                    line=dict(color='#FF4757')
                ),
                row=1, col=1
            )
            
            # Throughput
            fig_errors_throughput.add_trace(
                go.Scatter(
                    x=self.real_time_data['timestamps'],
                    y=self.real_time_data['throughput'],
                    mode='lines',
                    name='Requests/sec',
                    line=dict(color='#5352ED')
                ),
                row=2, col=1
            )
            
            fig_errors_throughput.update_layout(
                title='Error Rate and Throughput',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                height=600
            )
            
            charts['errors_throughput'] = fig_errors_throughput
        
        return charts
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status for dashboard."""
        latest_data = {}
        
        for metric in ['response_times', 'cpu_utilization', 'memory_utilization', 'error_rates', 'throughput']:
            if self.real_time_data[metric]:
                latest_data[metric] = self.real_time_data[metric][-1]
            else:
                latest_data[metric] = 0
        
        # Determine overall health
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
            health_status = 'critical' if any(alert.severity.value == 'critical' for alert in self.active_alerts) else 'warning'
        
        return {
            'health_status': health_status,
            'health_issues': health_issues,
            'latest_metrics': latest_data,
            'active_alerts': len(self.active_alerts),
            'data_collection_status': 'running' if self.running else 'stopped',
            'last_update': datetime.now(timezone.utc).isoformat(),
            'trends': self._get_trend_data()
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        if not self.real_time_data['timestamps']:
            return {'error': 'No data available'}
        
        # Filter data for time period
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Simple summary (in practice would filter by timestamp)
        summary = {
            'time_period_hours': hours,
            'total_data_points': len(self.real_time_data['timestamps']),
            'metrics_summary': {}
        }
        
        for metric in ['response_times', 'cpu_utilization', 'memory_utilization', 'error_rates', 'throughput']:
            data = self.real_time_data[metric]
            if data:
                summary['metrics_summary'][metric] = {
                    'current': data[-1],
                    'average': sum(data) / len(data),
                    'maximum': max(data),
                    'minimum': min(data)
                }
        
        return summary
    
    async def stop_dashboard(self) -> None:
        """Stop the dashboard and cleanup resources."""
        self.running = False
        
        # Close WebSocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except Exception:
                pass
        
        self.websocket_connections.clear()
        logger.info("Dashboard stopped")

# Streamlit Dashboard Application (separate file: dashboard_app.py)
def create_streamlit_dashboard_app():
    """Create Streamlit dashboard application."""
    if not STREAMLIT_AVAILABLE:
        return None
    
    # This would be in a separate dashboard_app.py file
    dashboard_code = '''
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
from enhanced_dashboard_integration import PerformanceDashboard

# Configure Streamlit page
st.set_page_config(
    page_title="Performance Baseline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dashboard
@st.cache_resource
def get_dashboard():
    return PerformanceDashboard()

dashboard = get_dashboard()

# Sidebar controls
st.sidebar.title("Dashboard Controls")
refresh_rate = st.sidebar.selectbox("Refresh Rate", [5, 10, 30, 60], index=0)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

if st.sidebar.button("Manual Refresh"):
    st.rerun()

# Main dashboard
st.title("🚀 Performance Baseline Dashboard")

# Status indicators
status = dashboard.get_current_status()

col1, col2, col3, col4 = st.columns(4)

with col1:
    health_color = {
        "healthy": "green",
        "warning": "orange", 
        "critical": "red"
    }.get(status["health_status"], "gray")
    
    st.metric(
        "System Health",
        status["health_status"].title(),
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Response Time",
        f"{status['latest_metrics']['response_times']:.1f}ms",
        delta=f"Target: 200ms"
    )

with col3:
    st.metric(
        "CPU Usage",
        f"{status['latest_metrics']['cpu_utilization']:.1f}%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Active Alerts",
        status["active_alerts"],
        delta_color="inverse"
    )

# Charts
charts = dashboard.create_performance_charts()

if "response_time" in charts:
    st.plotly_chart(charts["response_time"], use_container_width=True)

if "resources" in charts:
    st.plotly_chart(charts["resources"], use_container_width=True)

if "errors_throughput" in charts:
    st.plotly_chart(charts["errors_throughput"], use_container_width=True)

# Performance Summary
st.subheader("Performance Summary (Last 24 Hours)")
summary = dashboard.get_performance_summary(24)
if "metrics_summary" in summary:
    for metric, data in summary["metrics_summary"].items():
        with st.expander(f"{metric.replace('_', ' ').title()}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current", f"{data['current']:.2f}")
            col2.metric("Average", f"{data['average']:.2f}")
            col3.metric("Maximum", f"{data['maximum']:.2f}")
            col4.metric("Minimum", f"{data['minimum']:.2f}")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
'''
    
    return dashboard_code

# Global dashboard instance
_performance_dashboard: Optional[PerformanceDashboard] = None

def get_performance_dashboard() -> PerformanceDashboard:
    """Get global performance dashboard instance."""
    global _performance_dashboard
    if _performance_dashboard is None:
        _performance_dashboard = PerformanceDashboard()
    return _performance_dashboard

# Convenience functions
async def start_performance_dashboard(port: int = 8501) -> None:
    """Start the performance dashboard."""
    dashboard = get_performance_dashboard()
    await dashboard.start_dashboard_server(port)

async def get_dashboard_data() -> Dict[str, Any]:
    """Get current dashboard data."""
    dashboard = get_performance_dashboard()
    return {
        'status': dashboard.get_current_status(),
        'charts': dashboard.create_performance_charts(),
        'summary': dashboard.get_performance_summary()
    }