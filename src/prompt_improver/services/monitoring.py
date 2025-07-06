"""
Real-Time Monitoring Service for APES
Phase 3B: Advanced Monitoring & Analytics Implementation
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
from rich.text import Text
from rich.columns import Columns
from rich import box

from ..database import get_session
from ..services.analytics import AnalyticsService
from ..database.models import RulePerformance


@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    response_time_ms: int = 200
    cache_hit_ratio: float = 90.0
    database_connections: int = 15
    memory_usage_mb: int = 256
    error_rate_percent: float = 5.0


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    timestamp: datetime
    alert_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str


class RealTimeMonitor:
    """Real-time performance monitoring with alerting"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.analytics = AnalyticsService()
        self.alert_thresholds = AlertThreshold()
        self.active_alerts: List[PerformanceAlert] = []
        self.monitoring_active = False
        
    async def start_monitoring_dashboard(self, refresh_seconds: int = 5):
        """Live dashboard using existing analytics + Rich interface"""
        
        self.monitoring_active = True
        self.console.print("🚀 Starting APES Real-Time Monitoring Dashboard", style="green")
        
        with Live(auto_refresh=False) as live:
            while self.monitoring_active:
                try:
                    # Use existing analytics methods
                    performance_data = await self.analytics.get_performance_trends(days=1)
                    rule_effectiveness = await self.analytics.get_rule_effectiveness()
                    user_satisfaction = await self.analytics.get_user_satisfaction(days=1)
                    
                    # Get real-time system metrics
                    system_metrics = await self.collect_system_metrics()
                    
                    # Create Rich dashboard
                    dashboard = self.create_dashboard({
                        'performance': performance_data,
                        'effectiveness': rule_effectiveness, 
                        'satisfaction': user_satisfaction,
                        'system': system_metrics
                    })
                    
                    live.update(dashboard)
                    
                    # Check alerting thresholds
                    await self.check_performance_alerts(performance_data, system_metrics)
                    
                    await asyncio.sleep(refresh_seconds)
                    
                except Exception as e:
                    error_panel = Panel(
                        f"[red]Monitoring Error: {e}[/red]",
                        title="⚠️ Dashboard Error",
                        border_style="red"
                    )
                    live.update(error_panel)
                    await asyncio.sleep(refresh_seconds * 2)  # Longer delay on error
    
    def create_dashboard(self, data: Dict[str, Any]) -> Layout:
        """Create Rich dashboard layout"""
        
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )
        
        # Header with system status
        header_content = self.create_header_panel(data.get('system', {}))
        layout["header"].update(header_content)
        
        # Body with metrics
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Left column: Performance & System metrics
        layout["left"].split_column(
            Layout(name="performance", ratio=2),
            Layout(name="alerts", ratio=1)
        )
        
        # Right column: Rule effectiveness & User satisfaction
        layout["right"].split_column(
            Layout(name="effectiveness"),
            Layout(name="satisfaction")
        )
        
        # Populate sections
        layout["performance"].update(self.create_performance_panel(data.get('performance', {}), data.get('system', {})))
        layout["effectiveness"].update(self.create_effectiveness_panel(data.get('effectiveness', {})))
        layout["satisfaction"].update(self.create_satisfaction_panel(data.get('satisfaction', {})))
        layout["alerts"].update(self.create_alerts_panel())
        layout["footer"].update(self.create_footer_panel())
        
        return layout
    
    def create_header_panel(self, system_data: Dict[str, Any]) -> Panel:
        """Create header panel with system overview"""
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_time = system_data.get('avg_response_time_ms', 0)
        memory_usage = system_data.get('memory_usage_mb', 0)
        db_connections = system_data.get('database_connections', 0)
        
        # Status indicators
        response_status = "🟢" if response_time < 200 else "🟡" if response_time < 300 else "🔴"
        memory_status = "🟢" if memory_usage < 200 else "🟡" if memory_usage < 300 else "🔴"
        
        header_text = Text()
        header_text.append("APES Real-Time Monitoring", style="bold blue")
        header_text.append(f" | {current_time}", style="dim")
        header_text.append(f" | Response: {response_status} {response_time:.1f}ms", style="")
        header_text.append(f" | Memory: {memory_status} {memory_usage:.1f}MB", style="")
        header_text.append(f" | DB Connections: {db_connections}", style="")
        
        return Panel(header_text, border_style="blue")
    
    def create_performance_panel(self, perf_data: Dict[str, Any], system_data: Dict[str, Any]) -> Panel:
        """Create performance metrics panel"""
        
        table = Table(title="📊 Performance Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="magenta")
        table.add_column("Target", style="green")
        table.add_column("Status", justify="center")
        
        # Response time
        response_time = system_data.get('avg_response_time_ms', 0)
        response_status = "✅" if response_time < 200 else "⚠️" if response_time < 300 else "❌"
        table.add_row("Response Time", f"{response_time:.1f}ms", "< 200ms", response_status)
        
        # Cache hit ratio
        cache_ratio = perf_data.get('cache_hit_ratio', 0)
        cache_status = "✅" if cache_ratio > 90 else "⚠️" if cache_ratio > 80 else "❌"
        table.add_row("Cache Hit Ratio", f"{cache_ratio:.1f}%", "> 90%", cache_status)
        
        # Database connections
        db_conn = system_data.get('database_connections', 0)
        db_status = "✅" if db_conn < 15 else "⚠️" if db_conn < 20 else "❌"
        table.add_row("DB Connections", str(db_conn), "< 15", db_status)
        
        # Memory usage
        memory = system_data.get('memory_usage_mb', 0)
        memory_status = "✅" if memory < 200 else "⚠️" if memory < 300 else "❌"
        table.add_row("Memory Usage", f"{memory:.1f}MB", "< 200MB", memory_status)
        
        # Throughput (requests per minute)
        throughput = perf_data.get('throughput_rpm', 0)
        table.add_row("Throughput", f"{throughput:.0f} req/min", "-", "📈")
        
        return Panel(table, border_style="cyan")
    
    def create_effectiveness_panel(self, effectiveness_data: Dict[str, Any]) -> Panel:
        """Create rule effectiveness panel"""
        
        table = Table(title="🎯 Rule Effectiveness", box=box.ROUNDED)
        table.add_column("Rule", style="cyan")
        table.add_column("Effectiveness", style="magenta")
        table.add_column("Usage", style="yellow")
        table.add_column("Trend", justify="center")
        
        rules = effectiveness_data.get('rules', [])
        for rule in rules[:5]:  # Top 5 rules
            name = rule.get('rule_name', 'Unknown')[:15]  # Truncate long names
            effectiveness = rule.get('effectiveness_score', 0)
            usage_count = rule.get('usage_count', 0)
            trend = rule.get('trend', 'stable')
            
            trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
            
            table.add_row(
                name,
                f"{effectiveness:.1f}%",
                str(usage_count),
                trend_icon
            )
        
        return Panel(table, border_style="green")
    
    def create_satisfaction_panel(self, satisfaction_data: Dict[str, Any]) -> Panel:
        """Create user satisfaction panel"""
        
        table = Table(title="😊 User Satisfaction", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Change", style="yellow")
        
        avg_rating = satisfaction_data.get('average_rating', 0)
        total_feedback = satisfaction_data.get('total_feedback', 0)
        satisfaction_trend = satisfaction_data.get('trend', 0)
        
        trend_icon = "📈" if satisfaction_trend > 0 else "📉" if satisfaction_trend < 0 else "➡️"
        trend_text = f"{satisfaction_trend:+.1f}" if satisfaction_trend != 0 else "0.0"
        
        table.add_row("Average Rating", f"{avg_rating:.1f}/5.0", f"{trend_text} {trend_icon}")
        table.add_row("Total Feedback", str(total_feedback), "-")
        
        # Satisfaction distribution
        excellent = satisfaction_data.get('rating_distribution', {}).get('5', 0)
        good = satisfaction_data.get('rating_distribution', {}).get('4', 0)
        poor = satisfaction_data.get('rating_distribution', {}).get('1', 0) + satisfaction_data.get('rating_distribution', {}).get('2', 0)
        
        table.add_row("Excellent (5★)", str(excellent), "")
        table.add_row("Good (4★)", str(good), "")
        table.add_row("Poor (1-2★)", str(poor), "")
        
        return Panel(table, border_style="magenta")
    
    def create_alerts_panel(self) -> Panel:
        """Create active alerts panel"""
        
        if not self.active_alerts:
            content = Text("🟢 All systems operating normally", style="green")
            return Panel(content, title="🚨 Active Alerts", border_style="green")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Time", style="cyan")
        table.add_column("Alert", style="red")
        table.add_column("Value", style="yellow")
        
        # Show last 3 alerts
        for alert in self.active_alerts[-3:]:
            time_str = alert.timestamp.strftime("%H:%M:%S")
            severity_icon = "🔴" if alert.severity == "critical" else "🟡"
            
            table.add_row(
                time_str,
                f"{severity_icon} {alert.message}",
                f"{alert.current_value:.1f}"
            )
        
        return Panel(table, title="🚨 Active Alerts", border_style="red")
    
    def create_footer_panel(self) -> Panel:
        """Create footer with controls and status"""
        
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("Ctrl+C", style="red")
        controls.append(" to stop | ", style="")
        controls.append("Auto-refresh: 5s", style="dim")
        controls.append(" | ", style="")
        controls.append(f"Active Alerts: {len(self.active_alerts)}", style="yellow")
        
        return Panel(controls, border_style="dim")
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect real-time system metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'avg_response_time_ms': 0,
            'database_connections': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0
        }
        
        try:
            async with get_session() as session:
                # Test database response time
                start_time = time.time()
                await session.execute("SELECT 1")
                end_time = time.time()
                metrics['avg_response_time_ms'] = (end_time - start_time) * 1000
                
                # Get active database connections
                result = await session.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                metrics['database_connections'] = result.scalar() or 0
                
        except Exception as e:
            # Database unreachable
            metrics['avg_response_time_ms'] = 999
            metrics['database_connections'] = 0
        
        # Get system resource usage (mock for now, would use psutil in production)
        try:
            import psutil
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
            metrics['cpu_usage_percent'] = process.cpu_percent()
        except ImportError:
            # Fallback values
            metrics['memory_usage_mb'] = 50.0
            metrics['cpu_usage_percent'] = 5.0
        
        return metrics
    
    async def check_performance_alerts(self, performance_data: Dict[str, Any], system_metrics: Dict[str, Any]):
        """Check for performance threshold violations and generate alerts"""
        
        current_time = datetime.now()
        new_alerts = []
        
        # Check response time
        response_time = system_metrics.get('avg_response_time_ms', 0)
        if response_time > self.alert_thresholds.response_time_ms:
            severity = "critical" if response_time > 300 else "warning"
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="performance",
                metric_name="response_time",
                current_value=response_time,
                threshold_value=self.alert_thresholds.response_time_ms,
                severity=severity,
                message=f"High response time: {response_time:.1f}ms"
            )
            new_alerts.append(alert)
        
        # Check database connections
        db_connections = system_metrics.get('database_connections', 0)
        if db_connections > self.alert_thresholds.database_connections:
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="resource",
                metric_name="database_connections",
                current_value=db_connections,
                threshold_value=self.alert_thresholds.database_connections,
                severity="warning",
                message=f"High DB connections: {db_connections}"
            )
            new_alerts.append(alert)
        
        # Check memory usage
        memory_usage = system_metrics.get('memory_usage_mb', 0)
        if memory_usage > self.alert_thresholds.memory_usage_mb:
            severity = "critical" if memory_usage > 400 else "warning"
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="resource",
                metric_name="memory_usage",
                current_value=memory_usage,
                threshold_value=self.alert_thresholds.memory_usage_mb,
                severity=severity,
                message=f"High memory usage: {memory_usage:.1f}MB"
            )
            new_alerts.append(alert)
        
        # Check cache hit ratio
        cache_ratio = performance_data.get('cache_hit_ratio', 100)
        if cache_ratio < self.alert_thresholds.cache_hit_ratio:
            alert = PerformanceAlert(
                timestamp=current_time,
                alert_type="performance",
                metric_name="cache_hit_ratio",
                current_value=cache_ratio,
                threshold_value=self.alert_thresholds.cache_hit_ratio,
                severity="warning",
                message=f"Low cache hit ratio: {cache_ratio:.1f}%"
            )
            new_alerts.append(alert)
        
        # Add new alerts and maintain alert history (keep last 10)
        self.active_alerts.extend(new_alerts)
        if len(self.active_alerts) > 10:
            self.active_alerts = self.active_alerts[-10:]
        
        # Log alerts for persistence
        for alert in new_alerts:
            await self.log_alert(alert)
    
    async def log_alert(self, alert: PerformanceAlert):
        """Log alert to database for historical analysis"""
        
        try:
            async with get_session() as session:
                # Store alert in rule performance table as monitoring entry
                perf_metric = RulePerformance(
                    rule_id="monitoring_alert",
                    rule_name=f"{alert.alert_type}_alert",
                    improvement_score=0.0,  # Alert indicates a problem
                    confidence_level=1.0,
                    execution_time_ms=int(alert.current_value) if alert.metric_name == "response_time" else None,
                    rule_parameters={"alert_type": alert.alert_type, "severity": alert.severity, "threshold": alert.threshold_value},
                    before_metrics={"current_value": alert.current_value},
                    after_metrics={"threshold": alert.threshold_value},
                    prompt_characteristics={"monitoring": True}
                )
                
                session.add(perf_metric)
                await session.commit()
                
        except Exception as e:
            # Don't let logging errors break monitoring
            pass
    
    def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        self.monitoring_active = False
    
    async def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get monitoring summary for the specified time period"""
        
        try:
            # Get performance trends
            performance_data = await self.analytics.get_performance_trends(days=max(1, hours // 24))
            
            # Get current system metrics
            current_metrics = await self.collect_system_metrics()
            
            # Calculate alert statistics
            recent_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [a for a in self.active_alerts if a.timestamp >= recent_time]
            
            alert_summary = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in recent_alerts if a.severity == 'warning']),
                'most_common_alert': self._get_most_common_alert_type(recent_alerts)
            }
            
            return {
                'monitoring_period_hours': hours,
                'timestamp': datetime.now().isoformat(),
                'current_performance': current_metrics,
                'performance_trends': performance_data,
                'alert_summary': alert_summary,
                'health_status': self._calculate_health_status(current_metrics, recent_alerts)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'monitoring_period_hours': hours
            }
    
    def _get_most_common_alert_type(self, alerts: List[PerformanceAlert]) -> Optional[str]:
        """Get the most common alert type from a list of alerts"""
        
        if not alerts:
            return None
        
        alert_counts = {}
        for alert in alerts:
            alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type, 0) + 1
        
        return max(alert_counts, key=alert_counts.get)
    
    def _calculate_health_status(self, current_metrics: Dict[str, Any], recent_alerts: List[PerformanceAlert]) -> str:
        """Calculate overall system health status"""
        
        response_time = current_metrics.get('avg_response_time_ms', 0)
        memory_usage = current_metrics.get('memory_usage_mb', 0)
        db_connections = current_metrics.get('database_connections', 0)
        
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        
        # Determine health status
        if critical_alerts or response_time > 300 or memory_usage > 400:
            return 'critical'
        elif response_time > 200 or memory_usage > 256 or db_connections > 15:
            return 'warning'
        else:
            return 'healthy'


class HealthMonitor:
    """System health monitoring with automated diagnostics"""
    
    def __init__(self):
        self.analytics = AnalyticsService()
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Database connectivity check
        db_status = await self._check_database_health()
        health_results['checks']['database'] = db_status
        
        # MCP server performance check
        mcp_status = await self._check_mcp_performance()
        health_results['checks']['mcp_server'] = mcp_status
        
        # Analytics service check
        analytics_status = await self._check_analytics_service()
        health_results['checks']['analytics'] = analytics_status
        
        # ML service check
        ml_status = await self._check_ml_service()
        health_results['checks']['ml_service'] = ml_status
        
        # System resources check
        system_status = await self._check_system_resources()
        health_results['checks']['system_resources'] = system_status
        
        # Determine overall status
        failed_checks = [name for name, check in health_results['checks'].items() 
                        if check.get('status') == 'failed']
        warning_checks = [name for name, check in health_results['checks'].items() 
                         if check.get('status') == 'warning']
        
        if failed_checks:
            health_results['overall_status'] = 'failed'
            health_results['failed_checks'] = failed_checks
        elif warning_checks:
            health_results['overall_status'] = 'warning'
            health_results['warning_checks'] = warning_checks
        
        return health_results
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        
        try:
            start_time = time.time()
            async with get_session() as session:
                await session.execute("SELECT 1")
                response_time = (time.time() - start_time) * 1000
                
                # Check for long-running queries
                result = await session.execute("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < NOW() - INTERVAL '30 seconds'
                """)
                long_queries = result.scalar() or 0
                
                status = 'healthy'
                if response_time > 100:
                    status = 'warning'
                if response_time > 500 or long_queries > 0:
                    status = 'failed'
                
                return {
                    'status': status,
                    'response_time_ms': response_time,
                    'long_running_queries': long_queries,
                    'message': f"Database responding in {response_time:.1f}ms"
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Database connection failed'
            }
    
    async def _check_mcp_performance(self) -> Dict[str, Any]:
        """Check MCP server performance"""
        
        try:
            from ..mcp_server.mcp_server import improve_prompt
            
            start_time = time.time()
            result = await improve_prompt(
                prompt="Health check test prompt",
                context={"domain": "health_check"},
                session_id="health_check"
            )
            response_time = (time.time() - start_time) * 1000
            
            status = 'healthy'
            if response_time > 200:
                status = 'warning'
            if response_time > 500:
                status = 'failed'
            
            return {
                'status': status,
                'response_time_ms': response_time,
                'message': f"MCP server responding in {response_time:.1f}ms"
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'MCP server health check failed'
            }
    
    async def _check_analytics_service(self) -> Dict[str, Any]:
        """Check analytics service functionality"""
        
        try:
            start_time = time.time()
            result = await self.analytics.get_performance_trends(days=1)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'data_points': len(result.get('trends', [])),
                'message': f"Analytics service responding in {response_time:.1f}ms"
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Analytics service check failed'
            }
    
    async def _check_ml_service(self) -> Dict[str, Any]:
        """Check ML service availability"""
        
        try:
            from .ml_integration import get_ml_service
            
            start_time = time.time()
            ml_service = await get_ml_service()
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'message': f"ML service available in {response_time:.1f}ms"
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'error': str(e),
                'message': 'ML service unavailable (fallback to rule-based)'
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = 'healthy'
            warnings = []
            
            if memory_usage_percent > 80:
                status = 'warning'
                warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")
            
            if disk_usage_percent > 85:
                status = 'warning'
                warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")
            
            if cpu_percent > 80:
                status = 'warning'
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            return {
                'status': status,
                'memory_usage_percent': memory_usage_percent,
                'disk_usage_percent': disk_usage_percent,
                'cpu_usage_percent': cpu_percent,
                'warnings': warnings,
                'message': f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_usage_percent:.1f}%, Disk {disk_usage_percent:.1f}%"
            }
            
        except ImportError:
            return {
                'status': 'warning',
                'message': 'psutil not available for system monitoring'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'System resource check failed'
            }