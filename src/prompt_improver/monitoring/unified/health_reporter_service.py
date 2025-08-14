"""Health Reporter Service.

Focused service for health reporting, dashboard integration, and status aggregation.
Extracted from unified_monitoring_manager.py.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    health_reporter_tracer = trace.get_tracer(__name__ + ".health_reporter")
    health_reporter_meter = metrics.get_meter(__name__ + ".health_reporter")
    
    health_reports_generated = health_reporter_meter.create_counter(
        "health_reports_generated_total",
        description="Total health reports generated",
        unit="1",
    )
    
    health_report_generation_duration = health_reporter_meter.create_histogram(
        "health_report_generation_duration_seconds",
        description="Time taken to generate health reports",
        unit="s",
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    health_reporter_tracer = None
    health_reporter_meter = None
    health_reports_generated = None
    health_report_generation_duration = None

from .protocols import HealthReporterProtocol, MonitoringRepositoryProtocol
from .types import (
    HealthCheckResult, 
    HealthStatus, 
    MonitoringConfig, 
    SystemHealthSummary
)

logger = logging.getLogger(__name__)


class HealthReporterService:
    """Service for health reporting and status aggregation.
    
    Provides:
    - Comprehensive health report generation
    - Historical health trend analysis
    - Dashboard-friendly health status summaries
    - Health metric aggregation and rollups
    - Custom health report formatting
    """
    
    def __init__(
        self,
        config: MonitoringConfig,
        repository: Optional[MonitoringRepositoryProtocol] = None,
    ):
        self.config = config
        self.repository = repository
        
        # Report caching
        self._cached_reports: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_ttl = 60  # 1 minute cache TTL
        
        # Health trend tracking
        self._health_history: List[SystemHealthSummary] = []
        self._max_history_size = 1000
        
        # Custom formatters
        self._custom_formatters: Dict[str, callable] = {}
        
        logger.info("HealthReporterService initialized")
    
    async def generate_comprehensive_health_report(
        self,
        include_history: bool = True,
        include_trends: bool = True,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"comprehensive_health_{include_history}_{include_trends}_{include_recommendations}"
            if self._is_cached_report_valid(cache_key):
                logger.debug("Using cached comprehensive health report")
                return self._cached_reports[cache_key]
            
            report = {
                "report_type": "comprehensive_health",
                "generated_at": datetime.now(UTC).isoformat(),
                "generated_by": "health_reporter_service",
            }
            
            # Get current health status
            current_health = await self._get_current_health_status()
            report["current_status"] = current_health
            
            # Add historical data if requested
            if include_history:
                report["historical_data"] = await self._get_health_history_summary()
            
            # Add trend analysis if requested
            if include_trends:
                report["trend_analysis"] = await self._generate_health_trends()
            
            # Add recommendations if requested
            if include_recommendations:
                report["recommendations"] = await self._generate_health_recommendations(current_health)
            
            # Add component details
            report["component_details"] = await self._get_detailed_component_status()
            
            # Add performance metrics
            report["performance_metrics"] = await self._get_health_performance_metrics()
            
            # Cache the report
            self._cache_report(cache_key, report)
            
            # Record telemetry
            duration = time.time() - start_time
            if OPENTELEMETRY_AVAILABLE and health_reports_generated:
                health_reports_generated.add(1, {"type": "comprehensive"})
            
            if OPENTELEMETRY_AVAILABLE and health_report_generation_duration:
                health_report_generation_duration.record(
                    duration,
                    {"type": "comprehensive"}
                )
            
            logger.info(f"Generated comprehensive health report in {duration:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive health report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(UTC).isoformat(),
                "report_type": "comprehensive_health_error",
            }
    
    async def generate_dashboard_summary(self) -> Dict[str, Any]:
        """Generate dashboard-friendly health summary."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = "dashboard_summary"
            if self._is_cached_report_valid(cache_key):
                logger.debug("Using cached dashboard summary")
                return self._cached_reports[cache_key]
            
            # Get current health
            current_health = await self._get_current_health_status()
            
            # Calculate key metrics
            total_components = current_health.get("total_components", 0)
            healthy_components = current_health.get("healthy_components", 0)
            health_percentage = (
                (healthy_components / total_components * 100) if total_components > 0 else 0
            )
            
            # Determine overall status color/indicator
            overall_status = current_health.get("overall_status", "unknown")
            status_indicator = self._get_status_indicator(overall_status)
            
            summary = {
                "overall_status": overall_status,
                "status_indicator": status_indicator,
                "health_percentage": round(health_percentage, 1),
                "total_components": total_components,
                "healthy_components": healthy_components,
                "unhealthy_components": current_health.get("unhealthy_components", 0),
                "degraded_components": current_health.get("degraded_components", 0),
                "last_check_duration_ms": current_health.get("check_duration_ms", 0),
                "last_updated": datetime.now(UTC).isoformat(),
                "critical_issues": current_health.get("critical_issues", []),
                "trending": await self._get_health_trend_indicator(),
            }
            
            # Add quick component status breakdown
            summary["component_breakdown"] = await self._get_component_status_breakdown()
            
            # Cache the summary
            self._cache_report(cache_key, summary)
            
            # Record telemetry
            duration = time.time() - start_time
            if OPENTELEMETRY_AVAILABLE and health_reports_generated:
                health_reports_generated.add(1, {"type": "dashboard"})
            
            logger.debug(f"Generated dashboard summary in {duration:.2f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard summary: {e}")
            return {
                "overall_status": "error",
                "status_indicator": "error",
                "error": str(e),
                "last_updated": datetime.now(UTC).isoformat(),
            }
    
    async def generate_component_health_report(
        self,
        component_name: str,
        include_history: bool = True,
    ) -> Dict[str, Any]:
        """Generate health report for specific component."""
        start_time = time.time()
        
        try:
            cache_key = f"component_health_{component_name}_{include_history}"
            if self._is_cached_report_valid(cache_key):
                logger.debug(f"Using cached component health report for {component_name}")
                return self._cached_reports[cache_key]
            
            report = {
                "component_name": component_name,
                "report_type": "component_health",
                "generated_at": datetime.now(UTC).isoformat(),
            }
            
            # Get current component status
            component_status = await self._get_component_status(component_name)
            if not component_status:
                report["error"] = "Component not found"
                return report
            
            report["current_status"] = component_status
            
            # Add historical data if requested
            if include_history:
                report["historical_data"] = await self._get_component_history(component_name)
                report["trend_analysis"] = await self._analyze_component_trends(component_name)
            
            # Add component-specific metrics
            report["metrics"] = await self._get_component_metrics(component_name)
            
            # Add recommendations
            report["recommendations"] = await self._generate_component_recommendations(
                component_name, component_status
            )
            
            # Cache the report
            self._cache_report(cache_key, report)
            
            # Record telemetry
            duration = time.time() - start_time
            if OPENTELEMETRY_AVAILABLE and health_reports_generated:
                health_reports_generated.add(1, {"type": "component"})
            
            logger.debug(f"Generated component health report for {component_name} in {duration:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate component health report for {component_name}: {e}")
            return {
                "component_name": component_name,
                "error": str(e),
                "generated_at": datetime.now(UTC).isoformat(),
                "report_type": "component_health_error",
            }
    
    async def generate_health_trends_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate health trends report for specified time window."""
        start_time = time.time()
        
        try:
            cache_key = f"health_trends_{hours}"
            if self._is_cached_report_valid(cache_key):
                logger.debug(f"Using cached health trends report for {hours}h")
                return self._cached_reports[cache_key]
            
            report = {
                "report_type": "health_trends",
                "time_window_hours": hours,
                "generated_at": datetime.now(UTC).isoformat(),
            }
            
            # Get historical health data
            historical_data = await self._get_health_history(hours)
            
            if not historical_data:
                report["message"] = "No historical data available"
                return report
            
            # Analyze trends
            trends = self._analyze_health_trends(historical_data)
            report["trends"] = trends
            
            # Calculate key statistics
            report["statistics"] = self._calculate_health_statistics(historical_data)
            
            # Identify patterns and anomalies
            report["patterns"] = await self._identify_health_patterns(historical_data)
            report["anomalies"] = await self._identify_health_anomalies(historical_data)
            
            # Add forecasting if enough data
            if len(historical_data) >= 24:  # At least 24 data points
                report["forecast"] = await self._generate_health_forecast(historical_data)
            
            # Cache the report
            self._cache_report(cache_key, report)
            
            # Record telemetry
            duration = time.time() - start_time
            if OPENTELEMETRY_AVAILABLE and health_reports_generated:
                health_reports_generated.add(1, {"type": "trends"})
            
            logger.info(f"Generated health trends report in {duration:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate health trends report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(UTC).isoformat(),
                "report_type": "health_trends_error",
            }
    
    async def _get_current_health_status(self) -> Dict[str, Any]:
        """Get current health status from repository or live checks."""
        try:
            if self.repository:
                # Try to get recent health data from repository
                recent_health = await self.repository.get_recent_health_data(minutes=5)
                if recent_health:
                    return recent_health
            
            # Fallback: return basic status
            return {
                "overall_status": "unknown",
                "total_components": 0,
                "healthy_components": 0,
                "degraded_components": 0,
                "unhealthy_components": 0,
                "unknown_components": 0,
                "check_duration_ms": 0,
                "critical_issues": [],
            }
            
        except Exception as e:
            logger.error(f"Failed to get current health status: {e}")
            return {"error": str(e)}
    
    async def _get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get current status for specific component."""
        try:
            if self.repository:
                return await self.repository.get_component_health(component_name)
            return None
        except Exception as e:
            logger.error(f"Failed to get component status for {component_name}: {e}")
            return None
    
    async def _get_health_history_summary(self) -> Dict[str, Any]:
        """Get summary of health history."""
        try:
            if not self._health_history:
                return {"message": "No historical data available"}
            
            # Calculate summary statistics from recent history
            recent_history = self._health_history[-100:]  # Last 100 entries
            
            return {
                "data_points": len(recent_history),
                "avg_health_percentage": sum(
                    h.health_percentage for h in recent_history
                ) / len(recent_history),
                "avg_check_duration_ms": sum(
                    h.check_duration_ms for h in recent_history
                ) / len(recent_history),
                "status_distribution": self._calculate_status_distribution(recent_history),
            }
            
        except Exception as e:
            logger.error(f"Failed to get health history summary: {e}")
            return {"error": str(e)}
    
    async def _generate_health_trends(self) -> Dict[str, Any]:
        """Generate health trend analysis."""
        try:
            if len(self._health_history) < 2:
                return {"message": "Insufficient data for trend analysis"}
            
            recent = self._health_history[-10:]  # Last 10 entries
            older = self._health_history[-20:-10] if len(self._health_history) >= 20 else []
            
            trends = {}
            
            # Health percentage trend
            if recent and older:
                recent_avg = sum(h.health_percentage for h in recent) / len(recent)
                older_avg = sum(h.health_percentage for h in older) / len(older)
                trends["health_percentage_trend"] = "improving" if recent_avg > older_avg else "declining"
                trends["health_percentage_change"] = recent_avg - older_avg
            
            # Response time trend
            if recent and older:
                recent_avg_time = sum(h.check_duration_ms for h in recent) / len(recent)
                older_avg_time = sum(h.check_duration_ms for h in older) / len(older)
                trends["response_time_trend"] = "improving" if recent_avg_time < older_avg_time else "declining"
                trends["response_time_change_ms"] = recent_avg_time - older_avg_time
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to generate health trends: {e}")
            return {"error": str(e)}
    
    async def _generate_health_recommendations(self, current_health: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        try:
            overall_status = current_health.get("overall_status", "unknown")
            health_percentage = current_health.get("health_percentage", 0)
            unhealthy_components = current_health.get("unhealthy_components", 0)
            degraded_components = current_health.get("degraded_components", 0)
            
            # General recommendations based on overall health
            if overall_status == "unhealthy":
                recommendations.append("URGENT: Investigate unhealthy components immediately")
                recommendations.append("Consider implementing emergency procedures")
                
            elif overall_status == "degraded":
                recommendations.append("Review degraded components for potential issues")
                recommendations.append("Monitor closely for further degradation")
                
            # Specific recommendations based on metrics
            if health_percentage < 70:
                recommendations.append("Health percentage is below 70% - investigate failing components")
                
            if unhealthy_components > 0:
                recommendations.append(f"Address {unhealthy_components} unhealthy component(s)")
                
            if degraded_components > 2:
                recommendations.append("Multiple degraded components detected - system stability may be at risk")
                
            # Performance recommendations
            check_duration = current_health.get("check_duration_ms", 0)
            if check_duration > 5000:  # 5 seconds
                recommendations.append("Health check duration is high - consider optimizing health check implementations")
                
            # Default recommendation if system is healthy
            if not recommendations and overall_status == "healthy":
                recommendations.append("System health is good - continue regular monitoring")
            
        except Exception as e:
            logger.error(f"Failed to generate health recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations
    
    def _get_status_indicator(self, status: str) -> str:
        """Get status indicator for dashboard display."""
        status_indicators = {
            "healthy": "success",
            "degraded": "warning", 
            "unhealthy": "danger",
            "unknown": "info",
            "error": "danger",
        }
        return status_indicators.get(status.lower(), "info")
    
    async def _get_health_trend_indicator(self) -> str:
        """Get simple trend indicator for dashboard."""
        try:
            if len(self._health_history) < 5:
                return "stable"
            
            recent = self._health_history[-5:]
            health_percentages = [h.health_percentage for h in recent]
            
            # Simple trend analysis
            if health_percentages[-1] > health_percentages[0]:
                return "improving"
            elif health_percentages[-1] < health_percentages[0]:
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    async def _get_component_status_breakdown(self) -> Dict[str, int]:
        """Get component status breakdown for dashboard."""
        try:
            current_health = await self._get_current_health_status()
            
            return {
                "healthy": current_health.get("healthy_components", 0),
                "degraded": current_health.get("degraded_components", 0),
                "unhealthy": current_health.get("unhealthy_components", 0),
                "unknown": current_health.get("unknown_components", 0),
            }
            
        except Exception as e:
            logger.error(f"Failed to get component status breakdown: {e}")
            return {"error": 1}
    
    def _is_cached_report_valid(self, cache_key: str) -> bool:
        """Check if cached report is still valid."""
        if cache_key not in self._cached_reports:
            return False
        
        expiry_time = self._cache_expiry.get(cache_key, 0)
        return time.time() < expiry_time
    
    def _cache_report(self, cache_key: str, report: Dict[str, Any]) -> None:
        """Cache a report with TTL."""
        self._cached_reports[cache_key] = report
        self._cache_expiry[cache_key] = time.time() + self._cache_ttl
    
    def add_custom_formatter(self, format_name: str, formatter: callable) -> None:
        """Add custom report formatter."""
        self._custom_formatters[format_name] = formatter
        logger.info(f"Added custom formatter: {format_name}")
    
    async def format_report(self, report: Dict[str, Any], format_name: str) -> Any:
        """Format report using custom formatter."""
        if format_name not in self._custom_formatters:
            return report
        
        try:
            formatter = self._custom_formatters[format_name]
            if asyncio.iscoroutinefunction(formatter):
                return await formatter(report)
            else:
                return formatter(report)
        except Exception as e:
            logger.error(f"Custom formatter {format_name} failed: {e}")
            return report
    
    def add_health_summary(self, summary: SystemHealthSummary) -> None:
        """Add health summary to history tracking."""
        self._health_history.append(summary)
        
        # Maintain history size limit
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size:]
    
    def clear_cache(self) -> None:
        """Clear all cached reports."""
        self._cached_reports.clear()
        self._cache_expiry.clear()
        logger.debug("Cleared report cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_reports": len(self._cached_reports),
            "cache_hit_rate": "not_implemented",  # Would need hit/miss tracking
            "cache_ttl_seconds": self._cache_ttl,
        }