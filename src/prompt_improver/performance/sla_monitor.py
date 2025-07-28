"""SLA Monitoring and Performance Tracking System.

Implements real-time SLA monitoring with 95th percentile response time tracking
and automatic performance optimization triggers.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import statistics

import coredis

logger = logging.getLogger(__name__)

class SLAStatus(str, Enum):
    """SLA compliance status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"

@dataclass
class SLAMetrics:
    """SLA performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    sla_violations: int = 0
    sla_compliance_rate: float = 1.0
    current_status: SLAStatus = SLAStatus.HEALTHY
    last_updated: float = field(default_factory=time.time)

@dataclass
class RequestMetrics:
    """Individual request metrics."""
    request_id: str
    endpoint: str
    response_time_ms: float
    success: bool
    timestamp: float
    agent_type: str
    error_type: Optional[str] = None

class SLAMonitor:
    """Real-time SLA monitoring system with performance optimization.
    
    Features:
    - <200ms SLA enforcement with 95% compliance target
    - Real-time 95th percentile response time tracking
    - Automatic performance degradation detection
    - Redis-based metrics aggregation for distributed monitoring
    - Configurable alerting thresholds
    """
    
    def __init__(
        self,
        sla_target_ms: float = 200.0,
        compliance_target: float = 0.95,
        redis_url: str = "redis://localhost:6379/5"
    ):
        """Initialize SLA monitor.
        
        Args:
            sla_target_ms: SLA target response time in milliseconds
            compliance_target: Target compliance rate (0.95 = 95%)
            redis_url: Redis URL for distributed metrics
        """
        self.sla_target_ms = sla_target_ms
        self.compliance_target = compliance_target
        self.redis_url = redis_url
        self._redis_client = None
        
        # In-memory metrics storage
        self._response_times = deque(maxlen=1000)  # Last 1000 requests
        self._request_history = deque(maxlen=100)   # Last 100 detailed requests
        
        # Current metrics
        self._current_metrics = SLAMetrics()
        
        # Performance thresholds
        self.warning_threshold_ms = sla_target_ms * 0.8  # 160ms
        self.critical_threshold_ms = sla_target_ms * 1.2  # 240ms
        
        # Monitoring configuration
        self.monitoring_window_seconds = 300  # 5 minutes
        self.metrics_update_interval = 10     # 10 seconds
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Background monitoring task
        self._monitoring_task = None
        self._monitoring_enabled = True

    async def get_redis_client(self) -> coredis.Redis:
        """Get Redis client for distributed metrics."""
        if self._redis_client is None:
            self._redis_client = coredis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def record_request(
        self,
        request_id: str,
        endpoint: str,
        response_time_ms: float,
        success: bool,
        agent_type: str = "unknown",
        error_type: Optional[str] = None
    ) -> None:
        """Record request metrics for SLA monitoring.
        
        Args:
            request_id: Unique request identifier
            endpoint: API endpoint or operation name
            response_time_ms: Response time in milliseconds
            success: Whether request succeeded
            agent_type: Type of agent making request
            error_type: Type of error if request failed
        """
        # Create request metrics
        request_metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            response_time_ms=response_time_ms,
            success=success,
            timestamp=time.time(),
            agent_type=agent_type,
            error_type=error_type
        )
        
        # Store in memory
        self._response_times.append(response_time_ms)
        self._request_history.append(request_metrics)
        
        # Update current metrics
        await self._update_current_metrics(request_metrics)
        
        # Store in Redis for distributed monitoring
        await self._store_metrics_in_redis(request_metrics)
        
        # Check for SLA violations and alerts
        await self._check_sla_compliance(request_metrics)

    async def _update_current_metrics(self, request_metrics: RequestMetrics) -> None:
        """Update current SLA metrics.
        
        Args:
            request_metrics: Latest request metrics
        """
        metrics = self._current_metrics
        
        # Update counters
        metrics.total_requests += 1
        if request_metrics.success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update response time statistics
        if self._response_times:
            response_times = list(self._response_times)
            metrics.avg_response_time_ms = statistics.mean(response_times)
            
            if len(response_times) >= 2:
                sorted_times = sorted(response_times)
                n = len(sorted_times)
                
                # Calculate percentiles
                metrics.p50_response_time_ms = sorted_times[int(0.50 * n)]
                metrics.p95_response_time_ms = sorted_times[int(0.95 * n)]
                metrics.p99_response_time_ms = sorted_times[int(0.99 * n)]
        
        # Check for SLA violation
        if request_metrics.response_time_ms > self.sla_target_ms:
            metrics.sla_violations += 1
        
        # Calculate compliance rate
        metrics.sla_compliance_rate = (
            (metrics.total_requests - metrics.sla_violations) / 
            max(1, metrics.total_requests)
        )
        
        # Update status
        metrics.current_status = self._determine_sla_status(metrics)
        metrics.last_updated = time.time()

    def _determine_sla_status(self, metrics: SLAMetrics) -> SLAStatus:
        """Determine current SLA status based on metrics.
        
        Args:
            metrics: Current SLA metrics
            
        Returns:
            SLA status level
        """
        # Check compliance rate
        if metrics.sla_compliance_rate < 0.90:  # Below 90%
            return SLAStatus.CRITICAL
        elif metrics.sla_compliance_rate < self.compliance_target:  # Below 95%
            return SLAStatus.WARNING
        
        # Check P95 response time
        if metrics.p95_response_time_ms > self.critical_threshold_ms:
            return SLAStatus.CRITICAL
        elif metrics.p95_response_time_ms > self.warning_threshold_ms:
            return SLAStatus.WARNING
        
        # Check recent trend
        if len(self._response_times) >= 10:
            recent_times = list(self._response_times)[-10:]
            recent_avg = statistics.mean(recent_times)
            
            if recent_avg > self.critical_threshold_ms:
                return SLAStatus.DEGRADED
            elif recent_avg > self.warning_threshold_ms:
                return SLAStatus.WARNING
        
        return SLAStatus.HEALTHY

    async def _check_sla_compliance(self, request_metrics: RequestMetrics) -> None:
        """Check SLA compliance and trigger alerts if needed.
        
        Args:
            request_metrics: Latest request metrics
        """
        current_status = self._current_metrics.current_status
        
        # Trigger alerts for status changes or violations
        if current_status in [SLAStatus.WARNING, SLAStatus.CRITICAL, SLAStatus.DEGRADED]:
            await self._trigger_alerts(current_status, request_metrics)
        
        # Log SLA violations
        if request_metrics.response_time_ms > self.sla_target_ms:
            logger.warning(
                f"SLA violation: {request_metrics.endpoint} took {request_metrics.response_time_ms:.1f}ms "
                f"(target: {self.sla_target_ms}ms, agent: {request_metrics.agent_type})"
            )

    async def _store_metrics_in_redis(self, request_metrics: RequestMetrics) -> None:
        """Store metrics in Redis for distributed monitoring.
        
        Args:
            request_metrics: Request metrics to store
        """
        try:
            redis = await self.get_redis_client()
            
            # Store individual request
            request_key = f"sla:request:{request_metrics.request_id}"
            request_data = {
                "endpoint": request_metrics.endpoint,
                "response_time_ms": request_metrics.response_time_ms,
                "success": int(request_metrics.success),
                "agent_type": request_metrics.agent_type,
                "timestamp": request_metrics.timestamp
            }
            
            await redis.hset(request_key, mapping=request_data)
            await redis.expire(request_key, self.monitoring_window_seconds)
            
            # Update aggregated metrics
            metrics_key = "sla:metrics:current"
            await redis.hset(metrics_key, mapping={
                "total_requests": self._current_metrics.total_requests,
                "successful_requests": self._current_metrics.successful_requests,
                "failed_requests": self._current_metrics.failed_requests,
                "avg_response_time_ms": self._current_metrics.avg_response_time_ms,
                "p95_response_time_ms": self._current_metrics.p95_response_time_ms,
                "sla_violations": self._current_metrics.sla_violations,
                "sla_compliance_rate": self._current_metrics.sla_compliance_rate,
                "current_status": self._current_metrics.current_status.value,
                "last_updated": self._current_metrics.last_updated
            })
            
        except Exception as e:
            logger.warning(f"Failed to store SLA metrics in Redis: {e}")

    async def _trigger_alerts(self, status: SLAStatus, request_metrics: RequestMetrics) -> None:
        """Trigger alerts for SLA violations.
        
        Args:
            status: Current SLA status
            request_metrics: Request that triggered the alert
        """
        alert_data = {
            "status": status.value,
            "endpoint": request_metrics.endpoint,
            "response_time_ms": request_metrics.response_time_ms,
            "p95_response_time_ms": self._current_metrics.p95_response_time_ms,
            "compliance_rate": self._current_metrics.sla_compliance_rate,
            "timestamp": time.time()
        }
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for SLA alerts.
        
        Args:
            callback: Async function to call on alerts
        """
        self._alert_callbacks.append(callback)

    async def get_current_metrics(self) -> SLAMetrics:
        """Get current SLA metrics.
        
        Returns:
            Current SLA metrics
        """
        return self._current_metrics

    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed SLA metrics and analysis.
        
        Returns:
            Detailed metrics dictionary
        """
        metrics = self._current_metrics
        
        # Analyze request patterns
        endpoint_stats = {}
        agent_stats = {}
        
        for request in self._request_history:
            # Endpoint statistics
            if request.endpoint not in endpoint_stats:
                endpoint_stats[request.endpoint] = {
                    "count": 0,
                    "avg_time": 0.0,
                    "violations": 0
                }
            
            endpoint_stats[request.endpoint]["count"] += 1
            endpoint_stats[request.endpoint]["avg_time"] += request.response_time_ms
            if request.response_time_ms > self.sla_target_ms:
                endpoint_stats[request.endpoint]["violations"] += 1
            
            # Agent statistics
            if request.agent_type not in agent_stats:
                agent_stats[request.agent_type] = {
                    "count": 0,
                    "avg_time": 0.0,
                    "violations": 0
                }
            
            agent_stats[request.agent_type]["count"] += 1
            agent_stats[request.agent_type]["avg_time"] += request.response_time_ms
            if request.response_time_ms > self.sla_target_ms:
                agent_stats[request.agent_type]["violations"] += 1
        
        # Calculate averages
        for stats in endpoint_stats.values():
            stats["avg_time"] /= max(1, stats["count"])
        
        for stats in agent_stats.values():
            stats["avg_time"] /= max(1, stats["count"])
        
        return {
            "current_metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.successful_requests / max(1, metrics.total_requests),
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "p50_response_time_ms": metrics.p50_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "p99_response_time_ms": metrics.p99_response_time_ms,
                "sla_violations": metrics.sla_violations,
                "sla_compliance_rate": metrics.sla_compliance_rate,
                "current_status": metrics.current_status.value
            },
            "sla_configuration": {
                "target_ms": self.sla_target_ms,
                "compliance_target": self.compliance_target,
                "warning_threshold_ms": self.warning_threshold_ms,
                "critical_threshold_ms": self.critical_threshold_ms
            },
            "endpoint_analysis": endpoint_stats,
            "agent_analysis": agent_stats,
            "performance_assessment": {
                "sla_compliant": metrics.sla_compliance_rate >= self.compliance_target,
                "p95_compliant": metrics.p95_response_time_ms <= self.sla_target_ms,
                "overall_status": metrics.current_status.value,
                "recommendations": self._generate_recommendations(metrics)
            }
        }

    def _generate_recommendations(self, metrics: SLAMetrics) -> List[str]:
        """Generate performance optimization recommendations.
        
        Args:
            metrics: Current SLA metrics
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if metrics.sla_compliance_rate < self.compliance_target:
            recommendations.append(
                f"SLA compliance below target ({metrics.sla_compliance_rate:.2f} < {self.compliance_target}). "
                "Consider query optimization or caching improvements."
            )
        
        if metrics.p95_response_time_ms > self.sla_target_ms:
            recommendations.append(
                f"P95 response time exceeds SLA ({metrics.p95_response_time_ms:.1f}ms > {self.sla_target_ms}ms). "
                "Review slow queries and database indexing."
            )
        
        if metrics.failed_requests > metrics.total_requests * 0.05:  # >5% error rate
            recommendations.append(
                f"High error rate detected ({metrics.failed_requests}/{metrics.total_requests}). "
                "Investigate error patterns and system stability."
            )
        
        if metrics.avg_response_time_ms > self.warning_threshold_ms:
            recommendations.append(
                f"Average response time elevated ({metrics.avg_response_time_ms:.1f}ms). "
                "Consider prepared statement caching and connection pool tuning."
            )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters. Continue monitoring.")
        
        return recommendations

    async def reset_metrics(self) -> None:
        """Reset all SLA metrics (for testing or maintenance)."""
        self._response_times.clear()
        self._request_history.clear()
        self._current_metrics = SLAMetrics()
        
        try:
            redis = await self.get_redis_client()
            await redis.delete("sla:metrics:current")
        except Exception as e:
            logger.warning(f"Failed to reset Redis metrics: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for SLA monitoring system.
        
        Returns:
            Health status information
        """
        metrics = await self.get_current_metrics()
        
        health_issues = []
        
        if metrics.current_status == SLAStatus.CRITICAL:
            health_issues.append("SLA monitoring in critical state")
        elif metrics.current_status == SLAStatus.DEGRADED:
            health_issues.append("Performance degradation detected")
        
        if metrics.sla_compliance_rate < 0.90:
            health_issues.append(f"SLA compliance critically low: {metrics.sla_compliance_rate:.2f}")
        
        try:
            redis = await self.get_redis_client()
            await redis.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False
            health_issues.append("Redis connection failed")
        
        overall_status = "healthy" if not health_issues else "degraded"
        
        return {
            "status": overall_status,
            "issues": health_issues,
            "sla_status": metrics.current_status.value,
            "compliance_rate": metrics.sla_compliance_rate,
            "p95_response_time_ms": metrics.p95_response_time_ms,
            "redis_connected": redis_healthy,
            "monitoring_enabled": self._monitoring_enabled,
            "timestamp": time.time()
        }
