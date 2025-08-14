"""Cache Monitoring Service.

Focused service for comprehensive cache monitoring across L1/L2/L3 levels.
Migrated from monitoring/cache/ components.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    cache_monitoring_tracer = trace.get_tracer(__name__ + ".cache_monitoring")
    cache_monitoring_meter = metrics.get_meter(__name__ + ".cache_monitoring")
    
    cache_operations_monitored = cache_monitoring_meter.create_counter(
        "cache_operations_monitored_total",
        description="Total cache operations monitored",
        unit="1",
    )
    
    cache_monitoring_duration = cache_monitoring_meter.create_histogram(
        "cache_monitoring_duration_seconds",
        description="Cache monitoring operation duration",
        unit="s",
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    cache_monitoring_tracer = None
    cache_monitoring_meter = None
    cache_operations_monitored = None
    cache_monitoring_duration = None

from prompt_improver.core.types import AccessPattern, CacheLevel, CoordinationAction, InvalidationType

from .cache_performance_calculator import CachePerformanceCalculator
from .protocols import CacheMonitoringProtocol, MonitoringRepositoryProtocol  
from .types import MonitoringConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Cache alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CacheEntry:
    """Cache entry tracking."""
    key: str
    value: Any = None
    size_bytes: int = 0
    access_count: int = 0
    last_access: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: Optional[int] = None
    access_pattern: AccessPattern = AccessPattern.COLD
    promotion_score: float = 0.0
    levels_present: Set[CacheLevel] = field(default_factory=set)


@dataclass
class CachePerformanceAlert:
    """Cache performance alert."""
    alert_id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    cache_level: CacheLevel
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    affected_operations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class InvalidationEvent:
    """Cache invalidation event tracking."""
    event_id: str
    invalidation_type: InvalidationType
    affected_keys: List[str]
    cache_levels: List[CacheLevel]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    cascade_depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheMonitoringService:
    """Service for comprehensive cache monitoring across all levels.
    
    Consolidates cache monitoring functionality from legacy cache/ components:
    - Performance monitoring and alerting
    - Cross-level coordination tracking
    - Cache invalidation monitoring  
    - SLO compliance tracking
    - Predictive warming analysis
    """
    
    def __init__(
        self,
        config: MonitoringConfig,
        repository: Optional[MonitoringRepositoryProtocol] = None,
    ):
        self.config = config
        self.repository = repository
        
        # Cache operation tracking
        self._operation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._cache_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self._alert_thresholds = {
            "hit_rate_warning": 0.7,
            "hit_rate_critical": 0.5,
            "response_time_warning": 100.0,  # ms
            "response_time_critical": 500.0,  # ms
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.15,
        }
        
        # Alerting
        self._active_alerts: Dict[str, CachePerformanceAlert] = {}
        self._alert_callbacks: List[callable] = []
        self._alert_cooldown = 300  # 5 minutes
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Invalidation tracking
        self._invalidation_history: deque = deque(maxlen=10000)
        self._invalidation_callbacks: List[callable] = []
        
        # Cache dependencies and coordination
        self._cache_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Warming patterns
        self._warming_patterns: Dict[str, Dict[str, Any]] = {}
        self._warming_queue: List[str] = []
        
        # Cross-level coordination stats
        self._promotion_stats = defaultdict(int)
        self._demotion_stats = defaultdict(int)
        self._coordination_events: deque = deque(maxlen=5000)
        
        logger.info("CacheMonitoringService initialized")
    
    async def start_monitoring(self) -> None:
        """Start cache monitoring background tasks."""
        try:
            # Start monitoring loops
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._warming_analysis_loop())
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("Cache monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start cache monitoring: {e}")
            raise
    
    def record_cache_operation(
        self,
        operation: str,
        cache_level: CacheLevel,
        hit: bool,
        duration_ms: float,
        key: Optional[str] = None,
        value_size: int = 0,
    ) -> None:
        """Record cache operation for monitoring."""
        try:
            metric_key = f"{operation}_{cache_level.value}"
            
            operation_data = {
                "timestamp": time.time(),
                "operation": operation,
                "cache_level": cache_level,
                "hit": hit,
                "duration_ms": duration_ms,
                "key": key,
                "value_size": value_size,
            }
            
            self._operation_history[metric_key].append(operation_data)
            
            # Update warming patterns if cache hit
            if hit and key:
                self._update_warming_pattern(key)
            
            # Record telemetry
            if OPENTELEMETRY_AVAILABLE and cache_operations_monitored:
                cache_operations_monitored.add(
                    1,
                    {
                        "operation": operation,
                        "cache_level": cache_level.value,
                        "hit": str(hit).lower(),
                    }
                )
            
            # Check for performance alerts
            asyncio.create_task(self._check_performance_alerts(metric_key))
            
        except Exception as e:
            logger.error(f"Failed to record cache operation: {e}")
    
    async def record_invalidation_event(
        self,
        event_id: str,
        invalidation_type: InvalidationType,
        affected_keys: List[str],
        cache_levels: List[CacheLevel],
        cascade_depth: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record cache invalidation event."""
        try:
            event = InvalidationEvent(
                event_id=event_id,
                invalidation_type=invalidation_type,
                affected_keys=affected_keys,
                cache_levels=cache_levels,
                cascade_depth=cascade_depth,
                metadata=metadata or {},
            )
            
            self._invalidation_history.append(event)
            
            # Notify callbacks
            for callback in self._invalidation_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Invalidation callback failed: {e}")
            
            # Store in repository
            if self.repository:
                await self.repository.store_invalidation_event(event)
            
            logger.debug(f"Recorded invalidation event {event_id} affecting {len(affected_keys)} keys")
            
        except Exception as e:
            logger.error(f"Failed to record invalidation event: {e}")
    
    def record_coordination_event(
        self,
        action: CoordinationAction,
        key: str,
        source_level: Optional[CacheLevel] = None,
        target_level: Optional[CacheLevel] = None,
        success: bool = True,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record cross-level coordination event."""
        try:
            event = {
                "event_id": f"coord_{int(time.time() * 1000)}_{hash(key) % 10000}",
                "action": action,
                "key": key,
                "source_level": source_level,
                "target_level": target_level,
                "success": success,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(UTC),
                "metadata": metadata or {},
            }
            
            self._coordination_events.append(event)
            
            # Update promotion/demotion stats
            if action == CoordinationAction.PROMOTE and success:
                promotion_key = f"{source_level.value}_to_{target_level.value}"
                self._promotion_stats[promotion_key] += 1
            elif action == CoordinationAction.DEMOTE and success:
                demotion_key = f"{source_level.value}_to_{target_level.value}"
                self._demotion_stats[demotion_key] += 1
            
            logger.debug(f"Recorded coordination event: {action.value} {key}")
            
        except Exception as e:
            logger.error(f"Failed to record coordination event: {e}")
    
    def add_cache_dependency(self, key: str, depends_on: str) -> None:
        """Add cache dependency relationship."""
        try:
            self._cache_dependencies[key].add(depends_on)
            self._dependency_graph[depends_on].add(key)
            
            logger.debug(f"Added cache dependency: {key} depends on {depends_on}")
            
        except Exception as e:
            logger.error(f"Failed to add cache dependency: {e}")
    
    def remove_cache_dependency(self, key: str, depends_on: str) -> None:
        """Remove cache dependency relationship."""
        try:
            self._cache_dependencies[key].discard(depends_on)
            self._dependency_graph[depends_on].discard(key)
            
            # Clean up empty entries
            if not self._cache_dependencies[key]:
                del self._cache_dependencies[key]
            if not self._dependency_graph[depends_on]:
                del self._dependency_graph[depends_on]
            
            logger.debug(f"Removed cache dependency: {key} no longer depends on {depends_on}")
            
        except Exception as e:
            logger.error(f"Failed to remove cache dependency: {e}")
    
    def get_dependent_keys(self, key: str, max_depth: int = 5) -> Set[str]:
        """Get all keys that depend on the given key."""
        dependents = set()
        queue = [(key, 0)]
        visited = set()
        
        while queue:
            current_key, depth = queue.pop(0)
            
            if current_key in visited or depth >= max_depth:
                continue
            
            visited.add(current_key)
            direct_dependents = self._dependency_graph.get(current_key, set())
            
            for dependent in direct_dependents:
                if dependent not in dependents:
                    dependents.add(dependent)
                    queue.append((dependent, depth + 1))
        
        return dependents
    
    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance report."""
        start_time = time.time()
        
        try:
            report = {
                "report_type": "cache_performance",
                "generated_at": datetime.now(UTC).isoformat(),
                "time_window_minutes": 60,  # Report on last hour
            }
            
            # Overall statistics
            overall_stats = self._calculate_overall_stats()
            report["overall_stats"] = overall_stats
            
            # Per-level breakdown
            level_stats = {}
            for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
                level_stats[level.value] = self._calculate_level_stats(level)
            report["level_breakdown"] = level_stats
            
            # Performance metrics
            report["performance_metrics"] = self._calculate_performance_metrics()
            
            # Active alerts
            report["active_alerts"] = [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "metric": alert.metric_name,
                    "message": alert.message,
                    "cache_level": alert.cache_level.value,
                }
                for alert in self._active_alerts.values()
            ]
            
            # Invalidation statistics
            report["invalidation_stats"] = self._calculate_invalidation_stats()
            
            # Coordination statistics
            report["coordination_stats"] = {
                "promotions": dict(self._promotion_stats),
                "demotions": dict(self._demotion_stats),
                "recent_events": len([
                    e for e in self._coordination_events
                    if (datetime.now(UTC) - e["timestamp"]).total_seconds() < 3600
                ]),
            }
            
            # Warming analysis
            report["warming_analysis"] = await self._analyze_warming_opportunities()
            
            # Record telemetry
            duration = time.time() - start_time
            if OPENTELEMETRY_AVAILABLE and cache_monitoring_duration:
                cache_monitoring_duration.record(
                    duration,
                    {"operation": "performance_report"}
                )
            
            logger.info(f"Generated cache performance report in {duration:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate cache performance report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(UTC).isoformat(),
                "report_type": "cache_performance_error",
            }
    
    def _update_warming_pattern(self, key: str) -> None:
        """Update warming pattern based on cache access."""
        try:
            now = datetime.now(UTC)
            
            if key not in self._warming_patterns:
                self._warming_patterns[key] = {
                    "frequency": 1.0,
                    "success_rate": 1.0,
                    "last_access": now,
                    "access_count": 1,
                    "predicted_next_access": None,
                }
            else:
                pattern = self._warming_patterns[key]
                
                # Update access count
                pattern["access_count"] += 1
                
                # Update frequency (exponential moving average)
                time_since_last = (now - pattern["last_access"]).total_seconds() / 3600
                if time_since_last > 0:
                    new_frequency = 1.0 / time_since_last
                    pattern["frequency"] = 0.9 * pattern["frequency"] + 0.1 * new_frequency
                
                pattern["last_access"] = now
                
                # Predict next access
                if pattern["frequency"] > 0:
                    hours_until_next = 1.0 / pattern["frequency"]
                    pattern["predicted_next_access"] = now + timedelta(hours=hours_until_next)
            
        except Exception as e:
            logger.error(f"Failed to update warming pattern for {key}: {e}")
    
    async def _check_performance_alerts(self, metric_key: str) -> None:
        """Check for performance degradation and generate alerts."""
        try:
            history = self._operation_history[metric_key]
            if len(history) < 10:
                return
            
            recent_data = list(history)[-10:]
            
            # Calculate metrics
            hit_rates = [d["hit"] for d in recent_data]
            durations = [d["duration_ms"] for d in recent_data]
            
            hit_rate = sum(hit_rates) / len(hit_rates)
            avg_duration = sum(durations) / len(durations)
            
            # Check thresholds and generate alerts
            alerts_to_generate = []
            
            # Hit rate alerts
            if hit_rate < self._alert_thresholds["hit_rate_critical"]:
                alerts_to_generate.append(
                    self._create_hit_rate_alert(metric_key, hit_rate, AlertSeverity.CRITICAL)
                )
            elif hit_rate < self._alert_thresholds["hit_rate_warning"]:
                alerts_to_generate.append(
                    self._create_hit_rate_alert(metric_key, hit_rate, AlertSeverity.WARNING)
                )
            
            # Response time alerts
            if avg_duration > self._alert_thresholds["response_time_critical"]:
                alerts_to_generate.append(
                    self._create_response_time_alert(metric_key, avg_duration, AlertSeverity.CRITICAL)
                )
            elif avg_duration > self._alert_thresholds["response_time_warning"]:
                alerts_to_generate.append(
                    self._create_response_time_alert(metric_key, avg_duration, AlertSeverity.WARNING)
                )
            
            # Process alerts
            for alert in alerts_to_generate:
                await self._maybe_generate_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to check performance alerts for {metric_key}: {e}")
    
    def _create_hit_rate_alert(
        self,
        metric_key: str,
        hit_rate: float,
        severity: AlertSeverity,
    ) -> CachePerformanceAlert:
        """Create cache hit rate alert."""
        cache_level = self._extract_cache_level_from_metric_key(metric_key)
        threshold = (
            self._alert_thresholds["hit_rate_critical"]
            if severity == AlertSeverity.CRITICAL
            else self._alert_thresholds["hit_rate_warning"]
        )
        
        return CachePerformanceAlert(
            alert_id=f"cache_hit_rate_{metric_key}_{int(time.time())}",
            severity=severity,
            metric_name="cache_hit_rate",
            current_value=hit_rate,
            threshold_value=threshold,
            message=f"{severity.value.upper()}: Cache hit rate {hit_rate:.1%} below threshold {threshold:.1%} for {metric_key}",
            cache_level=cache_level,
            recommendations=[
                "Consider cache warming for frequently accessed keys",
                "Review cache TTL settings",
                "Analyze access patterns for optimization opportunities",
            ],
        )
    
    def _create_response_time_alert(
        self,
        metric_key: str,
        avg_duration: float,
        severity: AlertSeverity,
    ) -> CachePerformanceAlert:
        """Create cache response time alert."""
        cache_level = self._extract_cache_level_from_metric_key(metric_key)
        threshold = (
            self._alert_thresholds["response_time_critical"]
            if severity == AlertSeverity.CRITICAL
            else self._alert_thresholds["response_time_warning"]
        )
        
        return CachePerformanceAlert(
            alert_id=f"cache_response_time_{metric_key}_{int(time.time())}",
            severity=severity,
            metric_name="cache_response_time",
            current_value=avg_duration,
            threshold_value=threshold,
            message=f"{severity.value.upper()}: Cache response time {avg_duration:.1f}ms above threshold {threshold:.1f}ms for {metric_key}",
            cache_level=cache_level,
            recommendations=[
                "Check cache backend performance",
                "Review serialization/deserialization overhead",
                "Consider cache data size optimization",
            ],
        )
    
    def _extract_cache_level_from_metric_key(self, metric_key: str) -> CacheLevel:
        """Extract cache level from metric key."""
        if "l1" in metric_key.lower():
            return CacheLevel.L1
        elif "l2" in metric_key.lower():
            return CacheLevel.L2  
        elif "l3" in metric_key.lower():
            return CacheLevel.L3
        else:
            return CacheLevel.L1  # Default
    
    async def _maybe_generate_alert(self, alert: CachePerformanceAlert) -> None:
        """Generate alert if cooldown period has passed."""
        try:
            alert_key = f"{alert.metric_name}_{alert.cache_level.value}"
            now = datetime.now(UTC)
            
            # Check cooldown
            if alert_key in self._last_alert_times:
                time_since_last = (now - self._last_alert_times[alert_key]).total_seconds()
                if time_since_last < self._alert_cooldown:
                    return
            
            # Store and process alert
            self._active_alerts[alert.alert_id] = alert
            self._last_alert_times[alert_key] = now
            
            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Cache alert callback failed: {e}")
            
            logger.warning(f"Generated cache performance alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to generate cache alert: {e}")
    
    def _calculate_overall_stats(self) -> Dict[str, Any]:
        """Calculate overall cache statistics.""" 
        return CachePerformanceCalculator.calculate_overall_stats(self._operation_history)
    
    def _calculate_level_stats(self, level: CacheLevel) -> Dict[str, Any]:
        """Calculate statistics for specific cache level."""
        return CachePerformanceCalculator.calculate_level_stats(self._operation_history, level)
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        return CachePerformanceCalculator.calculate_performance_metrics(self._operation_history)
    
    def _calculate_invalidation_stats(self) -> Dict[str, Any]:
        """Calculate invalidation statistics."""
        return CachePerformanceCalculator.calculate_invalidation_stats(self._invalidation_history)
    
    def _calculate_warming_stats(self) -> Dict[str, Any]:
        """Calculate cache warming statistics."""
        return CachePerformanceCalculator.calculate_warming_stats(self._warming_patterns)
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for cache performance alerts."""
        self._alert_callbacks.append(callback)
        logger.info("Added cache alert callback")
    
    def add_invalidation_callback(self, callback: callable) -> None:
        """Add callback for invalidation events."""
        self._invalidation_callbacks.append(callback)
        logger.info("Added cache invalidation callback")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Update cache statistics
                self._update_cache_statistics()
                
                # Check for stale alerts
                await self._cleanup_stale_alerts()
                
        except Exception as e:
            logger.error(f"Performance monitoring loop failed: {e}")
    
    async def _warming_analysis_loop(self) -> None:
        """Background loop for warming analysis."""
        try:
            while True:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Analyze warming opportunities
                await self._analyze_warming_opportunities()
                
        except Exception as e:
            logger.error(f"Warming analysis loop failed: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        try:
            while True:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Clean up old operation history
                self._cleanup_old_operation_data()
                
                # Clean up old warming patterns
                self._cleanup_old_warming_patterns()
                
        except Exception as e:
            logger.error(f"Cleanup loop failed: {e}")
    
    def _update_cache_statistics(self) -> None:
        """Update internal cache statistics."""
        # This would be called periodically to update stats
        pass
    
    async def _cleanup_stale_alerts(self) -> None:
        """Clean up resolved or stale alerts."""
        # Remove alerts older than 24 hours
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        
        stale_alerts = [
            alert_id for alert_id, alert in self._active_alerts.items()
            if alert.timestamp < cutoff_time
        ]
        
        for alert_id in stale_alerts:
            del self._active_alerts[alert_id]
        
        if stale_alerts:
            logger.info(f"Cleaned up {len(stale_alerts)} stale cache alerts")
    
    def _cleanup_old_operation_data(self) -> None:
        """Clean up old operation data."""
        # Operation history is already limited by deque maxlen
        # Additional cleanup could be added here if needed
        pass
    
    def _cleanup_old_warming_patterns(self) -> None:
        """Clean up old warming patterns."""
        cutoff_time = datetime.now(UTC) - timedelta(days=7)
        
        stale_patterns = [
            key for key, pattern in self._warming_patterns.items()
            if pattern["last_access"] < cutoff_time
        ]
        
        for key in stale_patterns:
            del self._warming_patterns[key]
        
        if stale_patterns:
            logger.info(f"Cleaned up {len(stale_patterns)} old warming patterns")