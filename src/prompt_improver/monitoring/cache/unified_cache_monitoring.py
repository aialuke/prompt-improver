"""
Unified Cache Monitoring System
===============================

Comprehensive monitoring and alerting system for all consolidated cache operations,
providing enhanced OpenTelemetry integration, coordinated invalidation, and 
performance metrics unification across L1/L2/L3 cache levels.
"""

import asyncio
import json
import logging
import time
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    
    # Get tracer and meter for unified cache operations
    cache_tracer = trace.get_tracer(__name__ + ".unified_cache")
    cache_meter = metrics.get_meter(__name__ + ".unified_cache")
    
    # Create comprehensive cache metrics
    cache_operations_total = cache_meter.create_counter(
        "unified_cache_operations_total",
        description="Total unified cache operations by type, level, and status",
        unit="1"
    )
    
    cache_hit_ratio = cache_meter.create_gauge(
        "unified_cache_hit_ratio",
        description="Unified cache hit ratio by level and operation type",
        unit="ratio"
    )
    
    cache_operation_duration = cache_meter.create_histogram(
        "unified_cache_operation_duration_seconds",
        description="Unified cache operation duration by type and level",
        unit="s"
    )
    
    cache_invalidation_events = cache_meter.create_counter(
        "unified_cache_invalidation_events_total",
        description="Total cache invalidation events by type and scope",
        unit="1"
    )
    
    cache_dependency_violations = cache_meter.create_counter(
        "unified_cache_dependency_violations_total",
        description="Total cache dependency violations detected",
        unit="1"
    )
    
    cache_warming_effectiveness = cache_meter.create_gauge(
        "unified_cache_warming_effectiveness",
        description="Cache warming effectiveness ratio",
        unit="ratio"
    )
    
    cache_cross_level_promotion = cache_meter.create_counter(
        "unified_cache_cross_level_promotion_total",
        description="Total cache promotions between levels",
        unit="1"
    )
    
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    cache_tracer = None
    cache_meter = None
    cache_operations_total = None
    cache_hit_ratio = None
    cache_operation_duration = None
    cache_invalidation_events = None
    cache_dependency_violations = None
    cache_warming_effectiveness = None
    cache_cross_level_promotion = None

# Background task manager import
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

logger = logging.getLogger(__name__)

class InvalidationType(Enum):
    """Types of cache invalidation operations."""
    PATTERN = "pattern"
    DEPENDENCY = "dependency"
    CASCADE = "cascade"
    MANUAL = "manual"
    TTL_EXPIRED = "ttl_expired"
    CAPACITY = "capacity"

class CacheLevel(Enum):
    """Cache levels for unified monitoring."""
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    ALL = "all"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class CacheDependency:
    """Represents a cache dependency relationship."""
    key: str
    depends_on: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    cascade_level: int = 1  # How many levels to cascade invalidation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class InvalidationEvent:
    """Represents a cache invalidation event."""
    event_id: str
    invalidation_type: InvalidationType
    affected_keys: List[str]
    cache_levels: List[CacheLevel]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cascade_depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CachePerformanceAlert:
    """Represents a cache performance alert."""
    alert_id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    cache_level: CacheLevel
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    affected_operations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class CacheWarmingPattern:
    """Represents a cache warming pattern."""
    pattern: str
    frequency: float  # Access frequency per hour
    success_rate: float  # Warming success rate
    last_access: datetime
    predicted_next_access: Optional[datetime] = None
    warming_priority: int = 1  # 1-5, higher is more important

class UnifiedCacheMonitor:
    """
    Unified monitoring system for all consolidated cache operations.
    
    Provides comprehensive observability, coordinated invalidation, and
    performance analytics across L1/L2/L3 cache levels.
    """
    
    def __init__(self, unified_manager=None):
        """Initialize unified cache monitoring.
        
        Args:
            unified_manager: Optional UnifiedConnectionManager instance
        """
        self._unified_manager = unified_manager
        
        # Dependency tracking
        self._dependencies: Dict[str, CacheDependency] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Invalidation tracking
        self._invalidation_history: deque = deque(maxlen=10000)
        self._invalidation_callbacks: List[Callable] = []
        
        # Performance monitoring
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._alert_thresholds = {
            "hit_rate_warning": 0.7,
            "hit_rate_critical": 0.5,
            "response_time_warning": 100.0,  # ms
            "response_time_critical": 500.0,  # ms
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.15
        }
        
        # Cache warming intelligence
        self._warming_patterns: Dict[str, CacheWarmingPattern] = {}
        self._warming_queue: List[Tuple[str, int]] = []  # (key, priority)
        
        # Cross-level coordination
        self._promotion_stats = defaultdict(int)
        self._demotion_stats = defaultdict(int)
        self._cross_level_misses = defaultdict(int)
        
        # Alert management
        self._active_alerts: Dict[str, CachePerformanceAlert] = {}
        self._alert_callbacks: List[Callable] = []
        self._alert_cooldown = 300  # 5 minutes
        self._last_alert_times: Dict[str, datetime] = {}
        
        logger.info("UnifiedCacheMonitor initialized with comprehensive observability")
    
    # ========== Dependency Tracking ==========
    
    def add_dependency(self, key: str, depends_on: str, cascade_level: int = 1):
        """Add cache dependency relationship.
        
        Args:
            key: Cache key that depends on another
            depends_on: Cache key that this key depends on
            cascade_level: How many levels to cascade invalidation
        """
        if key not in self._dependencies:
            self._dependencies[key] = CacheDependency(key=key)
        
        self._dependencies[key].depends_on.add(depends_on)
        self._dependencies[key].cascade_level = max(
            self._dependencies[key].cascade_level, cascade_level
        )
        
        # Update dependency graph
        self._dependency_graph[depends_on].add(key)
        self._reverse_dependencies[key].add(depends_on)
        
        logger.debug(f"Added dependency: {key} depends on {depends_on} (cascade: {cascade_level})")
    
    def remove_dependency(self, key: str, depends_on: str):
        """Remove cache dependency relationship."""
        if key in self._dependencies:
            self._dependencies[key].depends_on.discard(depends_on)
            if not self._dependencies[key].depends_on:
                del self._dependencies[key]
        
        self._dependency_graph[depends_on].discard(key)
        self._reverse_dependencies[key].discard(depends_on)
        
        logger.debug(f"Removed dependency: {key} no longer depends on {depends_on}")
    
    def get_dependent_keys(self, key: str, max_depth: int = 5) -> Set[str]:
        """Get all keys that depend on the given key.
        
        Args:
            key: Cache key to find dependents for
            max_depth: Maximum dependency depth to traverse
            
        Returns:
            Set of dependent cache keys
        """
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
    
    # ========== Coordinated Invalidation ==========
    
    @asynccontextmanager
    async def trace_invalidation(self, invalidation_type: InvalidationType, keys: List[str]):
        """Context manager for tracing cache invalidation operations."""
        event_id = f"inv_{int(time.time() * 1000)}_{hash(tuple(keys)) % 10000}"
        
        span = None
        if OPENTELEMETRY_AVAILABLE and cache_tracer:
            span = cache_tracer.start_span(
                f"cache_invalidation_{invalidation_type.value}",
                attributes={
                    "cache.invalidation.type": invalidation_type.value,
                    "cache.invalidation.key_count": len(keys),
                    "cache.invalidation.event_id": event_id
                }
            )
        
        start_time = time.time()
        
        try:
            yield event_id
            
            if span:
                span.set_status(Status(StatusCode.OK))
            
            # Record successful invalidation
            if OPENTELEMETRY_AVAILABLE and cache_invalidation_events:
                cache_invalidation_events.add(1, {
                    "invalidation_type": invalidation_type.value,
                    "status": "success"
                })
        
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            # Record failed invalidation
            if OPENTELEMETRY_AVAILABLE and cache_invalidation_events:
                cache_invalidation_events.add(1, {
                    "invalidation_type": invalidation_type.value,
                    "status": "error"
                })
            
            logger.error(f"Cache invalidation failed: {e}")
            raise
        
        finally:
            duration = time.time() - start_time
            
            # Create invalidation event
            event = InvalidationEvent(
                event_id=event_id,
                invalidation_type=invalidation_type,
                affected_keys=keys,
                cache_levels=[CacheLevel.ALL],
                metadata={"duration_seconds": duration}
            )
            
            self._invalidation_history.append(event)
            
            if span:
                span.end()
    
    async def invalidate_by_pattern(self, pattern: str, cache_levels: Optional[List[CacheLevel]] = None) -> int:
        """Invalidate cache entries by pattern with dependency cascade.
        
        Args:
            pattern: Pattern to match for invalidation
            cache_levels: Cache levels to invalidate (default: all)
            
        Returns:
            Number of keys invalidated
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.ALL]
        
        async with self.trace_invalidation(InvalidationType.PATTERN, [pattern]):
            total_invalidated = 0
            
            # Get matching keys from unified manager if available
            if self._unified_manager and hasattr(self._unified_manager, 'scan_keys'):
                matching_keys = await self._unified_manager.scan_keys(pattern)
            else:
                # Fallback: simulate pattern matching
                matching_keys = [key for key in self._dependencies.keys() if pattern in key]
            
            # Invalidate matching keys with dependency cascade
            for key in matching_keys:
                invalidated_count = await self._invalidate_with_cascade(key, cache_levels)
                total_invalidated += invalidated_count
            
            logger.info(f"Pattern invalidation '{pattern}' affected {total_invalidated} keys")
            return total_invalidated
    
    async def invalidate_by_dependency(self, root_key: str, cache_levels: Optional[List[CacheLevel]] = None) -> int:
        """Invalidate cache entries by dependency relationships.
        
        Args:
            root_key: Root key whose dependents should be invalidated
            cache_levels: Cache levels to invalidate (default: all)
            
        Returns:
            Number of keys invalidated
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.ALL]
        
        dependent_keys = self.get_dependent_keys(root_key)
        all_keys = [root_key] + list(dependent_keys)
        
        async with self.trace_invalidation(InvalidationType.DEPENDENCY, all_keys):
            total_invalidated = 0
            
            for key in all_keys:
                invalidated_count = await self._invalidate_single_key(key, cache_levels)
                total_invalidated += invalidated_count
            
            logger.info(f"Dependency invalidation from '{root_key}' affected {total_invalidated} keys")
            return total_invalidated
    
    async def _invalidate_with_cascade(self, key: str, cache_levels: List[CacheLevel], depth: int = 0) -> int:
        """Invalidate key with dependency cascade."""
        if depth > 5:  # Prevent infinite recursion
            logger.warning(f"Maximum cascade depth reached for key: {key}")
            return 0
        
        total_invalidated = await self._invalidate_single_key(key, cache_levels)
        
        # Get cascade configuration
        cascade_level = 1
        if key in self._dependencies:
            cascade_level = self._dependencies[key].cascade_level
        
        if depth < cascade_level:
            # Cascade to dependents
            dependents = self._dependency_graph.get(key, set())
            for dependent in dependents:
                cascade_count = await self._invalidate_with_cascade(
                    dependent, cache_levels, depth + 1
                )
                total_invalidated += cascade_count
        
        return total_invalidated
    
    async def _invalidate_single_key(self, key: str, cache_levels: List[CacheLevel]) -> int:
        """Invalidate a single key across specified cache levels."""
        invalidated_count = 0
        
        if not self._unified_manager:
            return invalidated_count
        
        try:
            # Invalidate across cache levels
            if CacheLevel.L1 in cache_levels or CacheLevel.ALL in cache_levels:
                if hasattr(self._unified_manager, '_l1_cache') and self._unified_manager._l1_cache:
                    if self._unified_manager._l1_cache.delete(key):
                        invalidated_count += 1
            
            if CacheLevel.L2 in cache_levels or CacheLevel.ALL in cache_levels:
                if hasattr(self._unified_manager, '_redis_master') and self._unified_manager._redis_master:
                    deleted = await self._unified_manager._redis_master.delete(key)
                    invalidated_count += deleted
            
            # L3 would be database-level caching (if implemented)
            if CacheLevel.L3 in cache_levels or CacheLevel.ALL in cache_levels:
                # Future: database query cache invalidation
                pass
        
        except Exception as e:
            logger.error(f"Error invalidating key {key}: {e}")
        
        return invalidated_count
    
    # ========== Performance Monitoring ==========
    
    def record_cache_operation(
        self,
        operation: str,
        cache_level: CacheLevel,
        hit: bool,
        duration_ms: float,
        key: Optional[str] = None
    ):
        """Record cache operation for performance monitoring.
        
        Args:
            operation: Operation type (get, set, delete, exists)
            cache_level: Cache level where operation occurred
            hit: Whether operation was a cache hit
            duration_ms: Operation duration in milliseconds
            key: Optional cache key for pattern analysis
        """
        # Record in performance history
        metric_key = f"{operation}_{cache_level.value}"
        self._performance_history[metric_key].append({
            "timestamp": time.time(),
            "hit": hit,
            "duration_ms": duration_ms,
            "key": key
        })
        
        # Record OpenTelemetry metrics
        if OPENTELEMETRY_AVAILABLE:
            labels = {
                "operation": operation,
                "cache_level": cache_level.value,
                "hit": str(hit).lower()
            }
            
            if cache_operations_total:
                cache_operations_total.add(1, labels)
            
            if cache_operation_duration:
                cache_operation_duration.record(duration_ms / 1000.0, labels)
        
        # Update warming patterns if it's a cache hit
        if hit and key:
            self._update_warming_pattern(key)
        
        # Check for performance alerts
        asyncio.create_task(self._check_performance_alerts(operation, cache_level))
    
    def _update_warming_pattern(self, key: str):
        """Update cache warming pattern based on access."""
        now = datetime.now(timezone.utc)
        
        if key not in self._warming_patterns:
            self._warming_patterns[key] = CacheWarmingPattern(
                pattern=key,
                frequency=1.0,
                success_rate=1.0,
                last_access=now
            )
        else:
            pattern = self._warming_patterns[key]
            # Update frequency (exponential moving average)
            time_since_last = (now - pattern.last_access).total_seconds() / 3600  # hours
            if time_since_last > 0:
                pattern.frequency = 0.9 * pattern.frequency + 0.1 * (1.0 / time_since_last)
            pattern.last_access = now
            
            # Predict next access
            if pattern.frequency > 0:
                hours_until_next = 1.0 / pattern.frequency
                pattern.predicted_next_access = now + timedelta(hours=hours_until_next)
    
    async def _check_performance_alerts(self, operation: str, cache_level: CacheLevel):
        """Check for performance degradation and generate alerts."""
        metric_key = f"{operation}_{cache_level.value}"
        history = self._performance_history[metric_key]
        
        if len(history) < 10:  # Need enough data
            return
        
        # Calculate recent performance
        recent_data = list(history)[-10:]  # Last 10 operations
        hit_rate = sum(1 for d in recent_data if d["hit"]) / len(recent_data)
        avg_duration = sum(d["duration_ms"] for d in recent_data) / len(recent_data)
        
        alerts_to_generate = []
        
        # Check hit rate thresholds
        if hit_rate < self._alert_thresholds["hit_rate_critical"]:
            alerts_to_generate.append(self._create_hit_rate_alert(
                metric_key, hit_rate, AlertSeverity.CRITICAL, cache_level
            ))
        elif hit_rate < self._alert_thresholds["hit_rate_warning"]:
            alerts_to_generate.append(self._create_hit_rate_alert(
                metric_key, hit_rate, AlertSeverity.WARNING, cache_level
            ))
        
        # Check response time thresholds
        if avg_duration > self._alert_thresholds["response_time_critical"]:
            alerts_to_generate.append(self._create_response_time_alert(
                metric_key, avg_duration, AlertSeverity.CRITICAL, cache_level
            ))
        elif avg_duration > self._alert_thresholds["response_time_warning"]:
            alerts_to_generate.append(self._create_response_time_alert(
                metric_key, avg_duration, AlertSeverity.WARNING, cache_level
            ))
        
        # Generate alerts with cooldown
        for alert in alerts_to_generate:
            await self._maybe_generate_alert(alert)
    
    def _create_hit_rate_alert(
        self, metric_key: str, hit_rate: float, severity: AlertSeverity, cache_level: CacheLevel
    ) -> CachePerformanceAlert:
        """Create cache hit rate alert."""
        threshold = (self._alert_thresholds["hit_rate_critical"] 
                    if severity == AlertSeverity.CRITICAL 
                    else self._alert_thresholds["hit_rate_warning"])
        
        return CachePerformanceAlert(
            alert_id=f"hit_rate_{metric_key}_{int(time.time())}",
            severity=severity,
            metric_name="cache_hit_rate",
            current_value=hit_rate,
            threshold_value=threshold,
            message=f"{severity.value.upper()}: Cache hit rate {hit_rate:.1%} below threshold {threshold:.1%} for {metric_key}",
            cache_level=cache_level,
            recommendations=[
                "Consider cache warming for frequently accessed keys",
                "Review cache TTL settings",
                "Analyze access patterns for optimization opportunities"
            ]
        )
    
    def _create_response_time_alert(
        self, metric_key: str, avg_duration: float, severity: AlertSeverity, cache_level: CacheLevel
    ) -> CachePerformanceAlert:
        """Create cache response time alert."""
        threshold = (self._alert_thresholds["response_time_critical"]
                    if severity == AlertSeverity.CRITICAL
                    else self._alert_thresholds["response_time_warning"])
        
        return CachePerformanceAlert(
            alert_id=f"response_time_{metric_key}_{int(time.time())}",
            severity=severity,
            metric_name="cache_response_time",
            current_value=avg_duration,
            threshold_value=threshold,
            message=f"{severity.value.upper()}: Cache response time {avg_duration:.1f}ms above threshold {threshold:.1f}ms for {metric_key}",
            cache_level=cache_level,
            recommendations=[
                "Check cache backend performance",
                "Review serialization/deserialization overhead",
                "Consider cache data size optimization"
            ]
        )
    
    async def _maybe_generate_alert(self, alert: CachePerformanceAlert):
        """Generate alert if cooldown period has passed."""
        alert_key = f"{alert.metric_name}_{alert.cache_level.value}"
        now = datetime.now(timezone.utc)
        
        # Check cooldown
        if alert_key in self._last_alert_times:
            time_since_last = (now - self._last_alert_times[alert_key]).total_seconds()
            if time_since_last < self._alert_cooldown:
                return
        
        # Generate alert
        self._active_alerts[alert.alert_id] = alert
        self._last_alert_times[alert_key] = now
        
        # Execute alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Generated cache performance alert: {alert.message}")
    
    # ========== Cache Warming Intelligence ==========
    
    async def analyze_warming_opportunities(self) -> List[str]:
        """Analyze cache access patterns to identify warming opportunities.
        
        Returns:
            List of cache keys that should be warmed
        """
        now = datetime.now(timezone.utc)
        warming_candidates = []
        
        for key, pattern in self._warming_patterns.items():
            # Check if key is likely to be accessed soon
            if (pattern.predicted_next_access and 
                pattern.predicted_next_access <= now + timedelta(minutes=30) and
                pattern.frequency > 0.5):  # At least once every 2 hours
                
                warming_candidates.append(key)
        
        # Sort by priority (frequency * success_rate)
        warming_candidates.sort(
            key=lambda k: self._warming_patterns[k].frequency * self._warming_patterns[k].success_rate,
            reverse=True
        )
        
        return warming_candidates[:50]  # Return top 50 candidates
    
    async def execute_predictive_warming(self):
        """Execute predictive cache warming based on access patterns."""
        if not self._unified_manager:
            return
        
        candidates = await self.analyze_warming_opportunities()
        
        if not candidates:
            logger.debug("No cache warming candidates identified")
            return
        
        logger.info(f"Starting predictive warming for {len(candidates)} cache keys")
        
        # Submit warming tasks through background task manager
        task_manager = get_background_task_manager()
        
        if task_manager:
            for key in candidates:
                await task_manager.submit_enhanced_task(
                    task_id=f"cache_warming_{key}_{int(time.time())}",
                    coroutine=self._warm_single_key(key),
                    priority=TaskPriority.BACKGROUND,
                    tags={
                        "service": "cache_monitoring",
                        "type": "warming",
                        "component": "unified_cache_monitor"
                    }
                )
    
    async def _warm_single_key(self, key: str):
        """Warm a single cache key."""
        try:
            # This would need to be implemented based on how your cache warming works
            # For now, we'll just check if key exists and record the warming attempt
            if hasattr(self._unified_manager, 'exists_cached'):
                exists = await self._unified_manager.exists_cached(key)
                
                if key in self._warming_patterns:
                    pattern = self._warming_patterns[key]
                    if exists:
                        pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * 1.0
                    else:
                        pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * 0.0
        
        except Exception as e:
            logger.error(f"Cache warming failed for key {key}: {e}")
            
            if key in self._warming_patterns:
                pattern = self._warming_patterns[key]
                pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * 0.0
    
    # ========== Unified Statistics and Reporting ==========
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache monitoring statistics.
        
        Returns:
            Dictionary with all cache monitoring metrics
        """
        # Get base stats from unified manager if available
        base_stats = {}
        if self._unified_manager and hasattr(self._unified_manager, 'get_cache_stats'):
            base_stats = self._unified_manager.get_cache_stats()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Dependency statistics
        dependency_stats = {
            "total_dependencies": len(self._dependencies),
            "dependency_graph_size": sum(len(deps) for deps in self._dependency_graph.values()),
            "max_dependency_depth": self._calculate_max_dependency_depth()
        }
        
        # Invalidation statistics
        invalidation_stats = self._calculate_invalidation_stats()
        
        # Warming statistics
        warming_stats = self._calculate_warming_stats()
        
        # Cross-level coordination stats
        coordination_stats = {
            "l1_to_l2_promotions": self._promotion_stats.get("l1_to_l2", 0),
            "l2_to_l1_demotions": self._demotion_stats.get("l2_to_l1", 0),
            "cross_level_misses": dict(self._cross_level_misses)
        }
        
        # Active alerts
        alert_stats = {
            "active_alerts": len(self._active_alerts),
            "alert_breakdown": {
                severity.value: sum(1 for alert in self._active_alerts.values() 
                                  if alert.severity == severity)
                for severity in AlertSeverity
            }
        }
        
        return {
            **base_stats,
            "enhanced_monitoring": {
                "performance_metrics": performance_metrics,
                "dependency_tracking": dependency_stats,
                "invalidation_coordination": invalidation_stats,
                "predictive_warming": warming_stats,
                "cross_level_coordination": coordination_stats,
                "alerting": alert_stats,
                "monitoring_health": {
                    "callback_count": len(self._invalidation_callbacks),
                    "pattern_count": len(self._warming_patterns),
                    "history_size": len(self._invalidation_history)
                }
            }
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        for metric_key, history in self._performance_history.items():
            if not history:
                continue
            
            recent_data = list(history)[-100:]  # Last 100 operations
            
            hit_rates = [d["hit"] for d in recent_data]
            durations = [d["duration_ms"] for d in recent_data]
            
            metrics[metric_key] = {
                "operation_count": len(recent_data),
                "hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "p95_duration_ms": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else 0,
                "p99_duration_ms": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else 0
            }
        
        return metrics
    
    def _calculate_max_dependency_depth(self) -> int:
        """Calculate maximum dependency depth in the graph."""
        max_depth = 0
        
        for root_key in self._dependencies.keys():
            depth = self._calculate_dependency_depth(root_key, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_dependency_depth(self, key: str, visited: Set[str]) -> int:
        """Calculate dependency depth for a specific key."""
        if key in visited:
            return 0  # Circular dependency
        
        visited.add(key)
        max_child_depth = 0
        
        for dependent in self._dependency_graph.get(key, set()):
            child_depth = self._calculate_dependency_depth(dependent, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth + 1
    
    def _calculate_invalidation_stats(self) -> Dict[str, Any]:
        """Calculate invalidation statistics."""
        if not self._invalidation_history:
            return {"total_events": 0}
        
        recent_events = [event for event in self._invalidation_history
                        if (datetime.now(timezone.utc) - event.timestamp).total_seconds() < 3600]
        
        type_counts = defaultdict(int)
        for event in recent_events:
            type_counts[event.invalidation_type.value] += 1
        
        return {
            "total_events": len(self._invalidation_history),
            "recent_events_1h": len(recent_events),
            "events_by_type": dict(type_counts),
            "avg_cascade_depth": statistics.mean([e.cascade_depth for e in recent_events]) if recent_events else 0
        }
    
    def _calculate_warming_stats(self) -> Dict[str, Any]:
        """Calculate cache warming statistics."""
        if not self._warming_patterns:
            return {"active_patterns": 0}
        
        patterns = list(self._warming_patterns.values())
        avg_frequency = statistics.mean([p.frequency for p in patterns])
        avg_success_rate = statistics.mean([p.success_rate for p in patterns])
        
        # Calculate warming effectiveness
        effectiveness = 0.0
        if OPENTELEMETRY_AVAILABLE and cache_warming_effectiveness:
            cache_warming_effectiveness.set(avg_success_rate)
            effectiveness = avg_success_rate
        
        return {
            "active_patterns": len(self._warming_patterns),
            "avg_access_frequency": avg_frequency,
            "avg_success_rate": avg_success_rate,
            "warming_effectiveness": effectiveness,
            "queue_size": len(self._warming_queue)
        }
    
    # ========== Callback Management ==========
    
    def register_invalidation_callback(self, callback: Callable[[InvalidationEvent], None]):
        """Register callback for invalidation events."""
        self._invalidation_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable[[CachePerformanceAlert], None]):
        """Register callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    # ========== Integration Methods ==========
    
    async def integrate_with_unified_manager(self, unified_manager):
        """Integrate monitoring with UnifiedConnectionManager."""
        self._unified_manager = unified_manager
        
        # Start background monitoring tasks
        task_manager = get_background_task_manager()
        
        if task_manager:
            # Schedule periodic warming analysis
            await task_manager.submit_enhanced_task(
                task_id="cache_warming_analysis",
                coroutine=self._periodic_warming_analysis(),
                priority=TaskPriority.BACKGROUND,
                tags={
                    "service": "cache_monitoring",
                    "type": "analysis",
                    "component": "unified_cache_monitor"
                }
            )
        
        logger.info("UnifiedCacheMonitor integrated with UnifiedConnectionManager")
    
    async def _periodic_warming_analysis(self):
        """Periodic analysis and warming execution."""
        while True:
            try:
                await self.execute_predictive_warming()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Periodic warming analysis failed: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

# Global monitor instance
_unified_cache_monitor: Optional[UnifiedCacheMonitor] = None

def get_unified_cache_monitor() -> UnifiedCacheMonitor:
    """Get or create global unified cache monitor instance."""
    global _unified_cache_monitor
    
    if _unified_cache_monitor is None:
        _unified_cache_monitor = UnifiedCacheMonitor()
    
    return _unified_cache_monitor

def integrate_cache_monitoring(unified_manager):
    """Integrate cache monitoring with UnifiedConnectionManager.
    
    Args:
        unified_manager: UnifiedConnectionManager instance
    """
    monitor = get_unified_cache_monitor()
    asyncio.create_task(monitor.integrate_with_unified_manager(unified_manager))