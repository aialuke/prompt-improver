"""Cross-Level Cache Coordination
==============================

Enhanced coordination system for L1/L2/L3 cache levels, providing intelligent
promotion/demotion, cache coherence, and performance optimization across
all cache layers.
"""
import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    coordination_tracer = trace.get_tracer(__name__ + '.cross_level')
    coordination_meter = metrics.get_meter(__name__ + '.cross_level')
    cache_level_promotions = coordination_meter.create_counter('cache_level_promotions_total', description='Total cache promotions between levels', unit='1')
    cache_level_demotions = coordination_meter.create_counter('cache_level_demotions_total', description='Total cache demotions between levels', unit='1')
    cache_coherence_violations = coordination_meter.create_counter('cache_coherence_violations_total', description='Total cache coherence violations detected', unit='1')
    cache_coordination_efficiency = coordination_meter.create_gauge('cache_coordination_efficiency', description='Cache coordination efficiency ratio', unit='ratio')
    cross_level_latency = coordination_meter.create_histogram('cross_level_operation_duration_seconds', description='Cross-level cache operation duration', unit='s')
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    coordination_tracer = None
    coordination_meter = None
    cache_level_promotions = None
    cache_level_demotions = None
    cache_coherence_violations = None
    cache_coordination_efficiency = None
    cross_level_latency = None
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels for coordination."""
    L1 = 'l1'
    L2 = 'l2'
    L3 = 'l3'

class CoordinationAction(Enum):
    """Types of coordination actions."""
    PROMOTE = 'promote'
    DEMOTE = 'demote'
    REPLICATE = 'replicate'
    INVALIDATE = 'invalidate'
    WARM = 'warm'

class AccessPattern(Enum):
    """Cache access patterns."""
    HOT = 'hot'
    WARM = 'warm'
    COLD = 'cold'
    TEMPORAL = 'temporal'
    BURST = 'burst'

@dataclass
class CacheEntry:
    """Represents a cache entry across levels."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int | None = None
    access_pattern: AccessPattern = AccessPattern.COLD
    promotion_score: float = 0.0
    levels_present: set[CacheLevel] = field(default_factory=set)

@dataclass
class CoordinationEvent:
    """Represents a cache coordination event."""
    event_id: str
    action: CoordinationAction
    key: str
    source_level: CacheLevel | None
    target_level: CacheLevel | None
    success: bool
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationStrategy:
    """Configuration for cache coordination strategies."""
    promotion_threshold: float = 0.7
    demotion_threshold: float = 0.3
    max_l1_size: int = 1000
    max_l2_size: int = 10000
    coherence_timeout: int = 30
    access_window: timedelta = timedelta(minutes=15)
    min_access_count: int = 5

class CrossLevelCoordinator:
    """Enhanced coordination system for L1/L2/L3 cache levels.

    Provides intelligent cache management across all levels with
    performance optimization and coherence guarantees.
    """

    def __init__(self, unified_manager=None, strategy: CoordinationStrategy=None):
        """Initialize cross-level coordinator.

        Args:
            unified_manager: UnifiedConnectionManager instance
            strategy: Coordination strategy configuration
        """
        self._unified_manager = unified_manager
        self._strategy = strategy or CoordinationStrategy()
        self._entry_metadata: dict[str, CacheEntry] = {}
        self._access_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._coordination_events: deque = deque(maxlen=1000)
        self._pending_operations: dict[str, asyncio.Task] = {}
        self._coordination_stats = {'promotions': defaultdict(int), 'demotions': defaultdict(int), 'coherence_checks': 0, 'coherence_violations': 0, 'coordination_efficiency': 0.0}
        self._level_usage = {CacheLevel.L1: {'count': 0, 'size_bytes': 0}, CacheLevel.L2: {'count': 0, 'size_bytes': 0}, CacheLevel.L3: {'count': 0, 'size_bytes': 0}}
        self._coordination_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        logger.info('CrossLevelCoordinator initialized with intelligent cache management')

    def track_cache_access(self, key: str, cache_level: CacheLevel, hit: bool, value_size: int=0, operation: str='get'):
        """Track cache access for coordination decisions.

        Args:
            key: Cache key
            cache_level: Cache level where access occurred
            hit: Whether access was a hit
            value_size: Size of value in bytes
            operation: Operation type (get, set, delete)
        """
        now = datetime.now(UTC)
        if key not in self._entry_metadata:
            self._entry_metadata[key] = CacheEntry(key=key, value=None, size_bytes=value_size, access_count=0, last_access=now, created_at=now)
        entry = self._entry_metadata[key]
        entry.access_count += 1
        entry.last_access = now
        if hit:
            entry.levels_present.add(cache_level)
        self._access_history[key].append({'timestamp': now.timestamp(), 'level': cache_level.value, 'hit': hit, 'operation': operation})
        if hit and operation == 'set':
            self._level_usage[cache_level]['count'] += 1
            self._level_usage[cache_level]['size_bytes'] += value_size
        entry.access_pattern = self._analyze_access_pattern(key)
        entry.promotion_score = self._calculate_promotion_score(entry)
        asyncio.create_task(self._maybe_coordinate(key, entry))

    def _analyze_access_pattern(self, key: str) -> AccessPattern:
        """Analyze access pattern for a cache key."""
        history = list(self._access_history[key])
        if len(history) < 3:
            return AccessPattern.COLD
        now = time.time()
        recent_window = now - self._strategy.access_window.total_seconds()
        recent_accesses = [h for h in history if h['timestamp'] >= recent_window]
        if not recent_accesses:
            return AccessPattern.COLD
        access_count = len(recent_accesses)
        time_span = now - recent_accesses[0]['timestamp']
        frequency = access_count / (time_span / 60) if time_span > 0 else 0
        if frequency > 10:
            return AccessPattern.HOT
        if frequency > 2:
            return AccessPattern.WARM
        if self._detect_burst_pattern(recent_accesses):
            return AccessPattern.BURST
        if self._detect_temporal_pattern(recent_accesses):
            return AccessPattern.TEMPORAL
        return AccessPattern.COLD

    def _detect_burst_pattern(self, accesses: list[dict]) -> bool:
        """Detect burst access pattern."""
        if len(accesses) < 5:
            return False
        timestamps = [a['timestamp'] for a in accesses]
        time_range = max(timestamps) - min(timestamps)
        if time_range > 0:
            burst_window = time_range * 0.2
            burst_start = max(timestamps) - burst_window
            burst_accesses = sum((1 for t in timestamps if t >= burst_start))
            return burst_accesses >= len(accesses) * 0.8
        return False

    def _detect_temporal_pattern(self, accesses: list[dict]) -> bool:
        """Detect temporal access pattern (regular intervals)."""
        if len(accesses) < 4:
            return False
        timestamps = sorted([a['timestamp'] for a in accesses])
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        if len(intervals) < 3:
            return False
        mean_interval = statistics.mean(intervals)
        std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0
        return std_dev < mean_interval * 0.3 if mean_interval > 0 else False

    def _calculate_promotion_score(self, entry: CacheEntry) -> float:
        """Calculate promotion score for cache entry.

        Score factors:
        - Access frequency
        - Recency of access
        - Access pattern type
        - Size efficiency
        """
        now = datetime.now(UTC)
        time_window = self._strategy.access_window.total_seconds()
        recent_accesses = [a for a in self._access_history[entry.key] if now.timestamp() - a['timestamp'] <= time_window]
        frequency_score = min(1.0, len(recent_accesses) / 20.0)
        time_since_access = (now - entry.last_access).total_seconds()
        recency_score = max(0.0, 1.0 - time_since_access / time_window)
        pattern_scores = {AccessPattern.HOT: 1.0, AccessPattern.WARM: 0.7, AccessPattern.BURST: 0.8, AccessPattern.TEMPORAL: 0.6, AccessPattern.COLD: 0.2}
        pattern_score = pattern_scores.get(entry.access_pattern, 0.2)
        size_score = max(0.1, 1.0 - entry.size_bytes / 10000.0)
        weights = {'frequency': 0.4, 'recency': 0.3, 'pattern': 0.2, 'size': 0.1}
        final_score = frequency_score * weights['frequency'] + recency_score * weights['recency'] + pattern_score * weights['pattern'] + size_score * weights['size']
        return final_score

    async def _maybe_coordinate(self, key: str, entry: CacheEntry):
        """Decide if coordination action is needed for a cache entry."""
        if not self._unified_manager:
            return
        if key in self._pending_operations:
            return
        try:
            async with self._coordination_locks[key]:
                action = self._determine_coordination_action(entry)
                if action:
                    await self._execute_coordination_action(key, entry, action)
        except Exception as e:
            logger.error('Coordination failed for key {key}: %s', e)

    def _determine_coordination_action(self, entry: CacheEntry) -> tuple[CoordinationAction, CacheLevel, CacheLevel] | None:
        """Determine what coordination action to take."""
        if entry.promotion_score >= self._strategy.promotion_threshold and entry.access_count >= self._strategy.min_access_count:
            if CacheLevel.L2 in entry.levels_present and CacheLevel.L1 not in entry.levels_present and self._has_l1_capacity():
                return (CoordinationAction.PROMOTE, CacheLevel.L2, CacheLevel.L1)
            if CacheLevel.L3 in entry.levels_present and CacheLevel.L2 not in entry.levels_present and self._has_l2_capacity():
                return (CoordinationAction.PROMOTE, CacheLevel.L3, CacheLevel.L2)
        if entry.promotion_score <= self._strategy.demotion_threshold:
            if CacheLevel.L1 in entry.levels_present and self._needs_l1_space():
                return (CoordinationAction.DEMOTE, CacheLevel.L1, CacheLevel.L2)
            if CacheLevel.L2 in entry.levels_present and self._needs_l2_space():
                return (CoordinationAction.DEMOTE, CacheLevel.L2, CacheLevel.L3)
        return None

    def _has_l1_capacity(self) -> bool:
        """Check if L1 cache has capacity for new entries."""
        return self._level_usage[CacheLevel.L1]['count'] < self._strategy.max_l1_size

    def _has_l2_capacity(self) -> bool:
        """Check if L2 cache has capacity for new entries."""
        return self._level_usage[CacheLevel.L2]['count'] < self._strategy.max_l2_size

    def _needs_l1_space(self) -> bool:
        """Check if L1 cache needs space (for eviction)."""
        return self._level_usage[CacheLevel.L1]['count'] >= self._strategy.max_l1_size * 0.9

    def _needs_l2_space(self) -> bool:
        """Check if L2 cache needs space (for eviction)."""
        return self._level_usage[CacheLevel.L2]['count'] >= self._strategy.max_l2_size * 0.9

    @asynccontextmanager
    async def _trace_coordination(self, action: CoordinationAction, key: str, source_level: CacheLevel, target_level: CacheLevel):
        """Context manager for tracing coordination operations."""
        event_id = f'coord_{action.value}_{int(time.time() * 1000)}'
        span = None
        if OPENTELEMETRY_AVAILABLE and coordination_tracer:
            span = coordination_tracer.start_span(f'cache_coordination_{action.value}', attributes={'coordination.action': action.value, 'coordination.key': key, 'coordination.source_level': source_level.value, 'coordination.target_level': target_level.value, 'coordination.event_id': event_id})
        start_time = time.time()
        success = False
        try:
            yield event_id
            success = True
            if span:
                span.set_status(Status(StatusCode.OK))
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            event = CoordinationEvent(event_id=event_id, action=action, key=key, source_level=source_level, target_level=target_level, success=success, duration_ms=duration_ms)
            self._coordination_events.append(event)
            if OPENTELEMETRY_AVAILABLE:
                labels = {'action': action.value, 'source_level': source_level.value, 'target_level': target_level.value, 'success': str(success).lower()}
                if action == CoordinationAction.PROMOTE and cache_level_promotions:
                    cache_level_promotions.add(1, labels)
                elif action == CoordinationAction.DEMOTE and cache_level_demotions:
                    cache_level_demotions.add(1, labels)
                if cross_level_latency:
                    cross_level_latency.record(duration_ms / 1000.0, labels)
            if span:
                span.end()

    async def _execute_coordination_action(self, key: str, entry: CacheEntry, action_info: tuple[CoordinationAction, CacheLevel, CacheLevel]):
        """Execute a coordination action."""
        action, source_level, target_level = action_info
        async with self._trace_coordination(action, key, source_level, target_level):
            if action == CoordinationAction.PROMOTE:
                await self._execute_promotion(key, entry, source_level, target_level)
            elif action == CoordinationAction.DEMOTE:
                await self._execute_demotion(key, entry, source_level, target_level)
            if action == CoordinationAction.PROMOTE:
                self._coordination_stats['promotions'][f'{source_level.value}_to_{target_level.value}'] += 1
            elif action == CoordinationAction.DEMOTE:
                self._coordination_stats['demotions'][f'{source_level.value}_to_{target_level.value}'] += 1

    async def _execute_promotion(self, key: str, entry: CacheEntry, source_level: CacheLevel, target_level: CacheLevel):
        """Execute cache entry promotion."""
        try:
            value = await self._get_from_level(key, source_level)
            if value is None:
                logger.warning('Could not retrieve value for promotion: %s', key)
                return
            success = await self._set_to_level(key, value, target_level, entry.ttl_seconds)
            if success:
                entry.levels_present.add(target_level)
                logger.debug('Promoted %s from %s to %s', key, source_level.value, target_level.value)
            else:
                logger.warning('Failed to promote {key} to %s', target_level.value)
        except Exception as e:
            logger.error('Promotion execution failed for {key}: %s', e)
            raise

    async def _execute_demotion(self, key: str, entry: CacheEntry, source_level: CacheLevel, target_level: CacheLevel):
        """Execute cache entry demotion."""
        try:
            value = await self._get_from_level(key, source_level)
            if value is None:
                logger.warning('Could not retrieve value for demotion: %s', key)
                return
            success = await self._set_to_level(key, value, target_level, entry.ttl_seconds)
            if success:
                await self._delete_from_level(key, source_level)
                entry.levels_present.discard(source_level)
                entry.levels_present.add(target_level)
                logger.debug('Demoted %s from %s to %s', key, source_level.value, target_level.value)
            else:
                logger.warning('Failed to demote {key} to %s', target_level.value)
        except Exception as e:
            logger.error('Demotion execution failed for {key}: %s', e)
            raise

    async def _get_from_level(self, key: str, level: CacheLevel) -> Any:
        """Get value from specific cache level."""
        if not self._unified_manager:
            return None
        try:
            if level == CacheLevel.L1:
                if hasattr(self._unified_manager, '_l1_cache') and self._unified_manager._l1_cache:
                    return self._unified_manager._l1_cache.get(key)
            elif level == CacheLevel.L2:
                if hasattr(self._unified_manager, '_redis_master') and self._unified_manager._redis_master:
                    value = await self._unified_manager._redis_master.get(key)
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            elif level == CacheLevel.L3:
                pass
        except Exception as e:
            logger.error('Error getting {key} from {level.value}: %s', e)
        return None

    async def _set_to_level(self, key: str, value: Any, level: CacheLevel, ttl: int | None=None) -> bool:
        """Set value to specific cache level."""
        if not self._unified_manager:
            return False
        try:
            if level == CacheLevel.L1:
                if hasattr(self._unified_manager, '_l1_cache') and self._unified_manager._l1_cache:
                    self._unified_manager._l1_cache.set(key, value, ttl)
                    return True
            elif level == CacheLevel.L2:
                if hasattr(self._unified_manager, '_redis_master') and self._unified_manager._redis_master:
                    if ttl:
                        await self._unified_manager._redis_master.setex(key, ttl, str(value))
                    else:
                        await self._unified_manager._redis_master.set(key, str(value))
                    return True
            elif level == CacheLevel.L3:
                pass
        except Exception as e:
            logger.error('Error setting {key} to {level.value}: %s', e)
        return False

    async def _delete_from_level(self, key: str, level: CacheLevel) -> bool:
        """Delete value from specific cache level."""
        if not self._unified_manager:
            return False
        try:
            if level == CacheLevel.L1:
                if hasattr(self._unified_manager, '_l1_cache') and self._unified_manager._l1_cache:
                    return self._unified_manager._l1_cache.delete(key)
            elif level == CacheLevel.L2:
                if hasattr(self._unified_manager, '_redis_master') and self._unified_manager._redis_master:
                    deleted = await self._unified_manager._redis_master.delete(key)
                    return deleted > 0
            elif level == CacheLevel.L3:
                pass
        except Exception as e:
            logger.error('Error deleting {key} from {level.value}: %s', e)
        return False

    async def check_cache_coherence(self, keys: list[str] | None=None) -> dict[str, list[str]]:
        """Check cache coherence across levels.

        Args:
            keys: Specific keys to check (default: check all tracked keys)

        Returns:
            Dictionary of coherence violations by key
        """
        if keys is None:
            keys = list(self._entry_metadata.keys())
        violations = {}
        for key in keys:
            try:
                key_violations = await self._check_key_coherence(key)
                if key_violations:
                    violations[key] = key_violations
            except Exception as e:
                logger.error('Coherence check failed for {key}: %s', e)
        self._coordination_stats['coherence_checks'] += len(keys)
        self._coordination_stats['coherence_violations'] += len(violations)
        if OPENTELEMETRY_AVAILABLE and cache_coherence_violations:
            for key, key_violations in violations.items():
                cache_coherence_violations.add(len(key_violations), {'violation_type': 'coherence_mismatch'})
        return violations

    async def _check_key_coherence(self, key: str) -> list[str]:
        """Check coherence for a specific key across cache levels."""
        violations = []
        l1_value = await self._get_from_level(key, CacheLevel.L1)
        l2_value = await self._get_from_level(key, CacheLevel.L2)
        l3_value = await self._get_from_level(key, CacheLevel.L3)
        values = {CacheLevel.L1: l1_value, CacheLevel.L2: l2_value, CacheLevel.L3: l3_value}
        present_values = {level: value for level, value in values.items() if value is not None}
        if len(present_values) > 1:
            unique_values = set((str(v) for v in present_values.values()))
            if len(unique_values) > 1:
                violation_msg = f'Value mismatch across levels: {dict(present_values)}'
                violations.append(violation_msg)
        return violations

    async def repair_cache_coherence(self, key: str) -> bool:
        """Repair cache coherence for a specific key.

        Strategy: Use the value from the highest level (L1 > L2 > L3)
        and propagate it to other levels.
        """
        try:
            authoritative_value = None
            authoritative_level = None
            for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
                value = await self._get_from_level(key, level)
                if value is not None:
                    authoritative_value = value
                    authoritative_level = level
                    break
            if authoritative_value is None:
                logger.warning('No value found for coherence repair: %s', key)
                return False
            entry = self._entry_metadata.get(key)
            if not entry:
                return False
            success = True
            for level in entry.levels_present:
                if level != authoritative_level:
                    set_success = await self._set_to_level(key, authoritative_value, level, entry.ttl_seconds)
                    if not set_success:
                        success = False
                        logger.warning('Failed to repair coherence for %s at %s', key, level.value)
            logger.info('Repaired cache coherence for %s using value from %s', key, authoritative_level.value)
            return success
        except Exception as e:
            logger.error('Coherence repair failed for {key}: %s', e)
            return False

    def get_coordination_stats(self) -> dict[str, Any]:
        """Get comprehensive coordination statistics."""
        total_actions = sum(self._coordination_stats['promotions'].values()) + sum(self._coordination_stats['demotions'].values())
        successful_events = sum((1 for event in self._coordination_events if event.success))
        efficiency = successful_events / len(self._coordination_events) if self._coordination_events else 1.0
        self._coordination_stats['coordination_efficiency'] = efficiency
        if OPENTELEMETRY_AVAILABLE and cache_coordination_efficiency:
            cache_coordination_efficiency.set(efficiency)
        pattern_distribution = defaultdict(int)
        for entry in self._entry_metadata.values():
            pattern_distribution[entry.access_pattern.value] += 1
        level_distribution = defaultdict(int)
        for entry in self._entry_metadata.values():
            for level in entry.levels_present:
                level_distribution[level.value] += 1
        return {'coordination_events': {'total_events': len(self._coordination_events), 'successful_events': successful_events, 'efficiency': efficiency}, 'promotions': dict(self._coordination_stats['promotions']), 'demotions': dict(self._coordination_stats['demotions']), 'coherence': {'checks_performed': self._coordination_stats['coherence_checks'], 'violations_detected': self._coordination_stats['coherence_violations'], 'violation_rate': self._coordination_stats['coherence_violations'] / max(1, self._coordination_stats['coherence_checks'])}, 'cache_levels': {'usage': dict(self._level_usage), 'distribution': dict(level_distribution)}, 'access_patterns': {'distribution': dict(pattern_distribution), 'total_tracked_entries': len(self._entry_metadata)}, 'performance': {'avg_coordination_time_ms': statistics.mean([e.duration_ms for e in self._coordination_events]) if self._coordination_events else 0}}

    async def integrate_with_unified_manager(self, unified_manager):
        """Integrate coordinator with UnifiedConnectionManager."""
        self._unified_manager = unified_manager
        from prompt_improver.performance.monitoring.health.background_manager import TaskPriority, get_background_task_manager
        task_manager = get_background_task_manager()
        if task_manager:
            await task_manager.submit_enhanced_task(task_id='cache_coherence_check', coroutine=self._periodic_coherence_check(), priority=TaskPriority.BACKGROUND, tags={'service': 'cache_coordination', 'type': 'coherence_check', 'component': 'cross_level_coordinator'})
            await task_manager.submit_enhanced_task(task_id='cache_level_optimization', coroutine=self._periodic_optimization(), priority=TaskPriority.LOW, tags={'service': 'cache_coordination', 'type': 'optimization', 'component': 'cross_level_coordinator'})
        logger.info('CrossLevelCoordinator integrated with UnifiedConnectionManager')

    async def _periodic_coherence_check(self):
        """Periodic coherence checking task."""
        while True:
            try:
                violations = await self.check_cache_coherence()
                if violations:
                    logger.warning('Detected %s coherence violations', len(violations))
                    for key in violations.keys():
                        await self.repair_cache_coherence(key)
                await asyncio.sleep(self._strategy.coherence_timeout)
            except Exception as e:
                logger.error('Periodic coherence check failed: %s', e)
                await asyncio.sleep(30)

    async def _periodic_optimization(self):
        """Periodic cache level optimization task."""
        while True:
            try:
                await self._cleanup_old_entries()
                await self._optimize_cache_distribution()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error('Periodic optimization failed: %s', e)
                await asyncio.sleep(60)

    async def _cleanup_old_entries(self):
        """Clean up old entry metadata."""
        now = datetime.now(UTC)
        cleanup_threshold = now - timedelta(hours=1)
        keys_to_remove = []
        for key, entry in self._entry_metadata.items():
            if entry.last_access < cleanup_threshold and entry.access_count < 5:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._entry_metadata[key]
            if key in self._access_history:
                del self._access_history[key]
        if keys_to_remove:
            logger.debug('Cleaned up %s old cache entries', len(keys_to_remove))

    async def _optimize_cache_distribution(self):
        """Optimize cache distribution across levels."""
        for key, entry in self._entry_metadata.items():
            if CacheLevel.L1 in entry.levels_present and entry.promotion_score < self._strategy.demotion_threshold:
                await self._maybe_coordinate(key, entry)
_cross_level_coordinator: CrossLevelCoordinator | None = None

def get_cross_level_coordinator() -> CrossLevelCoordinator:
    """Get or create global cross-level coordinator instance."""
    global _cross_level_coordinator
    if _cross_level_coordinator is None:
        _cross_level_coordinator = CrossLevelCoordinator()
    return _cross_level_coordinator

def integrate_cross_level_coordination(unified_manager):
    """Integrate cross-level coordination with UnifiedConnectionManager.

    Args:
        unified_manager: UnifiedConnectionManager instance
    """
    coordinator = get_cross_level_coordinator()
    asyncio.create_task(coordinator.integrate_with_unified_manager(unified_manager))
