"""
Integration module for Phase 2 database optimizations.

Coordinates query caching, connection pooling, and performance monitoring
to achieve 50% database load reduction.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .cache_layer import DatabaseCacheLayer, CachePolicy, CacheStrategy
from .connection_pool_optimizer import get_connection_pool_optimizer
from .query_optimizer import get_query_executor
from .performance_monitor import get_performance_monitor
from ..ml.orchestration.events.event_bus import get_event_bus

logger = logging.getLogger(__name__)

class DatabaseOptimizationIntegration:
    """
    Coordinates all database optimization components for Phase 2.
    
    Integrates:
    - Query result caching
    - Connection pool optimization
    - Performance monitoring
    - Event-driven coordination
    """
    
    def __init__(self):
        self.cache_layer = None
        self.pool_optimizer = None
        self.query_executor = None
        self.performance_monitor = None
        self.event_bus = None
        self._initialized = False
        self._monitoring_task = None
    
    async def initialize(self, cache_policy: Optional[CachePolicy] = None):
        """Initialize all optimization components"""
        if self._initialized:
            return
        
        logger.info("Initializing database optimization components...")
        
        # Get event bus for coordination
        self.event_bus = get_event_bus()
        
        # Initialize cache layer with custom policy
        self.cache_layer = DatabaseCacheLayer(
            cache_policy or CachePolicy(
                ttl_seconds=300,  # 5 minutes default
                strategy=CacheStrategy.SMART,
                warm_on_startup=True
            )
        )
        
        # Initialize connection pool optimizer
        self.pool_optimizer = get_connection_pool_optimizer()
        
        # Initialize query executor (already has caching)
        self.query_executor = get_query_executor()
        
        # Initialize performance monitor with event bus
        self.performance_monitor = await get_performance_monitor(self.event_bus)
        
        # Subscribe to relevant events
        await self._setup_event_handlers()
        
        self._initialized = True
        logger.info("Database optimization components initialized")
    
    async def _setup_event_handlers(self):
        """Set up event handlers for coordination"""
        from ..ml.orchestration.events.event_types import EventType
        
        # Handle database performance events
        async def handle_slow_queries(event):
            if event.data.get("avg_query_time_ms", 0) > 100:
                # Aggressive caching for slow queries
                logger.warning("Slow queries detected, switching to aggressive caching")
                self.cache_layer.policy.strategy = CacheStrategy.AGGRESSIVE
        
        async def handle_cache_hit_ratio_low(event):
            if event.data.get("cache_hit_ratio", 100) < 50:
                # Optimize cache policy
                logger.warning("Low cache hit ratio, optimizing cache policy")
                await self.cache_layer.optimize_cache_policy()
        
        async def handle_pool_stressed(event):
            # Optimize pool size
            logger.warning("Connection pool stressed, optimizing pool size")
            await self.pool_optimizer.optimize_pool_size()
        
        # Subscribe to events
        self.event_bus.subscribe(EventType.DATABASE_SLOW_QUERY_DETECTED, handle_slow_queries)
        self.event_bus.subscribe(EventType.DATABASE_CACHE_HIT_RATIO_LOW, handle_cache_hit_ratio_low)
        self.event_bus.subscribe(EventType.DATABASE_PERFORMANCE_DEGRADED, handle_pool_stressed)
    
    async def start_optimization(self):
        """Start all optimization processes"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Starting database optimization processes...")
        
        # Implement connection multiplexing
        multiplex_result = await self.pool_optimizer.implement_connection_multiplexing()
        logger.info(f"Connection multiplexing: {multiplex_result}")
        
        # Warm cache with common queries
        await self._warm_cache()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
        
        logger.info("Database optimization started")
    
    async def _warm_cache(self):
        """Warm cache with frequently used queries"""
        from .unified_connection_manager import get_unified_manager, ManagerMode
        
        client = get_unified_manager(ManagerMode.ASYNC_MODERN)
        
        # Common queries to warm up
        warm_queries = [
            ("SELECT * FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 20", {}),
            ("SELECT COUNT(*) FROM sessions WHERE created_at > NOW() - INTERVAL '24 hours'", {}),
            ("SELECT id, name FROM rules ORDER BY usage_count DESC LIMIT 10", {}),
        ]
        
        # Execute function
        async def executor(query, params):
            return await client.fetch_raw(query, params)
        
        # Warm the cache
        warm_list = [(q, p, executor) for q, p in warm_queries]
        await self.cache_layer.warm_cache(warm_list)
    
    async def _continuous_monitoring(self):
        """Continuous monitoring and optimization"""
        while True:
            try:
                # Take performance snapshot
                snapshot = await self.performance_monitor.take_performance_snapshot()
                
                # Check if optimization is needed
                if snapshot.avg_query_time_ms > 75:  # Getting close to 100ms threshold
                    await self.pool_optimizer.optimize_pool_size()
                
                if snapshot.cache_hit_ratio < 70:  # Below optimal
                    await self.cache_layer.optimize_cache_policy()
                
                # Monitor every 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        # Collect all metrics
        cache_stats = await self.cache_layer.get_cache_stats()
        pool_stats = await self.pool_optimizer.get_optimization_summary()
        executor_stats = await self.query_executor.get_performance_summary()
        perf_summary = await self.performance_monitor.get_performance_summary(hours=1)
        
        # Calculate overall database load reduction
        # Each component contributes to load reduction
        cache_reduction = cache_stats.get("database_load_reduction_percent", 0)
        pool_reduction = pool_stats["optimization"]["database_load_reduction_percent"]
        
        # Combined effect (not simply additive)
        overall_reduction = cache_reduction + (pool_reduction * (100 - cache_reduction) / 100)
        
        return {
            "status": "optimized" if overall_reduction >= 45 else "optimizing",
            "database_load_reduction_percent": round(overall_reduction, 1),
            "target_achieved": overall_reduction >= 50,
            "cache_performance": {
                "hit_rate": cache_stats["cache_hit_rate"],
                "queries_cached": cache_stats["queries_cached"],
                "db_calls_avoided": cache_stats["db_calls_avoided"],
                "mb_saved": cache_stats["mb_saved"]
            },
            "pool_performance": {
                "utilization": pool_stats["pool_metrics"]["utilization_percent"],
                "efficiency": pool_stats["performance"]["pool_efficiency_percent"],
                "connections_saved": pool_stats["optimization"]["connections_saved"],
                "health": pool_stats["current_state"]
            },
            "query_performance": {
                "avg_query_time_ms": executor_stats.get("avg_execution_time_ms", 0),
                "cache_hit_rate": executor_stats.get("cache_hit_rate", 0),
                "queries_meeting_target": executor_stats.get("target_compliance_rate", 0) * 100
            },
            "system_performance": {
                "cache_hit_ratio": perf_summary.get("avg_cache_hit_ratio", 0),
                "avg_query_time": perf_summary.get("avg_query_time_ms", 0),
                "performance_status": perf_summary.get("performance_status", "UNKNOWN")
            }
        }
    
    async def stop_optimization(self):
        """Stop optimization processes"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.pool_optimizer and self.pool_optimizer._monitoring:
            self.pool_optimizer.stop_monitoring()
        
        if self.performance_monitor and self.performance_monitor._monitoring:
            self.performance_monitor.stop_monitoring()
        
        logger.info("Database optimization stopped")

# Global integration instance
_optimization_integration: Optional[DatabaseOptimizationIntegration] = None

async def get_optimization_integration() -> DatabaseOptimizationIntegration:
    """Get or create global optimization integration"""
    global _optimization_integration
    if _optimization_integration is None:
        _optimization_integration = DatabaseOptimizationIntegration()
        await _optimization_integration.initialize()
    return _optimization_integration

async def start_database_optimization():
    """Start database optimization (convenience function)"""
    integration = await get_optimization_integration()
    await integration.start_optimization()
    return integration

async def get_database_optimization_status() -> Dict[str, Any]:
    """Get database optimization status (convenience function)"""
    integration = await get_optimization_integration()
    return await integration.get_optimization_status()