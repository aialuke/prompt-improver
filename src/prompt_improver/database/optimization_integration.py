"""Integration module for Phase 2 database optimizations.

Coordinates query caching, connection pooling, and performance monitoring
to achieve 50% database load reduction.
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from prompt_improver.core.di.ml_container import MLServiceContainer

from prompt_improver.core.protocols.ml_protocols import (
    DatabaseServiceProtocol,
    MLflowServiceProtocol,
)
from prompt_improver.database import (
    ManagerMode,
    get_database_services,
)
from prompt_improver.database.cache_layer import (
    CachePolicy,
    CacheStrategy,
    DatabaseCacheLayer,
)
from prompt_improver.database.performance_monitor import get_performance_monitor
from prompt_improver.services.cache.cache_facade import CacheFacade
# Removed ML orchestration import - using optional service registry pattern instead
from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)

logger = logging.getLogger(__name__)


class DatabaseOptimizationIntegration:
    """Coordinates all database optimization components for Phase 2.

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
        self._monitoring_task_id = None
        self._ml_container: Optional["MLServiceContainer"] = None

    async def initialize(self, cache_policy: CachePolicy | None = None):
        """Initialize all optimization components"""
        if self._initialized:
            return
        logger.info("Initializing database optimization components...")
        # Use optional service registry instead of direct ML event bus
        self.cache_layer = DatabaseCacheLayer(
            cache_policy
            or CachePolicy(
                ttl_seconds=300, strategy=CacheStrategy.SMART, warm_on_startup=True
            )
        )
        self.pool_optimizer = await get_database_services(ManagerMode.ASYNC_MODERN)
        self.query_cache = CacheFacade(l1_max_size=500, l2_default_ttl=600, enable_l2=True, enable_l3=False)
        self.performance_monitor = await get_performance_monitor(None)  # No event bus dependency
        await self._setup_event_handlers()
        self._initialized = True
        logger.info("Database optimization components initialized")

    async def _setup_event_handlers(self):
        """Set up optimization event handlers using protocol-based approach."""
        from prompt_improver.database.services.optional_registry import get_optional_services_registry
        from prompt_improver.database.protocols.events import OptimizationEventProtocol

        class DatabaseOptimizationHandler:
            """Handles database optimization events without ML dependencies."""
            
            def __init__(self, integration_instance):
                self.integration = integration_instance
                
            async def handle_performance_degradation(self, metrics):
                """Handle performance degradation by optimizing various components."""
                avg_query_time = metrics.get("avg_query_time_ms", 0)
                cache_hit_ratio = metrics.get("cache_hit_ratio", 100)
                
                if avg_query_time > 100:
                    logger.warning("Slow queries detected, switching to aggressive caching")
                    if hasattr(self.integration.cache_layer, 'policy'):
                        self.integration.cache_layer.policy.strategy = CacheStrategy.AGGRESSIVE
                        
                if cache_hit_ratio < 50:
                    logger.warning("Low cache hit ratio, optimizing cache policy")
                    if hasattr(self.integration.cache_layer, 'optimize_cache_policy'):
                        await self.integration.cache_layer.optimize_cache_policy()
                        
            async def handle_cache_optimization(self, cache_stats):
                """Handle cache optimization requests."""
                logger.info("Cache optimization requested")
                if hasattr(self.integration.cache_layer, 'optimize_cache_policy'):
                    await self.integration.cache_layer.optimize_cache_policy()
                    
            async def handle_query_optimization(self, query_data):
                """Handle query optimization requests."""
                logger.info("Query optimization requested")
                if hasattr(self.integration.pool_optimizer, 'optimize_pool_size'):
                    await self.integration.pool_optimizer.optimize_pool_size()

        # Register our optimization handler with the optional services registry
        handler = DatabaseOptimizationHandler(self)
        registry = get_optional_services_registry()
        registry.register_optimization_handler(handler)
        
        logger.info("Database optimization event handlers registered successfully")

    async def start_optimization(self):
        """Start all optimization processes"""
        if not self._initialized:
            await self.initialize()
        logger.info("Starting database optimization processes...")
        multiplex_result = await self.pool_optimizer.optimize_pool_size()
        logger.info(f"Connection pool optimization: {multiplex_result}")
        await self._warm_cache()
        task_manager = get_background_task_manager()
        self._monitoring_task_id = await task_manager.submit_enhanced_task(
            task_id=f"db_optimization_monitoring_{str(uuid.uuid4())[:8]}",
            coroutine=self._continuous_monitoring(),
            priority=TaskPriority.HIGH,
            tags={
                "service": "database",
                "type": "optimization",
                "component": "continuous_monitoring",
                "operation": "performance_monitoring",
            },
        )
        logger.info("Database optimization started")

    async def _warm_cache(self):
        """Warm cache with frequently used queries"""
        from prompt_improver.database import (
            ManagerMode,
            get_database_services,
        )

        client = await get_database_services(ManagerMode.ASYNC_MODERN)
        warm_queries = [
            (
                "SELECT * FROM rules WHERE active = true ORDER BY created_at DESC LIMIT 20",
                {},
            ),
            (
                "SELECT COUNT(*) FROM sessions WHERE created_at > NOW() - INTERVAL '24 hours'",
                {},
            ),
            ("SELECT id, name FROM rules ORDER BY usage_count DESC LIMIT 10", {}),
        ]

        async def executor(query, params):
            return await client.fetch_raw(query, params)

        warm_list = [(q, p, executor) for q, p in warm_queries]
        await self.cache_layer.warm_cache(warm_list)

    async def _continuous_monitoring(self):
        """Continuous monitoring and optimization"""
        while True:
            try:
                snapshot = await self.performance_monitor.take_performance_snapshot()
                if snapshot.avg_query_time_ms > 75:
                    await self.pool_optimizer.optimize_pool_size()
                if snapshot.cache_hit_ratio < 70:
                    await self.cache_layer.optimize_cache_policy()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)

    async def get_optimization_status(self) -> dict[str, Any]:
        """Get comprehensive optimization status"""
        if not self._initialized:
            return {"error": "Not initialized"}
        cache_stats = await self.cache_layer.get_cache_stats()
        pool_stats = await self.pool_optimizer.get_health_status()
        query_cache_stats = self.query_cache.get_performance_stats()  # Use unified cache metrics
        perf_summary = await self.performance_monitor.get_performance_summary(hours=1)
        cache_reduction = cache_stats.get("database_load_reduction_percent", 0)
        pool_reduction = pool_stats["optimization"]["database_load_reduction_percent"]
        overall_reduction = (
            cache_reduction + pool_reduction * (100 - cache_reduction) / 100
        )
        return {
            "status": "optimized" if overall_reduction >= 45 else "optimizing",
            "database_load_reduction_percent": round(overall_reduction, 1),
            "target_achieved": overall_reduction >= 50,
            "cache_performance": {
                "hit_rate": cache_stats["cache_hit_rate"],
                "queries_cached": cache_stats["queries_cached"],
                "db_calls_avoided": cache_stats["db_calls_avoided"],
                "mb_saved": cache_stats["mb_saved"],
            },
            "pool_performance": {
                "utilization": pool_stats["pool_metrics"]["utilization_percent"],
                "efficiency": pool_stats["performance"]["pool_efficiency_percent"],
                "connections_saved": pool_stats["optimization"]["connections_saved"],
                "health": pool_stats["current_state"],
            },
            "query_performance": {
                "avg_query_time_ms": query_cache_stats.get("avg_response_time_ms", 0),
                "cache_hit_rate": query_cache_stats.get("hit_rate", 0),
                "queries_meeting_target": query_cache_stats.get("l1_hits", 0) / max(1, query_cache_stats.get("total_requests", 1)) * 100,
            },
            "system_performance": {
                "cache_hit_ratio": perf_summary.get("avg_cache_hit_ratio", 0),
                "avg_query_time": perf_summary.get("avg_query_time_ms", 0),
                "performance_status": perf_summary.get("performance_status", "UNKNOWN"),
            },
        }

    async def stop_optimization(self):
        """Stop optimization processes"""
        if self._monitoring_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._monitoring_task_id)
            self._monitoring_task_id = None
        if self.pool_optimizer and hasattr(self.pool_optimizer, "_monitoring"):
            logger.info("Pool monitoring handled by DatabaseServices")
        if self.performance_monitor and self.performance_monitor._monitoring:
            self.performance_monitor.stop_monitoring()
        logger.info("Database optimization stopped")


_optimization_integration: DatabaseOptimizationIntegration | None = None


def _get_ml_container_optional() -> Optional["MLServiceContainer"]:
    """Get ML container with lazy loading to avoid torch dependencies."""
    try:
        from prompt_improver.core.di.ml_container import MLServiceContainer
        return MLServiceContainer()
    except ImportError:
        logger.info("ML container not available (torch not installed)")
        return None
    except Exception as e:
        logger.warning(f"ML container initialization failed: {e}")
        return None


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


async def get_database_optimization_status() -> dict[str, Any]:
    """Get database optimization status (convenience function)"""
    integration = await get_optimization_integration()
    return await integration.get_optimization_status()
