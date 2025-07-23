"""
Database event handler for ML pipeline orchestration.

Handles database performance monitoring, optimization, and health events
to ensure optimal database performance for ML operations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..event_types import EventType, MLEvent


class DatabaseEventHandler:
    """
    Handles database-related events in the ML pipeline orchestration.
    
    Responds to database performance events and coordinates optimization
    actions across database components.
    """
    
    def __init__(self, orchestrator=None, event_bus=None):
        """Initialize the database event handler."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Event handling statistics
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time: Optional[datetime] = None
        
        # Database monitoring state
        self.performance_snapshots = []
        self.optimization_history = []
        self.alert_thresholds = {
            "cache_hit_ratio_min": 90.0,
            "query_time_max_ms": 50.0,
            "connection_count_max": 20
        }
        
    async def handle_event(self, event: MLEvent) -> None:
        """Handle a database-related event."""
        self.logger.debug(f"Handling database event: {event.event_type}")
        
        try:
            self.events_processed += 1
            self.last_event_time = datetime.now(timezone.utc)
            
            # Route to specific handler based on event type
            if event.event_type == EventType.DATABASE_PERFORMANCE_SNAPSHOT_TAKEN:
                await self._handle_performance_snapshot(event)
            elif event.event_type == EventType.DATABASE_CACHE_HIT_RATIO_LOW:
                await self._handle_cache_hit_ratio_low(event)
            elif event.event_type == EventType.DATABASE_SLOW_QUERY_DETECTED:
                await self._handle_slow_query_detected(event)
            elif event.event_type == EventType.DATABASE_CONNECTION_OPTIMIZED:
                await self._handle_connection_optimized(event)
            elif event.event_type == EventType.DATABASE_INDEXES_CREATED:
                await self._handle_indexes_created(event)
            elif event.event_type == EventType.DATABASE_PERFORMANCE_DEGRADED:
                await self._handle_performance_degraded(event)
            elif event.event_type == EventType.DATABASE_PERFORMANCE_IMPROVED:
                await self._handle_performance_improved(event)
            elif event.event_type == EventType.DATABASE_RESOURCE_ANALYSIS_COMPLETED:
                await self._handle_resource_analysis_completed(event)
            else:
                self.logger.warning(f"Unknown database event type: {event.event_type}")
                
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Error handling database event {event.event_type}: {e}")
            
            # Emit error event for monitoring
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.ALERT_TRIGGERED,
                    source="database_handler",
                    data={
                        "error": str(e),
                        "original_event": event.event_type.value,
                        "severity": "error"
                    }
                ))
    
    async def _handle_performance_snapshot(self, event: MLEvent) -> None:
        """Handle database performance snapshot taken event."""
        snapshot_data = event.data.get("snapshot", {})
        self.performance_snapshots.append({
            "timestamp": event.data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "data": snapshot_data
        })
        
        # Keep only last 100 snapshots
        if len(self.performance_snapshots) > 100:
            self.performance_snapshots.pop(0)
        
        # Check for performance issues
        cache_hit_ratio = snapshot_data.get("cache_hit_ratio", 100.0)
        avg_query_time = snapshot_data.get("avg_query_time_ms", 0.0)
        active_connections = snapshot_data.get("active_connections", 0)
        
        # Emit alerts if thresholds are exceeded
        if cache_hit_ratio < self.alert_thresholds["cache_hit_ratio_min"]:
            await self._emit_cache_hit_ratio_alert(cache_hit_ratio)
        
        if avg_query_time > self.alert_thresholds["query_time_max_ms"]:
            await self._emit_slow_query_alert(avg_query_time)
        
        if active_connections > self.alert_thresholds["connection_count_max"]:
            await self._emit_connection_count_alert(active_connections)
        
        self.logger.debug(f"Processed performance snapshot: cache_hit={cache_hit_ratio}%, query_time={avg_query_time}ms")
    
    async def _handle_cache_hit_ratio_low(self, event: MLEvent) -> None:
        """Handle low cache hit ratio event."""
        cache_ratio = event.data.get("cache_hit_ratio", 0.0)
        self.logger.warning(f"Low cache hit ratio detected: {cache_ratio}%")
        
        # Trigger optimization if available
        if self.orchestrator and hasattr(self.orchestrator, 'get_component'):
            try:
                optimizer = await self.orchestrator.get_component("database_connection_optimizer")
                if optimizer:
                    await optimizer.execute_capability("optimize_connection_settings", {})
                    self.logger.info("Triggered database optimization due to low cache hit ratio")
            except Exception as e:
                self.logger.error(f"Failed to trigger optimization: {e}")
    
    async def _handle_slow_query_detected(self, event: MLEvent) -> None:
        """Handle slow query detected event."""
        query_info = event.data.get("query_info", {})
        execution_time = query_info.get("execution_time_ms", 0.0)
        
        self.logger.warning(f"Slow query detected: {execution_time}ms")
        
        # Log slow query for analysis
        slow_query_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_text": query_info.get("query_text", ""),
            "execution_time_ms": execution_time,
            "calls": query_info.get("calls", 1)
        }
        
        # Emit alert for monitoring systems
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ALERT_TRIGGERED,
                source="database_handler",
                data={
                    "alert_type": "slow_query",
                    "query_data": slow_query_data,
                    "severity": "warning"
                }
            ))
    
    async def _handle_connection_optimized(self, event: MLEvent) -> None:
        """Handle database connection optimized event."""
        optimization_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_type": "connection_settings",
            "applied_settings": event.data.get("applied_settings", {}),
            "system_resources": event.data.get("system_resources", {})
        }
        
        self.optimization_history.append(optimization_data)
        
        # Keep only last 50 optimization records
        if len(self.optimization_history) > 50:
            self.optimization_history.pop(0)
        
        self.logger.info("Database connection settings optimized")
    
    async def _handle_indexes_created(self, event: MLEvent) -> None:
        """Handle database indexes created event."""
        index_data = event.data.get("index_data", {})
        self.logger.info(f"Performance indexes created: {index_data}")
        
        # Record index creation for tracking
        optimization_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_type": "index_creation",
            "indexes_created": index_data.get("indexes_created", []),
            "performance_impact": index_data.get("performance_impact", {})
        }
        
        self.optimization_history.append(optimization_data)
    
    async def _handle_performance_degraded(self, event: MLEvent) -> None:
        """Handle database performance degraded event."""
        degradation_data = event.data.get("degradation_data", {})
        self.logger.error(f"Database performance degraded: {degradation_data}")
        
        # Trigger immediate optimization
        if self.orchestrator and hasattr(self.orchestrator, 'get_component'):
            try:
                optimizer = await self.orchestrator.get_component("database_connection_optimizer")
                monitor = await self.orchestrator.get_component("database_performance_monitor")
                
                if optimizer and monitor:
                    # Run system analysis
                    analysis_result = await optimizer.execute_capability("system_resource_analysis", {})
                    
                    # Take performance snapshot
                    snapshot_result = await monitor.execute_capability("real_time_monitoring", {})
                    
                    self.logger.info("Triggered emergency database optimization")
                    
            except Exception as e:
                self.logger.error(f"Failed to trigger emergency optimization: {e}")
    
    async def _handle_performance_improved(self, event: MLEvent) -> None:
        """Handle database performance improved event."""
        improvement_data = event.data.get("improvement_data", {})
        self.logger.info(f"Database performance improved: {improvement_data}")
    
    async def _handle_resource_analysis_completed(self, event: MLEvent) -> None:
        """Handle database resource analysis completed event."""
        analysis_data = event.data.get("analysis_data", {})
        self.logger.info(f"Database resource analysis completed: {analysis_data}")
    
    async def _emit_cache_hit_ratio_alert(self, cache_ratio: float) -> None:
        """Emit cache hit ratio alert."""
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATABASE_CACHE_HIT_RATIO_LOW,
                source="database_handler",
                data={
                    "cache_hit_ratio": cache_ratio,
                    "threshold": self.alert_thresholds["cache_hit_ratio_min"],
                    "severity": "warning"
                }
            ))
    
    async def _emit_slow_query_alert(self, avg_query_time: float) -> None:
        """Emit slow query alert."""
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATABASE_SLOW_QUERY_DETECTED,
                source="database_handler",
                data={
                    "avg_query_time_ms": avg_query_time,
                    "threshold": self.alert_thresholds["query_time_max_ms"],
                    "severity": "warning"
                }
            ))
    
    async def _emit_connection_count_alert(self, connection_count: int) -> None:
        """Emit connection count alert."""
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.ALERT_TRIGGERED,
                source="database_handler",
                data={
                    "alert_type": "high_connection_count",
                    "connection_count": connection_count,
                    "threshold": self.alert_thresholds["connection_count_max"],
                    "severity": "warning"
                }
            ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database event handler statistics."""
        return {
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "performance_snapshots_count": len(self.performance_snapshots),
            "optimization_history_count": len(self.optimization_history),
            "alert_thresholds": self.alert_thresholds
        }
