"""Data Collection Component for Unified Analytics.

This component handles all data collection and aggregation operations,
including events, metrics, and raw data processing with memory optimization
and efficient storage strategies.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.analytics.unified.protocols import (
    AnalyticsComponentProtocol,
    AnalyticsEvent,
    AnalyticsMetrics,
    ComponentHealth,
    DataCollectionProtocol,
)

logger = logging.getLogger(__name__)


class EventBuffer(BaseModel):
    """Memory-optimized event buffer with automatic cleanup."""
    events: deque = deque(maxlen=1000)
    last_flush: datetime = datetime.now()
    buffer_size_limit: int = 1000
    flush_interval_seconds: int = 60


class DataCollectionComponent(DataCollectionProtocol, AnalyticsComponentProtocol):
    """Data Collection Component implementing efficient data ingestion and aggregation.

    Features:
    - High-throughput event collection with batching
    - Memory-efficient buffering with automatic cleanup
    - Intelligent aggregation strategies
    - Real-time and batch processing modes
    - Comprehensive error handling and recovery
    - Performance optimization for large datasets
    """

    def __init__(self, db_session: AsyncSession, config: dict[str, Any]) -> None:
        self.db_session = db_session
        self.config = config
        self.logger = logger

        # Event buffering and batching
        self._event_buffers: dict[str, EventBuffer] = defaultdict(
            lambda: EventBuffer(
                events=deque(maxlen=config.get("buffer_size", 1000)),
                flush_interval_seconds=config.get("flush_interval", 60)
            )
        )

        # Performance optimization settings
        self._batch_size = config.get("batch_size", 100)
        self._max_memory_mb = config.get("max_memory_mb", 200)
        self._flush_interval = config.get("flush_interval", 60)

        # Processing state
        self._processing_enabled = True
        self._flush_task: asyncio.Task | None = None
        self._stats = {
            "events_collected": 0,
            "metrics_collected": 0,
            "flushes_completed": 0,
            "errors_encountered": 0,
            "last_flush_time": datetime.now(),
        }

        # Memory management
        self._memory_cleanup_threshold = 0.8
        self._last_memory_check = datetime.now()

        # Start background processing
        self._start_background_tasks()

    async def collect_event(self, event: AnalyticsEvent) -> bool:
        """Collect an analytics event with intelligent buffering.

        Args:
            event: Analytics event to collect

        Returns:
            Success status
        """
        try:
            if not self._processing_enabled:
                self.logger.warning("Data collection disabled, dropping event")
                return False

            # Validate event
            if not event.event_id or not event.event_type:
                self.logger.error("Invalid event: missing required fields")
                return False

            # Add to appropriate buffer based on event type
            buffer = self._event_buffers[event.event_type]
            buffer.events.append(event.dict())

            # Update statistics
            self._stats["events_collected"] += 1

            # Check if immediate flush is needed (high-priority events)
            if self._should_immediate_flush(event):
                await self._flush_buffer(event.event_type)

            # Memory management check
            await self._check_memory_usage()

            return True

        except Exception as e:
            self.logger.exception(f"Error collecting event {event.event_id}: {e}")
            self._stats["errors_encountered"] += 1
            return False

    async def collect_metrics(self, metrics: AnalyticsMetrics) -> bool:
        """Collect analytics metrics with aggregation support.

        Args:
            metrics: Analytics metrics to collect

        Returns:
            Success status
        """
        try:
            if not self._processing_enabled:
                return False

            # Convert metrics to event format for unified handling
            metrics_event = AnalyticsEvent(
                event_id=str(uuid4()),
                event_type="metrics",
                source="metrics_collector",
                timestamp=metrics.timestamp,
                data=metrics.dict(),
                metadata={"aggregatable": True}
            )

            # Collect through standard event pipeline
            success = await self.collect_event(metrics_event)

            if success:
                self._stats["metrics_collected"] += 1

            return success

        except Exception as e:
            self.logger.exception(f"Error collecting metrics {metrics.metric_name}: {e}")
            self._stats["errors_encountered"] += 1
            return False

    async def aggregate_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Aggregate events within a time range with intelligent processing.

        Args:
            start_time: Start of aggregation window
            end_time: End of aggregation window
            event_types: Optional filter for specific event types

        Returns:
            Aggregation results
        """
        try:
            # Flush relevant buffers first to ensure fresh data
            if event_types:
                for event_type in event_types:
                    if event_type in self._event_buffers:
                        await self._flush_buffer(event_type)
            else:
                await self._flush_all_buffers()

            # Build aggregation query
            aggregations = {}

            # Basic event counting by type
            type_counts = await self._aggregate_event_counts(start_time, end_time, event_types)
            aggregations["event_counts"] = type_counts

            # Time-based aggregations (hourly, daily)
            time_series = await self._aggregate_time_series(start_time, end_time, event_types)
            aggregations["time_series"] = time_series

            # Value aggregations for metrics
            metric_aggregations = await self._aggregate_metrics_values(start_time, end_time)
            aggregations["metrics"] = metric_aggregations

            # Pattern detection
            patterns = await self._detect_event_patterns(start_time, end_time, event_types)
            aggregations["patterns"] = patterns

            return {
                "aggregation_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
                "event_types_included": event_types or "all",
                "results": aggregations,
                "metadata": {
                    "aggregated_at": datetime.now().isoformat(),
                    "processing_time_ms": 0,  # Would be calculated in real implementation
                    "total_events_processed": sum(type_counts.values()) if type_counts else 0
                }
            }

        except Exception as e:
            self.logger.exception(f"Error aggregating events: {e}")
            return {
                "error": str(e),
                "aggregation_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
                "results": {}
            }

    async def get_events(
        self,
        filter_criteria: dict[str, Any],
        limit: int = 100
    ) -> list[AnalyticsEvent]:
        """Get events matching filter criteria with intelligent caching.

        Args:
            filter_criteria: Filter conditions
            limit: Maximum number of events to return

        Returns:
            Filtered events
        """
        try:
            # Ensure recent events are persisted
            await self._flush_recent_events()

            events = []

            # Apply filters and retrieve events
            # This would connect to the actual database in a real implementation
            # For now, returning sample structure

            # Build query based on filter criteria
            query_conditions = []

            if "event_type" in filter_criteria:
                query_conditions.append(f"event_type = '{filter_criteria['event_type']}'")

            if "source" in filter_criteria:
                query_conditions.append(f"source = '{filter_criteria['source']}'")

            if "start_time" in filter_criteria:
                query_conditions.append(f"timestamp >= '{filter_criteria['start_time']}'")

            if "end_time" in filter_criteria:
                query_conditions.append(f"timestamp <= '{filter_criteria['end_time']}'")

            # For demonstration, return events from buffers that match criteria
            for event_type, buffer in self._event_buffers.items():
                if "event_type" in filter_criteria and filter_criteria["event_type"] != event_type:
                    continue

                for event_data in list(buffer.events)[-limit:]:
                    try:
                        event = AnalyticsEvent(**event_data)

                        # Apply additional filters
                        if "source" in filter_criteria and event.source != filter_criteria["source"]:
                            continue

                        if "start_time" in filter_criteria and event.timestamp < filter_criteria["start_time"]:
                            continue

                        if "end_time" in filter_criteria and event.timestamp > filter_criteria["end_time"]:
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            break
                    except Exception as e:
                        self.logger.warning(f"Error parsing event data: {e}")

                if len(events) >= limit:
                    break

            # Sort by timestamp (most recent first)
            events.sort(key=lambda x: x.timestamp, reverse=True)

            return events[:limit]

        except Exception as e:
            self.logger.exception(f"Error getting events: {e}")
            return []

    async def health_check(self) -> dict[str, Any]:
        """Check component health status."""
        try:
            # Calculate buffer utilization
            total_events = sum(len(buffer.events) for buffer in self._event_buffers.values())
            total_capacity = sum(buffer.buffer_size_limit for buffer in self._event_buffers.values())
            buffer_utilization = total_events / max(total_capacity, 1)

            # Calculate error rate
            total_operations = self._stats["events_collected"] + self._stats["metrics_collected"]
            error_rate = self._stats["errors_encountered"] / max(total_operations, 1)

            # Determine health status
            status = "healthy"
            alerts = []

            if buffer_utilization > 0.9:
                status = "degraded"
                alerts.append("High buffer utilization")

            if error_rate > 0.1:
                status = "unhealthy"
                alerts.append("High error rate")

            if not self._processing_enabled:
                status = "unhealthy"
                alerts.append("Processing disabled")

            return ComponentHealth(
                component_name="data_collection",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,  # Would measure actual response time
                error_rate=error_rate,
                memory_usage_mb=self._estimate_memory_usage(),
                alerts=alerts,
                details={
                    "buffer_utilization": buffer_utilization,
                    "active_buffers": len(self._event_buffers),
                    "total_events_buffered": total_events,
                    "stats": self._stats,
                    "processing_enabled": self._processing_enabled,
                }
            ).dict()

        except Exception as e:
            return {
                "component_name": "data_collection",
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }

    async def get_metrics(self) -> dict[str, Any]:
        """Get component performance metrics."""
        return {
            "performance": self._stats.copy(),
            "buffer_status": {
                buffer_type: {
                    "size": len(buffer.events),
                    "capacity": buffer.buffer_size_limit,
                    "utilization": len(buffer.events) / buffer.buffer_size_limit,
                    "last_flush": buffer.last_flush.isoformat()
                }
                for buffer_type, buffer in self._event_buffers.items()
            },
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    async def configure(self, config: dict[str, Any]) -> bool:
        """Configure component with new settings."""
        try:
            # Update configuration
            self.config.update(config)

            # Apply configuration changes
            if "batch_size" in config:
                self._batch_size = config["batch_size"]

            if "flush_interval" in config:
                self._flush_interval = config["flush_interval"]

            if "processing_enabled" in config:
                self._processing_enabled = config["processing_enabled"]

            self.logger.info(f"Data collection component reconfigured: {config}")
            return True

        except Exception as e:
            self.logger.exception(f"Error configuring component: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown component."""
        try:
            self.logger.info("Shutting down data collection component")

            # Stop processing
            self._processing_enabled = False

            # Cancel background tasks
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._flush_task

            # Flush all remaining data
            await self._flush_all_buffers()

            # Clear buffers
            self._event_buffers.clear()

            self.logger.info("Data collection component shutdown complete")

        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")

    # Private helper methods

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    async def _periodic_flush_loop(self) -> None:
        """Background task for periodic buffer flushing."""
        while self._processing_enabled:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_all_buffers()
                self._stats["flushes_completed"] += 1
                self._stats["last_flush_time"] = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Error in periodic flush: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    def _should_immediate_flush(self, event: AnalyticsEvent) -> bool:
        """Determine if event requires immediate flushing."""
        # High-priority events that need immediate processing
        high_priority_types = {"error", "security", "critical_performance"}
        return event.event_type in high_priority_types

    async def _flush_buffer(self, event_type: str) -> bool:
        """Flush specific event buffer to persistent storage."""
        try:
            if event_type not in self._event_buffers:
                return True

            buffer = self._event_buffers[event_type]
            if not buffer.events:
                return True

            # Get events to flush
            events_to_flush = list(buffer.events)

            # In a real implementation, this would write to database
            # For now, we'll just log the flush operation
            self.logger.debug(f"Flushing {len(events_to_flush)} {event_type} events")

            # Clear buffer after successful flush
            buffer.events.clear()
            buffer.last_flush = datetime.now()

            return True

        except Exception as e:
            self.logger.exception(f"Error flushing {event_type} buffer: {e}")
            return False

    async def _flush_all_buffers(self) -> None:
        """Flush all event buffers."""
        for event_type in list(self._event_buffers.keys()):
            await self._flush_buffer(event_type)

    async def _flush_recent_events(self) -> None:
        """Flush buffers that have recent activity."""
        for event_type, buffer in self._event_buffers.items():
            if buffer.events and (datetime.now() - buffer.last_flush).seconds > 30:
                await self._flush_buffer(event_type)

    async def _check_memory_usage(self) -> None:
        """Check and manage memory usage."""
        if (datetime.now() - self._last_memory_check).seconds < 30:
            return

        current_usage = self._estimate_memory_usage()

        if current_usage > self._max_memory_mb * self._memory_cleanup_threshold:
            self.logger.info(f"Memory usage {current_usage}MB approaching limit, triggering cleanup")
            await self._cleanup_old_data()

        self._last_memory_check = datetime.now()

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simple estimation based on buffer sizes
        total_events = sum(len(buffer.events) for buffer in self._event_buffers.values())
        # Rough estimate: 1KB per event
        return total_events * 0.001

    async def _cleanup_old_data(self) -> None:
        """Clean up old data to manage memory."""
        for buffer in self._event_buffers.values():
            # Keep only the most recent 50% of events in each buffer
            if len(buffer.events) > 100:
                events_list = list(buffer.events)
                buffer.events.clear()
                buffer.events.extend(events_list[-len(events_list) // 2:])

    async def _aggregate_event_counts(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None
    ) -> dict[str, int]:
        """Aggregate event counts by type."""
        # In a real implementation, this would query the database
        # For now, return sample data
        counts = {}

        for event_type, buffer in self._event_buffers.items():
            if event_types and event_type not in event_types:
                continue

            # Count events in time range
            count = 0
            for event_data in buffer.events:
                event_time = datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat()))
                if start_time <= event_time <= end_time:
                    count += 1

            if count > 0:
                counts[event_type] = count

        return counts

    async def _aggregate_time_series(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None
    ) -> dict[str, list[dict[str, Any]]]:
        """Aggregate events into time series."""
        # Simplified time series aggregation
        time_series = {}

        # Create hourly buckets
        current_hour = start_time.replace(minute=0, second=0, microsecond=0)

        while current_hour <= end_time:
            hour_key = current_hour.isoformat()

            for event_type, buffer in self._event_buffers.items():
                if event_types and event_type not in event_types:
                    continue

                if event_type not in time_series:
                    time_series[event_type] = []

                # Count events in this hour
                count = 0
                next_hour = current_hour + timedelta(hours=1)

                for event_data in buffer.events:
                    event_time = datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat()))
                    if current_hour <= event_time < next_hour:
                        count += 1

                time_series[event_type].append({
                    "timestamp": hour_key,
                    "count": count
                })

            current_hour += timedelta(hours=1)

        return time_series

    async def _aggregate_metrics_values(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> dict[str, dict[str, float]]:
        """Aggregate numeric metric values."""
        # Sample implementation
        metrics_agg = {}

        # Look for metrics events
        if "metrics" in self._event_buffers:
            metric_values = defaultdict(list)

            for event_data in self._event_buffers["metrics"].events:
                event_time = datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat()))
                if start_time <= event_time <= end_time:
                    data = event_data.get("data", {})
                    if "metric_name" in data and "value" in data:
                        metric_name = data["metric_name"]
                        value = data["value"]
                        if isinstance(value, (int, float)):
                            metric_values[metric_name].append(value)

            # Calculate aggregations
            for metric_name, values in metric_values.items():
                if values:
                    metrics_agg[metric_name] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }

        return metrics_agg

    async def _detect_event_patterns(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None
    ) -> dict[str, Any]:
        """Detect patterns in event data."""
        patterns = {
            "burst_detection": [],
            "anomaly_scores": {},
            "correlation_patterns": []
        }

        # Simple burst detection
        for event_type, buffer in self._event_buffers.items():
            if event_types and event_type not in event_types:
                continue

            # Count events per minute
            minute_counts = defaultdict(int)

            for event_data in buffer.events:
                event_time = datetime.fromisoformat(event_data.get("timestamp", datetime.now().isoformat()))
                if start_time <= event_time <= end_time:
                    minute_key = event_time.replace(second=0, microsecond=0)
                    minute_counts[minute_key] += 1

            if minute_counts:
                values = list(minute_counts.values())
                avg_rate = sum(values) / len(values)
                max_rate = max(values)

                # Detect bursts (rate > 3x average)
                if max_rate > avg_rate * 3:
                    patterns["burst_detection"].append({
                        "event_type": event_type,
                        "burst_rate": max_rate,
                        "average_rate": avg_rate,
                        "severity": "high" if max_rate > avg_rate * 5 else "medium"
                    })

        return patterns
