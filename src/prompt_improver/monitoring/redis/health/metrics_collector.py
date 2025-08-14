"""Redis Metrics Collector Service

Focused Redis performance metrics collection service for throughput analysis and monitoring.
Designed for comprehensive metrics gathering with <25ms operations following SRE best practices.
"""

import asyncio
import logging
import re
import statistics
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import coredis

from prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry

from .protocols import RedisClientProviderProtocol, RedisMetricsCollectorProtocol
from .types import PerformanceMetrics

logger = logging.getLogger(__name__)
_metrics_registry = get_metrics_registry()

# Performance metrics
REDIS_OPERATIONS_TOTAL = _metrics_registry.get_or_create_counter(
    "redis_operations_total",
    "Total Redis operations",
    ["operation_type"]
)

REDIS_MEMORY_USAGE = _metrics_registry.get_or_create_gauge(
    "redis_memory_usage_bytes",
    "Redis memory usage in bytes",
    ["memory_type"]
)

REDIS_HIT_RATE = _metrics_registry.get_or_create_gauge(
    "redis_cache_hit_rate_percent",
    "Redis cache hit rate percentage"
)

REDIS_LATENCY_PERCENTILES = _metrics_registry.get_or_create_histogram(
    "redis_operation_latency_ms",
    "Redis operation latency percentiles in milliseconds",
    ["operation"]
)

REDIS_SLOW_QUERIES = _metrics_registry.get_or_create_counter(
    "redis_slow_queries_total",
    "Total Redis slow queries",
    ["command"]
)


class RedisMetricsCollector:
    """Redis metrics collector service for performance monitoring and analysis.
    
    Provides comprehensive performance metrics collection including memory usage,
    throughput analysis, slow query monitoring, and trend analysis for SRE monitoring.
    """
    
    def __init__(
        self,
        client_provider: RedisClientProviderProtocol,
        collection_interval: float = 30.0,
        slow_query_threshold_ms: float = 100.0,
        max_metrics_history: int = 100
    ):
        """Initialize Redis metrics collector.
        
        Args:
            client_provider: Redis client provider for connections
            collection_interval: Interval for metrics collection in seconds
            slow_query_threshold_ms: Threshold for slow query detection
            max_metrics_history: Maximum metrics history to maintain
        """
        self.client_provider = client_provider
        self.collection_interval = collection_interval
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.max_metrics_history = max_metrics_history
        
        # Metrics state
        self._last_metrics: Optional[PerformanceMetrics] = None
        self._metrics_history: List[PerformanceMetrics] = []
        
        # Tracking state
        self._last_command_count = 0
        self._last_collection_time = 0.0
        self._is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Key sampling for memory analysis
        self._key_sample_size = 1000
        self._large_key_threshold_bytes = 1024 * 1024  # 1MB
        
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics.
        
        Returns:
            Current performance metrics with detailed analysis
        """
        start_time = time.time()
        
        try:
            metrics = PerformanceMetrics()
            
            client = await self.client_provider.get_client()
            if not client:
                return metrics
            
            # Collect Redis info
            info = await client.info()
            
            # Command performance metrics
            await self._collect_command_metrics(info, metrics)
            
            # Cache performance metrics  
            await self._collect_cache_metrics(info, metrics)
            
            # Memory metrics
            await self._collect_memory_metrics(info, metrics)
            
            # Latency sampling
            await self._collect_latency_metrics(client, metrics)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Cache metrics and update history
            self._last_metrics = metrics
            self._update_metrics_history(metrics)
            
            collection_duration = (time.time() - start_time) * 1000
            logger.debug(f"Metrics collection completed in {collection_duration:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics()
    
    async def collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory usage and fragmentation metrics.
        
        Returns:
            Detailed memory metrics dictionary
        """
        try:
            client = await self.client_provider.get_client()
            if not client:
                return {"error": "Redis client not available"}
            
            info = await client.info()
            
            memory_metrics = {
                "used_memory": {
                    "bytes": self._safe_int(info.get("used_memory", 0)),
                    "human": info.get("used_memory_human", "0B"),
                    "rss_bytes": self._safe_int(info.get("used_memory_rss", 0)),
                    "peak_bytes": self._safe_int(info.get("used_memory_peak", 0)),
                    "peak_human": info.get("used_memory_peak_human", "0B"),
                },
                "fragmentation": {
                    "ratio": self._safe_float(info.get("mem_fragmentation_ratio", 1.0)),
                    "fragmentation_bytes": max(0, 
                        self._safe_int(info.get("used_memory_rss", 0)) - 
                        self._safe_int(info.get("used_memory", 0))
                    ),
                },
                "system": {
                    "total_system_memory": self._safe_int(info.get("total_system_memory", 0)),
                    "maxmemory": self._safe_int(info.get("maxmemory", 0)),
                    "maxmemory_human": info.get("maxmemory_human", "0B"),
                    "maxmemory_policy": info.get("maxmemory_policy", "noeviction"),
                },
                "dataset": {
                    "used_memory_overhead": self._safe_int(info.get("used_memory_overhead", 0)),
                    "used_memory_dataset": self._safe_int(info.get("used_memory_dataset", 0)),
                    "dataset_percentage": 0.0,
                },
            }
            
            # Calculate dataset percentage
            total_memory = memory_metrics["used_memory"]["bytes"]
            if total_memory > 0:
                dataset_memory = memory_metrics["dataset"]["used_memory_dataset"]
                memory_metrics["dataset"]["dataset_percentage"] = (
                    dataset_memory / total_memory * 100
                )
            
            # Calculate memory usage percentage if maxmemory is set
            maxmemory = memory_metrics["system"]["maxmemory"]
            if maxmemory > 0:
                memory_metrics["usage_percentage"] = (
                    memory_metrics["used_memory"]["bytes"] / maxmemory * 100
                )
            
            return memory_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
            return {"error": str(e)}
    
    async def collect_throughput_metrics(self) -> Dict[str, Any]:
        """Collect throughput and command statistics.
        
        Returns:
            Throughput metrics dictionary
        """
        try:
            client = await self.client_provider.get_client()
            if not client:
                return {"error": "Redis client not available"}
            
            info = await client.info()
            current_time = time.time()
            
            # Basic throughput metrics
            throughput_metrics = {
                "instantaneous_ops_per_sec": self._safe_int(
                    info.get("instantaneous_ops_per_sec", 0)
                ),
                "total_commands_processed": self._safe_int(
                    info.get("total_commands_processed", 0)
                ),
                "total_connections_received": self._safe_int(
                    info.get("total_connections_received", 0)
                ),
                "rejected_connections": self._safe_int(
                    info.get("rejected_connections", 0)
                ),
            }
            
            # Calculate average ops per second
            if hasattr(self, "_last_command_count") and self._last_collection_time > 0:
                time_diff = current_time - self._last_collection_time
                command_diff = (
                    throughput_metrics["total_commands_processed"] - self._last_command_count
                )
                
                if time_diff > 0:
                    throughput_metrics["avg_ops_per_sec"] = command_diff / time_diff
                else:
                    throughput_metrics["avg_ops_per_sec"] = 0.0
            else:
                throughput_metrics["avg_ops_per_sec"] = 0.0
            
            # Update tracking state
            self._last_command_count = throughput_metrics["total_commands_processed"]
            self._last_collection_time = current_time
            
            # Get command stats if available
            try:
                command_stats = await client.info("commandstats")
                throughput_metrics["command_statistics"] = self._parse_command_stats(command_stats)
            except Exception as e:
                logger.debug(f"Failed to get command stats: {e}")
                throughput_metrics["command_statistics"] = {}
            
            return throughput_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect throughput metrics: {e}")
            return {"error": str(e)}
    
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries for performance optimization.
        
        Returns:
            List of slow query analysis results
        """
        try:
            client = await self.client_provider.get_client()
            if not client:
                return []
            
            # Get slow log entries
            slow_entries = await client.slowlog_get(100)  # Last 100 entries
            
            slow_queries = []
            current_time = time.time()
            
            for entry in slow_entries:
                try:
                    slow_query = {
                        "id": self._safe_int(entry[0]),
                        "timestamp": self._safe_int(entry[1]),
                        "duration_microseconds": self._safe_int(entry[2]),
                        "duration_ms": self._safe_int(entry[2]) / 1000.0,
                        "command": [str(cmd) for cmd in entry[3]] if len(entry) > 3 else [],
                        "client_info": {
                            "ip": str(entry[4]) if len(entry) > 4 else "",
                            "name": str(entry[5]) if len(entry) > 5 else "",
                        },
                        "age_seconds": current_time - self._safe_int(entry[1]),
                    }
                    
                    # Command analysis
                    if slow_query["command"]:
                        slow_query["command_type"] = slow_query["command"][0].upper()
                        slow_query["command_summary"] = " ".join(
                            str(cmd) for cmd in slow_query["command"][:5]
                        )
                    
                    # Performance analysis
                    if slow_query["duration_ms"] > self.slow_query_threshold_ms:
                        slow_queries.append(slow_query)
                        
                        # Update metrics
                        REDIS_SLOW_QUERIES.labels(
                            command=slow_query.get("command_type", "UNKNOWN")
                        ).inc()
                
                except (IndexError, TypeError, ValueError) as e:
                    logger.debug(f"Failed to parse slow log entry: {e}")
                    continue
            
            # Sort by duration descending
            slow_queries.sort(key=lambda x: x["duration_ms"], reverse=True)
            
            return slow_queries
            
        except Exception as e:
            logger.error(f"Failed to analyze slow queries: {e}")
            return []
    
    def get_metrics_history(self, duration_minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics history for trend analysis.
        
        Args:
            duration_minutes: How far back to retrieve metrics
            
        Returns:
            List of historical performance metrics
        """
        if not self._metrics_history:
            return []
        
        # Filter by time window
        cutoff_time = time.time() - (duration_minutes * 60)
        
        # Note: This is a simplified implementation
        # In production, you'd want to add timestamps to PerformanceMetrics
        return self._metrics_history[-duration_minutes:]  # Simplified approach
    
    async def start_collection(self) -> None:
        """Start continuous metrics collection."""
        if self._is_collecting:
            logger.warning("Metrics collection already started")
            return
        
        self._is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Redis metrics collection started")
    
    async def stop_collection(self) -> None:
        """Stop continuous metrics collection."""
        if not self._is_collecting:
            return
        
        self._is_collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        
        logger.info("Redis metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Continuous metrics collection loop."""
        logger.info("Starting Redis metrics collection loop")
        
        while self._is_collecting:
            try:
                await self.collect_performance_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Brief delay on error
    
    async def _collect_command_metrics(
        self, 
        info: Dict[str, Any], 
        metrics: PerformanceMetrics
    ) -> None:
        """Collect command performance metrics.
        
        Args:
            info: Redis info dictionary
            metrics: PerformanceMetrics object to populate
        """
        metrics.total_commands = self._safe_int(info.get("total_commands_processed", 0))
        metrics.instantaneous_ops_per_sec = self._safe_int(
            info.get("instantaneous_ops_per_sec", 0)
        )
        
        # Calculate average ops per second
        current_time = time.time()
        if self._last_command_count > 0 and self._last_collection_time > 0:
            time_diff = current_time - self._last_collection_time
            command_diff = metrics.total_commands - self._last_command_count
            
            if time_diff > 0:
                metrics.avg_ops_per_sec = command_diff / time_diff
        
        # Update tracking
        self._last_command_count = metrics.total_commands
        self._last_collection_time = current_time
    
    async def _collect_cache_metrics(
        self, 
        info: Dict[str, Any], 
        metrics: PerformanceMetrics
    ) -> None:
        """Collect cache performance metrics.
        
        Args:
            info: Redis info dictionary
            metrics: PerformanceMetrics object to populate
        """
        metrics.keyspace_hits = self._safe_int(info.get("keyspace_hits", 0))
        metrics.keyspace_misses = self._safe_int(info.get("keyspace_misses", 0))
        
        # Calculate hit rate
        metrics.calculate_hit_rate()
    
    async def _collect_memory_metrics(
        self, 
        info: Dict[str, Any], 
        metrics: PerformanceMetrics
    ) -> None:
        """Collect memory metrics.
        
        Args:
            info: Redis info dictionary
            metrics: PerformanceMetrics object to populate
        """
        metrics.used_memory_bytes = self._safe_int(info.get("used_memory", 0))
        metrics.peak_memory_bytes = self._safe_int(info.get("used_memory_peak", 0))
        metrics.fragmentation_ratio = self._safe_float(
            info.get("mem_fragmentation_ratio", 1.0)
        )
        
        # Calculate fragmentation bytes
        rss_memory = self._safe_int(info.get("used_memory_rss", 0))
        if rss_memory > metrics.used_memory_bytes:
            metrics.fragmentation_bytes = rss_memory - metrics.used_memory_bytes
    
    async def _collect_latency_metrics(
        self, 
        client: coredis.Redis, 
        metrics: PerformanceMetrics
    ) -> None:
        """Collect latency metrics through sampling.
        
        Args:
            client: Redis client instance
            metrics: PerformanceMetrics object to populate
        """
        try:
            # Sample latency with ping commands
            latency_samples = []
            
            for _ in range(5):  # Take 5 samples
                start_time = time.time()
                await client.ping()
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latency_samples.append(latency_ms)
                
                # Add to metrics
                metrics.add_latency_sample(latency_ms)
                
                # Update Prometheus histogram
                REDIS_LATENCY_PERCENTILES.labels(operation="ping").observe(latency_ms)
            
        except Exception as e:
            logger.debug(f"Failed to collect latency metrics: {e}")
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update Prometheus metrics registry.
        
        Args:
            metrics: PerformanceMetrics to update in registry
        """
        try:
            # Update operation counters
            REDIS_OPERATIONS_TOTAL.labels(operation_type="total").inc(
                metrics.total_commands - REDIS_OPERATIONS_TOTAL.labels(operation_type="total")._value.get()
            )
            
            # Update memory metrics
            REDIS_MEMORY_USAGE.labels(memory_type="used").set(metrics.used_memory_bytes)
            REDIS_MEMORY_USAGE.labels(memory_type="peak").set(metrics.peak_memory_bytes)
            
            # Update hit rate
            REDIS_HIT_RATE.set(metrics.hit_rate_percentage)
            
        except Exception as e:
            logger.debug(f"Failed to update Prometheus metrics: {e}")
    
    def _update_metrics_history(self, metrics: PerformanceMetrics) -> None:
        """Update metrics history for trend analysis.
        
        Args:
            metrics: Current metrics to add to history
        """
        self._metrics_history.append(metrics)
        
        # Keep history size manageable
        if len(self._metrics_history) > self.max_metrics_history:
            self._metrics_history = self._metrics_history[-self.max_metrics_history:]
    
    def _parse_command_stats(self, command_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Redis command statistics.
        
        Args:
            command_stats: Raw command stats from Redis
            
        Returns:
            Parsed command statistics
        """
        parsed_stats = {}
        
        for key, value in command_stats.items():
            if key.startswith("cmdstat_"):
                cmd_name = key[8:]  # Remove "cmdstat_" prefix
                
                # Parse the stats string: "calls=X,usec=Y,usec_per_call=Z"
                stats = {}
                if isinstance(value, str):
                    for stat in value.split(","):
                        if "=" in stat:
                            stat_key, stat_value = stat.split("=", 1)
                            stats[stat_key] = self._safe_float(stat_value)
                
                parsed_stats[cmd_name] = stats
        
        return parsed_stats
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for quick analysis.
        
        Returns:
            Performance summary dictionary
        """
        if not self._last_metrics:
            return {"error": "No metrics available"}
        
        metrics = self._last_metrics
        
        return {
            "throughput": {
                "instantaneous_ops_per_sec": metrics.instantaneous_ops_per_sec,
                "avg_ops_per_sec": round(metrics.avg_ops_per_sec, 2),
                "total_commands": metrics.total_commands,
            },
            "cache_performance": {
                "hit_rate_percent": round(metrics.hit_rate_percentage, 2),
                "keyspace_hits": metrics.keyspace_hits,
                "keyspace_misses": metrics.keyspace_misses,
            },
            "memory": {
                "used_memory_mb": round(metrics.used_memory_bytes / 1024 / 1024, 2),
                "peak_memory_mb": round(metrics.peak_memory_bytes / 1024 / 1024, 2),
                "fragmentation_ratio": round(metrics.fragmentation_ratio, 2),
            },
            "latency": {
                "avg_ms": round(metrics.avg_latency_ms, 2),
                "p95_ms": round(metrics.p95_latency_ms, 2),
                "p99_ms": round(metrics.p99_latency_ms, 2),
                "max_ms": round(metrics.max_latency_ms, 2),
            },
            "collection_status": {
                "is_collecting": self._is_collecting,
                "history_size": len(self._metrics_history),
                "last_collection": self._last_collection_time,
            }
        }