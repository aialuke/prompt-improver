"""Automated performance baseline collection engine."""

import asyncio
import json
import logging
import time
import tracemalloc
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import uuid

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Database monitoring
try:
    from ...database import get_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Redis monitoring
try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Prometheus integration
try:
    from ..monitoring.metrics_registry import get_metrics_registry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Background task manager
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

from .models import (
    BaselineMetrics, MetricValue, MetricDefinition, MetricType,
    STANDARD_METRICS, get_metric_definition
)

logger = logging.getLogger(__name__)

class BaselineCollector:
    """Automated performance baseline collection engine.
    
    Continuously collects key performance metrics with statistical analysis
    and trend detection capabilities.
    """

    def __init__(
        self,
        collection_interval: int = 60,  # seconds
        storage_path: Optional[Path] = None,
        metrics_config: Optional[Dict[str, MetricDefinition]] = None,
        enable_system_metrics: bool = True,
        enable_database_metrics: bool = True,
        enable_redis_metrics: bool = True,
        enable_application_metrics: bool = True,
        custom_collectors: Optional[List[Callable]] = None
    ):
        """Initialize baseline collector.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            storage_path: Where to store collected data
            metrics_config: Custom metric definitions
            enable_system_metrics: Collect system resource metrics
            enable_database_metrics: Collect database performance metrics
            enable_redis_metrics: Collect Redis/cache metrics
            enable_application_metrics: Collect application-specific metrics
            custom_collectors: Custom metric collection functions
        """
        self.collection_interval = collection_interval
        self.storage_path = storage_path or Path("./baselines")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metric definitions
        self.metrics_config = metrics_config or STANDARD_METRICS.copy()
        
        # Collection flags
        self.enable_system_metrics = enable_system_metrics and PSUTIL_AVAILABLE
        self.enable_database_metrics = enable_database_metrics and DATABASE_AVAILABLE
        self.enable_redis_metrics = enable_redis_metrics and REDIS_AVAILABLE
        self.enable_application_metrics = enable_application_metrics
        
        # Custom collectors
        self.custom_collectors = custom_collectors or []
        
        # State management
        self._running = False
        self._collection_task_id: Optional[str] = None
        self._current_baseline: Optional[BaselineMetrics] = None
        self._last_collection_time = datetime.now(timezone.utc)
        
        # Performance tracking
        self._request_counter = 0
        self._error_counter = 0
        self._response_times = []
        self._collection_lock = asyncio.Lock()
        
        # Prometheus integration
        if PROMETHEUS_AVAILABLE:
            self.metrics_registry = get_metrics_registry()
            self._setup_prometheus_metrics()
        
        logger.info(f"BaselineCollector initialized with {len(self.metrics_config)} metrics")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for baseline collection."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.baseline_counter = self.metrics_registry.get_or_create_counter(
                'baseline_collections_total',
                'Total number of baseline collections',
                ['status']
            )
            
            self.baseline_duration = self.metrics_registry.get_or_create_histogram(
                'baseline_collection_duration_seconds',
                'Time spent collecting baseline metrics',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            )
            
            self.metric_count = self.metrics_registry.get_or_create_gauge(
                'baseline_metrics_collected',
                'Number of metrics in last baseline collection'
            )
            
        except Exception as e:
            logger.warning(f"Failed to setup Prometheus metrics: {e}")

    async def start_collection(self) -> None:
        """Start automated baseline collection."""
        if self._running:
            logger.warning("Collection already running")
            return
        
        self._running = True
        # Start collection loop using background task manager
        task_manager = get_background_task_manager()
        self._collection_task_id = await task_manager.submit_enhanced_task(
            task_id=f"baseline_collection_{str(uuid.uuid4())[:8]}",
            coroutine=self._collection_loop(),
            priority=TaskPriority.LOW,
            tags={
                "service": "performance",
                "type": "baseline",
                "component": "collector",
                "operation": "data_collection"
            }
        )
        logger.info(f"Started baseline collection (interval: {self.collection_interval}s)")

    async def stop_collection(self) -> None:
        """Stop automated baseline collection."""
        if not self._running:
            return
        
        self._running = False
        
        if self._collection_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self._collection_task_id)
            self._collection_task_id = None
        
        # Save final baseline if available
        if self._current_baseline:
            await self.save_baseline(self._current_baseline)
        
        logger.info("Stopped baseline collection")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                start_time = time.time()
                
                # Collect metrics
                baseline = await self.collect_baseline()
                
                # Save baseline
                await self.save_baseline(baseline)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and hasattr(self, 'baseline_counter'):
                    collection_duration = time.time() - start_time
                    self.baseline_counter.labels(status='success').inc()
                    self.baseline_duration.observe(collection_duration)
                    
                    # Count metrics collected
                    metric_count = (
                        len(baseline.response_times) + 
                        len(baseline.cpu_utilization) + 
                        len(baseline.memory_utilization) +
                        sum(len(values) for values in baseline.custom_metrics.values())
                    )
                    self.metric_count.set(metric_count)
                
                logger.debug(f"Baseline collection completed in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                if PROMETHEUS_AVAILABLE and hasattr(self, 'baseline_counter'):
                    self.baseline_counter.labels(status='error').inc()
            
            # Wait for next collection
            await asyncio.sleep(self.collection_interval)

    async def collect_baseline(self) -> BaselineMetrics:
        """Collect a complete baseline measurement."""
        start_time = time.time()
        collection_timestamp = datetime.now(timezone.utc)
        
        baseline = BaselineMetrics(
            baseline_id=str(uuid.uuid4()),
            collection_timestamp=collection_timestamp,
            duration_seconds=0.0  # Will be updated at the end
        )
        
        async with self._collection_lock:
            # Collect different types of metrics
            if self.enable_system_metrics:
                await self._collect_system_metrics(baseline)
            
            if self.enable_database_metrics:
                await self._collect_database_metrics(baseline)
            
            if self.enable_redis_metrics:
                await self._collect_redis_metrics(baseline)
            
            if self.enable_application_metrics:
                await self._collect_application_metrics(baseline)
            
            # Run custom collectors
            for collector in self.custom_collectors:
                try:
                    await collector(baseline)
                except Exception as e:
                    logger.error(f"Custom collector failed: {e}")
        
        baseline.duration_seconds = time.time() - start_time
        self._current_baseline = baseline
        
        return baseline

    async def _collect_system_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            baseline.cpu_utilization.append(cpu_percent)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            baseline.memory_utilization.append(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            baseline.disk_usage.append(disk.percent)
            
            # Network I/O (bytes per second over last interval)
            network_io = psutil.net_io_counters()
            if hasattr(self, '_last_network_io'):
                bytes_sent_delta = network_io.bytes_sent - self._last_network_io.bytes_sent
                bytes_recv_delta = network_io.bytes_recv - self._last_network_io.bytes_recv
                total_bytes = bytes_sent_delta + bytes_recv_delta
                baseline.network_io.append(total_bytes / self.collection_interval)
            self._last_network_io = network_io
            
            logger.debug(f"System metrics: CPU={cpu_percent}%, Memory={memory.percent}%, Disk={disk.percent}%")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _collect_database_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect database performance metrics."""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            # Database connection time
            start_time = time.time()
            async with get_session() as session:
                connection_time = (time.time() - start_time) * 1000
                baseline.database_connection_time.append(connection_time)
                
                # Simple query response time
                query_start = time.time()
                await session.execute("SELECT 1")
                query_time = (time.time() - query_start) * 1000
                baseline.add_metric_value('database_query_time', query_time)
                
                # Connection pool metrics
                pool_info = session.get_bind().pool.status()
                if hasattr(pool_info, 'pool_size'):
                    baseline.add_metric_value('database_pool_size', pool_info.pool_size)
                if hasattr(pool_info, 'checked_out'):
                    baseline.add_metric_value('database_active_connections', pool_info.checked_out)
            
            logger.debug(f"Database metrics: Connection={connection_time:.2f}ms, Query={query_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")

    async def _collect_redis_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect Redis/cache performance metrics."""
        if not REDIS_AVAILABLE:
            return
        
        try:
            # Use UnifiedConnectionManager for Redis baseline collection
            from ...database.unified_connection_manager import get_unified_manager, ManagerMode
            unified_manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
            if not unified_manager._is_initialized:
                await unified_manager.initialize()
            
            # Get Redis client from UnifiedConnectionManager
            if hasattr(unified_manager, '_redis_master') and unified_manager._redis_master:
                redis_client = unified_manager._redis_master
            else:
                logger.warning("Redis client not available via UnifiedConnectionManager for baseline collection")
                return
            
            # Ping time
            ping_start = time.time()
            await redis_client.ping()
            ping_time = (time.time() - ping_start) * 1000
            baseline.add_metric_value('redis_ping_time', ping_time)
            
            # Memory usage and stats
            info = await redis_client.info('memory')
            if 'used_memory' in info:
                used_memory_mb = info['used_memory'] / (1024 * 1024)
                baseline.add_metric_value('redis_memory_usage', used_memory_mb)
            
            # Keyspace statistics for hit rate calculation
            keyspace_info = await redis_client.info('stats')
            if 'keyspace_hits' in keyspace_info and 'keyspace_misses' in keyspace_info:
                hits = keyspace_info['keyspace_hits']
                misses = keyspace_info['keyspace_misses']
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                baseline.cache_hit_rate.append(hit_rate)
            
            await redis_client.close()
            
            logger.debug(f"Redis metrics: Ping={ping_time:.2f}ms, Hit rate={hit_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")

    async def _collect_application_metrics(self, baseline: BaselineMetrics) -> None:
        """Collect application-specific metrics."""
        try:
            # Response time metrics (from tracked requests)
            if self._response_times:
                baseline.response_times.extend(self._response_times)
                self._response_times.clear()
            
            # Error rate calculation
            if self._request_counter > 0:
                error_rate = (self._error_counter / self._request_counter) * 100
                baseline.error_rates.append(error_rate)
            
            # Throughput calculation (requests per second)
            current_time = datetime.now(timezone.utc)
            time_delta = (current_time - self._last_collection_time).total_seconds()
            if time_delta > 0:
                throughput = self._request_counter / time_delta
                baseline.throughput_values.append(throughput)
            
            # Reset counters
            self._request_counter = 0
            self._error_counter = 0
            self._last_collection_time = current_time
            
            # Memory profiling if tracemalloc is enabled
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                baseline.add_metric_value('memory_traced_current', current / (1024 * 1024))
                baseline.add_metric_value('memory_traced_peak', peak / (1024 * 1024))
            
            logger.debug(f"Application metrics: Requests={self._request_counter}, Errors={self._error_counter}")
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")

    async def record_request(
        self, 
        response_time_ms: float, 
        is_error: bool = False,
        operation_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a request for performance tracking.
        
        Args:
            response_time_ms: Response time in milliseconds
            is_error: Whether the request resulted in an error
            operation_name: Name of the operation (for segmentation)
            metadata: Additional metadata about the request
        """
        async with self._collection_lock:
            self._request_counter += 1
            self._response_times.append(response_time_ms)
            
            if is_error:
                self._error_counter += 1
            
            # Store operation-specific metrics
            if operation_name and self._current_baseline:
                metric_name = f'{operation_name}_response_time'
                self._current_baseline.add_metric_value(metric_name, response_time_ms)
                
                if is_error:
                    error_metric_name = f'{operation_name}_error_count'
                    self._current_baseline.add_metric_value(error_metric_name, 1)

    async def record_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a custom metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: When the metric was recorded
        """
        if self._current_baseline:
            async with self._collection_lock:
                self._current_baseline.add_metric_value(metric_name, value)

    async def save_baseline(self, baseline: BaselineMetrics) -> None:
        """Save baseline to storage."""
        try:
            # Create filename with timestamp
            timestamp_str = baseline.collection_timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_{timestamp_str}_{baseline.baseline_id[:8]}.json"
            filepath = self.storage_path / filename
            
            # Save as JSON
            baseline_data = baseline.to_dict()
            with open(filepath, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
            
            logger.info(f"Saved baseline to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")

    async def load_baseline(self, baseline_id: str) -> Optional[BaselineMetrics]:
        """Load a specific baseline by ID."""
        try:
            # Find file by baseline ID
            for filepath in self.storage_path.glob("baseline_*.json"):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if data.get('baseline_id') == baseline_id:
                        return BaselineMetrics.from_dict(data)
            
            logger.warning(f"Baseline {baseline_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load baseline {baseline_id}: {e}")
            return None

    async def load_recent_baselines(self, hours: int = 24) -> List[BaselineMetrics]:
        """Load baselines from the last N hours."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            baselines = []
            
            for filepath in sorted(self.storage_path.glob("baseline_*.json")):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    baseline = BaselineMetrics.from_dict(data)
                    
                    if baseline.collection_timestamp >= cutoff_time:
                        baselines.append(baseline)
            
            logger.info(f"Loaded {len(baselines)} baselines from last {hours} hours")
            return baselines
            
        except Exception as e:
            logger.error(f"Failed to load recent baselines: {e}")
            return []

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status."""
        return {
            'running': self._running,
            'collection_interval': self.collection_interval,
            'last_collection': self._last_collection_time.isoformat() if self._last_collection_time else None,
            'current_baseline_id': self._current_baseline.baseline_id if self._current_baseline else None,
            'metrics_enabled': {
                'system': self.enable_system_metrics,
                'database': self.enable_database_metrics,
                'redis': self.enable_redis_metrics,
                'application': self.enable_application_metrics
            },
            'custom_collectors': len(self.custom_collectors),
            'request_counter': self._request_counter,
            'error_counter': self._error_counter,
            'response_times_pending': len(self._response_times)
        }

    async def collect_single_baseline(
        self, 
        duration_seconds: int = 60,
        custom_tags: Optional[Dict[str, str]] = None
    ) -> BaselineMetrics:
        """Collect a single baseline over a specified duration.
        
        Args:
            duration_seconds: How long to collect data
            custom_tags: Additional tags to add to the baseline
            
        Returns:
            Collected baseline metrics
        """
        logger.info(f"Starting single baseline collection for {duration_seconds}s")
        
        baseline = BaselineMetrics(
            baseline_id=str(uuid.uuid4()),
            collection_timestamp=datetime.now(timezone.utc),
            duration_seconds=duration_seconds,
            tags=custom_tags or {}
        )
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Collect metrics over the duration
        collection_count = 0
        while time.time() < end_time:
            try:
                if self.enable_system_metrics:
                    await self._collect_system_metrics(baseline)
                
                if self.enable_database_metrics:
                    await self._collect_database_metrics(baseline)
                
                if self.enable_redis_metrics:
                    await self._collect_redis_metrics(baseline)
                
                if self.enable_application_metrics:
                    await self._collect_application_metrics(baseline)
                
                collection_count += 1
                
                # Wait before next collection (sample every 5 seconds)
                await asyncio.sleep(min(5, duration_seconds / 10))
                
            except Exception as e:
                logger.error(f"Error during single baseline collection: {e}")
        
        baseline.metadata['collection_count'] = collection_count
        baseline.metadata['actual_duration'] = time.time() - start_time
        
        logger.info(f"Completed single baseline collection: {collection_count} samples")
        return baseline

# Global collector instance
_global_collector: Optional[BaselineCollector] = None

def get_baseline_collector() -> BaselineCollector:
    """Get the global baseline collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = BaselineCollector()
    return _global_collector

def set_baseline_collector(collector: BaselineCollector) -> None:
    """Set the global baseline collector instance."""
    global _global_collector
    _global_collector = collector

# Convenience functions for request tracking
async def record_operation(
    operation_name: str,
    response_time_ms: float,
    is_error: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Record an operation for baseline collection."""
    collector = get_baseline_collector()
    await collector.record_request(response_time_ms, is_error, operation_name, metadata)

@asynccontextmanager
async def track_operation(operation_name: str):
    """Context manager to automatically track operation timing."""
    start_time = time.time()
    error_occurred = False
    
    try:
        yield
    except Exception as e:
        error_occurred = True
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        await record_operation(operation_name, duration_ms, error_occurred)