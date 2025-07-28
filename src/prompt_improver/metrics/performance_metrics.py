"""
Performance Business Metrics Collectors for Prompt Improver.

Tracks request processing pipeline stages, database query performance,
cache effectiveness, and external API dependency performance with real-time analysis.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque

from ..performance.monitoring.metrics_registry import get_metrics_registry

class PipelineStage(Enum):
    """Request processing pipeline stages."""
    INGRESS = "ingress"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    INPUT_VALIDATION = "input_validation"
    BUSINESS_LOGIC = "business_logic"
    ML_PROCESSING = "ml_processing"
    DATA_ACCESS = "data_access"
    EXTERNAL_API = "external_api"
    RESPONSE_FORMATTING = "response_formatting"
    EGRESS = "egress"

class DatabaseOperation(Enum):
    """Types of database operations."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"
    BULK_UPDATE = "bulk_update"
    TRANSACTION = "transaction"
    MIGRATION = "migration"
    INDEX_SCAN = "index_scan"
    FULL_TABLE_SCAN = "full_table_scan"

class CacheType(Enum):
    """Types of cache systems."""
    APPLICATION = "application"
    ML_MODEL = "ml_model"
    CONFIGURATION = "configuration"
    SESSION = "session"
    REDIS = "redis"
    MEMORY = "memory"
    DATABASE_QUERY = "database_query"
    API_RESPONSE = "api_response"

class ExternalAPIType(Enum):
    """Types of external APIs."""
    ML_INFERENCE = "ml_inference"
    AUTHENTICATION = "authentication"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    THIRD_PARTY = "third_party"

@dataclass
class RequestPipelineMetric:
    """Metrics for request processing pipeline stages."""
    request_id: str
    stage: PipelineStage
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_type: Optional[str]
    memory_usage_mb: float
    cpu_usage_percent: float
    stage_specific_data: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    endpoint: str
    method: str
    queue_time_ms: Optional[float]
    retry_count: int

@dataclass
class DatabasePerformanceMetric:
    """Metrics for database operations."""
    operation_type: DatabaseOperation
    table_name: str
    query_hash: str
    execution_time_ms: float
    rows_affected: int
    rows_examined: int
    index_usage: List[str]
    query_plan_type: str
    connection_pool_size: int
    active_connections: int
    wait_time_ms: float
    lock_time_ms: float
    temp_tables_created: int
    bytes_sent: int
    bytes_received: int
    success: bool
    error_type: Optional[str]
    timestamp: datetime
    transaction_id: Optional[str]

@dataclass
class CachePerformanceMetric:
    """Metrics for cache operations."""
    cache_type: CacheType
    operation: str  # "get", "set", "delete", "exists"
    key_pattern: str
    hit: bool
    response_time_ms: float
    cache_size_bytes: Optional[int]
    eviction_triggered: bool
    ttl_remaining_seconds: Optional[int]
    serialization_time_ms: Optional[float]
    network_time_ms: Optional[float]
    compression_used: bool
    compression_ratio: Optional[float]
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]

@dataclass
class ExternalAPIMetric:
    """Metrics for external API calls."""
    api_type: ExternalAPIType
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    request_size_bytes: int
    response_size_bytes: int
    success: bool
    error_type: Optional[str]
    retry_count: int
    circuit_breaker_state: str
    rate_limited: bool
    timeout_occurred: bool
    dns_lookup_time_ms: Optional[float]
    tcp_connect_time_ms: Optional[float]
    tls_handshake_time_ms: Optional[float]
    ttfb_ms: Optional[float]  # Time to first byte
    timestamp: datetime
    request_id: str

class PerformanceMetricsCollector:
    """
    Collects and aggregates performance business metrics.
    
    Provides real-time tracking of request pipeline efficiency, database
    performance, cache effectiveness, and external dependency reliability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance metrics collector."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage with size limits
        self.pipeline_metrics: deque = deque(maxlen=self.config.get("max_pipeline_metrics", 15000))
        self.database_metrics: deque = deque(maxlen=self.config.get("max_database_metrics", 10000))
        self.cache_metrics: deque = deque(maxlen=self.config.get("max_cache_metrics", 20000))
        self.external_api_metrics: deque = deque(maxlen=self.config.get("max_external_api_metrics", 10000))
        
        # Real-time tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.pipeline_stage_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.database_query_cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.aggregation_window_minutes = self.config.get("aggregation_window_minutes", 5)
        self.retention_hours = self.config.get("retention_hours", 48)
        self.slow_query_threshold_ms = self.config.get("slow_query_threshold_ms", 1000)
        self.cache_efficiency_threshold = self.config.get("cache_efficiency_threshold", 0.8)
        
        # Collection statistics
        self.collection_stats = {
            "pipeline_stages_tracked": 0,
            "database_queries_tracked": 0,
            "cache_operations_tracked": 0,
            "external_api_calls_tracked": 0,
            "active_requests_count": 0,
            "slow_queries_detected": 0,
            "cache_misses_detected": 0,
            "last_aggregation": None
        }
        
        # Prometheus metrics integration
        self.metrics_registry = get_metrics_registry()
        self._initialize_prometheus_metrics()
        
        # Background processing
        self.is_running = False
        self.aggregation_task = None
        self.system_monitoring_task = None
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics for performance tracking."""
        # Request pipeline metrics
        self.pipeline_stage_duration = self.metrics_registry.get_or_create_histogram(
            "performance_pipeline_stage_duration_seconds",
            "Duration of request pipeline stages",
            ["stage", "endpoint", "success"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.pipeline_queue_time = self.metrics_registry.get_or_create_histogram(
            "performance_pipeline_queue_time_seconds",
            "Time spent queuing before pipeline stage execution",
            ["stage"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.pipeline_throughput = self.metrics_registry.get_or_create_gauge(
            "performance_pipeline_throughput_requests_per_second",
            "Request processing throughput by stage",
            ["stage", "time_window"]
        )
        
        self.pipeline_error_rate = self.metrics_registry.get_or_create_gauge(
            "performance_pipeline_error_rate",
            "Error rate by pipeline stage",
            ["stage", "error_type"]
        )
        
        # Database performance metrics
        self.database_query_duration = self.metrics_registry.get_or_create_histogram(
            "performance_database_query_duration_seconds",
            "Database query execution duration",
            ["operation", "table", "query_plan"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.database_connection_pool_utilization = self.metrics_registry.get_or_create_gauge(
            "performance_database_connection_pool_utilization",
            "Database connection pool utilization ratio",
            ["pool_name"]
        )
        
        self.database_slow_queries = self.metrics_registry.get_or_create_counter(
            "performance_database_slow_queries_total",
            "Total count of slow database queries",
            ["table", "operation"]
        )
        
        self.database_lock_time = self.metrics_registry.get_or_create_histogram(
            "performance_database_lock_time_seconds",
            "Time spent waiting for database locks",
            ["table", "operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # Cache performance metrics
        self.cache_hit_rate = self.metrics_registry.get_or_create_gauge(
            "performance_cache_hit_rate",
            "Cache hit rate by cache type",
            ["cache_type", "time_window"]
        )
        
        self.cache_operation_duration = self.metrics_registry.get_or_create_histogram(
            "performance_cache_operation_duration_seconds",
            "Cache operation duration",
            ["cache_type", "operation"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        )
        
        self.cache_eviction_rate = self.metrics_registry.get_or_create_counter(
            "performance_cache_evictions_total",
            "Total cache evictions",
            ["cache_type", "reason"]
        )
        
        self.cache_size_utilization = self.metrics_registry.get_or_create_gauge(
            "performance_cache_size_utilization_ratio",
            "Cache size utilization ratio",
            ["cache_type"]
        )
        
        # External API metrics
        self.external_api_response_time = self.metrics_registry.get_or_create_histogram(
            "performance_external_api_response_time_seconds",
            "External API response time distribution",
            ["api_type", "endpoint", "status_class"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.external_api_success_rate = self.metrics_registry.get_or_create_gauge(
            "performance_external_api_success_rate",
            "External API success rate",
            ["api_type", "endpoint", "time_window"]
        )
        
        self.external_api_circuit_breaker_state = self.metrics_registry.get_or_create_gauge(
            "performance_external_api_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            ["api_type", "endpoint"]
        )
        
        # System resource metrics
        self.system_cpu_utilization = self.metrics_registry.get_or_create_gauge(
            "performance_system_cpu_utilization_percent",
            "System CPU utilization percentage",
            ["core"]
        )
        
        self.system_memory_utilization = self.metrics_registry.get_or_create_gauge(
            "performance_system_memory_utilization_percent",
            "System memory utilization percentage",
            ["type"]  # "physical", "virtual", "available"
        )
        
        self.system_disk_io = self.metrics_registry.get_or_create_counter(
            "performance_system_disk_io_bytes_total",
            "Total disk I/O bytes",
            ["operation", "device"]  # operation: "read" or "write"
        )
    
    async def start_collection(self):
        """Start background metrics collection and processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.system_monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        self.logger.info("Started performance metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection and processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        for task in [self.aggregation_task, self.system_monitoring_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Stopped performance metrics collection")
    
    async def _aggregation_loop(self):
        """Background aggregation of metrics."""
        try:
            while self.is_running:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_window_minutes * 60)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in performance metrics aggregation: {e}")
    
    async def _system_monitoring_loop(self):
        """Background system resource monitoring."""
        try:
            while self.is_running:
                await self._collect_system_metrics()
                await asyncio.sleep(30)  # Collect system metrics every 30 seconds
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in system monitoring: {e}")
    
    async def record_pipeline_stage(self, metric: RequestPipelineMetric):
        """Record a request pipeline stage metric."""
        try:
            self.pipeline_metrics.append(metric)
            self.collection_stats["pipeline_stages_tracked"] += 1
            
            # Track active requests
            if metric.request_id not in self.active_requests:
                self.active_requests[metric.request_id] = {
                    "start_time": metric.start_time,
                    "stages": [],
                    "user_id": metric.user_id,
                    "endpoint": metric.endpoint
                }
            
            self.active_requests[metric.request_id]["stages"].append({
                "stage": metric.stage,
                "duration_ms": metric.duration_ms,
                "success": metric.success
            })
            
            # Update pipeline stage timings
            self.pipeline_stage_timings[metric.stage.value].append(metric.duration_ms)
            
            # Update Prometheus metrics
            success_label = "true" if metric.success else "false"
            
            self.pipeline_stage_duration.labels(
                stage=metric.stage.value,
                endpoint=metric.endpoint,
                success=success_label
            ).observe(metric.duration_ms / 1000.0)
            
            if metric.queue_time_ms is not None:
                self.pipeline_queue_time.labels(
                    stage=metric.stage.value
                ).observe(metric.queue_time_ms / 1000.0)
            
            # Error tracking
            if not metric.success and metric.error_type:
                self.pipeline_error_rate.labels(
                    stage=metric.stage.value,
                    error_type=metric.error_type
                ).set(1.0)  # Will be aggregated later
            
            self.logger.debug(f"Recorded pipeline stage: {metric.stage.value} for {metric.request_id}")
        
        except Exception as e:
            self.logger.error(f"Error recording pipeline stage metric: {e}")
    
    async def record_database_operation(self, metric: DatabasePerformanceMetric):
        """Record a database operation metric."""
        try:
            self.database_metrics.append(metric)
            self.collection_stats["database_queries_tracked"] += 1
            
            # Track slow queries
            if metric.execution_time_ms > self.slow_query_threshold_ms:
                self.collection_stats["slow_queries_detected"] += 1
                
                self.database_slow_queries.labels(
                    table=metric.table_name,
                    operation=metric.operation_type.value
                ).inc()
            
            # Update query cache for analysis
            query_key = f"{metric.operation_type.value}:{metric.table_name}:{metric.query_hash}"
            if query_key not in self.database_query_cache:
                self.database_query_cache[query_key] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "max_time": 0
                }
            
            cache_entry = self.database_query_cache[query_key]
            cache_entry["count"] += 1
            cache_entry["total_time"] += metric.execution_time_ms
            cache_entry["avg_time"] = cache_entry["total_time"] / cache_entry["count"]
            cache_entry["max_time"] = max(cache_entry["max_time"], metric.execution_time_ms)
            
            # Update Prometheus metrics
            self.database_query_duration.labels(
                operation=metric.operation_type.value,
                table=metric.table_name,
                query_plan=metric.query_plan_type
            ).observe(metric.execution_time_ms / 1000.0)
            
            if metric.connection_pool_size > 0:
                utilization = metric.active_connections / metric.connection_pool_size
                self.database_connection_pool_utilization.labels(
                    pool_name="default"
                ).set(utilization)
            
            if metric.lock_time_ms > 0:
                self.database_lock_time.labels(
                    table=metric.table_name,
                    operation=metric.operation_type.value
                ).observe(metric.lock_time_ms / 1000.0)
            
            self.logger.debug(f"Recorded database operation: {metric.operation_type.value} on {metric.table_name}")
        
        except Exception as e:
            self.logger.error(f"Error recording database operation metric: {e}")
    
    async def record_cache_operation(self, metric: CachePerformanceMetric):
        """Record a cache operation metric."""
        try:
            self.cache_metrics.append(metric)
            self.collection_stats["cache_operations_tracked"] += 1
            
            # Track cache misses
            if not metric.hit:
                self.collection_stats["cache_misses_detected"] += 1
            
            # Update Prometheus metrics
            self.cache_operation_duration.labels(
                cache_type=metric.cache_type.value,
                operation=metric.operation
            ).observe(metric.response_time_ms / 1000.0)
            
            # Cache eviction tracking
            if metric.eviction_triggered:
                self.cache_eviction_rate.labels(
                    cache_type=metric.cache_type.value,
                    reason="size_limit"  # Could be parameterized
                ).inc()
            
            self.logger.debug(f"Recorded cache operation: {metric.operation} on {metric.cache_type.value}")
        
        except Exception as e:
            self.logger.error(f"Error recording cache operation metric: {e}")
    
    async def record_external_api_call(self, metric: ExternalAPIMetric):
        """Record an external API call metric."""
        try:
            self.external_api_metrics.append(metric)
            self.collection_stats["external_api_calls_tracked"] += 1
            
            # Update Prometheus metrics
            status_class = f"{metric.status_code // 100}xx"
            
            self.external_api_response_time.labels(
                api_type=metric.api_type.value,
                endpoint=metric.endpoint,
                status_class=status_class
            ).observe(metric.response_time_ms / 1000.0)
            
            # Circuit breaker state mapping
            cb_state_value = {
                "closed": 0,
                "half_open": 1,
                "open": 2
            }.get(metric.circuit_breaker_state.lower(), 0)
            
            self.external_api_circuit_breaker_state.labels(
                api_type=metric.api_type.value,
                endpoint=metric.endpoint
            ).set(cb_state_value)
            
            self.logger.debug(f"Recorded external API call: {metric.api_type.value} to {metric.endpoint}")
        
        except Exception as e:
            self.logger.error(f"Error recording external API call metric: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu_usage in enumerate(cpu_percent):
                self.system_cpu_utilization.labels(core=f"cpu{i}").set(cpu_usage)
            
            # Overall CPU
            overall_cpu = psutil.cpu_percent(interval=1)
            self.system_cpu_utilization.labels(core="overall").set(overall_cpu)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.system_memory_utilization.labels(type="physical").set(memory.percent)
            self.system_memory_utilization.labels(type="available").set(
                (memory.available / memory.total) * 100
            )
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.system_disk_io.labels(operation="read", device="total").inc(disk_io.read_bytes)
                self.system_disk_io.labels(operation="write", device="total").inc(disk_io.write_bytes)
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics over time windows."""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(minutes=self.aggregation_window_minutes)
            
            # Aggregate pipeline metrics
            await self._aggregate_pipeline_metrics(window_start, current_time)
            
            # Aggregate database metrics
            await self._aggregate_database_metrics(window_start, current_time)
            
            # Aggregate cache metrics
            await self._aggregate_cache_metrics(window_start, current_time)
            
            # Aggregate external API metrics
            await self._aggregate_external_api_metrics(window_start, current_time)
            
            # Clean up completed requests
            await self._cleanup_completed_requests()
            
            # Clean up old metrics
            await self._cleanup_old_metrics(current_time)
            
            self.collection_stats["last_aggregation"] = current_time
            self.collection_stats["active_requests_count"] = len(self.active_requests)
            
        except Exception as e:
            self.logger.error(f"Error in performance metrics aggregation: {e}")
    
    async def _aggregate_pipeline_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate pipeline metrics."""
        window_metrics = [
            m for m in self.pipeline_metrics
            if window_start <= m.start_time <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by stage
        stage_groups = defaultdict(list)
        for metric in window_metrics:
            stage_groups[metric.stage].append(metric)
        
        # Calculate throughput and error rates
        window_duration_seconds = (window_end - window_start).total_seconds()
        
        for stage, metrics in stage_groups.items():
            if not metrics:
                continue
            
            # Throughput
            throughput = len(metrics) / window_duration_seconds if window_duration_seconds > 0 else 0
            self.pipeline_throughput.labels(
                stage=stage.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(throughput)
            
            # Error rate calculation
            error_groups = defaultdict(int)
            total_requests = len(metrics)
            
            for metric in metrics:
                if not metric.success and metric.error_type:
                    error_groups[metric.error_type] += 1
            
            for error_type, count in error_groups.items():
                error_rate = count / total_requests if total_requests > 0 else 0
                self.pipeline_error_rate.labels(
                    stage=stage.value,
                    error_type=error_type
                ).set(error_rate)
        
        self.logger.debug(f"Aggregated pipeline metrics: {len(window_metrics)} events")
    
    async def _aggregate_database_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate database metrics."""
        window_metrics = [
            m for m in self.database_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Calculate connection pool utilization
        if window_metrics:
            avg_pool_utilization = statistics.mean(
                m.active_connections / m.connection_pool_size
                for m in window_metrics
                if m.connection_pool_size > 0
            )
            
            self.database_connection_pool_utilization.labels(
                pool_name="aggregated"
            ).set(avg_pool_utilization)
        
        self.logger.debug(f"Aggregated database metrics: {len(window_metrics)} queries")
    
    async def _aggregate_cache_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate cache metrics."""
        window_metrics = [
            m for m in self.cache_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by cache type
        cache_groups = defaultdict(list)
        for metric in window_metrics:
            cache_groups[metric.cache_type].append(metric)
        
        # Calculate hit rates
        for cache_type, metrics in cache_groups.items():
            if not metrics:
                continue
            
            hits = sum(1 for m in metrics if m.hit)
            hit_rate = hits / len(metrics) if metrics else 0
            
            self.cache_hit_rate.labels(
                cache_type=cache_type.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(hit_rate)
        
        self.logger.debug(f"Aggregated cache metrics: {len(window_metrics)} operations")
    
    async def _aggregate_external_api_metrics(self, window_start: datetime, window_end: datetime):
        """Aggregate external API metrics."""
        window_metrics = [
            m for m in self.external_api_metrics
            if window_start <= m.timestamp <= window_end
        ]
        
        if not window_metrics:
            return
        
        # Group by API type and endpoint
        api_groups = defaultdict(list)
        for metric in window_metrics:
            key = (metric.api_type, metric.endpoint)
            api_groups[key].append(metric)
        
        # Calculate success rates
        for (api_type, endpoint), metrics in api_groups.items():
            if not metrics:
                continue
            
            successful_calls = sum(1 for m in metrics if m.success)
            success_rate = successful_calls / len(metrics) if metrics else 0
            
            self.external_api_success_rate.labels(
                api_type=api_type.value,
                endpoint=endpoint,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(success_rate)
        
        self.logger.debug(f"Aggregated external API metrics: {len(window_metrics)} calls")
    
    async def _cleanup_completed_requests(self):
        """Clean up completed requests from active tracking."""
        current_time = datetime.now(timezone.utc)
        completed_requests = []
        
        for request_id, request_data in self.active_requests.items():
            # Consider a request completed if it hasn't been updated in the last 5 minutes
            last_stage_time = max(
                (stage.get("timestamp", request_data["start_time"]) for stage in request_data.get("stages", [])),
                default=request_data["start_time"]
            )
            
            if isinstance(last_stage_time, str):
                # Handle string timestamps
                continue
            
            if current_time - last_stage_time > timedelta(minutes=5):
                completed_requests.append(request_id)
        
        for request_id in completed_requests:
            del self.active_requests[request_id]
        
        if completed_requests:
            self.logger.debug(f"Cleaned up {len(completed_requests)} completed requests")
    
    async def _cleanup_old_metrics(self, current_time: datetime):
        """Clean up metrics older than retention period."""
        cutoff_time = current_time - timedelta(hours=self.retention_hours)
        
        # Clean pipeline metrics
        original_pipeline_count = len(self.pipeline_metrics)
        self.pipeline_metrics = deque(
            (m for m in self.pipeline_metrics if m.start_time > cutoff_time),
            maxlen=self.pipeline_metrics.maxlen
        )
        
        # Clean database metrics
        original_db_count = len(self.database_metrics)
        self.database_metrics = deque(
            (m for m in self.database_metrics if m.timestamp > cutoff_time),
            maxlen=self.database_metrics.maxlen
        )
        
        # Clean cache metrics
        original_cache_count = len(self.cache_metrics)
        self.cache_metrics = deque(
            (m for m in self.cache_metrics if m.timestamp > cutoff_time),
            maxlen=self.cache_metrics.maxlen
        )
        
        # Clean external API metrics
        original_api_count = len(self.external_api_metrics)
        self.external_api_metrics = deque(
            (m for m in self.external_api_metrics if m.timestamp > cutoff_time),
            maxlen=self.external_api_metrics.maxlen
        )
        
        cleaned_total = (
            (original_pipeline_count - len(self.pipeline_metrics)) +
            (original_db_count - len(self.database_metrics)) +
            (original_cache_count - len(self.cache_metrics)) +
            (original_api_count - len(self.external_api_metrics))
        )
        
        if cleaned_total > 0:
            self.logger.debug(f"Cleaned up {cleaned_total} old performance metrics")
    
    async def get_pipeline_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get pipeline performance summary."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.pipeline_metrics if m.start_time > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "hours": hours}
        
        # Group by stage
        stage_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0,
            "success_count": 0,
            "queue_times": [],
            "memory_usage": [],
            "error_types": defaultdict(int)
        })
        
        for metric in recent_metrics:
            stats = stage_stats[metric.stage]
            stats["count"] += 1
            stats["total_duration"] += metric.duration_ms
            if metric.success:
                stats["success_count"] += 1
            else:
                stats["error_types"][metric.error_type or "unknown"] += 1
            
            if metric.queue_time_ms is not None:
                stats["queue_times"].append(metric.queue_time_ms)
            stats["memory_usage"].append(metric.memory_usage_mb)
        
        # Calculate derived metrics
        pipeline_summary = {}
        for stage, stats in stage_stats.items():
            count = stats["count"]
            
            pipeline_summary[stage.value] = {
                "request_count": count,
                "avg_duration_ms": stats["total_duration"] / count if count > 0 else 0,
                "success_rate": stats["success_count"] / count if count > 0 else 0,
                "requests_per_hour": count / hours,
                "avg_queue_time_ms": statistics.mean(stats["queue_times"]) if stats["queue_times"] else 0,
                "avg_memory_usage_mb": statistics.mean(stats["memory_usage"]) if stats["memory_usage"] else 0,
                "error_breakdown": dict(stats["error_types"])
            }
        
        return {
            "total_pipeline_events": len(recent_metrics),
            "active_requests": len(self.active_requests),
            "stage_performance": pipeline_summary,
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_database_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get database performance summary."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.database_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "hours": hours}
        
        # Analyze query performance
        table_stats = defaultdict(lambda: {
            "queries": [],
            "operations": defaultdict(int),
            "slow_queries": 0
        })
        
        for metric in recent_metrics:
            stats = table_stats[metric.table_name]
            stats["queries"].append(metric)
            stats["operations"][metric.operation_type.value] += 1
            
            if metric.execution_time_ms > self.slow_query_threshold_ms:
                stats["slow_queries"] += 1
        
        # Calculate performance metrics
        db_summary = {}
        for table, stats in table_stats.items():
            queries = stats["queries"]
            
            db_summary[table] = {
                "total_queries": len(queries),
                "avg_execution_time_ms": statistics.mean(q.execution_time_ms for q in queries),
                "max_execution_time_ms": max(q.execution_time_ms for q in queries),
                "slow_query_count": stats["slow_queries"],
                "slow_query_rate": stats["slow_queries"] / len(queries) if queries else 0,
                "avg_rows_affected": statistics.mean(q.rows_affected for q in queries),
                "avg_connection_pool_utilization": statistics.mean(
                    q.active_connections / q.connection_pool_size
                    for q in queries if q.connection_pool_size > 0
                ) if any(q.connection_pool_size > 0 for q in queries) else 0,
                "operation_breakdown": dict(stats["operations"])
            }
        
        return {
            "total_queries": len(recent_metrics),
            "slow_queries_detected": self.collection_stats["slow_queries_detected"],
            "table_performance": db_summary,
            "query_cache_entries": len(self.database_query_cache),
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        return {
            **self.collection_stats,
            "current_metrics_count": {
                "pipeline_stages": len(self.pipeline_metrics),
                "database_operations": len(self.database_metrics),
                "cache_operations": len(self.cache_metrics),
                "external_api_calls": len(self.external_api_metrics)
            },
            "pipeline_stage_cache_size": sum(len(timings) for timings in self.pipeline_stage_timings.values()),
            "database_query_cache_size": len(self.database_query_cache),
            "is_running": self.is_running,
            "config": {
                "aggregation_window_minutes": self.aggregation_window_minutes,
                "retention_hours": self.retention_hours,
                "slow_query_threshold_ms": self.slow_query_threshold_ms,
                "cache_efficiency_threshold": self.cache_efficiency_threshold
            }
        }

# Global instance
_performance_metrics_collector: Optional[PerformanceMetricsCollector] = None

def get_performance_metrics_collector(config: Optional[Dict[str, Any]] = None) -> PerformanceMetricsCollector:
    """Get global performance metrics collector instance."""
    global _performance_metrics_collector
    if _performance_metrics_collector is None:
        _performance_metrics_collector = PerformanceMetricsCollector(config)
    return _performance_metrics_collector

# Convenience functions for recording metrics
async def record_pipeline_stage_timing(
    request_id: str,
    stage: PipelineStage,
    start_time: datetime,
    end_time: datetime,
    success: bool = True,
    error_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    endpoint: str = "unknown",
    method: str = "unknown",
    queue_time_ms: Optional[float] = None,
    retry_count: int = 0,
    stage_specific_data: Optional[Dict[str, Any]] = None
):
    """Record pipeline stage timing (convenience function)."""
    collector = get_performance_metrics_collector()
    
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    # Get current system metrics
    try:
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        cpu_usage_percent = process.cpu_percent()
    except:
        memory_usage_mb = 0
        cpu_usage_percent = 0
    
    metric = RequestPipelineMetric(
        request_id=request_id,
        stage=stage,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        success=success,
        error_type=error_type,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent,
        stage_specific_data=stage_specific_data or {},
        user_id=user_id,
        session_id=session_id,
        endpoint=endpoint,
        method=method,
        queue_time_ms=queue_time_ms,
        retry_count=retry_count
    )
    
    await collector.record_pipeline_stage(metric)