"""
Comprehensive Redis Health Monitoring System

This module provides extensive Redis health monitoring capabilities including:
- Memory usage tracking and analysis
- Performance metrics with latency percentiles  
- Persistence health (RDB/AOF status)
- Replication monitoring for master/replica setups
- Client connection tracking and analysis
- Keyspace analytics with memory profiling
- Slow log analysis and performance insights
- Real-time metrics collection with async operations
- Production-ready monitoring with actual Redis connections

Uses coredis for async operations and integrates with the existing health system.
"""

import asyncio
import time
import statistics
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re

import coredis

# Import configuration for Redis connection
from ..core.config import get_config

# Import metrics registry for error tracking
from ..performance.monitoring.metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)

# Initialize metrics for cache error tracking
_metrics_registry = get_metrics_registry()
CACHE_ERRORS = _metrics_registry.get_or_create_counter(
    'cache_errors_total',
    'Total cache operation errors',
    ['operation', 'error_type']
)

# Global Redis client instance
_redis_client: Optional[coredis.Redis] = None

async def get_redis_client() -> Optional[coredis.Redis]:
    """Get or create global Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            config = get_config()
            redis_url = config.get_redis_url()
            _redis_client = coredis.Redis.from_url(redis_url, decode_responses=True)
            await _redis_client.ping()
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")
            return None
    return _redis_client

# For backward compatibility
redis_client = None  # Will be set by get_redis_client()

def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert Redis response to int."""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert Redis response to float."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _safe_str(value: Any, default: str = "") -> str:
    """Safely convert Redis response to string."""
    try:
        return str(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert Redis response to boolean."""
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return default
    except (ValueError, TypeError):
        return default

class RedisHealthStatus(Enum):
    """Redis health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class RedisRole(Enum):
    """Redis server roles."""
    MASTER = "master"
    SLAVE = "slave"  # Redis uses 'slave' terminology
    REPLICA = "replica"
    STANDALONE = "standalone"

@dataclass
class MemoryMetrics:
    """Redis memory usage metrics."""
    used_memory: int = 0
    used_memory_human: str = "0B"
    used_memory_rss: int = 0
    used_memory_peak: int = 0
    used_memory_peak_human: str = "0B"
    mem_fragmentation_ratio: float = 1.0
    used_memory_overhead: int = 0
    used_memory_dataset: int = 0
    total_system_memory: int = 0
    maxmemory: int = 0
    maxmemory_human: str = "0B"
    maxmemory_policy: str = "noeviction"
    memory_usage_percentage: float = 0.0
    fragmentation_bytes: int = 0
    
    def is_healthy(self) -> bool:
        """Check if memory metrics indicate healthy state."""
        # Critical if using >85% of max memory or fragmentation >3.0
        if self.maxmemory > 0 and self.memory_usage_percentage > 85.0:
            return False
        if self.mem_fragmentation_ratio > 3.0:
            return False
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on memory metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if using >70% memory or fragmentation >2.0
        if self.maxmemory > 0 and self.memory_usage_percentage > 70.0:
            return RedisHealthStatus.WARNING
        if self.mem_fragmentation_ratio > 2.0:
            return RedisHealthStatus.WARNING
            
        return RedisHealthStatus.HEALTHY

@dataclass
class PerformanceMetrics:
    """Redis performance metrics."""
    ops_per_sec: float = 0.0
    instantaneous_ops_per_sec: int = 0
    total_commands_processed: int = 0
    keyspace_hits: int = 0
    keyspace_misses: int = 0
    hit_rate: float = 0.0
    expired_keys: int = 0
    evicted_keys: int = 0
    rejected_connections: int = 0
    total_connections_received: int = 0
    
    # Latency tracking
    latency_samples: List[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    def calculate_hit_rate(self):
        """Calculate cache hit rate."""
        total = self.keyspace_hits + self.keyspace_misses
        self.hit_rate = (self.keyspace_hits / total * 100) if total > 0 else 0.0
    
    def add_latency_sample(self, latency_ms: float):
        """Add a latency sample and update percentiles."""
        self.latency_samples.append(latency_ms)
        # Keep only last 1000 samples for percentile calculations
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
        
        if self.latency_samples:
            sorted_samples = sorted(self.latency_samples)
            n = len(sorted_samples)
            
            self.avg_latency_ms = statistics.mean(sorted_samples)
            self.p50_latency_ms = sorted_samples[int(n * 0.5)]
            self.p95_latency_ms = sorted_samples[int(n * 0.95)]
            self.p99_latency_ms = sorted_samples[int(n * 0.99)]
            self.max_latency_ms = max(sorted_samples)
    
    def is_healthy(self) -> bool:
        """Check if performance metrics indicate healthy state."""
        # Critical if hit rate <50% or P95 latency >500ms
        if self.hit_rate < 50.0:
            return False
        if self.p95_latency_ms > 500.0:
            return False
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on performance metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if hit rate <80% or P95 latency >200ms
        if self.hit_rate < 80.0:
            return RedisHealthStatus.WARNING
        if self.p95_latency_ms > 200.0:
            return RedisHealthStatus.WARNING
            
        return RedisHealthStatus.HEALTHY

@dataclass
class PersistenceMetrics:
    """Redis persistence health metrics."""
    # RDB metrics
    rdb_changes_since_last_save: int = 0
    rdb_bgsave_in_progress: bool = False
    rdb_last_save_time: int = 0
    rdb_last_bgsave_status: str = "ok"
    rdb_last_bgsave_time_sec: int = 0
    rdb_current_bgsave_time_sec: int = 0
    
    # AOF metrics
    aof_enabled: bool = False
    aof_rewrite_in_progress: bool = False
    aof_rewrite_scheduled: bool = False
    aof_last_rewrite_time_sec: int = 0
    aof_current_rewrite_time_sec: int = 0
    aof_last_bgrewrite_status: str = "ok"
    aof_current_size: int = 0
    aof_base_size: int = 0
    aof_pending_rewrite: bool = False
    aof_buffer_length: int = 0
    aof_rewrite_buffer_length: int = 0
    aof_pending_bio_fsync: int = 0
    aof_delayed_fsync: int = 0
    
    # Last save time analysis
    last_save_age_minutes: float = 0.0
    
    def update_last_save_age(self):
        """Update the age of last save operation."""
        if self.rdb_last_save_time > 0:
            current_time = time.time()
            self.last_save_age_minutes = (current_time - self.rdb_last_save_time) / 60.0
    
    def is_healthy(self) -> bool:
        """Check if persistence metrics indicate healthy state."""
        # Critical if last save was >60 minutes ago and changes exist
        self.update_last_save_age()
        if (self.last_save_age_minutes > 60.0 and 
            self.rdb_changes_since_last_save > 1000):
            return False
        
        # Critical if last save/rewrite failed
        if (self.rdb_last_bgsave_status != "ok" or 
            (self.aof_enabled and self.aof_last_bgrewrite_status != "ok")):
            return False
            
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on persistence metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if last save was >30 minutes ago
        if self.last_save_age_minutes > 30.0 and self.rdb_changes_since_last_save > 100:
            return RedisHealthStatus.WARNING
        
        # Warning if operations are in progress for too long
        if (self.rdb_current_bgsave_time_sec > 300 or  # 5 minutes
            self.aof_current_rewrite_time_sec > 300):
            return RedisHealthStatus.WARNING
            
        return RedisHealthStatus.HEALTHY

@dataclass 
class ReplicationMetrics:
    """Redis replication metrics."""
    role: RedisRole = RedisRole.STANDALONE
    connected_slaves: int = 0
    master_replid: str = ""
    master_replid2: str = ""
    master_repl_offset: int = 0
    second_repl_offset: int = -1
    repl_backlog_active: bool = False
    repl_backlog_size: int = 0
    repl_backlog_first_byte_offset: int = 0
    repl_backlog_histlen: int = 0
    
    # Master-specific metrics
    slave_info: List[Dict[str, Any]] = field(default_factory=list)
    
    # Slave-specific metrics  
    master_host: str = ""
    master_port: int = 0
    master_link_status: str = "up"
    master_last_io_seconds_ago: int = 0
    master_sync_in_progress: bool = False
    slave_repl_offset: int = 0
    slave_priority: int = 100
    slave_read_only: bool = True
    
    # Replication lag
    replication_lag_bytes: int = 0
    replication_lag_seconds: float = 0.0
    
    def calculate_replication_lag(self):
        """Calculate replication lag metrics."""
        if self.role == RedisRole.SLAVE and self.master_repl_offset > 0:
            self.replication_lag_bytes = max(0, self.master_repl_offset - self.slave_repl_offset)
            # Estimate lag in seconds based on typical Redis throughput
            # This is approximate - actual lag depends on write load
            if self.master_last_io_seconds_ago >= 0:
                self.replication_lag_seconds = float(self.master_last_io_seconds_ago)
    
    def is_healthy(self) -> bool:
        """Check if replication metrics indicate healthy state."""
        if self.role == RedisRole.SLAVE:
            # Critical if master link is down or lag is high
            if self.master_link_status != "up":
                return False
            if self.replication_lag_seconds > 30.0:  # 30 second lag
                return False
            if self.master_last_io_seconds_ago > 60:  # No communication for 1 minute
                return False
        
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on replication metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        if self.role == RedisRole.SLAVE:
            # Warning if moderate lag or communication delay
            if self.replication_lag_seconds > 10.0:
                return RedisHealthStatus.WARNING
            if self.master_last_io_seconds_ago > 30:
                return RedisHealthStatus.WARNING
                
        return RedisHealthStatus.HEALTHY

@dataclass
class ConnectionMetrics:
    """Redis client connection metrics."""
    connected_clients: int = 0
    client_recent_max_input_buffer: int = 0
    client_recent_max_output_buffer: int = 0
    blocked_clients: int = 0
    tracking_clients: int = 0
    clients_in_timeout_table: int = 0
    total_connections_received: int = 0
    rejected_connections: int = 0
    maxclients: int = 10000
    
    # Connection pool analysis
    connection_utilization: float = 0.0
    connections_per_second: float = 0.0
    
    def calculate_utilization(self):
        """Calculate connection pool utilization."""
        if self.maxclients > 0:
            self.connection_utilization = (self.connected_clients / self.maxclients) * 100.0
    
    def is_healthy(self) -> bool:
        """Check if connection metrics indicate healthy state."""
        self.calculate_utilization()
        
        # Critical if using >90% of max connections or many blocked clients
        if self.connection_utilization > 90.0:
            return False
        if self.blocked_clients > self.connected_clients * 0.5:  # >50% blocked
            return False
        if self.rejected_connections > 0:
            return False
            
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on connection metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if using >70% connections or some blocked clients
        if self.connection_utilization > 70.0:
            return RedisHealthStatus.WARNING
        if self.blocked_clients > 0:
            return RedisHealthStatus.WARNING
            
        return RedisHealthStatus.HEALTHY

@dataclass
class KeyspaceInfo:
    """Keyspace database information."""
    db_id: int
    keys: int = 0
    expires: int = 0
    avg_ttl: float = 0.0
    memory_usage_mb: float = 0.0
    
    def get_expiry_ratio(self) -> float:
        """Get ratio of keys with expiration."""
        return (self.expires / self.keys * 100.0) if self.keys > 0 else 0.0

@dataclass
class KeyspaceMetrics:
    """Redis keyspace analytics."""
    databases: Dict[int, KeyspaceInfo] = field(default_factory=dict)
    total_keys: int = 0
    sample_key_memory: Dict[str, float] = field(default_factory=dict)  # Key pattern -> avg memory
    large_keys: List[Tuple[str, int]] = field(default_factory=list)  # Key name, size in bytes
    key_distribution: Dict[str, int] = field(default_factory=dict)  # Pattern -> count
    
    def calculate_totals(self):
        """Calculate total metrics from all databases."""
        self.total_keys = sum(db.keys for db in self.databases.values())
    
    def is_healthy(self) -> bool:
        """Check if keyspace metrics indicate healthy state."""
        # Check for extremely large keys (>100MB)
        for _, size in self.large_keys:
            if size > 100 * 1024 * 1024:  # 100MB
                return False
        
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on keyspace metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if large keys exist (>10MB)
        for _, size in self.large_keys:
            if size > 10 * 1024 * 1024:  # 10MB
                return RedisHealthStatus.WARNING
                
        return RedisHealthStatus.HEALTHY

@dataclass
class SlowLogEntry:
    """Redis slow log entry."""
    id: int
    timestamp: int
    duration_microseconds: int
    command: List[str]
    client_ip: str = ""
    client_name: str = ""
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration_microseconds / 1000.0
    
    @property
    def command_str(self) -> str:
        """Get command as string."""
        return " ".join(str(cmd) for cmd in self.command[:10])  # Limit to first 10 args

@dataclass
class SlowLogMetrics:
    """Redis slow log analysis."""
    entries: List[SlowLogEntry] = field(default_factory=list)
    slowlog_len: int = 0
    slowlog_max_len: int = 128
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    command_frequency: Dict[str, int] = field(default_factory=dict)
    recent_slow_commands: int = 0  # Commands in last 5 minutes
    
    def analyze_entries(self):
        """Analyze slow log entries."""
        if not self.entries:
            return
        
        durations = [entry.duration_ms for entry in self.entries]
        self.avg_duration_ms = statistics.mean(durations)
        self.max_duration_ms = max(durations)
        
        # Count command frequencies
        self.command_frequency.clear()
        for entry in self.entries:
            cmd = entry.command[0].upper() if entry.command else "UNKNOWN"
            self.command_frequency[cmd] = self.command_frequency.get(cmd, 0) + 1
        
        # Count recent slow commands (last 5 minutes)
        current_time = time.time()
        five_minutes_ago = current_time - 300
        self.recent_slow_commands = sum(
            1 for entry in self.entries 
            if entry.timestamp >= five_minutes_ago
        )
    
    def is_healthy(self) -> bool:
        """Check if slow log metrics indicate healthy state."""
        # Critical if many recent slow commands or very slow commands
        if self.recent_slow_commands > 10:
            return False
        if self.max_duration_ms > 1000.0:  # 1 second
            return False
            
        return True
    
    def get_status(self) -> RedisHealthStatus:
        """Get health status based on slow log metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        
        # Warning if some recent slow commands or moderately slow
        if self.recent_slow_commands > 2:
            return RedisHealthStatus.WARNING
        if self.max_duration_ms > 100.0:  # 100ms
            return RedisHealthStatus.WARNING
            
        return RedisHealthStatus.HEALTHY

class RedisHealthMonitor:
    """
    Comprehensive Redis health monitoring system.
    
    Provides extensive monitoring capabilities including:
    - Memory usage tracking and fragmentation analysis
    - Performance metrics with latency percentiles
    - Persistence health monitoring (RDB/AOF)
    - Replication status and lag monitoring
    - Client connection analysis
    - Keyspace analytics and memory profiling
    - Slow log analysis and optimization insights
    - Real-time metrics collection with circuit breaker protection
    """
    
    def __init__(self, client: Optional[coredis.Redis] = None):
        """
        Initialize Redis health monitor.

        Args:
            client: Optional Redis client. Uses global client if not provided.
        """
        self.client = client
        self._client_initialized = False
        self.last_check_time: Optional[datetime] = None
        self.check_count = 0
        
        # Metrics storage
        self.memory_metrics = MemoryMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.persistence_metrics = PersistenceMetrics()
        self.replication_metrics = ReplicationMetrics()
        self.connection_metrics = ConnectionMetrics()
        self.keyspace_metrics = KeyspaceMetrics()
        self.slowlog_metrics = SlowLogMetrics()
        
        # Configuration
        self.max_large_keys_to_track = 50
        self.slowlog_entries_to_analyze = 100
        self.keyspace_sample_size = 1000

    async def _ensure_client(self) -> bool:
        """Ensure Redis client is available."""
        if self.client is None and not self._client_initialized:
            self.client = await get_redis_client()
            self._client_initialized = True
        return self.client is not None
        
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all Redis health metrics.
        
        Returns:
            Comprehensive health metrics dictionary
        """
        start_time = time.time()
        
        try:
            # Ensure Redis client is available
            if not await self._ensure_client():
                return {
                    "status": RedisHealthStatus.FAILED.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "Redis client not available",
                    "check_duration_ms": round((time.time() - start_time) * 1000, 2),
                    "check_count": self.check_count
                }

            # Test basic connectivity first
            await self._test_connectivity()
            
            # Collect all metrics in parallel where possible
            await asyncio.gather(
                self._collect_memory_metrics(),
                self._collect_performance_metrics(),
                self._collect_persistence_metrics(),
                self._collect_replication_metrics(),
                self._collect_connection_metrics(),
                self._collect_keyspace_metrics(),
                self._collect_slowlog_metrics(),
                return_exceptions=True
            )
            
            # Calculate overall health status
            overall_status = self._calculate_overall_status()
            
            self.last_check_time = datetime.now(timezone.utc)
            self.check_count += 1
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": overall_status.value,
                "timestamp": self.last_check_time.isoformat(),
                "check_duration_ms": round(duration_ms, 2),
                "check_count": self.check_count,
                "memory": self._memory_metrics_to_dict(),
                "performance": self._performance_metrics_to_dict(),
                "persistence": self._persistence_metrics_to_dict(),
                "replication": self._replication_metrics_to_dict(),
                "connections": self._connection_metrics_to_dict(),
                "keyspace": self._keyspace_metrics_to_dict(),
                "slowlog": self._slowlog_metrics_to_dict(),
                "recommendations": self._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect Redis health metrics: {e}")
            CACHE_ERRORS.labels(operation='health_check').inc()
            
            return {
                "status": RedisHealthStatus.FAILED.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "check_duration_ms": round((time.time() - start_time) * 1000, 2),
                "check_count": self.check_count
            }
    
    async def _test_connectivity(self):
        """Test basic Redis connectivity."""
        if not self.client:
            raise Exception("Redis client not available")
        start_time = time.time()
        await self.client.ping()
        latency_ms = (time.time() - start_time) * 1000
        self.performance_metrics.add_latency_sample(latency_ms)
    
    async def _collect_memory_metrics(self):
        """Collect Redis memory usage metrics."""
        try:
            if not self.client:
                return
            info = await self.client.info()

            self.memory_metrics.used_memory = _safe_int(info.get("used_memory", 0))
            self.memory_metrics.used_memory_human = _safe_str(info.get("used_memory_human", "0B"))
            self.memory_metrics.used_memory_rss = _safe_int(info.get("used_memory_rss", 0))
            self.memory_metrics.used_memory_peak = _safe_int(info.get("used_memory_peak", 0))
            self.memory_metrics.used_memory_peak_human = _safe_str(info.get("used_memory_peak_human", "0B"))
            self.memory_metrics.mem_fragmentation_ratio = _safe_float(info.get("mem_fragmentation_ratio", 1.0))
            self.memory_metrics.used_memory_overhead = _safe_int(info.get("used_memory_overhead", 0))
            self.memory_metrics.used_memory_dataset = _safe_int(info.get("used_memory_dataset", 0))
            self.memory_metrics.total_system_memory = _safe_int(info.get("total_system_memory", 0))
            self.memory_metrics.maxmemory = _safe_int(info.get("maxmemory", 0))
            self.memory_metrics.maxmemory_human = _safe_str(info.get("maxmemory_human", "0B"))
            self.memory_metrics.maxmemory_policy = _safe_str(info.get("maxmemory_policy", "noeviction"))
            
            # Calculate derived metrics
            if self.memory_metrics.maxmemory > 0:
                self.memory_metrics.memory_usage_percentage = (
                    self.memory_metrics.used_memory / self.memory_metrics.maxmemory * 100.0
                )
            
            if self.memory_metrics.mem_fragmentation_ratio > 1.0:
                self.memory_metrics.fragmentation_bytes = (
                    self.memory_metrics.used_memory_rss - self.memory_metrics.used_memory
                )
                
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect Redis performance metrics."""
        try:
            if not self.client:
                return
            info = await self.client.info()

            self.performance_metrics.instantaneous_ops_per_sec = _safe_int(info.get("instantaneous_ops_per_sec", 0))
            self.performance_metrics.total_commands_processed = _safe_int(info.get("total_commands_processed", 0))
            self.performance_metrics.keyspace_hits = _safe_int(info.get("keyspace_hits", 0))
            self.performance_metrics.keyspace_misses = _safe_int(info.get("keyspace_misses", 0))
            self.performance_metrics.expired_keys = _safe_int(info.get("expired_keys", 0))
            self.performance_metrics.evicted_keys = _safe_int(info.get("evicted_keys", 0))
            self.performance_metrics.rejected_connections = _safe_int(info.get("rejected_connections", 0))
            self.performance_metrics.total_connections_received = _safe_int(info.get("total_connections_received", 0))
            
            # Calculate hit rate
            self.performance_metrics.calculate_hit_rate()
            
            # Calculate operations per second if we have previous data
            current_time = time.time()
            if hasattr(self, '_last_command_count') and hasattr(self, '_last_check_time'):
                time_diff = current_time - self._last_check_time
                if time_diff > 0:
                    command_diff = self.performance_metrics.total_commands_processed - self._last_command_count
                    self.performance_metrics.ops_per_sec = command_diff / time_diff
            
            self._last_command_count = self.performance_metrics.total_commands_processed
            self._last_check_time = current_time
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _collect_persistence_metrics(self):
        """Collect Redis persistence metrics."""
        try:
            if not self.client:
                return
            info = await self.client.info()
            
            # RDB metrics
            self.persistence_metrics.rdb_changes_since_last_save = _safe_int(info.get("rdb_changes_since_last_save", 0))
            self.persistence_metrics.rdb_bgsave_in_progress = _safe_bool(info.get("rdb_bgsave_in_progress", 0))
            self.persistence_metrics.rdb_last_save_time = _safe_int(info.get("rdb_last_save_time", 0))
            self.persistence_metrics.rdb_last_bgsave_status = _safe_str(info.get("rdb_last_bgsave_status", "ok"))
            self.persistence_metrics.rdb_last_bgsave_time_sec = _safe_int(info.get("rdb_last_bgsave_time_sec", 0))
            self.persistence_metrics.rdb_current_bgsave_time_sec = _safe_int(info.get("rdb_current_bgsave_time_sec", 0))

            # AOF metrics
            self.persistence_metrics.aof_enabled = _safe_bool(info.get("aof_enabled", 0))
            if self.persistence_metrics.aof_enabled:
                self.persistence_metrics.aof_rewrite_in_progress = _safe_bool(info.get("aof_rewrite_in_progress", 0))
                self.persistence_metrics.aof_rewrite_scheduled = _safe_bool(info.get("aof_rewrite_scheduled", 0))
                self.persistence_metrics.aof_last_rewrite_time_sec = _safe_int(info.get("aof_last_rewrite_time_sec", 0))
                self.persistence_metrics.aof_current_rewrite_time_sec = _safe_int(info.get("aof_current_rewrite_time_sec", 0))
                self.persistence_metrics.aof_last_bgrewrite_status = _safe_str(info.get("aof_last_bgrewrite_status", "ok"))
                self.persistence_metrics.aof_current_size = _safe_int(info.get("aof_current_size", 0))
                self.persistence_metrics.aof_base_size = _safe_int(info.get("aof_base_size", 0))
                self.persistence_metrics.aof_pending_rewrite = _safe_bool(info.get("aof_pending_rewrite", 0))
                self.persistence_metrics.aof_buffer_length = _safe_int(info.get("aof_buffer_length", 0))
                self.persistence_metrics.aof_rewrite_buffer_length = _safe_int(info.get("aof_rewrite_buffer_length", 0))
                self.persistence_metrics.aof_pending_bio_fsync = _safe_int(info.get("aof_pending_bio_fsync", 0))
                self.persistence_metrics.aof_delayed_fsync = _safe_int(info.get("aof_delayed_fsync", 0))
            
        except Exception as e:
            logger.error(f"Failed to collect persistence metrics: {e}")
    
    async def _collect_replication_metrics(self):
        """Collect Redis replication metrics."""
        try:
            if not self.client:
                return
            info = await self.client.info()
            
            # Role detection
            role = _safe_str(info.get("role", "master"))
            if role == "master":
                self.replication_metrics.role = RedisRole.MASTER
            elif role in ["slave", "replica"]:
                self.replication_metrics.role = RedisRole.SLAVE
            else:
                self.replication_metrics.role = RedisRole.STANDALONE

            # Common metrics
            self.replication_metrics.connected_slaves = _safe_int(info.get("connected_slaves", 0))
            self.replication_metrics.master_replid = _safe_str(info.get("master_replid", ""))
            self.replication_metrics.master_replid2 = _safe_str(info.get("master_replid2", ""))
            self.replication_metrics.master_repl_offset = _safe_int(info.get("master_repl_offset", 0))
            self.replication_metrics.second_repl_offset = _safe_int(info.get("second_repl_offset", -1))
            self.replication_metrics.repl_backlog_active = _safe_bool(info.get("repl_backlog_active", 0))
            self.replication_metrics.repl_backlog_size = _safe_int(info.get("repl_backlog_size", 0))
            self.replication_metrics.repl_backlog_first_byte_offset = _safe_int(info.get("repl_backlog_first_byte_offset", 0))
            self.replication_metrics.repl_backlog_histlen = _safe_int(info.get("repl_backlog_histlen", 0))
            
            # Slave-specific metrics
            if self.replication_metrics.role == RedisRole.SLAVE:
                self.replication_metrics.master_host = _safe_str(info.get("master_host", ""))
                self.replication_metrics.master_port = _safe_int(info.get("master_port", 0))
                self.replication_metrics.master_link_status = _safe_str(info.get("master_link_status", "up"))
                self.replication_metrics.master_last_io_seconds_ago = _safe_int(info.get("master_last_io_seconds_ago", 0))
                self.replication_metrics.master_sync_in_progress = _safe_bool(info.get("master_sync_in_progress", 0))
                self.replication_metrics.slave_repl_offset = _safe_int(info.get("slave_repl_offset", 0))
                self.replication_metrics.slave_priority = _safe_int(info.get("slave_priority", 100))
                self.replication_metrics.slave_read_only = _safe_bool(info.get("slave_read_only", 1))
                
                # Calculate replication lag
                self.replication_metrics.calculate_replication_lag()
            
            # Master-specific: parse slave info
            if self.replication_metrics.role == RedisRole.MASTER:
                slave_info = []
                for i in range(self.replication_metrics.connected_slaves):
                    slave_key = f"slave{i}"
                    if slave_key in info:
                        slave_data = _safe_str(info[slave_key])
                        # Parse slave info string: "ip=127.0.0.1,port=6380,state=online,offset=123,lag=0"
                        slave_dict = {}
                        if slave_data:
                            for item in slave_data.split(','):
                                if '=' in item:
                                    key, value = item.split('=', 1)
                                    slave_dict[key] = value
                        slave_info.append(slave_dict)
                self.replication_metrics.slave_info = slave_info
            
        except Exception as e:
            logger.error(f"Failed to collect replication metrics: {e}")
    
    async def _collect_connection_metrics(self):
        """Collect Redis connection metrics."""
        try:
            if not self.client:
                return
            info = await self.client.info()

            self.connection_metrics.connected_clients = _safe_int(info.get("connected_clients", 0))
            self.connection_metrics.client_recent_max_input_buffer = _safe_int(info.get("client_recent_max_input_buffer", 0))
            self.connection_metrics.client_recent_max_output_buffer = _safe_int(info.get("client_recent_max_output_buffer", 0))
            self.connection_metrics.blocked_clients = _safe_int(info.get("blocked_clients", 0))
            self.connection_metrics.tracking_clients = _safe_int(info.get("tracking_clients", 0))
            self.connection_metrics.clients_in_timeout_table = _safe_int(info.get("clients_in_timeout_table", 0))
            
            # Stats from same info call (no separate section needed)
            self.connection_metrics.total_connections_received = _safe_int(info.get("total_connections_received", 0))
            self.connection_metrics.rejected_connections = _safe_int(info.get("rejected_connections", 0))

            # Config for maxclients
            try:
                config = await self.client.config_get(["maxclients"])
                if config and "maxclients" in config:
                    self.connection_metrics.maxclients = _safe_int(config["maxclients"])
            except:
                # Default maxclients if config get fails
                self.connection_metrics.maxclients = 10000
            
            # Calculate derived metrics
            self.connection_metrics.calculate_utilization()
            
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
    
    async def _collect_keyspace_metrics(self):
        """Collect Redis keyspace analytics."""
        try:
            if not self.client:
                return
            info = await self.client.info()
            
            # Parse keyspace info
            self.keyspace_metrics.databases.clear()
            for key, value in info.items():
                if isinstance(key, str) and key.startswith("db"):
                    try:
                        db_id = int(key[2:])  # Extract number from "db0", "db1", etc.

                        # Parse db info: "keys=123,expires=45,avg_ttl=67890"
                        db_info = KeyspaceInfo(db_id=db_id)
                        value_str = _safe_str(value)
                        if value_str:
                            for item in value_str.split(','):
                                if '=' in item:
                                    metric_key, metric_value = item.split('=', 1)
                                    if metric_key == "keys":
                                        db_info.keys = _safe_int(metric_value)
                                    elif metric_key == "expires":
                                        db_info.expires = _safe_int(metric_value)
                                    elif metric_key == "avg_ttl":
                                        db_info.avg_ttl = _safe_float(metric_value)

                        self.keyspace_metrics.databases[db_id] = db_info
                    except (ValueError, TypeError):
                        continue  # Skip invalid database entries
            
            # Calculate totals
            self.keyspace_metrics.calculate_totals()
            
            # Sample key memory usage and find large keys
            await self._analyze_key_patterns()
            
        except Exception as e:
            logger.error(f"Failed to collect keyspace metrics: {e}")
    
    async def _analyze_key_patterns(self):
        """Analyze key patterns and memory usage."""
        try:
            if not self.client:
                return

            # Reset collections
            self.keyspace_metrics.sample_key_memory.clear()
            self.keyspace_metrics.large_keys.clear()
            self.keyspace_metrics.key_distribution.clear()

            # Temporary collection for memory samples (pattern -> list of memory values)
            temp_memory_samples: Dict[str, List[float]] = {}

            # Sample keys from each database
            for _ in self.keyspace_metrics.databases.keys():
                # Use SCAN to sample keys
                cursor = 0
                sampled_keys = 0
                key_patterns = {}

                while cursor != 0 or sampled_keys == 0:
                    if sampled_keys >= self.keyspace_sample_size:
                        break

                    # Scan with limited count to avoid blocking
                    cursor, keys = await self.client.scan(cursor, count=100)
                    
                    for key in keys[:min(50, len(keys))]:  # Limit processing per batch
                        if sampled_keys >= self.keyspace_sample_size:
                            break
                        
                        try:
                            # Get memory usage for this key
                            memory_usage = await self.client.memory_usage(key)
                            if memory_usage is not None:
                                # Extract pattern from key name
                                pattern = self._extract_key_pattern(key)
                                
                                # Track pattern distribution
                                key_patterns[pattern] = key_patterns.get(pattern, 0) + 1
                                
                                # Track memory usage by pattern
                                if pattern not in temp_memory_samples:
                                    temp_memory_samples[pattern] = []
                                temp_memory_samples[pattern].append(memory_usage)
                                
                                # Track large keys
                                if memory_usage > 1024 * 1024:  # >1MB
                                    self.keyspace_metrics.large_keys.append((key.decode() if isinstance(key, bytes) else key, memory_usage))
                                
                                sampled_keys += 1
                        
                        except Exception as key_error:
                            # Skip problematic keys
                            logger.debug(f"Error analyzing key {key}: {key_error}")
                            continue
                    
                    if cursor == 0:
                        break
                
                # Update key distribution
                self.keyspace_metrics.key_distribution.update(key_patterns)
            
            # Sort large keys by size and limit
            self.keyspace_metrics.large_keys.sort(key=lambda x: x[1], reverse=True)
            self.keyspace_metrics.large_keys = self.keyspace_metrics.large_keys[:self.max_large_keys_to_track]
            
            # Average memory usage by pattern
            for pattern, memory_list in temp_memory_samples.items():
                self.keyspace_metrics.sample_key_memory[pattern] = statistics.mean(memory_list)
                
        except Exception as e:
            logger.error(f"Failed to analyze key patterns: {e}")
    
    def _extract_key_pattern(self, key: Union[str, bytes]) -> str:
        """Extract pattern from Redis key."""
        # Convert to string safely
        key_str = _safe_str(key)
        if not key_str:
            return "unknown"

        # Common patterns to identify
        patterns = [
            (r'^session:\w+$', 'session:*'),
            (r'^user:\d+:', 'user:id:*'),
            (r'^cache:\w+:', 'cache:*'),
            (r'^queue:\w+$', 'queue:*'),
            (r'^lock:\w+$', 'lock:*'),
            (r'^\w+:\d+$', 'prefix:id'),
            (r'^\w+:\w+:\w+$', 'prefix:key:value'),
        ]

        for pattern_regex, pattern_name in patterns:
            if re.match(pattern_regex, key_str):
                return pattern_name

        # Default pattern based on prefix
        if ':' in key_str:
            return key_str.split(':', 1)[0] + ':*'
        else:
            return "simple_key"
    
    async def _collect_slowlog_metrics(self):
        """Collect and analyze Redis slow log."""
        try:
            if not self.client:
                return

            # Get slow log entries
            slowlog_entries = await self.client.slowlog_get(self.slowlog_entries_to_analyze)

            # Get slow log configuration
            slowlog_len = await self.client.slowlog_len()

            try:
                config = await self.client.config_get(["slowlog-max-len"])
                slowlog_max_len = _safe_int(config.get("slowlog-max-len", 128)) if config else 128
            except:
                slowlog_max_len = 128
            
            # Parse slow log entries
            self.slowlog_metrics.entries.clear()
            for entry in slowlog_entries:
                slow_entry = SlowLogEntry(
                    id=_safe_int(entry[0]),
                    timestamp=_safe_int(entry[1]),
                    duration_microseconds=_safe_int(entry[2]),
                    command=[_safe_str(cmd) for cmd in entry[3]] if len(entry) > 3 and entry[3] else [],
                    client_ip=_safe_str(entry[4]) if len(entry) > 4 else "",
                    client_name=_safe_str(entry[5]) if len(entry) > 5 else ""
                )
                self.slowlog_metrics.entries.append(slow_entry)
            
            self.slowlog_metrics.slowlog_len = slowlog_len
            self.slowlog_metrics.slowlog_max_len = slowlog_max_len
            
            # Analyze entries
            self.slowlog_metrics.analyze_entries()
            
        except Exception as e:
            logger.error(f"Failed to collect slow log metrics: {e}")
    
    def _calculate_overall_status(self) -> RedisHealthStatus:
        """Calculate overall health status from all metrics."""
        statuses = [
            self.memory_metrics.get_status(),
            self.performance_metrics.get_status(),
            self.persistence_metrics.get_status(),
            self.replication_metrics.get_status(),
            self.connection_metrics.get_status(),
            self.keyspace_metrics.get_status(),
            self.slowlog_metrics.get_status()
        ]
        
        # Overall status is the worst individual status
        if RedisHealthStatus.FAILED in statuses:
            return RedisHealthStatus.FAILED
        elif RedisHealthStatus.CRITICAL in statuses:
            return RedisHealthStatus.CRITICAL
        elif RedisHealthStatus.WARNING in statuses:
            return RedisHealthStatus.WARNING
        else:
            return RedisHealthStatus.HEALTHY
    
    def _memory_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert memory metrics to dictionary."""
        return {
            "status": self.memory_metrics.get_status().value,
            "used_memory_mb": round(self.memory_metrics.used_memory / 1024 / 1024, 2),
            "used_memory_human": self.memory_metrics.used_memory_human,
            "used_memory_rss_mb": round(self.memory_metrics.used_memory_rss / 1024 / 1024, 2),
            "peak_memory_mb": round(self.memory_metrics.used_memory_peak / 1024 / 1024, 2),
            "peak_memory_human": self.memory_metrics.used_memory_peak_human,
            "fragmentation_ratio": round(self.memory_metrics.mem_fragmentation_ratio, 2),
            "fragmentation_bytes": self.memory_metrics.fragmentation_bytes,
            "overhead_mb": round(self.memory_metrics.used_memory_overhead / 1024 / 1024, 2),
            "dataset_mb": round(self.memory_metrics.used_memory_dataset / 1024 / 1024, 2),
            "maxmemory_mb": round(self.memory_metrics.maxmemory / 1024 / 1024, 2) if self.memory_metrics.maxmemory > 0 else None,
            "maxmemory_human": self.memory_metrics.maxmemory_human,
            "maxmemory_policy": self.memory_metrics.maxmemory_policy,
            "memory_usage_percentage": round(self.memory_metrics.memory_usage_percentage, 2),
            "system_memory_mb": round(self.memory_metrics.total_system_memory / 1024 / 1024, 2) if self.memory_metrics.total_system_memory > 0 else None
        }
    
    def _performance_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            "status": self.performance_metrics.get_status().value,
            "ops_per_sec": round(self.performance_metrics.ops_per_sec, 2),
            "instantaneous_ops_per_sec": self.performance_metrics.instantaneous_ops_per_sec,
            "total_commands": self.performance_metrics.total_commands_processed,
            "keyspace_hits": self.performance_metrics.keyspace_hits,
            "keyspace_misses": self.performance_metrics.keyspace_misses,
            "hit_rate_percentage": round(self.performance_metrics.hit_rate, 2),
            "expired_keys": self.performance_metrics.expired_keys,
            "evicted_keys": self.performance_metrics.evicted_keys,
            "rejected_connections": self.performance_metrics.rejected_connections,
            "latency": {
                "avg_ms": round(self.performance_metrics.avg_latency_ms, 2),
                "p50_ms": round(self.performance_metrics.p50_latency_ms, 2),
                "p95_ms": round(self.performance_metrics.p95_latency_ms, 2),
                "p99_ms": round(self.performance_metrics.p99_latency_ms, 2),
                "max_ms": round(self.performance_metrics.max_latency_ms, 2),
                "samples": len(self.performance_metrics.latency_samples)
            }
        }
    
    def _persistence_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert persistence metrics to dictionary."""
        return {
            "status": self.persistence_metrics.get_status().value,
            "rdb": {
                "changes_since_last_save": self.persistence_metrics.rdb_changes_since_last_save,
                "bgsave_in_progress": self.persistence_metrics.rdb_bgsave_in_progress,
                "last_save_time": self.persistence_metrics.rdb_last_save_time,
                "last_save_age_minutes": round(self.persistence_metrics.last_save_age_minutes, 2),
                "last_bgsave_status": self.persistence_metrics.rdb_last_bgsave_status,
                "last_bgsave_duration_sec": self.persistence_metrics.rdb_last_bgsave_time_sec,
                "current_bgsave_duration_sec": self.persistence_metrics.rdb_current_bgsave_time_sec
            },
            "aof": {
                "enabled": self.persistence_metrics.aof_enabled,
                "rewrite_in_progress": self.persistence_metrics.aof_rewrite_in_progress,
                "rewrite_scheduled": self.persistence_metrics.aof_rewrite_scheduled,
                "last_rewrite_duration_sec": self.persistence_metrics.aof_last_rewrite_time_sec,
                "current_rewrite_duration_sec": self.persistence_metrics.aof_current_rewrite_time_sec,
                "last_bgrewrite_status": self.persistence_metrics.aof_last_bgrewrite_status,
                "current_size_mb": round(self.persistence_metrics.aof_current_size / 1024 / 1024, 2),
                "base_size_mb": round(self.persistence_metrics.aof_base_size / 1024 / 1024, 2),
                "pending_rewrite": self.persistence_metrics.aof_pending_rewrite,
                "buffer_length": self.persistence_metrics.aof_buffer_length,
                "rewrite_buffer_length": self.persistence_metrics.aof_rewrite_buffer_length,
                "pending_bio_fsync": self.persistence_metrics.aof_pending_bio_fsync,
                "delayed_fsync": self.persistence_metrics.aof_delayed_fsync
            } if self.persistence_metrics.aof_enabled else {"enabled": False}
        }
    
    def _replication_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert replication metrics to dictionary."""
        result = {
            "status": self.replication_metrics.get_status().value,
            "role": self.replication_metrics.role.value,
            "master_replid": self.replication_metrics.master_replid,
            "master_repl_offset": self.replication_metrics.master_repl_offset,
            "backlog_active": self.replication_metrics.repl_backlog_active,
            "backlog_size": self.replication_metrics.repl_backlog_size,
            "backlog_histlen": self.replication_metrics.repl_backlog_histlen
        }
        
        if self.replication_metrics.role == RedisRole.MASTER:
            result["master"] = {
                "connected_slaves": self.replication_metrics.connected_slaves,
                "slaves": self.replication_metrics.slave_info
            }
        elif self.replication_metrics.role == RedisRole.SLAVE:
            result["slave"] = {
                "master_host": self.replication_metrics.master_host,
                "master_port": self.replication_metrics.master_port,
                "master_link_status": self.replication_metrics.master_link_status,
                "master_last_io_seconds_ago": self.replication_metrics.master_last_io_seconds_ago,
                "master_sync_in_progress": self.replication_metrics.master_sync_in_progress,
                "slave_repl_offset": self.replication_metrics.slave_repl_offset,
                "slave_priority": self.replication_metrics.slave_priority,
                "slave_read_only": self.replication_metrics.slave_read_only,
                "replication_lag_bytes": self.replication_metrics.replication_lag_bytes,
                "replication_lag_seconds": round(self.replication_metrics.replication_lag_seconds, 2)
            }
        
        return result
    
    def _connection_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert connection metrics to dictionary."""
        return {
            "status": self.connection_metrics.get_status().value,
            "connected_clients": self.connection_metrics.connected_clients,
            "blocked_clients": self.connection_metrics.blocked_clients,
            "tracking_clients": self.connection_metrics.tracking_clients,
            "clients_in_timeout_table": self.connection_metrics.clients_in_timeout_table,
            "total_connections_received": self.connection_metrics.total_connections_received,
            "rejected_connections": self.connection_metrics.rejected_connections,
            "maxclients": self.connection_metrics.maxclients,
            "connection_utilization_percentage": round(self.connection_metrics.connection_utilization, 2),
            "recent_max_input_buffer": self.connection_metrics.client_recent_max_input_buffer,
            "recent_max_output_buffer": self.connection_metrics.client_recent_max_output_buffer
        }
    
    def _keyspace_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert keyspace metrics to dictionary."""
        databases = {}
        for db_id, db_info in self.keyspace_metrics.databases.items():
            databases[f"db{db_id}"] = {
                "keys": db_info.keys,
                "expires": db_info.expires,
                "avg_ttl": db_info.avg_ttl,
                "expiry_ratio_percentage": round(db_info.get_expiry_ratio(), 2),
                "memory_usage_mb": round(db_info.memory_usage_mb, 2)
            }
        
        return {
            "status": self.keyspace_metrics.get_status().value,
            "total_keys": self.keyspace_metrics.total_keys,
            "databases": databases,
            "key_patterns": {
                pattern: {
                    "count": count,
                    "avg_memory_bytes": round(self.keyspace_metrics.sample_key_memory.get(pattern, 0))
                }
                for pattern, count in list(self.keyspace_metrics.key_distribution.items())[:20]  # Top 20 patterns
            },
            "large_keys": [
                {
                    "key": key_name,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "size_bytes": size
                }
                for key_name, size in self.keyspace_metrics.large_keys[:10]  # Top 10 large keys
            ]
        }
    
    def _slowlog_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert slow log metrics to dictionary."""
        return {
            "status": self.slowlog_metrics.get_status().value,
            "total_entries": self.slowlog_metrics.slowlog_len,
            "max_entries": self.slowlog_metrics.slowlog_max_len,
            "recent_slow_commands": self.slowlog_metrics.recent_slow_commands,
            "avg_duration_ms": round(self.slowlog_metrics.avg_duration_ms, 2),
            "max_duration_ms": round(self.slowlog_metrics.max_duration_ms, 2),
            "command_frequency": dict(list(self.slowlog_metrics.command_frequency.items())[:10]),  # Top 10 commands
            "recent_entries": [
                {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "duration_ms": round(entry.duration_ms, 2),
                    "command": entry.command_str,
                    "client_ip": entry.client_ip,
                    "client_name": entry.client_name
                }
                for entry in self.slowlog_metrics.entries[:5]  # Most recent 5 entries
            ]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance and configuration recommendations."""
        recommendations = []
        
        # Memory recommendations
        if self.memory_metrics.mem_fragmentation_ratio > 2.0:
            recommendations.append(
                f"High memory fragmentation ({self.memory_metrics.mem_fragmentation_ratio:.2f}). "
                "Consider using MEMORY PURGE or restarting Redis during low traffic."
            )
        
        if self.memory_metrics.memory_usage_percentage > 80.0:
            recommendations.append(
                f"High memory usage ({self.memory_metrics.memory_usage_percentage:.1f}%). "
                "Consider increasing maxmemory or implementing key expiration policies."
            )
        
        # Performance recommendations
        if self.performance_metrics.hit_rate < 80.0:
            recommendations.append(
                f"Low cache hit rate ({self.performance_metrics.hit_rate:.1f}%). "
                "Review caching strategy and key TTL settings."
            )
        
        if self.performance_metrics.p95_latency_ms > 100.0:
            recommendations.append(
                f"High P95 latency ({self.performance_metrics.p95_latency_ms:.1f}ms). "
                "Check for slow commands and consider optimizing queries."
            )
        
        # Connection recommendations
        if self.connection_metrics.connection_utilization > 80.0:
            recommendations.append(
                f"High connection utilization ({self.connection_metrics.connection_utilization:.1f}%). "
                "Consider increasing maxclients or implementing connection pooling."
            )
        
        if self.connection_metrics.blocked_clients > 0:
            recommendations.append(
                f"{self.connection_metrics.blocked_clients} blocked clients detected. "
                "Review blocking operations like BLPOP, BRPOP, or BZPOPMIN."
            )
        
        # Persistence recommendations
        if self.persistence_metrics.last_save_age_minutes > 60.0:
            recommendations.append(
                f"Last RDB save was {self.persistence_metrics.last_save_age_minutes:.1f} minutes ago. "
                "Consider more frequent BGSAVE or enabling AOF for better durability."
            )
        
        if not self.persistence_metrics.aof_enabled and self.persistence_metrics.rdb_changes_since_last_save > 1000:
            recommendations.append(
                "AOF is disabled with many unsaved changes. Consider enabling AOF for better durability."
            )
        
        # Replication recommendations
        if (self.replication_metrics.role == RedisRole.SLAVE and 
            self.replication_metrics.replication_lag_seconds > 10.0):
            recommendations.append(
                f"High replication lag ({self.replication_metrics.replication_lag_seconds:.1f}s). "
                "Check network connectivity and master load."
            )
        
        # Keyspace recommendations
        for key_name, size in self.keyspace_metrics.large_keys[:3]:  # Top 3 large keys
            if size > 10 * 1024 * 1024:  # >10MB
                recommendations.append(
                    f"Large key detected: '{key_name}' ({size / 1024 / 1024:.1f}MB). "
                    "Consider breaking into smaller keys or using different data structures."
                )
        
        # Slow log recommendations
        if self.slowlog_metrics.recent_slow_commands > 5:
            recommendations.append(
                f"{self.slowlog_metrics.recent_slow_commands} slow commands in last 5 minutes. "
                "Review slow log for optimization opportunities."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Redis is performing well. No immediate optimizations needed.")
        
        return recommendations

# Convenience function for health endpoint integration
async def get_redis_health_summary() -> Dict[str, Any]:
    """
    Get a summary of Redis health for integration with main health endpoint.
    
    Returns:
        Simplified health summary suitable for main health checks
    """
    monitor = RedisHealthMonitor()
    
    try:
        # Ensure client is available
        if not await monitor._ensure_client():
            return {
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Redis client not available",
                "ping_ms": None,
                "memory_mb": None,
                "hit_rate": None
            }

        # Type assertion for client availability
        assert monitor.client is not None, "Client should be available after _ensure_client"

        # Quick connectivity test
        start_time = time.time()
        await monitor.client.ping()
        ping_time_ms = (time.time() - start_time) * 1000

        # Get basic info
        info = await monitor.client.info()

        # Calculate basic metrics
        memory_usage_mb = _safe_int(info.get("used_memory", 0)) / 1024 / 1024
        hit_rate = 0.0
        total_ops = _safe_int(info.get("keyspace_hits", 0)) + _safe_int(info.get("keyspace_misses", 0))
        if total_ops > 0:
            hit_rate = (_safe_int(info.get("keyspace_hits", 0)) / total_ops) * 100.0
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if ping_time_ms > 100:
            status = "warning"
            issues.append(f"High ping latency: {ping_time_ms:.1f}ms")
        
        if hit_rate < 80.0 and total_ops > 100:
            status = "warning" 
            issues.append(f"Low hit rate: {hit_rate:.1f}%")
        
        fragmentation_ratio = _safe_float(info.get("mem_fragmentation_ratio", 1.0))
        if fragmentation_ratio > 2.0:
            status = "warning"
            issues.append(f"High fragmentation: {fragmentation_ratio:.2f}")

        return {
            "healthy": status == "healthy",
            "status": status,
            "response_time_ms": round(ping_time_ms, 2),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "hit_rate_percentage": round(hit_rate, 2),
            "connected_clients": _safe_int(info.get("connected_clients", 0)),
            "total_commands": info.get("total_commands_processed", 0),
            "fragmentation_ratio": round(fragmentation_ratio, 2),
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Redis health summary failed: {e}")
        return {
            "healthy": False,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Health checker integration
class RedisHealthChecker:
    """Redis health checker for integration with existing health system."""
    
    def __init__(self):
        self.name = "redis"
        self.monitor = RedisHealthMonitor()
    
    async def check(self) -> Dict[str, Any]:
        """Perform Redis health check compatible with existing health system."""
        return await get_redis_health_summary()

# For backward compatibility with existing health system
async def create_redis_health_checker():
    """Create Redis health checker instance."""
    return RedisHealthChecker()