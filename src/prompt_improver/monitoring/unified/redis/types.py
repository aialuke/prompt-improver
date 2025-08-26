"""Shared types and data classes for Redis health monitoring.

This module contains all data structures used across Redis monitoring components,
following the clean architecture principle of shared types between components.
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RedisHealthStatus(Enum):
    """Redis health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class RedisRole(Enum):
    """Redis server roles."""

    MASTER = "master"
    SLAVE = "slave"
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
        if self.maxmemory > 0 and self.memory_usage_percentage > 85.0:
            return False
        return not self.mem_fragmentation_ratio > 3.0

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on memory metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
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
    latency_samples: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def calculate_hit_rate(self):
        """Calculate cache hit rate."""
        total = self.keyspace_hits + self.keyspace_misses
        self.hit_rate = self.keyspace_hits / total * 100 if total > 0 else 0.0

    def add_latency_sample(self, latency_ms: float):
        """Add a latency sample and update percentiles."""
        self.latency_samples.append(latency_ms)
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
        if self.hit_rate < 50.0:
            return False
        return not self.p95_latency_ms > 500.0

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on performance metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        if self.hit_rate < 80.0:
            return RedisHealthStatus.WARNING
        if self.p95_latency_ms > 200.0:
            return RedisHealthStatus.WARNING
        return RedisHealthStatus.HEALTHY


@dataclass
class PersistenceMetrics:
    """Redis persistence health metrics."""

    rdb_changes_since_last_save: int = 0
    rdb_bgsave_in_progress: bool = False
    rdb_last_save_time: int = 0
    rdb_last_bgsave_status: str = "ok"
    rdb_last_bgsave_time_sec: int = 0
    rdb_current_bgsave_time_sec: int = 0
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
    last_save_age_minutes: float = 0.0

    def update_last_save_age(self):
        """Update the age of last save operation."""
        if self.rdb_last_save_time > 0:
            current_time = time.time()
            self.last_save_age_minutes = (current_time - self.rdb_last_save_time) / 60.0

    def is_healthy(self) -> bool:
        """Check if persistence metrics indicate healthy state."""
        self.update_last_save_age()
        if (
            self.last_save_age_minutes > 60.0
            and self.rdb_changes_since_last_save > 1000
        ):
            return False
        return not (self.rdb_last_bgsave_status != "ok" or (self.aof_enabled and self.aof_last_bgrewrite_status != "ok"))

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on persistence metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        if self.last_save_age_minutes > 30.0 and self.rdb_changes_since_last_save > 100:
            return RedisHealthStatus.WARNING
        if (
            self.rdb_current_bgsave_time_sec > 300
            or self.aof_current_rewrite_time_sec > 300
        ):
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
    slave_info: list[dict[str, Any]] = field(default_factory=list)
    master_host: str = ""
    master_port: int = 0
    master_link_status: str = "up"
    master_last_io_seconds_ago: int = 0
    master_sync_in_progress: bool = False
    slave_repl_offset: int = 0
    slave_priority: int = 100
    slave_read_only: bool = True
    replication_lag_bytes: int = 0
    replication_lag_seconds: float = 0.0

    def calculate_replication_lag(self):
        """Calculate replication lag metrics."""
        if self.role == RedisRole.SLAVE and self.master_repl_offset > 0:
            self.replication_lag_bytes = max(
                0, self.master_repl_offset - self.slave_repl_offset
            )
            if self.master_last_io_seconds_ago >= 0:
                self.replication_lag_seconds = float(self.master_last_io_seconds_ago)

    def is_healthy(self) -> bool:
        """Check if replication metrics indicate healthy state."""
        if self.role == RedisRole.SLAVE:
            if self.master_link_status != "up":
                return False
            if self.replication_lag_seconds > 30.0:
                return False
            if self.master_last_io_seconds_ago > 60:
                return False
        return True

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on replication metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        if self.role == RedisRole.SLAVE:
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
    connection_utilization: float = 0.0
    connections_per_second: float = 0.0

    def calculate_utilization(self):
        """Calculate connection pool utilization."""
        if self.maxclients > 0:
            self.connection_utilization = (
                self.connected_clients / self.maxclients * 100.0
            )

    def is_healthy(self) -> bool:
        """Check if connection metrics indicate healthy state."""
        self.calculate_utilization()
        if self.connection_utilization > 90.0:
            return False
        if self.blocked_clients > self.connected_clients * 0.5:
            return False
        return not self.rejected_connections > 0

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on connection metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
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
        return self.expires / self.keys * 100.0 if self.keys > 0 else 0.0


@dataclass
class KeyspaceMetrics:
    """Redis keyspace analytics."""

    databases: dict[int, KeyspaceInfo] = field(default_factory=dict)
    total_keys: int = 0
    sample_key_memory: dict[str, float] = field(default_factory=dict)
    large_keys: list[tuple[str, int]] = field(default_factory=list)
    key_distribution: dict[str, int] = field(default_factory=dict)

    def calculate_totals(self):
        """Calculate total metrics from all databases."""
        self.total_keys = sum(db.keys for db in self.databases.values())

    def is_healthy(self) -> bool:
        """Check if keyspace metrics indicate healthy state."""
        return all(size <= 100 * 1024 * 1024 for _, size in self.large_keys)

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on keyspace metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
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
    command: list[str]
    client_ip: str = ""
    client_name: str = ""

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration_microseconds / 1000.0

    @property
    def command_str(self) -> str:
        """Get command as string."""
        return " ".join(str(cmd) for cmd in self.command[:10])


@dataclass
class SlowLogMetrics:
    """Redis slow log analysis."""

    entries: list[SlowLogEntry] = field(default_factory=list)
    slowlog_len: int = 0
    slowlog_max_len: int = 128
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    command_frequency: dict[str, int] = field(default_factory=dict)
    recent_slow_commands: int = 0

    def analyze_entries(self):
        """Analyze slow log entries."""
        if not self.entries:
            return
        durations = [entry.duration_ms for entry in self.entries]
        self.avg_duration_ms = statistics.mean(durations)
        self.max_duration_ms = max(durations)
        self.command_frequency.clear()
        for entry in self.entries:
            cmd = entry.command[0].upper() if entry.command else "UNKNOWN"
            self.command_frequency[cmd] = self.command_frequency.get(cmd, 0) + 1
        current_time = time.time()
        five_minutes_ago = current_time - 300
        self.recent_slow_commands = sum(
            1 for entry in self.entries if entry.timestamp >= five_minutes_ago
        )

    def is_healthy(self) -> bool:
        """Check if slow log metrics indicate healthy state."""
        if self.recent_slow_commands > 10:
            return False
        return not self.max_duration_ms > 1000.0

    def get_status(self) -> RedisHealthStatus:
        """Get health status based on slow log metrics."""
        if not self.is_healthy():
            return RedisHealthStatus.CRITICAL
        if self.recent_slow_commands > 2:
            return RedisHealthStatus.WARNING
        if self.max_duration_ms > 100.0:
            return RedisHealthStatus.WARNING
        return RedisHealthStatus.HEALTHY


@dataclass
class RedisHealthSummary:
    """Comprehensive Redis health summary."""

    overall_status: RedisHealthStatus
    memory_status: RedisHealthStatus
    performance_status: RedisHealthStatus
    persistence_status: RedisHealthStatus
    replication_status: RedisHealthStatus
    connection_status: RedisHealthStatus
    keyspace_status: RedisHealthStatus
    slowlog_status: RedisHealthStatus
    check_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    recommendations: list[str] = field(default_factory=list)

    def get_health_percentage(self) -> float:
        """Calculate overall health percentage."""
        statuses = [
            self.memory_status,
            self.performance_status,
            self.persistence_status,
            self.replication_status,
            self.connection_status,
            self.keyspace_status,
            self.slowlog_status,
        ]

        healthy_count = sum(1 for status in statuses if status == RedisHealthStatus.HEALTHY)
        return (healthy_count / len(statuses)) * 100.0

    def get_critical_issues(self) -> list[str]:
        """Get list of critical issues."""
        issues = []

        status_map = {
            "memory": self.memory_status,
            "performance": self.performance_status,
            "persistence": self.persistence_status,
            "replication": self.replication_status,
            "connection": self.connection_status,
            "keyspace": self.keyspace_status,
            "slowlog": self.slowlog_status,
        }

        for component, status in status_map.items():
            if status == RedisHealthStatus.CRITICAL:
                issues.append(f"{component} status is critical")
            elif status == RedisHealthStatus.FAILED:
                issues.append(f"{component} check failed")

        return issues
