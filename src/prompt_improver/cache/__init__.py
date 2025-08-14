"""Cache Module

This module provides comprehensive caching and Redis health monitoring capabilities.

Available Components:
- RedisHealthMonitor: Comprehensive Redis health monitoring with detailed metrics
- get_redis_health_summary: Quick Redis health summary for integration
- RedisHealthChecker: Health checker compatible with existing health system

Features:
- Memory usage tracking and fragmentation analysis
- Performance metrics with latency percentiles
- Persistence health monitoring (RDB/AOF status)
- Replication status and lag monitoring
- Client connection analysis and utilization tracking
- Keyspace analytics with memory profiling
- Slow log analysis for performance optimization
- Real-time metrics collection with circuit breaker protection
"""

# Import from new focused services
from prompt_improver.monitoring.redis.health import (
    RedisHealthManager,
    RedisHealthChecker as NewRedisHealthChecker,
    RedisHealthStatus,
    RedisRole,
    HealthMetrics,
    ConnectionPoolMetrics as ConnectionMetrics,
    PerformanceMetrics,
)

# Legacy compatibility removed - clean break modernization

# Legacy metric types (simplified for compatibility)
MemoryMetrics = HealthMetrics  # Backward compatibility
PersistenceMetrics = dict  # Legacy - use dict for compatibility
ReplicationMetrics = dict  # Legacy - use dict for compatibility
KeyspaceMetrics = dict  # Legacy - use dict for compatibility
SlowLogMetrics = dict  # Legacy - use dict for compatibility

__all__ = [
    # New focused services (recommended)
    "RedisHealthManager",
    "NewRedisHealthChecker",
    "RedisHealthStatus",
    "RedisRole",
    "HealthMetrics",
    
    # Compatibility metric types
    "ConnectionMetrics",
    "PerformanceMetrics",
    "MemoryMetrics",
    "KeyspaceMetrics",
    "PersistenceMetrics",
    "ReplicationMetrics", 
    "SlowLogMetrics",
]
