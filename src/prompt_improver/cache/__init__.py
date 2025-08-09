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
from prompt_improver.cache.redis_health import ConnectionMetrics, KeyspaceMetrics, MemoryMetrics, PerformanceMetrics, PersistenceMetrics, RedisHealthChecker, RedisHealthMonitor, RedisHealthStatus, RedisRole, ReplicationMetrics, SlowLogMetrics, create_redis_health_checker, get_redis_health_summary
__all__ = ['RedisHealthMonitor', 'RedisHealthChecker', 'get_redis_health_summary', 'create_redis_health_checker', 'RedisHealthStatus', 'RedisRole', 'MemoryMetrics', 'PerformanceMetrics', 'PersistenceMetrics', 'ReplicationMetrics', 'ConnectionMetrics', 'KeyspaceMetrics', 'SlowLogMetrics']
