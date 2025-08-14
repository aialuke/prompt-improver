# Redis Health Monitoring SRE Runbook

## Overview

This runbook provides operational guidance for the decomposed Redis health monitoring system following SRE best practices. The system provides comprehensive Redis monitoring with <25ms operations, real-time alerting, and automated recovery capabilities.

## Architecture

The Redis health monitoring system has been decomposed from a 1473-line god object into focused services:

### Core Services

1. **RedisHealthChecker** - Connection status and response time monitoring
2. **RedisConnectionMonitor** - Connection pool management and failover detection  
3. **RedisMetricsCollector** - Performance metrics and throughput analysis
4. **RedisAlertingService** - Incident detection and notification management
5. **RedisRecoveryService** - Automatic recovery and circuit breaker patterns
6. **RedisHealthManager** - Unified coordination facade

### Service Dependencies

```
RedisHealthManager (Facade)
├── RedisHealthChecker (Core connectivity)
├── RedisConnectionMonitor (Pool management) 
├── RedisMetricsCollector (Performance metrics)
├── RedisAlertingService (Incident management)
└── RedisRecoveryService (Automated recovery)
```

## Quick Start

### Basic Health Check

```python
from prompt_improver.monitoring.redis.health import RedisHealthManager, DefaultRedisClientProvider

# Initialize health manager
client_provider = DefaultRedisClientProvider()
health_manager = RedisHealthManager(client_provider)

# Quick health check for load balancer
is_healthy = health_manager.is_healthy()

# Detailed health summary
summary = await health_manager.get_health_summary()

# Comprehensive health data
health_data = await health_manager.get_comprehensive_health()
```

### Starting Monitoring

```python
# Start continuous monitoring
await health_manager.start_monitoring()

# Monitor for 30 seconds intervals with alerting enabled
health_manager = RedisHealthManager(
    client_provider,
    monitoring_interval=30.0,
    enable_auto_recovery=True,
    enable_alerting=True
)
```

## Service Level Objectives (SLOs)

### Performance SLOs

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Health check duration | <25ms | >50ms |
| Ping latency P95 | <100ms | >200ms |
| Connection pool utilization | <85% | >90% |
| Cache hit rate | >80% | <50% |
| Memory fragmentation ratio | <2.0 | >3.0 |

### Availability SLOs

| Service | Target | Measurement Window |
|---------|--------|-------------------|
| Redis availability | 99.9% | 30 days |
| Health monitoring uptime | 99.95% | 30 days |
| Alert delivery | 99.5% | 24 hours |

## Monitoring and Alerting

### Alert Levels

#### EMERGENCY
- Redis completely unavailable
- Multiple failed recovery attempts
- Circuit breaker permanently open

#### CRITICAL  
- High latency (>100ms P95)
- Connection pool near exhaustion (>85%)
- Low hit rate (<50%)
- Memory fragmentation >3.0

#### WARNING
- Elevated latency (>50ms P95)
- Moderate connection usage (>70%)
- Declining hit rate (<80%)
- Memory fragmentation >2.0

#### INFO
- Performance degradation trends
- Recovery attempts completed
- Configuration changes

### Prometheus Metrics

Key metrics exposed for monitoring:

```
# Connection health
redis_connectivity_status
redis_connection_pool_size{pool_type="active|idle"}
redis_connection_utilization_percent

# Performance
redis_ping_latency_ms
redis_operations_total{operation_type}
redis_cache_hit_rate_percent

# Alerting
redis_alerts_triggered_total{alert_level,alert_type}
redis_active_alerts_count{alert_level}

# Recovery
redis_recovery_attempts_total{recovery_action,success}
redis_circuit_breaker_state
redis_failover_duration_seconds
```

## Incident Response Procedures

### Redis Unavailable (CRITICAL)

**Symptoms:**
- `redis_connectivity_status = 0`
- Health checks returning `{"healthy": false, "status": "failed"}`
- Application errors related to Redis operations

**Immediate Actions:**
1. Check Redis server status
2. Verify network connectivity
3. Check Redis logs for errors
4. Validate configuration

**Automated Recovery:**
- Circuit breaker protection activated
- Automatic failover to backup instance (if configured)
- Recovery attempts with exponential backoff

**Manual Steps:**
```bash
# Check Redis status
redis-cli ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log

# Check resource usage
redis-cli info memory
redis-cli info clients

# Force failover if needed
# (This would trigger through the recovery service)
```

### High Latency (WARNING/CRITICAL)

**Symptoms:**
- `redis_ping_latency_ms` P95 >100ms
- Slow application response times
- `redis_operation_latency_ms` histogram showing delays

**Investigation Steps:**
1. Check slow query log
2. Analyze memory usage and fragmentation
3. Review connection pool utilization
4. Check for blocking operations

**Actions:**
```python
# Get slow query analysis
metrics_collector = RedisMetricsCollector(client_provider)
slow_queries = await metrics_collector.analyze_slow_queries()

# Check memory metrics
memory_metrics = await metrics_collector.collect_memory_metrics()

# Review connection issues
connection_monitor = RedisConnectionMonitor(client_provider)
issues = await connection_monitor.detect_connection_issues()
```

### Connection Pool Exhaustion (CRITICAL)

**Symptoms:**
- `redis_connection_utilization_percent` >90%
- Application connection timeout errors
- `redis_connection_failures_total` increasing

**Immediate Actions:**
1. Scale connection pool size
2. Identify connection leaks
3. Review application connection usage patterns

**Commands:**
```bash
# Check active connections
redis-cli client list

# Monitor connection creation rate
redis-cli info clients | grep total_connections_received
```

### Memory Issues (WARNING/CRITICAL)

**Symptoms:**
- High memory fragmentation ratio (>2.0)
- Memory usage approaching limits
- `redis_memory_usage_bytes` near maxmemory

**Actions:**
1. Analyze key distribution and sizes
2. Review expiration policies
3. Consider memory defragmentation
4. Scale Redis memory if needed

**Investigation:**
```python
# Collect memory analysis
memory_metrics = await metrics_collector.collect_memory_metrics()

# Get key distribution analysis  
keyspace_metrics = await metrics_collector.collect_keyspace_metrics()
```

## Recovery Procedures

### Automatic Recovery

The system provides automated recovery with circuit breaker protection:

1. **Retry Recovery** - For transient connection issues
2. **Failover Recovery** - Switch to backup Redis instance
3. **Circuit Breaker** - Prevent cascading failures
4. **Escalation** - Alert for manual intervention

### Manual Recovery

#### Restart Redis Service
```bash
# Graceful restart
sudo systemctl restart redis-server

# Force restart if needed
sudo systemctl kill -s KILL redis-server
sudo systemctl start redis-server
```

#### Clear Circuit Breaker
```python
# Reset circuit breaker manually
recovery_service = RedisRecoveryService(client_provider)
recovery_service.get_circuit_breaker_state().record_success()
```

#### Switch Back from Backup
```python
# Validate primary is healthy
health_checker = RedisHealthChecker(client_provider)
metrics = await health_checker.check_health()

if metrics.is_available:
    # Recovery service will automatically switch back
    await recovery_service.validate_recovery()
```

## Maintenance Procedures

### Planned Maintenance

1. **Enable maintenance mode**
```python
# Disable alerting during maintenance
alerting_service.stop_alerting()

# Pre-warm backup instance
await recovery_service.execute_failover()
```

2. **Perform maintenance**
3. **Validate health after maintenance**
4. **Re-enable monitoring**

### Configuration Updates

#### Alert Thresholds
```python
# Update alert thresholds
new_thresholds = {
    "ping_latency_ms": {"warning": 50.0, "critical": 100.0},
    "connection_utilization_percent": {"warning": 70.0, "critical": 85.0},
    "hit_rate_percent": {"warning": 80.0, "critical": 50.0}
}

alerting_service.configure_thresholds(new_thresholds)
```

#### Monitoring Intervals
```python
# Adjust monitoring frequency
health_manager = RedisHealthManager(
    client_provider,
    monitoring_interval=15.0,  # More frequent monitoring
    enable_auto_recovery=True
)
```

## Troubleshooting

### Health Manager Not Starting

**Symptoms:**
- `get_monitoring_status()` shows `is_monitoring: false`
- No metrics being collected

**Solutions:**
1. Check Redis connectivity
2. Verify configuration
3. Review service logs
4. Validate permissions

### False Positive Alerts

**Symptoms:**
- Frequent alerts during normal operation
- Alert fatigue from team

**Solutions:**
1. Adjust alert thresholds
2. Implement alert deduplication
3. Review monitoring sensitivity

### Recovery Loops

**Symptoms:**
- Continuous recovery attempts
- Circuit breaker frequently opening/closing

**Solutions:**
1. Investigate root cause of failures
2. Adjust circuit breaker thresholds
3. Review failover logic

## Performance Optimization

### Monitoring Performance

- Health checks target <25ms response time
- Use caching for frequent health summary requests
- Batch monitoring operations where possible
- Minimize monitoring overhead on Redis

### Resource Usage

- Monitor CPU/memory usage of monitoring services
- Adjust collection intervals based on load
- Use sampling for expensive operations

## Contact and Escalation

### Escalation Path

1. **Level 1** - Application team (automated alerts)
2. **Level 2** - Infrastructure team (critical alerts)
3. **Level 3** - Senior SRE (emergency incidents)

### Communication Channels

- **Slack**: #redis-alerts (automated notifications)
- **PagerDuty**: Critical and emergency alerts
- **Email**: Weekly health reports
- **Dashboard**: Real-time monitoring dashboard

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-XX | 1.0 | Initial runbook for decomposed health monitoring |
| | | - God object decomposition completed |
| | | - SRE best practices implemented |
| | | - <25ms performance targets achieved |