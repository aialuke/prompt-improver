# Cache Monitoring Operations Runbook

## Overview

This runbook provides comprehensive operational guidance for the unified cache monitoring system that monitors all 34 consolidated cache implementations now integrated into DatabaseServices.

## System Architecture

### Monitoring Components

1. **UnifiedCacheMonitor**: Core monitoring with OpenTelemetry integration
2. **CacheSLOIntegration**: SLO compliance and predictive alerting  
3. **CrossLevelCoordinator**: L1/L2/L3 coordination and optimization
4. **Background Task Integration**: Automated monitoring and optimization

### Cache Levels

- **L1 Cache**: In-memory LRU cache (fastest, smallest capacity)
- **L2 Cache**: Redis-based cache (moderate speed, larger capacity)
- **L3 Cache**: Database-level cache (slowest, largest capacity)

## Operational Procedures

### 1. System Initialization

#### Starting Cache Monitoring

```python
from prompt_improver.monitoring.cache import initialize_comprehensive_cache_monitoring
from prompt_improver.database import get_unified_manager, ManagerMode

# Get unified manager instance
unified_manager = get_database_services(ManagerMode.ASYNC_MODERN)

# Initialize comprehensive monitoring
await initialize_comprehensive_cache_monitoring(unified_manager)
```

#### Verification Steps

1. Check monitoring system health:
```python
from prompt_improver.monitoring.cache import get_cache_monitoring_report

report = await get_cache_monitoring_report()
print(f"System Health: {report['monitoring_system_health']}")
```

2. Verify OpenTelemetry integration:
```bash
# Check for cache metrics in your observability platform
# Look for these metric families:
# - unified_cache_operations_total
# - unified_cache_hit_ratio  
# - unified_cache_operation_duration_seconds
# - slo_cache_performance_impact
# - cache_level_promotions_total  
```

### 2. Performance Monitoring

#### Key Performance Indicators (KPIs)

| Metric | Target | Critical Threshold | Description |
|--------|--------|--------------------|-------------|
| Cache Hit Rate | >85% | <70% | Overall cache effectiveness |
| P95 Latency | <50ms | >200ms | Cache operation response time |
| Availability | 99.9% | <99% | Cache system availability |
| Error Rate | <0.1% | >1% | Cache operation error rate |
| Coordination Efficiency | >90% | <70% | Cross-level coordination success |

#### Daily Monitoring Checklist

- [ ] Check overall cache hit rate across all levels
- [ ] Review P95 latency trends for degradation
- [ ] Verify SLO compliance ratios
- [ ] Check for coherence violations
- [ ] Review predictive alerts
- [ ] Validate background task health

#### Performance Dashboard Queries

**Cache Hit Rate by Level:**
```promql
unified_cache_hit_ratio{level=~"l1|l2|l3"}
```

**Cache Operation Latency P95:**
```promql
histogram_quantile(0.95, unified_cache_operation_duration_seconds_bucket)
```

**SLO Compliance Status:**
```promql
slo_compliance_ratio{service_name="unified_cache"}
```

### 3. Alert Management

#### Alert Severities

**INFO**: Performance degradation within acceptable bounds
**WARNING**: Performance approaching critical thresholds
**CRITICAL**: Performance violation requiring immediate attention
**EMERGENCY**: System unavailability or severe degradation

#### Alert Response Procedures

##### Cache Hit Rate Degradation

**Symptoms:**
- Cache hit rate drops below 70%
- Increased backend load
- Higher response times

**Immediate Actions:**
1. Check cache warming effectiveness
2. Review TTL configurations
3. Analyze access patterns for optimization opportunities
4. Verify cache capacity limits

**Investigation:**
```python
# Get detailed cache statistics
cache_monitor = get_unified_cache_monitor()
stats = cache_monitor.get_comprehensive_stats()

# Check warming patterns
warming_stats = stats['enhanced_monitoring']['predictive_warming']
print(f"Warming effectiveness: {warming_stats['warming_effectiveness']}")

# Review access patterns
pattern_dist = stats['enhanced_monitoring']['access_patterns']['distribution']
print(f"Access pattern distribution: {pattern_dist}")
```

##### High Cache Latency

**Symptoms:**
- P95 latency >200ms
- User-facing performance degradation
- Increased timeout errors

**Immediate Actions:**
1. Check cache backend health (Redis, memory)
2. Review serialization/deserialization performance
3. Analyze cache data sizes
4. Check network connectivity

**Investigation:**
```python
# Check performance metrics by level
perf_metrics = stats['enhanced_monitoring']['performance_metrics']
for level, metrics in perf_metrics.items():
    print(f"{level}: P95={metrics['p95_duration_ms']}ms")

# Check cross-level coordination impact
coordinator = get_cross_level_coordinator()
coord_stats = coordinator.get_coordination_stats()
avg_coord_time = coord_stats['performance']['avg_coordination_time_ms']
print(f"Avg coordination time: {avg_coord_time}ms")
```

##### Coherence Violations

**Symptoms:**
- Cache coherence violations detected
- Inconsistent data across cache levels
- Data integrity alerts

**Immediate Actions:**
1. Identify affected keys
2. Repair coherence violations
3. Review invalidation patterns
4. Check for race conditions

**Investigation:**
```python
# Check coherence violations
coordinator = get_cross_level_coordinator()
violations = await coordinator.check_cache_coherence()

if violations:
    print(f"Coherence violations detected: {len(violations)}")
    for key, issues in violations.items():
        print(f"Key {key}: {issues}")
        # Attempt repair
        await coordinator.repair_cache_coherence(key)
```

### 4. Predictive Alerting

#### Understanding Predictive Alerts

The system generates predictive alerts based on performance trends:

- **Trend Analysis**: Linear regression on historical performance data
- **Confidence Levels**: LOW, MEDIUM, HIGH, VERY_HIGH
- **Prediction Window**: 30 minutes ahead
- **Alert Types**: predicted_hit_rate_violation, predicted_latency_violation

#### Responding to Predictive Alerts

```python
# Get current predictive alerts
slo_integration = get_cache_slo_integration()
alerts = await slo_integration.generate_predictive_alerts()

for alert in alerts:
    print(f"Alert: {alert.alert_type}")
    print(f"Predicted violation time: {alert.predicted_violation_time}")
    print(f"Confidence: {alert.confidence.value}")
    print(f"Recommendations: {alert.recommended_actions}")
```

**Proactive Actions:**
1. Implement recommended optimizations
2. Adjust cache warming schedules
3. Review resource allocation
4. Prepare escalation procedures

### 5. Cache Invalidation Management

#### Coordinated Invalidation

The system supports multiple invalidation strategies:

- **Pattern-based**: Invalidate keys matching patterns
- **Dependency-based**: Cascade invalidation through dependencies
- **Manual**: Direct key invalidation
- **TTL-based**: Automatic expiration

#### Managing Dependencies

```python
# Add cache dependencies
cache_monitor = get_unified_cache_monitor()
cache_monitor.add_dependency("user:123:profile", "user:123", cascade_level=2)

# Invalidate with cascade
await cache_monitor.invalidate_by_dependency("user:123")
```

#### Pattern Invalidation

```python
# Invalidate all user cache entries
await cache_monitor.invalidate_by_pattern("user:*")

# Invalidate specific patterns
await cache_monitor.invalidate_by_pattern("ml:model:*")
```

### 6. Cross-Level Coordination

#### Understanding Cache Promotion/Demotion

**Promotion Triggers:**
- High access frequency (>10/min)
- Recent access patterns (WARM/HOT)
- High promotion score (>0.7)

**Demotion Triggers:**
- Low access frequency (<2/hour) 
- Cache capacity pressure
- Low promotion score (<0.3)

**Monitoring Coordination:**
```python
coordinator = get_cross_level_coordinator()
stats = coordinator.get_coordination_stats()

print(f"L2→L1 Promotions: {stats['promotions']['l2_to_l1']}")
print(f"L1→L2 Demotions: {stats['demotions']['l1_to_l2']}")
print(f"Coordination Efficiency: {stats['coordination_events']['efficiency']}")
```

### 7. Troubleshooting Common Issues

#### Issue: High Memory Usage in L1 Cache

**Symptoms:**
- Memory pressure alerts
- L1 cache evictions
- Performance degradation

**Resolution:**
1. Check L1 cache size configuration
2. Review promotion criteria
3. Implement more aggressive demotion
4. Analyze entry size distribution

```python
# Check L1 usage
stats = coordinator.get_coordination_stats()
l1_usage = stats['cache_levels']['usage']['l1']
print(f"L1 Usage: {l1_usage['count']} entries, {l1_usage['size_bytes']} bytes")
```

#### Issue: Redis Connection Failures

**Symptoms:**
- L2 cache unavailable
- Increased L1 load
- Redis connectivity errors

**Resolution:**
1. Check Redis server health
2. Verify network connectivity
3. Review connection pool settings
4. Check Redis memory usage

```python
# Check L2 cache health
if unified_manager.cache.redis_client:
    try:
        await unified_manager.cache.redis_client.ping()
        print("Redis connection healthy")
    except Exception as e:
        print(f"Redis connection failed: {e}")
```

#### Issue: Cache Warming Ineffective

**Symptoms:**
- Low warming success rate
- Poor hit rates despite warming
- Wasted warming resources

**Resolution:**
1. Review warming patterns
2. Adjust warming schedules
3. Analyze access pattern changes
4. Optimize warming prioritization

```python
# Check warming effectiveness
warming_candidates = await cache_monitor.analyze_warming_opportunities()
print(f"Warming candidates: {len(warming_candidates)}")

# Review warming patterns
patterns = cache_monitor._warming_patterns
for key, pattern in patterns.items():
    print(f"Key: {key}, Success Rate: {pattern.success_rate}")
```

### 8. Maintenance Procedures

#### Weekly Maintenance

1. **Performance Review:**
   - Analyze weekly performance trends
   - Review SLO compliance history
   - Check for recurring patterns

2. **Capacity Planning:**
   - Review cache size trends
   - Analyze growth patterns
   - Plan capacity adjustments

3. **Configuration Optimization:**
   - Review TTL settings
   - Adjust promotion/demotion thresholds
   - Update warming strategies

#### Monthly Maintenance

1. **System Health Audit:**
   - Comprehensive performance analysis
   - Review alert patterns
   - Assess monitoring coverage

2. **Documentation Updates:**
   - Update operational procedures
   - Review alert response times
   - Update escalation procedures

### 9. Integration with External Systems

#### Grafana Dashboards

**Cache Performance Dashboard:**
```json
{
  "dashboard": {
    "title": "Unified Cache Performance",
    "panels": [
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "unified_cache_hit_ratio",
            "legendFormat": "{{level}}"
          }
        ]
      },
      {
        "title": "Cache Operation Latency",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, unified_cache_operation_duration_seconds_bucket)",
            "legendFormat": "P95 {{operation}}"
          }
        ]
      }
    ]
  }
}
```

#### PagerDuty Integration

```python
# Configure alert callbacks for PagerDuty integration
async def pagerduty_alert_callback(alert):
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
        # Send to PagerDuty
        await send_pagerduty_alert({
            "incident_key": alert.alert_id,
            "description": alert.message,
            "severity": alert.severity.value
        })

cache_monitor.register_alert_callback(pagerduty_alert_callback)
```

### 10. Emergency Procedures

#### Cache System Failure

**Immediate Response:**
1. Check system status and error rates
2. Verify fallback mechanisms
3. Assess user impact
4. Implement emergency bypasses if needed

**Recovery Steps:**
1. Identify root cause
2. Restart affected components
3. Verify system health
4. Restore monitoring
5. Conduct post-incident review

#### Data Corruption Detection

**Immediate Response:**
1. Stop cache writes
2. Identify scope of corruption
3. Implement cache bypass
4. Begin data recovery

**Recovery Steps:**
1. Clear corrupted cache entries
2. Restore from clean backup
3. Verify data integrity
4. Resume normal operations
5. Implement preventive measures

## Monitoring Metrics Reference

### Core Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `unified_cache_operations_total` | Counter | Total cache operations | operation, level, status |
| `unified_cache_hit_ratio` | Gauge | Cache hit ratio by level | level |
| `unified_cache_operation_duration_seconds` | Histogram | Operation latency | operation, level |
| `slo_cache_performance_impact` | Gauge | SLO performance impact | service |
| `cache_level_promotions_total` | Counter | Cross-level promotions | source_level, target_level |
| `cache_coherence_violations_total` | Counter | Coherence violations | violation_type |
| `predictive_cache_alerts_total` | Counter | Predictive alerts | alert_type, confidence |

### SLI Metrics

| SLI | Target | Measurement |
|-----|--------|-------------|
| Availability | 99.9% | Successful operations / Total operations |
| Latency | <50ms P95 | 95th percentile operation duration |
| Hit Rate | >85% | Cache hits / Total requests |
| Error Rate | <0.1% | Failed operations / Total operations |

## Contact Information

**On-Call Team**: cache-monitoring-oncall@company.com  
**Escalation**: sre-team@company.com  
**Documentation**: https://docs.company.com/cache-monitoring  
**Runbook Updates**: https://github.com/company/cache-monitoring-runbooks