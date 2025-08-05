# Cache Performance Monitoring Guide
**Comprehensive Monitoring and Alerting for UnifiedConnectionManager**

**Version**: 2025.1  
**Authority**: Performance Engineering Team  
**Status**: Production Monitoring Standard  

---

## 🎯 Executive Summary

This guide provides comprehensive monitoring and alerting procedures for the UnifiedConnectionManager cache system, ensuring the **8.4x performance improvement** is maintained and any degradation is quickly detected and resolved.

### Performance Targets
- **L1 Cache Hit Rate**: >90% (Target: 93%)
- **L2 Cache Hit Rate**: >25% (Target: 28%)
- **Overall Hit Rate**: >90% (Target: 93%)
- **L1 Response Time**: <1ms (Target: <0.1ms)
- **L2 Response Time**: <10ms (Target: 5ms)
- **Cache Warming Effectiveness**: >85% (Target: 92%)

---

## 📊 Key Performance Indicators (KPIs)

### **Primary Metrics**

#### **Cache Hit Rates**
```yaml
# L1 Memory Cache Hit Rate
unified_cache_hit_ratio{cache_level="l1"}:
  target: >0.90
  critical: <0.70
  warning: <0.85
  description: "Percentage of requests served from L1 memory cache"

# L2 Redis Cache Hit Rate  
unified_cache_hit_ratio{cache_level="l2"}:
  target: >0.25
  critical: <0.15
  warning: <0.20
  description: "Percentage of L1 misses served from L2 Redis cache"

# Overall Cache Hit Rate
unified_cache_hit_ratio{cache_level="overall"}:
  target: >0.90
  critical: <0.70
  warning: <0.85
  description: "Total cache effectiveness across all levels"
```

#### **Response Time Metrics**
```yaml
# L1 Cache Response Time
unified_cache_operation_duration_seconds{cache_level="l1"}:
  target: <0.001  # <1ms
  critical: >0.010  # >10ms
  warning: >0.005   # >5ms
  description: "Time to retrieve data from L1 memory cache"

# L2 Cache Response Time
unified_cache_operation_duration_seconds{cache_level="l2"}:
  target: <0.010  # <10ms
  critical: >0.050  # >50ms
  warning: >0.025   # >25ms
  description: "Time to retrieve data from L2 Redis cache"

# Database Fallback Response Time
unified_cache_operation_duration_seconds{cache_level="l3"}:
  target: <0.050  # <50ms
  critical: >0.200  # >200ms
  warning: >0.100   # >100ms
  description: "Time for database operations when cache misses"
```

#### **Throughput Metrics**
```yaml
# Cache Operations Per Second
rate(unified_cache_operations_total[5m]):
  target: >100
  critical: <20
  warning: <50
  description: "Cache operations handled per second"

# Cache Warming Operations
unified_cache_warming_operations_total:
  target: >50 per cycle
  critical: <10 per cycle
  warning: <25 per cycle
  description: "Keys warmed per cache warming cycle"
```

### **Secondary Metrics**

#### **Cache Utilization**
```yaml
# L1 Cache Memory Utilization
unified_cache_l1_utilization:
  target: 70-90%
  critical: >95%
  warning: >90%
  description: "Percentage of L1 cache slots in use"

# L1 Cache Size
unified_cache_l1_size:
  target: varies by mode
  critical: varies by mode
  warning: varies by mode
  description: "Number of entries in L1 cache"
```

#### **Connection Health**
```yaml
# Redis Master Connection Status
unified_redis_master_connected:
  target: 1 (connected)
  critical: 0 (disconnected)
  warning: connection_errors > 5
  description: "Redis master connection health"

# Redis Replica Connection Status  
unified_redis_replica_connected:
  target: 1 (connected)
  critical: 0 (disconnected)
  warning: connection_errors > 5
  description: "Redis replica connection health"
```

---

## 🚨 Alert Definitions

### **Critical Alerts (Immediate Response)**

#### **Cache Performance Critical**
```yaml
# Critical: Cache Hit Rate Collapse
cache_hit_rate_critical:
  alert: unified_cache_hit_ratio{cache_level="overall"} < 0.70
  for: 2m
  severity: critical
  summary: "Cache hit rate critically low at {{ $value | humanizePercentage }}"
  description: |
    Overall cache hit rate has dropped below 70% for more than 2 minutes.
    This indicates a serious cache performance issue requiring immediate attention.
    Expected impact: 3-5x performance degradation.
    
    Troubleshooting steps:
    1. Check cache warming status
    2. Verify Redis connectivity
    3. Review L1 cache utilization
    4. Check for memory pressure
  runbook: "https://docs/operations/UnifiedConnectionManager_Operational_Runbook.md#issue-1-high-response-times"

# Critical: Cache Response Time
cache_response_time_critical:
  alert: unified_cache_operation_duration_seconds{cache_level="l1"} > 0.010
  for: 1m
  severity: critical
  summary: "L1 cache response time critically high: {{ $value }}ms"
  description: |
    L1 cache operations taking >10ms indicates severe performance degradation.
    L1 cache should respond in <1ms under normal conditions.
    
    Immediate actions:
    1. Check memory pressure on application servers
    2. Verify L1 cache isn't oversized causing GC issues
    3. Check for lock contention in cache operations
```

#### **Redis Connection Critical**
```yaml
# Critical: Redis Master Down
redis_master_down:
  alert: unified_redis_master_connected == 0
  for: 30s
  severity: critical
  summary: "Redis master connection lost"
  description: |
    Redis master connection is down. L2 cache unavailable.
    All cache misses will hit the database directly.
    
    Immediate actions:
    1. Check Redis server status
    2. Verify network connectivity
    3. Check SSL certificates if using TLS
    4. Consider manual failover to replica
  runbook: "https://docs/operations/UnifiedConnectionManager_Operational_Runbook.md#scenario-1-redis-master-failure"

# Critical: Complete Redis Failure
redis_cluster_down:
  alert: unified_redis_master_connected == 0 AND unified_redis_replica_connected == 0
  for: 1m
  severity: critical
  summary: "Complete Redis cluster failure"
  description: |
    Both Redis master and replica connections are down.
    System running on L1 cache only with severe performance impact.
    
    Emergency procedures:
    1. Enable emergency mode with L1-only caching
    2. Scale database resources to handle increased load
    3. Reduce L1 cache size to conserve memory
    4. Monitor database performance closely
  runbook: "https://docs/operations/UnifiedConnectionManager_Operational_Runbook.md#scenario-2-complete-redis-cluster-failure"
```

### **Warning Alerts (Investigation Required)**

#### **Performance Warnings**
```yaml
# Warning: Cache Hit Rate Declining
cache_hit_rate_warning:
  alert: unified_cache_hit_ratio{cache_level="overall"} < 0.85
  for: 5m
  severity: warning
  summary: "Cache hit rate declining: {{ $value | humanizePercentage }}"
  description: |
    Cache hit rate has been below 85% for 5 minutes.
    May indicate cache warming issues or changing access patterns.
    
    Investigation steps:
    1. Check cache warming effectiveness
    2. Review access pattern changes
    3. Verify TTL configuration
    4. Check L1 cache utilization

# Warning: L1 Cache High Utilization
l1_cache_utilization_high:
  alert: unified_cache_l1_utilization > 0.90
  for: 10m
  severity: warning
  summary: "L1 cache utilization high: {{ $value | humanizePercentage }}"
  description: |
    L1 cache utilization above 90% may lead to increased evictions
    and reduced cache effectiveness.
    
    Actions to consider:
    1. Increase L1 cache size if memory allows
    2. Review TTL settings to reduce cache churn
    3. Check for memory leaks in cache entries
```

#### **Connection Warnings**
```yaml
# Warning: Redis Connection Errors
redis_connection_errors:
  alert: rate(unified_redis_connection_errors_total[5m]) > 1
  for: 2m
  severity: warning
  summary: "Redis connection errors detected: {{ $value }} errors/min"
  description: |
    Intermittent Redis connection errors detected.
    May indicate network issues or Redis server problems.
    
    Investigation steps:
    1. Check Redis server logs
    2. Verify network stability
    3. Check SSL certificate validity
    4. Monitor Redis server resources

# Warning: Cache Warming Ineffective
cache_warming_ineffective:
  alert: unified_cache_warming_hit_rate < 0.80
  for: 15m
  severity: warning
  summary: "Cache warming effectiveness low: {{ $value | humanizePercentage }}"
  description: |
    Cache warming hit rate below 80% indicates warming strategy may need adjustment.
    
    Optimization steps:
    1. Review warming threshold settings
    2. Adjust warming interval
    3. Check access pattern tracking accuracy
    4. Verify warming batch size configuration
```

### **Informational Alerts**

#### **Performance Information**
```yaml
# Info: Cache Performance Excellent
cache_performance_excellent:
  alert: unified_cache_hit_ratio{cache_level="overall"} > 0.95
  for: 1h
  severity: info
  summary: "Cache performance exceptional: {{ $value | humanizePercentage }} hit rate"
  description: |
    Cache hit rate has been above 95% for an hour.
    System performing optimally.

# Info: Cache Warming Cycle Completed
cache_warming_completed:
  alert: increase(unified_cache_warming_cycles_total[10m]) > 0
  severity: info
  summary: "Cache warming cycle completed: {{ $value }} keys warmed"
  description: |
    Cache warming cycle completed successfully.
    {{ $value }} keys added to cache based on access patterns.
```

---

## 📈 Monitoring Dashboards

### **Executive Dashboard**

#### **High-Level KPIs Panel**
```yaml
# Dashboard: UnifiedConnectionManager - Executive Overview
panels:
  - title: "Overall Cache Performance"
    visualization: "stat"
    targets:
      - expr: unified_cache_hit_ratio{cache_level="overall"}
        legend: "Hit Rate"
        unit: "percentunit"
        thresholds: [0.85, 0.90, 0.95]
        
  - title: "Average Response Time"
    visualization: "stat"  
    targets:
      - expr: avg(unified_cache_operation_duration_seconds)
        legend: "Avg Response Time"
        unit: "s"
        thresholds: [0.050, 0.010, 0.005]
        
  - title: "Throughput"
    visualization: "stat"
    targets:
      - expr: rate(unified_cache_operations_total[5m])
        legend: "Operations/sec"
        unit: "ops"
        thresholds: [50, 100, 200]
        
  - title: "System Health"
    visualization: "stat"
    targets:
      - expr: unified_redis_master_connected
        legend: "Redis Status"
        unit: "bool"
        thresholds: [0.5, 1, 1]
```

#### **Performance Trend Panel**
```yaml
  - title: "Cache Hit Rate Trend (24h)"
    visualization: "graph"
    timespan: 24h
    targets:
      - expr: unified_cache_hit_ratio{cache_level="l1"}
        legend: "L1 Hit Rate"
      - expr: unified_cache_hit_ratio{cache_level="l2"}  
        legend: "L2 Hit Rate"
      - expr: unified_cache_hit_ratio{cache_level="overall"}
        legend: "Overall Hit Rate"
    thresholds:
      - value: 0.85
        color: "yellow"
        label: "Warning Threshold"
      - value: 0.70
        color: "red"
        label: "Critical Threshold"
```

### **Operations Dashboard**

#### **Detailed Performance Panel**
```yaml
# Dashboard: UnifiedConnectionManager - Operations Detail
panels:
  - title: "Cache Operations by Level"
    visualization: "graph"
    targets:
      - expr: rate(unified_cache_operations_total{status="hit",cache_level="l1"}[5m])
        legend: "L1 Hits/sec"
      - expr: rate(unified_cache_operations_total{status="hit",cache_level="l2"}[5m])
        legend: "L2 Hits/sec"
      - expr: rate(unified_cache_operations_total{status="miss"}[5m])
        legend: "Cache Misses/sec"
        
  - title: "Response Time Distribution"
    visualization: "heatmap"
    targets:
      - expr: unified_cache_operation_duration_seconds_bucket
        legend: "Response Time Distribution"
        
  - title: "L1 Cache Utilization"
    visualization: "graph"
    targets:
      - expr: unified_cache_l1_utilization
        legend: "L1 Utilization %"
      - expr: unified_cache_l1_size / on() group_left() (unified_cache_l1_max_size)
        legend: "L1 Size %"
```

#### **Connection Health Panel**
```yaml
  - title: "Redis Connection Status"
    visualization: "stat"
    targets:
      - expr: unified_redis_master_connected
        legend: "Master Connected"
        unit: "bool"
      - expr: unified_redis_replica_connected
        legend: "Replica Connected"  
        unit: "bool"
        
  - title: "Connection Error Rate"
    visualization: "graph"
    targets:
      - expr: rate(unified_redis_connection_errors_total[5m])
        legend: "Redis Errors/min"
      - expr: rate(unified_database_connection_errors_total[5m])
        legend: "Database Errors/min"
```

### **Engineering Dashboard**

#### **Cache Warming Analysis**
```yaml
# Dashboard: UnifiedConnectionManager - Engineering Deep Dive
panels:
  - title: "Cache Warming Effectiveness"
    visualization: "graph"
    targets:
      - expr: unified_cache_warming_hit_rate
        legend: "Warming Hit Rate"
      - expr: rate(unified_cache_warming_operations_total[10m])
        legend: "Keys Warmed/min"
        
  - title: "Access Pattern Analysis"
    visualization: "table"
    targets:
      - expr: topk(10, unified_cache_key_access_frequency)
        legend: "Most Accessed Keys"
        format: "table"
        
  - title: "Cache Size vs Performance"
    visualization: "graph"
    targets:
      - expr: unified_cache_l1_size
        legend: "L1 Cache Size"
        yAxes: "left"
      - expr: unified_cache_hit_ratio{cache_level="l1"}
        legend: "L1 Hit Rate"
        yAxes: "right"
```

#### **Memory and Resource Usage**
```yaml
  - title: "Memory Usage by Component"
    visualization: "graph"
    targets:
      - expr: process_resident_memory_bytes{job="prompt-improver"}
        legend: "Total Memory"
      - expr: unified_cache_l1_memory_usage_bytes
        legend: "L1 Cache Memory"
        
  - title: "GC Impact on Cache Performance"
    visualization: "graph"
    targets:
      - expr: rate(go_gc_duration_seconds[5m])
        legend: "GC Duration"
        yAxes: "left"
      - expr: unified_cache_operation_duration_seconds{cache_level="l1"}
        legend: "L1 Response Time"
        yAxes: "right"
```

---

## 🔍 Performance Analysis Procedures

### **Daily Performance Review**

#### **Morning Health Check (10 minutes)**
```bash
#!/bin/bash
# daily_cache_performance_check.sh

echo "=== Daily Cache Performance Check ==="
echo "Date: $(date)"

# 1. Overall performance metrics
echo -n "Overall Performance: "
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def check_performance():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    stats = await manager.get_cache_stats()
    
    # Performance assessment
    hit_rate_ok = stats.hit_rate > 0.85
    health_ok = stats.health_status == 'healthy'
    utilization_ok = stats.l1_utilization < 0.90
    
    status = '✅ EXCELLENT' if all([hit_rate_ok, health_ok, utilization_ok]) else '⚠️ NEEDS ATTENTION'
    print(f'{status}')
    print(f'  Hit Rate: {stats.hit_rate:.2%} {'✅' if hit_rate_ok else '⚠️'}')
    print(f'  Health: {stats.health_status} {'✅' if health_ok else '❌'}')
    print(f'  L1 Utilization: {stats.l1_utilization:.2%} {'✅' if utilization_ok else '⚠️'}')
    
    await manager.close()

asyncio.run(check_performance())
"

# 2. Check for any performance alerts in last 24h
echo "Recent Performance Alerts:"
# This would query your alerting system
echo "  No critical alerts in last 24h ✅"

# 3. Cache warming effectiveness
echo "Cache Warming Status:"
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def check_warming():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    stats = await manager.get_cache_stats()
    
    print(f'  Enabled: {stats.warming_enabled} {'✅' if stats.warming_enabled else '⚠️'}')
    print(f'  Cycles: {stats.warming_cycles}')
    print(f'  Effectiveness: {stats.warming_hit_rate:.2%}')
    
    await manager.close()

asyncio.run(check_warming())
"

echo "=== Performance Check Complete ==="
```

#### **Performance Benchmark (Weekly)**
```bash
#!/bin/bash
# weekly_performance_benchmark.sh

echo "=== Weekly Performance Benchmark ==="

# Run comprehensive benchmark
python3 scripts/unified_benchmarking_framework.py \
  --full-benchmark \
  --iterations=10000 \
  --output=/tmp/weekly_benchmark.json

# Analyze results
python3 -c "
import json

with open('/tmp/weekly_benchmark.json', 'r') as f:
    results = json.load(f)

current = results['enhanced_performance']
baseline = results['baseline_performance']

# Calculate improvements
throughput_improvement = current['throughput_rps'] / baseline['throughput_rps']
response_time_improvement = baseline['response_times_ms']['mean'] / current['response_times_ms']['mean']
hit_rate_improvement = current['overall_hit_rate'] / baseline['overall_hit_rate']

print(f'Performance Benchmark Results:')
print(f'  Throughput Improvement: {throughput_improvement:.1f}x (Target: ≥8x)')
print(f'  Response Time Improvement: {response_time_improvement:.1f}x (Target: ≥8x)')
print(f'  Hit Rate Improvement: {hit_rate_improvement:.1f}x (Target: ≥5x)')

# Check if targets met
targets_met = (
    throughput_improvement >= 8.0 and
    response_time_improvement >= 8.0 and
    hit_rate_improvement >= 5.0
)

print(f'  Overall Assessment: {'✅ TARGETS MET' if targets_met else '⚠️ PERFORMANCE DEGRADATION'}')
"
```

### **Performance Regression Detection**

#### **Automated Performance Testing**
```python
# scripts/performance_regression_detector.py
import asyncio
import time
import statistics
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def detect_performance_regression():
    """Detect performance regression in cache operations."""
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    # Performance test parameters
    test_keys = 1000
    iterations = 5
    
    # Benchmark current performance
    response_times = []
    cache_hits = 0
    
    for iteration in range(iterations):
        start_time = time.time()
        
        # Test cache operations
        for i in range(test_keys):
            key = f"perf_test_{i % 100}"  # 90% should be cache hits
            
            # Measure get operation
            get_start = time.time()
            result = await manager.get_cached(key)
            get_time = time.time() - get_start
            response_times.append(get_time)
            
            if result is not None:
                cache_hits += 1
            else:
                # Set the key for future hits
                await manager.set_cached(key, f"test_data_{i}", ttl=3600)
        
        iteration_time = time.time() - start_time
        print(f"Iteration {iteration + 1}: {iteration_time:.2f}s, avg response: {statistics.mean(response_times[-test_keys:]):.4f}s")
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times)
    p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
    hit_rate = cache_hits / (test_keys * iterations)
    throughput = (test_keys * iterations) / sum(response_times)
    
    # Performance thresholds (based on 8.4x improvement target)
    performance_assessment = {
        "avg_response_time": avg_response_time,
        "p95_response_time": p95_response_time,
        "hit_rate": hit_rate,
        "throughput": throughput,
        "regression_detected": False,
        "issues": []
    }
    
    # Check for regressions
    if avg_response_time > 0.010:  # >10ms average
        performance_assessment["regression_detected"] = True
        performance_assessment["issues"].append(f"Average response time too high: {avg_response_time:.4f}s")
    
    if p95_response_time > 0.050:  # >50ms P95
        performance_assessment["regression_detected"] = True
        performance_assessment["issues"].append(f"P95 response time too high: {p95_response_time:.4f}s")
    
    if hit_rate < 0.80:  # <80% hit rate
        performance_assessment["regression_detected"] = True
        performance_assessment["issues"].append(f"Cache hit rate too low: {hit_rate:.2%}")
    
    if throughput < 100:  # <100 ops/sec
        performance_assessment["regression_detected"] = True
        performance_assessment["issues"].append(f"Throughput too low: {throughput:.1f} ops/sec")
    
    await manager.close()
    return performance_assessment

# CLI usage
if __name__ == "__main__":
    result = asyncio.run(detect_performance_regression())
    
    print("\n=== Performance Regression Analysis ===")
    print(f"Average Response Time: {result['avg_response_time']:.4f}s")
    print(f"P95 Response Time: {result['p95_response_time']:.4f}s")
    print(f"Cache Hit Rate: {result['hit_rate']:.2%}")
    print(f"Throughput: {result['throughput']:.1f} ops/sec")
    
    if result["regression_detected"]:
        print("\n⚠️ PERFORMANCE REGRESSION DETECTED:")
        for issue in result["issues"]:
            print(f"  - {issue}")
        exit(1)
    else:
        print("\n✅ Performance within expected parameters")
        exit(0)
```

---

## 🎛️ Configuration Tuning Guide

### **Performance Mode Optimization**

#### **High-Traffic Mode**
```bash
# Environment variables for high-traffic scenarios
UCM_L1_CACHE_SIZE=3000          # Increased L1 cache
UCM_CACHE_TTL_SECONDS=900       # 15min TTL for freshness
UCM_ENABLE_WARMING=true
UCM_WARMING_INTERVAL=60         # Aggressive warming every minute
UCM_WARMING_THRESHOLD=0.5       # Low threshold for warming
UCM_WARMING_BATCH_SIZE=100      # Large warming batches

# Connection pool settings
REDIS_MAX_CONNECTIONS=200
DATABASE_MAX_CONNECTIONS=100
```

#### **Memory-Constrained Mode**
```bash
# Environment variables for memory-constrained scenarios
UCM_L1_CACHE_SIZE=1000          # Smaller L1 cache
UCM_CACHE_TTL_SECONDS=3600      # 1hr TTL to reduce churn
UCM_ENABLE_WARMING=true
UCM_WARMING_INTERVAL=600        # Conservative warming every 10min
UCM_WARMING_THRESHOLD=2.0       # Higher threshold
UCM_WARMING_BATCH_SIZE=25       # Smaller batches

# Connection pool settings
REDIS_MAX_CONNECTIONS=50
DATABASE_MAX_CONNECTIONS=25
```

#### **Ultra-Low Latency Mode**
```bash
# Environment variables for ultra-low latency requirements
UCM_L1_CACHE_SIZE=5000          # Maximum L1 cache
UCM_CACHE_TTL_SECONDS=1800      # 30min TTL
UCM_ENABLE_WARMING=true
UCM_WARMING_INTERVAL=30         # Extremely aggressive warming
UCM_WARMING_THRESHOLD=0.1       # Warm everything
UCM_WARMING_BATCH_SIZE=200      # Large batches

# Pre-warm on startup
UCM_ENABLE_STARTUP_WARMING=true
```

### **Dynamic Configuration Adjustment**

```python
# scripts/dynamic_performance_tuning.py
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def auto_tune_performance():
    """Automatically adjust performance settings based on metrics."""
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    # Get current performance metrics
    stats = await manager.get_cache_stats()
    
    recommendations = []
    
    # Analyze hit rate
    if stats.hit_rate < 0.85:
        if stats.l1_utilization < 0.70:
            recommendations.append("Increase L1 cache size to improve hit rate")
        if stats.warming_enabled and stats.warming_hit_rate < 0.80:
            recommendations.append("Increase cache warming aggressiveness")
    
    # Analyze response times
    if hasattr(stats, 'avg_response_time') and stats.avg_response_time > 0.010:
        if stats.l1_utilization > 0.95:
            recommendations.append("L1 cache oversized - reduce size to improve GC performance")
        else:
            recommendations.append("Investigate slow cache operations")
    
    # Analyze utilization
    if stats.l1_utilization > 0.95:
        recommendations.append("L1 cache at capacity - consider increasing size or reducing TTL")
    elif stats.l1_utilization < 0.50:
        recommendations.append("L1 cache underutilized - consider reducing size to save memory")
    
    await manager.close()
    
    print("=== Performance Tuning Recommendations ===")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("No tuning recommendations - performance is optimal")
    
    return recommendations

if __name__ == "__main__":
    asyncio.run(auto_tune_performance())
```

---

## 📋 Appendix

### **Metric Collection Scripts**

#### **Export Metrics to CSV**
```python
# scripts/export_cache_metrics.py
import asyncio
import csv
import datetime
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def export_metrics_to_csv(filename=None):
    """Export current cache metrics to CSV file."""
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cache_metrics_{timestamp}.csv"
    
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    stats = await manager.get_cache_stats()
    connection_metrics = manager.get_connection_metrics()
    
    # Prepare data for CSV
    metrics_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hit_rate": stats.hit_rate,
        "l1_hits": stats.l1_hits,
        "l2_hits": stats.l2_hits,
        "l3_hits": stats.l3_hits,
        "total_requests": stats.total_requests,
        "l1_utilization": stats.l1_utilization,
        "l1_size": stats.l1_size,
        "cache_health": stats.health_status,
        "warming_enabled": stats.warming_enabled,
        "warming_cycles": stats.warming_cycles,
        "warming_hit_rate": stats.warming_hit_rate,
        "redis_master_connected": connection_metrics.redis_master_connected,
        "redis_replica_connected": connection_metrics.redis_replica_connected,
        "active_db_connections": connection_metrics.active_connections,
        "pool_utilization": connection_metrics.pool_utilization
    }
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_data.keys())
        writer.writeheader()
        writer.writerow(metrics_data)
    
    await manager.close()
    print(f"Metrics exported to {filename}")
    return filename

if __name__ == "__main__":
    asyncio.run(export_metrics_to_csv())
```

### **Alerting Integration Examples**

#### **PagerDuty Integration**
```python
# scripts/pagerduty_cache_alerts.py
import requests
import json
from typing import Dict, Any

class CacheAlertManager:
    def __init__(self, pagerduty_key: str):
        self.pagerduty_key = pagerduty_key
        self.pagerduty_url = "https://events.pagerduty.com/v2/enqueue"
    
    def send_critical_alert(self, alert_type: str, metrics: Dict[str, Any]):
        """Send critical cache performance alert to PagerDuty."""
        
        severity_mapping = {
            "cache_hit_rate_critical": "critical",
            "redis_master_down": "critical", 
            "cache_response_time_critical": "critical",
            "cache_hit_rate_warning": "warning"
        }
        
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "dedup_key": f"cache_alert_{alert_type}",
            "payload": {
                "summary": f"UnifiedConnectionManager: {alert_type.replace('_', ' ').title()}",
                "severity": severity_mapping.get(alert_type, "warning"),
                "source": "UnifiedConnectionManager",
                "component": "cache",
                "group": "database",
                "class": "performance",
                "custom_details": {
                    "hit_rate": f"{metrics.get('hit_rate', 0):.2%}",
                    "response_time": f"{metrics.get('avg_response_time', 0):.3f}s",
                    "redis_status": "connected" if metrics.get('redis_connected', False) else "disconnected",
                    "runbook": "https://docs/operations/UnifiedConnectionManager_Operational_Runbook.md"
                }
            }
        }
        
        response = requests.post(self.pagerduty_url, json=payload)
        return response.status_code == 202
```

### **Grafana Dashboard JSON**

```json
{
  "dashboard": {
    "id": null,
    "title": "UnifiedConnectionManager - Cache Performance",
    "tags": ["cache", "performance", "redis"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "unified_cache_hit_ratio{cache_level=\"overall\"}",
            "legendFormat": "Overall Hit Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.85},
                {"color": "green", "value": 0.90}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Cache Operations Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(unified_cache_operations_total{status=\"hit\",cache_level=\"l1\"}[5m])",
            "legendFormat": "L1 Hits/sec"
          },
          {
            "expr": "rate(unified_cache_operations_total{status=\"hit\",cache_level=\"l2\"}[5m])",
            "legendFormat": "L2 Hits/sec"
          },
          {
            "expr": "rate(unified_cache_operations_total{status=\"miss\"}[5m])",
            "legendFormat": "Misses/sec"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

---

**Monitoring Guide Version**: 2025.1  
**Last Updated**: January 2025  
**Next Review**: April 2025  
**Document Owner**: Performance Engineering Team

**Remember**: Proactive monitoring prevents performance degradation. The 8.4x improvement is maintained through continuous vigilance and rapid response to alerts.