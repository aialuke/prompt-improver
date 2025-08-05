# EnhancedBackgroundTaskManager Operational Runbook

**Document Version**: 1.0  
**Last Updated**: 2025-08-01  
**Scope**: Day-to-day operations, monitoring, and troubleshooting  
**Audience**: DevOps, SRE, Development Teams  

## Executive Summary

This runbook provides comprehensive operational procedures for the EnhancedBackgroundTaskManager, the unified async infrastructure established by ADR-007. All background services in the prompt-improver system use this centralized task management system for enhanced reliability, observability, and operational simplicity.

## Table of Contents

1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring & Health Checks](#monitoring--health-checks)
4. [Common Issues & Resolution](#common-issues--resolution)
5. [Emergency Procedures](#emergency-procedures)
6. [Performance Tuning](#performance-tuning)
7. [Maintenance Tasks](#maintenance-tasks)
8. [Integration with Monitoring](#integration-with-monitoring)
9. [Troubleshooting Reference](#troubleshooting-reference)

---

## System Overview

### Architecture
- **Single Source of Truth**: All background services use EnhancedBackgroundTaskManager
- **Priority-Based Scheduling**: CRITICAL → HIGH → NORMAL → LOW → BACKGROUND
- **Enhanced Reliability**: Built-in retry mechanisms, circuit breakers, dead letter queues
- **Comprehensive Observability**: Task-level metrics, logging, and health monitoring

### Key Components
- **Task Manager**: `src/prompt_improver/performance/monitoring/health/background_manager.py`
- **Global Instance**: `get_background_task_manager()` provides singleton access
- **Task Priorities**: TaskPriority enum (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
- **Monitoring Integration**: OpenTelemetry metrics and structured logging

### Service Coverage
All background services including: ML serving, health monitoring, metrics collection, database monitoring, event processing, CLI orchestration, WebSocket management, performance testing, and utility services.

---

## Daily Operations

### Health Check Procedures

#### **Morning System Verification**
```bash
# 1. Check task manager status
python3 -c "
import asyncio
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager

async def check_status():
    tm = get_background_task_manager()
    stats = tm.get_statistics()
    print(f'Active tasks: {len(tm.running_tasks)}')
    print(f'Queue depth: {len(tm.priority_queue)}')
    print(f'Success rate: {stats.get(\"success_rate\", \"N/A\")}')
    print(f'Total tasks processed: {stats.get(\"total_processed\", 0)}')

asyncio.run(check_status())
"

# 2. Check for failed tasks
grep "Task failed" /var/log/prompt-improver/*.log | tail -20

# 3. Verify circuit breaker status
python3 -c "
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager
tm = get_background_task_manager()
for name, cb in tm.circuit_breakers.items():
    print(f'{name}: {cb.state}')
"
```

#### **Key Metrics to Monitor Daily**

| Metric | Normal Range | Investigation Threshold | Critical Threshold |
|--------|--------------|------------------------|-------------------|
| Task Success Rate | >95% | <95% | <90% |
| Average Queue Depth | <10 tasks | >20 tasks | >50 tasks |
| Task Processing Time | <30s average | >60s average | >120s average |
| Circuit Breaker Status | All closed | 1-2 half-open | Any open |
| Dead Letter Queue | <5 tasks | >10 tasks | >20 tasks |
| Memory Usage | <500MB | >800MB | >1GB |

### Routine Checks

#### **Task Health Verification**
```python
# Check task manager health
async def daily_health_check():
    tm = get_background_task_manager()
    
    # 1. Running tasks audit
    running = tm.get_running_tasks()
    print(f"Currently running: {len(running)} tasks")
    for task_id in running:
        task = tm.tasks.get(task_id)
        if task:
            duration = (datetime.now(UTC) - task.started_at).total_seconds()
            print(f"  {task_id}: {duration:.1f}s")
    
    # 2. Queue analysis
    queue_depth = len(tm.priority_queue)
    if queue_depth > 10:
        print(f"⚠️  High queue depth: {queue_depth}")
        
    # 3. Failed task review
    failed_count = len([t for t in tm.tasks.values() if t.status == TaskStatus.FAILED])
    if failed_count > 0:
        print(f"❌ Failed tasks: {failed_count}")
        
    # 4. Dead letter queue check
    dl_count = len(tm.dead_letter_queue)
    if dl_count > 0:
        print(f"💀 Dead letter queue: {dl_count} tasks")
```

---

## Monitoring & Health Checks

### Automated Health Monitoring

#### **System Health Indicators**
- **Green**: Success rate >95%, queue depth <10, all circuits closed
- **Yellow**: Success rate 90-95%, queue depth 10-20, circuits half-open
- **Red**: Success rate <90%, queue depth >20, circuits open

#### **Health Check Endpoints**
```python
# Health check integration
async def system_health_check():
    tm = get_background_task_manager()
    
    health_status = {
        "status": "healthy",
        "task_manager": {
            "running_tasks": len(tm.running_tasks),
            "queue_depth": len(tm.priority_queue),
            "dead_letter_count": len(tm.dead_letter_queue),
            "circuit_breakers": {name: cb.state for name, cb in tm.circuit_breakers.items()}
        }
    }
    
    # Determine overall health
    if len(tm.priority_queue) > 20 or len(tm.dead_letter_queue) > 10:
        health_status["status"] = "degraded"
    
    return health_status
```

### Performance Monitoring

#### **Key Performance Indicators**
```python
# Performance metrics collection
async def collect_performance_metrics():
    tm = get_background_task_manager()
    
    metrics = {
        "task_throughput": tm.get_throughput_metrics(),
        "average_processing_time": tm.get_average_processing_time(),
        "memory_usage": tm.get_memory_usage(),
        "resource_utilization": tm.get_resource_utilization()
    }
    
    return metrics
```

---

## Common Issues & Resolution

### High Task Failure Rate (>10%)

#### **Symptoms**
- Task success rate drops below 90%
- Increasing number of tasks in dead letter queue
- Circuit breakers opening frequently

#### **Investigation Steps**
```bash
# 1. Check recent task failures
python3 -c "
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager
tm = get_background_task_manager()
failed_tasks = [t for t in tm.tasks.values() if t.status.value == 'failed']
for task in failed_tasks[-10:]:  # Last 10 failures
    print(f'{task.task_id}: {task.error}')
"

# 2. Review error patterns
grep "Task execution failed" /var/log/prompt-improver/*.log | awk '{print $NF}' | sort | uniq -c | sort -nr

# 3. Check resource constraints
top -p $(pgrep -f "python.*prompt-improver")
```

#### **Resolution Actions**
1. **Resource Issues**: Scale up resources if CPU/memory constrained
2. **Dependency Failures**: Check external service health (database, Redis, APIs)
3. **Code Issues**: Review recent deployments for introduced bugs
4. **Configuration**: Verify task timeout and retry configurations

### Tasks Stuck in Queue

#### **Symptoms**
- Queue depth consistently >20 tasks
- Tasks not progressing from QUEUED to RUNNING
- Increasing task creation rate vs completion rate

#### **Investigation Steps**
```python
# Analyze queue composition
async def analyze_queue():
    tm = get_background_task_manager()
    
    # Group by priority
    priority_distribution = {}
    for task in tm.priority_queue:
        priority = task.priority.name
        priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
    
    print("Queue by priority:", priority_distribution)
    
    # Check for blocked tasks
    oldest_queued = min(tm.priority_queue, key=lambda t: t.created_at)
    wait_time = (datetime.now(UTC) - oldest_queued.created_at).total_seconds()
    print(f"Oldest queued task waiting: {wait_time:.1f}s")
```

#### **Resolution Actions**
1. **Increase Concurrency**: Adjust `max_concurrent_tasks` if resources allow
2. **Priority Rebalancing**: Review task priority assignments
3. **Blocking Operations**: Identify and fix tasks with blocking I/O
4. **Resource Allocation**: Scale horizontally or vertically

### Memory Growth Issues

#### **Symptoms**
- Steadily increasing memory usage
- Task manager process memory >1GB
- Out of memory errors in logs

#### **Investigation Steps**
```python
# Memory usage analysis
import psutil
import gc

async def memory_analysis():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # Check task cleanup
    tm = get_background_task_manager()
    print(f"Total tasks in memory: {len(tm.tasks)}")
    print(f"Completed tasks: {len([t for t in tm.tasks.values() if t.status.value == 'completed'])}")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Garbage collected: {collected} objects")
```

#### **Resolution Actions**
1. **Task Cleanup**: Implement periodic cleanup of completed tasks
2. **Memory Leaks**: Profile for memory leaks in task implementations
3. **Resource Limits**: Set task memory limits and timeouts
4. **Garbage Collection**: Tune garbage collection parameters

---

## Emergency Procedures

### Complete System Restart

#### **Graceful Shutdown Sequence**
```python
# Emergency graceful shutdown
async def emergency_shutdown():
    tm = get_background_task_manager()
    
    print("🛑 Initiating emergency shutdown...")
    
    # Stop accepting new tasks
    tm._accepting_tasks = False
    
    # Wait for current tasks to complete (up to 60 seconds)
    await tm.stop(timeout=60.0)
    
    # Force shutdown if needed
    await shutdown_background_task_manager(timeout=30.0)
    
    print("✅ Emergency shutdown complete")
```

#### **Recovery Sequence**
```python
# System recovery after shutdown
async def emergency_recovery():
    print("🔄 Starting system recovery...")
    
    # Initialize new task manager
    tm = await init_background_task_manager(max_concurrent_tasks=10)
    
    # Re-register critical services
    critical_services = [
        "health_monitoring",
        "database_monitoring", 
        "metrics_collection"
    ]
    
    for service in critical_services:
        print(f"Restarting {service}...")
        # Service-specific restart logic here
    
    print("✅ System recovery complete")
```

### Circuit Breaker Recovery

#### **Manual Circuit Breaker Reset**
```python
# Reset circuit breakers after issue resolution
async def reset_circuit_breakers():
    tm = get_background_task_manager()
    
    for name, cb in tm.circuit_breakers.items():
        if cb.state == "open":
            print(f"Resetting circuit breaker: {name}")
            cb.failure_count = 0
            cb.state = "closed"
            cb.last_failure_time = None
```

### Dead Letter Queue Processing

#### **Manual Dead Letter Queue Recovery**
```python
# Process dead letter queue manually
async def process_dead_letter_queue():
    tm = get_background_task_manager()
    
    print(f"Processing {len(tm.dead_letter_queue)} dead letter tasks...")
    
    for task in tm.dead_letter_queue.copy():
        # Manual review and re-submission
        if should_retry_task(task):
            # Reset task status and retry
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.error = None
            
            # Re-submit task
            await tm.submit_enhanced_task(
                task_id=f"{task.task_id}_recovery",
                coroutine=task.coroutine,
                priority=task.priority,
                tags=task.tags
            )
            
            tm.dead_letter_queue.remove(task)
```

---

## Performance Tuning

### Concurrency Optimization

#### **Determining Optimal Concurrency**
```python
# Performance testing for optimal concurrency
async def find_optimal_concurrency():
    import time
    
    concurrency_levels = [5, 10, 15, 20, 25]
    results = {}
    
    for level in concurrency_levels:
        print(f"Testing concurrency level: {level}")
        
        # Restart with new concurrency
        await shutdown_background_task_manager()
        tm = await init_background_task_manager(max_concurrent_tasks=level)
        
        # Submit test workload
        start_time = time.time()
        test_tasks = []
        
        for i in range(100):
            task_id = await tm.submit_enhanced_task(
                task_id=f"perf_test_{i}",
                coroutine=lambda: asyncio.sleep(1),
                priority=TaskPriority.NORMAL
            )
            test_tasks.append(task_id)
        
        # Wait for completion
        while any(tm.tasks[tid].status != TaskStatus.COMPLETED for tid in test_tasks):
            await asyncio.sleep(0.1)
        
        duration = time.time() - start_time
        results[level] = duration
        print(f"Completed in {duration:.2f}s")
    
    optimal = min(results.items(), key=lambda x: x[1])
    print(f"Optimal concurrency: {optimal[0]} ({optimal[1]:.2f}s)")
```

### Priority Tuning Guidelines

#### **Task Priority Assignment**
- **CRITICAL**: System health, security operations, emergency procedures
- **HIGH**: User-facing operations, data integrity, real-time processing
- **NORMAL**: Standard background processing, metrics collection, routine operations
- **LOW**: Cleanup operations, non-essential monitoring, optimization tasks
- **BACKGROUND**: Archival, reporting, analytics, optional optimizations

#### **Priority Distribution Monitoring**
```python
# Monitor priority distribution
async def monitor_priority_distribution():
    tm = get_background_task_manager()
    
    distribution = {}
    for task in tm.tasks.values():
        priority = task.priority.name
        distribution[priority] = distribution.get(priority, 0) + 1
    
    total = sum(distribution.values())
    for priority, count in distribution.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{priority}: {count} tasks ({percentage:.1f}%)")
```

---

## Maintenance Tasks

### Weekly Maintenance

#### **Task Cleanup & Analysis**
```bash
#!/bin/bash
# Weekly maintenance script

echo "🔧 Starting weekly maintenance..."

# 1. Clean up completed tasks older than 24 hours
python3 -c "
import asyncio
from datetime import datetime, timedelta, UTC
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager

async def cleanup():
    tm = get_background_task_manager()
    cutoff = datetime.now(UTC) - timedelta(days=1)
    
    cleaned = 0
    for task_id, task in list(tm.tasks.items()):
        if task.status.value == 'completed' and task.completed_at and task.completed_at < cutoff:
            del tm.tasks[task_id]
            cleaned += 1
    
    print(f'Cleaned up {cleaned} old completed tasks')

asyncio.run(cleanup())
"

# 2. Generate performance report
python3 -c "
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager
tm = get_background_task_manager()
stats = tm.get_statistics()
print('Weekly Performance Report:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"

# 3. Check for memory leaks
ps aux | grep python | grep prompt-improver | awk '{print "Memory usage: " $6/1024 " MB"}'

echo "✅ Weekly maintenance complete"
```

### Monthly Maintenance  

#### **Capacity Planning Analysis**
```python
# Monthly capacity analysis
async def monthly_capacity_analysis():
    import json
    from datetime import datetime, timedelta
    
    tm = get_background_task_manager()
    
    # Collect metrics over past month
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "peak_concurrent_tasks": max([len(tm.running_tasks)]),  # Would need historical tracking
        "average_queue_depth": sum([len(tm.priority_queue)]) / 30,  # Would need historical data
        "task_volume_trends": {},  # Would analyze historical task volumes
        "performance_trends": {},  # Would analyze response times
        "resource_utilization": {
            "cpu_average": 0,  # Would collect from monitoring
            "memory_average": 0,  # Would collect from monitoring
            "peak_memory": 0     # Would collect from monitoring
        },
        "recommendations": []
    }
    
    # Generate recommendations
    if analysis["peak_concurrent_tasks"] > tm.max_concurrent_tasks * 0.8:
        analysis["recommendations"].append("Consider increasing max_concurrent_tasks")
    
    if analysis["average_queue_depth"] > 15:
        analysis["recommendations"].append("High queue depth indicates need for more workers")
    
    # Save analysis
    with open(f"/tmp/capacity_analysis_{datetime.now().strftime('%Y%m')}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    return analysis
```

### Quarterly Reviews

#### **Architecture & Performance Review**
1. **Performance Baseline Update**: Re-establish performance baselines
2. **Priority Adjustment**: Review and adjust task priority assignments based on usage
3. **Capacity Planning**: Plan infrastructure scaling based on growth trends
4. **Configuration Tuning**: Optimize timeouts, retry policies, and resource limits
5. **Security Review**: Audit task execution permissions and access controls

---

## Integration with Monitoring

### OpenTelemetry Metrics

#### **Key Metrics Integration**
```python
# OpenTelemetry metrics integration
from opentelemetry import metrics

# Task execution metrics
task_counter = metrics.get_meter("background_tasks").create_counter(
    "task_executions_total",
    description="Total number of task executions",
    unit="1"
)

task_duration_histogram = metrics.get_meter("background_tasks").create_histogram(
    "task_duration_seconds", 
    description="Task execution duration",
    unit="s"
)

# Queue depth gauge
queue_depth_gauge = metrics.get_meter("background_tasks").create_up_down_counter(
    "queue_depth",
    description="Current queue depth",
    unit="1"
)
```

### Prometheus Integration

#### **Custom Metrics Export**
```python
# Prometheus metrics for task manager
from prometheus_client import Counter, Histogram, Gauge

TASK_EXECUTIONS = Counter('background_task_executions_total', 'Total task executions', ['status', 'priority'])
TASK_DURATION = Histogram('background_task_duration_seconds', 'Task execution duration', ['task_type'])
QUEUE_DEPTH = Gauge('background_task_queue_depth', 'Current queue depth')
ACTIVE_TASKS = Gauge('background_task_active_count', 'Number of active tasks')
```

### Alerting Configuration

#### **Critical Alerts**
- **Task Manager Down**: No heartbeat for >5 minutes
- **High Failure Rate**: Success rate <90% for >10 minutes
- **Queue Backup**: Queue depth >50 for >5 minutes  
- **Circuit Breaker Open**: Any critical circuit breaker open
- **Memory Leak**: Memory usage growing >10% per hour

#### **Warning Alerts**
- **Degraded Performance**: Average task time >60s
- **High Queue Depth**: Queue depth >20 for >5 minutes
- **Dead Letter Queue**: >10 tasks in dead letter queue
- **Resource Pressure**: CPU >80% or Memory >80%

---

## Troubleshooting Reference

### Quick Diagnostic Commands

```bash
# Task manager status
python3 -c "from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager; tm = get_background_task_manager(); print(f'Running: {len(tm.running_tasks)}, Queued: {len(tm.priority_queue)}, Failed: {len([t for t in tm.tasks.values() if t.status.value == \"failed\"])}')"

# Recent errors
tail -100 /var/log/prompt-improver/background-tasks.log | grep ERROR

# Memory usage
ps aux | grep "python.*background_manager" | awk '{print $6/1024 " MB"}'

# Task distribution
python3 -c "from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager; tm = get_background_task_manager(); priorities = {}; [priorities.update({t.priority.name: priorities.get(t.priority.name, 0) + 1}) for t in tm.tasks.values()]; print(priorities)"
```

### Log Analysis Patterns

#### **Common Error Patterns**
```bash
# Task timeout errors
grep "Task timeout" /var/log/prompt-improver/*.log

# Circuit breaker activations  
grep "Circuit breaker opened" /var/log/prompt-improver/*.log

# Memory pressure
grep "Memory pressure" /var/log/prompt-improver/*.log

# Database connection issues
grep "Database.*connection.*failed" /var/log/prompt-improver/*.log
```

### Performance Troubleshooting

#### **Slow Task Execution**
1. Check resource constraints (CPU, memory, I/O)
2. Review task implementation for blocking operations
3. Analyze database query performance
4. Check external service response times
5. Consider task priority adjustments

#### **High Memory Usage**
1. Review task cleanup procedures
2. Check for memory leaks in task implementations
3. Analyze object retention patterns
4. Consider task timeout adjustments
5. Review garbage collection settings

---

## Contact Information

### Escalation Contacts
- **Primary On-Call**: DevOps Team
- **Secondary**: SRE Team  
- **Architecture Questions**: System Architecture Team
- **Performance Issues**: Performance Engineering Team

### Documentation Updates
This runbook should be updated whenever:
- New monitoring capabilities are added  
- Performance characteristics change significantly
- New failure modes are discovered
- Emergency procedures are modified

---

**Document End**

*This runbook is a living document. Please update it based on operational experience and system evolution.*