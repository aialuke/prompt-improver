# DatabaseServices Operational Runbook
**Production Operations Guide**

**Version**: 2025.2  
**Authority**: Database Operations Team  
**Status**: Production-Ready Operations Guide  
**Architecture**: Service Composition with Clean Architecture

---

## 🎯 Executive Summary

The DatabaseServices architecture implements clean service composition replacing the previous monolithic approach. This runbook provides comprehensive operational procedures for monitoring, maintaining, and troubleshooting the distributed database service components in production environments.

### Key Performance Indicators
- **3,922-line monolith eliminated** with zero backwards compatibility
- **Multi-level cache performance**: L1 (~0.001ms), L2 (~1-5ms), L3 (~10-50ms)
- **Response Time**: 0.1-500ms target with P95 <100ms
- **Memory Usage**: Optimized within 10-1000MB range
- **99.9% Availability** with circuit breaker protection
- **Zero Single Points of Failure** through service composition

---

## 🏗️ System Architecture Overview

### **DatabaseServices Components**

```
┌─────────────────────────────────────────────────────────────┐
│                   DatabaseServices                         │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│ PostgresPool│ CacheManager│ LockManager │   HealthManager │
│   Manager   │             │             │                 │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ • Connection│ • L1 Memory │ • Redis Lock│ • Circuit       │
│   Pooling   │ • L2 Redis  │ • Timeout   │   Breaker       │
│ • Session   │ • L3 DB     │ • Lease Mgmt│ • Health Checks │
│   Lifecycle │ • Warming   │ • Contention│ • Failure Det.  │
└─────────────┴─────────────┴─────────────┴─────────────────┘
                            │
                   ┌────────┴────────┐
                   │   PubSubManager │
                   │ • Channel Mgmt  │
                   │ • Message Route │
                   │ • Broadcasting  │
                   └─────────────────┘
```

### **Service Composition Pattern**

```python
# Composition Root (src/prompt_improver/database/composition.py)
class DatabaseServices:
    def __init__(self, mode: ManagerMode):
        self.database = PostgresPoolManager(pool_config)
        self.cache = CacheManager(database=self.database, redis_config=redis_config)
        self.lock_manager = DistributedLockManager(redis_client=self.cache.redis_client)
        self.pubsub = PubSubManager(redis_client=self.cache.redis_client)
        self.health_manager = HealthManager(database=self.database, cache=self.cache)
```

---

## 🚀 Deployment Procedures

### 1. **Pre-Deployment Validation**

**Service Health Verification:**
```bash
# Verify all service components are healthy
python -c "
import asyncio
from prompt_improver.database import get_database_services, ManagerMode

async def check_health():
    services = await get_database_services(ManagerMode.ASYNC_MODERN)
    await services.initialize()
    
    # Check each service component
    db_health = await services.database.health_check()
    cache_health = await services.cache.health_check()
    lock_health = await services.lock_manager.health_check()
    
    print(f'Database: {db_health.status}')
    print(f'Cache: {cache_health.status}')
    print(f'Locks: {lock_health.status}')

asyncio.run(check_health())
"
```

**Performance Baseline:**
```bash
# Run performance validation
python -c "
import asyncio
import time
from prompt_improver.database import get_database_services, ManagerMode

async def performance_check():
    services = await get_database_services(ManagerMode.ASYNC_MODERN)
    
    # L1 Cache Performance (should be <0.001ms)
    start = time.time()
    await services.cache.set('test_key', 'test_value')
    await services.cache.get('test_key')  # L1 hit
    l1_time = (time.time() - start) * 1000
    
    print(f'L1 Cache: {l1_time:.3f}ms (target: <1ms)')
    assert l1_time < 1, 'L1 cache performance degraded'

asyncio.run(performance_check())
"
```

### 2. **Service Initialization Order**

**Correct Startup Sequence:**
```python
# 1. Initialize database services
services = await get_database_services(ManagerMode.ASYNC_MODERN)

# 2. Initialize individual components (automatic in composition)
await services.initialize()  # Handles proper ordering internally

# 3. Verify health status
health_status = await services.health_manager.check_health()
assert health_status.status == "healthy"
```

**Environment Variables Required:**
```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=0

# Redis Configuration
REDIS_URL=redis://host:6379/0
REDIS_POOL_MIN_SIZE=5
REDIS_POOL_MAX_SIZE=50

# Cache Configuration
CACHE_L1_SIZE=1000
CACHE_L2_TTL=300
CACHE_L3_ENABLED=true
```

---

## 📊 Monitoring & Observability

### **Service Health Monitoring**

**Health Check Endpoint:**
```python
# API endpoint for service health
@app.get("/health/database-services")
async def database_services_health():
    services = await get_database_services()
    
    # Comprehensive health check
    health_report = {
        "database": await services.database.health_check(),
        "cache": await services.cache.health_check(),
        "locks": await services.lock_manager.health_check(),
        "pubsub": await services.pubsub.health_check(),
        "overall": await services.health_manager.check_health()
    }
    
    return health_report
```

**Key Metrics to Monitor:**

1. **Database Metrics:**
   - Connection pool utilization: `services.database.get_pool_stats()`
   - Query response times: Should be <100ms P95
   - Active connections: Monitor for connection leaks

2. **Cache Metrics:**
   - L1 hit rate: Target >90%
   - L2 hit rate: Target >85%  
   - Cache warming effectiveness: Monitor pattern accuracy

3. **Lock Manager Metrics:**
   - Lock contention rate: Monitor for high contention
   - Lock timeout rate: Should be <1%
   - Average lock duration: Monitor for long-held locks

4. **Health Manager Metrics:**
   - Circuit breaker state: Monitor for OPEN state
   - Failure detection rate: Track service failures
   - Recovery time: Monitor time to recover from failures

### **Performance Baselines**

```bash
# Cache Performance Verification
curl -s http://localhost:8000/health/cache | jq '.l1_hit_rate'
# Expected: >0.90

# Database Performance Verification  
curl -s http://localhost:8000/health/database | jq '.avg_response_time_ms'
# Expected: <100ms

# Overall System Health
curl -s http://localhost:8000/health/database-services | jq '.overall.status'
# Expected: "healthy"
```

---

## 🚨 Troubleshooting Guide

### **Common Issues & Solutions**

#### 1. **Service Initialization Failures**

**Symptoms:**
- `DatabaseServices initialization failed`
- Services unable to connect to PostgreSQL/Redis

**Diagnosis:**
```python
# Check individual service health
services = await get_database_services(ManagerMode.ASYNC_MODERN)

try:
    await services.database.health_check()
except Exception as e:
    print(f"Database issue: {e}")

try:
    await services.cache.health_check()  
except Exception as e:
    print(f"Cache issue: {e}")
```

**Solutions:**
1. Verify environment variables are correctly set
2. Check PostgreSQL/Redis connectivity
3. Validate configuration values
4. Check security context permissions

#### 2. **Cache Performance Degradation**

**Symptoms:**
- L1 cache hit rate <90%
- L2 cache response times >10ms
- Overall cache effectiveness declining

**Diagnosis:**
```python
# Get detailed cache statistics
cache_stats = await services.cache.get_stats()
print(f"L1 hit rate: {cache_stats['l1_hit_rate']}")
print(f"L2 hit rate: {cache_stats['l2_hit_rate']}")
print(f"Cache warming accuracy: {cache_stats['warming_accuracy']}")
```

**Solutions:**
1. Increase L1 cache size if memory allows
2. Review cache warming patterns for accuracy
3. Check TTL settings for optimal expiration
4. Monitor for cache key hotspots

#### 3. **Connection Pool Exhaustion**

**Symptoms:**  
- `asyncpg.exceptions.TooManyConnectionsError`
- Database queries timing out
- High connection pool utilization

**Diagnosis:**
```python
# Check pool statistics
pool_stats = await services.database.get_pool_stats()
print(f"Active connections: {pool_stats['active_connections']}")
print(f"Pool size: {pool_stats['pool_size']}")
print(f"Overflow: {pool_stats['overflow']}")
```

**Solutions:**
1. Increase `DATABASE_POOL_SIZE` if resources allow
2. Check for connection leaks in application code
3. Verify proper async context manager usage
4. Review query patterns for optimization

#### 4. **Circuit Breaker Activation**

**Symptoms:**
- `CircuitBreakerOpen` exceptions
- Services failing fast without attempting operations
- Health checks showing degraded status

**Diagnosis:**
```python
# Check circuit breaker status
cb_status = await services.health_manager.get_circuit_breaker_status()
print(f"State: {cb_status['state']}")
print(f"Failure count: {cb_status['failure_count']}")
print(f"Last failure: {cb_status['last_failure_time']}")
```

**Solutions:**
1. Identify root cause of failures triggering circuit breaker
2. Wait for automatic recovery (typically 30-60 seconds)
3. Manually reset circuit breaker if safe to do so
4. Adjust circuit breaker thresholds if needed

---

## 🔧 Maintenance Procedures

### **Routine Maintenance Tasks**

#### Daily Health Checks
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== DatabaseServices Daily Health Check ==="
echo "Date: $(date)"

# Overall service health
curl -s http://localhost:8000/health/database-services | jq .

# Performance metrics
echo "Performance Metrics:"
curl -s http://localhost:8000/metrics/database-services | jq .

# Check for any errors in logs
echo "Recent Errors:"
grep -i error /var/log/prompt-improver/database-services.log | tail -10
```

#### Weekly Performance Review
```bash
#!/bin/bash
# weekly_performance_review.sh

echo "=== Weekly Performance Review ==="
echo "Week ending: $(date)"

# Cache performance trends
echo "Cache Hit Rates (7-day average):"
curl -s http://localhost:8000/metrics/cache/weekly | jq '.hit_rates'

# Database performance trends  
echo "Database Response Times (7-day average):"
curl -s http://localhost:8000/metrics/database/weekly | jq '.response_times'

# Resource utilization
echo "Resource Utilization:"
curl -s http://localhost:8000/metrics/resources/weekly | jq .
```

### **Service Updates & Migrations**

#### Safe Update Procedure
1. **Pre-update validation**: Run health checks and performance baselines
2. **Gradual rollout**: Use blue-green deployment with health monitoring
3. **Rollback preparation**: Ensure quick rollback capability
4. **Post-update verification**: Comprehensive health and performance validation

#### Configuration Changes
```python
# Safe configuration update pattern
async def update_configuration(new_config):
    # 1. Validate new configuration
    validated_config = await validate_config(new_config)
    
    # 2. Create new services with updated config
    new_services = await create_database_services(
        mode=ManagerMode.ASYNC_MODERN,
        config=validated_config
    )
    
    # 3. Health check new services
    health = await new_services.health_manager.check_health()
    if health.status != "healthy":
        raise ConfigurationError("New configuration failed health check")
    
    # 4. Gracefully replace old services
    await old_services.shutdown()
    return new_services
```

---

## 🔒 Security Operations

### **Security Context Management**

**Security Tier Verification:**
```python
# Verify security context is properly applied
security_context = create_security_context(
    agent_id="production_api",
    tier=SecurityTier.ENTERPRISE
)

services = await create_database_services(
    mode=ManagerMode.ASYNC_MODERN,
    security_context=security_context
)

# Verify permissions
permissions = await services.security_context.get_permissions()
print(f"Tier: {permissions.tier}")
print(f"Rate limit: {permissions.rate_limit}")
```

**Security Monitoring:**
- Monitor authentication failures
- Track tier-based rate limiting violations  
- Audit security context changes
- Log all administrative operations

### **Data Protection Compliance**

**Encryption Verification:**
```bash
# Verify all connections use encryption
netstat -an | grep :5432 | grep ESTABLISHED  # PostgreSQL connections
netstat -an | grep :6379 | grep ESTABLISHED  # Redis connections

# Check TLS configuration
openssl s_client -connect postgres-host:5432 -starttls postgres
```

---

## 📈 Performance Optimization

### **Optimization Strategies**

#### Cache Optimization
```python
# Monitor and optimize cache performance
async def optimize_cache_performance():
    stats = await services.cache.get_stats()
    
    # L1 optimization
    if stats['l1_hit_rate'] < 0.90:
        await services.cache.increase_l1_size()
    
    # L2 optimization  
    if stats['l2_response_time'] > 5.0:
        await services.cache.optimize_l2_connections()
    
    # Cache warming optimization
    if stats['warming_accuracy'] < 0.85:
        await services.cache.retrain_warming_model()
```

#### Database Optimization
```python
# Monitor and optimize database performance
async def optimize_database_performance():
    pool_stats = await services.database.get_pool_stats()
    
    # Pool size optimization
    if pool_stats['utilization'] > 0.80:
        await services.database.scale_pool_size()
    
    # Query optimization
    slow_queries = await services.database.get_slow_queries()
    for query in slow_queries:
        await services.database.optimize_query(query)
```

### **Scaling Strategies**

**Horizontal Scaling:**
- Deploy multiple DatabaseServices instances
- Use load balancer for request distribution
- Implement consistent hashing for cache distribution

**Vertical Scaling:**
- Increase connection pool sizes
- Expand cache memory allocation
- Optimize hardware resources

---

## 📋 Runbook Checklist

### **Pre-Production Deployment**
- [ ] Environment variables configured
- [ ] Service health checks passing  
- [ ] Performance baselines validated
- [ ] Security context properly configured
- [ ] Monitoring alerts configured
- [ ] Backup and recovery procedures tested

### **Production Operations**
- [ ] Daily health checks completed
- [ ] Performance metrics within targets
- [ ] No critical errors in logs
- [ ] Resource utilization optimal
- [ ] Security monitoring active
- [ ] Backup procedures functioning

### **Incident Response**
- [ ] Service degradation detected
- [ ] Root cause identified
- [ ] Mitigation actions taken
- [ ] Service recovery validated
- [ ] Post-incident review scheduled
- [ ] Documentation updated

---

## 📞 Support & Escalation

**Tier 1 Support:** Application Operations Team
- Service health monitoring
- Basic troubleshooting
- Configuration management

**Tier 2 Support:** Database Engineering Team  
- Performance optimization
- Complex troubleshooting
- Architecture modifications

**Tier 3 Support:** Principal Engineers
- Critical system failures
- Architecture redesign
- Emergency escalation

**Emergency Contact:** `oncall-database@company.com`

---

*This runbook reflects the DatabaseServices architecture implemented in 2025. The legacy UnifiedConnectionManager has been completely replaced with clean service composition patterns.*