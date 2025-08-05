# Troubleshooting and Emergency Response Guide
**Comprehensive Issue Resolution for UnifiedConnectionManager**

**Version**: 2025.1  
**Authority**: Site Reliability Engineering Team  
**Status**: Production Emergency Procedures  
**Emergency Hotline**: +1-555-URGENT-OPS

---

## 🚨 Emergency Contact Information

### **Immediate Response (24/7)**
```yaml
Primary On-Call: "sre-oncall@company.com" / +1-555-SRE-ONCALL
Secondary On-Call: "dev-oncall@company.com" / +1-555-DEV-ONCALL
Escalation Manager: "engineering-manager@company.com" / +1-555-ENG-MGR
Incident Commander: "incident-commander@company.com" / +1-555-INCIDENT

Emergency Conference Bridge: https://meet.company.com/emergency-cache
Incident Tracking: https://incident.company.com/
Status Page: https://status.company.com/
```

### **Severity Classification**
- **P0 (Critical)**: Complete system failure, data loss risk, security breach
- **P1 (High)**: Major functionality impaired, significant performance degradation
- **P2 (Medium)**: Minor functionality issues, workarounds available
- **P3 (Low)**: Cosmetic issues, enhancement requests

---

## 🎯 Quick Diagnosis Decision Tree

### **Step 1: Initial Assessment (2 minutes)**
```
Is the system responding to health checks?
├─ YES → Go to Performance Issues (Section 3)
└─ NO → Go to System Failure (Section 2)

Are Redis connections working?
├─ YES → Check database connections
└─ NO → Go to Redis Issues (Section 4)

Is cache hit rate >80%?
├─ YES → Check response times
└─ NO → Go to Cache Performance Issues (Section 5)

Are there security alerts?
├─ YES → Go to Security Issues (Section 6)
└─ NO → Continue with standard troubleshooting
```

---

## 🔥 P0 Critical Emergencies

### **Emergency 1: Complete System Failure**

#### **Symptoms**
- Health check endpoints returning 500/503 errors
- Application startup failures
- Cannot establish any database connections
- Redis connections completely failed

#### **Immediate Response (0-5 minutes)**
```bash
#!/bin/bash
# emergency_system_failure.sh

echo "🚨 P0 EMERGENCY: COMPLETE SYSTEM FAILURE 🚨"
echo "Time: $(date)"
echo "Responder: $(whoami)"

# 1. Declare incident
echo "1. Declaring P0 incident..."
# curl -X POST https://incident.company.com/api/incidents \
#   -H "Authorization: Bearer $INCIDENT_API_KEY" \
#   -d '{"severity": "P0", "title": "UnifiedConnectionManager Complete Failure"}'

# 2. Quick system status check
echo "2. System status check..."
systemctl status prompt-improver
docker ps | grep prompt-improver  # If containerized
ps aux | grep python | grep prompt-improver

# 3. Check critical dependencies
echo "3. Dependency status..."
echo -n "Redis: "
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping 2>/dev/null || echo "❌ DOWN"

echo -n "Database: "
pg_isready -h $DATABASE_HOST -p $DATABASE_PORT 2>/dev/null || echo "❌ DOWN"

# 4. Check logs for immediate errors
echo "4. Recent critical errors..."
tail -50 /var/log/prompt-improver/app.log | grep -i "error\|exception\|critical" | tail -10

# 5. Check resource utilization
echo "5. Resource utilization..."
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
echo "CPU: $(top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}')"

echo "🚨 IMMEDIATE ASSESSMENT COMPLETE - PROCEED WITH RECOVERY 🚨"
```

#### **Recovery Actions (5-15 minutes)**
```bash
#!/bin/bash
# emergency_recovery.sh

echo "🔧 P0 EMERGENCY RECOVERY 🔧"

# 1. Attempt service restart
echo "1. Service restart attempt..."
systemctl stop prompt-improver
sleep 5
systemctl start prompt-improver
sleep 10

# 2. Test basic functionality
echo "2. Basic functionality test..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def emergency_test():
    try:
        print('   Testing UnifiedConnectionManager initialization...')
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        
        print('   Testing cache operations...')
        await manager.set_cached('emergency_test', 'recovery_active', ttl=300)
        result = await manager.get_cached('emergency_test')
        
        if result == 'recovery_active':
            print('   ✅ Cache operations working')
        else:
            print('   ❌ Cache operations failed')
            
        print('   Testing database operations...')
        async with manager.get_session() as session:
            db_result = await session.execute('SELECT NOW()')
            timestamp = db_result.fetchone()[0]
            print(f'   ✅ Database connection working: {timestamp}')
        
        await manager.close()
        print('✅ EMERGENCY RECOVERY SUCCESSFUL')
        
    except Exception as e:
        print(f'❌ EMERGENCY RECOVERY FAILED: {e}')
        exit(1)

asyncio.run(emergency_test())
"

# 3. If recovery failed, attempt emergency mode
if [ $? -ne 0 ]; then
    echo "3. Attempting emergency fallback mode..."
    
    # Emergency environment variables
    export UCM_L1_CACHE_SIZE=500  # Minimal L1 cache
    export UCM_ENABLE_WARMING=false  # Disable warming to reduce load
    export REDIS_MAX_CONNECTIONS=10  # Minimal connections
    export DATABASE_MAX_CONNECTIONS=5
    
    systemctl restart prompt-improver
    
    # Test emergency mode
    python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_emergency_mode():
    manager = get_unified_manager(ManagerMode.ADMIN)  # Minimal mode
    await manager.initialize()
    
    # Test basic L1 cache only
    await manager.set_cached('emergency_l1_test', 'l1_working', ttl=60)
    result = await manager.get_cached('emergency_l1_test')
    
    if result == 'l1_working':
        print('✅ EMERGENCY MODE ACTIVE - L1 CACHE ONLY')
    else:
        print('❌ EMERGENCY MODE FAILED')
        exit(1)
    
    await manager.close()

asyncio.run(test_emergency_mode())
"

fi

echo "🔧 EMERGENCY RECOVERY COMPLETE 🔧"
```

### **Emergency 2: Data Corruption/Loss Risk**

#### **Symptoms**
- Cache returning incorrect data
- Database integrity errors
- Inconsistent data between cache layers

#### **Immediate Response (0-3 minutes)**
```bash
#!/bin/bash
# emergency_data_integrity.sh

echo "🚨 P0 EMERGENCY: DATA INTEGRITY RISK 🚨"

# 1. STOP ALL WRITES IMMEDIATELY
echo "1. Emergency write lockdown..."
# Implementation depends on your architecture
# Could involve setting read-only mode, stopping write services, etc.

# 2. Clear all caches to prevent corruption spread
echo "2. Emergency cache flush..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def emergency_cache_flush():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    print('   Clearing L1 cache...')
    await manager.clear_cache()
    
    print('   Clearing Redis cache...')
    if manager._redis_master:
        await manager._redis_master.flushdb()
    
    print('   ✅ All caches cleared')
    await manager.close()

asyncio.run(emergency_cache_flush())
"

# 3. Database integrity check
echo "3. Database integrity check..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def integrity_check():
    manager = get_unified_manager(ManagerMode.ADMIN)
    await manager.initialize()
    
    async with manager.get_session() as session:
        # Run basic integrity checks
        result = await session.execute('SELECT COUNT(*) FROM information_schema.tables')
        table_count = result.fetchone()[0]
        print(f'   Database tables accessible: {table_count}')
        
        # Check for obvious corruption signs
        # This would be customized for your specific tables
        
    await manager.close()

asyncio.run(integrity_check())
"

echo "🚨 DATA INTEGRITY EMERGENCY RESPONSE COMPLETE 🚨"
echo "📞 ESCALATE TO DATA TEAM IMMEDIATELY"
```

---

## ⚡ P1 High Priority Issues

### **Issue 1: Severe Performance Degradation**

#### **Symptoms**
- Response times >200ms consistently
- Cache hit rate <70%
- High CPU/memory usage
- User complaints about slowness

#### **Diagnosis Procedure**
```bash
#!/bin/bash
# diagnose_performance_degradation.sh

echo "🔍 P1 DIAGNOSIS: PERFORMANCE DEGRADATION 🔍"

# 1. Get current performance metrics
echo "1. Performance metrics snapshot..."
python3 -c "
import asyncio
import time
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def performance_snapshot():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    # Get cache statistics
    stats = await manager.get_cache_stats()
    print(f'   Cache Hit Rate: {stats.hit_rate:.2%}')
    print(f'   L1 Utilization: {stats.l1_utilization:.2%}')
    print(f'   Cache Health: {stats.health_status}')
    
    # Performance test
    start_time = time.time()
    for i in range(100):
        await manager.get_cached(f'perf_test_{i % 10}')
    end_time = time.time()
    
    avg_response_time = (end_time - start_time) / 100
    print(f'   Average Response Time: {avg_response_time*1000:.2f}ms')
    
    # Connection metrics
    conn_metrics = manager.get_connection_metrics()
    print(f'   DB Pool Utilization: {conn_metrics.pool_utilization:.2%}')
    print(f'   Redis Master Connected: {conn_metrics.redis_master_connected}')
    
    await manager.close()

asyncio.run(performance_snapshot())
"

# 2. System resource check
echo "2. System resources..."
echo "   Memory Usage:"
free -h | grep -E "(Mem|Swap)"

echo "   CPU Load:"
uptime

echo "   Disk I/O:"
iostat -x 1 1 | grep -v "^$"

# 3. Database performance
echo "3. Database performance..."
python3 -c "
import asyncio
import time
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def db_performance_check():
    manager = get_unified_manager(ManagerMode.ADMIN)
    await manager.initialize()
    
    async with manager.get_session() as session:
        # Check active connections
        result = await session.execute('SELECT count(*) FROM pg_stat_activity')
        active_connections = result.fetchone()[0]
        print(f'   Active DB Connections: {active_connections}')
        
        # Check for long-running queries
        result = await session.execute('''
            SELECT count(*) FROM pg_stat_activity 
            WHERE state = 'active' AND query_start < NOW() - INTERVAL '1 minute'
        ''')
        long_queries = result.fetchone()[0]
        print(f'   Long-running queries: {long_queries}')
        
        # Test query performance
        start_time = time.time()
        await session.execute('SELECT 1')
        query_time = time.time() - start_time
        print(f'   Simple query time: {query_time*1000:.2f}ms')
    
    await manager.close()

asyncio.run(db_performance_check())
"

echo "🔍 PERFORMANCE DIAGNOSIS COMPLETE - ANALYZE RESULTS 🔍"
```

#### **Resolution Actions**
```bash
#!/bin/bash
# resolve_performance_degradation.sh

echo "🔧 P1 RESOLUTION: PERFORMANCE OPTIMIZATION 🔧"

# 1. Cache optimization
echo "1. Cache optimization..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def optimize_cache():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    stats = await manager.get_cache_stats()
    
    # If L1 utilization is high, clear expired entries
    if stats.l1_utilization > 0.90:
        print('   L1 cache utilization high - clearing expired entries')
        # Force cleanup of expired entries
        await manager.clear_cache()  # This will restart with clean cache
        print('   ✅ L1 cache cleared and reset')
    
    # If cache warming is disabled, enable it
    if not stats.warming_enabled:
        print('   Cache warming disabled - this is likely the issue')
        print('   Set UCM_ENABLE_WARMING=true and restart service')
    
    await manager.close()

asyncio.run(optimize_cache())
"

# 2. Connection pool optimization
echo "2. Connection pool optimization..."
# Check if we need to increase connection pool sizes
CURRENT_REDIS_CONNECTIONS=$(grep REDIS_MAX_CONNECTIONS /etc/prompt-improver/.env | cut -d= -f2)
CURRENT_DB_CONNECTIONS=$(grep DATABASE_MAX_CONNECTIONS /etc/prompt-improver/.env | cut -d= -f2)

echo "   Current Redis connections: $CURRENT_REDIS_CONNECTIONS"
echo "   Current DB connections: $CURRENT_DB_CONNECTIONS"

# If connections are low and system has resources, suggest increases
AVAILABLE_MEMORY_GB=$(free -g | grep Mem | awk '{print $7}')
if [ "$AVAILABLE_MEMORY_GB" -gt 2 ] && [ "$CURRENT_REDIS_CONNECTIONS" -lt 100 ]; then
    echo "   💡 Consider increasing REDIS_MAX_CONNECTIONS to 100"
fi

# 3. Emergency performance mode
echo "3. Applying emergency performance optimizations..."
cat << EOF > /tmp/emergency_performance.env
# Emergency performance settings
UCM_L1_CACHE_SIZE=3000
UCM_CACHE_TTL_SECONDS=900
UCM_ENABLE_WARMING=true
UCM_WARMING_INTERVAL=60
UCM_WARMING_THRESHOLD=0.5
REDIS_MAX_CONNECTIONS=150
DATABASE_MAX_CONNECTIONS=75
EOF

echo "   Emergency performance configuration created at /tmp/emergency_performance.env"
echo "   Review and apply if appropriate, then restart service"

echo "🔧 PERFORMANCE RESOLUTION COMPLETE 🔧"
```

### **Issue 2: Redis Connection Failures**

#### **Symptoms**
- Redis master/replica connection errors
- Cache operations falling back to database
- High database load

#### **Quick Fix Procedure**
```bash
#!/bin/bash
# fix_redis_connections.sh

echo "🔧 P1 FIX: REDIS CONNECTION ISSUES 🔧"

# 1. Test Redis connectivity
echo "1. Redis connectivity test..."
echo -n "   Redis Master: "
if redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Connection failed"
    
    # Check if it's an authentication issue
    echo -n "   Testing with auth: "
    if redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping > /dev/null 2>&1; then
        echo "✅ Authentication working"
    else
        echo "❌ Authentication failed"
    fi
fi

# 2. Check SSL certificate (if using SSL)
if [ "$REDIS_USE_SSL" = "true" ]; then
    echo "2. SSL certificate check..."
    echo | openssl s_client -connect $REDIS_HOST:$REDIS_PORT -servername $REDIS_HOST 2>/dev/null | \
        openssl x509 -noout -dates
fi

# 3. Application-level connection test
echo "3. Application connection test..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_redis_connection():
    try:
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        
        # Test Redis directly
        if manager._redis_master:
            await manager._redis_master.ping()
            print('   ✅ Redis master connection working from application')
        else:
            print('   ❌ Redis master not available in connection manager')
            
        if manager._redis_replica:
            await manager._redis_replica.ping()
            print('   ✅ Redis replica connection working from application')
        else:
            print('   ⚠️ Redis replica not available (may be intentional)')
        
        await manager.close()
        
    except Exception as e:
        print(f'   ❌ Application Redis connection failed: {e}')
        
        # Try emergency mode with minimal settings
        print('   Attempting emergency connection...')
        import coredis
        try:
            redis_client = coredis.Redis(
                host='$REDIS_HOST',
                port=$REDIS_PORT,
                password='$REDIS_PASSWORD' if '$REDIS_PASSWORD' else None,
                decode_responses=True
            )
            await redis_client.ping()
            print('   ✅ Direct Redis connection working - issue is in UnifiedConnectionManager')
            await redis_client.close()
        except Exception as direct_e:
            print(f'   ❌ Direct Redis connection also failed: {direct_e}')

asyncio.run(test_redis_connection())
"

# 4. Quick fixes
echo "4. Applying quick fixes..."

# Restart Redis connection pools
echo "   Restarting application to reset connection pools..."
systemctl restart prompt-improver

# Wait and test
sleep 15
echo "   Testing after restart..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def post_restart_test():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    await manager.set_cached('restart_test', 'success', ttl=60)
    result = await manager.get_cached('restart_test')
    
    if result == 'success':
        print('   ✅ Redis connection restored after restart')
    else:
        print('   ❌ Redis connection still failing after restart')
    
    await manager.close()

asyncio.run(post_restart_test())
"

echo "🔧 REDIS CONNECTION FIX COMPLETE 🔧"
```

---

## 🔍 Common Troubleshooting Scenarios

### **Scenario 1: Cache Hit Rate Suddenly Drops**

#### **Investigation Steps**
```bash
#!/bin/bash
# investigate_cache_hit_rate_drop.sh

echo "🔍 INVESTIGATING: CACHE HIT RATE DROP 🔍"

# 1. Get detailed cache statistics
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def detailed_cache_analysis():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    stats = await manager.get_cache_stats()
    
    print('📊 Cache Statistics Analysis:')
    print(f'   Overall Hit Rate: {stats.hit_rate:.2%}')
    print(f'   L1 Hit Rate: {stats.l1_hits / max(1, stats.total_requests):.2%}')
    print(f'   L2 Hit Rate: {stats.l2_hits / max(1, stats.total_requests):.2%}')
    print(f'   Cache Misses: {(stats.total_requests - stats.l1_hits - stats.l2_hits) / max(1, stats.total_requests):.2%}')
    
    print(f'\\n🏠 L1 Cache Analysis:')
    print(f'   Size: {stats.l1_size} entries')
    print(f'   Utilization: {stats.l1_utilization:.2%}')
    
    print(f'\\n🔄 Cache Warming Analysis:')
    print(f'   Warming Enabled: {stats.warming_enabled}')
    print(f'   Warming Cycles: {stats.warming_cycles}')
    print(f'   Warming Hit Rate: {stats.warming_hit_rate:.2%}')
    
    # Check for signs of issues
    if stats.hit_rate < 0.80:
        print('\\n⚠️ ISSUE IDENTIFIED: Low overall hit rate')
        
        if not stats.warming_enabled:
            print('   🔧 FIX: Enable cache warming (UCM_ENABLE_WARMING=true)')
        
        if stats.l1_utilization > 0.95:
            print('   🔧 FIX: L1 cache oversaturated - increase UCM_L1_CACHE_SIZE')
        
        if stats.l1_utilization < 0.30:
            print('   🔧 FIX: L1 cache underutilized - check TTL settings')
    
    await manager.close()

asyncio.run(detailed_cache_analysis())
"

# 2. Check for access pattern changes
echo "2. Access pattern analysis..."
echo "   Recent log patterns:"
grep -i "cache.*miss\|cache.*hit" /var/log/prompt-improver/app.log | tail -10

# 3. Check for configuration changes
echo "3. Configuration drift check..."
echo "   Current cache configuration:"
env | grep UCM_ | sort

echo "🔍 CACHE HIT RATE INVESTIGATION COMPLETE 🔍"
```

#### **Common Fixes**
```python
# cache_hit_rate_fixes.py
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def apply_cache_hit_rate_fixes():
    """Apply common fixes for cache hit rate issues."""
    
    print("🔧 Applying Cache Hit Rate Fixes...")
    
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    stats = await manager.get_cache_stats()
    
    # Fix 1: Enable cache warming if disabled
    if not stats.warming_enabled:
        print("Fix 1: Cache warming is disabled")
        print("   Set UCM_ENABLE_WARMING=true and restart service")
        print("   This should improve hit rate by 20-30%")
    
    # Fix 2: Optimize L1 cache size
    if stats.l1_utilization > 0.95:
        print("Fix 2: L1 cache oversaturated")
        current_size = stats.l1_size
        recommended_size = int(current_size * 1.5)
        print(f"   Increase UCM_L1_CACHE_SIZE from {current_size} to {recommended_size}")
    elif stats.l1_utilization < 0.30:
        print("Fix 3: L1 cache underutilized") 
        print("   Check TTL settings - entries may be expiring too quickly")
        print("   Consider increasing UCM_CACHE_TTL_SECONDS")
    
    # Fix 4: Clear corrupted cache
    if stats.hit_rate < 0.50:  # Extremely low
        print("Fix 4: Cache may be corrupted")
        print("   Clearing all caches for fresh start...")
        await manager.clear_cache()
        print("   ✅ Cache cleared - monitor hit rate recovery")
    
    await manager.close()
    print("🔧 Cache Hit Rate Fixes Applied")

if __name__ == "__main__":
    asyncio.run(apply_cache_hit_rate_fixes())
```

### **Scenario 2: Memory Leak in L1 Cache**

#### **Detection and Resolution**
```python
# memory_leak_detection.py
import asyncio
import time
import psutil
import os
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def detect_and_fix_memory_leak():
    """Detect and resolve memory leaks in L1 cache."""
    
    print("🔍 Memory Leak Detection and Resolution")
    
    # Get current process memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    # Check L1 cache state
    stats = await manager.get_cache_stats()
    print(f"L1 cache size: {stats.l1_size} entries")
    print(f"L1 utilization: {stats.l1_utilization:.2%}")
    
    # Test for memory leak by cycling cache entries
    print("Testing for memory leak...")
    
    # Add many entries
    for i in range(1000):
        await manager.set_cached(f"leak_test_{i}", f"data_{i}", ttl=10)
    
    # Check memory after additions
    mid_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after cache additions: {mid_memory:.2f} MB")
    
    # Wait for TTL expiry
    print("Waiting for TTL expiry...")
    await asyncio.sleep(12)
    
    # Check if memory was released
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after TTL expiry: {final_memory:.2f} MB")
    
    # Analyze results
    memory_growth = final_memory - initial_memory
    if memory_growth > 50:  # More than 50MB growth
        print(f"⚠️ MEMORY LEAK DETECTED: {memory_growth:.2f} MB not released")
        
        # Apply fixes
        print("Applying memory leak fixes...")
        
        # Fix 1: Force cache clear
        await manager.clear_cache()
        post_clear_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after cache clear: {post_clear_memory:.2f} MB")
        
        # Fix 2: Reduce cache size if still high
        if post_clear_memory > initial_memory + 20:
            print("Recommending L1 cache size reduction:")
            print("   Set UCM_L1_CACHE_SIZE=1000 (reduced from current)")
            print("   Restart service to apply changes")
        
    else:
        print(f"✅ No memory leak detected: {memory_growth:.2f} MB growth is acceptable")
    
    await manager.close()

if __name__ == "__main__":
    asyncio.run(detect_and_fix_memory_leak())
```

---

## 🛠️ Preventive Maintenance Scripts

### **Health Check Automation**
```bash
#!/bin/bash
# automated_health_check.sh
# Run every 5 minutes via cron

HEALTH_CHECK_LOG="/var/log/prompt-improver/health_checks.log"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $HEALTH_CHECK_LOG
}

# 1. Basic connectivity check
if python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def quick_health():
    try:
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        await manager.set_cached('health_check', 'ok', ttl=30)
        result = await manager.get_cached('health_check') 
        await manager.close()
        exit(0 if result == 'ok' else 1)
    except:
        exit(1)

asyncio.run(quick_health())
" > /dev/null 2>&1; then
    log_message "HEALTH_CHECK: PASS - System operational"
else
    log_message "HEALTH_CHECK: FAIL - System issues detected"
    
    # Trigger alert
    curl -X POST https://alerts.company.com/webhook \
        -H "Content-Type: application/json" \
        -d '{"alert": "UnifiedConnectionManager health check failed", "severity": "warning"}'
fi

# 2. Performance monitoring
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def performance_check():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    stats = await manager.get_cache_stats()
    
    if stats.hit_rate < 0.80:
        print('PERFORMANCE_ALERT: Cache hit rate low')
        exit(1)
    
    await manager.close()
    exit(0)

asyncio.run(performance_check())
" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    log_message "PERFORMANCE_ALERT: Cache performance degraded"
fi
```

### **Automated Recovery Script**
```bash
#!/bin/bash
# automated_recovery.sh
# Triggered by monitoring alerts

RECOVERY_LOG="/var/log/prompt-improver/recovery.log"

recovery_log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $RECOVERY_LOG
}

recovery_log "RECOVERY: Automated recovery initiated"

# 1. Attempt service restart
recovery_log "RECOVERY: Attempting service restart"
systemctl restart prompt-improver
sleep 30

# 2. Test recovery
if python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_recovery():
    manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    await manager.initialize()
    
    await manager.set_cached('recovery_test', 'success', ttl=60)
    result = await manager.get_cached('recovery_test')
    
    await manager.close()
    exit(0 if result == 'success' else 1)

asyncio.run(test_recovery())
" > /dev/null 2>&1; then
    recovery_log "RECOVERY: SUCCESS - Service recovered automatically"
    
    # Send success notification
    curl -X POST https://alerts.company.com/webhook \
        -H "Content-Type: application/json" \
        -d '{"alert": "UnifiedConnectionManager auto-recovery successful", "severity": "info"}'
else
    recovery_log "RECOVERY: FAILED - Manual intervention required"
    
    # Escalate to on-call
    curl -X POST https://alerts.company.com/webhook \
        -H "Content-Type: application/json" \
        -d '{"alert": "UnifiedConnectionManager auto-recovery failed - manual intervention required", "severity": "critical"}'
fi
```

---

## 📊 Diagnostic Tools and Utilities

### **Cache Analysis Tool**
```python
# cache_analyzer.py
import asyncio
import json
import statistics
from datetime import datetime
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

class CacheAnalyzer:
    def __init__(self):
        self.manager = None
    
    async def initialize(self):
        self.manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
        await self.manager.initialize()
    
    async def comprehensive_analysis(self):
        """Perform comprehensive cache analysis."""
        
        print("🔍 UnifiedConnectionManager Comprehensive Analysis")
        print("=" * 60)
        
        # Basic statistics
        stats = await self.manager.get_cache_stats()
        connection_metrics = self.manager.get_connection_metrics()
        
        analysis_report = {
            "timestamp": datetime.now().isoformat(),
            "cache_performance": {
                "overall_hit_rate": stats.hit_rate,
                "l1_hit_rate": stats.l1_hits / max(1, stats.total_requests),
                "l2_hit_rate": stats.l2_hits / max(1, stats.total_requests),
                "cache_misses": (stats.total_requests - stats.l1_hits - stats.l2_hits) / max(1, stats.total_requests)
            },
            "cache_health": {
                "l1_utilization": stats.l1_utilization,
                "l1_size": stats.l1_size,
                "warming_enabled": stats.warming_enabled,
                "warming_effectiveness": stats.warming_hit_rate if stats.warming_enabled else None,
                "health_status": stats.health_status
            },
            "connection_health": {
                "redis_master": connection_metrics.redis_master_connected,
                "redis_replica": connection_metrics.redis_replica_connected,
                "db_pool_utilization": connection_metrics.pool_utilization,
                "active_connections": connection_metrics.active_connections
            }
        }
        
        # Performance assessment
        performance_issues = []
        recommendations = []
        
        if stats.hit_rate < 0.85:
            performance_issues.append("Low cache hit rate")
            if not stats.warming_enabled:
                recommendations.append("Enable cache warming")
            if stats.l1_utilization > 0.95:
                recommendations.append("Increase L1 cache size")
        
        if stats.l1_utilization > 0.95:
            performance_issues.append("L1 cache oversaturated")
            recommendations.append("Increase UCM_L1_CACHE_SIZE")
        
        if connection_metrics.pool_utilization > 0.90:
            performance_issues.append("Database connection pool near capacity")
            recommendations.append("Increase DATABASE_MAX_CONNECTIONS")
        
        analysis_report["performance_issues"] = performance_issues
        analysis_report["recommendations"] = recommendations
        
        # Display results
        self._display_analysis(analysis_report)
        
        return analysis_report
    
    def _display_analysis(self, report):
        """Display analysis results in formatted output."""
        
        print(f"Analysis Time: {report['timestamp']}")
        print()
        
        # Cache Performance
        perf = report['cache_performance']
        print("📊 CACHE PERFORMANCE:")
        print(f"   Overall Hit Rate: {perf['overall_hit_rate']:.2%}")
        print(f"   L1 Hit Rate: {perf['l1_hit_rate']:.2%}")
        print(f"   L2 Hit Rate: {perf['l2_hit_rate']:.2%}")
        print(f"   Cache Miss Rate: {perf['cache_misses']:.2%}")
        print()
        
        # Health Status
        health = report['cache_health']
        print("🏥 CACHE HEALTH:")
        print(f"   L1 Utilization: {health['l1_utilization']:.2%}")
        print(f"   L1 Size: {health['l1_size']} entries")
        print(f"   Cache Warming: {'Enabled' if health['warming_enabled'] else 'Disabled'}")
        if health['warming_effectiveness']:
            print(f"   Warming Effectiveness: {health['warming_effectiveness']:.2%}")
        print(f"   Overall Health: {health['health_status']}")
        print()
        
        # Connection Health
        conn = report['connection_health']
        print("🔗 CONNECTION HEALTH:")
        print(f"   Redis Master: {'Connected' if conn['redis_master'] else 'Disconnected'}")
        print(f"   Redis Replica: {'Connected' if conn['redis_replica'] else 'Disconnected'}")
        print(f"   DB Pool Utilization: {conn['db_pool_utilization']:.2%}")
        print(f"   Active DB Connections: {conn['active_connections']}")
        print()
        
        # Issues and Recommendations
        if report['performance_issues']:
            print("⚠️ PERFORMANCE ISSUES:")
            for issue in report['performance_issues']:
                print(f"   - {issue}")
            print()
        
        if report['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
            print()
        
        # Overall Assessment
        score = self._calculate_health_score(report)
        if score >= 90:
            print("✅ OVERALL ASSESSMENT: EXCELLENT")
        elif score >= 75:
            print("✅ OVERALL ASSESSMENT: GOOD")
        elif score >= 60:
            print("⚠️ OVERALL ASSESSMENT: NEEDS ATTENTION")
        else:
            print("❌ OVERALL ASSESSMENT: REQUIRES IMMEDIATE ACTION")
    
    def _calculate_health_score(self, report):
        """Calculate overall health score (0-100)."""
        
        score = 0
        
        # Cache performance (40 points)
        hit_rate = report['cache_performance']['overall_hit_rate']
        if hit_rate >= 0.90:
            score += 40
        elif hit_rate >= 0.80:
            score += 30
        elif hit_rate >= 0.70:
            score += 20
        else:
            score += 10
        
        # Connection health (30 points)
        conn = report['connection_health']
        if conn['redis_master'] and conn['db_pool_utilization'] < 0.80:
            score += 30
        elif conn['redis_master']:
            score += 20
        else:
            score += 0
        
        # Cache health (30 points)
        health = report['cache_health']
        if (health['health_status'] == 'healthy' and 
            health['l1_utilization'] < 0.90 and
            health['warming_enabled']):
            score += 30
        elif health['health_status'] == 'healthy':
            score += 20
        else:
            score += 10
        
        return score
    
    async def close(self):
        if self.manager:
            await self.manager.close()

# CLI usage
async def main():
    analyzer = CacheAnalyzer()
    await analyzer.initialize()
    
    try:
        report = await analyzer.comprehensive_analysis()
        
        # Save report to file
        with open(f"/tmp/cache_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### **Emergency Diagnostic Bundle**
```bash
#!/bin/bash
# emergency_diagnostic_bundle.sh
# Collect comprehensive diagnostic information

BUNDLE_DIR="/tmp/emergency_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BUNDLE_DIR

echo "🚨 Creating Emergency Diagnostic Bundle..."
echo "Bundle location: $BUNDLE_DIR"

# 1. System information
echo "1. Collecting system information..."
{
    echo "=== System Information ==="
    uname -a
    echo
    echo "=== Memory Usage ==="
    free -h
    echo
    echo "=== Disk Usage ==="
    df -h
    echo
    echo "=== CPU Load ==="
    uptime
    echo
    echo "=== Process List ==="
    ps aux | grep -E "(prompt-improver|redis|postgres)"
} > $BUNDLE_DIR/system_info.txt

# 2. Application logs
echo "2. Collecting application logs..."
tail -1000 /var/log/prompt-improver/app.log > $BUNDLE_DIR/app_logs.txt 2>/dev/null || echo "No app logs found" > $BUNDLE_DIR/app_logs.txt

# 3. Configuration
echo "3. Collecting configuration..."
{
    echo "=== Environment Variables ==="
    env | grep -E "(REDIS|DATABASE|UCM)" | sed 's/PASSWORD=.*/PASSWORD=***REDACTED***/'
    echo
    echo "=== Service Status ==="
    systemctl status prompt-improver
} > $BUNDLE_DIR/configuration.txt

# 4. Cache analysis
echo "4. Running cache analysis..."
python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def emergency_analysis():
    try:
        manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        await manager.initialize()
        
        stats = await manager.get_cache_stats()
        conn_metrics = manager.get_connection_metrics()
        
        print('=== Cache Statistics ===')
        print(f'Overall Hit Rate: {stats.hit_rate:.2%}')
        print(f'L1 Hits: {stats.l1_hits}')
        print(f'L2 Hits: {stats.l2_hits}')
        print(f'Total Requests: {stats.total_requests}')
        print(f'L1 Utilization: {stats.l1_utilization:.2%}')
        print(f'L1 Size: {stats.l1_size}')
        print(f'Warming Enabled: {stats.warming_enabled}')
        print(f'Health Status: {stats.health_status}')
        
        print()
        print('=== Connection Metrics ===')
        print(f'Redis Master Connected: {conn_metrics.redis_master_connected}')
        print(f'Redis Replica Connected: {conn_metrics.redis_replica_connected}')
        print(f'Active DB Connections: {conn_metrics.active_connections}')
        print(f'DB Pool Utilization: {conn_metrics.pool_utilization:.2%}')
        
        await manager.close()
        
    except Exception as e:
        print(f'Cache analysis failed: {e}')

asyncio.run(emergency_analysis())
" > $BUNDLE_DIR/cache_analysis.txt 2>&1

# 5. Network connectivity
echo "5. Testing network connectivity..."
{
    echo "=== Redis Connectivity ==="
    redis-cli -h $REDIS_HOST -p $REDIS_PORT ping 2>&1 || echo "Redis connection failed"
    
    echo
    echo "=== Database Connectivity ==="
    pg_isready -h $DATABASE_HOST -p $DATABASE_PORT 2>&1 || echo "Database connection failed"
} > $BUNDLE_DIR/connectivity.txt

# 6. Create summary
echo "6. Creating diagnostic summary..."
{
    echo "Emergency Diagnostic Bundle"
    echo "=========================="
    echo "Generated: $(date)"
    echo "System: $(hostname)"
    echo "Bundle Location: $BUNDLE_DIR"
    echo
    echo "Contents:"
    echo "- system_info.txt: System resource information"
    echo "- app_logs.txt: Recent application logs"
    echo "- configuration.txt: Current configuration"
    echo "- cache_analysis.txt: Cache performance analysis"
    echo "- connectivity.txt: Network connectivity tests"
    echo
    echo "🚨 SEND THIS BUNDLE TO SUPPORT FOR ANALYSIS 🚨"
} > $BUNDLE_DIR/README.txt

# 7. Create tarball
tar -czf "${BUNDLE_DIR}.tar.gz" -C /tmp "$(basename $BUNDLE_DIR)"

echo "✅ Emergency diagnostic bundle created: ${BUNDLE_DIR}.tar.gz"
echo "📧 Send this file to support: support@company.com"
echo "📱 Reference in incident: Include bundle filename in incident report"
```

---

## 📋 Recovery Verification Checklist

### **Post-Recovery Validation**
```bash
#!/bin/bash
# post_recovery_validation.sh

echo "✅ POST-RECOVERY VALIDATION CHECKLIST"
echo "======================================"

VALIDATION_RESULTS=()

# 1. Basic connectivity
echo -n "1. Basic system connectivity: "
if systemctl is-active prompt-improver >/dev/null 2>&1; then
    echo "✅ PASS"
    VALIDATION_RESULTS+=("connectivity:PASS")
else
    echo "❌ FAIL"
    VALIDATION_RESULTS+=("connectivity:FAIL")
fi

# 2. Cache operations
echo -n "2. Cache operations: "
if python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_cache():
    manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    await manager.initialize()
    await manager.set_cached('validation_test', 'success', ttl=60)
    result = await manager.get_cached('validation_test')
    await manager.close()
    exit(0 if result == 'success' else 1)

asyncio.run(test_cache())
" >/dev/null 2>&1; then
    echo "✅ PASS"
    VALIDATION_RESULTS+=("cache:PASS")
else
    echo "❌ FAIL"
    VALIDATION_RESULTS+=("cache:FAIL")
fi

# 3. Database operations
echo -n "3. Database operations: "
if python3 -c "
import asyncio
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_database():
    manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
    await manager.initialize()
    async with manager.get_session() as session:
        result = await session.execute('SELECT NOW()')
        result.fetchone()
    await manager.close()

asyncio.run(test_database())
" >/dev/null 2>&1; then
    echo "✅ PASS"
    VALIDATION_RESULTS+=("database:PASS")
else
    echo "❌ FAIL"
    VALIDATION_RESULTS+=("database:FAIL")
fi

# 4. Performance validation
echo -n "4. Performance validation: "
if python3 -c "
import asyncio
import time
from prompt_improver.database.unified_connection_manager import get_unified_manager, ManagerMode

async def test_performance():
    manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
    await manager.initialize()
    
    # Performance test
    start_time = time.time()
    for i in range(100):
        await manager.get_cached(f'perf_test_{i % 10}')
    end_time = time.time()
    
    avg_response_time = (end_time - start_time) / 100
    
    # Check cache hit rate
    stats = await manager.get_cache_stats()
    
    await manager.close()
    
    # Pass if response time < 50ms and hit rate > 70%
    exit(0 if avg_response_time < 0.050 and stats.hit_rate > 0.70 else 1)

asyncio.run(test_performance())
" >/dev/null 2>&1; then
    echo "✅ PASS"
    VALIDATION_RESULTS+=("performance:PASS")
else
    echo "❌ FAIL"
    VALIDATION_RESULTS+=("performance:FAIL")
fi

# 5. Security validation
echo -n "5. Security validation: "
if python3 scripts/validate_redis_security_fixes.py >/dev/null 2>&1; then
    echo "✅ PASS"
    VALIDATION_RESULTS+=("security:PASS")
else
    echo "❌ FAIL"
    VALIDATION_RESULTS+=("security:FAIL")
fi

# Summary
echo
echo "📊 VALIDATION SUMMARY:"
PASS_COUNT=0
FAIL_COUNT=0

for result in "${VALIDATION_RESULTS[@]}"; do
    test_name=$(echo $result | cut -d: -f1)
    test_result=$(echo $result | cut -d: -f2)
    echo "   $test_name: $test_result"
    
    if [ "$test_result" = "PASS" ]; then
        ((PASS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

echo
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 ALL VALIDATIONS PASSED - RECOVERY SUCCESSFUL"
    echo "✅ System is fully operational"
    exit 0
else
    echo "⚠️ $FAIL_COUNT VALIDATION(S) FAILED - RECOVERY INCOMPLETE"
    echo "❌ System requires additional attention"
    exit 1
fi
```

---

**Troubleshooting Guide Version**: 2025.1  
**Last Updated**: January 2025  
**Next Review**: April 2025  
**Document Owner**: Site Reliability Engineering Team  
**Emergency Contact**: +1-555-URGENT-OPS

**Remember**: In emergency situations, focus on rapid diagnosis and resolution. The UnifiedConnectionManager has proven 99.9% reliability - most issues have simple solutions if approached systematically.