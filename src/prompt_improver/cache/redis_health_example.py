#!/usr/bin/env python3
"""
Redis Health Monitoring Example

This script demonstrates the comprehensive Redis health monitoring capabilities,
including memory analysis, performance tracking, keyspace analytics, and more.

Usage:
    python redis_health_example.py
"""

import asyncio
import json
import time
from datetime import datetime

from .redis_health import RedisHealthMonitor, get_redis_health_summary

async def demonstrate_basic_monitoring():
    """Demonstrate basic Redis health monitoring."""
    print("=" * 60)
    print("BASIC REDIS HEALTH MONITORING")
    print("=" * 60)
    
    # Get quick health summary
    print("\n1. Quick Health Summary:")
    print("-" * 30)
    
    summary = await get_redis_health_summary()
    print(json.dumps(summary, indent=2))

async def demonstrate_comprehensive_monitoring():
    """Demonstrate comprehensive Redis health monitoring."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REDIS HEALTH MONITORING")
    print("=" * 60)
    
    monitor = RedisHealthMonitor()
    
    # Collect all metrics
    print("\n2. Comprehensive Health Check:")
    print("-" * 35)
    
    start_time = time.time()
    metrics = await monitor.collect_all_metrics()
    duration = time.time() - start_time
    
    print(f"✓ Health check completed in {duration:.2f} seconds")
    print(f"✓ Overall status: {metrics['status'].upper()}")
    print(f"✓ Total checks performed: {metrics.get('check_count', 0)}")
    
    # Display key metrics
    print(f"\n📊 Key Metrics Summary:")
    
    # Memory metrics
    if 'memory' in metrics:
        memory = metrics['memory']
        print(f"  Memory Usage: {memory.get('used_memory_mb', 0):.1f}MB")
        print(f"  Fragmentation: {memory.get('fragmentation_ratio', 1.0):.2f}")
        print(f"  Peak Memory: {memory.get('peak_memory_mb', 0):.1f}MB")
        if memory.get('memory_usage_percentage'):
            print(f"  Usage Percentage: {memory['memory_usage_percentage']:.1f}%")
    
    # Performance metrics
    if 'performance' in metrics:
        perf = metrics['performance']
        print(f"  Hit Rate: {perf.get('hit_rate_percentage', 0):.1f}%")
        print(f"  Ops/Sec: {perf.get('ops_per_sec', 0):.1f}")
        if 'latency' in perf:
            latency = perf['latency']
            print(f"  P95 Latency: {latency.get('p95_ms', 0):.1f}ms")
    
    # Connection metrics
    if 'connections' in metrics:
        conn = metrics['connections']
        print(f"  Connected Clients: {conn.get('connected_clients', 0)}")
        print(f"  Connection Utilization: {conn.get('connection_utilization_percentage', 0):.1f}%")
    
    # Recommendations
    if 'recommendations' in metrics and metrics['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(metrics['recommendations'][:3], 1):  # Show top 3
            print(f"  {i}. {rec}")

async def demonstrate_specific_metrics():
    """Demonstrate specific metric collection."""
    print("\n" + "=" * 60)
    print("SPECIFIC METRIC ANALYSIS")
    print("=" * 60)
    
    monitor = RedisHealthMonitor()
    
    # Memory analysis
    print("\n3. Memory Analysis:")
    print("-" * 20)
    
    await monitor._collect_memory_metrics()
    memory_data = monitor._memory_metrics_to_dict()
    
    print(f"  Used Memory: {memory_data.get('used_memory_human', 'N/A')}")
    print(f"  RSS Memory: {memory_data.get('used_memory_rss_mb', 0):.1f}MB")
    print(f"  Fragmentation Ratio: {memory_data.get('fragmentation_ratio', 1.0):.2f}")
    print(f"  Dataset Size: {memory_data.get('dataset_mb', 0):.1f}MB")
    print(f"  Overhead: {memory_data.get('overhead_mb', 0):.1f}MB")
    
    if memory_data.get('maxmemory_mb'):
        print(f"  Max Memory: {memory_data['maxmemory_mb']:.1f}MB")
        print(f"  Memory Policy: {memory_data.get('maxmemory_policy', 'N/A')}")
    
    # Performance analysis
    print("\n4. Performance Analysis:")
    print("-" * 25)
    
    # Generate some test operations for latency tracking
    test_key = "health_monitor_test"
    for i in range(10):
        start = time.time()
        await monitor.client.set(test_key, f"test_value_{i}", ex=60)
        await monitor.client.get(test_key)
        latency_ms = (time.time() - start) * 1000
        monitor.performance_metrics.add_latency_sample(latency_ms)
    
    await monitor.client.delete([test_key])  # Cleanup
    
    await monitor._collect_performance_metrics()
    perf_data = monitor._performance_metrics_to_dict()
    
    print(f"  Instantaneous Ops/Sec: {perf_data.get('instantaneous_ops_per_sec', 0)}")
    print(f"  Total Commands: {perf_data.get('total_commands', 0):,}")
    print(f"  Cache Hit Rate: {perf_data.get('hit_rate_percentage', 0):.1f}%")
    print(f"  Expired Keys: {perf_data.get('expired_keys', 0):,}")
    print(f"  Evicted Keys: {perf_data.get('evicted_keys', 0):,}")
    
    if 'latency' in perf_data:
        latency = perf_data['latency']
        print(f"  Average Latency: {latency.get('avg_ms', 0):.2f}ms")
        print(f"  P95 Latency: {latency.get('p95_ms', 0):.2f}ms")
        print(f"  P99 Latency: {latency.get('p99_ms', 0):.2f}ms")

async def demonstrate_keyspace_analysis():
    """Demonstrate keyspace analysis capabilities."""
    print("\n" + "=" * 60)
    print("KEYSPACE ANALYSIS")
    print("=" * 60)
    
    monitor = RedisHealthMonitor()
    
    # Create some test data for analysis
    print("\n5. Creating Test Data for Analysis:")
    print("-" * 40)
    
    test_keys = [
        ("user:123:profile", "User profile data" * 100),
        ("session:abc123", "Session data" * 50),
        ("cache:expensive_query", "Query result" * 200),
        ("queue:email_notifications", "Email queue item" * 10),
        ("lock:resource_123", "Lock data"),
        ("analytics:2024:01:15", "Analytics data" * 75),
    ]
    
    # Set test keys with expiration
    for key, value in test_keys:
        await monitor.client.set(key, value, ex=300)  # 5 minute expiration
    
    print(f"✓ Created {len(test_keys)} test keys")
    
    # Analyze keyspace
    await monitor._collect_keyspace_metrics()
    keyspace_data = monitor._keyspace_metrics_to_dict()
    
    print(f"\n📊 Keyspace Analysis Results:")
    print(f"  Total Keys: {keyspace_data.get('total_keys', 0):,}")
    
    if 'databases' in keyspace_data:
        for db_name, db_info in keyspace_data['databases'].items():
            print(f"  {db_name.upper()}:")
            print(f"    Keys: {db_info.get('keys', 0):,}")
            print(f"    Expires: {db_info.get('expires', 0):,}")
            print(f"    Expiry Ratio: {db_info.get('expiry_ratio_percentage', 0):.1f}%")
            if db_info.get('avg_ttl', 0) > 0:
                print(f"    Average TTL: {db_info['avg_ttl'] / 1000:.1f}s")
    
    if 'key_patterns' in keyspace_data and keyspace_data['key_patterns']:
        print(f"\n🔍 Key Patterns Found:")
        for pattern, info in list(keyspace_data['key_patterns'].items())[:5]:
            print(f"  {pattern}: {info.get('count', 0)} keys, avg {info.get('avg_memory_bytes', 0)} bytes")
    
    if 'large_keys' in keyspace_data and keyspace_data['large_keys']:
        print(f"\n🔍 Large Keys Detected:")
        for key_info in keyspace_data['large_keys'][:3]:
            print(f"  {key_info.get('key', 'N/A')}: {key_info.get('size_mb', 0):.2f}MB")
    
    # Cleanup test keys
    test_key_names = [key for key, _ in test_keys]
    await monitor.client.delete(test_key_names)
    print(f"\n✓ Cleaned up {len(test_key_names)} test keys")

async def demonstrate_slowlog_analysis():
    """Demonstrate slow log analysis."""
    print("\n" + "=" * 60)
    print("SLOW LOG ANALYSIS")  
    print("=" * 60)
    
    monitor = RedisHealthMonitor()
    
    print("\n6. Slow Log Analysis:")
    print("-" * 25)
    
    # Generate some intentionally slow operations for demonstration
    # (In production, you'd analyze actual slow operations)
    large_data = "x" * 10000  # 10KB of data
    for i in range(3):
        await monitor.client.set(f"slow_test_{i}", large_data, ex=60)
        # Simulate processing time
        await asyncio.sleep(0.01)
    
    await monitor._collect_slowlog_metrics()
    slowlog_data = monitor._slowlog_metrics_to_dict()
    
    print(f"  Total Slow Log Entries: {slowlog_data.get('total_entries', 0)}")
    print(f"  Max Entries: {slowlog_data.get('max_entries', 0)}")
    print(f"  Recent Slow Commands: {slowlog_data.get('recent_slow_commands', 0)}")
    
    if slowlog_data.get('avg_duration_ms', 0) > 0:
        print(f"  Average Duration: {slowlog_data['avg_duration_ms']:.2f}ms")
        print(f"  Max Duration: {slowlog_data.get('max_duration_ms', 0):.2f}ms")
    
    if 'command_frequency' in slowlog_data and slowlog_data['command_frequency']:
        print(f"\n  Top Slow Commands:")
        for cmd, count in list(slowlog_data['command_frequency'].items())[:3]:
            print(f"    {cmd}: {count} times")
    
    if 'recent_entries' in slowlog_data and slowlog_data['recent_entries']:
        print(f"\n  Recent Slow Operations:")
        for entry in slowlog_data['recent_entries'][:2]:
            print(f"    {entry.get('command', 'N/A')} - {entry.get('duration_ms', 0):.2f}ms")
    
    # Cleanup
    await monitor.client.delete([f"slow_test_{i}" for i in range(3)])

async def demonstrate_replication_monitoring():
    """Demonstrate replication monitoring (if applicable)."""
    print("\n" + "=" * 60)
    print("REPLICATION MONITORING")
    print("=" * 60)
    
    monitor = RedisHealthMonitor()
    
    print("\n7. Replication Status:")
    print("-" * 25)
    
    await monitor._collect_replication_metrics()
    replication_data = monitor._replication_metrics_to_dict()
    
    print(f"  Role: {replication_data.get('role', 'unknown').upper()}")
    print(f"  Master Repl ID: {replication_data.get('master_replid', 'N/A')[:16]}...")
    print(f"  Master Offset: {replication_data.get('master_repl_offset', 0):,}")
    print(f"  Backlog Active: {replication_data.get('backlog_active', False)}")
    
    if 'master' in replication_data:
        master_info = replication_data['master']
        print(f"  Connected Slaves: {master_info.get('connected_slaves', 0)}")
        if master_info.get('slaves'):
            print(f"  Slave Details: {len(master_info['slaves'])} slaves connected")
    
    if 'slave' in replication_data:
        slave_info = replication_data['slave']
        print(f"  Master: {slave_info.get('master_host', 'N/A')}:{slave_info.get('master_port', 0)}")
        print(f"  Link Status: {slave_info.get('master_link_status', 'unknown')}")
        print(f"  Last IO: {slave_info.get('master_last_io_seconds_ago', 0)}s ago")
        print(f"  Replication Lag: {slave_info.get('replication_lag_seconds', 0):.2f}s")

async def main():
    """Main demonstration function."""
    print("🔍 Redis Health Monitoring Demonstration")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demonstrations
        await demonstrate_basic_monitoring()
        await demonstrate_comprehensive_monitoring()  
        await demonstrate_specific_metrics()
        await demonstrate_keyspace_analysis()
        await demonstrate_slowlog_analysis()
        await demonstrate_replication_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe Redis health monitoring system provides:")
        print("• Comprehensive memory usage tracking and fragmentation analysis")
        print("• Real-time performance metrics with latency percentiles")
        print("• Persistence health monitoring (RDB/AOF status)")
        print("• Replication status and lag monitoring")
        print("• Client connection analysis and utilization tracking")
        print("• Keyspace analytics with memory profiling")
        print("• Slow log analysis for performance optimization")
        print("• Automated recommendations for performance tuning")
        print("\nIntegrated with existing health endpoints for production monitoring.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure Redis is running and accessible.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())