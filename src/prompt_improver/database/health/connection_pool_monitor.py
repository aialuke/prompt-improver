"""
Connection Pool Monitor with Real psycopg3 Metrics

Provides detailed connection pool monitoring including:
- Active/idle/waiting connections with age tracking
- Connection lifecycle management
- Pool utilization analysis
- Connection health assessment
- Pool optimization recommendations
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

# psycopg_client removed in Phase 1 - using unified_connection_manager instead
from ..unified_connection_manager import get_unified_manager, ManagerMode

logger = logging.getLogger(__name__)

@dataclass
class ConnectionInfo:
    """Individual connection information"""
    connection_id: str
    pid: Optional[int] = None
    backend_start: Optional[datetime] = None
    query_start: Optional[datetime] = None  
    state: Optional[str] = None
    application_name: Optional[str] = None
    client_addr: Optional[str] = None
    current_query: Optional[str] = None
    wait_event_type: Optional[str] = None
    wait_event: Optional[str] = None
    
    @property
    def age_seconds(self) -> float:
        """Connection age in seconds"""
        if self.backend_start:
            return (datetime.now(timezone.utc) - self.backend_start.replace(tzinfo=timezone.utc)).total_seconds()
        return 0.0
    
    @property
    def query_duration_seconds(self) -> float:
        """Current query duration in seconds"""
        if self.query_start:
            return (datetime.now(timezone.utc) - self.query_start.replace(tzinfo=timezone.utc)).total_seconds()
        return 0.0

@dataclass
class ConnectionPoolMetrics:
    """Comprehensive connection pool metrics"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Pool size metrics
    min_pool_size: int = 0
    max_pool_size: int = 0
    current_pool_size: int = 0
    
    # Connection state counts
    active_connections: int = 0
    idle_connections: int = 0
    idle_in_transaction_connections: int = 0
    waiting_connections: int = 0
    
    # Pool utilization
    utilization_percent: float = 0.0
    
    # Connection age statistics
    avg_connection_age_seconds: float = 0.0
    max_connection_age_seconds: float = 0.0
    min_connection_age_seconds: float = 0.0
    
    # Connection details
    connections: List[ConnectionInfo] = field(default_factory=list)
    
    # Pool health indicators
    connections_over_max_lifetime: int = 0
    long_running_queries: int = 0
    blocked_connections: int = 0
    
    # Performance metrics
    pool_efficiency_score: float = 100.0
    recommendations: List[str] = field(default_factory=list)

class ConnectionPoolMonitor:
    """
    Monitor PostgreSQL connection pool using psycopg3 pool introspection
    and pg_stat_activity for detailed connection analysis.
    """
    
    def __init__(self, client: Optional[Any] = None):
        self.client = client
        
        # Configuration thresholds
        self.max_connection_lifetime_seconds = 1800  # 30 minutes
        self.long_query_threshold_seconds = 300      # 5 minutes
        self.utilization_warning_threshold = 80.0    # 80%
        self.utilization_critical_threshold = 95.0   # 95%
    
    async def get_client(self):
        """Get database client"""
        if self.client is None:
            return get_unified_manager(ManagerMode.ASYNC_MODERN)
        return self.client
    
    async def collect_connection_pool_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive connection pool metrics from PostgreSQL
        """
        logger.debug("Collecting connection pool metrics")
        start_time = time.perf_counter()
        
        try:
            # Get pool stats from psycopg client
            client = await self.get_client()
            pool_stats = await client.get_pool_stats()
            
            # Get detailed connection information from PostgreSQL
            connection_details = await self._get_connection_details()
            
            # Analyze connection ages and states
            age_stats = self._analyze_connection_ages(connection_details)
            state_stats = self._analyze_connection_states(connection_details)
            
            # Calculate pool utilization
            current_size = pool_stats.get("pool_size", 0)
            active_count = state_stats["active"]
            utilization = (active_count / current_size * 100) if current_size > 0 else 0
            
            # Build comprehensive metrics
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pool_configuration": {
                    "min_size": pool_stats.get("pool_min_size", 0),
                    "max_size": pool_stats.get("pool_max_size", 0),
                    "current_size": current_size,
                    "timeout_seconds": pool_stats.get("pool_timeout", 0),
                    "max_lifetime_seconds": pool_stats.get("pool_max_lifetime", 0)
                },
                "connection_states": {
                    "active": state_stats["active"],
                    "idle": state_stats["idle"],
                    "idle_in_transaction": state_stats["idle_in_transaction"],
                    "idle_in_transaction_aborted": state_stats["idle_in_transaction_aborted"],
                    "fastpath_function_call": state_stats["fastpath_function_call"],
                    "disabled": state_stats["disabled"]
                },
                "utilization_metrics": {
                    "utilization_percent": utilization,
                    "available_connections": current_size - active_count,
                    "waiting_requests": pool_stats.get("requests_waiting", 0),
                    "pool_efficiency_score": self._calculate_efficiency_score(state_stats, age_stats)
                },
                "age_statistics": age_stats,
                "connection_details": connection_details[:10],  # Limit to first 10 for response size
                "health_indicators": {
                    "connections_over_max_lifetime": age_stats["over_max_lifetime_count"],
                    "long_running_queries": len([c for c in connection_details if c.query_duration_seconds > self.long_query_threshold_seconds]),
                    "blocked_connections": len([c for c in connection_details if c.wait_event_type and "Lock" in c.wait_event_type]),
                    "problematic_connections": self._identify_problematic_connections(connection_details)
                },
                "recommendations": self._generate_pool_recommendations(utilization, age_stats, state_stats)
            }
            
            collection_time = (time.perf_counter() - start_time) * 1000
            metrics["collection_time_ms"] = round(collection_time, 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect connection pool metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_time_ms": (time.perf_counter() - start_time) * 1000
            }
    
    async def _get_connection_details(self) -> List[ConnectionInfo]:
        """
        Get detailed connection information from pg_stat_activity
        """
        async with get_session_context() as session:
            query = text("""
                SELECT 
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    client_hostname,
                    client_port,
                    backend_start,
                    xact_start,
                    query_start,
                    state_change,
                    state,
                    backend_xid,
                    backend_xmin,
                    query,
                    backend_type,
                    wait_event_type,
                    wait_event
                FROM pg_stat_activity 
                WHERE backend_type = 'client backend'
                    AND pid IS NOT NULL
                ORDER BY backend_start
            """)
            
            result = await session.execute(query)
            connections = []
            
            for row in result:
                conn = ConnectionInfo(
                    connection_id=f"pid_{row[0]}",
                    pid=row[0],
                    backend_start=row[6],
                    query_start=row[8],
                    state=row[10],
                    application_name=row[2],
                    client_addr=str(row[3]) if row[3] else None,
                    current_query=row[13][:200] + "..." if row[13] and len(row[13]) > 200 else row[13],
                    wait_event_type=row[15],
                    wait_event=row[16]
                )
                connections.append(conn)
            
            return connections
    
    def _analyze_connection_ages(self, connections: List[ConnectionInfo]) -> Dict[str, Any]:
        """
        Analyze connection age statistics
        """
        if not connections:
            return {
                "total_connections": 0,
                "avg_age_seconds": 0,
                "max_age_seconds": 0,
                "min_age_seconds": 0,
                "over_max_lifetime_count": 0,
                "age_distribution": {}
            }
        
        ages = [conn.age_seconds for conn in connections]
        over_max_lifetime = len([age for age in ages if age > self.max_connection_lifetime_seconds])
        
        # Age distribution buckets
        age_buckets = {
            "0-60s": len([age for age in ages if 0 <= age < 60]),
            "1-5min": len([age for age in ages if 60 <= age < 300]),
            "5-30min": len([age for age in ages if 300 <= age < 1800]),
            "30min+": len([age for age in ages if age >= 1800])
        }
        
        return {
            "total_connections": len(connections),
            "avg_age_seconds": sum(ages) / len(ages),
            "max_age_seconds": max(ages),
            "min_age_seconds": min(ages),
            "over_max_lifetime_count": over_max_lifetime,
            "age_distribution": age_buckets,
            "oldest_connections": sorted(
                [(c.connection_id, c.age_seconds, c.state) for c in connections], 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def _analyze_connection_states(self, connections: List[ConnectionInfo]) -> Dict[str, int]:
        """
        Analyze connection state distribution
        """
        states = {
            "active": 0,
            "idle": 0,
            "idle_in_transaction": 0,
            "idle_in_transaction_aborted": 0,
            "fastpath_function_call": 0,
            "disabled": 0
        }
        
        for conn in connections:
            state = conn.state or "unknown"
            if state in states:
                states[state] += 1
            else:
                states["unknown"] = states.get("unknown", 0) + 1
        
        return states
    
    def _calculate_efficiency_score(self, state_stats: Dict[str, int], age_stats: Dict[str, Any]) -> float:
        """
        Calculate pool efficiency score (0-100)
        """
        score = 100.0
        total_connections = sum(state_stats.values())
        
        if total_connections == 0:
            return score
        
        # Penalize idle in transaction connections (they hold locks)
        idle_in_txn_ratio = state_stats["idle_in_transaction"] / total_connections
        score -= idle_in_txn_ratio * 30  # Up to 30 point penalty
        
        # Penalize connections over max lifetime
        over_lifetime_ratio = age_stats["over_max_lifetime_count"] / total_connections
        score -= over_lifetime_ratio * 20  # Up to 20 point penalty
        
        # Penalize low utilization (too many idle connections)
        idle_ratio = state_stats["idle"] / total_connections
        if idle_ratio > 0.7:  # More than 70% idle
            score -= (idle_ratio - 0.7) * 50  # Penalty for over-provisioning
        
        return max(0.0, min(100.0, score))
    
    def _identify_problematic_connections(self, connections: List[ConnectionInfo]) -> List[Dict[str, Any]]:
        """
        Identify connections that may be causing issues
        """
        problematic = []
        
        for conn in connections:
            issues = []
            
            # Long-running queries
            if conn.query_duration_seconds > self.long_query_threshold_seconds:
                issues.append(f"Long-running query ({conn.query_duration_seconds:.1f}s)")
            
            # Old connections
            if conn.age_seconds > self.max_connection_lifetime_seconds:
                issues.append(f"Connection exceeds max lifetime ({conn.age_seconds:.1f}s)")
            
            # Idle in transaction
            if conn.state == "idle in transaction":
                issues.append("Idle in transaction (may hold locks)")
            
            # Waiting on locks
            if conn.wait_event_type and "Lock" in conn.wait_event_type:
                issues.append(f"Waiting on lock: {conn.wait_event}")
            
            if issues:
                problematic.append({
                    "connection_id": conn.connection_id,
                    "pid": conn.pid,
                    "application_name": conn.application_name,
                    "state": conn.state,
                    "age_seconds": conn.age_seconds,
                    "query_duration_seconds": conn.query_duration_seconds,
                    "issues": issues,
                    "current_query": conn.current_query
                })
        
        return problematic
    
    def _generate_pool_recommendations(self, utilization: float, age_stats: Dict[str, Any], state_stats: Dict[str, int]) -> List[str]:
        """
        Generate actionable recommendations for pool optimization
        """
        recommendations = []
        
        # Utilization-based recommendations
        if utilization > self.utilization_critical_threshold:
            recommendations.append(f"CRITICAL: Pool utilization very high ({utilization:.1f}%). Consider increasing max pool size immediately.")
        elif utilization > self.utilization_warning_threshold:
            recommendations.append(f"WARNING: Pool utilization high ({utilization:.1f}%). Consider increasing max pool size.")
        elif utilization < 30:
            recommendations.append(f"INFO: Pool utilization low ({utilization:.1f}%). Consider reducing min pool size to save resources.")
        
        # Age-based recommendations
        over_lifetime_count = age_stats["over_max_lifetime_count"]
        if over_lifetime_count > 0:
            recommendations.append(f"Consider reducing max_lifetime setting. {over_lifetime_count} connections exceed current max lifetime.")
        
        if age_stats["avg_age_seconds"] > self.max_connection_lifetime_seconds:
            recommendations.append("Average connection age exceeds recommended lifetime. Enable connection recycling.")
        
        # State-based recommendations
        total_connections = sum(state_stats.values())
        if total_connections > 0:
            idle_in_txn_percent = (state_stats["idle_in_transaction"] / total_connections) * 100
            if idle_in_txn_percent > 10:
                recommendations.append(f"High percentage of idle-in-transaction connections ({idle_in_txn_percent:.1f}%). Review application transaction handling.")
            
            idle_percent = (state_stats["idle"] / total_connections) * 100
            if idle_percent > 70:
                recommendations.append(f"High percentage of idle connections ({idle_percent:.1f}%). Consider reducing pool size or implementing connection multiplexing.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Pool appears healthy. Continue monitoring for optimal performance.")
        
        return recommendations
    
    async def get_connection_pool_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of connection pool health
        """
        metrics = await self.collect_connection_pool_metrics()
        
        if "error" in metrics:
            return {
                "status": "error",
                "message": "Failed to collect pool metrics",
                "error": metrics["error"]
            }
        
        utilization = metrics["utilization_metrics"]["utilization_percent"]
        efficiency_score = metrics["utilization_metrics"]["pool_efficiency_score"]
        problematic_count = len(metrics["health_indicators"]["problematic_connections"])
        
        # Determine overall health status
        if utilization > 95 or efficiency_score < 60 or problematic_count > 5:
            status = "critical"
        elif utilization > 80 or efficiency_score < 80 or problematic_count > 2:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "utilization_percent": utilization,
            "efficiency_score": efficiency_score,
            "total_connections": metrics["pool_configuration"]["current_size"],
            "active_connections": metrics["connection_states"]["active"],
            "problematic_connections": problematic_count,
            "key_recommendations": metrics["recommendations"][:3],  # Top 3 recommendations
            "timestamp": metrics["timestamp"]
        }
    
    async def monitor_connection_lifecycle(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Monitor connection lifecycle over a specified duration
        """
        logger.info(f"Starting connection lifecycle monitoring for {duration_seconds} seconds")
        
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration_seconds:
            try:
                metrics = await self.collect_connection_pool_metrics()
                if "error" not in metrics:
                    samples.append({
                        "timestamp": time.time(),
                        "utilization": metrics["utilization_metrics"]["utilization_percent"],
                        "active_connections": metrics["connection_states"]["active"],
                        "total_connections": metrics["pool_configuration"]["current_size"]
                    })
                
                await asyncio.sleep(10)  # Sample every 10 seconds
                
            except Exception as e:
                logger.error(f"Error during lifecycle monitoring: {e}")
                break
        
        if not samples:
            return {"error": "No samples collected during monitoring period"}
        
        # Analyze lifecycle patterns
        utilizations = [s["utilization"] for s in samples]
        active_counts = [s["active_connections"] for s in samples]
        
        return {
            "monitoring_duration_seconds": duration_seconds,
            "samples_collected": len(samples),
            "utilization_stats": {
                "min": min(utilizations),
                "max": max(utilizations),
                "avg": sum(utilizations) / len(utilizations),
                "trend": "increasing" if utilizations[-1] > utilizations[0] else "decreasing" if utilizations[-1] < utilizations[0] else "stable"
            },
            "active_connection_stats": {
                "min": min(active_counts),
                "max": max(active_counts),
                "avg": sum(active_counts) / len(active_counts),
                "volatility": max(active_counts) - min(active_counts)
            },
            "samples": samples[-20:] if len(samples) > 20 else samples  # Return last 20 samples
        }