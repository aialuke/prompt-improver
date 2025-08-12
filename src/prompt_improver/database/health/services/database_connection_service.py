"""Database Connection Health Monitoring Service.

Provides comprehensive connection pool monitoring, connection lifecycle management,
and connection health assessment. This service focuses solely on connection-related
health metrics and analysis.

Features:
- Connection pool utilization tracking
- Connection age and lifecycle monitoring  
- Connection state analysis
- Problematic connection identification
- Pool efficiency scoring
- Connection-specific recommendations
"""

import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from prompt_improver.core.common import get_logger
from prompt_improver.repositories.protocols.session_manager_protocol import SessionManagerProtocol
from .health_protocols import DatabaseConnectionServiceProtocol
from .health_types import ConnectionHealthMetrics

logger = get_logger(__name__)


class DatabaseConnectionService:
    """Service for database connection health monitoring and assessment.
    
    This service provides comprehensive monitoring of database connections,
    including pool utilization, connection lifecycle, and health assessment.
    """
    
    def __init__(self, session_manager: SessionManagerProtocol):
        """Initialize the connection service.
        
        Args:
            session_manager: Database session manager for executing queries
        """
        self.session_manager = session_manager
        
        # Configuration thresholds
        self.max_connection_lifetime_seconds = 1800  # 30 minutes
        self.long_query_threshold_seconds = 300      # 5 minutes
        self.utilization_warning_threshold = 80.0    # 80%
        self.utilization_critical_threshold = 95.0   # 95%
    
    async def collect_connection_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive connection pool metrics.
        
        Returns:
            Dictionary containing detailed connection metrics
        """
        logger.debug("Collecting database connection pool metrics")
        start_time = time.perf_counter()
        
        try:
            # Get connection info from session manager
            connection_info = await self.session_manager.get_connection_info()
            
            # Get detailed connection information
            connection_details = await self.get_connection_details()
            
            # Analyze connection patterns
            age_stats = self.analyze_connection_ages(connection_details)
            state_stats = self.analyze_connection_states(connection_details)
            
            # Calculate utilization metrics
            current_size = connection_info.get("pool_size", 0)
            active_count = state_stats["active"]
            utilization = active_count / current_size * 100 if current_size > 0 else 0
            
            # Build comprehensive metrics
            metrics = ConnectionHealthMetrics(
                pool_configuration={
                    "min_size": connection_info.get("pool_min_size", 0),
                    "max_size": connection_info.get("pool_max_size", 0),
                    "current_size": current_size,
                    "timeout_seconds": connection_info.get("pool_timeout", 0),
                    "max_lifetime_seconds": connection_info.get("pool_max_lifetime", 0),
                },
                connection_states={
                    "active": state_stats["active"],
                    "idle": state_stats["idle"],
                    "idle_in_transaction": state_stats["idle_in_transaction"],
                    "idle_in_transaction_aborted": state_stats["idle_in_transaction_aborted"],
                    "fastpath_function_call": state_stats["fastpath_function_call"],
                    "disabled": state_stats["disabled"],
                },
                utilization_metrics={
                    "utilization_percent": utilization,
                    "available_connections": current_size - active_count,
                    "waiting_requests": connection_info.get("requests_waiting", 0),
                    "pool_efficiency_score": self._calculate_efficiency_score(
                        state_stats, age_stats
                    ),
                },
                age_statistics=age_stats,
                connection_details=connection_details[:10],  # Limit for performance
                health_indicators={
                    "connections_over_max_lifetime": age_stats["over_max_lifetime_count"],
                    "long_running_queries": len([
                        c for c in connection_details
                        if c.get("query_duration_seconds", 0) > self.long_query_threshold_seconds
                    ]),
                    "blocked_connections": len([
                        c for c in connection_details
                        if c.get("wait_event_type") and "Lock" in c["wait_event_type"]
                    ]),
                    "problematic_connections": self.identify_problematic_connections(
                        connection_details
                    ),
                },
                recommendations=self._generate_pool_recommendations(
                    utilization, age_stats, state_stats
                ),
                collection_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
            
            return self._metrics_to_dict(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect connection pool metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
                "collection_time_ms": (time.perf_counter() - start_time) * 1000,
            }
    
    async def get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed connection information from pg_stat_activity.
        
        Returns:
            List of connection details with timing and state information
        """
        async with self.session_manager.session_context() as session:
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
            rows = result.fetchall()
            
            connections = []
            for row in rows:
                connection_info = {
                    "connection_id": str(row[0]),
                    "pid": row[0],
                    "backend_start": row[6],
                    "query_start": row[8],
                    "state": row[10],
                    "application_name": row[2],
                    "client_addr": str(row[3]) if row[3] else None,
                    "current_query": row[13],
                    "wait_event_type": row[15],
                    "wait_event": row[16],
                }
                
                # Calculate connection age
                if row[6]:
                    connection_info["age_seconds"] = (
                        datetime.now(UTC) - row[6].replace(tzinfo=UTC)
                    ).total_seconds()
                else:
                    connection_info["age_seconds"] = 0.0
                
                # Calculate query duration
                if row[8]:
                    connection_info["query_duration_seconds"] = (
                        datetime.now(UTC) - row[8].replace(tzinfo=UTC)
                    ).total_seconds()
                else:
                    connection_info["query_duration_seconds"] = 0.0
                
                connections.append(connection_info)
            
            return connections
    
    def analyze_connection_ages(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze connection age distribution and identify old connections.
        
        Args:
            connections: List of connection details
            
        Returns:
            Dictionary with age analysis statistics
        """
        if not connections:
            return {
                "total_connections": 0,
                "average_age_seconds": 0.0,
                "max_age_seconds": 0.0,
                "min_age_seconds": 0.0,
                "over_max_lifetime_count": 0,
                "age_distribution": {},
            }
        
        ages = [conn["age_seconds"] for conn in connections]
        over_max_lifetime = [
            age for age in ages if age > self.max_connection_lifetime_seconds
        ]
        
        # Age distribution buckets
        distribution = {
            "0-5min": len([age for age in ages if age <= 300]),
            "5-30min": len([age for age in ages if 300 < age <= 1800]),
            "30min-2h": len([age for age in ages if 1800 < age <= 7200]),
            "2h+": len([age for age in ages if age > 7200]),
        }
        
        return {
            "total_connections": len(connections),
            "average_age_seconds": sum(ages) / len(ages),
            "max_age_seconds": max(ages),
            "min_age_seconds": min(ages),
            "over_max_lifetime_count": len(over_max_lifetime),
            "age_distribution": distribution,
        }
    
    def analyze_connection_states(self, connections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze connection state distribution.
        
        Args:
            connections: List of connection details
            
        Returns:
            Dictionary with connection state counts
        """
        states = {
            "active": 0,
            "idle": 0,
            "idle_in_transaction": 0,
            "idle_in_transaction_aborted": 0,
            "fastpath_function_call": 0,
            "disabled": 0,
        }
        
        for conn in connections:
            state = conn.get("state", "unknown")
            if state in states:
                states[state] += 1
            else:
                states["disabled"] += 1
        
        return states
    
    def identify_problematic_connections(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify connections that may be problematic.
        
        Args:
            connections: List of connection details
            
        Returns:
            List of problematic connections with issue descriptions
        """
        problematic = []
        
        for conn in connections:
            issues = []
            
            # Check for long-running queries
            if conn["query_duration_seconds"] > self.long_query_threshold_seconds:
                issues.append(
                    f"Long-running query: {conn['query_duration_seconds']:.1f}s"
                )
            
            # Check for old connections
            if conn["age_seconds"] > self.max_connection_lifetime_seconds:
                issues.append(f"Old connection: {conn['age_seconds'] / 3600:.1f}h")
            
            # Check for blocking
            if conn.get("wait_event_type") and "Lock" in conn["wait_event_type"]:
                issues.append(f"Blocked: {conn['wait_event']}")
            
            # Check for idle in transaction
            if conn.get("state") == "idle_in_transaction":
                issues.append("Idle in transaction")
            
            if issues:
                problematic.append({
                    "pid": conn["pid"],
                    "application_name": conn["application_name"],
                    "client_addr": conn["client_addr"],
                    "issues": issues,
                    "query": conn["current_query"][:100] if conn["current_query"] else None,
                })
        
        return problematic
    
    async def get_pool_health_summary(self) -> Dict[str, Any]:
        """Get connection pool health summary with status assessment.
        
        Returns:
            Dictionary with health status and key metrics
        """
        try:
            metrics_dict = await self.collect_connection_metrics()
            
            # Extract key metrics for assessment
            utilization = metrics_dict.get("utilization_metrics", {}).get(
                "utilization_percent", 0
            )
            efficiency_score = metrics_dict.get("utilization_metrics", {}).get(
                "pool_efficiency_score", 0
            )
            problematic_count = len(
                metrics_dict.get("health_indicators", {}).get("problematic_connections", [])
            )
            
            # Determine overall status
            if (
                utilization > self.utilization_critical_threshold
                or efficiency_score < 50
            ):
                status = "critical"
            elif (
                utilization > self.utilization_warning_threshold
                or efficiency_score < 70
                or problematic_count > 0
            ):
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "utilization_percent": utilization,
                "efficiency_score": efficiency_score,
                "total_connections": metrics_dict.get("age_statistics", {}).get(
                    "total_connections", 0
                ),
                "problematic_connections_count": problematic_count,
                "recommendations": metrics_dict.get("recommendations", []),
                "summary": f"Pool utilization: {utilization:.1f}%, Efficiency: {efficiency_score:.1f}/100",
                "detailed_metrics": metrics_dict,
            }
            
        except Exception as e:
            logger.error(f"Failed to get connection pool health summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "summary": "Connection pool health check failed",
            }
    
    def _calculate_efficiency_score(
        self, state_stats: Dict[str, int], age_stats: Dict[str, Any]
    ) -> float:
        """Calculate pool efficiency score (0-100).
        
        Args:
            state_stats: Connection state distribution
            age_stats: Connection age statistics
            
        Returns:
            Pool efficiency score between 0 and 100
        """
        total_connections = sum(state_stats.values())
        if total_connections == 0:
            return 100.0
        
        # Calculate ratios
        active_ratio = state_stats["active"] / total_connections
        idle_ratio = state_stats["idle"] / total_connections
        idle_in_transaction_ratio = state_stats["idle_in_transaction"] / total_connections
        long_lived_ratio = age_stats["over_max_lifetime_count"] / total_connections
        
        # Weighted efficiency score
        efficiency = (
            active_ratio * 40          # Active connections are good
            + idle_ratio * 30          # Some idle connections are normal
            - idle_in_transaction_ratio * 30  # Idle in transaction is bad
            - long_lived_ratio * 40    # Long-lived connections are problematic
        ) * 100
        
        return max(0.0, min(100.0, efficiency))
    
    def _generate_pool_recommendations(
        self, utilization: float, age_stats: Dict[str, Any], state_stats: Dict[str, int]
    ) -> List[str]:
        """Generate connection pool optimization recommendations.
        
        Args:
            utilization: Pool utilization percentage
            age_stats: Connection age statistics
            state_stats: Connection state distribution
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Utilization recommendations
        if utilization > self.utilization_critical_threshold:
            recommendations.append(
                f"CRITICAL: Pool utilization at {utilization:.1f}% - consider increasing pool size"
            )
        elif utilization > self.utilization_warning_threshold:
            recommendations.append(
                f"WARNING: Pool utilization at {utilization:.1f}% - monitor closely"
            )
        
        # Age-based recommendations
        if age_stats["over_max_lifetime_count"] > 0:
            recommendations.append(
                f"Found {age_stats['over_max_lifetime_count']} connections over max lifetime - consider connection recycling"
            )
        
        # State-based recommendations
        if state_stats["idle_in_transaction"] > 0:
            recommendations.append(
                f"Found {state_stats['idle_in_transaction']} idle-in-transaction connections - check application transaction handling"
            )
        
        # Efficiency recommendations
        if utilization < 10 and sum(state_stats.values()) > 5:
            recommendations.append(
                f"Low utilization ({utilization:.1f}%) - consider reducing pool size"
            )
        
        return recommendations
    
    def _metrics_to_dict(self, metrics: ConnectionHealthMetrics) -> Dict[str, Any]:
        """Convert ConnectionHealthMetrics dataclass to dictionary.
        
        Args:
            metrics: Connection health metrics dataclass
            
        Returns:
            Dictionary representation of metrics
        """
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "pool_configuration": metrics.pool_configuration,
            "connection_states": metrics.connection_states,
            "utilization_metrics": metrics.utilization_metrics,
            "age_statistics": metrics.age_statistics,
            "connection_details": metrics.connection_details,
            "health_indicators": metrics.health_indicators,
            "recommendations": metrics.recommendations,
            "collection_time_ms": metrics.collection_time_ms,
        }