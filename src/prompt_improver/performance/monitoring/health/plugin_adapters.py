"""
Plugin Adapters for Health Checkers

Converts existing health checkers to the unified plugin system:
- EnhancedMLServiceHealthChecker 
- AnalyticsServiceHealthChecker
- MLServiceHealthChecker
- DatabaseHealthChecker (5 database types)
- RedisHealthChecker (3 variants)
- API endpoint health checkers
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from .unified_health_system import (
    HealthCheckPlugin,
    HealthCheckCategory,
    HealthCheckPluginConfig
)
from ....core.protocols.health_protocol import HealthCheckResult, HealthStatus
from .base import HealthResult, HealthStatus as BaseHealthStatus

# Import existing health checkers
from .checkers import (
    DatabaseHealthChecker,
    AnalyticsServiceHealthChecker, 
    MLServiceHealthChecker,
    RedisHealthChecker,
    SystemResourcesHealthChecker,
    QueueHealthChecker,
    MCPServerHealthChecker
)
from .enhanced_checkers import (
    EnhancedMLServiceHealthChecker,
    EnhancedAnalyticsServiceHealthChecker
)
from .ml_specific_checkers import (
    MLModelHealthChecker,
    MLDataQualityChecker,
    MLTrainingHealthChecker,
    MLPerformanceHealthChecker
)
from .ml_orchestration_checkers import (
    MLOrchestratorHealthChecker,
    MLComponentRegistryHealthChecker,
    MLResourceManagerHealthChecker,
    MLWorkflowEngineHealthChecker,
    MLEventBusHealthChecker
)

# Import database health monitors
try:
    from ...database.health.database_health_monitor import DatabaseHealthMonitor
    from ...database.health.connection_pool_monitor import ConnectionPoolMonitor
    from ...database.health.query_performance_analyzer import QueryPerformanceAnalyzer
    from ...database.health.index_health_assessor import IndexHealthAssessor
    from ...database.health.table_bloat_detector import TableBloatDetector
    DATABASE_HEALTH_AVAILABLE = True
except ImportError:
    DATABASE_HEALTH_AVAILABLE = False

# Import Redis health monitor
try:
    from ...cache.redis_health import RedisHealthChecker as DetailedRedisHealthChecker
    REDIS_HEALTH_AVAILABLE = True
except ImportError:
    REDIS_HEALTH_AVAILABLE = False

# Import API health checkers
try:
    from ...api.health import HealthRouter
    API_HEALTH_AVAILABLE = True
except ImportError:
    API_HEALTH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _convert_health_status(status: BaseHealthStatus) -> HealthStatus:
    """Convert base health status to protocol health status"""
    if status == BaseHealthStatus.HEALTHY:
        return HealthStatus.HEALTHY
    elif status == BaseHealthStatus.WARNING:
        return HealthStatus.DEGRADED
    elif status == BaseHealthStatus.FAILED:
        return HealthStatus.UNHEALTHY
    else:
        return HealthStatus.UNKNOWN


def _convert_health_result(result: HealthResult) -> HealthCheckResult:
    """Convert base health result to protocol health result"""
    return HealthCheckResult(
        status=_convert_health_status(result.status),
        message=result.message or "",
        details=result.details or {},
        check_name=result.component,
        duration_ms=result.response_time_ms or 0.0
    )




# ML Service Health Check Plugins - Direct Plugin Implementations
class MLServicePlugin(HealthCheckPlugin):
    """ML Service health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_service",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0)
        )
        self.checker = MLServiceHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML service check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class EnhancedMLServicePlugin(HealthCheckPlugin):
    """Enhanced ML Service health check plugin with 2025 features"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="enhanced_ml_service",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=20.0)
        )
        self.checker = EnhancedMLServiceHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Enhanced ML service check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLModelPlugin(HealthCheckPlugin):
    """ML Model health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_model",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0)
        )
        self.checker = MLModelHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML model check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLDataQualityPlugin(HealthCheckPlugin):
    """ML Data Quality health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_data_quality",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=15.0)
        )
        self.checker = MLDataQualityChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML data quality check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLTrainingPlugin(HealthCheckPlugin):
    """ML Training health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_training",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=20.0)
        )
        self.checker = MLTrainingHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML training check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLPerformancePlugin(HealthCheckPlugin):
    """ML Performance health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_performance",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        self.checker = MLPerformanceHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML performance check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# ML Orchestration Health Check Plugins
class MLOrchestratorPlugin(HealthCheckPlugin):
    """ML Orchestrator health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_orchestrator",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0)
        )
        self.checker = MLOrchestratorHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML orchestrator check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLComponentRegistryPlugin(HealthCheckPlugin):
    """ML Component Registry health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_component_registry",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        self.checker = MLComponentRegistryHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML component registry check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLResourceManagerPlugin(HealthCheckPlugin):
    """ML Resource Manager health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_resource_manager",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        self.checker = MLResourceManagerHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML resource manager check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLWorkflowEnginePlugin(HealthCheckPlugin):
    """ML Workflow Engine health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_workflow_engine",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=15.0)
        )
        self.checker = MLWorkflowEngineHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML workflow engine check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MLEventBusPlugin(HealthCheckPlugin):
    """ML Event Bus health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="ml_event_bus",
            category=HealthCheckCategory.ML,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        self.checker = MLEventBusHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"ML event bus check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# Database Health Check Plugins
class DatabasePlugin(HealthCheckPlugin):
    """Basic database health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="database",
            category=HealthCheckCategory.DATABASE,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0)
        )
        self.checker = DatabaseHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class DatabaseConnectionPoolPlugin(HealthCheckPlugin):
    """Database connection pool health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="database_connection_pool",
            category=HealthCheckCategory.DATABASE,
            config=config or HealthCheckPluginConfig(timeout_seconds=5.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        if not DATABASE_HEALTH_AVAILABLE:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Database health monitoring not available",
                check_name=self.name
            )
        
        try:
            monitor = ConnectionPoolMonitor()
            metrics = await monitor.get_pool_metrics()
            
            total_connections = metrics.get('total_connections', 0)
            active_connections = metrics.get('active_connections', 0)
            
            # Check for high connection usage
            if total_connections > 0:
                usage_ratio = active_connections / total_connections
                if usage_ratio > 0.9:
                    status = HealthStatus.UNHEALTHY
                    message = f"High connection pool usage: {usage_ratio:.1%}"
                elif usage_ratio > 0.75:
                    status = HealthStatus.DEGRADED
                    message = f"Moderate connection pool usage: {usage_ratio:.1%}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Connection pool healthy: {usage_ratio:.1%} usage"
            else:
                status = HealthStatus.HEALTHY
                message = "No active connections"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=metrics,
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Connection pool check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class DatabaseQueryPerformancePlugin(HealthCheckPlugin):
    """Database query performance health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="database_query_performance",
            category=HealthCheckCategory.DATABASE,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        if not DATABASE_HEALTH_AVAILABLE:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Database health monitoring not available",
                check_name=self.name
            )
        
        try:
            analyzer = QueryPerformanceAnalyzer()
            metrics = await analyzer.analyze_query_performance()
            
            avg_query_time = metrics.get('average_query_time_ms', 0)
            slow_queries = metrics.get('slow_queries_count', 0)
            
            if slow_queries > 10 or avg_query_time > 1000:
                status = HealthStatus.UNHEALTHY
                message = f"Poor query performance: {slow_queries} slow queries, {avg_query_time:.1f}ms avg"
            elif slow_queries > 5 or avg_query_time > 500:
                status = HealthStatus.DEGRADED
                message = f"Moderate query performance: {slow_queries} slow queries, {avg_query_time:.1f}ms avg"
            else:
                status = HealthStatus.HEALTHY
                message = f"Good query performance: {avg_query_time:.1f}ms avg"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=metrics,
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Query performance check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class DatabaseIndexHealthPlugin(HealthCheckPlugin):
    """Database index health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="database_index_health",
            category=HealthCheckCategory.DATABASE,
            config=config or HealthCheckPluginConfig(timeout_seconds=15.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        if not DATABASE_HEALTH_AVAILABLE:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Database health monitoring not available",
                check_name=self.name
            )
        
        try:
            assessor = IndexHealthAssessor()
            assessment = await assessor.assess_index_health()
            
            unused_indexes = assessment.get('unused_indexes', [])
            missing_indexes = assessment.get('missing_indexes', [])
            
            if len(unused_indexes) > 5 or len(missing_indexes) > 3:
                status = HealthStatus.DEGRADED
                message = f"Index issues: {len(unused_indexes)} unused, {len(missing_indexes)} missing"
            elif len(unused_indexes) > 2 or len(missing_indexes) > 1:
                status = HealthStatus.DEGRADED
                message = f"Minor index issues: {len(unused_indexes)} unused, {len(missing_indexes)} missing"  
            else:
                status = HealthStatus.HEALTHY
                message = "Index health good"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=assessment,
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Index health check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class DatabaseBloatPlugin(HealthCheckPlugin):
    """Database table bloat health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="database_bloat",
            category=HealthCheckCategory.DATABASE,
            config=config or HealthCheckPluginConfig(timeout_seconds=20.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        if not DATABASE_HEALTH_AVAILABLE:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Database health monitoring not available",
                check_name=self.name
            )
        
        try:
            detector = TableBloatDetector()
            bloat_info = await detector.detect_table_bloat()
            
            high_bloat_tables = [
                table for table, info in bloat_info.items()
                if info.get('bloat_percentage', 0) > 25
            ]
            
            if len(high_bloat_tables) > 3:
                status = HealthStatus.UNHEALTHY
                message = f"High table bloat: {len(high_bloat_tables)} tables need attention"
            elif len(high_bloat_tables) > 1:
                status = HealthStatus.DEGRADED
                message = f"Some table bloat: {len(high_bloat_tables)} tables need attention"
            else:
                status = HealthStatus.HEALTHY
                message = "Table bloat within acceptable limits"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={"bloat_info": bloat_info, "high_bloat_tables": high_bloat_tables},
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Bloat detection failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# Redis Health Check Plugins
class RedisPlugin(HealthCheckPlugin):
    """Basic Redis health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="redis",
            category=HealthCheckCategory.REDIS,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=5.0)
        )
        self.checker = RedisHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class RedisDetailedPlugin(HealthCheckPlugin):
    """Detailed Redis health check plugin with comprehensive metrics"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="redis_detailed",
            category=HealthCheckCategory.REDIS,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        if not REDIS_HEALTH_AVAILABLE:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Detailed Redis health monitoring not available",
                check_name=self.name
            )
        
        try:
            # Use the detailed Redis health checker
            checker = DetailedRedisHealthChecker()
            health_report = await checker.comprehensive_health_check()
            
            overall_status = health_report.get('overall_status', 'unknown')
            if overall_status == 'healthy':
                status = HealthStatus.HEALTHY
            elif overall_status == 'degraded':
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                status=status,
                message=health_report.get('summary', 'Redis health check completed'),
                details=health_report,
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Detailed Redis check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class RedisMemoryPlugin(HealthCheckPlugin):
    """Redis memory usage health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="redis_memory",
            category=HealthCheckCategory.REDIS,
            config=config or HealthCheckPluginConfig(timeout_seconds=5.0)
        )
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            # Simple Redis memory check using basic Redis connection
            from ...cache.redis_client import get_redis_client
            
            redis = await get_redis_client()
            info = await redis.info('memory')
            
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory > 0:
                memory_usage = used_memory / max_memory
                if memory_usage > 0.9:
                    status = HealthStatus.UNHEALTHY
                    message = f"High Redis memory usage: {memory_usage:.1%}"
                elif memory_usage > 0.75:
                    status = HealthStatus.DEGRADED
                    message = f"Moderate Redis memory usage: {memory_usage:.1%}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Redis memory usage normal: {memory_usage:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis memory usage: {used_memory / 1024 / 1024:.1f}MB"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={"used_memory": used_memory, "max_memory": max_memory},
                check_name=self.name
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Redis memory check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# Analytics and API Health Check Plugins
class AnalyticsServicePlugin(HealthCheckPlugin):
    """Analytics Service health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="analytics_service",
            category=HealthCheckCategory.API,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0)
        )
        self.checker = AnalyticsServiceHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Analytics service check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class EnhancedAnalyticsServicePlugin(HealthCheckPlugin):
    """Enhanced Analytics Service health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="enhanced_analytics_service", 
            category=HealthCheckCategory.API,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0)
        )
        self.checker = EnhancedAnalyticsServiceHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Enhanced analytics service check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class MCPServerPlugin(HealthCheckPlugin):
    """MCP Server health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="mcp_server",
            category=HealthCheckCategory.API,
            config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0)
        )
        self.checker = MCPServerHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"MCP server check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# System Health Check Plugins  
class SystemResourcesPlugin(HealthCheckPlugin):
    """System resources health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="system_resources",
            category=HealthCheckCategory.SYSTEM,
            config=config or HealthCheckPluginConfig(timeout_seconds=5.0)
        )
        self.checker = SystemResourcesHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"System resources check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


class QueuePlugin(HealthCheckPlugin):
    """Queue health check plugin"""
    
    def __init__(self, config: Optional[HealthCheckPluginConfig] = None):
        super().__init__(
            name="queue_service",
            category=HealthCheckCategory.SYSTEM,
            config=config or HealthCheckPluginConfig(timeout_seconds=10.0)
        )
        self.checker = QueueHealthChecker()
        
    async def execute_check(self) -> HealthCheckResult:
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Queue service check failed: {str(e)}",
                details={"error": str(e)},
                check_name=self.name
            )


# Plugin Factory Functions
def create_ml_plugins() -> list[HealthCheckPlugin]:
    """Create all ML-related health check plugins"""
    return [
        MLServicePlugin(),
        EnhancedMLServicePlugin(),
        MLModelPlugin(),
        MLDataQualityPlugin(),
        MLTrainingPlugin(),
        MLPerformancePlugin(),
        MLOrchestratorPlugin(),
        MLComponentRegistryPlugin(),
        MLResourceManagerPlugin(), 
        MLWorkflowEnginePlugin(),
        MLEventBusPlugin()
    ]


def create_database_plugins() -> list[HealthCheckPlugin]:
    """Create all database-related health check plugins"""
    return [
        DatabasePlugin(),
        DatabaseConnectionPoolPlugin(),
        DatabaseQueryPerformancePlugin(),
        DatabaseIndexHealthPlugin(),
        DatabaseBloatPlugin()
    ]


def create_redis_plugins() -> list[HealthCheckPlugin]:
    """Create all Redis-related health check plugins"""
    return [
        RedisPlugin(),
        RedisDetailedPlugin(),
        RedisMemoryPlugin()
    ]


def create_api_plugins() -> list[HealthCheckPlugin]:
    """Create all API-related health check plugins"""
    return [
        AnalyticsServicePlugin(),
        EnhancedAnalyticsServicePlugin(),
        MCPServerPlugin()
    ]


def create_system_plugins() -> list[HealthCheckPlugin]:
    """Create all system-related health check plugins"""
    return [
        SystemResourcesPlugin(),
        QueuePlugin()
    ]


def create_all_plugins() -> list[HealthCheckPlugin]:
    """Create all available health check plugins"""
    plugins = []
    plugins.extend(create_ml_plugins())
    plugins.extend(create_database_plugins())
    plugins.extend(create_redis_plugins())
    plugins.extend(create_api_plugins())
    plugins.extend(create_system_plugins())
    return plugins


def register_all_plugins(monitor) -> int:
    """Register all plugins with the unified health monitor"""
    plugins = create_all_plugins()
    registered_count = 0
    
    for plugin in plugins:
        try:
            if monitor.register_plugin(plugin):
                registered_count += 1
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.name}: {e}")
    
    logger.info(f"Registered {registered_count}/{len(plugins)} health check plugins")
    return registered_count