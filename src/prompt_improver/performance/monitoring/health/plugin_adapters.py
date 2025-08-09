"""Plugin Adapters for Health Checkers

Converts existing health checkers to the unified plugin system:
- EnhancedMLServiceHealthChecker
- EnhancedAnalyticsServiceHealthChecker
- DatabaseHealthChecker (5 database types)
- RedisHealthChecker (3 variants)
- API endpoint health checkers
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional
from prompt_improver.core.protocols.ml_protocols import EventBusProtocol
from prompt_improver.ml.orchestration.events.adaptive_event_bus import AdaptiveEventBus
from prompt_improver.performance.monitoring.health.base import HealthResult, HealthStatus as BaseHealthStatus
from prompt_improver.performance.monitoring.health.unified_health_system import HealthCheckCategory, HealthCheckPlugin, HealthCheckPluginConfig
logger = logging.getLogger(__name__)

def _get_database_health_checker():
    """Lazy import DatabaseHealthChecker"""
    from prompt_improver.performance.monitoring.health.checkers import DatabaseHealthChecker
    return DatabaseHealthChecker

def _get_ml_service_health_checker():
    """Lazy import EnhancedMLServiceHealthChecker"""
    from prompt_improver.performance.monitoring.health.enhanced_checkers import EnhancedMLServiceHealthChecker
    return EnhancedMLServiceHealthChecker

def _get_analytics_service_health_checker():
    """Lazy import EnhancedAnalyticsServiceHealthChecker"""
    from prompt_improver.performance.monitoring.health.enhanced_checkers import EnhancedAnalyticsServiceHealthChecker
    return EnhancedAnalyticsServiceHealthChecker

def _get_redis_health_checker():
    """Lazy import RedisHealthChecker"""
    from prompt_improver.performance.monitoring.health.checkers import RedisHealthChecker
    return RedisHealthChecker

def _get_system_resources_health_checker():
    """Lazy import SystemResourcesHealthChecker"""
    from prompt_improver.performance.monitoring.health.checkers import SystemResourcesHealthChecker
    return SystemResourcesHealthChecker

def _get_queue_health_checker():
    """Lazy import QueueHealthChecker"""
    from prompt_improver.performance.monitoring.health.checkers import QueueHealthChecker
    return QueueHealthChecker

def _get_mcp_server_health_checker():
    """Lazy import MCPServerHealthChecker"""
    from prompt_improver.performance.monitoring.health.checkers import MCPServerHealthChecker
    return MCPServerHealthChecker

def _get_enhanced_ml_service_health_checker():
    """Lazy import EnhancedMLServiceHealthChecker"""
    from prompt_improver.performance.monitoring.health.enhanced_checkers import EnhancedMLServiceHealthChecker
    return EnhancedMLServiceHealthChecker

def _get_enhanced_analytics_service_health_checker():
    """Lazy import EnhancedAnalyticsServiceHealthChecker"""
    from prompt_improver.performance.monitoring.health.enhanced_checkers import EnhancedAnalyticsServiceHealthChecker
    return EnhancedAnalyticsServiceHealthChecker

def _get_ml_model_health_checker():
    """Lazy import MLModelHealthChecker"""
    from prompt_improver.performance.monitoring.health.ml_specific_checkers import MLModelHealthChecker
    return MLModelHealthChecker

def _get_ml_data_quality_checker():
    """Lazy import MLDataQualityChecker"""
    from prompt_improver.performance.monitoring.health.ml_specific_checkers import MLDataQualityChecker
    return MLDataQualityChecker

def _get_ml_training_health_checker():
    """Lazy import MLTrainingHealthChecker"""
    from prompt_improver.performance.monitoring.health.ml_specific_checkers import MLTrainingHealthChecker
    return MLTrainingHealthChecker

def _get_ml_performance_health_checker():
    """Lazy import MLPerformanceHealthChecker"""
    from prompt_improver.performance.monitoring.health.ml_specific_checkers import MLPerformanceHealthChecker
    return MLPerformanceHealthChecker

def _get_database_health_monitor():
    """Lazy import DatabaseHealthMonitor"""
    try:
        from prompt_improver.performance.database.health.database_health_monitor import DatabaseHealthMonitor, get_database_health_monitor
        return get_database_health_monitor()
    except ImportError:
        return None

def _get_detailed_redis_health_checker():
    """Lazy import detailed RedisHealthChecker"""
    try:
        from prompt_improver.performance.cache.redis_health import RedisHealthChecker as DetailedRedisHealthChecker
        return DetailedRedisHealthChecker
    except ImportError:
        return None

def _get_health_protocol_types():
    """Lazy import health protocol types to avoid circular imports"""
    from prompt_improver.core.protocols.health_protocol import HealthCheckResult, HealthMonitorProtocol, HealthStatus
    return (HealthMonitorProtocol, HealthCheckResult, HealthStatus)

def _create_health_check_result(status, message, details=None, check_name='', duration_ms=0.0):
    """Helper function to create HealthCheckResult with lazy loading"""
    _, HealthCheckResult, _ = _get_health_protocol_types()
    return _create_health_check_result(status=status, message=message, details=details or {}, check_name=check_name, duration_ms=duration_ms)

def _convert_health_result(health_result: HealthResult):
    """Convert HealthResult to HealthCheckResult"""
    _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
    status_mapping = {BaseHealthStatus.HEALTHY: HealthStatus.HEALTHY, BaseHealthStatus.WARNING: HealthStatus.DEGRADED, BaseHealthStatus.FAILED: HealthStatus.UNHEALTHY}
    return _create_health_check_result(status=status_mapping.get(health_result.status, HealthStatus.UNKNOWN), message=health_result.message or '', details=health_result.details or {}, check_name=health_result.component, duration_ms=health_result.response_time_ms or 0.0)

def _convert_health_status(status: BaseHealthStatus):
    """Convert base health status to protocol health status"""
    _, _, HealthStatus = _get_health_protocol_types()
    if status == BaseHealthStatus.HEALTHY:
        return HealthStatus.HEALTHY
    if status == BaseHealthStatus.WARNING:
        return HealthStatus.DEGRADED
    if status == BaseHealthStatus.FAILED:
        return HealthStatus.UNHEALTHY
    return HealthStatus.UNKNOWN

def _convert_health_result(result: HealthResult):
    """Convert base health result to protocol health result"""
    _, HealthCheckResult, _ = _get_health_protocol_types()
    return _create_health_check_result(status=_convert_health_status(result.status), message=result.message or '', details=result.details or {}, check_name=result.component, duration_ms=result.response_time_ms or 0.0)

class MLServicePlugin(HealthCheckPlugin):
    """ML Service health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_service', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0))
        self.checker = EnhancedMLServiceHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML service check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class EnhancedMLServicePlugin(HealthCheckPlugin):
    """Enhanced ML Service health check plugin with 2025 features"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='enhanced_ml_service', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=20.0))
        self.checker = EnhancedMLServiceHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Enhanced ML service check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLModelPlugin(HealthCheckPlugin):
    """ML Model health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_model', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0))
        self.checker = MLModelHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML model check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLDataQualityPlugin(HealthCheckPlugin):
    """ML Data Quality health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_data_quality', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=15.0))
        self.checker = MLDataQualityChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML data quality check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLTrainingPlugin(HealthCheckPlugin):
    """ML Training health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_training', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=20.0))
        self.checker = MLTrainingHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML training check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLPerformancePlugin(HealthCheckPlugin):
    """ML Performance health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_performance', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))
        self.checker = MLPerformanceHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML performance check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLOrchestratorPlugin(HealthCheckPlugin):
    """ML Orchestrator health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_orchestrator', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0))
        self.checker = MLOrchestratorHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML orchestrator check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLComponentRegistryPlugin(HealthCheckPlugin):
    """ML Component Registry health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_component_registry', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))
        self.checker = MLComponentRegistryHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML component registry check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLResourceManagerPlugin(HealthCheckPlugin):
    """ML Resource Manager health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_resource_manager', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))
        self.checker = MLResourceManagerHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML resource manager check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLWorkflowEnginePlugin(HealthCheckPlugin):
    """ML Workflow Engine health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_workflow_engine', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=15.0))
        self.checker = MLWorkflowEngineHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML workflow engine check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MLEventBusPlugin(HealthCheckPlugin):
    """ML Event Bus health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='ml_event_bus', category=HealthCheckCategory.ML, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))
        self.checker = MLEventBusHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'ML event bus check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class DatabasePlugin(HealthCheckPlugin):
    """Enhanced comprehensive database health check plugin with <10ms performance target"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='database', category=HealthCheckCategory.DATABASE, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=8.0, retry_count=2))
        self._cached_monitor = None

    def _get_monitor(self):
        """Get database health monitor with lazy loading and caching"""
        if self._cached_monitor is None:
            try:
                from prompt_improver.performance.database.health.database_health_monitor import get_database_health_monitor
                self._cached_monitor = get_database_health_monitor()
            except ImportError:
                return None
        return self._cached_monitor

    async def execute_check(self):
        try:
            monitor = self._get_monitor()
            if not monitor:
                _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
                return _create_health_check_result(status=HealthStatus.UNKNOWN, message='Database health monitor not available', check_name=self.name)
            start_time = time.time()
            health_metrics = await monitor.get_comprehensive_health()
            duration_ms = (time.time() - start_time) * 1000
            connection_health = health_metrics.get('connection_health', {})
            storage_health = health_metrics.get('storage_health', {})
            cache_metrics = health_metrics.get('cache_metrics', {})
            pool_utilization = connection_health.get('utilization_percent', 0)
            cache_hit_ratio = cache_metrics.get('overall_cache_hit_ratio_percent', 100)
            db_size_gb = storage_health.get('database_size_bytes', 0) / 1024 ** 3
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            issues = []
            if pool_utilization > 90:
                issues.append(f'High pool utilization: {pool_utilization:.1f}%')
            if cache_hit_ratio < 90:
                issues.append(f'Low cache hit ratio: {cache_hit_ratio:.1f}%')
            if db_size_gb > 50:
                issues.append(f'Large database: {db_size_gb:.1f}GB')
            if len(issues) > 2:
                status = HealthStatus.UNHEALTHY
                message = f"Database health critical: {', '.join(issues[:2])}"
            elif len(issues) > 0:
                status = HealthStatus.DEGRADED
                message = f"Database health degraded: {', '.join(issues[:1])}"
            else:
                status = HealthStatus.HEALTHY
                message = f'Database healthy: {pool_utilization:.1f}% pool, {cache_hit_ratio:.1f}% cache hit'
            return _create_health_check_result(status=status, message=message, details={'pool_utilization_percent': pool_utilization, 'cache_hit_ratio_percent': cache_hit_ratio, 'database_size_gb': round(db_size_gb, 2), 'issues_detected': issues, 'performance_ms': round(duration_ms, 2), 'comprehensive_metrics': {'connection_health': connection_health.get('health_status', 'unknown'), 'storage_health': storage_health.get('health_status', 'unknown'), 'cache_efficiency': cache_metrics.get('cache_efficiency', 'unknown')}}, check_name=self.name, duration_ms=duration_ms)
        except Exception as e:
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Database health check failed: {e!s}', details={'error': str(e), 'error_type': type(e).__name__}, check_name=self.name)

class DatabaseConnectionPoolPlugin(HealthCheckPlugin):
    """Database connection pool health check plugin - integrated with UnifiedConnectionManager"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='database_connection_pool', category=HealthCheckCategory.DATABASE, config=config or HealthCheckPluginConfig(timeout_seconds=5.0))

    async def execute_check(self):
        try:
            from prompt_improver.database.unified_connection_manager import get_unified_connection_manager
            unified_manager = get_unified_connection_manager()
            pool_metrics = await unified_manager.get_ml_telemetry_metrics()
            coordination_status = await unified_manager.coordinate_pools()
            utilization = pool_metrics.get('pool_utilization', 0)
            active_connections = pool_metrics.get('active_connections', 0)
            pool_size = pool_metrics.get('pool_size', 0)
            health_status = pool_metrics.get('health_status', 'unknown')
            is_healthy = True
            issues = []
            if utilization > 0.9:
                is_healthy = False
                issues.append(f'High pool utilization: {utilization:.1%}')
            if pool_size > 0 and active_connections >= pool_size:
                is_healthy = False
                issues.append('Pool exhausted: all connections in use')
            if health_status != 'healthy':
                is_healthy = False
                issues.append(f'Pool health degraded: {health_status}')
            healthy_pools = coordination_status.get('healthy_pools', 0)
            if healthy_pools == 0:
                is_healthy = False
                issues.append('No healthy pools available in coordination')
            metadata = {'pool_utilization': utilization, 'active_connections': active_connections, 'available_connections': pool_metrics.get('available_connections', 0), 'pool_size': pool_size, 'avg_connection_time_ms': pool_metrics.get('avg_connection_time_ms', 0), 'avg_query_time_ms': pool_metrics.get('avg_query_time_ms', 0), 'coordination_status': coordination_status.get('status', 'unknown'), 'healthy_pools': healthy_pools, 'issues': issues}
            if is_healthy:
                if utilization > 0.75:
                    status = HealthStatus.DEGRADED
                    message = f'Pool utilization high but stable: {utilization:.1%}'
                else:
                    status = HealthStatus.HEALTHY
                    message = f'Connection pool healthy: {utilization:.1%} utilization'
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Connection pool unhealthy: {', '.join(issues)}"
            return _create_health_check_result(status=status, message=message, details=metadata, check_name=self.name)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Connection pool check failed: {e!s}', details={'error': str(e), 'source': 'UnifiedConnectionManager'}, check_name=self.name)

class DatabaseQueryPerformancePlugin(HealthCheckPlugin):
    """Database query performance health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='database_query_performance', category=HealthCheckCategory.DATABASE, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))

    async def execute_check(self):
        if not DATABASE_HEALTH_AVAILABLE:
            return _create_health_check_result(status=HealthStatus.UNKNOWN, message='Database health monitoring not available', check_name=self.name)
        try:
            from prompt_improver.performance.database.health.database_health_monitor import get_database_health_monitor
            analyzer = get_database_health_monitor()
            metrics = await analyzer.analyze_query_performance()
            avg_query_time = metrics.get('average_query_time_ms', 0)
            slow_queries = metrics.get('slow_queries_count', 0)
            if slow_queries > 10 or avg_query_time > 1000:
                status = HealthStatus.UNHEALTHY
                message = f'Poor query performance: {slow_queries} slow queries, {avg_query_time:.1f}ms avg'
            elif slow_queries > 5 or avg_query_time > 500:
                status = HealthStatus.DEGRADED
                message = f'Moderate query performance: {slow_queries} slow queries, {avg_query_time:.1f}ms avg'
            else:
                status = HealthStatus.HEALTHY
                message = f'Good query performance: {avg_query_time:.1f}ms avg'
            return _create_health_check_result(status=status, message=message, details=metrics, check_name=self.name)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Query performance check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class DatabaseIndexHealthPlugin(HealthCheckPlugin):
    """Enhanced database index health check plugin with <10ms performance target"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='database_index_health', category=HealthCheckCategory.DATABASE, config=config or HealthCheckPluginConfig(timeout_seconds=8.0, critical=False, retry_count=1))
        self._cached_monitor = None

    def _get_monitor(self):
        """Get database health monitor with lazy loading and caching"""
        if self._cached_monitor is None:
            try:
                from prompt_improver.performance.database.health.database_health_monitor import get_database_health_monitor
                self._cached_monitor = get_database_health_monitor()
            except ImportError:
                return None
        return self._cached_monitor

    async def execute_check(self):
        try:
            monitor = self._get_monitor()
            if not monitor:
                _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
                return _create_health_check_result(status=HealthStatus.UNKNOWN, message='Database health monitor not available', check_name=self.name)
            start_time = time.time()
            assessment_report = await monitor.index_assessor.assess_index_health()
            duration_ms = (time.time() - start_time) * 1000
            if hasattr(assessment_report, 'unused_indexes'):
                unused_count = len(assessment_report.unused_indexes)
                redundant_count = len(assessment_report.redundant_indexes)
                bloated_count = len(assessment_report.bloated_indexes)
                low_usage_count = len(assessment_report.low_usage_indexes)
                missing_suggestions = len(assessment_report.missing_index_suggestions)
                health_score = assessment_report.health_score
                potential_savings_mb = assessment_report.potential_space_savings_bytes / 1024 ** 2
            else:
                unused_count = len(assessment_report.get('unused_indexes', []))
                redundant_count = len(assessment_report.get('redundant_indexes', []))
                bloated_count = len(assessment_report.get('bloated_indexes', []))
                low_usage_count = len(assessment_report.get('low_usage_indexes', []))
                missing_suggestions = len(assessment_report.get('missing_index_suggestions', []))
                health_score = assessment_report.get('health_score', 100)
                potential_savings_mb = assessment_report.get('potential_space_savings_bytes', 0) / 1024 ** 2
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            total_issues = unused_count + redundant_count + bloated_count
            if health_score < 60 or total_issues > 10 or potential_savings_mb > 500:
                status = HealthStatus.UNHEALTHY
                message = f'Critical index issues: {total_issues} problematic indexes, {potential_savings_mb:.1f}MB potential savings'
            elif health_score < 80 or total_issues > 5 or missing_suggestions > 3:
                status = HealthStatus.DEGRADED
                message = f'Index optimization needed: {total_issues} issues, {missing_suggestions} missing indexes'
            else:
                status = HealthStatus.HEALTHY
                message = f'Index health good: score {health_score:.1f}, {total_issues} minor issues'
            return _create_health_check_result(status=status, message=message, details={'health_score': health_score, 'unused_indexes': unused_count, 'redundant_indexes': redundant_count, 'bloated_indexes': bloated_count, 'low_usage_indexes': low_usage_count, 'missing_suggestions': missing_suggestions, 'potential_savings_mb': round(potential_savings_mb, 2), 'total_issues': total_issues, 'performance_ms': round(duration_ms, 2)}, check_name=self.name, duration_ms=duration_ms)
        except Exception as e:
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Index health check failed: {e!s}', details={'error': str(e), 'error_type': type(e).__name__}, check_name=self.name)

class DatabaseBloatPlugin(HealthCheckPlugin):
    """Enhanced database table bloat detection plugin with <10ms performance target"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='database_bloat', category=HealthCheckCategory.DATABASE, config=config or HealthCheckPluginConfig(timeout_seconds=8.0, critical=False, retry_count=1))
        self._cached_monitor = None

    def _get_monitor(self):
        """Get database health monitor with lazy loading and caching"""
        if self._cached_monitor is None:
            try:
                from prompt_improver.performance.database.health.database_health_monitor import get_database_health_monitor
                self._cached_monitor = get_database_health_monitor()
            except ImportError:
                return None
        return self._cached_monitor

    async def execute_check(self):
        try:
            monitor = self._get_monitor()
            if not monitor:
                _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
                return _create_health_check_result(status=HealthStatus.UNKNOWN, message='Database health monitor not available', check_name=self.name)
            start_time = time.time()
            bloat_report = await monitor.bloat_detector.detect_table_bloat()
            duration_ms = (time.time() - start_time) * 1000
            high_bloat_tables = []
            moderate_bloat_tables = []
            total_bloat_bytes = 0
            if hasattr(bloat_report, 'bloated_tables'):
                high_bloat_tables = [t.table_name for t in bloat_report.bloated_tables if t.bloat_ratio_percent > 30]
                moderate_bloat_tables = [t.table_name for t in bloat_report.bloated_tables if 15 <= t.bloat_ratio_percent <= 30]
                total_bloat_bytes = sum((t.bloat_bytes for t in bloat_report.bloated_tables))
                total_tables = bloat_report.total_tables_analyzed
            else:
                for table_name, info in bloat_report.items():
                    bloat_pct = info.get('bloat_percentage', 0)
                    if bloat_pct > 30:
                        high_bloat_tables.append(table_name)
                    elif bloat_pct >= 15:
                        moderate_bloat_tables.append(table_name)
                    total_bloat_bytes += info.get('bloat_bytes', 0)
                total_tables = len(bloat_report)
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            if len(high_bloat_tables) > 5 or total_bloat_bytes > 1024 ** 3:
                status = HealthStatus.UNHEALTHY
                message = f'Critical table bloat: {len(high_bloat_tables)} tables with >30% bloat'
            elif len(high_bloat_tables) > 2 or len(moderate_bloat_tables) > 5:
                status = HealthStatus.DEGRADED
                message = f'Moderate table bloat: {len(high_bloat_tables)} high, {len(moderate_bloat_tables)} moderate'
            else:
                status = HealthStatus.HEALTHY
                message = f'Table bloat acceptable: {total_tables} tables analyzed'
            return _create_health_check_result(status=status, message=message, details={'high_bloat_tables': high_bloat_tables[:10], 'moderate_bloat_tables': moderate_bloat_tables[:10], 'total_tables_analyzed': total_tables, 'total_bloat_bytes': total_bloat_bytes, 'total_bloat_mb': round(total_bloat_bytes / 1024 ** 2, 2), 'performance_ms': round(duration_ms, 2)}, check_name=self.name, duration_ms=duration_ms)
        except Exception as e:
            _, HealthCheckResult, HealthStatus = _get_health_protocol_types()
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Database bloat check failed: {e!s}', details={'error': str(e), 'error_type': type(e).__name__}, check_name=self.name)

class RedisPlugin(HealthCheckPlugin):
    """Basic Redis health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='redis', category=HealthCheckCategory.REDIS, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=5.0))
        self.checker = RedisHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Redis check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class RedisDetailedPlugin(HealthCheckPlugin):
    """Detailed Redis health check plugin with comprehensive metrics"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='redis_detailed', category=HealthCheckCategory.REDIS, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))

    async def execute_check(self):
        if not REDIS_HEALTH_AVAILABLE:
            return _create_health_check_result(status=HealthStatus.UNKNOWN, message='Detailed Redis health monitoring not available', check_name=self.name)
        try:
            checker = DetailedRedisHealthChecker()
            health_report = await checker.comprehensive_health_check()
            overall_status = health_report.get('overall_status', 'unknown')
            if overall_status == 'healthy':
                status = HealthStatus.HEALTHY
            elif overall_status == 'degraded':
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            return _create_health_check_result(status=status, message=health_report.get('summary', 'Redis health check completed'), details=health_report, check_name=self.name)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Detailed Redis check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class RedisMemoryPlugin(HealthCheckPlugin):
    """Redis memory usage health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='redis_memory', category=HealthCheckCategory.REDIS, config=config or HealthCheckPluginConfig(timeout_seconds=5.0))

    async def execute_check(self):
        try:
            from prompt_improver.performance.cache.redis_client import get_redis_client
            redis = await get_redis_client()
            info = await redis.info('memory')
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            if max_memory > 0:
                memory_usage = used_memory / max_memory
                if memory_usage > 0.9:
                    status = HealthStatus.UNHEALTHY
                    message = f'High Redis memory usage: {memory_usage:.1%}'
                elif memory_usage > 0.75:
                    status = HealthStatus.DEGRADED
                    message = f'Moderate Redis memory usage: {memory_usage:.1%}'
                else:
                    status = HealthStatus.HEALTHY
                    message = f'Redis memory usage normal: {memory_usage:.1%}'
            else:
                status = HealthStatus.HEALTHY
                message = f'Redis memory usage: {used_memory / 1024 / 1024:.1f}MB'
            return _create_health_check_result(status=status, message=message, details={'used_memory': used_memory, 'max_memory': max_memory}, check_name=self.name)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Redis memory check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class AnalyticsServicePlugin(HealthCheckPlugin):
    """Analytics Service health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='analytics_service', category=HealthCheckCategory.API, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0))
        self.checker = EnhancedAnalyticsServiceHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Analytics service check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class EnhancedAnalyticsServicePlugin(HealthCheckPlugin):
    """Enhanced Analytics Service health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='enhanced_analytics_service', category=HealthCheckCategory.API, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=15.0))
        self.checker = EnhancedAnalyticsServiceHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Enhanced analytics service check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class MCPServerPlugin(HealthCheckPlugin):
    """MCP Server health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='mcp_server', category=HealthCheckCategory.API, config=config or HealthCheckPluginConfig(critical=True, timeout_seconds=10.0))
        self.checker = MCPServerHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'MCP server check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class SystemResourcesPlugin(HealthCheckPlugin):
    """System resources health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='system_resources', category=HealthCheckCategory.SYSTEM, config=config or HealthCheckPluginConfig(timeout_seconds=5.0))
        self.checker = SystemResourcesHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'System resources check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

class QueuePlugin(HealthCheckPlugin):
    """Queue health check plugin"""

    def __init__(self, config: HealthCheckPluginConfig | None=None):
        super().__init__(name='queue_service', category=HealthCheckCategory.SYSTEM, config=config or HealthCheckPluginConfig(timeout_seconds=10.0))
        self.checker = QueueHealthChecker()

    async def execute_check(self):
        try:
            result = await self.checker.check()
            return _convert_health_result(result)
        except Exception as e:
            return _create_health_check_result(status=HealthStatus.UNHEALTHY, message=f'Queue service check failed: {e!s}', details={'error': str(e)}, check_name=self.name)

def create_ml_plugins() -> list[HealthCheckPlugin]:
    """Create all ML-related health check plugins"""
    return [MLServicePlugin(), EnhancedMLServicePlugin(), MLModelPlugin(), MLDataQualityPlugin(), MLTrainingPlugin(), MLPerformancePlugin(), MLOrchestratorPlugin(), MLComponentRegistryPlugin(), MLResourceManagerPlugin(), MLWorkflowEnginePlugin(), MLEventBusPlugin()]

def create_database_plugins() -> list[HealthCheckPlugin]:
    """Create all database-related health check plugins"""
    return [DatabasePlugin(), DatabaseConnectionPoolPlugin(), DatabaseQueryPerformancePlugin(), DatabaseIndexHealthPlugin(), DatabaseBloatPlugin()]

def create_redis_plugins() -> list[HealthCheckPlugin]:
    """Create all Redis-related health check plugins"""
    return [RedisPlugin(), RedisDetailedPlugin(), RedisMemoryPlugin()]

def create_api_plugins() -> list[HealthCheckPlugin]:
    """Create all API-related health check plugins"""
    return [AnalyticsServicePlugin(), EnhancedAnalyticsServicePlugin(), MCPServerPlugin()]

def create_system_plugins() -> list[HealthCheckPlugin]:
    """Create all system-related health check plugins"""
    return [SystemResourcesPlugin(), QueuePlugin()]

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
            logger.error('Failed to register plugin {plugin.name}: %s', e)
    logger.info('Registered {registered_count}/%s health check plugins', len(plugins))
    return registered_count
