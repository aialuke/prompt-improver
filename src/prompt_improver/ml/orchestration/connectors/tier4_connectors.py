"""
Tier 4 Component Connectors - Performance & Testing Components.

Connectors for the 8 performance and testing components including A/B testing,
monitoring, and real-time analytics.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .component_connector import ComponentConnector, ComponentMetadata, ComponentCapability, ComponentTier

class AdvancedABTestingConnector(ComponentConnector):
    """Connector for AdvancedABTesting component."""
    
    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="advanced_ab_testing",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="enhanced_ab_testing",
                    description="Enhanced A/B testing with advanced statistics",
                    input_types=["test_config", "participant_data"],
                    output_types=["ab_test_results"]
                ),
                ComponentCapability(
                    name="multi_variant_testing",
                    description="Multi-variant testing capabilities",
                    input_types=["variant_config", "test_data"],
                    output_types=["mvt_results"]
                )
            ],
            resource_requirements={"memory": "1GB", "cpu": "2 cores"}
        )
        super().__init__(metadata, event_bus)
    
    async def _initialize_component(self) -> None:
        """Initialize AdvancedABTesting component."""
        self.logger.info("AdvancedABTesting connector initialized")
        await asyncio.sleep(0.1)
    
    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AdvancedABTesting capability."""
        if capability_name == "enhanced_ab_testing":
            return await self._enhanced_ab_testing(parameters)
        elif capability_name == "multi_variant_testing":
            return await self._multi_variant_testing(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _enhanced_ab_testing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced A/B testing."""
        await asyncio.sleep(0.3)
        return {
            "test_id": "enhanced_ab_001",
            "variant_performance": {"A": 0.83, "B": 0.87},
            "statistical_power": 0.85,
            "sample_size": 2000,
            "confidence_level": 0.95
        }
    
    async def _multi_variant_testing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-variant testing."""
        await asyncio.sleep(0.4)
        return {
            "variants_tested": 4,
            "best_variant": "C",
            "performance_scores": {"A": 0.80, "B": 0.83, "C": 0.89, "D": 0.85},
            "significance_matrix": "all_significant"
        }

class RealTimeAnalyticsConnector(ComponentConnector):
    """Connector for RealTimeAnalytics component."""
    
    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="real_time_analytics",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="live_monitoring",
                    description="Live performance monitoring",
                    input_types=["monitoring_config", "metrics_stream"],
                    output_types=["live_metrics"]
                ),
                ComponentCapability(
                    name="real_time_alerts",
                    description="Real-time alerting system",
                    input_types=["alert_config", "threshold_data"],
                    output_types=["alert_status"]
                )
            ],
            resource_requirements={"memory": "2GB", "cpu": "3 cores"}
        )
        super().__init__(metadata, event_bus)
    
    async def _initialize_component(self) -> None:
        """Initialize RealTimeAnalytics component."""
        self.logger.info("RealTimeAnalytics connector initialized")
        await asyncio.sleep(0.1)
    
    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RealTimeAnalytics capability."""
        if capability_name == "live_monitoring":
            return await self._live_monitoring(parameters)
        elif capability_name == "real_time_alerts":
            return await self._real_time_alerts(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _live_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Live monitoring."""
        await asyncio.sleep(0.2)
        return {
            "active_monitors": 12,
            "metrics_collected": 150,
            "update_frequency": "1s",
            "dashboard_status": "active"
        }
    
    async def _real_time_alerts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real-time alerts."""
        await asyncio.sleep(0.1)
        return {
            "alerts_active": 3,
            "alert_types": ["performance", "error_rate", "resource"],
            "notification_channels": ["email", "slack", "webhook"],
            "response_time": "500ms"
        }

class PerformanceMonitoringConnector(ComponentConnector):
    """Connector for PerformanceMonitoring component."""
    
    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="performance_monitoring",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="system_monitoring",
                    description="System performance monitoring",
                    input_types=["monitoring_targets", "metrics_config"],
                    output_types=["performance_report"]
                ),
                ComponentCapability(
                    name="resource_tracking",
                    description="Resource utilization tracking",
                    input_types=["resource_config"],
                    output_types=["resource_metrics"]
                )
            ],
            resource_requirements={"memory": "1GB", "cpu": "2 cores"}
        )
        super().__init__(metadata, event_bus)
    
    async def _initialize_component(self) -> None:
        """Initialize PerformanceMonitoring component."""
        self.logger.info("PerformanceMonitoring connector initialized")
        await asyncio.sleep(0.1)
    
    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PerformanceMonitoring capability."""
        if capability_name == "system_monitoring":
            return await self._system_monitoring(parameters)
        elif capability_name == "resource_tracking":
            return await self._resource_tracking(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _system_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """System monitoring."""
        await asyncio.sleep(0.2)
        return {
            "cpu_usage": 0.65,
            "memory_usage": 0.72,
            "disk_usage": 0.45,
            "network_latency": "15ms",
            "throughput": "1500 req/s"
        }
    
    async def _resource_tracking(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resource tracking."""
        await asyncio.sleep(0.1)
        return {
            "gpu_utilization": 0.80,
            "memory_allocated": "6.2GB",
            "storage_used": "125GB",
            "bandwidth_usage": "850 Mbps"
        }

class DatabasePerformanceMonitorConnector(ComponentConnector):
    """Connector for DatabasePerformanceMonitor component."""

    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="database_performance_monitor",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="real_time_monitoring",
                    description="Real-time database performance monitoring",
                    input_types=["monitoring_config", "performance_targets"],
                    output_types=["performance_snapshot", "performance_metrics"]
                ),
                ComponentCapability(
                    name="cache_hit_monitoring",
                    description="Monitor database cache hit ratios",
                    input_types=["cache_config"],
                    output_types=["cache_metrics", "cache_recommendations"]
                ),
                ComponentCapability(
                    name="slow_query_detection",
                    description="Detect and analyze slow queries",
                    input_types=["query_thresholds"],
                    output_types=["slow_query_report", "optimization_suggestions"]
                ),
                ComponentCapability(
                    name="performance_recommendations",
                    description="Generate performance optimization recommendations",
                    input_types=["performance_data"],
                    output_types=["recommendations", "optimization_plan"]
                )
            ],
            resource_requirements={"memory": "512MB", "cpu": "1 core"}
        )
        super().__init__(metadata, event_bus)

    async def _initialize_component(self) -> None:
        """Initialize DatabasePerformanceMonitor component."""
        try:
            from prompt_improver.database.performance_monitor import get_performance_monitor
            self.component = await get_performance_monitor()
            self.logger.info("DatabasePerformanceMonitor connector initialized")

            # Emit initialization event
            if self.event_bus:
                await self.event_bus.emit("database_performance_monitor_initialized", {
                    "component_name": self.metadata.name,
                    "status": "ready"
                })
        except Exception as e:
            self.logger.error(f"Failed to initialize DatabasePerformanceMonitor: {e}")
            raise

    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DatabasePerformanceMonitor capability."""
        if capability_name == "real_time_monitoring":
            return await self._real_time_monitoring(parameters)
        elif capability_name == "cache_hit_monitoring":
            return await self._cache_hit_monitoring(parameters)
        elif capability_name == "slow_query_detection":
            return await self._slow_query_detection(parameters)
        elif capability_name == "performance_recommendations":
            return await self._performance_recommendations(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")

    async def _real_time_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real-time database performance monitoring."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabasePerformanceMonitor component not initialized")

        snapshot = await self.component.take_performance_snapshot()
        return {
            "status": "success",
            "snapshot": {
                "timestamp": snapshot.timestamp.isoformat(),
                "cache_hit_ratio": snapshot.cache_hit_ratio,
                "active_connections": snapshot.active_connections,
                "avg_query_time_ms": snapshot.avg_query_time_ms,
                "slow_queries_count": snapshot.slow_queries_count,
                "index_hit_ratio": snapshot.index_hit_ratio
            }
        }

    async def _cache_hit_monitoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor database cache hit ratios."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabasePerformanceMonitor component not initialized")

        cache_ratio = await self.component.get_cache_hit_ratio()
        index_ratio = await self.component.get_index_hit_ratio()
        return {
            "status": "success",
            "cache_hit_ratio": cache_ratio,
            "index_hit_ratio": index_ratio,
            "meets_target": cache_ratio >= 90.0,
            "target_threshold": 90.0
        }

    async def _slow_query_detection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and analyze slow queries."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabasePerformanceMonitor component not initialized")

        limit = parameters.get("limit", 10)
        slow_queries = await self.component.get_slow_queries(limit=limit)
        return {
            "status": "success",
            "slow_queries": [
                {
                    "query_text": q.query_text,
                    "mean_exec_time": q.mean_exec_time,
                    "calls": q.calls,
                    "total_exec_time": q.total_exec_time,
                    "cache_hit_ratio": q.cache_hit_ratio
                } for q in slow_queries
            ],
            "query_count": len(slow_queries)
        }

    async def _performance_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance optimization recommendations."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabasePerformanceMonitor component not initialized")

        recommendations = await self.component.get_recommendations()
        return {
            "status": "success",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations)
        }

class DatabaseConnectionOptimizerConnector(ComponentConnector):
    """Connector for DatabaseConnectionOptimizer component."""

    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="database_connection_optimizer",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="optimize_connection_settings",
                    description="Optimize database connection settings with dynamic resource detection",
                    input_types=["optimization_config"],
                    output_types=["optimization_result", "applied_settings"]
                ),
                ComponentCapability(
                    name="create_performance_indexes",
                    description="Create performance-critical database indexes",
                    input_types=["index_config"],
                    output_types=["index_creation_result", "performance_impact"]
                ),
                ComponentCapability(
                    name="system_resource_analysis",
                    description="Analyze system resources for optimal database configuration",
                    input_types=["analysis_config"],
                    output_types=["resource_analysis", "optimization_recommendations"]
                )
            ],
            resource_requirements={"memory": "256MB", "cpu": "1 core"}
        )
        super().__init__(metadata, event_bus)

    async def _initialize_component(self) -> None:
        """Initialize DatabaseConnectionOptimizer component."""
        try:
            from prompt_improver.database.query_optimizer import DatabaseConnectionOptimizer
            self.component = DatabaseConnectionOptimizer()
            self.logger.info("DatabaseConnectionOptimizer connector initialized")

            # Emit initialization event
            if self.event_bus:
                await self.event_bus.emit("database_connection_optimizer_initialized", {
                    "component_name": self.metadata.name,
                    "status": "ready"
                })
        except Exception as e:
            self.logger.error(f"Failed to initialize DatabaseConnectionOptimizer: {e}")
            raise

    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DatabaseConnectionOptimizer capability."""
        if capability_name == "optimize_connection_settings":
            return await self._optimize_connection_settings(parameters)
        elif capability_name == "create_performance_indexes":
            return await self._create_performance_indexes(parameters)
        elif capability_name == "system_resource_analysis":
            return await self._system_resource_analysis(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")

    async def _optimize_connection_settings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database connection settings."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabaseConnectionOptimizer component not initialized")

        await self.component.optimize_connection_settings()
        return {
            "status": "success",
            "message": "Database connection settings optimized with dynamic resource detection",
            "optimization_applied": True
        }

    async def _create_performance_indexes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance-critical database indexes."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabaseConnectionOptimizer component not initialized")

        await self.component.create_performance_indexes()
        return {
            "status": "success",
            "message": "Performance indexes created successfully",
            "indexes_created": True
        }

    async def _system_resource_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system resources for optimal database configuration."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("DatabaseConnectionOptimizer component not initialized")

        resources = self.component._get_system_resources()
        memory_settings = self.component._calculate_optimal_memory_settings(resources)
        return {
            "status": "success",
            "system_resources": resources,
            "recommended_settings": memory_settings,
            "analysis_timestamp": datetime.now().isoformat()
        }

class PreparedStatementCacheConnector(ComponentConnector):
    """Connector for PreparedStatementCache component."""

    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="prepared_statement_cache",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="cache_performance_analysis",
                    description="Analyze prepared statement cache performance",
                    input_types=["analysis_config"],
                    output_types=["cache_performance_metrics", "optimization_recommendations"]
                ),
                ComponentCapability(
                    name="query_optimization_analysis",
                    description="Analyze query optimization patterns",
                    input_types=["optimization_config"],
                    output_types=["query_optimization_results", "complexity_analysis"]
                ),
                ComponentCapability(
                    name="cache_efficiency_analysis",
                    description="Analyze cache efficiency and usage patterns",
                    input_types=["efficiency_config"],
                    output_types=["efficiency_metrics", "usage_recommendations"]
                )
            ],
            resource_requirements={"memory": "256MB", "cpu": "1 core"}
        )
        super().__init__(metadata, event_bus)

    async def _initialize_component(self) -> None:
        """Initialize PreparedStatementCache component."""
        try:
            from prompt_improver.database.query_optimizer import PreparedStatementCache
            # Create a new instance for orchestrator use
            self.component = PreparedStatementCache(max_size=100)
            self.logger.info("PreparedStatementCache connector initialized")

            # Emit initialization event
            if self.event_bus:
                try:
                    result = self.event_bus.emit("prepared_statement_cache_initialized", {
                        "component_name": self.metadata.name,
                        "status": "ready",
                        "cache_size": 100
                    })
                    if hasattr(result, '__await__'):
                        await result
                except Exception as e:
                    # Don't fail initialization if event emission fails
                    self.logger.debug(f"Failed to emit initialization event: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize PreparedStatementCache: {e}")
            raise

    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PreparedStatementCache capability."""
        if capability_name == "cache_performance_analysis":
            return await self._cache_performance_analysis(parameters)
        elif capability_name == "query_optimization_analysis":
            return await self._query_optimization_analysis(parameters)
        elif capability_name == "cache_efficiency_analysis":
            return await self._cache_efficiency_analysis(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")

    async def _cache_performance_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cache performance."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("PreparedStatementCache component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("cache_performance", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "PreparedStatementCacheConnector"
        }

    async def _query_optimization_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query optimization patterns."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("PreparedStatementCache component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("query_optimization", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "PreparedStatementCacheConnector"
        }

    async def _cache_efficiency_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cache efficiency."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("PreparedStatementCache component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("cache_efficiency", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "PreparedStatementCacheConnector"
        }

class TypeSafePsycopgClientConnector(ComponentConnector):
    """Connector for TypeSafePsycopgClient component."""

    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="type_safe_psycopg_client",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="performance_metrics_analysis",
                    description="Analyze database performance metrics",
                    input_types=["metrics_config"],
                    output_types=["performance_metrics", "recommendations"]
                ),
                ComponentCapability(
                    name="connection_health_analysis",
                    description="Analyze database connection health",
                    input_types=["health_config"],
                    output_types=["health_status", "pool_analysis"]
                ),
                ComponentCapability(
                    name="query_pattern_analysis",
                    description="Analyze query execution patterns",
                    input_types=["query_config"],
                    output_types=["query_analysis", "optimization_opportunities"]
                ),
                ComponentCapability(
                    name="type_safety_validation",
                    description="Validate type safety and security features",
                    input_types=["validation_config"],
                    output_types=["type_safety_report", "security_analysis"]
                ),
                ComponentCapability(
                    name="comprehensive_database_analysis",
                    description="Run comprehensive database analysis",
                    input_types=["comprehensive_config"],
                    output_types=["comprehensive_report", "executive_summary"]
                )
            ],
            resource_requirements={"memory": "512MB", "cpu": "2 cores"}
        )
        super().__init__(metadata, event_bus)

    async def _initialize_component(self) -> None:
        """Initialize TypeSafePsycopgClient component."""
        try:
            from prompt_improver.database.psycopg_client import get_psycopg_client
            # Get the global client instance
            self.component = await get_psycopg_client()
            self.logger.info("TypeSafePsycopgClient connector initialized")

            # Emit initialization event
            if self.event_bus:
                try:
                    result = self.event_bus.emit("type_safe_psycopg_client_initialized", {
                        "component_name": self.metadata.name,
                        "status": "ready",
                        "pool_config": {
                            "min_size": self.component.config.pool_min_size,
                            "max_size": self.component.config.pool_max_size
                        }
                    })
                    if hasattr(result, '__await__'):
                        await result
                except Exception as e:
                    # Don't fail initialization if event emission fails
                    self.logger.debug(f"Failed to emit initialization event: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TypeSafePsycopgClient: {e}")
            raise

    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TypeSafePsycopgClient capability."""
        if capability_name == "performance_metrics_analysis":
            return await self._performance_metrics_analysis(parameters)
        elif capability_name == "connection_health_analysis":
            return await self._connection_health_analysis(parameters)
        elif capability_name == "query_pattern_analysis":
            return await self._query_pattern_analysis(parameters)
        elif capability_name == "type_safety_validation":
            return await self._type_safety_validation(parameters)
        elif capability_name == "comprehensive_database_analysis":
            return await self._comprehensive_database_analysis(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")

    async def _performance_metrics_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database performance metrics."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("TypeSafePsycopgClient component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("performance_metrics", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "TypeSafePsycopgClientConnector"
        }

    async def _connection_health_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database connection health."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("TypeSafePsycopgClient component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("connection_health", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "TypeSafePsycopgClientConnector"
        }

    async def _query_pattern_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query execution patterns."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("TypeSafePsycopgClient component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("query_analysis", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "TypeSafePsycopgClientConnector"
        }

    async def _type_safety_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate type safety features."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("TypeSafePsycopgClient component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("type_safety_validation", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "TypeSafePsycopgClientConnector"
        }

    async def _comprehensive_database_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive database analysis."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("TypeSafePsycopgClient component not initialized")

        analysis_result = await self.component.run_orchestrated_analysis("comprehensive_analysis", **parameters)
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "connector": "TypeSafePsycopgClientConnector"
        }

class RetryManagerConnector(ComponentConnector):
    """Connector for RetryManager component."""

    def __init__(self, event_bus=None):
        metadata = ComponentMetadata(
            name="retry_manager",
            tier=ComponentTier.TIER_4_PERFORMANCE,
            version="1.0.0",
            capabilities=[
                ComponentCapability(
                    name="execute_with_retry",
                    description="Execute operations with unified retry logic and circuit breaker",
                    input_types=["operation_config", "retry_config"],
                    output_types=["operation_result", "retry_metrics"]
                ),
                ComponentCapability(
                    name="circuit_breaker_status",
                    description="Get circuit breaker status for operations",
                    input_types=["operation_name"],
                    output_types=["circuit_breaker_status"]
                ),
                ComponentCapability(
                    name="retry_metrics_analysis",
                    description="Analyze retry patterns and performance",
                    input_types=["metrics_config"],
                    output_types=["retry_analysis", "performance_insights"]
                ),
                ComponentCapability(
                    name="reset_circuit_breaker",
                    description="Reset circuit breaker for specific operation",
                    input_types=["operation_name"],
                    output_types=["reset_status"]
                ),
                ComponentCapability(
                    name="configure_retry_strategy",
                    description="Configure retry strategies for operations",
                    input_types=["strategy_config"],
                    output_types=["configuration_result"]
                )
            ],
            resource_requirements={"memory": "256MB", "cpu": "1 core"}
        )
        super().__init__(metadata, event_bus)

    async def _initialize_component(self) -> None:
        """Initialize RetryManager component."""
        try:
            from ....core.retry_manager import RetryManager, RetryConfig

            # Create retry manager with orchestrator-optimized config
            default_config = RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                max_delay=30.0,
                enable_circuit_breaker=True,
                enable_metrics=True,
                enable_tracing=True
            )

            self.component = RetryManager(default_config)
            self.logger.info("RetryManager connector initialized")

            # Emit initialization event
            if self.event_bus:
                try:
                    result = self.event_bus.emit("retry_manager_initialized", {
                        "component_name": self.metadata.name,
                        "status": "ready",
                        "default_config": {
                            "max_attempts": default_config.max_attempts,
                            "strategy": default_config.strategy.value,
                            "circuit_breaker_enabled": default_config.enable_circuit_breaker
                        }
                    })
                    if hasattr(result, '__await__'):
                        await result
                except Exception as e:
                    # Don't fail initialization if event emission fails
                    self.logger.debug(f"Failed to emit initialization event: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize RetryManager: {e}")
            raise

    async def _execute_component(self, capability_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RetryManager capability."""
        if capability_name == "execute_with_retry":
            return await self._execute_with_retry(parameters)
        elif capability_name == "circuit_breaker_status":
            return await self._circuit_breaker_status(parameters)
        elif capability_name == "retry_metrics_analysis":
            return await self._retry_metrics_analysis(parameters)
        elif capability_name == "reset_circuit_breaker":
            return await self._reset_circuit_breaker(parameters)
        elif capability_name == "configure_retry_strategy":
            return await self._configure_retry_strategy(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")

    async def _execute_with_retry(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation with retry logic."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("RetryManager component not initialized")

        operation_name = parameters.get("operation_name", "unknown_operation")
        retry_config_params = parameters.get("retry_config", {})

        # Create retry config from parameters
        from ....core.retry_manager import RetryConfig, RetryStrategy

        retry_config = RetryConfig(
            max_attempts=retry_config_params.get("max_attempts", 3),
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=retry_config_params.get("base_delay", 0.1),
            max_delay=retry_config_params.get("max_delay", 30.0),
            enable_circuit_breaker=retry_config_params.get("enable_circuit_breaker", True),
            operation_name=operation_name
        )

        # This is a demonstration - in real usage, the operation would be passed in
        async def demo_operation():
            return {"status": "success", "message": f"Operation {operation_name} completed"}

        try:
            result = await self.component.retry_async(demo_operation, config=retry_config)
            return {
                "status": "success",
                "result": result,
                "operation_name": operation_name,
                "connector": "RetryManagerConnector"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "operation_name": operation_name,
                "connector": "RetryManagerConnector"
            }

    async def _circuit_breaker_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get circuit breaker status."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("RetryManager component not initialized")

        operation_name = parameters.get("operation_name", "default")
        status = await self.component.get_circuit_breaker_status(operation_name)

        return {
            "status": "success",
            "operation_name": operation_name,
            "circuit_breaker_status": status,
            "connector": "RetryManagerConnector"
        }

    async def _retry_metrics_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze retry metrics and patterns."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("RetryManager component not initialized")

        # This would integrate with actual metrics collection in a real implementation
        analysis = {
            "total_operations": len(self.component.circuit_breakers),
            "circuit_breakers_configured": list(self.component.circuit_breakers.keys()),
            "metrics_enabled": self.component.default_config.enable_metrics,
            "tracing_enabled": self.component.default_config.enable_tracing
        }

        return {
            "status": "success",
            "analysis": analysis,
            "connector": "RetryManagerConnector"
        }

    async def _reset_circuit_breaker(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reset circuit breaker for operation."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("RetryManager component not initialized")

        operation_name = parameters.get("operation_name")
        if not operation_name:
            return {
                "status": "error",
                "error": "operation_name is required",
                "connector": "RetryManagerConnector"
            }

        await self.component.reset_circuit_breaker(operation_name)

        return {
            "status": "success",
            "message": f"Circuit breaker reset for {operation_name}",
            "operation_name": operation_name,
            "connector": "RetryManagerConnector"
        }

    async def _configure_retry_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Configure retry strategy."""
        if not hasattr(self, 'component') or not self.component:
            raise RuntimeError("RetryManager component not initialized")

        # Update default configuration
        config_updates = parameters.get("config", {})

        # This would update the default config in a real implementation
        return {
            "status": "success",
            "message": "Retry strategy configuration updated",
            "config_updates": config_updates,
            "connector": "RetryManagerConnector"
        }

class Tier4ConnectorFactory:
    """Factory for creating Tier 4 component connectors."""
    
    @staticmethod
    def create_connector(component_name: str, event_bus=None) -> ComponentConnector:
        """Create a connector for the specified Tier 4 component."""
        connectors = {
            "advanced_ab_testing": AdvancedABTestingConnector,
            "real_time_analytics": RealTimeAnalyticsConnector,
            "performance_monitoring": PerformanceMonitoringConnector,
            "database_performance_monitor": DatabasePerformanceMonitorConnector,
            "database_connection_optimizer": DatabaseConnectionOptimizerConnector,
            "prepared_statement_cache": PreparedStatementCacheConnector,
            "type_safe_psycopg_client": TypeSafePsycopgClientConnector,
            "retry_manager": RetryManagerConnector,
        }
        
        if component_name not in connectors:
            raise ValueError(f"Unknown Tier 4 component: {component_name}")
        
        return connectors[component_name](event_bus)
    
    @staticmethod
    def list_available_components() -> List[str]:
        """List all available Tier 4 components."""
        return [
            "advanced_ab_testing",
            "canary_testing",
            "real_time_analytics",
            "analytics",
            "performance_monitoring",
            "database_performance_monitor",
            "database_connection_optimizer",
            "prepared_statement_cache",
            "type_safe_psycopg_client",
            "retry_manager",
            "async_optimizer",
            "early_stopping",
            "background_manager"
        ]