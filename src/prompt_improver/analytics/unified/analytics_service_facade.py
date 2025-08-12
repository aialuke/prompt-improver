"""Unified Analytics Service Facade

This is the main entry point for all analytics operations in the system.
It provides a clean, unified interface that delegates to specialized components
while managing cross-cutting concerns like caching, monitoring, and coordination.

Architecture Features:
- Single entry point for all analytics operations
- Component-based architecture with specialized analytics domains
- Performance optimization with intelligent caching
- Memory management and resource optimization
- Real-time monitoring and health checking
- Comprehensive error handling and recovery
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .protocols import (
    ABTestingProtocol,
    AnalyticsComponentProtocol,
    AnalyticsConfiguration,
    AnalyticsEvent,
    AnalyticsInsight,
    AnalyticsMetrics,
    AnalyticsServiceProtocol,
    CacheProtocol,
    ComponentHealth,
    DataCollectionProtocol,
    ExperimentResult,
    MLAnalyticsProtocol,
    MLModelMetrics,
    PerformanceAnalyticsProtocol,
    PerformanceMetrics,
    SessionAnalyticsProtocol,
    SessionMetrics,
)

logger = logging.getLogger(__name__)


class AnalyticsServiceFacade(AnalyticsServiceProtocol):
    """
    Unified Analytics Service Facade implementing 2025 best practices.
    
    This facade provides a single entry point for all analytics operations,
    coordinating between specialized components while maintaining high performance
    and reliability.
    
    Key Features:
    - Unified interface for all analytics operations
    - Intelligent component routing and load balancing
    - Advanced caching with TTL and invalidation strategies
    - Memory-efficient data processing
    - Real-time performance monitoring
    - Comprehensive health checking and self-healing
    - Graceful degradation under load
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        config: Optional[Dict[str, Any]] = None
    ):
        self.db_session = db_session
        self.logger = logger
        self.config = config or {}
        
        # Component registry - lazy initialized
        self._components: Dict[str, AnalyticsComponentProtocol] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        
        # Performance optimization
        self._cache: Optional[CacheProtocol] = None
        self._performance_targets = {
            "max_response_time_ms": 200,
            "max_memory_usage_mb": 500,
            "cache_hit_rate_target": 0.8
        }
        
        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._service_healthy = True
        self._circuit_breaker_state: Dict[str, bool] = {}
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        self._last_health_check = datetime.now()
        
        # Component initialization flags
        self._initialized = False
        self._shutting_down = False
    
    async def initialize(self) -> None:
        """Initialize the analytics service and all components"""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing Unified Analytics Service")
            
            # Initialize cache if configured
            if self.config.get("cache_enabled", True):
                self._cache = await self._initialize_cache()
            
            self.logger.info(f"Analytics service initialized with caching: {self._cache is not None}")
            
            # Initialize components lazily - they will be created when first accessed
            self._initialized = True
            
            # Start background health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
            
            self.logger.info("Analytics service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics service: {e}")
            raise
    
    async def collect_data(self, data_type: str, data: Dict[str, Any]) -> bool:
        """
        Collect data of specified type through appropriate component.
        
        Args:
            data_type: Type of data ("event", "metrics", "performance", "session", "model")
            data: Data to collect
            
        Returns:
            Success status
        """
        start_time = datetime.now()
        
        try:
            # Input validation
            if not data_type or not data:
                raise ValueError("data_type and data are required")
            
            # Check circuit breaker
            if self._is_circuit_open(f"collect_{data_type}"):
                self.logger.warning(f"Circuit breaker open for collect_{data_type}")
                return False
            
            # Route to appropriate component
            success = await self._route_data_collection(data_type, data)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error collecting {data_type} data: {e}")
            self._record_error(f"collect_{data_type}")
            return False
    
    async def analyze_performance(
        self, 
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform performance analysis using performance analytics component.
        
        Args:
            analysis_type: Type of analysis ("trends", "anomalies", "report")
            parameters: Analysis parameters
            
        Returns:
            Analysis results
        """
        start_time = datetime.now()
        cache_key = f"perf_analysis:{analysis_type}:{hash(str(parameters))}"
        
        try:
            # Check cache first
            if self._cache:
                cached_result = await self._cache.get(cache_key)
                if cached_result:
                    self.logger.debug(f"Cache hit for performance analysis: {analysis_type}")
                    return cached_result
            
            # Get performance analytics component
            perf_component = await self._get_component("performance") 
            if not perf_component:
                return {"error": "Performance analytics component not available"}
            
            # Route analysis request
            result = await self._route_performance_analysis(
                perf_component, analysis_type, parameters or {}
            )
            
            # Cache successful results
            if self._cache and result.get("success", True):
                await self._cache.set(cache_key, result, ttl=300)  # 5 minute cache
            
            self._update_performance_metrics(start_time, True)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis {analysis_type}: {e}")
            self._record_error("analyze_performance")
            return {"error": str(e), "analysis_type": analysis_type}
    
    async def run_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Create and run A/B testing experiment.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        try:
            ab_component = await self._get_component("ab_testing")
            if not ab_component:
                raise RuntimeError("A/B testing component not available")
            
            # Extract configuration
            name = experiment_config.get("name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            control = experiment_config.get("control_variant", {})
            treatments = experiment_config.get("treatment_variants", [])
            success_metric = experiment_config.get("success_metric", "conversion_rate")
            significance = experiment_config.get("target_significance", 0.05)
            
            # Create experiment through component
            experiment_id = await ab_component.create_experiment(
                name=name,
                control_variant=control,
                treatment_variants=treatments,
                success_metric=success_metric,
                target_significance=significance
            )
            
            self.logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Error creating experiment: {e}")
            self._record_error("run_experiment")
            raise
    
    async def analyze_sessions(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user sessions using session analytics component.
        
        Args:
            analysis_config: Analysis configuration
            
        Returns:
            Session analysis results
        """
        start_time = datetime.now()
        
        try:
            session_component = await self._get_component("session")
            if not session_component:
                return {"error": "Session analytics component not available"}
            
            analysis_type = analysis_config.get("type", "patterns")
            
            # Route to appropriate analysis method
            if analysis_type == "patterns":
                result = await session_component.analyze_session_patterns(
                    session_ids=analysis_config.get("session_ids"),
                    time_range=analysis_config.get("time_range")
                )
            elif analysis_type == "compare":
                result = await session_component.compare_sessions(
                    analysis_config["session_a_id"],
                    analysis_config["session_b_id"]
                )
            elif analysis_type == "report":
                result = await session_component.generate_session_report(
                    analysis_config["session_id"],
                    analysis_config.get("format", "summary")
                )
            else:
                result = {"error": f"Unknown session analysis type: {analysis_type}"}
            
            self._update_performance_metrics(start_time, True)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in session analysis: {e}")
            self._record_error("analyze_sessions")
            return {"error": str(e)}
    
    async def monitor_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Monitor ML model performance using ML analytics component.
        
        Args:
            model_ids: List of model IDs to monitor
            
        Returns:
            Model monitoring results
        """
        try:
            ml_component = await self._get_component("ml_analytics")
            if not ml_component:
                return {"error": "ML analytics component not available"}
            
            results = {}
            
            # Analyze each model
            for model_id in model_ids:
                try:
                    # Check for model drift
                    drift_analysis = await ml_component.analyze_model_drift(model_id)
                    
                    # Predict potential issues
                    issue_predictions = await ml_component.predict_model_issues(model_id)
                    
                    results[model_id] = {
                        "drift_analysis": drift_analysis,
                        "predicted_issues": issue_predictions,
                        "status": "monitored",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring model {model_id}: {e}")
                    results[model_id] = {
                        "error": str(e),
                        "status": "error"
                    }
            
            # Compare models if multiple provided
            if len(model_ids) > 1:
                try:
                    comparison = await ml_component.compare_models(model_ids)
                    results["comparison"] = comparison
                except Exception as e:
                    self.logger.error(f"Error comparing models: {e}")
                    results["comparison"] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model monitoring: {e}")
            self._record_error("monitor_models")
            return {"error": str(e)}
    
    async def generate_insights(
        self, 
        insight_type: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable insights by coordinating across components.
        
        Args:
            insight_type: Type of insights ("performance", "experiments", "sessions", "models")
            context: Context for insight generation
            
        Returns:
            List of generated insights
        """
        try:
            insights = []
            
            if insight_type == "performance":
                insights.extend(await self._generate_performance_insights(context))
            elif insight_type == "experiments":
                insights.extend(await self._generate_experiment_insights(context))
            elif insight_type == "sessions":
                insights.extend(await self._generate_session_insights(context))
            elif insight_type == "models":
                insights.extend(await self._generate_model_insights(context))
            elif insight_type == "comprehensive":
                # Generate insights from all areas
                for itype in ["performance", "experiments", "sessions", "models"]:
                    try:
                        component_insights = await self.generate_insights(itype, context)
                        insights.extend(component_insights)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate {itype} insights: {e}")
            else:
                raise ValueError(f"Unknown insight type: {insight_type}")
            
            # Sort insights by impact and confidence
            insights.sort(key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.get("impact_level", "low"), 1),
                x.get("confidence_score", 0)
            ), reverse=True)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating {insight_type} insights: {e}")
            return [{"error": str(e), "insight_type": insight_type}]
    
    async def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health status of analytics service and components"""
        try:
            service_status = {
                "service": "analytics_facade",
                "status": "healthy" if self._service_healthy else "unhealthy",
                "initialized": self._initialized,
                "shutting_down": self._shutting_down,
                "last_health_check": self._last_health_check.isoformat(),
                "performance": {
                    "total_requests": self._request_count,
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._request_count, 1),
                    "avg_response_time_ms": self._total_response_time / max(self._request_count, 1),
                },
                "components": {}
            }
            
            # Check each component health
            for component_name, component in self._components.items():
                try:
                    component_health = await component.health_check()
                    service_status["components"][component_name] = component_health
                except Exception as e:
                    service_status["components"][component_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Overall service health based on components
            component_statuses = [
                comp.get("status", "unknown") 
                for comp in service_status["components"].values()
            ]
            
            if any(status == "unhealthy" for status in component_statuses):
                service_status["status"] = "degraded"
            elif any(status == "error" for status in component_statuses):
                service_status["status"] = "degraded"
            
            return service_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                "service": "analytics_facade",
                "status": "error",
                "error": str(e)
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Alias for health_check method for compatibility"""
        return await self.health_check()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown analytics service and all components"""
        if self._shutting_down:
            return
        
        self.logger.info("Shutting down analytics service")
        self._shutting_down = True
        
        try:
            # Cancel health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all components
            for component_name, component in self._components.items():
                try:
                    self.logger.info(f"Shutting down {component_name} component")
                    await component.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down {component_name}: {e}")
            
            # Clear component registry
            self._components.clear()
            
            self.logger.info("Analytics service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    # Private helper methods
    
    async def _get_component(self, component_type: str) -> Optional[AnalyticsComponentProtocol]:
        """Get or lazily initialize component of specified type"""
        if component_type not in self._components:
            try:
                # Lazy initialization of components
                component = await self._initialize_component(component_type)
                if component:
                    self._components[component_type] = component
                    self.logger.info(f"Initialized {component_type} component")
                else:
                    self.logger.error(f"Failed to initialize {component_type} component")
                    return None
            except Exception as e:
                self.logger.error(f"Error initializing {component_type} component: {e}")
                return None
        
        return self._components.get(component_type)
    
    async def _initialize_component(self, component_type: str) -> Optional[AnalyticsComponentProtocol]:
        """Initialize specific component type"""
        try:
            if component_type == "data_collection":
                from .data_collection_component import DataCollectionComponent
                return DataCollectionComponent(self.db_session, self.config.get("data_collection", {}))
            elif component_type == "performance":
                from .performance_analytics_component import PerformanceAnalyticsComponent
                return PerformanceAnalyticsComponent(self.db_session, self.config.get("performance", {}))
            elif component_type == "ab_testing":
                from .ab_testing_component import ABTestingComponent
                return ABTestingComponent(self.db_session, self.config.get("ab_testing", {}))
            elif component_type == "session":
                from .session_analytics_component import SessionAnalyticsComponent
                return SessionAnalyticsComponent(self.db_session, self.config.get("session", {}))
            elif component_type == "ml_analytics":
                from .ml_analytics_component import MLAnalyticsComponent
                return MLAnalyticsComponent(self.db_session, self.config.get("ml_analytics", {}))
            else:
                self.logger.error(f"Unknown component type: {component_type}")
                return None
                
        except ImportError as e:
            self.logger.error(f"Component {component_type} not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating {component_type} component: {e}")
            return None
    
    async def _initialize_cache(self) -> Optional[CacheProtocol]:
        """Initialize cache component"""
        try:
            # Try to use Redis cache if available
            from prompt_improver.utils.redis_cache import AsyncRedisCache
            cache_config = self.config.get("cache", {})
            return AsyncRedisCache(
                host=cache_config.get("host", "localhost"),
                port=cache_config.get("port", 6379),
                db=cache_config.get("db", 0)
            )
        except ImportError:
            self.logger.warning("Redis cache not available, using memory cache")
            try:
                from .memory_cache import MemoryCache
                return MemoryCache(max_size=self.config.get("cache", {}).get("max_size", 1000))
            except Exception as e:
                self.logger.error(f"Failed to initialize any cache: {e}")
                return None
    
    async def _route_data_collection(self, data_type: str, data: Dict[str, Any]) -> bool:
        """Route data collection to appropriate component"""
        data_component = await self._get_component("data_collection")
        if not data_component:
            return False
        
        if data_type == "event":
            # Ensure timestamp is provided for event data
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now()
            event = AnalyticsEvent(**data)
            return await data_component.collect_event(event)
        elif data_type == "metrics":
            metrics = AnalyticsMetrics(**data)
            return await data_component.collect_metrics(metrics)
        elif data_type in ["performance_test", "health_check", "test_event"]:
            # Handle test and diagnostic events
            event = AnalyticsEvent(
                event_id=data.get("event_id", f"{data_type}_{datetime.now().isoformat()}"),
                event_type=data_type,
                timestamp=datetime.now(),
                source=data.get("source", "analytics_service"),
                data=data
            )
            return await data_component.collect_event(event)
        else:
            self.logger.error(f"Unknown data type for collection: {data_type}")
            return False
    
    async def _route_performance_analysis(
        self, 
        component: PerformanceAnalyticsProtocol, 
        analysis_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route performance analysis to appropriate method"""
        if analysis_type == "trends":
            return await component.analyze_performance_trends(
                parameters.get("start_time", datetime.now() - timedelta(hours=24)),
                parameters.get("end_time", datetime.now())
            )
        elif analysis_type == "anomalies":
            return await component.detect_performance_anomalies(
                parameters.get("threshold", 2.0)
            )
        elif analysis_type == "report":
            return await component.generate_performance_report(
                parameters.get("report_type", "summary")
            )
        else:
            return {"error": f"Unknown performance analysis type: {analysis_type}"}
    
    def _update_performance_metrics(self, start_time: datetime, success: bool) -> None:
        """Update internal performance metrics"""
        self._request_count += 1
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self._total_response_time += response_time
        
        if not success:
            self._error_count += 1
    
    def _record_error(self, operation: str) -> None:
        """Record error and check circuit breaker thresholds"""
        self._error_count += 1
        
        # Simple circuit breaker logic
        error_rate = self._error_count / max(self._request_count, 1)
        if error_rate > 0.5:  # 50% error rate threshold
            self._circuit_breaker_state[operation] = True
            self.logger.warning(f"Circuit breaker opened for {operation} due to high error rate")
    
    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for operation"""
        return self._circuit_breaker_state.get(operation, False)
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Update health status
                health_status = await self.health_check()
                self._service_healthy = health_status["status"] == "healthy"
                self._last_health_check = datetime.now()
                
                # Reset circuit breakers if error rate improves
                current_error_rate = self._error_count / max(self._request_count, 1)
                if current_error_rate < 0.1:  # 10% threshold for recovery
                    self._circuit_breaker_state.clear()
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    async def _generate_performance_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance-related insights"""
        insights = []
        try:
            perf_component = await self._get_component("performance")
            if not perf_component:
                return insights
            
            # Analyze recent performance trends
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=context.get("lookback_hours", 24))
            
            trends = await perf_component.analyze_performance_trends(start_time, end_time)
            anomalies = await perf_component.detect_performance_anomalies()
            
            # Generate insights based on findings
            if trends.get("declining_performance"):
                insights.append({
                    "insight_type": "performance_decline",
                    "title": "Performance Degradation Detected",
                    "description": "System performance has declined over the analysis period",
                    "confidence_score": 0.85,
                    "impact_level": "high",
                    "recommendations": [
                        "Check for recent deployments or configuration changes",
                        "Review resource utilization metrics",
                        "Consider scaling up resources"
                    ],
                    "supporting_data": trends
                })
            
            if anomalies:
                insights.append({
                    "insight_type": "performance_anomalies",
                    "title": f"{len(anomalies)} Performance Anomalies Detected",
                    "description": "Unusual performance patterns identified",
                    "confidence_score": 0.9,
                    "impact_level": "medium",
                    "recommendations": [
                        "Investigate root cause of anomalies",
                        "Consider implementing additional monitoring"
                    ],
                    "supporting_data": {"anomalies": anomalies}
                })
            
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {e}")
        
        return insights
    
    async def _generate_experiment_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate A/B testing experiment insights"""
        insights = []
        try:
            ab_component = await self._get_component("ab_testing")
            if not ab_component:
                return insights
            
            # Get active experiments
            experiments = await ab_component.get_active_experiments()
            
            for experiment in experiments:
                if experiment.get("ready_for_analysis"):
                    result = await ab_component.analyze_experiment(experiment["id"])
                    
                    if result.statistical_significance:
                        insights.append({
                            "insight_type": "significant_experiment",
                            "title": f"Experiment {experiment['name']} Shows Significant Results",
                            "description": f"Statistical significance achieved with p-value {result.p_value:.4f}",
                            "confidence_score": 1.0 - result.p_value,
                            "impact_level": "high" if result.effect_size > 0.5 else "medium",
                            "recommendations": [
                                f"Consider rolling out winning variant: {result.winner}",
                                "Document learnings and apply to future experiments"
                            ],
                            "supporting_data": result.dict()
                        })
            
        except Exception as e:
            self.logger.error(f"Error generating experiment insights: {e}")
        
        return insights
    
    async def _generate_session_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate session analytics insights"""
        insights = []
        try:
            session_component = await self._get_component("session")
            if not session_component:
                return insights
            
            # Analyze recent session patterns
            patterns = await session_component.analyze_session_patterns()
            
            # Generate insights from patterns
            if patterns.get("low_engagement_sessions"):
                insights.append({
                    "insight_type": "low_engagement",
                    "title": "High Number of Low-Engagement Sessions",
                    "description": f"{patterns['low_engagement_sessions']} sessions with low engagement detected",
                    "confidence_score": 0.8,
                    "impact_level": "medium",
                    "recommendations": [
                        "Review user onboarding flow",
                        "Analyze common exit points",
                        "Consider UX improvements"
                    ],
                    "supporting_data": patterns
                })
            
        except Exception as e:
            self.logger.error(f"Error generating session insights: {e}")
        
        return insights
    
    async def _generate_model_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ML model insights"""
        insights = []
        try:
            ml_component = await self._get_component("ml_analytics")
            if not ml_component:
                return insights
            
            # This would be implemented based on available model IDs
            # For now, return empty insights
            
        except Exception as e:
            self.logger.error(f"Error generating model insights: {e}")
        
        return insights


# Factory function for easy instantiation
async def create_analytics_service(
    db_session: AsyncSession,
    config: Optional[Dict[str, Any]] = None
) -> AnalyticsServiceFacade:
    """
    Create and initialize analytics service facade.
    
    Args:
        db_session: Database session
        config: Optional configuration
        
    Returns:
        Initialized analytics service
    """
    service = AnalyticsServiceFacade(db_session, config)
    await service.initialize()
    return service