"""
Unified Health Monitoring Orchestration System - 2025 SRE Best Practices

This module provides a comprehensive health monitoring orchestration system that:
- Unified Health Dashboard: Comprehensive health overview with all systems
- Health Check Orchestration: Coordinate health checks with dependencies and timeout handling  
- Alerting Integration: Connect to monitoring stack for automated alerts
- Performance Aggregation: Combine metrics from all health systems
- Circuit Breaker Coordination: Coordinate circuit breakers across all systems
- Health Score Algorithm: Overall system health scoring (0-100)
- Dependency Chain Management: Handle health check dependencies and cascade failures

Integrates with all existing health monitoring systems:
- Enhanced Health Service (general health)
- ML Health Integration Manager (ML-specific)
- External API Health Monitor (API dependencies) 
- Redis Health Monitor (cache)
- Database Health Monitor (persistence)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import statistics

# Prometheus/observability imports
try:
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import existing health monitoring systems
from ..performance.monitoring.health.service import get_health_service
from ..performance.monitoring.health.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..ml.health.integration_manager import get_ml_health_integration_manager
from .external_api_health import ExternalAPIHealthMonitor
from ..cache.redis_health import RedisHealthMonitor, get_redis_health_summary
from ..database.health.database_health_monitor import get_database_health_monitor

logger = logging.getLogger(__name__)

class SystemHealthStatus(Enum):
    """Overall system health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"

class DependencyLevel(Enum):
    """Dependency criticality levels"""
    CRITICAL = "critical"      # System cannot function without this
    IMPORTANT = "important"    # System degraded without this
    OPTIONAL = "optional"      # System can function without this

class AlertSeverity(Enum):
    """Alert severity levels for monitoring integration"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class HealthComponent:
    """Health monitoring component configuration"""
    name: str
    category: str  # database, cache, ml, external_api, system
    dependency_level: DependencyLevel
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    circuit_breaker_enabled: bool = True
    weight: float = 1.0  # Weight for health score calculation
    dependencies: List[str] = field(default_factory=list)  # Other components this depends on
    
@dataclass
class HealthCheckResult:
    """Result of a health check execution"""
    component: str
    status: SystemHealthStatus
    response_time_ms: float
    timestamp: datetime
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_checks: Dict[str, Any] = field(default_factory=dict)
    health_score: float = 100.0  # Component-specific health score (0-100)

@dataclass
class SystemHealthSnapshot:
    """Complete system health snapshot"""
    overall_status: SystemHealthStatus
    overall_health_score: float  # 0-100
    timestamp: datetime
    execution_time_ms: float
    
    # Component results
    component_results: Dict[str, HealthCheckResult]
    
    # Dependency analysis
    dependency_chain_status: Dict[str, str]
    cascade_failures: List[str]
    
    # Performance metrics
    aggregated_metrics: Dict[str, Any]
    
    # Circuit breaker status
    circuit_breaker_states: Dict[str, Dict[str, Any]]
    
    # Alerts generated
    active_alerts: List[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[str]

class HealthScoreCalculator:
    """Calculates overall system health score (0-100) with weighted components"""
    
    def __init__(self):
        # Component weights for health score calculation
        self.component_weights = {
            "database": 0.30,      # Critical - 30%
            "cache": 0.15,         # Important - 15% 
            "ml_system": 0.25,     # Important - 25%
            "external_apis": 0.15, # Important - 15%
            "system_resources": 0.10, # Optional - 10%
            "application": 0.05    # Optional - 5%
        }
        
        # Threshold mappings
        self.status_score_mapping = {
            SystemHealthStatus.HEALTHY: (90, 100),
            SystemHealthStatus.DEGRADED: (70, 89),
            SystemHealthStatus.CRITICAL: (30, 69),
            SystemHealthStatus.FAILED: (0, 29)
        }
    
    def calculate_component_score(
        self, 
        component_result: HealthCheckResult,
        component_config: HealthComponent
    ) -> float:
        """Calculate health score for individual component (0-100)"""
        base_score = component_result.health_score
        
        # Adjust based on response time (penalize slow responses)
        response_penalty = 0
        if component_result.response_time_ms > 5000:  # >5s is very slow
            response_penalty = 20
        elif component_result.response_time_ms > 1000:  # >1s is slow
            response_penalty = 10
        elif component_result.response_time_ms > 500:   # >500ms is concerning
            response_penalty = 5
            
        # Adjust based on status
        status_multiplier = 1.0
        if component_result.status == SystemHealthStatus.HEALTHY:
            status_multiplier = 1.0
        elif component_result.status == SystemHealthStatus.DEGRADED:
            status_multiplier = 0.7
        elif component_result.status == SystemHealthStatus.CRITICAL:
            status_multiplier = 0.4
        else:  # FAILED
            status_multiplier = 0.0
            
        final_score = max(0, (base_score - response_penalty) * status_multiplier)
        return min(100, final_score)
    
    def calculate_overall_score(
        self, 
        component_results: Dict[str, HealthCheckResult],
        component_configs: Dict[str, HealthComponent]
    ) -> float:
        """Calculate overall system health score (0-100)"""
        weighted_scores = []
        total_weight = 0
        
        for component_name, result in component_results.items():
            config = component_configs.get(component_name)
            if not config:
                continue
                
            component_score = self.calculate_component_score(result, config)
            
            # Apply dependency level weighting
            dependency_weight = 1.0
            if config.dependency_level == DependencyLevel.CRITICAL:
                dependency_weight = 1.5  # Critical components get 50% more weight
            elif config.dependency_level == DependencyLevel.IMPORTANT:
                dependency_weight = 1.0
            else:  # OPTIONAL
                dependency_weight = 0.5  # Optional components get 50% less weight
                
            # Get category weight
            category_weight = self.component_weights.get(config.category, 0.05)
            
            # Calculate final weighted score
            final_weight = config.weight * dependency_weight * category_weight
            weighted_scores.append(component_score * final_weight)
            total_weight += final_weight
        
        if total_weight == 0:
            return 0.0
            
        overall_score = sum(weighted_scores) / total_weight
        return min(100.0, max(0.0, overall_score))
    
    def determine_overall_status(self, overall_score: float) -> SystemHealthStatus:
        """Determine overall system status from health score"""
        if overall_score >= 90:
            return SystemHealthStatus.HEALTHY
        elif overall_score >= 70:
            return SystemHealthStatus.DEGRADED
        elif overall_score >= 30:
            return SystemHealthStatus.CRITICAL
        else:
            return SystemHealthStatus.FAILED

class DependencyManager:
    """Manages dependency chains and cascade failure detection"""
    
    def __init__(self, components: Dict[str, HealthComponent]):
        self.components = components
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build dependency graph from component configurations"""
        graph = defaultdict(set)
        
        for component_name, config in self.components.items():
            for dependency in config.dependencies:
                graph[dependency].add(component_name)  # dependency -> dependent
                
        return dict(graph)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get optimal execution order for health checks based on dependencies"""
        # Topological sort to get execution order
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for component_name, config in self.components.items():
            for dependency in config.dependencies:
                in_degree[component_name] += 1
                
        # Group components by dependency level for parallel execution
        execution_groups = []
        remaining_components = set(self.components.keys())
        
        while remaining_components:
            # Find components with no remaining dependencies
            current_group = []
            for component in list(remaining_components):
                config = self.components[component]
                if all(dep not in remaining_components for dep in config.dependencies):
                    current_group.append(component)
            
            if not current_group:
                # Circular dependency detected - add remaining components
                current_group = list(remaining_components)
                logger.warning("Circular dependency detected in health check components")
            
            execution_groups.append(current_group)
            remaining_components -= set(current_group)
        
        return execution_groups
    
    def analyze_cascade_failures(
        self, 
        component_results: Dict[str, HealthCheckResult]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Analyze cascade failures and dependency impact"""
        cascade_failures = []
        dependency_status = {}
        
        # Find failed components
        failed_components = {
            name for name, result in component_results.items()
            if result.status in [SystemHealthStatus.CRITICAL, SystemHealthStatus.FAILED]
        }
        
        # Analyze cascade impact
        for failed_component in failed_components:
            affected_components = self._get_dependent_components(failed_component)
            for affected in affected_components:
                if affected in component_results:
                    cascade_failures.append(f"{failed_component} -> {affected}")
                    dependency_status[affected] = f"degraded_due_to_{failed_component}"
        
        return cascade_failures, dependency_status
    
    def _get_dependent_components(self, component: str) -> Set[str]:
        """Get all components that depend on the given component"""
        dependents = set()
        queue = [component]
        
        while queue:
            current = queue.pop(0)
            if current in self.dependency_graph:
                for dependent in self.dependency_graph[current]:
                    if dependent not in dependents:
                        dependents.add(dependent)
                        queue.append(dependent)
        
        return dependents

class PerformanceAggregator:
    """Aggregates performance metrics from all health systems"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)  # Keep last 100 snapshots
    
    def aggregate_metrics(
        self, 
        component_results: Dict[str, HealthCheckResult]
    ) -> Dict[str, Any]:
        """Aggregate performance metrics from all components"""
        aggregated = {
            "response_times": {},
            "availability": {},
            "throughput": {},
            "error_rates": {},
            "sla_compliance": {},
            "trend_analysis": {}
        }
        
        # Response time analysis
        response_times = [r.response_time_ms for r in component_results.values()]
        if response_times:
            aggregated["response_times"] = {
                "p50": statistics.median(response_times),
                "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0],
                "p99": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else response_times[0],
                "avg": statistics.mean(response_times),
                "max": max(response_times),
                "min": min(response_times)
            }
        
        # Availability analysis
        total_components = len(component_results)
        healthy_components = len([
            r for r in component_results.values() 
            if r.status == SystemHealthStatus.HEALTHY
        ])
        
        aggregated["availability"] = {
            "overall_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "degraded_components": len([
                r for r in component_results.values()
                if r.status == SystemHealthStatus.DEGRADED
            ]),
            "failed_components": len([
                r for r in component_results.values()
                if r.status in [SystemHealthStatus.CRITICAL, SystemHealthStatus.FAILED]
            ])
        }
        
        # Component-specific metrics
        for component_name, result in component_results.items():
            if result.metadata:
                # Extract component-specific metrics
                if "throughput" in result.metadata:
                    aggregated["throughput"][component_name] = result.metadata["throughput"]
                if "error_rate" in result.metadata:
                    aggregated["error_rates"][component_name] = result.metadata["error_rate"]
                if "sla_compliance" in result.metadata:
                    aggregated["sla_compliance"][component_name] = result.metadata["sla_compliance"]
        
        # Store for trend analysis
        self.metrics_history.append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": aggregated.copy()
        })
        
        # Calculate trends
        aggregated["trend_analysis"] = self._calculate_trends()
        
        return aggregated
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trend analysis from historical metrics"""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Analyze response time trends
        recent_response_times = []
        older_response_times = []
        
        history_list = list(self.metrics_history)
        midpoint = len(history_list) // 2
        
        for entry in history_list[midpoint:]:
            if "response_times" in entry["metrics"] and "avg" in entry["metrics"]["response_times"]:
                recent_response_times.append(entry["metrics"]["response_times"]["avg"])
        
        for entry in history_list[:midpoint]:
            if "response_times" in entry["metrics"] and "avg" in entry["metrics"]["response_times"]:
                older_response_times.append(entry["metrics"]["response_times"]["avg"])
        
        trends = {}
        
        if recent_response_times and older_response_times:
            recent_avg = statistics.mean(recent_response_times)
            older_avg = statistics.mean(older_response_times)
            
            if recent_avg > older_avg * 1.2:
                trends["response_time"] = "degrading"
            elif recent_avg < older_avg * 0.8:
                trends["response_time"] = "improving"
            else:
                trends["response_time"] = "stable"
        
        # Analyze availability trends
        recent_availability = []
        older_availability = []
        
        for entry in history_list[midpoint:]:
            if "availability" in entry["metrics"] and "overall_percentage" in entry["metrics"]["availability"]:
                recent_availability.append(entry["metrics"]["availability"]["overall_percentage"])
        
        for entry in history_list[:midpoint]:
            if "availability" in entry["metrics"] and "overall_percentage" in entry["metrics"]["availability"]:
                older_availability.append(entry["metrics"]["availability"]["overall_percentage"])
        
        if recent_availability and older_availability:
            recent_avg = statistics.mean(recent_availability)
            older_avg = statistics.mean(older_availability)
            
            if recent_avg < older_avg - 5:  # 5% drop in availability
                trends["availability"] = "degrading"
            elif recent_avg > older_avg + 5:  # 5% improvement
                trends["availability"] = "improving"
            else:
                trends["availability"] = "stable"
        
        return trends

class AlertingManager:
    """Manages alerting integration with Prometheus/Grafana"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self.registry = None
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring integration"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.health_score_gauge = Gauge(
            'system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )
        
        self.component_health_gauge = Gauge(
            'component_health_score',
            'Individual component health scores',
            ['component', 'category'],
            registry=self.registry
        )
        
        self.active_alerts_gauge = Gauge(
            'active_health_alerts_total',
            'Number of active health alerts',
            ['severity'],
            registry=self.registry
        )
        
        self.health_check_duration_histogram = Histogram(
            'health_check_duration_seconds',
            'Health check execution duration',
            ['component'],
            registry=self.registry
        )
        
        self.cascade_failures_counter = Counter(
            'cascade_failures_total',
            'Total number of cascade failures detected',
            ['source_component', 'affected_component'],
            registry=self.registry
        )
    
    def process_health_snapshot(self, snapshot: SystemHealthSnapshot):
        """Process health snapshot and generate alerts"""
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.registry:
            self._update_prometheus_metrics(snapshot)
        
        # Generate new alerts
        new_alerts = self._generate_alerts(snapshot)
        
        # Update active alerts
        self._update_active_alerts(new_alerts)
        
        # Store in snapshot
        snapshot.active_alerts = list(self.active_alerts.values())
    
    def _update_prometheus_metrics(self, snapshot: SystemHealthSnapshot):
        """Update Prometheus metrics from health snapshot"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # Update overall health score
        self.health_score_gauge.set(snapshot.overall_health_score)
        
        # Update component health scores
        for component_name, result in snapshot.component_results.items():
            category = "unknown"  # Would need component config to get actual category
            self.component_health_gauge.labels(
                component=component_name,
                category=category
            ).set(result.health_score)
            
            # Update health check duration
            self.health_check_duration_histogram.labels(
                component=component_name
            ).observe(result.response_time_ms / 1000.0)
        
        # Update active alerts by severity
        alert_counts = defaultdict(int)
        for alert in snapshot.active_alerts:
            alert_counts[alert.get("severity", "unknown")] += 1
        
        for severity in ["critical", "warning", "info"]:
            self.active_alerts_gauge.labels(severity=severity).set(alert_counts[severity])
        
        # Update cascade failure metrics
        for cascade in snapshot.cascade_failures:
            if " -> " in cascade:
                source, affected = cascade.split(" -> ", 1)
                self.cascade_failures_counter.labels(
                    source_component=source,
                    affected_component=affected
                ).inc()
    
    def _generate_alerts(self, snapshot: SystemHealthSnapshot) -> List[Dict[str, Any]]:
        """Generate alerts based on health snapshot"""
        alerts = []
        current_time = datetime.now(timezone.utc)
        
        # Overall system health alerts
        if snapshot.overall_health_score < 30:
            alerts.append({
                "id": "system_health_critical",
                "severity": AlertSeverity.CRITICAL.value,
                "title": "System Health Critical",
                "message": f"Overall system health score is {snapshot.overall_health_score:.1f}%",
                "timestamp": current_time,
                "component": "system",
                "health_score": snapshot.overall_health_score
            })
        elif snapshot.overall_health_score < 70:
            alerts.append({
                "id": "system_health_degraded",
                "severity": AlertSeverity.WARNING.value,
                "title": "System Health Degraded", 
                "message": f"Overall system health score is {snapshot.overall_health_score:.1f}%",
                "timestamp": current_time,
                "component": "system",
                "health_score": snapshot.overall_health_score
            })
        
        # Component-specific alerts
        for component_name, result in snapshot.component_results.items():
            if result.status == SystemHealthStatus.FAILED:
                alerts.append({
                    "id": f"{component_name}_failed",
                    "severity": AlertSeverity.CRITICAL.value,
                    "title": f"{component_name} Failed",
                    "message": f"{component_name} health check failed: {result.error or result.message}",
                    "timestamp": current_time,
                    "component": component_name,
                    "health_score": result.health_score
                })
            elif result.status == SystemHealthStatus.CRITICAL:
                alerts.append({
                    "id": f"{component_name}_critical",
                    "severity": AlertSeverity.CRITICAL.value,
                    "title": f"{component_name} Critical",
                    "message": f"{component_name} is in critical state: {result.message}",
                    "timestamp": current_time,
                    "component": component_name,
                    "health_score": result.health_score
                })
            elif result.status == SystemHealthStatus.DEGRADED:
                alerts.append({
                    "id": f"{component_name}_degraded",
                    "severity": AlertSeverity.WARNING.value,
                    "title": f"{component_name} Degraded",
                    "message": f"{component_name} performance is degraded: {result.message}",
                    "timestamp": current_time,
                    "component": component_name,
                    "health_score": result.health_score
                })
        
        # Cascade failure alerts
        if snapshot.cascade_failures:
            alerts.append({
                "id": "cascade_failures_detected",
                "severity": AlertSeverity.CRITICAL.value,
                "title": "Cascade Failures Detected",
                "message": f"Detected {len(snapshot.cascade_failures)} cascade failures",
                "timestamp": current_time,
                "component": "system",
                "cascade_failures": snapshot.cascade_failures
            })
        
        # Performance alerts
        aggregated = snapshot.aggregated_metrics
        if "availability" in aggregated:
            availability = aggregated["availability"]["overall_percentage"]
            if availability < 95:
                alerts.append({
                    "id": "low_availability",
                    "severity": AlertSeverity.WARNING.value,
                    "title": "Low System Availability",
                    "message": f"System availability is {availability:.1f}%",
                    "timestamp": current_time,
                    "component": "system",
                    "availability": availability
                })
        
        return alerts
    
    def _update_active_alerts(self, new_alerts: List[Dict[str, Any]]):
        """Update active alerts, handling alert lifecycle"""
        current_time = datetime.now(timezone.utc)
        
        # Add new alerts
        for alert in new_alerts:
            alert_id = alert["id"]
            if alert_id not in self.active_alerts:
                # New alert
                alert["first_seen"] = current_time
                alert["last_seen"] = current_time
                alert["count"] = 1
                self.active_alerts[alert_id] = alert
                
                # Add to history
                self.alert_history.append({
                    "action": "created",
                    "alert": alert.copy(),
                    "timestamp": current_time
                })
                
                logger.warning(f"Health alert created: {alert['title']} - {alert['message']}")
            else:
                # Existing alert - update last seen and count
                existing = self.active_alerts[alert_id]
                existing["last_seen"] = current_time
                existing["count"] += 1
                existing.update({
                    k: v for k, v in alert.items() 
                    if k not in ["first_seen", "count"]
                })
        
        # Check for resolved alerts (not in new alerts)
        new_alert_ids = {alert["id"] for alert in new_alerts}
        resolved_alerts = []
        
        for alert_id in list(self.active_alerts.keys()):
            if alert_id not in new_alert_ids:
                # Alert resolved
                resolved_alert = self.active_alerts.pop(alert_id)
                resolved_alerts.append(resolved_alert)
                
                # Add to history
                self.alert_history.append({
                    "action": "resolved",
                    "alert": resolved_alert,
                    "timestamp": current_time
                })
                
                logger.info(f"Health alert resolved: {resolved_alert['title']}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in exposition format"""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return "# Prometheus not available\n"
        
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')

class CircuitBreakerCoordinator:
    """Coordinates circuit breakers across all health monitoring systems"""
    
    def __init__(self):
        self.component_breakers = {}
        self.global_breaker_config = CircuitBreakerConfig.from_global_config()
    
    def get_or_create_breaker(
        self, 
        component_name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        if component_name not in self.component_breakers:
            breaker_config = config or self.global_breaker_config
            self.component_breakers[component_name] = CircuitBreaker(
                name=component_name,
                config=breaker_config,
                on_state_change=self._on_breaker_state_change
            )
        
        return self.component_breakers[component_name]
    
    def _on_breaker_state_change(self, component_name: str, new_state):
        """Handle circuit breaker state changes"""
        logger.info(f"Circuit breaker state change: {component_name} -> {new_state.value}")
        
        # Additional logic for coordinated circuit breaker management
        if new_state.value == "open":
            # Component is failing - check if we need to open dependent breakers
            self._handle_cascade_breaker_opening(component_name)
    
    def _handle_cascade_breaker_opening(self, failed_component: str):
        """Handle cascade circuit breaker opening when a component fails"""
        # This would integrate with the DependencyManager to determine
        # which other components should have their breakers opened
        logger.warning(f"Evaluating cascade circuit breaker opening due to {failed_component} failure")
    
    def get_all_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        states = {}
        
        for component_name, breaker in self.component_breakers.items():
            states[component_name] = breaker.get_metrics()
        
        return states
    
    def reset_all_breakers(self):
        """Reset all circuit breakers (emergency recovery)"""
        for breaker in self.component_breakers.values():
            breaker.reset()
        
        logger.warning("All circuit breakers have been manually reset")

class HealthOrchestrator:
    """
    Unified Health Monitoring Orchestration System
    
    Central orchestrator that coordinates all health monitoring systems:
    - Manages health check execution with dependency ordering
    - Aggregates results from all health systems
    - Calculates overall system health score
    - Manages circuit breakers and alerting
    - Provides unified dashboard and API
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize component configuration
        self.components = self._initialize_components()
        
        # Initialize sub-systems
        self.health_score_calculator = HealthScoreCalculator()
        self.dependency_manager = DependencyManager(self.components)
        self.performance_aggregator = PerformanceAggregator()
        self.alerting_manager = AlertingManager()
        self.circuit_breaker_coordinator = CircuitBreakerCoordinator()
        
        # Health monitoring system instances
        self._enhanced_health_service = None
        self._ml_health_manager = None
        self._external_api_monitor = None
        self._redis_health_monitor = None
        self._database_health_monitor = None
        
        # Execution metrics
        self._last_execution_time = None
        self._execution_history = deque(maxlen=100)
        
        self.logger.info("Health Orchestrator initialized with comprehensive monitoring capabilities")
    
    def _initialize_components(self) -> Dict[str, HealthComponent]:
        """Initialize health monitoring component configurations"""
        components = {
            # Database components
            "database_primary": HealthComponent(
                name="database_primary",
                category="database",
                dependency_level=DependencyLevel.CRITICAL,
                timeout_seconds=15.0,
                weight=1.0,
                dependencies=[]
            ),
            
            # Cache components
            "redis_cache": HealthComponent(
                name="redis_cache", 
                category="cache",
                dependency_level=DependencyLevel.IMPORTANT,
                timeout_seconds=10.0,
                weight=0.8,
                dependencies=[]
            ),
            
            # ML system components
            "ml_models": HealthComponent(
                name="ml_models",
                category="ml_system", 
                dependency_level=DependencyLevel.IMPORTANT,
                timeout_seconds=30.0,
                weight=1.0,
                dependencies=["database_primary", "redis_cache"]
            ),
            
            "ml_orchestrator": HealthComponent(
                name="ml_orchestrator",
                category="ml_system",
                dependency_level=DependencyLevel.IMPORTANT,
                timeout_seconds=20.0,
                weight=0.9,
                dependencies=["database_primary", "ml_models"]
            ),
            
            # External API components
            "external_apis": HealthComponent(
                name="external_apis",
                category="external_apis",
                dependency_level=DependencyLevel.IMPORTANT,
                timeout_seconds=15.0,
                weight=0.7,
                dependencies=[]
            ),
            
            # System components
            "system_resources": HealthComponent(
                name="system_resources",
                category="system_resources",
                dependency_level=DependencyLevel.OPTIONAL,
                timeout_seconds=5.0,
                weight=0.5,
                dependencies=[]
            ),
            
            # Application components
            "application_health": HealthComponent(
                name="application_health",
                category="application",
                dependency_level=DependencyLevel.OPTIONAL,
                timeout_seconds=10.0,
                weight=0.6,
                dependencies=["database_primary", "redis_cache"]
            )
        }
        
        return components
    
    async def execute_comprehensive_health_check(
        self,
        timeout_seconds: float = 120.0,
        parallel_execution: bool = True,
        include_deep_analysis: bool = True
    ) -> SystemHealthSnapshot:
        """
        Execute comprehensive health check across all systems
        
        Args:
            timeout_seconds: Maximum time to spend on health checks
            parallel_execution: Whether to run checks in parallel where possible
            include_deep_analysis: Whether to include deep analysis (performance, trends, etc.)
            
        Returns:
            SystemHealthSnapshot with complete system health information
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info("Starting comprehensive system health check")
        
        try:
            # Initialize health monitoring systems if needed
            await self._initialize_health_systems()
            
            # Get execution order based on dependencies
            execution_groups = self.dependency_manager.get_execution_order()
            
            # Execute health checks in dependency order
            component_results = {}
            
            if parallel_execution:
                component_results = await self._execute_parallel_health_checks(
                    execution_groups, timeout_seconds
                )
            else:
                component_results = await self._execute_sequential_health_checks(
                    execution_groups, timeout_seconds
                )
            
            # Calculate overall health score
            overall_score = self.health_score_calculator.calculate_overall_score(
                component_results, self.components
            )
            overall_status = self.health_score_calculator.determine_overall_status(overall_score)
            
            # Analyze dependencies and cascade failures
            cascade_failures, dependency_status = self.dependency_manager.analyze_cascade_failures(
                component_results
            )
            
            # Aggregate performance metrics
            aggregated_metrics = self.performance_aggregator.aggregate_metrics(component_results)
            
            # Get circuit breaker states
            circuit_breaker_states = self.circuit_breaker_coordinator.get_all_breaker_states()
            
            # Create health snapshot
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            snapshot = SystemHealthSnapshot(
                overall_status=overall_status,
                overall_health_score=overall_score,
                timestamp=start_time,
                execution_time_ms=execution_time,
                component_results=component_results,
                dependency_chain_status=dependency_status,
                cascade_failures=cascade_failures,
                aggregated_metrics=aggregated_metrics,
                circuit_breaker_states=circuit_breaker_states,
                active_alerts=[],  # Will be populated by alerting manager
                recommendations=[]  # Will be populated below
            )
            
            # Process through alerting manager (generates alerts and updates Prometheus)
            self.alerting_manager.process_health_snapshot(snapshot)
            
            # Generate recommendations
            snapshot.recommendations = self._generate_recommendations(snapshot)
            
            # Store execution metrics
            self._last_execution_time = execution_time
            self._execution_history.append({
                "timestamp": start_time,
                "execution_time_ms": execution_time,
                "overall_score": overall_score,
                "overall_status": overall_status.value
            })
            
            self.logger.info(
                f"Comprehensive health check completed in {execution_time:.2f}ms. "
                f"Overall status: {overall_status.value} (score: {overall_score:.1f})"
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Comprehensive health check failed: {e}", exc_info=True)
            
            # Return failed snapshot
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return SystemHealthSnapshot(
                overall_status=SystemHealthStatus.FAILED,
                overall_health_score=0.0,
                timestamp=start_time,
                execution_time_ms=execution_time,
                component_results={},
                dependency_chain_status={},
                cascade_failures=[],
                aggregated_metrics={},
                circuit_breaker_states={},
                active_alerts=[{
                    "id": "health_check_execution_failed",
                    "severity": AlertSeverity.CRITICAL.value,
                    "title": "Health Check Execution Failed",
                    "message": f"Health check orchestration failed: {str(e)}",
                    "timestamp": datetime.now(timezone.utc),
                    "component": "orchestrator"
                }],
                recommendations=["Investigate health check orchestration system failure"]
            )
    
    async def _initialize_health_systems(self):
        """Initialize all health monitoring systems"""
        if self._enhanced_health_service is None:
            self._enhanced_health_service = get_health_service()
        
        if self._ml_health_manager is None:
            try:
                self._ml_health_manager = await get_ml_health_integration_manager()
            except Exception as e:
                self.logger.warning(f"ML health integration manager not available: {e}")
        
        if self._external_api_monitor is None:
            try:
                self._external_api_monitor = ExternalAPIHealthMonitor()
            except Exception as e:
                self.logger.warning(f"External API health monitor not available: {e}")
        
        if self._redis_health_monitor is None:
            try:
                self._redis_health_monitor = RedisHealthMonitor()
            except Exception as e:
                self.logger.warning(f"Redis health monitor not available: {e}")
        
        if self._database_health_monitor is None:
            try:
                self._database_health_monitor = await get_database_health_monitor()
            except Exception as e:
                self.logger.warning(f"Database health monitor not available: {e}")
        
    async def _execute_parallel_health_checks(
        self,
        execution_groups: List[List[str]],
        timeout_seconds: float
    ) -> Dict[str, HealthCheckResult]:
        """Execute health checks in parallel within dependency groups"""
        all_results = {}
        group_timeout = timeout_seconds / len(execution_groups) if execution_groups else timeout_seconds
        
        for group_index, component_group in enumerate(execution_groups):
            self.logger.debug(f"Executing health check group {group_index + 1}: {component_group}")
            
            # Execute all components in this group in parallel
            tasks = []
            for component_name in component_group:
                task = self._execute_single_component_health_check(
                    component_name, group_timeout
                )
                tasks.append(task)
            
            try:
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(group_results):
                    component_name = component_group[i]
                    if isinstance(result, Exception):
                        # Convert exception to failed health result
                        all_results[component_name] = HealthCheckResult(
                            component=component_name,
                            status=SystemHealthStatus.FAILED,
                            response_time_ms=0.0,
                            timestamp=datetime.now(timezone.utc),
                            error=str(result),
                            message=f"Health check execution failed: {result}"
                        )
                    else:
                        all_results[component_name] = result
                        
            except Exception as e:
                # Group execution failed
                self.logger.error(f"Health check group {group_index + 1} execution failed: {e}")
                
                # Create failed results for all components in group
                for component_name in component_group:
                    all_results[component_name] = HealthCheckResult(
                        component=component_name,
                        status=SystemHealthStatus.FAILED,
                        response_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        error=str(e),
                        message=f"Group execution failed: {e}"
                    )
        
        return all_results
    
    async def _execute_sequential_health_checks(
        self,
        execution_groups: List[List[str]],
        timeout_seconds: float
    ) -> Dict[str, HealthCheckResult]:
        """Execute health checks sequentially"""
        all_results = {}
        
        total_components = sum(len(group) for group in execution_groups)
        component_timeout = timeout_seconds / total_components if total_components > 0 else timeout_seconds
        
        for group_index, component_group in enumerate(execution_groups):
            for component_name in component_group:
                try:
                    result = await self._execute_single_component_health_check(
                        component_name, component_timeout
                    )
                    all_results[component_name] = result
                    
                except Exception as e:
                    all_results[component_name] = HealthCheckResult(
                        component=component_name,
                        status=SystemHealthStatus.FAILED,
                        response_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        error=str(e),
                        message=f"Health check failed: {e}"
                    )
        
        return all_results
    
    async def _execute_single_component_health_check(
        self,
        component_name: str,
        timeout_seconds: float
    ) -> HealthCheckResult:
        """Execute health check for a single component"""
        start_time = time.time()
        component_config = self.components.get(component_name)
        
        if not component_config:
            return HealthCheckResult(
                component=component_name,
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error="Component configuration not found",
                message=f"No configuration found for component: {component_name}"
            )
        
        # Get circuit breaker for component
        circuit_breaker = None
        if component_config.circuit_breaker_enabled:
            circuit_breaker = self.circuit_breaker_coordinator.get_or_create_breaker(component_name)
        
        try:
            # Execute health check with timeout and circuit breaker protection
            if circuit_breaker:
                result = await circuit_breaker.call(
                    self._call_component_health_check,
                    component_name,
                    component_config,
                    timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    self._call_component_health_check(component_name, component_config, timeout_seconds),
                    timeout=timeout_seconds
                )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            result.response_time_ms = response_time_ms
            
            return result
            
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=component_name,
                status=SystemHealthStatus.FAILED,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                error="Health check timeout",
                message=f"Health check timed out after {timeout_seconds}s"
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=component_name,
                status=SystemHealthStatus.FAILED,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"Health check failed: {e}"
            )
    
    async def _call_component_health_check(
        self,
        component_name: str,
        component_config: HealthComponent,
        timeout_seconds: float
    ) -> HealthCheckResult:
        """Call the appropriate health check method for a component"""
        
        if component_name == "database_primary":
            return await self._check_database_health()
        elif component_name == "redis_cache":
            return await self._check_redis_health()
        elif component_name in ["ml_models", "ml_orchestrator"]:
            return await self._check_ml_health(component_name)
        elif component_name == "external_apis":
            return await self._check_external_apis_health()
        elif component_name == "system_resources":
            return await self._check_system_resources_health()
        elif component_name == "application_health":
            return await self._check_application_health()
        else:
            return HealthCheckResult(
                component=component_name,
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error="Unknown component type",
                message=f"No health check implementation for component: {component_name}"
            )
    
    async def _check_database_health(self) -> HealthCheckResult:
        """Check database health using comprehensive database monitor"""
        try:
            if self._database_health_monitor:
                metrics = await self._database_health_monitor.collect_comprehensive_metrics()
                
                # Determine status based on health score
                if metrics.health_score >= 90:
                    status = SystemHealthStatus.HEALTHY
                elif metrics.health_score >= 70:
                    status = SystemHealthStatus.DEGRADED
                elif metrics.health_score >= 30:
                    status = SystemHealthStatus.CRITICAL
                else:
                    status = SystemHealthStatus.FAILED
                
                return HealthCheckResult(
                    component="database_primary",
                    status=status,
                    response_time_ms=0.0,  # Will be set by caller
                    timestamp=datetime.now(timezone.utc),
                    message=f"Database health score: {metrics.health_score:.1f}%",
                    health_score=metrics.health_score,
                    metadata={
                        "connection_pool": metrics.connection_pool,
                        "query_performance": metrics.query_performance,
                        "storage": metrics.storage,
                        "cache_hit_ratio": metrics.cache.get("overall_cache_hit_ratio_percent", 100)
                    },
                    sub_checks={
                        "connection_pool": metrics.connection_pool,
                        "query_performance": metrics.query_performance,
                        "replication": metrics.replication,
                        "storage": metrics.storage,
                        "locks": metrics.locks,
                        "cache": metrics.cache,
                        "transactions": metrics.transactions
                    }
                )
            else:
                # Fallback to basic database check via enhanced health service
                if self._enhanced_health_service:
                    result = await self._enhanced_health_service.run_specific_check("database")
                    return self._convert_health_result(result, "database_primary")
                
                return HealthCheckResult(
                    component="database_primary",
                    status=SystemHealthStatus.FAILED,
                    response_time_ms=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error="Database health monitor not available",
                    message="Database health monitoring system not initialized"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="database_primary",
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"Database health check failed: {e}"
            )
    
    async def _check_redis_health(self) -> HealthCheckResult:
        """Check Redis health using comprehensive Redis monitor"""
        try:
            if self._redis_health_monitor:
                result = await self._redis_health_monitor.collect_all_metrics()
                
                # Convert Redis health status to system health status
                redis_status = result.get("status", "failed")
                if redis_status == "healthy":
                    status = SystemHealthStatus.HEALTHY
                    health_score = 100.0
                elif redis_status == "warning":
                    status = SystemHealthStatus.DEGRADED  
                    health_score = 75.0
                elif redis_status == "critical":
                    status = SystemHealthStatus.CRITICAL
                    health_score = 40.0
                else:
                    status = SystemHealthStatus.FAILED
                    health_score = 0.0
                
                return HealthCheckResult(
                    component="redis_cache",
                    status=status,
                    response_time_ms=0.0,  # Will be set by caller
                    timestamp=datetime.now(timezone.utc),
                    message=f"Redis status: {redis_status}",
                    health_score=health_score,
                    metadata={
                        "memory_usage_pct": result.get("memory", {}).get("memory_usage_percentage", 0),
                        "hit_rate": result.get("performance", {}).get("hit_rate", 0),
                        "ops_per_sec": result.get("performance", {}).get("ops_per_sec", 0),
                        "connected_clients": result.get("clients", {}).get("connected_clients", 0)
                    },
                    sub_checks={
                        "memory": result.get("memory", {}),
                        "performance": result.get("performance", {}),
                        "persistence": result.get("persistence", {}),
                        "replication": result.get("replication", {}),
                        "clients": result.get("clients", {}),
                        "keyspace": result.get("keyspace", {}),
                        "slowlog": result.get("slowlog", {})
                    }
                )
            else:
                # Fallback to basic Redis check
                try:
                    result = await get_redis_health_summary()
                    
                    # Convert to health check result
                    healthy = result.get("healthy", False)
                    if healthy:
                        status = SystemHealthStatus.HEALTHY
                        health_score = 90.0
                    else:
                        status = SystemHealthStatus.FAILED
                        health_score = 0.0
                    
                    return HealthCheckResult(
                        component="redis_cache",
                        status=status,
                        response_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        message="Redis basic health check",
                        health_score=health_score,
                        metadata=result
                    )
                except Exception as e:
                    return HealthCheckResult(
                        component="redis_cache",
                        status=SystemHealthStatus.FAILED,
                        response_time_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                        error=str(e),
                        message=f"Redis health check failed: {e}"
                    )
                
        except Exception as e:
            return HealthCheckResult(
                component="redis_cache",
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"Redis health check failed: {e}"
            )
    
    async def _check_ml_health(self, component_name: str) -> HealthCheckResult:
        """Check ML system health using ML health integration manager"""
        try:
            if self._ml_health_manager:
                dashboard = await self._ml_health_manager.get_system_health_dashboard()
                
                # Determine overall ML health
                system_health = dashboard.get("system_health", {})
                ml_healthy = system_health.get("healthy", False)
                health_score = system_health.get("health_score", 0.0) * 100
                
                if ml_healthy and health_score >= 90:
                    status = SystemHealthStatus.HEALTHY
                elif ml_healthy and health_score >= 70:
                    status = SystemHealthStatus.DEGRADED
                elif health_score >= 30:
                    status = SystemHealthStatus.CRITICAL
                else:
                    status = SystemHealthStatus.FAILED
                
                # Component-specific details
                if component_name == "ml_models":
                    message = "ML models health check"
                    models_count = system_health.get("models", {}).get("total_loaded", 0)
                    metadata = {
                        "total_models_loaded": models_count,
                        "total_memory_mb": system_health.get("models", {}).get("total_memory_mb", 0),
                        "avg_memory_per_model": system_health.get("models", {}).get("memory_per_model_avg", 0)
                    }
                else:  # ml_orchestrator
                    message = "ML orchestrator health check"
                    metadata = {
                        "performance_overview": dashboard.get("performance_overview", {}),
                        "drift_overview": dashboard.get("drift_overview", {})
                    }
                
                return HealthCheckResult(
                    component=component_name,
                    status=status,
                    response_time_ms=0.0,  # Will be set by caller
                    timestamp=datetime.now(timezone.utc),
                    message=message,
                    health_score=health_score,
                    metadata=metadata,
                    sub_checks={
                        "system_health": system_health,
                        "model_summaries": dashboard.get("model_summaries", []),
                        "alerts": dashboard.get("alerts", []),
                        "recommendations": dashboard.get("recommendations", [])
                    }
                )
            else:
                # Fallback to enhanced health service ML check
                if self._enhanced_health_service:
                    result = await self._enhanced_health_service.run_specific_check("ml_service")
                    return self._convert_health_result(result, component_name)
                
                return HealthCheckResult(
                    component=component_name,
                    status=SystemHealthStatus.FAILED,
                    response_time_ms=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error="ML health manager not available",
                    message="ML health monitoring system not initialized"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=component_name,
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"ML health check failed: {e}"
            )
    
    async def _check_external_apis_health(self) -> HealthCheckResult:
        """Check external APIs health using external API monitor"""
        try:
            if self._external_api_monitor:
                health_snapshots = await self._external_api_monitor.check_all_endpoints()
                
                # Analyze overall API health
                total_apis = len(health_snapshots)
                healthy_apis = len([
                    s for s in health_snapshots.values() 
                    if s.status.value == "healthy"
                ])
                degraded_apis = len([
                    s for s in health_snapshots.values()
                    if s.status.value == "degraded"
                ])
                
                # Calculate health score
                if total_apis == 0:
                    health_score = 100.0
                    status = SystemHealthStatus.HEALTHY
                else:
                    healthy_percentage = (healthy_apis + degraded_apis * 0.7) / total_apis
                    health_score = healthy_percentage * 100
                    
                    if healthy_percentage >= 0.9:
                        status = SystemHealthStatus.HEALTHY
                    elif healthy_percentage >= 0.7:
                        status = SystemHealthStatus.DEGRADED
                    elif healthy_percentage >= 0.3:
                        status = SystemHealthStatus.CRITICAL
                    else:
                        status = SystemHealthStatus.FAILED
                
                # Aggregate response times
                response_times = [
                    s.current_response_time_ms 
                    for s in health_snapshots.values()
                    if s.current_response_time_ms is not None
                ]
                avg_response_time = statistics.mean(response_times) if response_times else 0.0
                
                return HealthCheckResult(
                    component="external_apis",
                    status=status,
                    response_time_ms=0.0,  # Will be set by caller
                    timestamp=datetime.now(timezone.utc),
                    message=f"External APIs: {healthy_apis}/{total_apis} healthy",
                    health_score=health_score,
                    metadata={
                        "total_apis": total_apis,
                        "healthy_apis": healthy_apis,
                        "degraded_apis": degraded_apis,
                        "avg_response_time_ms": avg_response_time,
                        "overall_availability": healthy_percentage
                    },
                    sub_checks={
                        endpoint_name: {
                            "status": snapshot.status.value,
                            "response_time_ms": snapshot.current_response_time_ms,
                            "availability": snapshot.availability,
                            "sla_compliance": snapshot.sla_compliance.value
                        }
                        for endpoint_name, snapshot in health_snapshots.items()
                    }
                )
            else:
                return HealthCheckResult(
                    component="external_apis",
                    status=SystemHealthStatus.FAILED,
                    response_time_ms=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error="External API monitor not available",
                    message="External API health monitoring system not initialized"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="external_apis",
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"External APIs health check failed: {e}"
            )
    
    async def _check_system_resources_health(self) -> HealthCheckResult:
        """Check system resources health"""
        try:
            if self._enhanced_health_service:
                # Get system metrics from enhanced health service
                result = await self._enhanced_health_service.run_specific_check("system")
                return self._convert_health_result(result, "system_resources")
            
            # Fallback to basic system check
            import psutil
            
            # Check CPU, memory, disk
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health based on resource usage
            health_issues = []
            if cpu_percent > 90:
                health_issues.append("High CPU usage")
            if memory.percent > 90:
                health_issues.append("High memory usage")
            if disk.percent > 85:
                health_issues.append("High disk usage")
            
            # Calculate health score
            cpu_score = max(0, 100 - cpu_percent)
            memory_score = max(0, 100 - memory.percent)  
            disk_score = max(0, 100 - disk.percent)
            health_score = (cpu_score + memory_score + disk_score) / 3
            
            # Determine status
            if not health_issues:
                status = SystemHealthStatus.HEALTHY
            elif len(health_issues) == 1 and health_score > 70:
                status = SystemHealthStatus.DEGRADED
            elif health_score > 30:
                status = SystemHealthStatus.CRITICAL
            else:
                status = SystemHealthStatus.FAILED
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                response_time_ms=0.0,  # Will be set by caller
                timestamp=datetime.now(timezone.utc),
                message=f"System resources check - Issues: {', '.join(health_issues) if health_issues else 'None'}",
                health_score=health_score,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"System resources health check failed: {e}"
            )
    
    async def _check_application_health(self) -> HealthCheckResult:
        """Check general application health"""
        try:
            if self._enhanced_health_service:
                # Run aggregated health check from enhanced service
                result = await self._enhanced_health_service.run_health_check(parallel=True)
                
                # Convert aggregated result
                if result.overall_status.value == "healthy":
                    status = SystemHealthStatus.HEALTHY
                    health_score = 95.0
                elif result.overall_status.value == "warning":
                    status = SystemHealthStatus.DEGRADED
                    health_score = 75.0
                else:
                    status = SystemHealthStatus.CRITICAL
                    health_score = 40.0
                
                return HealthCheckResult(
                    component="application_health",
                    status=status,
                    response_time_ms=0.0,  # Will be set by caller
                    timestamp=datetime.now(timezone.utc),
                    message=f"Application health: {result.overall_status.value}",
                    health_score=health_score,
                    metadata={
                        "total_checks": len(result.checks),
                        "failed_checks": len(result.failed_checks) if result.failed_checks else 0,
                        "warning_checks": len(result.warning_checks) if result.warning_checks else 0
                    },
                    sub_checks={
                        name: {
                            "status": check_result.status.value,
                            "message": check_result.message,
                            "error": check_result.error
                        }
                        for name, check_result in result.checks.items()
                    }
                )
            else:
                return HealthCheckResult(
                    component="application_health",
                    status=SystemHealthStatus.FAILED,
                    response_time_ms=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error="Enhanced health service not available",
                    message="Application health monitoring system not initialized"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="application_health",
                status=SystemHealthStatus.FAILED,
                response_time_ms=0.0,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
                message=f"Application health check failed: {e}"
            )
    
    def _convert_health_result(self, result, component_name: str) -> HealthCheckResult:
        """Convert legacy health result to new format"""
        # Map legacy status to new status
        if result.status.value == "healthy":
            status = SystemHealthStatus.HEALTHY
            health_score = 95.0
        elif result.status.value == "warning":
            status = SystemHealthStatus.DEGRADED
            health_score = 75.0
        else:
            status = SystemHealthStatus.FAILED
            health_score = 0.0
        
        return HealthCheckResult(
            component=component_name,
            status=status,
            response_time_ms=getattr(result, 'response_time_ms', 0.0),
            timestamp=result.timestamp or datetime.now(timezone.utc),
            message=result.message or "",
            error=result.error,
            health_score=health_score,
            metadata=getattr(result, 'metadata', {})
        )
    
    def _generate_recommendations(self, snapshot: SystemHealthSnapshot) -> List[str]:
        """Generate actionable recommendations based on health snapshot"""
        recommendations = []
        
        # Overall system recommendations
        if snapshot.overall_health_score < 50:
            recommendations.append(
                "🚨 CRITICAL: System health is severely degraded. Immediate intervention required."
            )
        elif snapshot.overall_health_score < 70:
            recommendations.append(
                "⚠️ WARNING: System health is degraded. Monitor closely and address issues."
            )
        
        # Component-specific recommendations
        for component_name, result in snapshot.component_results.items():
            if result.status == SystemHealthStatus.FAILED:
                recommendations.append(
                    f"🔴 {component_name}: Component is failed - investigate immediately"
                )
            elif result.status == SystemHealthStatus.CRITICAL:
                recommendations.append(
                    f"🟠 {component_name}: Component is critical - requires attention"
                )
            elif result.response_time_ms > 5000:
                recommendations.append(
                    f"⏱️ {component_name}: Slow response time ({result.response_time_ms:.1f}ms) - optimize performance"
                )
        
        # Cascade failure recommendations
        if snapshot.cascade_failures:
            recommendations.append(
                f"🔗 Cascade failures detected ({len(snapshot.cascade_failures)}) - address root cause components"
            )
        
        # Performance recommendations
        aggregated = snapshot.aggregated_metrics
        if "availability" in aggregated:
            availability = aggregated["availability"]["overall_percentage"]
            if availability < 95:
                recommendations.append(
                    f"📊 System availability is {availability:.1f}% - improve component reliability"
                )
        
        # Circuit breaker recommendations
        open_breakers = [
            name for name, state in snapshot.circuit_breaker_states.items()
            if state.get("state") == "open"
        ]
        if open_breakers:
            recommendations.append(
                f"⚡ Circuit breakers open for: {', '.join(open_breakers)} - investigate underlying issues"
            )
        
        # Trend-based recommendations
        if "trend_analysis" in aggregated:
            trends = aggregated["trend_analysis"]
            if trends.get("response_time") == "degrading":
                recommendations.append("📈 Response times are degrading - investigate performance bottlenecks")
            if trends.get("availability") == "degrading":
                recommendations.append("📉 Availability trend is negative - review system stability")
        
        # Default recommendation
        if not recommendations and snapshot.overall_health_score >= 90:
            recommendations.append("✅ System health is excellent - continue current monitoring practices")
        
        return recommendations
    
    async def get_health_dashboard(self) -> Dict[str, Any]:
        """Get unified health dashboard data"""
        try:
            # Execute comprehensive health check
            snapshot = await self.execute_comprehensive_health_check()
            
            # Build dashboard data structure
            dashboard = {
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_status": snapshot.overall_status.value,
                "overall_health_score": snapshot.overall_health_score,
                "execution_time_ms": snapshot.execution_time_ms,
                
                # Summary statistics
                "summary": {
                    "total_components": len(snapshot.component_results),
                    "healthy_components": len([
                        r for r in snapshot.component_results.values()
                        if r.status == SystemHealthStatus.HEALTHY
                    ]),
                    "degraded_components": len([
                        r for r in snapshot.component_results.values()
                        if r.status == SystemHealthStatus.DEGRADED
                    ]),
                    "critical_components": len([
                        r for r in snapshot.component_results.values()
                        if r.status == SystemHealthStatus.CRITICAL
                    ]),
                    "failed_components": len([
                        r for r in snapshot.component_results.values()
                        if r.status == SystemHealthStatus.FAILED
                    ])
                },
                
                # Component details
                "components": {
                    name: {
                        "status": result.status.value,
                        "health_score": result.health_score,
                        "response_time_ms": result.response_time_ms,
                        "message": result.message,
                        "error": result.error,
                        "timestamp": result.timestamp.isoformat(),
                        "metadata": result.metadata,
                        "sub_checks": result.sub_checks
                    }
                    for name, result in snapshot.component_results.items()
                },
                
                # Performance metrics
                "performance": snapshot.aggregated_metrics,
                
                # Dependency analysis
                "dependencies": {
                    "cascade_failures": snapshot.cascade_failures,
                    "dependency_status": snapshot.dependency_chain_status
                },
                
                # Circuit breaker status
                "circuit_breakers": snapshot.circuit_breaker_states,
                
                # Active alerts
                "alerts": snapshot.active_alerts,
                
                # Recommendations
                "recommendations": snapshot.recommendations,
                
                # Historical data
                "execution_history": list(self._execution_history)
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to generate health dashboard: {e}", exc_info=True)
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": "failed",
                "overall_health_score": 0.0,
                "error": str(e),
                "execution_time_ms": 0.0
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics for monitoring integration"""
        return self.alerting_manager.get_prometheus_metrics()
    
    async def reset_circuit_breakers(self) -> Dict[str, str]:
        """Reset all circuit breakers (emergency recovery)"""
        try:
            self.circuit_breaker_coordinator.reset_all_breakers()
            return {
                "status": "success",
                "message": "All circuit breakers have been reset",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to reset circuit breakers: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_component_dependencies(self) -> Dict[str, Any]:
        """Get component dependency information"""
        return {
            "components": {
                name: {
                    "name": config.name,
                    "category": config.category,
                    "dependency_level": config.dependency_level.value,
                    "dependencies": config.dependencies,
                    "weight": config.weight,
                    "timeout_seconds": config.timeout_seconds,
                    "circuit_breaker_enabled": config.circuit_breaker_enabled
                }
                for name, config in self.components.items()
            },
            "execution_order": self.dependency_manager.get_execution_order(),
            "dependency_graph": dict(self.dependency_manager.dependency_graph)
        }

# Global health orchestrator instance
_health_orchestrator: Optional[HealthOrchestrator] = None

async def get_health_orchestrator() -> HealthOrchestrator:
    """Get or create global health orchestrator instance"""
    global _health_orchestrator
    if _health_orchestrator is None:
        _health_orchestrator = HealthOrchestrator()
    return _health_orchestrator

def reset_health_orchestrator():
    """Reset global health orchestrator instance (useful for testing)"""
    global _health_orchestrator
    _health_orchestrator = None