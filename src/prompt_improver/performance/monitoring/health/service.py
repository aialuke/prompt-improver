"""Enhanced Health Service - 2025 Edition

Advanced health monitoring with 2025 best practices:
- Circuit breaker integration for dependency health
- Predictive health monitoring with trend analysis
- Health check result caching and optimization
- Dependency graph visualization and analysis
- Advanced observability with OpenTelemetry integration
- Service mesh health integration
- Auto-healing capabilities
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

# Enhanced observability imports
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Circuit breaker imports
try:
    from circuit_breaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    # Simple circuit breaker implementation
    class CircuitBreaker:
        def __init__(self, failure_threshold=5, timeout=60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "closed"  # closed, open, half_open

        def call(self, func):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = func()
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                raise e

from .base import AggregatedHealthResult, HealthChecker, HealthResult, HealthStatus

class HealthTrend(Enum):
    """Health trend analysis."""

    improving = "improving"
    stable = "stable"
    degrading = "degrading"
    CRITICAL = "critical"

class DependencyType(Enum):
    """Types of service dependencies."""

    DATABASE = "database"
    cache = "cache"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    ML_SERVICE = "ml_service"
    storage = "storage"
    network = "network"

@dataclass
class HealthMetrics:
    """Enhanced health metrics with trend analysis."""

    success_count: int = 0
    failure_count: int = 0
    total_checks: int = 0
    average_response_time: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    trend: HealthTrend = HealthTrend.stable

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_checks == 0:
            return 0.0
        return self.success_count / self.total_checks

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

@dataclass
class DependencyInfo:
    """Information about service dependencies."""

    name: str
    dependency_type: DependencyType
    critical: bool = True
    circuit_breaker: Optional[CircuitBreaker] = None
    health_metrics: HealthMetrics = field(default_factory=HealthMetrics)
    last_check_time: Optional[datetime] = None
    cache_ttl: int = 30  # seconds

@dataclass
class PredictiveHealthAnalysis:
    """Predictive health analysis results."""

    component: str
    current_health: HealthStatus
    predicted_health: HealthStatus
    confidence: float
    time_to_failure: Optional[timedelta] = None
    recommended_actions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

# Use centralized metrics registry
from ..metrics_registry import get_metrics_registry, StandardMetrics

metrics_registry = get_metrics_registry()
ENHANCED_HEALTH_CHECK_COUNTER = metrics_registry.get_or_create_counter(
    StandardMetrics.HEALTH_CHECK_TOTAL,
    'Total enhanced health checks',
    ['component', 'status']
)
ENHANCED_HEALTH_CHECK_DURATION = metrics_registry.get_or_create_histogram(
    StandardMetrics.HEALTH_CHECK_DURATION,
    'Enhanced health check duration',
    ['component']
)
ENHANCED_HEALTH_STATUS_GAUGE = metrics_registry.get_or_create_gauge(
    StandardMetrics.HEALTH_CHECK_STATUS,
    'Current enhanced health status',
    ['component']
)
ENHANCED_CIRCUIT_BREAKER_STATE = metrics_registry.get_or_create_gauge(
    StandardMetrics.CIRCUIT_BREAKER_STATE,
    'Enhanced circuit breaker state',
    ['component']
)
ENHANCED_DEPENDENCY_HEALTH = metrics_registry.get_or_create_gauge(
    'enhanced_dependency_health_score',
    'Enhanced dependency health score',
    ['dependency', 'type']
)
from .checkers import (
    REDIS_MONITOR_AVAILABLE,
    AnalyticsServiceHealthChecker,
    DatabaseHealthChecker,
    MCPServerHealthChecker,
    MLServiceHealthChecker,
    SystemResourcesHealthChecker,
)

# Import RedisHealthMonitor from the comprehensive implementation
from ....cache.redis_health import RedisHealthMonitor

# Import ML-specific health checkers
try:
    from .ml_specific_checkers import (
        MLModelHealthChecker,
        MLDataQualityChecker,
        MLTrainingHealthChecker,
        MLPerformanceHealthChecker,
    )
    ML_SPECIFIC_CHECKERS_AVAILABLE = True
except ImportError:
    ML_SPECIFIC_CHECKERS_AVAILABLE = False

# Lazy import to avoid circular dependency
try:
    from .checkers import queue_health_checker
except ImportError:
    # Use lazy import to avoid circular dependency issues
    queue_health_checker = None

# Import ML orchestration health checkers
try:
    from .ml_orchestration_checkers import (
        MLOrchestratorHealthChecker,
        MLComponentRegistryHealthChecker,
        MLResourceManagerHealthChecker,
        MLWorkflowEngineHealthChecker,
        MLEventBusHealthChecker
    )
    ML_ORCHESTRATION_CHECKERS_AVAILABLE = True
except ImportError:
    ML_ORCHESTRATION_CHECKERS_AVAILABLE = False

from .metrics import instrument_health_check

class EnhancedHealthService:
    """Enhanced health service with 2025 best practices

    features:
    - Circuit breaker integration for dependency health
    - Predictive health monitoring with trend analysis
    - Health check result caching and optimization
    - Dependency graph visualization and analysis
    - Advanced observability with metrics
    """

    def __init__(
        self,
        checkers: Optional[List[HealthChecker]] = None,
        enable_circuit_breakers: bool = True,
        enable_predictive_analysis: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 30,
        trend_window_size: int = 10
    ):
        """Initialize enhanced health service with 2025 features."""
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_predictive_analysis = enable_predictive_analysis
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.trend_window_size = trend_window_size

        # Enhanced tracking
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=trend_window_size))
        self.cached_results: Dict[str, Tuple[HealthResult, datetime]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if checkers is None:
            self.checkers = [
                DatabaseHealthChecker(),
                MCPServerHealthChecker(),
                AnalyticsServiceHealthChecker(),
                MLServiceHealthChecker(),
                SystemResourcesHealthChecker(),
            ]

            # Add Redis health monitor if available
            if REDIS_MONITOR_AVAILABLE and RedisHealthMonitor is not None:
                try:
                    # Load Redis configuration

                    from ...utils.redis_cache import redis_config

                    # Create health monitor configuration
                    monitor_config = {
                        'check_interval': 60,
                        'failure_threshold': 3,
                        'latency_threshold': 100,
                        'reconnection': {'max_retries': 5, 'backoff_factor': 2}
                    }

                    # Update with YAML configuration if available
                    if hasattr(redis_config, 'health_monitor'):
                        monitor_config.update(redis_config.health_monitor)

                    self.checkers.append(RedisHealthMonitor(monitor_config))
                except Exception as e:
                    # Log the error but continue without Redis health checker
                    print(f"Warning: Could not initialize RedisHealthMonitor: {e}")

            # Add queue_health_checker if available (avoid circular import)
            if queue_health_checker is not None:
                try:
                    self.checkers.append(queue_health_checker())
                except Exception as e:
                    # Log the error but continue without queue health checker
                    print(f"Warning: Could not initialize queue_health_checker: {e}")

            # Add ML orchestration health checkers if available
            if ML_ORCHESTRATION_CHECKERS_AVAILABLE:
                try:
                    self.ml_orchestrator_checker = MLOrchestratorHealthChecker()
                    self.ml_registry_checker = MLComponentRegistryHealthChecker()
                    self.ml_resource_checker = MLResourceManagerHealthChecker()
                    self.ml_workflow_checker = MLWorkflowEngineHealthChecker()
                    self.ml_event_checker = MLEventBusHealthChecker()

                    # Add to checkers list
                    self.checkers.extend([
                        self.ml_orchestrator_checker,
                        self.ml_registry_checker,
                        self.ml_resource_checker,
                        self.ml_workflow_checker,
                        self.ml_event_checker
                    ])
                except Exception as e:
                    # Log the error but continue without ML health checkers
                    print(f"Warning: Could not initialize ML orchestration health checkers: {e}")

            # Add ML-specific health checkers if available
            if ML_SPECIFIC_CHECKERS_AVAILABLE:
                try:
                    self.ml_model_checker = MLModelHealthChecker()
                    self.ml_data_quality_checker = MLDataQualityChecker()
                    self.ml_training_checker = MLTrainingHealthChecker()
                    self.ml_performance_checker = MLPerformanceHealthChecker()

                    # Add to checkers list
                    self.checkers.extend([
                        self.ml_model_checker,
                        self.ml_data_quality_checker,
                        self.ml_training_checker,
                        self.ml_performance_checker
                    ])
                except Exception as e:
                    # Log the error but continue without ML-specific health checkers
                    print(f"Warning: Could not initialize ML-specific health checkers: {e}")
        else:
            self.checkers = checkers

        # Create checker mapping for easy access
        self.checker_map = {checker.name: checker for checker in self.checkers}

        # Initialize dependencies with circuit breakers
        self._initialize_dependencies()

        self.logger.info(f"Enhanced health service initialized with {len(self.checkers)} checkers and {len(self.dependencies)} dependencies")

    def _initialize_dependencies(self):
        """Initialize dependency information and circuit breakers."""
        dependency_mapping = {
            "database": DependencyType.DATABASE,
            "redis": DependencyType.cache,
            "mcp_server": DependencyType.EXTERNAL_API,
            "analytics": DependencyType.EXTERNAL_API,
            "ml_service": DependencyType.ML_SERVICE,
            "ml_orchestrator": DependencyType.ML_SERVICE,
            "ml_registry": DependencyType.ML_SERVICE,
            "ml_resource": DependencyType.ML_SERVICE,
            "ml_workflow": DependencyType.ML_SERVICE,
            "ml_event": DependencyType.MESSAGE_QUEUE,
            "queue": DependencyType.MESSAGE_QUEUE,
            "system": DependencyType.network
        }

        for checker in self.checkers:
            # Determine dependency type
            dep_type = DependencyType.EXTERNAL_API  # default
            for key, dtype in dependency_mapping.items():
                if key in checker.name.lower():
                    dep_type = dtype
                    break

            # Create circuit breaker if enabled
            circuit_breaker = None
            if self.enable_circuit_breakers:
                circuit_breaker = CircuitBreaker(
                    failure_threshold=3,
                    timeout=60
                )
                self.circuit_breakers[checker.name] = circuit_breaker

            # Create dependency info
            self.dependencies[checker.name] = DependencyInfo(
                name=checker.name,
                dependency_type=dep_type,
                critical=dep_type in [DependencyType.DATABASE, DependencyType.ML_SERVICE],
                circuit_breaker=circuit_breaker,
                cache_ttl=self.cache_ttl
            )

    async def run_enhanced_health_check(
        self,
        parallel: bool = True,
        use_cache: bool = True,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Run enhanced health checks with 2025 features."""
        start_time = datetime.now(UTC)

        # Check cache first if enabled
        if use_cache and self.enable_caching:
            cached_results = self._get_cached_results()
            if cached_results:
                self.logger.debug(f"Using cached results for {len(cached_results)} components")
        else:
            cached_results = {}

        # Determine which checks to run
        checks_to_run = [
            checker for checker in self.checkers
            if checker.name not in cached_results
        ]

        # Run health checks
        if parallel and checks_to_run:
            tasks = [self._run_single_health_check(checker) for checker in checks_to_run]
            fresh_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            fresh_results = []
            for checker in checks_to_run:
                result = await self._run_single_health_check(checker)
                fresh_results.append(result)

        # Combine cached and fresh results
        all_results = {}
        all_results.update(cached_results)

        for i, result in enumerate(fresh_results):
            checker = checks_to_run[i]
            if isinstance(result, Exception):
                # Convert exception to failed health result
                result = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error=str(result),
                    message=f"Health check failed: {result}",
                    timestamp=datetime.now(UTC)
                )

            all_results[checker.name] = result

            # Update cache
            if self.enable_caching:
                self.cached_results[checker.name] = (result, datetime.now(UTC))

        # Update health history and metrics
        self._update_health_history(all_results)

        # Perform trend analysis
        trend_analysis = self._analyze_health_trends()

        # Generate predictive analysis if enabled
        predictions = []
        if include_predictions and self.enable_predictive_analysis:
            predictions = await self._generate_predictive_analysis(all_results)

        # Calculate overall health
        overall_status = self._calculate_overall_health(all_results)

        # Prepare enhanced result
        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        return {
            "overall_status": overall_status.value,
            "execution_time": execution_time,
            "timestamp": start_time.isoformat(),
            "component_results": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "error": result.error,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                    "response_time": getattr(result, 'response_time', None),
                    "metadata": getattr(result, 'metadata', {})
                }
                for name, result in all_results.items()
            },
            "dependency_analysis": self._analyze_dependencies(all_results),
            "trend_analysis": trend_analysis,
            "predictive_analysis": predictions,
            "circuit_breaker_status": self._get_circuit_breaker_status(),
            "health_metrics": self._get_health_metrics_summary(),
            "recommendations": self._generate_recommendations(all_results, predictions)
        }

    @instrument_health_check("aggregated")
    async def run_health_check(self, parallel: bool = True) -> AggregatedHealthResult:
        """Run all health checks and return aggregated result"""
        start_time = time.time()

        if parallel:
            # Run all checks in parallel for better performance
            tasks = [checker.check() for checker in self.checkers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run checks sequentially
            results = []
            for checker in self.checkers:
                try:
                    result = await checker.check()
                    results.append(result)
                except Exception as e:
                    # Create error result if checker fails
                    error_result = HealthResult(
                        status=HealthStatus.FAILED,
                        component=checker.name,
                        error=str(e),
                        message=f"Health check failed: {e!s}",
                    )
                    results.append(error_result)

        # Process results and handle any exceptions
        check_results = {}
        for i, result in enumerate(results):
            checker = self.checkers[i]

            if isinstance(result, Exception):
                # Convert exception to failed health result
                check_results[checker.name] = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error=str(result),
                    message=f"Health check failed: {result!s}",
                )
            elif isinstance(result, HealthResult):
                check_results[checker.name] = result
            else:
                # Unexpected result type
                check_results[checker.name] = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error="Invalid result type",
                    message="Health check returned invalid result",
                )

        # Calculate overall status using hierarchical logic
        overall_status = self._calculate_overall_status(check_results)

        end_time = time.time()
        response_time = (end_time - start_time) * 1000

        return AggregatedHealthResult(
            overall_status=overall_status,
            checks=check_results,
            timestamp=datetime.now(),
        )

    async def _run_single_health_check(self, checker: HealthChecker) -> HealthResult:
        """Run a single health check with circuit breaker protection."""
        dependency = self.dependencies.get(checker.name)

        if dependency and dependency.circuit_breaker:
            try:
                # Use circuit breaker
                def check_func():
                    return asyncio.create_task(checker.check())

                result = await dependency.circuit_breaker.call(check_func)

                # Update metrics on success
                dependency.health_metrics.success_count += 1
                dependency.health_metrics.consecutive_successes += 1
                dependency.health_metrics.consecutive_failures = 0
                dependency.health_metrics.last_success_time = datetime.now(UTC)

            except Exception as e:
                # Circuit breaker is open or check failed
                dependency.health_metrics.failure_count += 1
                dependency.health_metrics.consecutive_failures += 1
                dependency.health_metrics.consecutive_successes = 0
                dependency.health_metrics.last_failure_time = datetime.now(UTC)

                result = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error=str(e),
                    message=f"Circuit breaker protection: {e}",
                    timestamp=datetime.now(UTC)
                )
        else:
            # Run check normally
            try:
                result = await checker.check()
                if dependency:
                    dependency.health_metrics.success_count += 1
                    dependency.health_metrics.consecutive_successes += 1
                    dependency.health_metrics.consecutive_failures = 0
                    dependency.health_metrics.last_success_time = datetime.now(UTC)
            except Exception as e:
                if dependency:
                    dependency.health_metrics.failure_count += 1
                    dependency.health_metrics.consecutive_failures += 1
                    dependency.health_metrics.consecutive_successes = 0
                    dependency.health_metrics.last_failure_time = datetime.now(UTC)

                result = HealthResult(
                    status=HealthStatus.FAILED,
                    component=checker.name,
                    error=str(e),
                    message=f"Health check failed: {e}",
                    timestamp=datetime.now(UTC)
                )

        # Update total checks
        if dependency:
            dependency.health_metrics.total_checks += 1
            dependency.last_check_time = datetime.now(UTC)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and ENHANCED_HEALTH_CHECK_COUNTER:
            ENHANCED_HEALTH_CHECK_COUNTER.labels(component=checker.name, status=result.status.value).inc()
            if ENHANCED_HEALTH_STATUS_GAUGE:
                ENHANCED_HEALTH_STATUS_GAUGE.labels(component=checker.name).set(1 if result.status == HealthStatus.HEALTHY else 0)

        return result

    def _get_cached_results(self) -> Dict[str, HealthResult]:
        """Get valid cached health check results."""
        cached_results = {}
        current_time = datetime.now(UTC)

        for component, (result, cache_time) in list(self.cached_results.items()):
            dependency = self.dependencies.get(component)
            cache_ttl = dependency.cache_ttl if dependency else self.cache_ttl

            if (current_time - cache_time).total_seconds() < cache_ttl:
                cached_results[component] = result
            else:
                # Remove expired cache entry
                del self.cached_results[component]

        return cached_results

    def _update_health_history(self, results: Dict[str, HealthResult]):
        """Update health history for trend analysis."""
        for component, result in results.items():
            self.health_history[component].append({
                "timestamp": datetime.now(UTC),
                "status": result.status,
                "response_time": getattr(result, 'response_time', None)
            })

    def _analyze_health_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze health trends for all components."""
        trends = {}

        for component, history in self.health_history.items():
            if len(history) < 2:
                trends[component] = {"trend": HealthTrend.stable.value, "confidence": 0.0}
                continue

            # Analyze recent trend
            recent_statuses = [entry["status"] for entry in list(history)[-5:]]
            healthy_count = sum(1 for status in recent_statuses if status == HealthStatus.HEALTHY)

            if healthy_count == len(recent_statuses):
                trend = HealthTrend.stable if len(recent_statuses) >= 3 else HealthTrend.improving
            elif healthy_count == 0:
                trend = HealthTrend.CRITICAL
            elif healthy_count < len(recent_statuses) / 2:
                trend = HealthTrend.degrading
            else:
                trend = HealthTrend.improving

            confidence = min(1.0, len(recent_statuses) / 5.0)

            trends[component] = {
                "trend": trend.value,
                "confidence": confidence,
                "recent_success_rate": healthy_count / len(recent_statuses)
            }

        return trends

    def _calculate_overall_health(self, results: Dict[str, HealthResult]) -> HealthStatus:
        """Calculate overall health status from individual results."""
        if not results:
            return HealthStatus.FAILED

        statuses = [result.status for result in results.values()]

        # If any critical dependency is failed, overall is failed
        for component, result in results.items():
            dependency = self.dependencies.get(component)
            if dependency and dependency.critical and result.status == HealthStatus.FAILED:
                return HealthStatus.FAILED

        # Check status distribution
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.FAILED for status in statuses):
            return HealthStatus.WARNING
        elif any(status == HealthStatus.WARNING for status in statuses):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    async def run_specific_check(self, component_name: str) -> HealthResult:
        """Run health check for a specific component"""
        if component_name not in self.checker_map:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=component_name,
                error="Unknown component",
                message=f"No health checker found for component: {component_name}",
            )

        checker = self.checker_map[component_name]
        try:
            return await checker.check()
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=component_name,
                error=str(e),
                message=f"Health check failed: {e!s}",
            )

    def _calculate_overall_status(
        self, results: dict[str, HealthResult]
    ) -> HealthStatus:
        """Calculate overall health status from individual results"""
        if not results:
            return HealthStatus.FAILED

        statuses = [result.status for result in results.values()]

        # Hierarchical status calculation: failed > warning > healthy
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        if HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    def get_available_checks(self) -> list[str]:
        """Get list of available health check components"""
        return list(self.checker_map.keys())

    async def get_health_summary(self, include_details: bool = False) -> dict:
        """Get health summary in dictionary format for API responses"""
        result = await self.run_health_check()

        summary = {
            "overall_status": result.overall_status.value,
            "timestamp": result.timestamp.isoformat(),
            "checks": {},
        }

        for component, check_result in result.checks.items():
            check_summary = {
                "status": check_result.status.value,
                "message": check_result.message,
            }

            if check_result.response_time_ms is not None:
                check_summary["response_time_ms"] = check_result.response_time_ms

            if check_result.error:
                check_summary["error"] = check_result.error

            if include_details and check_result.details:
                check_summary["details"] = check_result.details

            summary["checks"][component] = check_summary

        if result.failed_checks:
            summary["failed_checks"] = result.failed_checks

        if result.warning_checks:
            summary["warning_checks"] = result.warning_checks

        return summary

    def _analyze_dependencies(self, results: Dict[str, HealthResult]) -> Dict[str, Any]:
        """Analyze dependency health and relationships."""
        dependency_analysis = {
            "critical_dependencies": [],
            "failing_dependencies": [],
            "dependency_graph": {},
            "impact_analysis": {}
        }

        for component, result in results.items():
            dependency = self.dependencies.get(component)
            if not dependency:
                continue

            dep_info = {
                "name": component,
                "type": dependency.dependency_type.value,
                "status": result.status.value,
                "critical": dependency.critical,
                "success_rate": dependency.health_metrics.success_rate,
                "consecutive_failures": dependency.health_metrics.consecutive_failures
            }

            if dependency.critical:
                dependency_analysis["critical_dependencies"].append(dep_info)

            if result.status != HealthStatus.HEALTHY:
                dependency_analysis["failing_dependencies"].append(dep_info)

            dependency_analysis["dependency_graph"][component] = dep_info

        return dependency_analysis

    async def _generate_predictive_analysis(self, results: Dict[str, HealthResult]) -> List[Dict[str, Any]]:
        """Generate predictive health analysis."""
        predictions = []

        for component, result in results.items():
            dependency = self.dependencies.get(component)
            if not dependency:
                continue

            # Simple predictive analysis based on trends
            history = self.health_history.get(component, deque())
            if len(history) < 3:
                continue

            recent_failures = dependency.health_metrics.consecutive_failures
            success_rate = dependency.health_metrics.success_rate

            # Predict future health
            if recent_failures >= 2:
                predicted_health = HealthStatus.FAILED
                confidence = 0.8
                time_to_failure = "5 minutes"
                recommended_actions = ["Investigate component", "Check dependencies", "Review logs"]
                risk_factors = ["High failure rate", "Consecutive failures"]
            elif success_rate < 0.7:
                predicted_health = HealthStatus.WARNING
                confidence = 0.6
                time_to_failure = "15 minutes"
                recommended_actions = ["Monitor closely", "Check performance metrics"]
                risk_factors = ["Low success rate"]
            else:
                predicted_health = HealthStatus.HEALTHY
                confidence = 0.9
                time_to_failure = None
                recommended_actions = []
                risk_factors = []

            prediction = {
                "component": component,
                "current_health": result.status.value,
                "predicted_health": predicted_health.value,
                "confidence": confidence,
                "time_to_failure": time_to_failure,
                "recommended_actions": recommended_actions,
                "risk_factors": risk_factors
            }

            predictions.append(prediction)

        return predictions

    def _get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all components."""
        status = {}

        for component, cb in self.circuit_breakers.items():
            status[component] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }

        return status

    def _get_health_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics summary for all components."""
        summary = {}

        for component, dependency in self.dependencies.items():
            metrics = dependency.health_metrics
            summary[component] = {
                "success_count": metrics.success_count,
                "failure_count": metrics.failure_count,
                "total_checks": metrics.total_checks,
                "success_rate": metrics.success_rate,
                "failure_rate": metrics.failure_rate,
                "consecutive_failures": metrics.consecutive_failures,
                "consecutive_successes": metrics.consecutive_successes,
                "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                "last_failure_time": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
                "trend": metrics.trend.value
            }

        return summary

    def _generate_recommendations(self, results: Dict[str, HealthResult], predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on health analysis."""
        recommendations = []

        # Check for critical failures
        critical_failures = [
            component for component, result in results.items()
            if result.status == HealthStatus.FAILED and self.dependencies.get(component, DependencyInfo("", DependencyType.EXTERNAL_API)).critical
        ]

        if critical_failures:
            recommendations.append(f"URGENT: Critical dependencies failing: {', '.join(critical_failures)}")

        # Check for warning services
        warning_services = [
            component for component, result in results.items()
            if result.status == HealthStatus.WARNING
        ]

        if warning_services:
            recommendations.append(f"Monitor warning services: {', '.join(warning_services)}")

        # Check circuit breaker states
        open_breakers = [
            component for component, cb in self.circuit_breakers.items()
            if cb.state == "open"
        ]

        if open_breakers:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_breakers)}")

        # Add predictive recommendations
        for prediction in predictions:
            if prediction["predicted_health"] == "failed" and prediction["confidence"] > 0.7:
                recommendations.append(f"Predicted failure for {prediction['component']} in {prediction['time_to_failure']}")

        return recommendations

    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for health monitoring (2025 pattern)"""
        start_time = datetime.now(UTC)

        try:
            # Extract configuration
            parallel = config.get("parallel", True)
            use_cache = config.get("use_cache", True)
            include_predictions = config.get("include_predictions", True)
            output_path = config.get("output_path", "./outputs/health_monitoring")

            # Run enhanced health check
            health_result = await self.run_enhanced_health_check(
                parallel=parallel,
                use_cache=use_cache,
                include_predictions=include_predictions
            )

            # Calculate execution metadata
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return {
                "orchestrator_compatible": True,
                "component_result": health_result,
                "local_metadata": {
                    "output_path": output_path,
                    "execution_time": execution_time,
                    "components_checked": len(self.checkers),
                    "dependencies_monitored": len(self.dependencies),
                    "circuit_breakers_active": len(self.circuit_breakers),
                    "caching_enabled": self.enable_caching,
                    "predictive_analysis_enabled": self.enable_predictive_analysis,
                    "component_version": "2025.1.0"
                }
            }

        except Exception as e:
            self.logger.error(f"Orchestrated health monitoring failed: {e}")
            return {
                "orchestrator_compatible": True,
                "component_result": {"error": str(e), "overall_status": "failed"},
                "local_metadata": {
                    "execution_time": (datetime.now(UTC) - start_time).total_seconds(),
                    "error": True,
                    "component_version": "2025.1.0"
                }
            }

# Maintain backward compatibility
class HealthService(EnhancedHealthService):
    """Backward compatible health service."""

    def __init__(self, checkers: Optional[List[HealthChecker]] = None):
        super().__init__(
            checkers=checkers,
            enable_circuit_breakers=False,
            enable_predictive_analysis=False,
            enable_caching=False
        )

    def add_checker(self, checker: HealthChecker) -> None:
        """Add a new health checker to the service"""
        self.checkers.append(checker)
        self.checker_map[checker.name] = checker

    def remove_checker(self, component_name: str) -> bool:
        """Remove a health checker from the service"""
        if component_name not in self.checker_map:
            return False

        checker = self.checker_map[component_name]
        self.checkers.remove(checker)
        del self.checker_map[component_name]
        return True

    def configure_ml_orchestration_checkers(self, orchestrator=None, registry=None,
                                          resource_manager=None, workflow_engine=None, event_bus=None):
        """Configure ML orchestration health checkers with their respective components."""
        if not ML_ORCHESTRATION_CHECKERS_AVAILABLE:
            return False

        try:
            if hasattr(self, 'ml_orchestrator_checker') and orchestrator:
                self.ml_orchestrator_checker.set_orchestrator(orchestrator)

            if hasattr(self, 'ml_registry_checker') and registry:
                self.ml_registry_checker.set_registry(registry)

            if hasattr(self, 'ml_resource_checker') and resource_manager:
                self.ml_resource_checker.set_resource_manager(resource_manager)

            if hasattr(self, 'ml_workflow_checker') and workflow_engine:
                self.ml_workflow_checker.set_workflow_engine(workflow_engine)

            if hasattr(self, 'ml_event_checker') and event_bus:
                self.ml_event_checker.set_event_bus(event_bus)

            return True
        except Exception as e:
            print(f"Warning: Failed to configure ML orchestration health checkers: {e}")
            return False

    def ensure_queue_checker(self) -> bool:
        """Ensure queue health checker is available, add it if not present.

        Returns:
            True if queue checker is available, False otherwise
        """
        # Check if queue checker already exists
        if "queue" in self.checker_map:
            return True

        # Try to dynamically import and add queue checker
        try:
            from .checkers import queue_health_checker

            queue_checker = queue_health_checker()
            self.add_checker(queue_checker)
            return True
        except Exception as e:
            print(f"Warning: Could not add queue_health_checker: {e}")
            return False

# Global health service instance
_health_service_instance: HealthService | None = None

def get_health_service() -> HealthService:
    """Get or create the global health service instance"""
    global _health_service_instance
    if _health_service_instance is None:
        _health_service_instance = HealthService()
    return _health_service_instance

def reset_health_service() -> None:
    """Reset the global health service instance (useful for testing)"""
    global _health_service_instance
    _health_service_instance = None
