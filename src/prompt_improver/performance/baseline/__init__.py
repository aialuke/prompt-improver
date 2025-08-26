"""Automated Performance Baseline System.

This module provides comprehensive performance baseline collection, analysis,
and regression detection for the prompt-improver system.

Key features:
- Continuous baseline collection and storage
- Statistical analysis of performance distributions
- Automated regression detection with alerting
- Performance trend analysis and forecasting
- Integration with existing monitoring systems
- Real production traffic analysis
- Capacity planning and optimization recommendations
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from prompt_improver.performance.baseline.automation import (
    BaselineAutomation,
    get_automation_dashboard,
    get_baseline_automation,
    start_automated_baseline_system,
    stop_automated_baseline_system,
)
from prompt_improver.performance.baseline.baseline_collector import (
    BaselineCollector,
    get_baseline_collector,
    record_operation,
    track_operation,
)
from prompt_improver.performance.baseline.enhanced_dashboard_integration import (
    PerformanceDashboard,
    get_dashboard_data,
    get_performance_dashboard,
    start_performance_dashboard,
)
from prompt_improver.performance.baseline.load_testing_integration import (
    LoadPattern,
    LoadTestConfig,
    LoadTestingIntegration,
    find_system_capacity,
    get_load_testing_integration,
    run_integrated_load_test,
)
from prompt_improver.performance.baseline.models import (
    STANDARD_METRICS,
    AlertSeverity,
    BaselineComparison,
    BaselineMetrics,
    MetricDefinition,
    MetricType,
    MetricValue,
    PerformanceTrend,
    ProfileData,
    RegressionAlert,
    TrendDirection,
    get_metric_definition,
)
from prompt_improver.performance.baseline.performance_validation_suite import (
    PerformanceValidationResult,
    PerformanceValidationSuite,
    SystemEfficiencyReport,
    get_validation_suite,
    quick_performance_check,
    validate_baseline_system_performance,
)
from prompt_improver.performance.baseline.production_optimization_guide import (
    DeploymentEnvironment,
    OptimizationLevel,
    ProductionOptimizationGuide,
    generate_production_checklist,
    get_development_config,
    get_optimization_guide,
    get_production_config,
    validate_for_production,
)
from prompt_improver.performance.baseline.profiler import (
    ContinuousProfiler,
    get_performance_summary,
    get_profiler,
    profile,
    profile_async_block,
    profile_block,
    start_continuous_profiling,
    stop_continuous_profiling,
)
from prompt_improver.performance.baseline.regression_detector import (
    RegressionDetector,
    check_baseline_for_regressions,
    get_regression_detector,
    setup_webhook_alerts,
)
from prompt_improver.performance.baseline.reporting import (
    BaselineReporter,
    generate_capacity_planning_report,
    generate_daily_performance_report,
    generate_weekly_performance_report,
    get_baseline_reporter,
)
from prompt_improver.performance.baseline.statistical_analyzer import (
    StatisticalAnalyzer,
    analyze_metric_trend,
    calculate_baseline_score,
    detect_performance_anomalies,
)
from prompt_improver.performance.monitoring.health.background_manager import (
    TaskPriority,
    get_background_task_manager,
)

logger = logging.getLogger(__name__)
_baseline_system_initialized = False
_baseline_system_running = False


class PerformanceBaselineSystem:
    """Integrated performance baseline system.

    Provides a unified interface to all baseline system components including
    collection, analysis, regression detection, profiling, automation, and reporting.
    """

    def __init__(
        self,
        enable_collection: bool = True,
        enable_analysis: bool = True,
        enable_regression_detection: bool = True,
        enable_profiling: bool = False,
        enable_automation: bool = True,
        enable_reporting: bool = True,
        integration_with_existing_monitoring: bool = True,
    ) -> None:
        """Initialize the performance baseline system.

        Args:
            enable_collection: Enable baseline collection
            enable_analysis: Enable trend analysis
            enable_regression_detection: Enable regression detection
            enable_profiling: Enable continuous profiling
            enable_automation: Enable automated scheduling
            enable_reporting: Enable automated reporting
            integration_with_existing_monitoring: Integrate with existing monitoring
        """
        self.enable_collection = enable_collection
        self.enable_analysis = enable_analysis
        self.enable_regression_detection = enable_regression_detection
        self.enable_profiling = enable_profiling
        self.enable_automation = enable_automation
        self.enable_reporting = enable_reporting
        self.integration_with_existing_monitoring = integration_with_existing_monitoring
        self.collector: BaselineCollector | None = None
        self.analyzer: StatisticalAnalyzer | None = None
        self.detector: RegressionDetector | None = None
        self.profiler: ContinuousProfiler | None = None
        self.automation: BaselineAutomation | None = None
        self.reporter: BaselineReporter | None = None
        self._running = False
        self._components_initialized = False
        logger.info("PerformanceBaselineSystem initialized")

    async def initialize(self) -> None:
        """Initialize all system components."""
        if self._components_initialized:
            logger.warning("System already initialized")
            return
        logger.info("Initializing performance baseline system components...")
        if self.enable_collection:
            self.collector = get_baseline_collector()
            logger.info("✓ Baseline collector initialized")
        if self.enable_analysis:
            self.analyzer = StatisticalAnalyzer()
            logger.info("✓ Statistical analyzer initialized")
        if self.enable_regression_detection:
            self.detector = get_regression_detector()
            logger.info("✓ Regression detector initialized")
        if self.enable_profiling:
            self.profiler = get_profiler()
            logger.info("✓ Continuous profiler initialized")
        if self.enable_automation:
            self.automation = get_baseline_automation()
            logger.info("✓ Baseline automation initialized")
        if self.enable_reporting:
            self.reporter = get_baseline_reporter()
            logger.info("✓ Baseline reporter initialized")
        if self.integration_with_existing_monitoring:
            await self._integrate_with_existing_monitoring()
        self._components_initialized = True
        logger.info("Performance baseline system initialization complete")

    async def start(self) -> None:
        """Start the performance baseline system."""
        if not self._components_initialized:
            await self.initialize()
        if self._running:
            logger.warning("System already running")
            return
        logger.info("Starting performance baseline system...")
        if self.collector and self.enable_collection:
            await self.collector.start_collection()
            logger.info("✓ Started baseline collection")
        if self.profiler and self.enable_profiling:
            await self.profiler.start_profiling()
            logger.info("✓ Started continuous profiling")
        if self.automation and self.enable_automation:
            await self.automation.start_automation()
            logger.info("✓ Started baseline automation")
        self._running = True
        logger.info("Performance baseline system started successfully")

    async def stop(self) -> None:
        """Stop the performance baseline system."""
        if not self._running:
            return
        logger.info("Stopping performance baseline system...")
        if self.automation:
            await self.automation.stop_automation()
            logger.info("✓ Stopped baseline automation")
        if self.profiler:
            await self.profiler.stop_profiling()
            logger.info("✓ Stopped continuous profiling")
        if self.collector:
            await self.collector.stop_collection()
            logger.info("✓ Stopped baseline collection")
        self._running = False
        logger.info("Performance baseline system stopped")

    async def _integrate_with_existing_monitoring(self) -> None:
        """Integrate with existing monitoring systems."""
        try:
            from prompt_improver.performance.monitoring.performance_monitor import (
                get_performance_monitor,
            )

            existing_monitor = get_performance_monitor()
            if self.collector:
                original_record = existing_monitor.record_operation

                async def enhanced_record_operation(
                    operation_name: str,
                    response_time_ms: float,
                    is_error: bool = False,
                    metadata: dict[str, Any] | None = None,
                ) -> None:
                    await original_record(
                        operation_name, response_time_ms, is_error, metadata
                    )
                    await self.collector.record_request(
                        response_time_ms, is_error, operation_name, metadata
                    )

                existing_monitor.record_operation = enhanced_record_operation
                logger.info("✓ Integrated with existing performance monitor")
            try:
                from prompt_improver.performance.monitoring.metrics_registry import (
                    get_metrics_registry,
                )

                metrics_registry = get_metrics_registry()
                baseline_metrics = {
                    "baseline_collection_total": metrics_registry.get_or_create_counter(
                        "baseline_collection_total",
                        "Total baseline collections performed",
                        ["status"],
                    ),
                    "baseline_analysis_duration": metrics_registry.get_or_create_histogram(
                        "baseline_analysis_duration_seconds",
                        "Time spent analyzing baselines",
                        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    ),
                    "baseline_regressions_detected": metrics_registry.get_or_create_counter(
                        "baseline_regressions_detected_total",
                        "Total performance regressions detected",
                        ["severity", "metric"],
                    ),
                }
                logger.info("✓ Integrated with metrics registry")
            except ImportError:
                logger.debug("Metrics registry not available for integration")
            try:
                from prompt_improver.performance.monitoring.health.service import (
                    HealthCheckService,
                )

                health_service = HealthCheckService()

                async def baseline_health_check():
                    """Health check for baseline system."""
                    if not self._running:
                        return (False, "Baseline system not running")
                    status = {}
                    if self.collector:
                        collector_status = self.collector.get_collection_status()
                        status["collector"] = collector_status["running"]
                    if self.automation:
                        automation_status = self.automation.get_automation_status()
                        status["automation"] = automation_status["running"]
                    if all(status.values()):
                        return (True, "All baseline components healthy")
                    failing_components = [
                        comp for comp, healthy in status.items() if not healthy
                    ]
                    return (
                        False,
                        f"Components not healthy: {', '.join(failing_components)}",
                    )

                logger.info("✓ Integrated with health check system")
            except ImportError:
                logger.debug("Health check service not available for integration")
        except Exception as e:
            logger.exception(f"Failed to integrate with existing monitoring: {e}")

    async def record_production_traffic(
        self,
        operation_name: str,
        response_time_ms: float,
        is_error: bool = False,
        user_id: str | None = None,
        request_size_bytes: int | None = None,
        response_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record real production traffic for baseline analysis.

        Args:
            operation_name: Name of the operation
            response_time_ms: Response time in milliseconds
            is_error: Whether the request resulted in an error
            user_id: User identifier (anonymized for privacy)
            request_size_bytes: Size of the request
            response_size_bytes: Size of the response
            metadata: Additional metadata
        """
        if not self.collector:
            return
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            "traffic_type": "production",
            "timestamp": datetime.now(UTC).isoformat(),
        })
        if user_id:
            enhanced_metadata["user_id_hash"] = hash(user_id) % 10000
        if request_size_bytes:
            enhanced_metadata["request_size_bytes"] = request_size_bytes
        if response_size_bytes:
            enhanced_metadata["response_size_bytes"] = response_size_bytes
        await self.collector.record_request(
            response_time_ms, is_error, operation_name, enhanced_metadata
        )

    async def analyze_performance_trends(
        self, hours: int = 24, metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze performance trends over a time period.

        Args:
            hours: Hours of data to analyze
            metrics: Specific metrics to analyze (defaults to all)

        Returns:
            Trend analysis results
        """
        if not self.analyzer or not self.collector:
            return {"error": "Analyzer or collector not available"}
        baselines = await self.collector.load_recent_baselines(hours)
        if not baselines:
            return {"error": "No baseline data available"}
        if metrics is None:
            metrics = [
                "response_time",
                "error_rate",
                "throughput",
                "cpu_utilization",
                "memory_utilization",
            ]
        trends = {}
        for metric_name in metrics:
            try:
                trend = await self.analyzer.analyze_trend(metric_name, baselines, hours)
                trends[metric_name] = {
                    "direction": trend.direction.value,
                    "magnitude": trend.magnitude,
                    "confidence_score": trend.confidence_score,
                    "is_significant": trend.is_significant(),
                    "sample_count": trend.sample_count,
                }
            except Exception as e:
                trends[metric_name] = {"error": str(e)}
        return {
            "timeframe_hours": hours,
            "baselines_analyzed": len(baselines),
            "trends": trends,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

    async def check_for_regressions(self) -> list[dict[str, Any]]:
        """Check for performance regressions."""
        if not self.detector or not self.collector:
            return []
        recent_baselines = await self.collector.load_recent_baselines(hours=2)
        if not recent_baselines:
            return []
        current_baseline = recent_baselines[-1]
        reference_baselines = await self.collector.load_recent_baselines(hours=168)
        alerts = await self.detector.check_for_regressions(
            current_baseline, reference_baselines[:-1]
        )
        return [alert.model_dump() for alert in alerts]

    async def generate_performance_report(
        self, report_type: str = "daily"
    ) -> dict[str, Any]:
        """Generate a performance report.

        Args:
            report_type: Type of report ('daily', 'weekly', 'monthly', 'capacity')

        Returns:
            Generated report data
        """
        if not self.reporter:
            return {"error": "Reporter not available"}
        try:
            if report_type == "daily":
                report = await self.reporter.generate_daily_report()
            elif report_type == "weekly":
                report = await self.reporter.generate_weekly_report()
            elif report_type == "monthly":
                report = await self.reporter.generate_monthly_report()
            elif report_type == "capacity":
                report = await self.reporter.generate_capacity_planning_report()
            else:
                return {"error": f"Unknown report type: {report_type}"}
            return report.model_dump()
        except Exception as e:
            return {"error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        status = {
            "initialized": self._components_initialized,
            "running": self._running,
            "components": {},
        }
        if self.collector:
            status["components"]["collector"] = self.collector.get_collection_status()
        if self.detector:
            status["components"]["detector"] = self.detector.get_alert_statistics()
        if self.profiler:
            status["components"]["profiler"] = self.profiler.get_profiler_status()
        if self.automation:
            status["components"]["automation"] = self.automation.get_automation_status()
        return status


_global_baseline_system: PerformanceBaselineSystem | None = None


def get_baseline_system() -> PerformanceBaselineSystem:
    """Get the global performance baseline system instance."""
    global _global_baseline_system
    if _global_baseline_system is None:
        _global_baseline_system = PerformanceBaselineSystem()
    return _global_baseline_system


def set_baseline_system(system: PerformanceBaselineSystem) -> None:
    """Set the global performance baseline system instance."""
    global _global_baseline_system
    _global_baseline_system = system


async def initialize_baseline_system(**kwargs) -> None:
    """Initialize the performance baseline system."""
    global _baseline_system_initialized
    if _baseline_system_initialized:
        logger.warning("Baseline system already initialized")
        return
    system = get_baseline_system()
    for key, value in kwargs.items():
        if hasattr(system, key):
            setattr(system, key, value)
    await system.initialize()
    _baseline_system_initialized = True
    logger.info("Performance baseline system initialized globally")


async def start_baseline_system() -> None:
    """Start the performance baseline system."""
    global _baseline_system_running
    if not _baseline_system_initialized:
        await initialize_baseline_system()
    if _baseline_system_running:
        logger.warning("Baseline system already running")
        return
    system = get_baseline_system()
    await system.start()
    _baseline_system_running = True
    logger.info("Performance baseline system started globally")


async def stop_baseline_system() -> None:
    """Stop the performance baseline system."""
    global _baseline_system_running
    if not _baseline_system_running:
        return
    system = get_baseline_system()
    await system.stop()
    _baseline_system_running = False
    logger.info("Performance baseline system stopped globally")


def is_baseline_system_running() -> bool:
    """Check if the baseline system is running."""
    return _baseline_system_running


async def record_production_request(
    operation_name: str, response_time_ms: float, is_error: bool = False, **kwargs
) -> None:
    """Record a production request for baseline analysis."""
    if _baseline_system_running:
        system = get_baseline_system()
        await system.record_production_traffic(
            operation_name, response_time_ms, is_error, **kwargs
        )


class track_production_operation:
    """Context manager for tracking production operations."""

    def __init__(self, operation_name: str, **metadata) -> None:
        self.operation_name = operation_name
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            is_error = exc_type is not None
            task_manager = get_background_task_manager()
            asyncio.create_task(
                task_manager.submit_enhanced_task(
                    task_id=f"baseline_production_record_{self.operation_name}_{str(uuid.uuid4())[:8]}",
                    coroutine=record_production_request(
                        self.operation_name, duration_ms, is_error, **self.metadata
                    ),
                    priority=TaskPriority.NORMAL,
                    tags={
                        "service": "performance",
                        "type": "tracking",
                        "component": "baseline",
                        "operation": self.operation_name,
                    },
                )
            )


__all__ = [
    "STANDARD_METRICS",
    "AlertSeverity",
    "BaselineAutomation",
    "BaselineCollector",
    "BaselineComparison",
    "BaselineMetrics",
    "BaselineReporter",
    "ContinuousProfiler",
    "DeploymentEnvironment",
    "LoadPattern",
    "LoadTestConfig",
    "LoadTestingIntegration",
    "MetricDefinition",
    "MetricType",
    "MetricValue",
    "OptimizationLevel",
    "PerformanceBaselineSystem",
    "PerformanceDashboard",
    "PerformanceTrend",
    "PerformanceValidationResult",
    "PerformanceValidationSuite",
    "ProductionOptimizationGuide",
    "ProfileData",
    "RegressionAlert",
    "RegressionDetector",
    "StatisticalAnalyzer",
    "SystemEfficiencyReport",
    "TrendDirection",
    "analyze_metric_trend",
    "calculate_baseline_score",
    "check_baseline_for_regressions",
    "detect_performance_anomalies",
    "find_system_capacity",
    "generate_capacity_planning_report",
    "generate_daily_performance_report",
    "generate_production_checklist",
    "generate_weekly_performance_report",
    "get_automation_dashboard",
    "get_baseline_automation",
    "get_baseline_collector",
    "get_baseline_reporter",
    "get_baseline_system",
    "get_dashboard_data",
    "get_development_config",
    "get_load_testing_integration",
    "get_metric_definition",
    "get_optimization_guide",
    "get_performance_dashboard",
    "get_performance_summary",
    "get_production_config",
    "get_profiler",
    "get_regression_detector",
    "get_validation_suite",
    "initialize_baseline_system",
    "is_baseline_system_running",
    "profile",
    "profile_async_block",
    "profile_block",
    "quick_performance_check",
    "record_operation",
    "record_production_request",
    "run_integrated_load_test",
    "set_baseline_system",
    "setup_webhook_alerts",
    "start_automated_baseline_system",
    "start_baseline_system",
    "start_continuous_profiling",
    "start_performance_dashboard",
    "stop_automated_baseline_system",
    "stop_baseline_system",
    "stop_continuous_profiling",
    "track_operation",
    "track_production_operation",
    "validate_baseline_system_performance",
    "validate_for_production",
]
