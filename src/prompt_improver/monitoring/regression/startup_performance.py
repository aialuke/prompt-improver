"""Startup Performance Tracker.
===========================

Monitors application startup performance and detects dependency contamination
that could reintroduce the 92.4% startup improvement regression.

Key Monitoring Areas:
- Import chain performance measurement
- Heavy dependency loading detection
- Memory usage during startup
- Module loading time analysis
- Dependency contamination alerts

Protects against:
- NumPy/torch contamination during import (1007ms penalty)
- SQLAlchemy/asyncpg loading at module level (134-223ms penalty)
- beartype/coredis automatic imports causing cascading loads
"""

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
    startup_tracer = trace.get_tracer(__name__)
    startup_meter = metrics.get_meter(__name__)

    # Startup performance metrics
    startup_duration_histogram = startup_meter.create_histogram(
        "startup_duration_seconds",
        description="Application startup duration by component",
        unit="s"
    )

    import_duration_histogram = startup_meter.create_histogram(
        "import_duration_seconds",
        description="Module import duration",
        unit="s"
    )

    memory_usage_gauge = startup_meter.create_gauge(
        "startup_memory_usage_mb",
        description="Memory usage during startup",
        unit="MB"
    )

    dependency_contamination_counter = startup_meter.create_counter(
        "dependency_contamination_violations_total",
        description="Dependencies loaded during startup that shouldn't be",
        unit="1"
    )

except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    startup_tracer = None
    startup_duration_histogram = None
    import_duration_histogram = None
    memory_usage_gauge = None
    dependency_contamination_counter = None

logger = logging.getLogger(__name__)


@dataclass
class ImportMetrics:
    """Metrics for a single import operation."""
    module_name: str
    duration_seconds: float
    memory_delta_mb: float
    is_heavy_dependency: bool
    dependency_chain: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StartupProfile:
    """Comprehensive startup performance profile."""
    startup_time_seconds: float
    total_imports: int
    heavy_imports: int
    memory_usage_mb: float
    memory_peak_mb: float
    contaminated_dependencies: list[str] = field(default_factory=list)
    import_metrics: list[ImportMetrics] = field(default_factory=list)
    performance_violations: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class StartupPerformanceTracker:
    """Track and analyze application startup performance.

    Monitors import chains, memory usage, and dependency loading to prevent
    performance regressions that could reintroduce startup penalties.
    """

    # Dependencies that should NEVER be imported during startup
    PROHIBITED_STARTUP_DEPS = {
        "torch", "transformers", "tensorflow",  # ML frameworks
        "pandas", "numpy",  # Data processing (unless explicitly needed)
        "matplotlib", "seaborn", "plotly",  # Visualization
        "jupyter", "ipython",  # Interactive tools
        "selenium", "playwright"  # Browser automation
    }

    # Dependencies that can be imported but should be monitored
    MONITORED_DEPS = {
        "sqlalchemy", "asyncpg", "psycopg2",  # Database
        "redis", "coredis",  # Cache
        "beartype", "pydantic",  # Validation
        "fastapi", "starlette",  # Web framework
        "opentelemetry"  # Observability
    }

    # Performance thresholds (based on Python 2025 realistic targets)
    PERFORMANCE_THRESHOLDS = {
        "total_startup_seconds": 0.5,  # 500ms total startup
        "single_import_seconds": 0.1,   # 100ms per import
        "memory_usage_mb": 100,         # 100MB memory limit
        "heavy_import_count": 3         # Max 3 heavy imports
    }

    def __init__(self) -> None:
        """Initialize startup performance tracker."""
        self.import_history: list[ImportMetrics] = []
        self.startup_profiles: list[StartupProfile] = []
        self.original_import = __builtins__['__import__']
        self.monitoring_active = False
        self.current_profile: StartupProfile | None = None

        # Memory tracking
        self.memory_baseline = 0
        self.memory_peak = 0

        logger.info("StartupPerformanceTracker initialized")

    @contextmanager
    def monitor_startup(self, component_name: str = "application"):
        """Context manager to monitor startup performance.

        Args:
            component_name: Name of the component being started
        """
        start_time = time.time()

        # Start memory monitoring
        tracemalloc.start()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.memory_baseline = process.memory_info().rss / 1024 / 1024  # MB

        # Start import monitoring
        self._start_import_monitoring()

        with startup_tracer.start_as_current_span(f"startup_{component_name}") if startup_tracer else None as span:
            try:
                self.current_profile = StartupProfile(
                    startup_time_seconds=0,
                    total_imports=0,
                    heavy_imports=0,
                    memory_usage_mb=0,
                    memory_peak_mb=0
                )

                yield self.current_profile

                # Calculate final metrics
                duration = time.time() - start_time
                self.current_profile.startup_time_seconds = duration

                if PSUTIL_AVAILABLE:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    self.current_profile.memory_usage_mb = current_memory - self.memory_baseline
                    self.current_profile.memory_peak_mb = self.memory_peak

                # Analyze performance violations
                self._analyze_performance_violations()

                # Record metrics
                if OPENTELEMETRY_AVAILABLE:
                    self._record_startup_metrics(component_name)

                # Store profile
                self.startup_profiles.append(self.current_profile)

                if span:
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("startup_duration", duration)
                    span.set_attribute("total_imports", self.current_profile.total_imports)
                    span.set_attribute("violations", len(self.current_profile.performance_violations))

                logger.info(
                    f"Startup completed for {component_name}: {duration:.3f}s, "
                    f"{self.current_profile.total_imports} imports, "
                    f"{self.current_profile.memory_usage_mb:.1f}MB memory"
                )

            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                logger.exception(f"Startup monitoring failed for {component_name}: {e}")
                raise
            finally:
                self._stop_import_monitoring()
                tracemalloc.stop()
                self.current_profile = None

    def _start_import_monitoring(self) -> None:
        """Start monitoring import operations."""
        self.monitoring_active = True

        def monitored_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Monitored version of __import__."""
            import_start = time.time()

            # Track memory before import
            memory_before = 0
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024
                except Exception:
                    pass

            try:
                # Perform the actual import
                result = self.original_import(name, globals, locals, fromlist, level)

                # Calculate metrics
                duration = time.time() - import_start
                memory_after = memory_before

                if PSUTIL_AVAILABLE:
                    try:
                        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                        self.memory_peak = max(self.memory_peak, memory_after)
                    except Exception:
                        pass

                memory_delta = memory_after - memory_before

                # Check if this is a heavy dependency
                is_heavy = any(dep in name for dep in
                              (self.PROHIBITED_STARTUP_DEPS | self.MONITORED_DEPS))

                # Create import metrics
                import_metrics = ImportMetrics(
                    module_name=name,
                    duration_seconds=duration,
                    memory_delta_mb=memory_delta,
                    is_heavy_dependency=is_heavy
                )

                # Store metrics
                self.import_history.append(import_metrics)
                if self.current_profile:
                    self.current_profile.import_metrics.append(import_metrics)
                    self.current_profile.total_imports += 1
                    if is_heavy:
                        self.current_profile.heavy_imports += 1

                # Check for prohibited dependencies
                if any(dep in name for dep in self.PROHIBITED_STARTUP_DEPS):
                    self.current_profile.contaminated_dependencies.append(name)

                    # Record violation
                    if dependency_contamination_counter:
                        dependency_contamination_counter.add(1, {"dependency": name})

                    logger.warning(f"Prohibited dependency imported during startup: {name}")

                # Record import duration
                if import_duration_histogram:
                    import_duration_histogram.record(duration, {"module": name, "is_heavy": str(is_heavy)})

                return result

            except Exception as e:
                logger.debug(f"Import monitoring error for {name}: {e}")
                # Fallback to original import
                return self.original_import(name, globals, locals, fromlist, level)

        # Replace __import__
        __builtins__['__import__'] = monitored_import

    def _stop_import_monitoring(self) -> None:
        """Stop monitoring import operations."""
        self.monitoring_active = False
        __builtins__['__import__'] = self.original_import

    def _analyze_performance_violations(self) -> None:
        """Analyze current profile for performance violations."""
        if not self.current_profile:
            return

        violations = []

        # Check total startup time
        if self.current_profile.startup_time_seconds > self.PERFORMANCE_THRESHOLDS["total_startup_seconds"]:
            violations.append({
                "type": "slow_startup",
                "severity": "high",
                "value": self.current_profile.startup_time_seconds,
                "threshold": self.PERFORMANCE_THRESHOLDS["total_startup_seconds"],
                "message": f"Startup took {self.current_profile.startup_time_seconds:.3f}s (>{self.PERFORMANCE_THRESHOLDS['total_startup_seconds']}s threshold)"
            })

        # Check memory usage
        if self.current_profile.memory_usage_mb > self.PERFORMANCE_THRESHOLDS["memory_usage_mb"]:
            violations.append({
                "type": "high_memory_usage",
                "severity": "medium",
                "value": self.current_profile.memory_usage_mb,
                "threshold": self.PERFORMANCE_THRESHOLDS["memory_usage_mb"],
                "message": f"Startup used {self.current_profile.memory_usage_mb:.1f}MB memory (>{self.PERFORMANCE_THRESHOLDS['memory_usage_mb']}MB threshold)"
            })

        # Check slow individual imports
        violations.extend({
                    "type": "slow_import",
                    "severity": "medium",
                    "value": import_metric.duration_seconds,
                    "threshold": self.PERFORMANCE_THRESHOLDS["single_import_seconds"],
                    "module": import_metric.module_name,
                    "message": f"Import of {import_metric.module_name} took {import_metric.duration_seconds:.3f}s"
                } for import_metric in self.current_profile.import_metrics if import_metric.duration_seconds > self.PERFORMANCE_THRESHOLDS["single_import_seconds"])

        # Check prohibited dependencies
        if self.current_profile.contaminated_dependencies:
            violations.append({
                "type": "dependency_contamination",
                "severity": "critical",
                "value": len(self.current_profile.contaminated_dependencies),
                "dependencies": self.current_profile.contaminated_dependencies,
                "message": f"Prohibited dependencies loaded during startup: {', '.join(self.current_profile.contaminated_dependencies)}"
            })

        # Check heavy import count
        if self.current_profile.heavy_imports > self.PERFORMANCE_THRESHOLDS["heavy_import_count"]:
            violations.append({
                "type": "too_many_heavy_imports",
                "severity": "high",
                "value": self.current_profile.heavy_imports,
                "threshold": self.PERFORMANCE_THRESHOLDS["heavy_import_count"],
                "message": f"Too many heavy imports during startup: {self.current_profile.heavy_imports}"
            })

        self.current_profile.performance_violations = violations

    def _record_startup_metrics(self, component_name: str) -> None:
        """Record startup metrics to OpenTelemetry."""
        if not OPENTELEMETRY_AVAILABLE or not self.current_profile:
            return

        # Record startup duration
        if startup_duration_histogram:
            startup_duration_histogram.record(
                self.current_profile.startup_time_seconds,
                {"component": component_name}
            )

        # Record memory usage
        if memory_usage_gauge:
            memory_usage_gauge.set(
                self.current_profile.memory_usage_mb,
                {"component": component_name, "type": "startup"}
            )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of startup performance."""
        if not self.startup_profiles:
            return {
                "status": "no_profiles",
                "message": "No startup profiles available"
            }

        latest_profile = self.startup_profiles[-1]

        # Calculate averages from recent profiles (last 10)
        recent_profiles = self.startup_profiles[-10:]
        avg_startup_time = sum(p.startup_time_seconds for p in recent_profiles) / len(recent_profiles)
        avg_memory_usage = sum(p.memory_usage_mb for p in recent_profiles) / len(recent_profiles)
        avg_imports = sum(p.total_imports for p in recent_profiles) / len(recent_profiles)

        return {
            "status": "healthy" if not latest_profile.performance_violations else "violations_detected",
            "latest_startup": {
                "duration_seconds": latest_profile.startup_time_seconds,
                "memory_usage_mb": latest_profile.memory_usage_mb,
                "total_imports": latest_profile.total_imports,
                "heavy_imports": latest_profile.heavy_imports,
                "contaminated_dependencies": latest_profile.contaminated_dependencies,
                "violations": len(latest_profile.performance_violations)
            },
            "averages": {
                "startup_time_seconds": avg_startup_time,
                "memory_usage_mb": avg_memory_usage,
                "imports_count": avg_imports
            },
            "performance_status": {
                "startup_time_ok": latest_profile.startup_time_seconds <= self.PERFORMANCE_THRESHOLDS["total_startup_seconds"],
                "memory_usage_ok": latest_profile.memory_usage_mb <= self.PERFORMANCE_THRESHOLDS["memory_usage_mb"],
                "no_contamination": len(latest_profile.contaminated_dependencies) == 0,
                "heavy_imports_ok": latest_profile.heavy_imports <= self.PERFORMANCE_THRESHOLDS["heavy_import_count"]
            },
            "thresholds": self.PERFORMANCE_THRESHOLDS,
            "total_profiles": len(self.startup_profiles),
            "timestamp": latest_profile.timestamp
        }

    def get_slow_imports(self, threshold_seconds: float = 0.05) -> list[dict[str, Any]]:
        """Get list of slow imports across all profiles.

        Args:
            threshold_seconds: Minimum duration to consider "slow"

        Returns:
            List of slow import information
        """
        slow_imports = [{
                    "module": import_metric.module_name,
                    "duration_seconds": import_metric.duration_seconds,
                    "memory_delta_mb": import_metric.memory_delta_mb,
                    "is_heavy": import_metric.is_heavy_dependency,
                    "timestamp": import_metric.timestamp
                } for import_metric in self.import_history if import_metric.duration_seconds >= threshold_seconds]

        # Sort by duration descending
        slow_imports.sort(key=lambda x: x["duration_seconds"], reverse=True)
        return slow_imports

    def analyze_dependency_trends(self) -> dict[str, Any]:
        """Analyze trends in dependency loading."""
        if not self.startup_profiles:
            return {"message": "No startup profiles to analyze"}

        # Group imports by module
        module_stats = {}
        for profile in self.startup_profiles:
            for import_metric in profile.import_metrics:
                module = import_metric.module_name
                if module not in module_stats:
                    module_stats[module] = {
                        "count": 0,
                        "total_duration": 0,
                        "total_memory": 0,
                        "is_heavy": import_metric.is_heavy_dependency
                    }

                stats = module_stats[module]
                stats["count"] += 1
                stats["total_duration"] += import_metric.duration_seconds
                stats["total_memory"] += import_metric.memory_delta_mb

        # Calculate averages and identify problematic modules
        problematic_modules = []
        for module, stats in module_stats.items():
            avg_duration = stats["total_duration"] / stats["count"]
            avg_memory = stats["total_memory"] / stats["count"]

            if (avg_duration > 0.05 or  # >50ms average
                avg_memory > 10 or      # >10MB average
                stats["is_heavy"]):

                problematic_modules.append({
                    "module": module,
                    "average_duration_seconds": avg_duration,
                    "average_memory_mb": avg_memory,
                    "import_count": stats["count"],
                    "is_heavy": stats["is_heavy"],
                    "total_duration": stats["total_duration"]
                })

        # Sort by total impact (duration * count)
        problematic_modules.sort(key=lambda x: x["total_duration"], reverse=True)

        return {
            "total_unique_modules": len(module_stats),
            "problematic_modules": problematic_modules[:20],  # Top 20
            "heavy_modules": [m for m in problematic_modules if m["is_heavy"]],
            "contamination_risk": len([m for m in problematic_modules
                                     if any(dep in m["module"] for dep in self.PROHIBITED_STARTUP_DEPS)])
        }

    async def validate_startup_performance(self, target_duration_seconds: float = 0.5) -> tuple[bool, dict[str, Any]]:
        """Validate that startup performance meets targets.

        Args:
            target_duration_seconds: Maximum allowed startup duration

        Returns:
            Tuple of (meets_target, detailed_analysis)
        """
        if not self.startup_profiles:
            return False, {"error": "No startup profiles to validate"}

        latest_profile = self.startup_profiles[-1]
        meets_target = (
            latest_profile.startup_time_seconds <= target_duration_seconds and
            len(latest_profile.contaminated_dependencies) == 0 and
            not any(v["severity"] == "critical" for v in latest_profile.performance_violations)
        )

        analysis = {
            "meets_target": meets_target,
            "target_duration_seconds": target_duration_seconds,
            "actual_duration_seconds": latest_profile.startup_time_seconds,
            "performance_delta": latest_profile.startup_time_seconds - target_duration_seconds,
            "contaminated_dependencies": latest_profile.contaminated_dependencies,
            "critical_violations": [v for v in latest_profile.performance_violations if v["severity"] == "critical"],
            "recommendations": []
        }

        # Generate recommendations
        if latest_profile.startup_time_seconds > target_duration_seconds:
            slow_imports = [m for m in latest_profile.import_metrics
                           if m.duration_seconds > 0.05]  # >50ms
            if slow_imports:
                analysis["recommendations"].append({
                    "type": "optimize_slow_imports",
                    "message": f"Optimize {len(slow_imports)} slow imports",
                    "slow_imports": [m.module_name for m in slow_imports[:5]]  # Top 5
                })

        if latest_profile.contaminated_dependencies:
            analysis["recommendations"].append({
                "type": "remove_prohibited_deps",
                "message": "Remove prohibited dependencies from startup path",
                "dependencies": latest_profile.contaminated_dependencies
            })

        if latest_profile.heavy_imports > self.PERFORMANCE_THRESHOLDS["heavy_import_count"]:
            analysis["recommendations"].append({
                "type": "reduce_heavy_imports",
                "message": f"Reduce heavy imports from {latest_profile.heavy_imports} to ≤{self.PERFORMANCE_THRESHOLDS['heavy_import_count']}",
                "heavy_import_count": latest_profile.heavy_imports
            })

        return meets_target, analysis
