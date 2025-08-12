"""Advanced Memory Leak Detection Framework

This module provides comprehensive memory profiling and leak detection for
validation operations, designed to handle 100k+ operations while maintaining
detailed tracking of memory allocations, deallocations, and growth patterns.

Key Features:
1. High-precision memory tracking with tracemalloc integration
2. Leak detection algorithms for different patterns (linear, exponential, cyclic)
3. Memory hotspot identification and allocation tracing
4. GC stress testing and optimization recommendations
5. Real-time memory monitoring with configurable thresholds
"""

import asyncio
import gc
import json
import logging
import statistics
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import psutil

logger = logging.getLogger(__name__)


class LeakPattern(Enum):
    """Memory leak pattern types."""

    LINEAR = "linear"  # Steady linear growth
    EXPONENTIAL = "exponential"  # Exponential growth pattern
    CYCLIC = "cyclic"  # Periodic memory spikes
    BURST = "burst"  # Sudden large allocations
    FRAGMENTATION = "fragmentation"  # Memory fragmentation issues


class MemoryThreshold(Enum):
    """Memory usage threshold levels."""

    LOW = 0.6  # 60% of available memory
    MEDIUM = 0.75  # 75% of available memory
    HIGH = 0.85  # 85% of available memory
    CRITICAL = 0.95  # 95% of available memory


@dataclass
class AllocationSnapshot:
    """Single memory allocation snapshot."""

    timestamp: datetime
    total_size: int  # Total allocated bytes
    peak_size: int  # Peak memory usage
    current_size: int  # Current allocated memory
    count: int  # Number of allocations
    top_files: list[tuple[str, int]]  # Top file allocations
    operation_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_size": self.total_size,
            "peak_size": self.peak_size,
            "current_size": self.current_size,
            "count": self.count,
            "top_files": self.top_files,
            "operation_context": self.operation_context,
            "metadata": self.metadata,
        }


@dataclass
class MemoryLeak:
    """Detected memory leak information."""

    operation_name: str
    pattern: LeakPattern
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    growth_rate_mb_per_hour: float
    total_leaked_mb: float
    detection_confidence: float  # 0.0 to 1.0
    first_detected: datetime
    last_confirmed: datetime
    allocation_sources: list[tuple[str, int, str]]  # (file, line, function)
    recommendations: list[str]
    is_confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "pattern": self.pattern.value,
            "severity": self.severity,
            "growth_rate_mb_per_hour": self.growth_rate_mb_per_hour,
            "total_leaked_mb": self.total_leaked_mb,
            "detection_confidence": self.detection_confidence,
            "first_detected": self.first_detected.isoformat(),
            "last_confirmed": self.last_confirmed.isoformat(),
            "allocation_sources": self.allocation_sources,
            "recommendations": self.recommendations,
            "is_confirmed": self.is_confirmed,
            "metadata": self.metadata,
        }


@dataclass
class GCStressTestResult:
    """Garbage collection stress test results."""

    operation_name: str
    pre_gc_memory_mb: float
    post_gc_memory_mb: float
    memory_freed_mb: float
    gc_time_seconds: float
    objects_collected: int
    objects_remaining: int
    gc_efficiency: float  # Percentage of memory freed
    leak_suspected: bool
    recommendations: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "pre_gc_memory_mb": self.pre_gc_memory_mb,
            "post_gc_memory_mb": self.post_gc_memory_mb,
            "memory_freed_mb": self.memory_freed_mb,
            "gc_time_seconds": self.gc_time_seconds,
            "objects_collected": self.objects_collected,
            "objects_remaining": self.objects_remaining,
            "gc_efficiency": self.gc_efficiency,
            "leak_suspected": self.leak_suspected,
            "recommendations": self.recommendations,
        }


class MemoryTracker:
    """Real-time memory tracking and analysis."""

    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.operation_snapshots: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self.baseline_memory_mb = self._get_current_memory_mb()

        # Tracking state
        self.is_tracking = False
        self.tracking_start_time: datetime | None = None

        # Weak references to track object lifecycle
        self.tracked_objects: set[weakref.ref] = set()

    def start_tracking(self) -> None:
        """Start memory tracking."""
        if not self.is_tracking:
            tracemalloc.start()
            self.is_tracking = True
            self.tracking_start_time = datetime.now(UTC)
            self.baseline_memory_mb = self._get_current_memory_mb()
            logger.info(
                f"Memory tracking started - baseline: {self.baseline_memory_mb:.2f}MB"
            )

    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        if self.is_tracking:
            tracemalloc.stop()
            self.is_tracking = False
            logger.info("Memory tracking stopped")

    def take_snapshot(self, operation_context: str = "") -> AllocationSnapshot:
        """Take a memory allocation snapshot."""
        if not self.is_tracking:
            self.start_tracking()

        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("filename")

        # Get system memory info
        process = psutil.Process()
        memory_info = process.memory_info()

        current_memory_mb = memory_info.rss / 1024 / 1024

        # Extract top file allocations
        top_files = []
        for stat in top_stats[:10]:
            formatted_traceback = stat.traceback.format()
            file_info = formatted_traceback[0] if formatted_traceback else "unknown"
            top_files.append((file_info, stat.size))

        allocation_snapshot = AllocationSnapshot(
            timestamp=datetime.now(UTC),
            total_size=sum(stat.size for stat in top_stats),
            peak_size=int(current_memory_mb * 1024 * 1024),
            current_size=memory_info.rss,
            count=len(top_stats),
            top_files=top_files,
            operation_context=operation_context,
            metadata={
                "vms": memory_info.vms,
                "shared": memory_info.shared if hasattr(memory_info, "shared") else 0,
                "cpu_percent": process.cpu_percent(),
            },
        )

        # Store snapshot
        self.snapshots.append(allocation_snapshot)
        if operation_context:
            self.operation_snapshots[operation_context].append(allocation_snapshot)

        return allocation_snapshot

    def track_object(self, obj: Any) -> None:
        """Track an object's lifecycle for leak detection."""
        try:
            weak_ref = weakref.ref(obj)
            self.tracked_objects.add(weak_ref)
        except TypeError:
            # Object doesn't support weak references
            pass

    def cleanup_dead_references(self) -> int:
        """Clean up dead weak references and return count."""
        before_count = len(self.tracked_objects)
        self.tracked_objects = {
            ref for ref in self.tracked_objects if ref() is not None
        }
        cleaned_count = before_count - len(self.tracked_objects)
        return cleaned_count

    def get_memory_growth_rate(
        self, operation_context: str, hours: float = 1.0
    ) -> float | None:
        """Calculate memory growth rate for an operation in MB/hour."""
        if operation_context not in self.operation_snapshots:
            return None

        snapshots = list(self.operation_snapshots[operation_context])
        if len(snapshots) < 2:
            return None

        # Filter snapshots within the time window
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]

        if len(recent_snapshots) < 2:
            return None

        # Calculate linear growth rate
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]

        time_diff_hours = (
            last_snapshot.timestamp - first_snapshot.timestamp
        ).total_seconds() / 3600
        if time_diff_hours == 0:
            return None

        memory_diff_mb = (
            (last_snapshot.current_size - first_snapshot.current_size) / 1024 / 1024
        )
        growth_rate = memory_diff_mb / time_diff_hours

        return growth_rate

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class MemoryLeakDetector:
    """Advanced memory leak detection and analysis."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("memory_analysis")
        self.data_dir.mkdir(exist_ok=True)

        self.tracker = MemoryTracker()
        self.detected_leaks: list[MemoryLeak] = []

        # Detection thresholds
        self.leak_thresholds = {
            "linear_growth_mb_per_hour": 5.0,  # 5MB/hour sustained growth
            "exponential_growth_factor": 1.5,  # 50% growth rate
            "burst_size_mb": 50.0,  # 50MB sudden allocation
            "fragmentation_ratio": 0.3,  # 30% fragmentation threshold
            "confidence_threshold": 0.7,  # 70% confidence for confirmed leaks
        }

    async def run_comprehensive_leak_detection(
        self, operations: int = 100000, operation_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Run comprehensive memory leak detection across multiple operations.

        Args:
            operations: Number of operations to run for each type
            operation_types: List of operation types to test (default: all validation types)

        Returns:
            Comprehensive leak detection results
        """
        if operation_types is None:
            operation_types = [
                "mcp_message_decode",
                "config_instantiation",
                "metrics_collection",
                "cache_operations",
                "concurrent_validation",
            ]

        logger.info(
            f"Starting comprehensive leak detection with {operations:,} operations per type"
        )

        results = {}

        for operation_type in operation_types:
            logger.info(f"Testing {operation_type} for memory leaks...")

            # Start tracking for this operation
            self.tracker.start_tracking()
            initial_snapshot = self.tracker.take_snapshot(f"{operation_type}_baseline")

            # Run operations with periodic memory snapshots
            leak_result = await self._detect_leaks_in_operation(
                operation_type=operation_type, iterations=operations
            )

            results[operation_type] = leak_result

            # GC stress test
            gc_result = await self._run_gc_stress_test(operation_type)
            results[f"{operation_type}_gc_test"] = gc_result

            # Clean up between operations
            gc.collect()
            await asyncio.sleep(0.1)

        # Generate comprehensive report
        comprehensive_report = await self._generate_leak_report(results)

        return comprehensive_report

    async def _detect_leaks_in_operation(
        self, operation_type: str, iterations: int
    ) -> dict[str, Any]:
        """Detect memory leaks in a specific operation type."""
        snapshots = []
        errors = 0

        # Take initial baseline
        baseline_snapshot = self.tracker.take_snapshot(f"{operation_type}_baseline")
        snapshots.append(baseline_snapshot)

        # Run operations with periodic memory tracking
        snapshot_interval = max(1, iterations // 50)  # Take 50 snapshots max

        for i in range(iterations):
            try:
                # Run the specific operation
                await self._execute_operation(operation_type, i)

                # Take memory snapshot periodically
                if i % snapshot_interval == 0:
                    snapshot = self.tracker.take_snapshot(f"{operation_type}_{i}")
                    snapshots.append(snapshot)

                # Track objects for lifecycle analysis
                if i % 1000 == 0:
                    # Create test object for tracking
                    test_obj = {"iteration": i, "timestamp": time.time()}
                    self.tracker.track_object(test_obj)

                # Yield control occasionally
                if i % 5000 == 0:
                    await asyncio.sleep(0.01)

            except Exception as e:
                errors += 1
                if errors > iterations * 0.01:  # More than 1% error rate
                    logger.warning(
                        f"High error rate in {operation_type}: {errors}/{i + 1}"
                    )
                    break

        # Take final snapshot
        final_snapshot = self.tracker.take_snapshot(f"{operation_type}_final")
        snapshots.append(final_snapshot)

        # Analyze memory patterns
        leak_analysis = self._analyze_memory_patterns(operation_type, snapshots)

        return {
            "operation_type": operation_type,
            "iterations_completed": iterations - errors,
            "error_count": errors,
            "total_snapshots": len(snapshots),
            "baseline_memory_mb": baseline_snapshot.current_size / 1024 / 1024,
            "final_memory_mb": final_snapshot.current_size / 1024 / 1024,
            "memory_growth_mb": (
                final_snapshot.current_size - baseline_snapshot.current_size
            )
            / 1024
            / 1024,
            "leak_analysis": leak_analysis,
            "snapshots": [
                s.model_dump() for s in snapshots[-10:]
            ],  # Keep last 10 snapshots
        }

    async def _execute_operation(self, operation_type: str, iteration: int) -> None:
        """Execute a specific operation for memory testing."""
        if operation_type == "mcp_message_decode":
            from prompt_improver.core.types import PromptImprovementRequest

            # Create and validate request
            request = PromptImprovementRequest(
                prompt=f"Memory test iteration {iteration}: Create a function to process data",
                session_id=f"mem_test_{iteration % 100}",
            )

            # Simulate some processing
            _ = len(request.prompt)
            _ = request.session_id.split("_")

        elif operation_type == "config_instantiation":
            from prompt_improver.core.config import (
                AppConfig,
                DatabaseConfig,
                RedisConfig,
            )

            # Create config with nested objects
            config = AppConfig(
                environment="testing",
                debug=iteration % 2 == 0,
                database=DatabaseConfig(database_pool_size=10 + (iteration % 5)),
                redis=RedisConfig(redis_max_connections=100 + (iteration % 10)),
            )

            # Access config properties
            _ = config.database.database_url
            _ = config.redis.redis_url

        elif operation_type == "metrics_collection":
            from prompt_improver.metrics.api_metrics import (
                APIUsageMetric,
                EndpointCategory,
                HTTPMethod,
            )

            # Create metrics object
            metric = APIUsageMetric(
                endpoint=f"/api/test/{iteration}",
                method=HTTPMethod.POST,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200,
                response_time_ms=float(iteration % 1000),
                request_size_bytes=1024 + (iteration % 512),
                response_size_bytes=2048 + (iteration % 1024),
            )

            # Simulate metric processing
            _ = metric.endpoint
            _ = metric.response_time_ms * 1000

        elif operation_type == "cache_operations":
            # Simulate cache operations
            cache_data = {
                f"key_{iteration}": {
                    "value": f"cached_value_{iteration}",
                    "metadata": {"created": time.time(), "iteration": iteration},
                }
            }

            # Simulate cache access patterns
            for key, value in cache_data.items():
                _ = len(str(value))

        elif operation_type == "concurrent_validation":
            # Simulate concurrent validation scenarios
            tasks_data = [
                {"id": f"task_{iteration}_{j}", "data": f"validation_data_{j}"}
                for j in range(min(10, iteration % 20 + 1))
            ]

            # Process tasks
            for task in tasks_data:
                _ = task["id"]
                _ = len(task["data"])

        # Force some object creation and cleanup
        temp_objects = [f"temp_{i}" for i in range(iteration % 10)]
        del temp_objects

    def _analyze_memory_patterns(
        self, operation_type: str, snapshots: list[AllocationSnapshot]
    ) -> dict[str, Any]:
        """Analyze memory allocation patterns for leak detection."""
        if len(snapshots) < 3:
            return {"error": "Insufficient snapshots for analysis"}

        # Extract memory usage over time
        memory_mb_series = [s.current_size / 1024 / 1024 for s in snapshots]
        time_series = [
            (s.timestamp - snapshots[0].timestamp).total_seconds() / 3600
            for s in snapshots
        ]  # Hours

        analysis = {
            "total_snapshots": len(snapshots),
            "memory_growth_mb": memory_mb_series[-1] - memory_mb_series[0],
            "peak_memory_mb": max(memory_mb_series),
            "average_memory_mb": statistics.mean(memory_mb_series),
            "memory_volatility": statistics.stdev(memory_mb_series)
            if len(memory_mb_series) > 1
            else 0,
            "leak_patterns_detected": [],
            "confidence_scores": {},
            "recommendations": [],
        }

        # Linear growth detection
        linear_leak = self._detect_linear_leak(time_series, memory_mb_series)
        if linear_leak:
            analysis["leak_patterns_detected"].append(linear_leak)

        # Exponential growth detection
        exponential_leak = self._detect_exponential_leak(time_series, memory_mb_series)
        if exponential_leak:
            analysis["leak_patterns_detected"].append(exponential_leak)

        # Burst allocation detection
        burst_leak = self._detect_burst_allocation(snapshots)
        if burst_leak:
            analysis["leak_patterns_detected"].append(burst_leak)

        # Cyclic pattern detection
        cyclic_leak = self._detect_cyclic_pattern(memory_mb_series)
        if cyclic_leak:
            analysis["leak_patterns_detected"].append(cyclic_leak)

        # Generate overall assessment
        if analysis["leak_patterns_detected"]:
            analysis["leak_suspected"] = True
            analysis["primary_pattern"] = analysis["leak_patterns_detected"][0][
                "pattern"
            ]
            analysis["recommendations"] = self._generate_leak_recommendations(
                operation_type, analysis["leak_patterns_detected"]
            )
        else:
            analysis["leak_suspected"] = False
            analysis["recommendations"] = [
                "No memory leaks detected - memory usage appears stable"
            ]

        return analysis

    def _detect_linear_leak(
        self, time_hours: list[float], memory_mb: list[float]
    ) -> dict[str, Any] | None:
        """Detect linear memory growth pattern."""
        if len(time_hours) < 3:
            return None

        try:
            import numpy as np
            from scipy import stats

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_hours, memory_mb
            )
            r_squared = r_value**2

            # Check if growth rate exceeds threshold
            growth_rate_mb_per_hour = slope

            if (
                growth_rate_mb_per_hour
                > self.leak_thresholds["linear_growth_mb_per_hour"]
                and r_squared > 0.7
                and p_value < 0.05
            ):
                return {
                    "pattern": LeakPattern.LINEAR.value,
                    "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
                    "r_squared": r_squared,
                    "p_value": p_value,
                    "confidence": min(r_squared, 1 - p_value),
                    "severity": self._classify_leak_severity(growth_rate_mb_per_hour),
                    "projected_24h_growth_mb": growth_rate_mb_per_hour * 24,
                }
        except ImportError:
            # Fallback to simple calculation if scipy is not available
            if len(memory_mb) >= 2:
                total_growth = memory_mb[-1] - memory_mb[0]
                total_time = (
                    time_hours[-1] - time_hours[0]
                    if time_hours[-1] > time_hours[0]
                    else 1
                )
                growth_rate = total_growth / total_time

                if growth_rate > self.leak_thresholds["linear_growth_mb_per_hour"]:
                    return {
                        "pattern": LeakPattern.LINEAR.value,
                        "growth_rate_mb_per_hour": growth_rate,
                        "confidence": 0.6,  # Lower confidence without statistical test
                        "severity": self._classify_leak_severity(growth_rate),
                    }

        return None

    def _detect_exponential_leak(
        self, time_hours: list[float], memory_mb: list[float]
    ) -> dict[str, Any] | None:
        """Detect exponential memory growth pattern."""
        if len(memory_mb) < 4:
            return None

        # Calculate growth factors between consecutive measurements
        growth_factors = []
        for i in range(1, len(memory_mb)):
            if memory_mb[i - 1] > 0:
                factor = memory_mb[i] / memory_mb[i - 1]
                growth_factors.append(factor)

        if not growth_factors:
            return None

        avg_growth_factor = statistics.mean(growth_factors)

        # Check for exponential growth (consistent factor > threshold)
        if (
            avg_growth_factor > self.leak_thresholds["exponential_growth_factor"]
            and len([f for f in growth_factors if f > 1.2]) >= len(growth_factors) * 0.7
        ):
            return {
                "pattern": LeakPattern.EXPONENTIAL.value,
                "average_growth_factor": avg_growth_factor,
                "consistent_growth_periods": len([
                    f for f in growth_factors if f > 1.1
                ]),
                "total_growth_factor": memory_mb[-1] / memory_mb[0]
                if memory_mb[0] > 0
                else 0,
                "confidence": min(0.9, (avg_growth_factor - 1.0) * 2),
                "severity": "HIGH" if avg_growth_factor > 2.0 else "MEDIUM",
            }

        return None

    def _detect_burst_allocation(
        self, snapshots: list[AllocationSnapshot]
    ) -> dict[str, Any] | None:
        """Detect sudden burst allocations."""
        if len(snapshots) < 2:
            return None

        burst_events = []

        for i in range(1, len(snapshots)):
            prev_memory = snapshots[i - 1].current_size / 1024 / 1024
            curr_memory = snapshots[i].current_size / 1024 / 1024

            memory_increase = curr_memory - prev_memory

            if memory_increase > self.leak_thresholds["burst_size_mb"]:
                burst_events.append({
                    "timestamp": snapshots[i].timestamp,
                    "increase_mb": memory_increase,
                    "context": snapshots[i].operation_context,
                })

        if burst_events:
            total_burst_memory = sum(event["increase_mb"] for event in burst_events)

            return {
                "pattern": LeakPattern.BURST.value,
                "burst_count": len(burst_events),
                "total_burst_memory_mb": total_burst_memory,
                "largest_burst_mb": max(event["increase_mb"] for event in burst_events),
                "burst_events": burst_events[-5:],  # Last 5 events
                "confidence": min(0.9, len(burst_events) / 10),
                "severity": self._classify_leak_severity(
                    total_burst_memory / len(snapshots)
                ),
            }

        return None

    def _detect_cyclic_pattern(self, memory_mb: list[float]) -> dict[str, Any] | None:
        """Detect cyclic memory allocation patterns."""
        if len(memory_mb) < 10:
            return None

        # Simple peak/valley detection
        peaks = []
        valleys = []

        for i in range(1, len(memory_mb) - 1):
            if memory_mb[i] > memory_mb[i - 1] and memory_mb[i] > memory_mb[i + 1]:
                peaks.append((i, memory_mb[i]))
            elif memory_mb[i] < memory_mb[i - 1] and memory_mb[i] < memory_mb[i + 1]:
                valleys.append((i, memory_mb[i]))

        if len(peaks) >= 3 and len(valleys) >= 2:
            # Calculate average cycle amplitude
            if peaks and valleys:
                avg_peak = statistics.mean(peak[1] for peak in peaks)
                avg_valley = statistics.mean(valley[1] for valley in valleys)
                cycle_amplitude = avg_peak - avg_valley

                # Check if there's an overall upward trend in baseline
                first_half_avg = statistics.mean(memory_mb[: len(memory_mb) // 2])
                second_half_avg = statistics.mean(memory_mb[len(memory_mb) // 2 :])
                baseline_growth = second_half_avg - first_half_avg

                if (
                    cycle_amplitude > 10 or baseline_growth > 5
                ):  # 10MB cycles or 5MB baseline growth
                    return {
                        "pattern": LeakPattern.CYCLIC.value,
                        "cycle_amplitude_mb": cycle_amplitude,
                        "baseline_growth_mb": baseline_growth,
                        "peak_count": len(peaks),
                        "valley_count": len(valleys),
                        "confidence": min(0.8, (len(peaks) + len(valleys)) / 20),
                        "severity": "MEDIUM" if baseline_growth > 10 else "LOW",
                    }

        return None

    def _classify_leak_severity(self, growth_rate_mb_per_hour: float) -> str:
        """Classify leak severity based on growth rate."""
        if growth_rate_mb_per_hour > 50:
            return "CRITICAL"
        if growth_rate_mb_per_hour > 20:
            return "HIGH"
        if growth_rate_mb_per_hour > 10:
            return "MEDIUM"
        return "LOW"

    def _generate_leak_recommendations(
        self, operation_type: str, detected_patterns: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on detected leak patterns."""
        recommendations = []

        for pattern_info in detected_patterns:
            pattern = pattern_info["pattern"]
            severity = pattern_info.get("severity", "LOW")

            if pattern == LeakPattern.LINEAR.value:
                if severity in ["CRITICAL", "HIGH"]:
                    recommendations.extend([
                        f"URGENT: Linear memory leak detected in {operation_type}",
                        "Review object lifecycle management and ensure proper cleanup",
                        "Check for circular references preventing garbage collection",
                        "Consider implementing object pooling or weak references",
                    ])
                else:
                    recommendations.extend([
                        f"Linear memory growth detected in {operation_type} - monitor trend",
                        "Review recent changes that might affect memory usage",
                        "Consider periodic garbage collection triggers",
                    ])

            elif pattern == LeakPattern.EXPONENTIAL.value:
                recommendations.extend([
                    f"CRITICAL: Exponential memory growth in {operation_type}",
                    "Immediate investigation required - likely recursive object creation",
                    "Check for infinite loops in object instantiation",
                    "Review caching mechanisms for unbounded growth",
                ])

            elif pattern == LeakPattern.BURST.value:
                recommendations.extend([
                    f"Burst allocations detected in {operation_type}",
                    "Review large object creation patterns",
                    "Consider streaming or chunking large data processing",
                    "Implement memory-aware processing limits",
                ])

            elif pattern == LeakPattern.CYCLIC.value:
                recommendations.extend([
                    f"Cyclic memory pattern in {operation_type}",
                    "Review periodic processing that may be accumulating memory",
                    "Ensure proper cleanup after each processing cycle",
                    "Consider memory compaction after intensive operations",
                ])

        if not recommendations:
            recommendations.append(
                f"No specific memory issues detected in {operation_type}"
            )

        return recommendations

    async def _run_gc_stress_test(self, operation_type: str) -> GCStressTestResult:
        """Run garbage collection stress test."""
        # Get pre-GC memory stats
        process = psutil.Process()
        pre_gc_memory = process.memory_info().rss / 1024 / 1024

        # Count objects before GC
        gc_stats_before = gc.get_stats()
        objects_before = len(gc.get_objects())

        # Force multiple GC cycles
        gc_start_time = time.perf_counter()

        collected_objects = 0
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects += collected

        gc_time = time.perf_counter() - gc_start_time

        # Get post-GC memory stats
        post_gc_memory = process.memory_info().rss / 1024 / 1024
        objects_after = len(gc.get_objects())

        memory_freed = pre_gc_memory - post_gc_memory
        gc_efficiency = (memory_freed / pre_gc_memory * 100) if pre_gc_memory > 0 else 0

        # Determine if leak is suspected
        leak_suspected = (
            memory_freed < 1.0  # Less than 1MB freed
            and pre_gc_memory > 100  # More than 100MB baseline
            and gc_efficiency < 5  # Less than 5% efficiency
        )

        recommendations = []
        if leak_suspected:
            recommendations.extend([
                f"Low GC efficiency ({gc_efficiency:.1f}%) suggests memory leaks",
                "Objects may be held by strong references preventing collection",
                "Review object lifecycle and reference management",
            ])
        elif memory_freed > 50:
            recommendations.append(f"Good GC efficiency - freed {memory_freed:.1f}MB")

        return GCStressTestResult(
            operation_name=operation_type,
            pre_gc_memory_mb=pre_gc_memory,
            post_gc_memory_mb=post_gc_memory,
            memory_freed_mb=memory_freed,
            gc_time_seconds=gc_time,
            objects_collected=collected_objects,
            objects_remaining=objects_after,
            gc_efficiency=gc_efficiency,
            leak_suspected=leak_suspected,
            recommendations=recommendations,
        )

    async def _generate_leak_report(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive memory leak detection report."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "total_operations_tested": len([
                    k for k in results if not k.endswith("_gc_test")
                ]),
                "operations_with_leaks": 0,
                "critical_leaks": 0,
                "high_leaks": 0,
                "medium_leaks": 0,
                "low_leaks": 0,
                "total_memory_growth_mb": 0,
                "gc_efficiency_average": 0,
            },
            "detailed_results": results,
            "recommendations": [],
            "leak_registry": [],
        }

        gc_efficiencies = []

        # Analyze results
        for operation_name, result in results.items():
            if operation_name.endswith("_gc_test"):
                gc_efficiencies.append(result.get("gc_efficiency", 0))
                continue

            if isinstance(result, dict) and result.get("leak_analysis", {}).get(
                "leak_suspected"
            ):
                report["summary"]["operations_with_leaks"] += 1

                leak_patterns = result["leak_analysis"].get(
                    "leak_patterns_detected", []
                )
                for pattern_info in leak_patterns:
                    severity = pattern_info.get("severity", "LOW")
                    if severity == "CRITICAL":
                        report["summary"]["critical_leaks"] += 1
                    elif severity == "HIGH":
                        report["summary"]["high_leaks"] += 1
                    elif severity == "MEDIUM":
                        report["summary"]["medium_leaks"] += 1
                    else:
                        report["summary"]["low_leaks"] += 1

            # Track total memory growth
            memory_growth = result.get("memory_growth_mb", 0)
            report["summary"]["total_memory_growth_mb"] += memory_growth

        # Calculate GC efficiency average
        if gc_efficiencies:
            report["summary"]["gc_efficiency_average"] = statistics.mean(
                gc_efficiencies
            )

        # Generate overall recommendations
        if report["summary"]["critical_leaks"] > 0:
            report["recommendations"].extend([
                "CRITICAL: Memory leaks detected - immediate action required",
                "Stop deployment and investigate leak sources",
                "Review object lifecycle management across all affected operations",
            ])
        elif report["summary"]["high_leaks"] > 0:
            report["recommendations"].extend([
                "HIGH PRIORITY: Significant memory leaks detected",
                "Schedule immediate investigation and fixes",
                "Monitor memory usage closely in production",
            ])
        elif report["summary"]["medium_leaks"] > 0:
            report["recommendations"].extend([
                "MEDIUM PRIORITY: Memory growth patterns detected",
                "Plan optimization work to prevent future issues",
                "Increase monitoring frequency",
            ])
        else:
            report["recommendations"].append(
                "No critical memory leaks detected - system appears stable"
            )

        # Save report to disk
        report_file = self.data_dir / f"memory_leak_report_{int(time.time())}.json"
        async with aiofiles.open(report_file, "w") as f:
            await f.write(json.dumps(report, indent=2))

        logger.info(f"Memory leak detection report saved to {report_file}")

        return report


# Factory function
def get_memory_leak_detector(data_dir: Path | None = None) -> MemoryLeakDetector:
    """Get or create memory leak detector instance."""
    return MemoryLeakDetector(data_dir)


# CLI interface
async def run_memory_leak_detection(
    operations: int = 100000,
    operation_types: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run memory leak detection from CLI.

    Args:
        operations: Number of operations per type
        operation_types: List of operation types to test
        output_dir: Output directory for results

    Returns:
        Detection results
    """
    detector = get_memory_leak_detector(Path(output_dir) if output_dir else None)

    return await detector.run_comprehensive_leak_detection(
        operations=operations, operation_types=operation_types
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Leak Detection Framework")
    parser.add_argument(
        "--operations", type=int, default=100000, help="Operations per test type"
    )
    parser.add_argument(
        "--output-dir", type=str, default="memory_analysis", help="Output directory"
    )
    parser.add_argument(
        "--operation-types", nargs="*", help="Specific operation types to test"
    )

    args = parser.parse_args()

    async def main():
        results = await run_memory_leak_detection(
            operations=args.operations,
            operation_types=args.operation_types,
            output_dir=args.output_dir,
        )

        print(
            json.dumps(
                {
                    "summary": results["summary"],
                    "recommendations": results["recommendations"],
                },
                indent=2,
            )
        )

        return results

    asyncio.run(main())
