"""SLI Calculator Components for SLO/SLA Monitoring
===============================================

Implements Service Level Indicator (SLI) calculations with multi-window analysis,
percentile computations, and availability tracking following Google SRE practices.
"""

import asyncio
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from prompt_improver.database import (
    ManagerMode,
    create_security_context,
    get_database_services,
)
from prompt_improver.monitoring.slo.framework import SLOTarget, SLOTimeWindow, SLOType

logger = logging.getLogger(__name__)


@dataclass
class SLIMeasurement:
    """Individual SLI measurement point"""

    timestamp: float
    value: float
    success: bool = True
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SLIResult:
    """Result of SLI calculation for a time window"""

    slo_target: SLOTarget
    time_window: SLOTimeWindow
    current_value: float
    target_value: float
    compliance_ratio: float
    measurement_count: int
    window_start: datetime
    window_end: datetime
    min_value: float | None = None
    max_value: float | None = None
    median_value: float | None = None
    std_deviation: float | None = None

    @property
    def is_compliant(self) -> bool:
        """Check if SLI meets the SLO target"""
        if self.slo_target.slo_type in [
            SLOType.AVAILABILITY,
            SLOType.THROUGHPUT,
            SLOType.QUALITY,
        ]:
            return self.current_value >= self.target_value
        return self.current_value <= self.target_value

    @property
    def compliance_percentage(self) -> float:
        """Get compliance as percentage (0-100)"""
        if self.slo_target.slo_type in [
            SLOType.AVAILABILITY,
            SLOType.THROUGHPUT,
            SLOType.QUALITY,
        ]:
            return min(100.0, self.current_value / self.target_value * 100.0)
        return max(0.0, 100.0 - self.current_value / self.target_value * 100.0)


class SLICalculator:
    """Base SLI calculator for single time window calculations"""

    def __init__(
        self, slo_target: SLOTarget, max_measurements: int = 10000, unified_manager=None
    ):
        self.slo_target = slo_target
        self.max_measurements = max_measurements
        self._unified_manager = unified_manager or await get_database_services(
            ManagerMode.ASYNC_MODERN
        )
        self._security_context = None
        self._measurements = deque(maxlen=max_measurements)
        self._result_cache: dict[str, tuple[SLIResult, float]] = {}
        self._cache_ttl = 60

    async def _ensure_security_context(self):
        """Ensure security context exists for Redis operations"""
        if self._security_context is None:
            self._security_context = await create_security_context(
                agent_id=f"slo_calculator_{self.slo_target.service_name}_{self.slo_target.name}",
                tier="professional",
                authenticated=True,
            )
        return self._security_context

    async def add_measurement(
        self,
        value: float,
        timestamp: float | None = None,
        success: bool = True,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new SLI measurement"""
        measurement = SLIMeasurement(
            timestamp=timestamp or time.time(),
            value=value,
            success=success,
            labels=labels or {},
            metadata=metadata or {},
        )
        self._measurements.append(measurement)
        self._result_cache.clear()
        import uuid

        from prompt_improver.performance.monitoring.health.background_manager import (
            TaskPriority,
            get_background_task_manager,
        )

        task_manager = get_background_task_manager()
        await task_manager.submit_enhanced_task(
            task_id=f"slo_unified_store_{self.slo_target.name}_{str(uuid.uuid4())[:8]}",
            coroutine=self._store_measurement_unified(measurement),
            priority=TaskPriority.NORMAL,
            tags={
                "service": "slo",
                "type": "unified_cache_storage",
                "component": "slo_calculator",
                "slo_name": self.slo_target.name,
            },
        )

    async def _store_measurement_unified(self, measurement: SLIMeasurement) -> None:
        """Store measurement in unified cache system for distributed access"""
        try:
            if not self.True:
                await self._unified_manager.initialize()
            security_context = await self._ensure_security_context()
            key = f"sli:{self.slo_target.service_name}:{self.slo_target.name}:{measurement.timestamp}"
            data = {
                "timestamp": measurement.timestamp,
                "value": measurement.value,
                "success": int(measurement.success),
                "labels": measurement.labels,
                "metadata": measurement.metadata,
            }
            max_window_seconds = max(window.seconds for window in SLOTimeWindow)
            ttl_seconds = max_window_seconds * 2
            success = await self._unified_manager.set_cached(
                key=key,
                value=data,
                ttl_seconds=ttl_seconds,
                security_context=security_context,
            )
            if success:
                logger.debug(f"Stored SLI measurement for {self.slo_target.name} in unified cache")
            else:
                logger.warning(f"Failed to store SLI measurement for {self.slo_target.name} in unified cache")
        except Exception as e:
            logger.warning(f"Failed to store measurement in unified cache: {e}")

    async def calculate_sli(
        self, time_window: SLOTimeWindow | None = None, end_time: datetime | None = None
    ) -> SLIResult:
        """Calculate SLI for specified time window"""
        time_window = time_window or self.slo_target.time_window
        end_time = end_time or datetime.now(UTC)
        start_time = end_time - timedelta(seconds=time_window.seconds)
        cache_key = f"sli_result:{self.slo_target.service_name}:{self.slo_target.name}:{time_window.value}:{end_time.timestamp():.0f}"
        try:
            if not self.True:
                await self._unified_manager.initialize()
            security_context = await self._ensure_security_context()
            cached_result = await self._unified_manager.get_cached(
                cache_key, security_context
            )
            if cached_result is not None:
                logger.debug(f"SLI result cache hit for {self.slo_target.name}")
                return SLIResult(**cached_result)
        except Exception as e:
            logger.warning(f"Failed to check unified cache for SLI result: {e}")
        if cache_key in self._result_cache:
            result, cache_time = self._result_cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                return result
        window_measurements = [
            m
            for m in self._measurements
            if start_time.timestamp() <= m.timestamp <= end_time.timestamp()
        ]
        if not window_measurements:
            result = SLIResult(
                slo_target=self.slo_target,
                time_window=time_window,
                current_value=0.0,
                target_value=self.slo_target.target_value,
                compliance_ratio=0.0,
                measurement_count=0,
                window_start=start_time,
                window_end=end_time,
            )
        else:
            current_value = self._calculate_value_by_type(window_measurements)
            compliance_ratio = self._calculate_compliance_ratio(current_value)
            values = [m.value for m in window_measurements]
            result = SLIResult(
                slo_target=self.slo_target,
                time_window=time_window,
                current_value=current_value,
                target_value=self.slo_target.target_value,
                compliance_ratio=compliance_ratio,
                measurement_count=len(window_measurements),
                window_start=start_time,
                window_end=end_time,
                min_value=min(values),
                max_value=max(values),
                median_value=statistics.median(values),
                std_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
            )
        self._result_cache[cache_key] = (result, time.time())
        try:
            security_context = await self._ensure_security_context()
            result_data = {
                "slo_target": {
                    "name": self.slo_target.name,
                    "service_name": self.slo_target.service_name,
                    "slo_type": self.slo_target.slo_type.value,
                    "target_value": self.slo_target.target_value,
                    "unit": self.slo_target.unit,
                },
                "time_window": time_window.value,
                "current_value": result.current_value,
                "target_value": result.target_value,
                "compliance_ratio": result.compliance_ratio,
                "measurement_count": result.measurement_count,
                "window_start": result.window_start.isoformat(),
                "window_end": result.window_end.isoformat(),
                "min_value": result.min_value,
                "max_value": result.max_value,
                "median_value": result.median_value,
                "std_deviation": result.std_deviation,
            }
            await self._unified_manager.set_cached(
                key=cache_key,
                value=result_data,
                ttl_seconds=self._cache_ttl,
                security_context=security_context,
            )
        except Exception as e:
            logger.warning(f"Failed to cache SLI result in unified cache: {e}")
        return result

    def _calculate_value_by_type(self, measurements: list[SLIMeasurement]) -> float:
        """Calculate current value based on SLO type"""
        values = [m.value for m in measurements]
        if self.slo_target.slo_type == SLOType.AVAILABILITY:
            successful = sum(1 for m in measurements if m.success)
            return successful / len(measurements) * 100.0
        if self.slo_target.slo_type == SLOType.LATENCY:
            if "p99" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 99)
            if "p95" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 95)
            if "p90" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 90)
            if "p50" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 50)
            return statistics.mean(values)
        if self.slo_target.slo_type == SLOType.ERROR_RATE:
            failed = sum(1 for m in measurements if not m.success)
            return failed / len(measurements) * 100.0
        if self.slo_target.slo_type == SLOType.THROUGHPUT:
            time_span = max(m.timestamp for m in measurements) - min(
                m.timestamp for m in measurements
            )
            if time_span > 0:
                return len(measurements) / time_span
            return len(measurements)
        if self.slo_target.slo_type == SLOType.QUALITY:
            return statistics.mean(values)
        return statistics.mean(values)

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = percentile / 100.0 * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        lower_index = int(math.floor(index))
        upper_index = int(math.ceil(index))
        weight = index - lower_index
        return (
            sorted_values[lower_index] * (1 - weight)
            + sorted_values[upper_index] * weight
        )

    def _calculate_compliance_ratio(self, current_value: float) -> float:
        """Calculate compliance ratio based on SLO type"""
        target = self.slo_target.target_value
        if target == 0:
            return 1.0 if current_value == 0 else 0.0
        if self.slo_target.slo_type in [
            SLOType.AVAILABILITY,
            SLOType.THROUGHPUT,
            SLOType.QUALITY,
        ]:
            return min(1.0, current_value / target)
        return max(0.0, 1.0 - current_value / target)

    def get_measurement_count(self) -> int:
        """Get total number of measurements stored"""
        return len(self._measurements)

    def clear_measurements(self) -> None:
        """Clear all stored measurements"""
        self._measurements.clear()
        self._result_cache.clear()


class MultiWindowSLICalculator:
    """Calculator for multiple time windows with trend analysis"""

    def __init__(
        self,
        slo_target: SLOTarget,
        windows: list[SLOTimeWindow] | None = None,
        max_measurements: int = 50000,
        redis_url: str | None = None,
        unified_manager=None,
    ):
        self.slo_target = slo_target
        self.windows = windows or [
            SLOTimeWindow.HOUR_1,
            SLOTimeWindow.DAY_1,
            SLOTimeWindow.WEEK_1,
            SLOTimeWindow.MONTH_1,
        ]
        self._unified_manager = unified_manager or await get_database_services(
            ManagerMode.ASYNC_MODERN
        )
        self.calculators = {
            window: SLICalculator(
                slo_target=slo_target,
                max_measurements=max_measurements,
                redis_url=redis_url,
                unified_manager=self._unified_manager,
            )
            for window in self.windows
        }
        self.trend_window = SLOTimeWindow.HOUR_1
        self.trend_points = 12

    def add_measurement(
        self,
        value: float,
        timestamp: float | None = None,
        success: bool = True,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add measurement to all window calculators"""
        for calculator in self.calculators.values():
            calculator.add_measurement(value, timestamp, success, labels, metadata)

    def calculate_all_windows(
        self, end_time: datetime | None = None
    ) -> dict[SLOTimeWindow, SLIResult]:
        """Calculate SLI for all configured time windows"""
        results = {}
        for window, calculator in self.calculators.items():
            try:
                results[window] = calculator.calculate_sli(window, end_time)
            except Exception as e:
                logger.error(f"Failed to calculate SLI for window {window}: {e}")
                results[window] = SLIResult(
                    slo_target=self.slo_target,
                    time_window=window,
                    current_value=0.0,
                    target_value=self.slo_target.target_value,
                    compliance_ratio=0.0,
                    measurement_count=0,
                    window_start=end_time or datetime.now(UTC),
                    window_end=end_time or datetime.now(UTC),
                )
        return results

    def analyze_trends(self, end_time: datetime | None = None) -> dict[str, Any]:
        """Analyze SLI trends over time"""
        end_time = end_time or datetime.now(UTC)
        trend_results = []
        interval_seconds = self.trend_window.seconds // self.trend_points
        for i in range(self.trend_points):
            point_end_time = end_time - timedelta(seconds=i * interval_seconds)
            try:
                result = self.calculators[self.trend_window].calculate_sli(
                    SLOTimeWindow.HOUR_1, point_end_time
                )
                trend_results.append({
                    "timestamp": point_end_time.timestamp(),
                    "value": result.current_value,
                    "compliance_ratio": result.compliance_ratio,
                })
            except Exception as e:
                logger.warning(f"Failed to calculate trend point {i}: {e}")
        if len(trend_results) < 2:
            return {
                "trend_direction": "unknown",
                "trend_slope": 0.0,
                "trend_confidence": 0.0,
                "data_points": trend_results,
            }
        values = [p["value"] for p in trend_results]
        timestamps = [p["timestamp"] for p in trend_results]
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum((x * y for x, y in zip(timestamps, values, strict=False)))
        sum_x2 = sum(x * x for x in timestamps)
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0.0
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = (
                "improving"
                if self.slo_target.slo_type
                in [SLOType.AVAILABILITY, SLOType.THROUGHPUT]
                else "degrading"
            )
        else:
            direction = (
                "degrading"
                if self.slo_target.slo_type
                in [SLOType.AVAILABILITY, SLOType.THROUGHPUT]
                else "improving"
            )
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        if ss_tot > 0:
            y_pred = [slope * x + (sum_y - slope * sum_x) / n for x in timestamps]
            ss_res = sum(((y - y_pred[i]) ** 2 for i, y in enumerate(values)))
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = 1.0
        return {
            "trend_direction": direction,
            "trend_slope": slope,
            "trend_confidence": max(0.0, min(1.0, r_squared)),
            "data_points": trend_results,
        }

    def get_summary(self, end_time: datetime | None = None) -> dict[str, Any]:
        """Get comprehensive SLI summary across all windows"""
        window_results = self.calculate_all_windows(end_time)
        trend_analysis = self.analyze_trends(end_time)
        worst_window = None
        worst_compliance = 1.0
        for window, result in window_results.items():
            if result.compliance_ratio < worst_compliance:
                worst_compliance = result.compliance_ratio
                worst_window = window
        compliances = [
            r.compliance_ratio
            for r in window_results.values()
            if r.measurement_count > 0
        ]
        health_score = statistics.mean(compliances) if compliances else 0.0
        return {
            "slo_target": self.slo_target.name,
            "service_name": self.slo_target.service_name,
            "slo_type": self.slo_target.slo_type.value,
            "overall_health_score": health_score,
            "worst_performing_window": worst_window.value if worst_window else None,
            "worst_compliance_ratio": worst_compliance,
            "window_results": {
                window.value: {
                    "current_value": result.current_value,
                    "target_value": result.target_value,
                    "compliance_ratio": result.compliance_ratio,
                    "is_compliant": result.is_compliant,
                    "measurement_count": result.measurement_count,
                }
                for window, result in window_results.items()
            },
            "trend_analysis": trend_analysis,
            "calculated_at": (end_time or datetime.now(UTC)).isoformat(),
        }


class PercentileCalculator:
    """Specialized calculator for latency percentiles"""

    def __init__(
        self, percentiles: list[int] = [50, 90, 95, 99], max_samples: int = 10000
    ):
        self.percentiles = percentiles
        self.max_samples = max_samples
        self._samples = deque(maxlen=max_samples)

    def add_sample(self, value: float, timestamp: float | None = None) -> None:
        """Add a latency sample"""
        self._samples.append({"value": value, "timestamp": timestamp or time.time()})

    def calculate_percentiles(self, window_seconds: int = 3600) -> dict[int, float]:
        """Calculate percentiles for specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        recent_samples = [
            s["value"] for s in self._samples if s["timestamp"] >= cutoff_time
        ]
        if not recent_samples:
            return dict.fromkeys(self.percentiles, 0.0)
        sorted_samples = sorted(recent_samples)
        results = {}
        for percentile in self.percentiles:
            index = percentile / 100.0 * (len(sorted_samples) - 1)
            if index.is_integer():
                results[percentile] = sorted_samples[int(index)]
            else:
                lower_index = int(math.floor(index))
                upper_index = int(math.ceil(index))
                weight = index - lower_index
                results[percentile] = (
                    sorted_samples[lower_index] * (1 - weight)
                    + sorted_samples[upper_index] * weight
                )
        return results

    def get_statistics(self, window_seconds: int = 3600) -> dict[str, float]:
        """Get comprehensive latency statistics"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        recent_samples = [
            s["value"] for s in self._samples if s["timestamp"] >= cutoff_time
        ]
        if not recent_samples:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
            }
        percentiles = self.calculate_percentiles(window_seconds)
        return {
            "count": len(recent_samples),
            "min": min(recent_samples),
            "max": max(recent_samples),
            "mean": statistics.mean(recent_samples),
            "median": statistics.median(recent_samples),
            "std_dev": statistics.stdev(recent_samples)
            if len(recent_samples) > 1
            else 0.0,
            **{f"p{p}": v for p, v in percentiles.items()},
        }


class AvailabilityCalculator:
    """Specialized calculator for availability metrics"""

    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events = deque(maxlen=max_events)

    def record_event(
        self,
        success: bool,
        timestamp: float | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record an availability event"""
        self._events.append({
            "success": success,
            "timestamp": timestamp or time.time(),
            "labels": labels or {},
        })

    def calculate_availability(self, window_seconds: int = 3600) -> dict[str, Any]:
        """Calculate availability metrics for time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        recent_events = [e for e in self._events if e["timestamp"] >= cutoff_time]
        if not recent_events:
            return {
                "availability_percentage": 100.0,
                "total_events": 0,
                "successful_events": 0,
                "failed_events": 0,
                "mttr_seconds": 0.0,
                "mtbf_seconds": 0.0,
            }
        total_events = len(recent_events)
        successful_events = sum(1 for e in recent_events if e["success"])
        failed_events = total_events - successful_events
        availability_percentage = successful_events / total_events * 100.0
        mttr = self._calculate_mttr(recent_events)
        mtbf = self._calculate_mtbf(recent_events)
        return {
            "availability_percentage": availability_percentage,
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "mttr_seconds": mttr,
            "mtbf_seconds": mtbf,
        }

    def _calculate_mttr(self, events: list[dict[str, Any]]) -> float:
        """Calculate Mean Time To Recovery"""
        recovery_times = []
        failure_start = None
        for event in sorted(events, key=lambda x: x["timestamp"]):
            if not event["success"] and failure_start is None:
                failure_start = event["timestamp"]
            elif event["success"] and failure_start is not None:
                recovery_time = event["timestamp"] - failure_start
                recovery_times.append(recovery_time)
                failure_start = None
        return statistics.mean(recovery_times) if recovery_times else 0.0

    def _calculate_mtbf(self, events: list[dict[str, Any]]) -> float:
        """Calculate Mean Time Between Failures"""
        failure_timestamps = [e["timestamp"] for e in events if not e["success"]]
        if len(failure_timestamps) < 2:
            return 0.0
        failure_timestamps.sort()
        intervals = [
            failure_timestamps[i + 1] - failure_timestamps[i]
            for i in range(len(failure_timestamps) - 1)
        ]
        return statistics.mean(intervals)
