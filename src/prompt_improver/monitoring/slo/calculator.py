"""
SLI Calculator Components for SLO/SLA Monitoring
===============================================

Implements Service Level Indicator (SLI) calculations with multi-window analysis,
percentile computations, and availability tracking following Google SRE practices.
"""

import time
import asyncio
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from collections import deque
import logging
import math

try:
    import coredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None

from .framework import SLOTarget, SLOTimeWindow, SLOType

logger = logging.getLogger(__name__)

@dataclass
class SLIMeasurement:
    """Individual SLI measurement point"""
    timestamp: float
    value: float
    success: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class SLIResult:
    """Result of SLI calculation for a time window"""
    slo_target: SLOTarget
    time_window: SLOTimeWindow
    current_value: float
    target_value: float
    compliance_ratio: float  # current/target or 1 - (current/target) for error rates
    measurement_count: int
    window_start: datetime
    window_end: datetime
    
    # Additional statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median_value: Optional[float] = None
    std_deviation: Optional[float] = None
    
    @property
    def is_compliant(self) -> bool:
        """Check if SLI meets the SLO target"""
        if self.slo_target.slo_type in [SLOType.AVAILABILITY, SLOType.THROUGHPUT, SLOType.QUALITY]:
            # Higher is better
            return self.current_value >= self.target_value
        else:
            # Lower is better (latency, error rate)
            return self.current_value <= self.target_value
    
    @property
    def compliance_percentage(self) -> float:
        """Get compliance as percentage (0-100)"""
        if self.slo_target.slo_type in [SLOType.AVAILABILITY, SLOType.THROUGHPUT, SLOType.QUALITY]:
            return min(100.0, (self.current_value / self.target_value) * 100.0)
        else:
            return max(0.0, 100.0 - ((self.current_value / self.target_value) * 100.0))

class SLICalculator:
    """Base SLI calculator for single time window calculations"""
    
    def __init__(
        self, 
        slo_target: SLOTarget,
        max_measurements: int = 10000,
        redis_url: Optional[str] = None
    ):
        self.slo_target = slo_target
        self.max_measurements = max_measurements
        self.redis_url = redis_url
        self._redis_client = None
        
        # In-memory storage for measurements
        self._measurements = deque(maxlen=max_measurements)
        
        # Cache for calculated results
        self._result_cache: Dict[str, Tuple[SLIResult, float]] = {}
        self._cache_ttl = 60  # Cache for 60 seconds
    
    async def get_redis_client(self) -> Optional[coredis.Redis]:
        """Get Redis client for distributed storage"""
        if not REDIS_AVAILABLE or not self.redis_url:
            return None
            
        if self._redis_client is None:
            try:
                self._redis_client = coredis.Redis.from_url(self.redis_url, decode_responses=True)
                await self._redis_client.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None
        
        return self._redis_client
    
    def add_measurement(
        self, 
        value: float, 
        timestamp: Optional[float] = None,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new SLI measurement"""
        measurement = SLIMeasurement(
            timestamp=timestamp or time.time(),
            value=value,
            success=success,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        self._measurements.append(measurement)
        
        # Clear cache as new measurement invalidates results
        self._result_cache.clear()
        
        # Store in Redis if available
        asyncio.create_task(self._store_measurement_redis(measurement))
    
    async def _store_measurement_redis(self, measurement: SLIMeasurement) -> None:
        """Store measurement in Redis for distributed access"""
        redis = await self.get_redis_client()
        if not redis:
            return
            
        try:
            key = f"sli:{self.slo_target.service_name}:{self.slo_target.name}"
            data = {
                "timestamp": measurement.timestamp,
                "value": measurement.value,
                "success": int(measurement.success),
                "labels": str(measurement.labels),
                "metadata": str(measurement.metadata)
            }
            
            # Store as hash with timestamp as field
            await redis.hset(key, str(measurement.timestamp), str(data))
            
            # Set expiration based on longest time window needed
            max_window_seconds = max(window.seconds for window in SLOTimeWindow)
            await redis.expire(key, max_window_seconds * 2)  # Keep extra for safety
            
        except Exception as e:
            logger.warning(f"Failed to store measurement in Redis: {e}")
    
    def calculate_sli(
        self, 
        time_window: Optional[SLOTimeWindow] = None,
        end_time: Optional[datetime] = None
    ) -> SLIResult:
        """Calculate SLI for specified time window"""
        time_window = time_window or self.slo_target.time_window
        end_time = end_time or datetime.now(UTC)
        start_time = end_time - timedelta(seconds=time_window.seconds)
        
        # Check cache first
        cache_key = f"{time_window.value}:{end_time.timestamp():.0f}"
        if cache_key in self._result_cache:
            result, cache_time = self._result_cache[cache_key]
            if time.time() - cache_time < self._cache_ttl:
                return result
        
        # Filter measurements to time window
        window_measurements = [
            m for m in self._measurements 
            if start_time.timestamp() <= m.timestamp <= end_time.timestamp()
        ]
        
        if not window_measurements:
            # No measurements in window
            result = SLIResult(
                slo_target=self.slo_target,
                time_window=time_window,
                current_value=0.0,
                target_value=self.slo_target.target_value,
                compliance_ratio=0.0,
                measurement_count=0,
                window_start=start_time,
                window_end=end_time
            )
        else:
            # Calculate SLI based on type
            current_value = self._calculate_value_by_type(window_measurements)
            
            # Calculate compliance ratio
            compliance_ratio = self._calculate_compliance_ratio(current_value)
            
            # Calculate statistics
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
                std_deviation=statistics.stdev(values) if len(values) > 1 else 0.0
            )
        
        # Cache result
        self._result_cache[cache_key] = (result, time.time())
        
        return result
    
    def _calculate_value_by_type(self, measurements: List[SLIMeasurement]) -> float:
        """Calculate current value based on SLO type"""
        values = [m.value for m in measurements]
        
        if self.slo_target.slo_type == SLOType.AVAILABILITY:
            # Availability: percentage of successful measurements
            successful = sum(1 for m in measurements if m.success)
            return (successful / len(measurements)) * 100.0
            
        elif self.slo_target.slo_type == SLOType.LATENCY:
            # Latency: Use percentile if specified in name, otherwise mean
            if "p99" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 99)
            elif "p95" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 95)
            elif "p90" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 90)
            elif "p50" in self.slo_target.name.lower():
                return self._calculate_percentile(values, 50)
            else:
                return statistics.mean(values)
                
        elif self.slo_target.slo_type == SLOType.ERROR_RATE:
            # Error rate: percentage of failed measurements
            failed = sum(1 for m in measurements if not m.success)
            return (failed / len(measurements)) * 100.0
            
        elif self.slo_target.slo_type == SLOType.THROUGHPUT:
            # Throughput: measurements per unit time
            time_span = max(m.timestamp for m in measurements) - min(m.timestamp for m in measurements)
            if time_span > 0:
                return len(measurements) / time_span
            else:
                return len(measurements)  # Single point in time
                
        elif self.slo_target.slo_type == SLOType.QUALITY:
            # Quality: average of measurement values
            return statistics.mean(values)
            
        else:
            # Default: mean
            return statistics.mean(values)
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(math.floor(index))
            upper_index = int(math.ceil(index))
            weight = index - lower_index
            
            return (sorted_values[lower_index] * (1 - weight) + 
                   sorted_values[upper_index] * weight)
    
    def _calculate_compliance_ratio(self, current_value: float) -> float:
        """Calculate compliance ratio based on SLO type"""
        target = self.slo_target.target_value
        
        if target == 0:
            return 1.0 if current_value == 0 else 0.0
            
        if self.slo_target.slo_type in [SLOType.AVAILABILITY, SLOType.THROUGHPUT, SLOType.QUALITY]:
            # Higher is better
            return min(1.0, current_value / target)
        else:
            # Lower is better (latency, error rate)
            return max(0.0, 1.0 - (current_value / target))
    
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
        windows: Optional[List[SLOTimeWindow]] = None,
        max_measurements: int = 50000,
        redis_url: Optional[str] = None
    ):
        self.slo_target = slo_target
        self.windows = windows or [
            SLOTimeWindow.HOUR_1,
            SLOTimeWindow.DAY_1, 
            SLOTimeWindow.WEEK_1,
            SLOTimeWindow.MONTH_1
        ]
        
        # Create individual calculators for each window
        self.calculators = {
            window: SLICalculator(
                slo_target=slo_target,
                max_measurements=max_measurements,
                redis_url=redis_url
            )
            for window in self.windows
        }
        
        # Trend analysis configuration
        self.trend_window = SLOTimeWindow.HOUR_1
        self.trend_points = 12  # Number of points for trend analysis
    
    def add_measurement(
        self,
        value: float,
        timestamp: Optional[float] = None,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add measurement to all window calculators"""
        for calculator in self.calculators.values():
            calculator.add_measurement(value, timestamp, success, labels, metadata)
    
    def calculate_all_windows(
        self, 
        end_time: Optional[datetime] = None
    ) -> Dict[SLOTimeWindow, SLIResult]:
        """Calculate SLI for all configured time windows"""
        results = {}
        
        for window, calculator in self.calculators.items():
            try:
                results[window] = calculator.calculate_sli(window, end_time)
            except Exception as e:
                logger.error(f"Failed to calculate SLI for window {window}: {e}")
                # Create empty result on error
                results[window] = SLIResult(
                    slo_target=self.slo_target,
                    time_window=window,
                    current_value=0.0,
                    target_value=self.slo_target.target_value,
                    compliance_ratio=0.0,
                    measurement_count=0,
                    window_start=end_time or datetime.now(UTC),
                    window_end=end_time or datetime.now(UTC)
                )
        
        return results
    
    def analyze_trends(
        self, 
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze SLI trends over time"""
        end_time = end_time or datetime.now(UTC)
        trend_results = []
        
        # Calculate SLI at regular intervals over trend window
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
                    "compliance_ratio": result.compliance_ratio
                })
            except Exception as e:
                logger.warning(f"Failed to calculate trend point {i}: {e}")
        
        if len(trend_results) < 2:
            return {
                "trend_direction": "unknown",
                "trend_slope": 0.0,
                "trend_confidence": 0.0,
                "data_points": trend_results
            }
        
        # Calculate trend slope using linear regression
        values = [p["value"] for p in trend_results]
        timestamps = [p["timestamp"] for p in trend_results]
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0.0
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Small threshold for "stable"
            direction = "stable"
        elif slope > 0:
            direction = "improving" if self.slo_target.slo_type in [SLOType.AVAILABILITY, SLOType.THROUGHPUT] else "degrading"
        else:
            direction = "degrading" if self.slo_target.slo_type in [SLOType.AVAILABILITY, SLOType.THROUGHPUT] else "improving"
        
        # Calculate R-squared for confidence
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        if ss_tot > 0:
            y_pred = [slope * x + (sum_y - slope * sum_x) / n for x in timestamps]
            ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(values))
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 1.0
        
        return {
            "trend_direction": direction,
            "trend_slope": slope,
            "trend_confidence": max(0.0, min(1.0, r_squared)),
            "data_points": trend_results
        }
    
    def get_summary(self, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive SLI summary across all windows"""
        window_results = self.calculate_all_windows(end_time)
        trend_analysis = self.analyze_trends(end_time)
        
        # Find worst performing window
        worst_window = None
        worst_compliance = 1.0
        
        for window, result in window_results.items():
            if result.compliance_ratio < worst_compliance:
                worst_compliance = result.compliance_ratio
                worst_window = window
        
        # Calculate overall health score
        compliances = [r.compliance_ratio for r in window_results.values() if r.measurement_count > 0]
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
                    "measurement_count": result.measurement_count
                }
                for window, result in window_results.items()
            },
            "trend_analysis": trend_analysis,
            "calculated_at": (end_time or datetime.now(UTC)).isoformat()
        }

class PercentileCalculator:
    """Specialized calculator for latency percentiles"""
    
    def __init__(
        self,
        percentiles: List[int] = [50, 90, 95, 99],
        max_samples: int = 10000
    ):
        self.percentiles = percentiles
        self.max_samples = max_samples
        self._samples = deque(maxlen=max_samples)
    
    def add_sample(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a latency sample"""
        self._samples.append({
            "value": value,
            "timestamp": timestamp or time.time()
        })
    
    def calculate_percentiles(
        self, 
        window_seconds: int = 3600
    ) -> Dict[int, float]:
        """Calculate percentiles for specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter samples to time window
        recent_samples = [
            s["value"] for s in self._samples 
            if s["timestamp"] >= cutoff_time
        ]
        
        if not recent_samples:
            return {p: 0.0 for p in self.percentiles}
        
        # Calculate percentiles
        sorted_samples = sorted(recent_samples)
        results = {}
        
        for percentile in self.percentiles:
            index = (percentile / 100.0) * (len(sorted_samples) - 1)
            
            if index.is_integer():
                results[percentile] = sorted_samples[int(index)]
            else:
                lower_index = int(math.floor(index))
                upper_index = int(math.ceil(index))
                weight = index - lower_index
                
                results[percentile] = (
                    sorted_samples[lower_index] * (1 - weight) + 
                    sorted_samples[upper_index] * weight
                )
        
        return results
    
    def get_statistics(self, window_seconds: int = 3600) -> Dict[str, float]:
        """Get comprehensive latency statistics"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_samples = [
            s["value"] for s in self._samples 
            if s["timestamp"] >= cutoff_time
        ]
        
        if not recent_samples:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0
            }
        
        percentiles = self.calculate_percentiles(window_seconds)
        
        return {
            "count": len(recent_samples),
            "min": min(recent_samples),
            "max": max(recent_samples),
            "mean": statistics.mean(recent_samples),
            "median": statistics.median(recent_samples),
            "std_dev": statistics.stdev(recent_samples) if len(recent_samples) > 1 else 0.0,
            **{f"p{p}": v for p, v in percentiles.items()}
        }

class AvailabilityCalculator:
    """Specialized calculator for availability metrics"""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events = deque(maxlen=max_events)
    
    def record_event(
        self, 
        success: bool, 
        timestamp: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record an availability event"""
        self._events.append({
            "success": success,
            "timestamp": timestamp or time.time(),
            "labels": labels or {}
        })
    
    def calculate_availability(
        self, 
        window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Calculate availability metrics for time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter events to time window
        recent_events = [
            e for e in self._events 
            if e["timestamp"] >= cutoff_time
        ]
        
        if not recent_events:
            return {
                "availability_percentage": 100.0,
                "total_events": 0,
                "successful_events": 0,
                "failed_events": 0,
                "mttr_seconds": 0.0,
                "mtbf_seconds": 0.0
            }
        
        # Calculate basic metrics
        total_events = len(recent_events)
        successful_events = sum(1 for e in recent_events if e["success"])
        failed_events = total_events - successful_events
        availability_percentage = (successful_events / total_events) * 100.0
        
        # Calculate MTTR (Mean Time To Recovery) and MTBF (Mean Time Between Failures)
        mttr = self._calculate_mttr(recent_events)
        mtbf = self._calculate_mtbf(recent_events)
        
        return {
            "availability_percentage": availability_percentage,
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "mttr_seconds": mttr,
            "mtbf_seconds": mtbf
        }
    
    def _calculate_mttr(self, events: List[Dict[str, Any]]) -> float:
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
    
    def _calculate_mtbf(self, events: List[Dict[str, Any]]) -> float:
        """Calculate Mean Time Between Failures"""
        failure_timestamps = [
            e["timestamp"] for e in events if not e["success"]
        ]
        
        if len(failure_timestamps) < 2:
            return 0.0
        
        failure_timestamps.sort()
        intervals = [
            failure_timestamps[i+1] - failure_timestamps[i]
            for i in range(len(failure_timestamps) - 1)
        ]
        
        return statistics.mean(intervals)