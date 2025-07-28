"""
Model Performance Tracker - 2025 Best Practices

Comprehensive performance tracking for ML models with real-time monitoring,
historical analysis, and performance degradation detection.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot for a model"""
    model_id: str
    timestamp: datetime
    
    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    
    # Throughput metrics
    requests_per_second: float
    concurrent_requests: int
    
    # Quality metrics
    success_rate: float
    error_rate: float
    timeout_rate: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    
    # Sample size
    sample_count: int = 0


@dataclass
class ModelPerformanceHistory:
    """Historical performance data for a model"""
    model_id: str
    snapshots: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_snapshot(self, snapshot: PerformanceSnapshot):
        """Add a new performance snapshot"""
        self.snapshots.append(snapshot)
    
    def get_recent_snapshots(self, hours: int = 24) -> List[PerformanceSnapshot]:
        """Get snapshots from the last N hours"""
        cutoff_time = aware_utc_now() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self.snapshots
            if snapshot.timestamp > cutoff_time
        ]


class ModelPerformanceTracker:
    """
    Advanced performance tracking for ML models.
    
    Provides real-time performance monitoring, historical analysis,
    and performance degradation detection.
    """
    
    def __init__(self, snapshot_interval_seconds: int = 60):
        self.snapshot_interval_seconds = snapshot_interval_seconds
        
        # Performance history by model
        self._performance_history: Dict[str, ModelPerformanceHistory] = defaultdict(
            lambda: ModelPerformanceHistory("")
        )
        
        # Real-time tracking
        self._active_requests: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._request_counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "error": 0, "timeout": 0}
        )
        
        # Performance baselines
        self._performance_baselines: Dict[str, Dict[str, float]] = {}
        
        logger.info("Model Performance Tracker initialized")
    
    async def start_request_tracking(
        self, 
        model_id: str, 
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start tracking a new inference request"""
        try:
            self._active_requests[model_id][request_id] = {
                "start_time": time.time(),
                "metadata": metadata or {},
                "status": "active"
            }
            
            self._request_counters[model_id]["total"] += 1
            
        except Exception as e:
            logger.error(f"Failed to start request tracking: {e}")
    
    async def end_request_tracking(
        self,
        model_id: str,
        request_id: str,
        success: bool,
        error_type: Optional[str] = None
    ) -> Optional[float]:
        """End tracking for an inference request and return latency"""
        try:
            if request_id not in self._active_requests[model_id]:
                logger.warning(f"Request {request_id} not found for model {model_id}")
                return None
            
            request_info = self._active_requests[model_id][request_id]
            latency_ms = (time.time() - request_info["start_time"]) * 1000
            
            # Update counters
            if success:
                self._request_counters[model_id]["success"] += 1
            else:
                self._request_counters[model_id]["error"] += 1
                if error_type == "timeout":
                    self._request_counters[model_id]["timeout"] += 1
            
            # Clean up
            del self._active_requests[model_id][request_id]
            
            return latency_ms
            
        except Exception as e:
            logger.error(f"Failed to end request tracking: {e}")
            return None
    
    async def create_performance_snapshot(
        self,
        model_id: str,
        cpu_usage: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None
    ) -> Optional[PerformanceSnapshot]:
        """Create a performance snapshot for a model"""
        try:
            counters = self._request_counters[model_id]
            
            if counters["total"] == 0:
                return None  # No requests to analyze
            
            # Collect recent latency data
            recent_snapshots = self._performance_history[model_id].get_recent_snapshots(
                hours=1
            )
            
            # Calculate latency metrics from recent data
            if recent_snapshots:
                recent_latencies = [s.avg_latency_ms for s in recent_snapshots]
                avg_latency = np.mean(recent_latencies)
                p50_latency = np.percentile(recent_latencies, 50)
                p95_latency = np.percentile(recent_latencies, 95)
                p99_latency = np.percentile(recent_latencies, 99)
                max_latency = np.max(recent_latencies)
            else:
                # Use current period estimates
                avg_latency = p50_latency = p95_latency = p99_latency = max_latency = 0.0
            
            # Calculate throughput (requests per second)
            time_window_minutes = 1.0  # Current snapshot window
            rps = counters["total"] / (time_window_minutes * 60)
            
            # Calculate rates
            success_rate = counters["success"] / counters["total"] if counters["total"] > 0 else 0.0
            error_rate = counters["error"] / counters["total"] if counters["total"] > 0 else 0.0
            timeout_rate = counters["timeout"] / counters["total"] if counters["total"] > 0 else 0.0
            
            # Get concurrent requests
            concurrent_requests = len(self._active_requests[model_id])
            
            snapshot = PerformanceSnapshot(
                model_id=model_id,
                timestamp=aware_utc_now(),
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                max_latency_ms=max_latency,
                requests_per_second=rps,
                concurrent_requests=concurrent_requests,
                success_rate=success_rate,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                cpu_usage_percent=cpu_usage or 0.0,
                memory_usage_mb=memory_usage_mb or 0.0,
                gpu_memory_mb=gpu_memory_mb,
                sample_count=counters["total"]
            )
            
            # Add to history
            self._performance_history[model_id].add_snapshot(snapshot)
            
            # Reset counters for next period
            self._request_counters[model_id] = {
                "total": 0, "success": 0, "error": 0, "timeout": 0
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create performance snapshot for {model_id}: {e}")
            return None
    
    async def get_model_performance_summary(
        self, 
        model_id: str, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary for a model"""
        try:
            history = self._performance_history[model_id]
            recent_snapshots = history.get_recent_snapshots(hours)
            
            if not recent_snapshots:
                return {
                    "model_id": model_id,
                    "period_hours": hours,
                    "status": "no_data",
                    "message": "No performance data available"
                }
            
            # Calculate aggregate metrics
            latencies = [s.avg_latency_ms for s in recent_snapshots]
            success_rates = [s.success_rate for s in recent_snapshots]
            error_rates = [s.error_rate for s in recent_snapshots]
            rps_values = [s.requests_per_second for s in recent_snapshots]
            
            # Performance trends
            performance_trend = self._calculate_performance_trend(recent_snapshots)
            
            # Degradation detection
            degradation_detected, degradation_score = await self._detect_performance_degradation(
                model_id, recent_snapshots
            )
            
            return {
                "model_id": model_id,
                "period_hours": hours,
                "timestamp": aware_utc_now().isoformat(),
                "snapshot_count": len(recent_snapshots),
                
                # Latency summary
                "latency_summary": {
                    "avg_ms": float(np.mean(latencies)),
                    "min_ms": float(np.min(latencies)),
                    "max_ms": float(np.max(latencies)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "std_ms": float(np.std(latencies))
                },
                
                # Throughput summary
                "throughput_summary": {
                    "avg_rps": float(np.mean(rps_values)),
                    "max_rps": float(np.max(rps_values)),
                    "total_requests": sum(s.sample_count for s in recent_snapshots)
                },
                
                # Quality summary
                "quality_summary": {
                    "avg_success_rate": float(np.mean(success_rates)),
                    "min_success_rate": float(np.min(success_rates)),
                    "avg_error_rate": float(np.mean(error_rates)),
                    "max_error_rate": float(np.max(error_rates))
                },
                
                # Performance analysis
                "performance_trend": performance_trend,
                "degradation_detected": degradation_detected,
                "degradation_score": degradation_score,
                
                # Current status
                "current_status": self._assess_current_performance(recent_snapshots[-1]),
                "recommendations": self._generate_performance_recommendations(
                    recent_snapshots, degradation_detected, performance_trend
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary for {model_id}: {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }
    
    async def get_all_models_performance(self) -> List[Dict[str, Any]]:
        """Get performance summaries for all monitored models"""
        summaries = []
        
        for model_id in self._performance_history.keys():
            summary = await self.get_model_performance_summary(model_id)
            summaries.append(summary)
        
        return summaries
    
    def _calculate_performance_trend(
        self, snapshots: List[PerformanceSnapshot]
    ) -> Dict[str, Any]:
        """Calculate performance trend from snapshots"""
        if len(snapshots) < 3:
            return {"trend": "insufficient_data", "direction": "stable"}
        
        try:
            # Use latency as primary performance indicator
            latencies = [s.avg_latency_ms for s in snapshots]
            
            # Calculate trend using simple moving average comparison
            recent_half = latencies[len(latencies)//2:]
            older_half = latencies[:len(latencies)//2]
            
            recent_avg = np.mean(recent_half)
            older_avg = np.mean(older_half)
            
            # Calculate percentage change
            if older_avg > 0:
                change_percent = ((recent_avg - older_avg) / older_avg) * 100
            else:
                change_percent = 0
            
            # Determine trend direction
            if abs(change_percent) < 5:  # Less than 5% change
                direction = "stable"
            elif change_percent > 0:
                direction = "degrading"  # Higher latency = worse performance
            else:
                direction = "improving"  # Lower latency = better performance
            
            return {
                "trend": "calculated",
                "direction": direction,
                "change_percent": float(change_percent),
                "recent_avg_latency": float(recent_avg),
                "baseline_avg_latency": float(older_avg)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance trend: {e}")
            return {"trend": "error", "direction": "unknown"}
    
    async def _detect_performance_degradation(
        self, 
        model_id: str, 
        recent_snapshots: List[PerformanceSnapshot]
    ) -> Tuple[bool, float]:
        """Detect performance degradation using baseline comparison"""
        try:
            if len(recent_snapshots) < 5:
                return False, 0.0
            
            # Get or establish baseline
            baseline = self._performance_baselines.get(model_id)
            if not baseline:
                baseline = await self._establish_performance_baseline(model_id, recent_snapshots)
                if not baseline:
                    return False, 0.0
            
            # Calculate current performance
            current_latency = np.mean([s.avg_latency_ms for s in recent_snapshots[-5:]])
            current_error_rate = np.mean([s.error_rate for s in recent_snapshots[-5:]])
            
            # Compare to baseline
            latency_degradation = (current_latency - baseline["avg_latency"]) / baseline["avg_latency"]
            error_rate_degradation = current_error_rate - baseline["avg_error_rate"]
            
            # Calculate degradation score (0-1, higher = more degraded)
            degradation_score = max(
                0.0,
                min(1.0, (latency_degradation + error_rate_degradation * 2) / 3)
            )
            
            # Detect degradation (>20% worse performance)
            degradation_detected = degradation_score > 0.2
            
            return degradation_detected, float(degradation_score)
            
        except Exception as e:
            logger.error(f"Failed to detect performance degradation: {e}")
            return False, 0.0
    
    async def _establish_performance_baseline(
        self, 
        model_id: str, 
        snapshots: List[PerformanceSnapshot]
    ) -> Optional[Dict[str, float]]:
        """Establish performance baseline for a model"""
        try:
            if len(snapshots) < 10:
                return None
            
            # Use first 70% of data as baseline
            baseline_size = int(len(snapshots) * 0.7)
            baseline_snapshots = snapshots[:baseline_size]
            
            baseline = {
                "avg_latency": float(np.mean([s.avg_latency_ms for s in baseline_snapshots])),
                "p95_latency": float(np.mean([s.p95_latency_ms for s in baseline_snapshots])),
                "avg_error_rate": float(np.mean([s.error_rate for s in baseline_snapshots])),
                "avg_success_rate": float(np.mean([s.success_rate for s in baseline_snapshots])),
                "established_at": aware_utc_now().timestamp()
            }
            
            self._performance_baselines[model_id] = baseline
            
            logger.info(f"Established performance baseline for {model_id}")
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to establish baseline for {model_id}: {e}")
            return None
    
    def _assess_current_performance(
        self, latest_snapshot: PerformanceSnapshot
    ) -> Dict[str, Any]:
        """Assess current performance status"""
        status = "good"
        issues = []
        
        # Check latency
        if latest_snapshot.p95_latency_ms > 5000:  # >5 seconds
            status = "poor"
            issues.append("High P95 latency detected")
        elif latest_snapshot.p95_latency_ms > 1000:  # >1 second
            status = "warning"
            issues.append("Elevated P95 latency")
        
        # Check error rate
        if latest_snapshot.error_rate > 0.1:  # >10% errors
            status = "poor"
            issues.append("High error rate detected")
        elif latest_snapshot.error_rate > 0.05:  # >5% errors
            if status == "good":
                status = "warning"
            issues.append("Elevated error rate")
        
        # Check success rate
        if latest_snapshot.success_rate < 0.9:  # <90% success
            status = "poor"
            issues.append("Low success rate detected")
        
        return {
            "status": status,
            "issues": issues,
            "metrics": {
                "avg_latency_ms": latest_snapshot.avg_latency_ms,
                "p95_latency_ms": latest_snapshot.p95_latency_ms,
                "success_rate": latest_snapshot.success_rate,
                "error_rate": latest_snapshot.error_rate,
                "requests_per_second": latest_snapshot.requests_per_second
            }
        }
    
    def _generate_performance_recommendations(
        self,
        snapshots: List[PerformanceSnapshot],
        degradation_detected: bool,
        performance_trend: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not snapshots:
            return ["No performance data available for analysis"]
        
        latest = snapshots[-1]
        
        # Latency recommendations
        if latest.p95_latency_ms > 2000:
            recommendations.append("🔴 High latency detected - investigate model optimization")
        elif latest.p95_latency_ms > 500:
            recommendations.append("🟡 Moderate latency - consider performance tuning")
        
        # Error rate recommendations
        if latest.error_rate > 0.05:
            recommendations.append("⚠️ High error rate - review error logs and input validation")
        
        # Throughput recommendations
        if latest.requests_per_second < 1:
            recommendations.append("📈 Low throughput - consider increasing concurrency or optimizing batch processing")
        
        # Trend-based recommendations
        trend_direction = performance_trend.get("direction", "unknown")
        if trend_direction == "degrading":
            recommendations.append("📉 Performance degrading trend - proactive optimization recommended")
        elif trend_direction == "improving":
            recommendations.append("📈 Performance improving - continue current optimization efforts")
        
        # Degradation recommendations
        if degradation_detected:
            recommendations.extend([
                "🚨 Performance degradation detected - consider model refresh or infrastructure scaling",
                "Review recent changes to model, data, or infrastructure"
            ])
        
        # Resource recommendations
        if latest.memory_usage_mb > 1000:  # >1GB
            recommendations.append("💾 High memory usage - monitor for memory leaks")
        
        if not recommendations:
            recommendations.append("✅ Performance metrics within normal ranges")
        
        return recommendations


# Global performance tracker instance
_performance_tracker: Optional[ModelPerformanceTracker] = None

async def get_performance_tracker() -> ModelPerformanceTracker:
    """Get or create global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = ModelPerformanceTracker()
    return _performance_tracker