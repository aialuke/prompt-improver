"""Training Metrics Service - Clean Architecture Implementation.

Implements performance monitoring, metrics collection, and analysis.
Extracted from training_system_manager.py (2109 lines) as part of decomposition.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from prompt_improver.cli.services.training_protocols import TrainingMetricsProtocol


class TrainingMetrics(TrainingMetricsProtocol):
    """Training metrics service implementing Clean Architecture patterns.

    Responsibilities:
    - Performance monitoring and metrics collection
    - Resource usage tracking and analysis
    - Training progress measurement
    - System health metrics aggregation
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("apes.training_metrics")

        # Metrics collection state
        self._metrics_history: list[dict[str, Any]] = []
        self._performance_baseline: dict[str, float] | None = None
        self._collection_start_time: float = time.time()

    async def get_resource_usage(self) -> dict[str, float]:
        """Get current system resource usage metrics.

        Returns:
            Resource usage metrics including memory, CPU, and I/O
        """
        try:
            import psutil

            process = psutil.Process()

            # Get detailed process metrics
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            # Get system-wide metrics for context
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent()

            resource_metrics = {
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": cpu_percent,
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "system_memory_available_gb": system_memory.available / (1024**3),
                "system_memory_percent": system_memory.percent,
                "system_cpu_percent": system_cpu,
            }

            # Add I/O metrics if available
            try:
                io_counters = process.io_counters()
                resource_metrics.update({
                    "io_read_bytes": io_counters.read_bytes,
                    "io_write_bytes": io_counters.write_bytes,
                    "io_read_count": io_counters.read_count,
                    "io_write_count": io_counters.write_count,
                })
            except (AttributeError, psutil.AccessDenied):
                # I/O counters not available on all platforms
                pass

            # Add network metrics if available
            try:
                network_connections = len(process.connections())
                resource_metrics["network_connections"] = network_connections
            except (AttributeError, psutil.AccessDenied):
                pass

            return resource_metrics

        except ImportError:
            # Fallback metrics when psutil is not available
            return {
                "memory_mb": 0,
                "cpu_percent": 0,
                "open_files": 0,
                "status": "psutil_not_available"
            }

    async def get_detailed_training_metrics(self) -> dict[str, Any]:
        """Get comprehensive training performance metrics.

        Returns:
            Detailed training metrics including performance trends and analysis
        """
        try:
            current_time = time.time()
            uptime_seconds = current_time - self._collection_start_time

            # Get current resource usage
            resource_usage = await self.get_resource_usage()

            # Calculate performance metrics
            performance_metrics = {
                "collection_uptime_seconds": uptime_seconds,
                "metrics_collection_count": len(self._metrics_history),
                "resource_usage": resource_usage,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Add performance trends if we have historical data
            if len(self._metrics_history) > 1:
                performance_metrics["trends"] = await self._calculate_performance_trends()

            # Add resource efficiency metrics
            performance_metrics["efficiency"] = await self._calculate_resource_efficiency(resource_usage)

            # Add performance health score
            performance_metrics["health_score"] = await self._calculate_performance_health_score(
                resource_usage, performance_metrics.get("trends", {})
            )

            # Store current metrics for trend analysis
            self._metrics_history.append({
                "timestamp": current_time,
                "resource_usage": resource_usage,
                "performance_metrics": performance_metrics,
            })

            # Keep only recent history (last 100 entries)
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

            return performance_metrics

        except Exception as e:
            self.logger.exception(f"Failed to get detailed training metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def get_current_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics for checkpoint creation.

        Returns:
            Current performance state for checkpointing
        """
        try:
            resource_usage = await self.get_resource_usage()

            performance_metrics = {
                "resource_usage": resource_usage,
                "collection_timestamp": datetime.now(UTC).isoformat(),
                "uptime_seconds": time.time() - self._collection_start_time,
                "metrics_available": len(self._metrics_history),
            }

            # Add performance indicators
            if resource_usage.get("memory_mb", 0) > 0:
                performance_metrics["performance_indicators"] = {
                    "memory_usage_category": self._categorize_memory_usage(
                        resource_usage["memory_mb"]
                    ),
                    "cpu_usage_category": self._categorize_cpu_usage(
                        resource_usage.get("cpu_percent", 0)
                    ),
                }

            # Add recent trend if available
            if len(self._metrics_history) >= 5:
                recent_metrics = self._metrics_history[-5:]
                performance_metrics["recent_trend"] = self._analyze_recent_trend(recent_metrics)

            return performance_metrics

        except Exception as e:
            self.logger.exception(f"Failed to get current performance metrics: {e}")
            return {"error": str(e)}

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get summarized performance metrics for status reporting.

        Returns:
            Performance summary with key metrics and insights
        """
        try:
            # Get current metrics
            current_metrics = await self.get_detailed_training_metrics()

            # Calculate summary statistics
            summary = {
                "current_performance": {
                    "memory_mb": current_metrics["resource_usage"].get("memory_mb", 0),
                    "cpu_percent": current_metrics["resource_usage"].get("cpu_percent", 0),
                    "health_score": current_metrics.get("health_score", 0),
                },
                "collection_statistics": {
                    "uptime_seconds": current_metrics.get("collection_uptime_seconds", 0),
                    "total_measurements": len(self._metrics_history),
                    "measurement_frequency": self._calculate_measurement_frequency(),
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Add performance insights
            if len(self._metrics_history) > 10:
                summary["performance_insights"] = await self._generate_performance_insights()

            # Add recommendations
            summary["recommendations"] = await self._generate_performance_recommendations(
                current_metrics
            )

            return summary

        except Exception as e:
            self.logger.exception(f"Failed to get performance summary: {e}")
            return {"error": str(e)}

    async def record_training_iteration(
        self,
        iteration: int,
        metrics: dict[str, float],
        duration_seconds: float
    ) -> None:
        """Record metrics for a training iteration.

        Args:
            iteration: Training iteration number
            metrics: Performance metrics for this iteration
            duration_seconds: Time taken for iteration
        """
        try:
            iteration_metrics = {
                "iteration": iteration,
                "duration_seconds": duration_seconds,
                "metrics": metrics,
                "timestamp": time.time(),
                "resource_usage": await self.get_resource_usage(),
            }

            self._metrics_history.append(iteration_metrics)

            # Update performance baseline if this is significantly better
            if self._performance_baseline is None:
                self._performance_baseline = metrics.copy()
            else:
                await self._update_performance_baseline(metrics)

            self.logger.debug(f"Recorded training iteration {iteration} metrics")

        except Exception as e:
            self.logger.exception(f"Failed to record training iteration metrics: {e}")

    async def _calculate_performance_trends(self) -> dict[str, Any]:
        """Calculate performance trends from historical metrics."""
        if len(self._metrics_history) < 2:
            return {"status": "insufficient_data"}

        try:
            # Extract resource usage trends
            memory_values = []
            cpu_values = []
            timestamps = []

            for entry in self._metrics_history[-20:]:  # Last 20 measurements
                if "resource_usage" in entry:
                    memory_values.append(entry["resource_usage"].get("memory_mb", 0))
                    cpu_values.append(entry["resource_usage"].get("cpu_percent", 0))
                    timestamps.append(entry.get("timestamp", time.time()))

            trends = {}

            # Memory trend
            if len(memory_values) >= 2:
                trends["memory_trend"] = {
                    "current": memory_values[-1],
                    "average": sum(memory_values) / len(memory_values),
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "direction": "increasing" if memory_values[-1] > memory_values[0] else "decreasing",
                }

            # CPU trend
            if len(cpu_values) >= 2:
                trends["cpu_trend"] = {
                    "current": cpu_values[-1],
                    "average": sum(cpu_values) / len(cpu_values),
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "direction": "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing",
                }

            # Performance stability
            if len(memory_values) >= 5:
                memory_variance = sum((x - trends["memory_trend"]["average"]) ** 2 for x in memory_values) / len(memory_values)
                trends["stability"] = {
                    "memory_variance": memory_variance,
                    "stability_score": max(0, 1 - (memory_variance / trends["memory_trend"]["average"])),
                }

            return trends

        except Exception as e:
            self.logger.exception(f"Failed to calculate performance trends: {e}")
            return {"error": str(e)}

    async def _calculate_resource_efficiency(self, resource_usage: dict[str, float]) -> dict[str, Any]:
        """Calculate resource efficiency metrics."""
        try:
            efficiency = {}

            # Memory efficiency
            memory_mb = resource_usage.get("memory_mb", 0)
            if memory_mb > 0:
                # Categorize memory usage efficiency
                if memory_mb < 100:
                    efficiency["memory_efficiency"] = "excellent"
                elif memory_mb < 500:
                    efficiency["memory_efficiency"] = "good"
                elif memory_mb < 1000:
                    efficiency["memory_efficiency"] = "acceptable"
                else:
                    efficiency["memory_efficiency"] = "needs_optimization"

            # CPU efficiency
            cpu_percent = resource_usage.get("cpu_percent", 0)
            if cpu_percent >= 0:
                if cpu_percent < 20:
                    efficiency["cpu_efficiency"] = "excellent"
                elif cpu_percent < 50:
                    efficiency["cpu_efficiency"] = "good"
                elif cpu_percent < 80:
                    efficiency["cpu_efficiency"] = "acceptable"
                else:
                    efficiency["cpu_efficiency"] = "needs_optimization"

            # Overall efficiency score
            memory_score = {"excellent": 1.0, "good": 0.8, "acceptable": 0.6, "needs_optimization": 0.3}.get(
                efficiency.get("memory_efficiency", "needs_optimization"), 0.3
            )
            cpu_score = {"excellent": 1.0, "good": 0.8, "acceptable": 0.6, "needs_optimization": 0.3}.get(
                efficiency.get("cpu_efficiency", "needs_optimization"), 0.3
            )

            efficiency["overall_efficiency_score"] = (memory_score + cpu_score) / 2

            return efficiency

        except Exception as e:
            self.logger.exception(f"Failed to calculate resource efficiency: {e}")
            return {"error": str(e)}

    async def _calculate_performance_health_score(
        self, resource_usage: dict[str, float], trends: dict[str, Any]
    ) -> float:
        """Calculate overall performance health score (0-1)."""
        try:
            score_components = []

            # Resource usage score
            memory_mb = resource_usage.get("memory_mb", 0)
            cpu_percent = resource_usage.get("cpu_percent", 0)

            # Memory score (lower is better)
            memory_score = max(0, 1 - (memory_mb / 2000))  # 2GB as high baseline
            score_components.append(memory_score * 0.4)

            # CPU score (moderate usage is optimal)
            if cpu_percent < 10:
                cpu_score = 0.8  # Too low might indicate underutilization
            elif cpu_percent < 50:
                cpu_score = 1.0  # Optimal range
            elif cpu_percent < 80:
                cpu_score = 0.7  # High but acceptable
            else:
                cpu_score = 0.3  # Too high
            score_components.append(cpu_score * 0.3)

            # Stability score from trends
            stability_score = trends.get("stability", {}).get("stability_score", 0.5)
            score_components.append(stability_score * 0.3)

            return sum(score_components)

        except Exception as e:
            self.logger.exception(f"Failed to calculate performance health score: {e}")
            return 0.0

    def _categorize_memory_usage(self, memory_mb: float) -> str:
        """Categorize memory usage level."""
        if memory_mb < 50:
            return "low"
        if memory_mb < 200:
            return "moderate"
        if memory_mb < 500:
            return "high"
        return "very_high"

    def _categorize_cpu_usage(self, cpu_percent: float) -> str:
        """Categorize CPU usage level."""
        if cpu_percent < 10:
            return "low"
        if cpu_percent < 30:
            return "moderate"
        if cpu_percent < 70:
            return "high"
        return "very_high"

    def _analyze_recent_trend(self, recent_metrics: list[dict[str, Any]]) -> dict[str, str]:
        """Analyze recent performance trend."""
        if len(recent_metrics) < 3:
            return {"status": "insufficient_data"}

        try:
            # Extract memory values from recent metrics
            memory_values = [entry["resource_usage"].get("memory_mb", 0) for entry in recent_metrics if "resource_usage" in entry]

            if len(memory_values) < 3:
                return {"status": "insufficient_memory_data"}

            # Simple trend analysis
            first_half = memory_values[:len(memory_values) // 2]
            second_half = memory_values[len(memory_values) // 2:]

            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            change_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0

            if abs(change_percent) < 5:
                trend = "stable"
            elif change_percent > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "trend": trend,
                "change_percent": round(change_percent, 2),
                "status": "analyzed"
            }

        except Exception as e:
            return {"status": "analysis_error", "error": str(e)}

    def _calculate_measurement_frequency(self) -> float:
        """Calculate average measurement frequency."""
        if len(self._metrics_history) < 2:
            return 0.0

        try:
            timestamps = [entry.get("timestamp", time.time()) for entry in self._metrics_history[-10:]]
            if len(timestamps) < 2:
                return 0.0

            intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            avg_interval = sum(intervals) / len(intervals)

            return 1 / avg_interval if avg_interval > 0 else 0.0  # Frequency in Hz

        except Exception as e:
            self.logger.exception(f"Failed to calculate measurement frequency: {e}")
            return 0.0

    async def _generate_performance_insights(self) -> list[str]:
        """Generate performance insights from historical data."""
        insights = []

        try:
            if len(self._metrics_history) < 10:
                return ["Insufficient data for performance insights"]

            # Analyze memory usage patterns
            recent_memory = [
                entry["resource_usage"].get("memory_mb", 0)
                for entry in self._metrics_history[-20:]
                if "resource_usage" in entry
            ]

            if recent_memory:
                avg_memory = sum(recent_memory) / len(recent_memory)
                max_memory = max(recent_memory)

                if max_memory > 1000:
                    insights.append(f"High memory usage detected (peak: {max_memory:.1f}MB)")

                if max_memory > avg_memory * 2:
                    insights.append("Memory usage shows high variability")

            # Analyze CPU usage patterns
            recent_cpu = [
                entry["resource_usage"].get("cpu_percent", 0)
                for entry in self._metrics_history[-20:]
                if "resource_usage" in entry
            ]

            if recent_cpu:
                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                if avg_cpu > 80:
                    insights.append(f"High CPU utilization (avg: {avg_cpu:.1f}%)")
                elif avg_cpu < 5:
                    insights.append("Low CPU utilization - system may be idle")

            # Collection frequency insights
            frequency = self._calculate_measurement_frequency()
            if frequency < 0.1:
                insights.append("Low measurement frequency - consider more frequent monitoring")

        except Exception as e:
            insights.append(f"Error generating insights: {e!s}")

        return insights if insights else ["No significant performance insights available"]

    async def _generate_performance_recommendations(self, current_metrics: dict[str, Any]) -> list[str]:
        """Generate performance recommendations based on current metrics."""
        recommendations = []

        try:
            resource_usage = current_metrics.get("resource_usage", {})
            efficiency = current_metrics.get("efficiency", {})

            # Memory recommendations
            memory_mb = resource_usage.get("memory_mb", 0)
            if memory_mb > 1000:
                recommendations.append("Consider reducing batch sizes or implementing memory optimization")

            if efficiency.get("memory_efficiency") == "needs_optimization":
                recommendations.append("Memory usage is high - investigate memory leaks or optimize data structures")

            # CPU recommendations
            cpu_percent = resource_usage.get("cpu_percent", 0)
            if cpu_percent > 90:
                recommendations.append("CPU usage is very high - consider reducing concurrent operations")
            elif cpu_percent < 5:
                recommendations.append("CPU usage is very low - consider increasing parallelism")

            # Health score recommendations
            health_score = current_metrics.get("health_score", 0)
            if health_score < 0.5:
                recommendations.append("Overall performance health is poor - review system configuration")

            # File handle recommendations
            open_files = resource_usage.get("open_files", 0)
            if open_files > 100:
                recommendations.append("High number of open files - ensure proper resource cleanup")

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e!s}")

        return recommendations if recommendations else ["System performance appears optimal"]

    async def _update_performance_baseline(self, new_metrics: dict[str, float]) -> None:
        """Update performance baseline with new metrics if they're significantly better."""
        try:
            if self._performance_baseline is None:
                self._performance_baseline = new_metrics.copy()
                return

            # Simple baseline update - use better values
            for key, value in new_metrics.items():
                if isinstance(value, (int, float)):
                    current_baseline = self._performance_baseline.get(key, value)
                    # For most metrics, lower is better (like loss), but effectiveness_score higher is better
                    if key == "effectiveness_score":
                        if value > current_baseline:
                            self._performance_baseline[key] = value
                    elif value < current_baseline:
                        self._performance_baseline[key] = value

        except Exception as e:
            self.logger.exception(f"Failed to update performance baseline: {e}")

    # Utility methods for external access
    def get_metrics_history(self, count: int | None = None) -> list[dict[str, Any]]:
        """Get metrics history for analysis.

        Args:
            count: Number of recent entries to return, None for all

        Returns:
            List of historical metrics entries
        """
        if count is None:
            return self._metrics_history.copy()
        return self._metrics_history[-count:] if count > 0 else []

    def reset_metrics(self) -> None:
        """Reset all collected metrics and baselines."""
        self._metrics_history.clear()
        self._performance_baseline = None
        self._collection_start_time = time.time()
        self.logger.info("Training metrics reset")
