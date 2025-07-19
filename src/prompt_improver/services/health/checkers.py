"""Individual health checker implementations for APES components.
PHASE 3: Health Check Consolidation - Component Checkers
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .base import HealthChecker, HealthResult, HealthStatus

# Import Redis Health Monitor
try:
    from .redis_monitor import RedisHealthMonitor
    
    REDIS_MONITOR_AVAILABLE = True
except Exception:
    REDIS_MONITOR_AVAILABLE = False
    RedisHealthMonitor = None

# Graceful imports for various services
try:
    from .background_manager import get_background_task_manager
    
    BACKGROUND_MANAGER_AVAILABLE = True
except Exception:
    BACKGROUND_MANAGER_AVAILABLE = False
    get_background_task_manager = None

try:
    from ...optimization.batch_processor import BatchProcessor
    
    BATCH_PROCESSOR_AVAILABLE = True
except Exception as e:
    BATCH_PROCESSOR_AVAILABLE = False
    BatchProcessor = None
    print(f"BatchProcessor import failed: {e}")

QUEUE_SERVICES_AVAILABLE = BACKGROUND_MANAGER_AVAILABLE or BATCH_PROCESSOR_AVAILABLE

# Graceful database import handling
try:
    from ...database import get_session

    DATABASE_AVAILABLE = True
except Exception:
    DATABASE_AVAILABLE = False
    get_session = None


class DatabaseHealthChecker(HealthChecker):
    """Database connectivity and performance health checker"""

    def __init__(self):
        super().__init__("database")

    async def check(self) -> HealthResult:
        """Check database connectivity and performance"""
        if not DATABASE_AVAILABLE or get_session is None:
            return HealthResult(
                status=HealthStatus.WARNING,
                component=self.name,
                message="Database configuration not available",
                error="Database credentials not configured",
            )

        try:
            from sqlalchemy import text

            from ...database import scalar

            start_time = time.time()
            async with get_session() as session:
                await scalar(session, text("SELECT 1"))
                response_time = (time.time() - start_time) * 1000

                # Check for long-running queries
                long_queries = await scalar(session, text("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < NOW() - INTERVAL '30 seconds'
                """))
                long_queries = long_queries or 0

                # Get active connections
                active_connections = await scalar(session, text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                ))
                active_connections = active_connections or 0

                # Determine status based on response time and query health
                if response_time > 500 or long_queries > 0:
                    status = HealthStatus.FAILED
                elif response_time > 100:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.HEALTHY

                return HealthResult(
                    status=status,
                    component=self.name,
                    response_time_ms=response_time,
                    message=f"Database responding in {response_time:.1f}ms",
                    details={
                        "long_running_queries": long_queries,
                        "active_connections": active_connections,
                    },
                )

        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="Database connection failed",
            )


class MCPServerHealthChecker(HealthChecker):
    """MCP server performance health checker"""

    def __init__(self):
        super().__init__("mcp_server")

    async def check(self) -> HealthResult:
        """Check MCP server performance"""
        try:
            try:
                from ...mcp_server.mcp_server import improve_prompt
            except ImportError:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="MCP server module not available",
                    error="MCP server not configured",
                )

            start_time = time.time()
            result = await improve_prompt(
                prompt="Health check test prompt",
                context={"domain": "health_check"},
                session_id="health_check",
            )
            response_time = (time.time() - start_time) * 1000

            # Determine status based on response time
            if response_time > 500:
                status = HealthStatus.FAILED
            elif response_time > 200:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=f"MCP server responding in {response_time:.1f}ms",
            )

        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="MCP server health check failed",
            )


class AnalyticsServiceHealthChecker(HealthChecker):
    """Analytics service functionality health checker"""

    def __init__(self):
        super().__init__("analytics")

    async def check(self) -> HealthResult:
        """Check analytics service functionality"""
        try:
            try:
                from ..analytics import AnalyticsService
            except ImportError:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="Analytics service module not available",
                    error="Analytics service not configured",
                )

            analytics = AnalyticsService()
            start_time = time.time()
            result = await analytics.get_performance_trends(days=1)
            response_time = (time.time() - start_time) * 1000

            return HealthResult(
                status=HealthStatus.HEALTHY,
                component=self.name,
                response_time_ms=response_time,
                message=f"Analytics service responding in {response_time:.1f}ms",
                details={"data_points": len(result.get("trends", []))},
            )

        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="Analytics service check failed",
            )


class MLServiceHealthChecker(HealthChecker):
    """ML service availability health checker"""

    def __init__(self):
        super().__init__("ml_service")

    async def check(self) -> HealthResult:
        """Check ML service availability"""
        try:
            try:
                from ..ml_integration import get_ml_service
            except ImportError:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="ML service module not available",
                    error="ML service not configured",
                )

            start_time = time.time()
            ml_service = await get_ml_service()
            response_time = (time.time() - start_time) * 1000

            return HealthResult(
                status=HealthStatus.HEALTHY,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML service available in {response_time:.1f}ms",
            )

        except Exception as e:
            return HealthResult(
                status=HealthStatus.WARNING,  # ML service is optional
                component=self.name,
                error=str(e),
                message="ML service unavailable (fallback to rule-based)",
            )


class QueueHealthChecker(HealthChecker):
    """Queue systems health checker for monitoring queue metrics"""

    def __init__(self, batch_processor: BatchProcessor | None = None):
        super().__init__("queue")
        self.batch_processor = batch_processor

    async def check(self) -> HealthResult:
        """Check queue health including length, retry backlog, and latency"""
        if not QUEUE_SERVICES_AVAILABLE:
            return HealthResult(
                status=HealthStatus.WARNING,
                component=self.name,
                message="Queue services not configured",
                error="Queue or background services not available",
            )

        try:
            start_time = time.time()
            metrics = await self._collect_queue_metrics()
            status = self._evaluate_queue_health(metrics)
            response_time = (time.time() - start_time) * 1000

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=self._get_status_message(status, metrics),
                details=metrics,
                timestamp=datetime.now(),
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message=f"Queue health check failed: {e!s}",
                response_time_ms=response_time,
                timestamp=datetime.now(),
            )

    async def _collect_queue_metrics(self) -> dict[str, Any]:
        """Collect comprehensive queue metrics"""
        metrics = {}

        # Get batch processor metrics
        if self.batch_processor:
            metrics.update(await self._get_batch_processor_metrics())

        # Get background task manager metrics
        metrics.update(await self._get_background_task_metrics())

        # Calculate derived metrics
        metrics.update(self._calculate_derived_metrics(metrics))

        return metrics

    async def _get_batch_processor_metrics(self) -> dict[str, Any]:
        """Get metrics from batch processor"""
        try:
            processor_metrics = {}

            # Queue size metrics
            processor_metrics["training_queue_size"] = (
                self.batch_processor.get_queue_size()
            )
            processor_metrics["priority_queue_enabled"] = (
                self.batch_processor.config.enable_priority_queue
            )
            processor_metrics["max_queue_size"] = (
                self.batch_processor.config.max_queue_size
            )

            # Configuration metrics
            processor_metrics["batch_size"] = self.batch_processor.config.batch_size
            processor_metrics["concurrency"] = self.batch_processor.config.concurrency
            processor_metrics["max_attempts"] = self.batch_processor.config.max_attempts

            # Processing state
            processor_metrics["processing"] = self.batch_processor.processing

            # Performance metrics (if available)
            if (
                hasattr(self.batch_processor, "metrics")
                and self.batch_processor.metrics
            ):
                batch_metrics = self.batch_processor.metrics
                processor_metrics["processed_count"] = batch_metrics.get("processed", 0)
                processor_metrics["failed_count"] = batch_metrics.get("failed", 0)
                processor_metrics["retry_count"] = batch_metrics.get("retries", 0)

                # Calculate success rate
                total = batch_metrics.get("processed", 0) + batch_metrics.get(
                    "failed", 0
                )
                if total > 0:
                    processor_metrics["success_rate"] = (
                        batch_metrics.get("processed", 0) / total
                    )
                else:
                    processor_metrics["success_rate"] = 1.0

                # Calculate uptime
                if "start_time" in batch_metrics:
                    processor_metrics["uptime_seconds"] = (
                        time.time() - batch_metrics["start_time"]
                    )

            return processor_metrics

        except Exception as e:
            return {"batch_processor_error": str(e)}

    async def _get_background_task_metrics(self) -> dict[str, Any]:
        """Get metrics from background task manager"""
        try:
            task_manager = get_background_task_manager()

            background_metrics = {}
            background_metrics["background_queue_size"] = task_manager.get_queue_size()
            background_metrics["running_tasks"] = len(task_manager.get_running_tasks())
            background_metrics["max_concurrent_tasks"] = (
                task_manager.max_concurrent_tasks
            )

            # Task count by status
            task_counts = task_manager.get_task_count()
            background_metrics["task_counts"] = task_counts

            # Calculate queue utilization
            if task_manager.max_concurrent_tasks > 0:
                utilization = (
                    len(task_manager.get_running_tasks())
                    / task_manager.max_concurrent_tasks
                )
                background_metrics["task_utilization"] = utilization
            else:
                background_metrics["task_utilization"] = 0.0

            return background_metrics

        except Exception as e:
            return {"background_task_error": str(e)}

    def _calculate_derived_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Calculate derived metrics from collected data"""
        derived = {}

        # Total queue backlog
        training_queue = metrics.get("training_queue_size", 0)
        background_queue = metrics.get("background_queue_size", 0)
        derived["total_queue_backlog"] = training_queue + background_queue

        # Queue capacity utilization
        max_queue_size = metrics.get("max_queue_size", 1000)
        if max_queue_size > 0:
            derived["queue_capacity_utilization"] = training_queue / max_queue_size
        else:
            derived["queue_capacity_utilization"] = 0.0

        # Retry backlog ratio
        retry_count = metrics.get("retry_count", 0)
        processed_count = metrics.get("processed_count", 0)
        total_processed = retry_count + processed_count
        if total_processed > 0:
            derived["retry_backlog_ratio"] = retry_count / total_processed
        else:
            derived["retry_backlog_ratio"] = 0.0

        # Average latency estimation (based on processing metrics)
        if "uptime_seconds" in metrics and processed_count > 0:
            derived["avg_processing_latency_ms"] = (
                metrics["uptime_seconds"] / processed_count
            ) * 1000
        else:
            derived["avg_processing_latency_ms"] = 0.0

        # Throughput (items per second)
        if "uptime_seconds" in metrics and metrics["uptime_seconds"] > 0:
            derived["throughput_per_second"] = (
                processed_count / metrics["uptime_seconds"]
            )
        else:
            derived["throughput_per_second"] = 0.0

        return derived

    def _evaluate_queue_health(self, metrics: dict[str, Any]) -> HealthStatus:
        """Evaluate overall queue health based on metrics"""
        # Check for critical issues
        if "batch_processor_error" in metrics or "background_task_error" in metrics:
            return HealthStatus.FAILED

        # Check queue capacity
        capacity_utilization = metrics.get("queue_capacity_utilization", 0.0)
        if capacity_utilization >= 0.95:  # 95% capacity
            return HealthStatus.FAILED
        if capacity_utilization >= 0.80:  # 80% capacity
            return HealthStatus.WARNING

        # Check retry backlog
        retry_ratio = metrics.get("retry_backlog_ratio", 0.0)
        if retry_ratio >= 0.5:  # 50% retry rate
            return HealthStatus.FAILED
        if retry_ratio >= 0.2:  # 20% retry rate
            return HealthStatus.WARNING

        # Check success rate
        success_rate = metrics.get("success_rate", 1.0)
        if success_rate < 0.8:  # Less than 80% success
            return HealthStatus.FAILED
        if success_rate < 0.95:  # Less than 95% success
            return HealthStatus.WARNING

        # Check task utilization
        task_utilization = metrics.get("task_utilization", 0.0)
        if task_utilization >= 0.95:  # 95% task utilization
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _get_status_message(self, status: HealthStatus, metrics: dict[str, Any]) -> str:
        """Get human-readable status message"""
        total_backlog = metrics.get("total_queue_backlog", 0)
        capacity_util = metrics.get("queue_capacity_utilization", 0.0)
        success_rate = metrics.get("success_rate", 1.0)

        if status == HealthStatus.HEALTHY:
            return f"Queue healthy - {total_backlog} items queued, {capacity_util:.1%} capacity, {success_rate:.1%} success"
        if status == HealthStatus.WARNING:
            return f"Queue warning - {total_backlog} items queued, {capacity_util:.1%} capacity, {success_rate:.1%} success"
        return f"Queue failed - {total_backlog} items queued, {capacity_util:.1%} capacity, {success_rate:.1%} success"


class SystemResourcesHealthChecker(HealthChecker):
    """System resource usage health checker"""

    def __init__(self):
        super().__init__("system_resources")

    async def check(self) -> HealthResult:
        """Check system resource usage"""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Event loop detection and latency benchmarking
            loop_info = await self._check_event_loop_performance()

            status = HealthStatus.HEALTHY
            warnings = []

            if memory_usage_percent > 80:
                status = HealthStatus.WARNING
                warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")

            if disk_usage_percent > 85:
                status = HealthStatus.WARNING
                warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")

            if cpu_percent > 80:
                status = HealthStatus.WARNING
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Check event loop latency
            if loop_info.get("avg_latency_ms", 0) > 10:
                status = HealthStatus.WARNING
                warnings.append(
                    f"High event loop latency: {loop_info['avg_latency_ms']:.2f}ms"
                )

            message = f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_usage_percent:.1f}%, Disk {disk_usage_percent:.1f}%"
            if loop_info.get("uvloop_enabled"):
                message += f" - uvloop: {loop_info['loop_type']}"

            if warnings:
                message += f" - Warnings: {', '.join(warnings)}"

            return HealthResult(
                status=status,
                component=self.name,
                message=message,
                details={
                    "memory_usage_percent": memory_usage_percent,
                    "disk_usage_percent": disk_usage_percent,
                    "cpu_usage_percent": cpu_percent,
                    "warnings": warnings,
                    "event_loop": loop_info,
                },
            )

        except ImportError:
            return HealthResult(
                status=HealthStatus.WARNING,
                component=self.name,
                message="psutil not available for system monitoring",
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="System resource check failed",
            )

    async def _check_event_loop_performance(self) -> dict[str, Any]:
        """Check event loop type and performance metrics using event loop manager."""
        try:
            from prompt_improver.utils.event_loop_manager import get_event_loop_manager

            # Get event loop manager and run performance check
            manager = get_event_loop_manager()

            # Get loop information
            loop_info = manager.get_loop_info()

            # Run latency benchmark
            latency_metrics = await manager.benchmark_loop_latency(samples=10)

            # Combine information
            return {
                **loop_info,
                **latency_metrics,
                "enhanced_monitoring": True,
            }

        except ImportError:
            # Fallback to original implementation if event loop manager not available
            return await self._fallback_event_loop_check()
        except Exception as e:
            return {
                "loop_type": "unknown",
                "uvloop_enabled": False,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "samples": 0,
                "error": str(e),
                "enhanced_monitoring": False,
            }

    async def _fallback_event_loop_check(self) -> dict[str, Any]:
        """Fallback event loop check using original implementation."""
        try:
            # Get current event loop
            loop = asyncio.get_running_loop()

            # Detect event loop type
            loop_type = type(loop).__name__
            uvloop_enabled = "uvloop" in loop_type.lower()

            # Benchmark event loop latency
            latencies = []

            for _ in range(10):
                start = time.perf_counter()
                await asyncio.sleep(0)  # Yield control to event loop
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            return {
                "loop_type": loop_type,
                "uvloop_enabled": uvloop_enabled,
                "avg_ms": avg_latency,  # Updated to match new format
                "min_ms": min_latency,  # Updated to match new format
                "max_ms": max_latency,  # Updated to match new format
                "samples": len(latencies),  # Updated to match new format
                "enhanced_monitoring": False,
            }

        except Exception as e:
            return {
                "loop_type": "unknown",
                "uvloop_enabled": False,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "samples": 0,
                "error": str(e),
                "enhanced_monitoring": False,
            }
