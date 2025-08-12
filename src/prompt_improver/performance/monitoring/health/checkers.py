"""Individual health checker implementations for APES components.
PHASE 3: Health Check Consolidation - Component Checkers
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Union

from prompt_improver.performance.monitoring.health.base import (
    HealthChecker,
    HealthResult,
    HealthStatus,
)

try:
    from prompt_improver.performance.monitoring.health.redis_monitor import (
        redis_health_monitor,
    )

    REDIS_MONITOR_AVAILABLE = True
except Exception:
    REDIS_MONITOR_AVAILABLE = False
    redis_health_monitor = None
try:
    from prompt_improver.performance.monitoring.health.background_manager import (
        get_background_task_manager,
    )

    BACKGROUND_MANAGER_AVAILABLE = True
except Exception:
    BACKGROUND_MANAGER_AVAILABLE = False
    get_background_task_manager = None
try:
    from prompt_improver.ml.optimization.batch import (
        UnifiedBatchProcessor as BatchProcessor,
    )

    BATCH_PROCESSOR_AVAILABLE = True
    batch_processor = BatchProcessor
except Exception as e:
    BATCH_PROCESSOR_AVAILABLE = False
    batch_processor = None
    print(f"batch_processor import failed: {e}")
QUEUE_SERVICES_AVAILABLE = BACKGROUND_MANAGER_AVAILABLE or BATCH_PROCESSOR_AVAILABLE
try:
    from prompt_improver.performance.database import get_session

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

            from prompt_improver.performance.database import scalar

            start_time = time.time()
            async with get_session() as session:
                await scalar(session, text("SELECT 1"))
                response_time = (time.time() - start_time) * 1000
                long_queries = await scalar(
                    session,
                    text(
                        "\n                    SELECT count(*)\n                    FROM pg_stat_activity\n                    WHERE state = 'active'\n                    AND query_start < NOW() - INTERVAL '30 seconds'\n                "
                    ),
                )
                long_queries = long_queries or 0
                active_connections = await scalar(
                    session,
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                    ),
                )
                active_connections = active_connections or 0
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
                from prompt_improver.performance.mcp_server.server import APESMCPServer
            except ImportError:
                return HealthResult(
                    status=HealthStatus.WARNING,
                    component=self.name,
                    message="MCP server module not available",
                    error="MCP server not configured",
                )
            mcp_server = APESMCPServer()
            start_time = time.time()
            result = await mcp_server._improve_prompt_impl(
                prompt="Health check test prompt",
                context={"domain": "health_check"},
                session_id="health_check",
                rate_limit_remaining=None,
            )
            response_time = (time.time() - start_time) * 1000
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


class QueueHealthChecker(HealthChecker):
    """Queue systems health checker for monitoring queue metrics"""

    def __init__(self, batch_processor: batch_processor | None = None):
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
        if self.batch_processor:
            metrics.update(await self._get_batch_processor_metrics())
        metrics.update(await self._get_background_task_metrics())
        metrics.update(self._calculate_derived_metrics(metrics))
        return metrics

    async def _get_batch_processor_metrics(self) -> dict[str, Any]:
        """Get metrics from batch processor"""
        try:
            processor_metrics = {}
            processor_metrics["training_queue_size"] = (
                self.batch_processor.get_queue_size()
            )
            processor_metrics["priority_queue_enabled"] = (
                self.batch_processor.config.enable_priority_queue
            )
            processor_metrics["max_queue_size"] = (
                self.batch_processor.config.max_queue_size
            )
            processor_metrics["batch_size"] = self.batch_processor.config.batch_size
            processor_metrics["concurrency"] = self.batch_processor.config.concurrency
            processor_metrics["max_attempts"] = self.batch_processor.config.max_attempts
            processor_metrics["processing"] = self.batch_processor.processing
            if (
                hasattr(self.batch_processor, "metrics")
                and self.batch_processor.metrics
            ):
                batch_metrics = self.batch_processor.metrics
                processor_metrics["processed_count"] = batch_metrics.get("processed", 0)
                processor_metrics["failed_count"] = batch_metrics.get("failed", 0)
                processor_metrics["retry_count"] = batch_metrics.get("retries", 0)
                total = batch_metrics.get("processed", 0) + batch_metrics.get(
                    "failed", 0
                )
                if total > 0:
                    processor_metrics["success_rate"] = (
                        batch_metrics.get("processed", 0) / total
                    )
                else:
                    processor_metrics["success_rate"] = 1.0
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
            task_counts = task_manager.get_task_count()
            background_metrics["task_counts"] = task_counts
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
        training_queue = metrics.get("training_queue_size", 0)
        background_queue = metrics.get("background_queue_size", 0)
        derived["total_queue_backlog"] = training_queue + background_queue
        max_queue_size = metrics.get("max_queue_size", 1000)
        if max_queue_size > 0:
            derived["queue_capacity_utilization"] = training_queue / max_queue_size
        else:
            derived["queue_capacity_utilization"] = 0.0
        retry_count = metrics.get("retry_count", 0)
        processed_count = metrics.get("processed_count", 0)
        total_processed = retry_count + processed_count
        if total_processed > 0:
            derived["retry_backlog_ratio"] = retry_count / total_processed
        else:
            derived["retry_backlog_ratio"] = 0.0
        if "uptime_seconds" in metrics and processed_count > 0:
            derived["avg_processing_latency_ms"] = (
                metrics["uptime_seconds"] / processed_count * 1000
            )
        else:
            derived["avg_processing_latency_ms"] = 0.0
        if "uptime_seconds" in metrics and metrics["uptime_seconds"] > 0:
            derived["throughput_per_second"] = (
                processed_count / metrics["uptime_seconds"]
            )
        else:
            derived["throughput_per_second"] = 0.0
        return derived

    def _evaluate_queue_health(self, metrics: dict[str, Any]) -> HealthStatus:
        """Evaluate overall queue health based on metrics"""
        if "batch_processor_error" in metrics or "background_task_error" in metrics:
            return HealthStatus.FAILED
        capacity_utilization = metrics.get("queue_capacity_utilization", 0.0)
        if capacity_utilization >= 0.95:
            return HealthStatus.FAILED
        if capacity_utilization >= 0.8:
            return HealthStatus.WARNING
        retry_ratio = metrics.get("retry_backlog_ratio", 0.0)
        if retry_ratio >= 0.5:
            return HealthStatus.FAILED
        if retry_ratio >= 0.2:
            return HealthStatus.WARNING
        success_rate = metrics.get("success_rate", 1.0)
        if success_rate < 0.8:
            return HealthStatus.FAILED
        if success_rate < 0.95:
            return HealthStatus.WARNING
        task_utilization = metrics.get("task_utilization", 0.0)
        if task_utilization >= 0.95:
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

            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent
            cpu_percent = psutil.cpu_percent(interval=1)
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
        """Check event loop type and performance metrics using database services."""
        try:
            from prompt_improver.database import (
                ManagerMode,
                get_database_services,
            )

            manager = await await get_database_services(ManagerMode.HIGH_AVAILABILITY)
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__
            uvloop_enabled = "uvloop" in loop_type.lower()
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                async with manager.get_async_session() as session:
                    await asyncio.sleep(0)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            return {
                "loop_type": loop_type,
                "uvloop_enabled": uvloop_enabled,
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "samples": len(latencies),
                "enhanced_monitoring": True,
            }
        except ImportError:
            return await self._fallback_event_loop_check()
        except Exception as e:
            return {
                "loop_type": "unknown",
                "uvloop_enabled": False,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "samples": 0,
                "error": str(e),
                "enhanced_monitoring": False,
            }

    async def _fallback_event_loop_check(self) -> dict[str, Any]:
        """Fallback event loop check using original implementation."""
        try:
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__
            uvloop_enabled = "uvloop" in loop_type.lower()
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                await asyncio.sleep(0)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            return {
                "loop_type": loop_type,
                "uvloop_enabled": uvloop_enabled,
                "avg_ms": avg_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "samples": len(latencies),
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


class RedisHealthChecker(HealthChecker):
    """Comprehensive Redis health checker using the advanced monitoring system"""

    def __init__(self):
        super().__init__("redis")

    async def check(self) -> HealthResult:
        """Check Redis health using comprehensive monitoring"""
        try:
            from prompt_improver.cache.redis_health import get_redis_health_summary

            health_data = await get_redis_health_summary()
            status_mapping = {
                "healthy": HealthStatus.HEALTHY,
                "warning": HealthStatus.WARNING,
                "critical": HealthStatus.WARNING,
                "failed": HealthStatus.FAILED,
            }
            redis_status = health_data.get("status", "failed")
            health_status = status_mapping.get(redis_status, HealthStatus.FAILED)
            message_parts = [
                f"Redis {redis_status}",
                f"{health_data.get(response_time_ms, 0):.1f}ms latency",
                f"{health_data.get(memory_usage_mb, 0):.1f}MB memory",
                f"{health_data.get(hit_rate_percentage, 0):.1f}% hit rate",
            ]
            if health_data.get("issues"):
                message_parts.extend(health_data["issues"][:2])
            message = ", ".join(message_parts)
            details = {
                "redis_metrics": {
                    "memory_usage_mb": health_data.get("memory_usage_mb", 0),
                    "hit_rate_percentage": health_data.get("hit_rate_percentage", 0),
                    "connected_clients": health_data.get("connected_clients", 0),
                    "total_commands": health_data.get("total_commands", 0),
                    "fragmentation_ratio": health_data.get("fragmentation_ratio", 1.0),
                    "response_time_ms": health_data.get("response_time_ms", 0),
                },
                "issues": health_data.get("issues", []),
                "timestamp": health_data.get("timestamp"),
            }
            return HealthResult(
                status=health_status,
                component=self.name,
                message=message,
                details=details,
            )
        except ImportError as e:
            return await self._basic_redis_check()
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message=f"Redis health check failed: {e}",
            )

    async def _basic_redis_check(self) -> HealthResult:
        """Basic Redis connectivity check as fallback"""
        try:
            import time

            start_time = time.time()
            await redis_client.ping()
            response_time_ms = (time.time() - start_time) * 1000
            info = await redis_client.info()
            message = f"Redis basic check OK, {response_time_ms:.1f}ms latency"
            if response_time_ms > 100:
                status = HealthStatus.WARNING
                message += " (high latency)"
            else:
                status = HealthStatus.HEALTHY
            details = {
                "response_time_ms": round(response_time_ms, 2),
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human"),
                "fallback_check": True,
            }
            return HealthResult(
                status=status, component=self.name, message=message, details=details
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message=f"Redis connectivity failed: {e}",
            )
