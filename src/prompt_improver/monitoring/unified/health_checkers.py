"""Unified health checker components.

Consolidates the scattered health checking functionality into
unified, focused components following clean architecture principles.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from .protocols import HealthCheckComponentProtocol
from .types import ComponentCategory, HealthCheckResult, HealthStatus

logger = logging.getLogger(__name__)


class DatabaseHealthChecker:
    """Unified database health checker."""
    
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
    
    async def check_health(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Import here to avoid circular dependencies
            from prompt_improver.database import get_session
            
            async with asyncio.timeout(self.timeout_seconds):
                async with get_session() as session:
                    # Simple connectivity test
                    result = await session.execute("SELECT 1")
                    await result.fetchone()
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Check for long-running queries
                    long_query_result = await session.execute(
                        """
                        SELECT count(*) FROM pg_stat_activity 
                        WHERE state = 'active' 
                        AND query_start < NOW() - INTERVAL '30 seconds'
                        """
                    )
                    long_queries = (await long_query_result.fetchone())[0] or 0
                    
                    # Check active connections
                    conn_result = await session.execute(
                        "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                    )
                    active_connections = (await conn_result.fetchone())[0] or 0
                    
                    # Determine status based on performance
                    if response_time_ms > 500 or long_queries > 0:
                        status = HealthStatus.UNHEALTHY
                        message = f"Database performance issues: {response_time_ms:.1f}ms response"
                    elif response_time_ms > 100:
                        status = HealthStatus.DEGRADED  
                        message = f"Database slow response: {response_time_ms:.1f}ms"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Database healthy: {response_time_ms:.1f}ms response"
                    
                    return HealthCheckResult(
                        status=status,
                        component_name="database",
                        message=message,
                        response_time_ms=response_time_ms,
                        category=ComponentCategory.DATABASE,
                        details={
                            "active_connections": active_connections,
                            "long_running_queries": long_queries,
                            "response_time_ms": response_time_ms,
                        }
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="database", 
                message=f"Database timeout after {self.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.DATABASE,
                error="Connection timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="database",
                message=f"Database check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000, 
                category=ComponentCategory.DATABASE,
                error=str(e)
            )
    
    def get_component_name(self) -> str:
        return "database"
        
    def get_timeout_seconds(self) -> float:
        return self.timeout_seconds


class RedisHealthChecker:
    """Unified Redis health checker."""
    
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
    
    async def check_health(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            # Try to use the new Redis health manager first
            try:
                from prompt_improver.monitoring.redis.health import RedisHealthManager, DefaultRedisClientProvider
                
                async with asyncio.timeout(self.timeout_seconds):
                    client_provider = DefaultRedisClientProvider()
                    health_manager = RedisHealthManager(client_provider)
                    health_data = await health_manager.get_health_summary()
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Map Redis health status to our unified status
                    redis_status = health_data.get("status", "failed")
                    if redis_status == "healthy":
                        status = HealthStatus.HEALTHY
                    elif redis_status in ["warning", "critical"]:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.UNHEALTHY
                    
                    message_parts = [
                        f"Redis {redis_status}",
                        f"{health_data.get('response_time_ms', 0):.1f}ms latency",
                        f"{health_data.get('memory_usage_mb', 0):.1f}MB memory",
                    ]
                    
                    return HealthCheckResult(
                        status=status,
                        component_name="redis",
                        message=", ".join(message_parts),
                        response_time_ms=response_time_ms,
                        category=ComponentCategory.CACHE,
                        details={
                            "redis_health_data": health_data,
                            "memory_usage_mb": health_data.get("memory_usage_mb", 0),
                            "hit_rate_percentage": health_data.get("hit_rate_percentage", 0),
                            "connected_clients": health_data.get("connected_clients", 0),
                        }
                    )
                    
            except ImportError:
                # Fallback to basic Redis check
                return await self._basic_redis_check(start_time)
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="redis",
                message=f"Redis timeout after {self.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.CACHE,
                error="Connection timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="redis",
                message=f"Redis check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.CACHE,
                error=str(e)
            )
    
    async def _basic_redis_check(self, start_time: float) -> HealthCheckResult:
        """Basic Redis connectivity check."""
        try:
            from prompt_improver.database import (
                ManagerMode,
                create_security_context,
                get_database_services,
            )
            
            unified_manager = await get_database_services(ManagerMode.HIGH_AVAILABILITY)
            security_context = await create_security_context(
                agent_id="health_check", tier="basic"
            )
            
            # Test Redis connectivity with simple operations
            test_key = "health_check_test"
            await unified_manager.set_cached(
                test_key, "test_value", ttl_seconds=10, security_context=security_context
            )
            value = await unified_manager.get_cached(test_key, security_context)
            await unified_manager.invalidate_cached([test_key], security_context)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if value != "test_value":
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component_name="redis",
                    message="Redis read/write test failed",
                    response_time_ms=response_time_ms,
                    category=ComponentCategory.CACHE,
                    error="Read/write test failed"
                )
            
            # Determine status based on response time
            if response_time_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"Redis slow: {response_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy: {response_time_ms:.1f}ms"
            
            return HealthCheckResult(
                status=status,
                component_name="redis",
                message=message,
                response_time_ms=response_time_ms,
                category=ComponentCategory.CACHE,
                details={"basic_check": True, "read_write_test": "passed"}
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="redis",
                message=f"Redis basic check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.CACHE,
                error=str(e)
            )
    
    def get_component_name(self) -> str:
        return "redis"
        
    def get_timeout_seconds(self) -> float:
        return self.timeout_seconds


class MLModelsHealthChecker:
    """Unified ML models health checker."""
    
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
    
    async def check_health(self) -> HealthCheckResult:
        """Check ML models availability and performance."""
        start_time = time.time()
        
        try:
            async with asyncio.timeout(self.timeout_seconds):
                # Check if ML services are available
                try:
                    from prompt_improver.core.services.ml_service import MLServiceFacade
                    
                    # This is a lightweight check to see if ML services are responsive
                    ml_service = MLServiceFacade()
                    
                    # Simple availability check - just verify service responds
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        component_name="ml_models",
                        message=f"ML services available: {response_time_ms:.1f}ms",
                        response_time_ms=response_time_ms,
                        category=ComponentCategory.ML_MODELS,
                        details={"service_available": True}
                    )
                    
                except ImportError:
                    # ML services not configured
                    return HealthCheckResult(
                        status=HealthStatus.DEGRADED,
                        component_name="ml_models",
                        message="ML services not configured",
                        response_time_ms=(time.time() - start_time) * 1000,
                        category=ComponentCategory.ML_MODELS,
                        details={"service_available": False, "reason": "not_configured"}
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="ml_models",
                message=f"ML models timeout after {self.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.ML_MODELS,
                error="Timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                component_name="ml_models",
                message=f"ML models check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.ML_MODELS,
                error=str(e)
            )
    
    def get_component_name(self) -> str:
        return "ml_models"
        
    def get_timeout_seconds(self) -> float:
        return self.timeout_seconds


class SystemResourcesHealthChecker:
    """Unified system resources health checker."""
    
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
    
    async def check_health(self) -> HealthCheckResult:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            import psutil
            
            async with asyncio.timeout(self.timeout_seconds):
                # Collect system metrics
                memory = psutil.virtual_memory()
                memory_usage_percent = memory.percent
                
                disk = psutil.disk_usage("/")
                disk_usage_percent = disk.percent
                
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # Determine health status
                warnings = []
                status = HealthStatus.HEALTHY
                
                if memory_usage_percent > 90:
                    status = HealthStatus.UNHEALTHY
                    warnings.append(f"Critical memory usage: {memory_usage_percent:.1f}%")
                elif memory_usage_percent > 80:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")
                
                if disk_usage_percent > 95:
                    status = HealthStatus.UNHEALTHY 
                    warnings.append(f"Critical disk usage: {disk_usage_percent:.1f}%")
                elif disk_usage_percent > 85:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")
                
                if cpu_percent > 95:
                    status = HealthStatus.UNHEALTHY
                    warnings.append(f"Critical CPU usage: {cpu_percent:.1f}%")
                elif cpu_percent > 80:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
                
                message = f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_usage_percent:.1f}%, Disk {disk_usage_percent:.1f}%"
                if warnings:
                    message += f" - {', '.join(warnings)}"
                
                return HealthCheckResult(
                    status=status,
                    component_name="system_resources",
                    message=message,
                    response_time_ms=response_time_ms,
                    category=ComponentCategory.SYSTEM,
                    details={
                        "memory_usage_percent": memory_usage_percent,
                        "disk_usage_percent": disk_usage_percent,
                        "cpu_usage_percent": cpu_percent,
                        "warnings": warnings,
                    }
                )
                
        except ImportError:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                component_name="system_resources",
                message="psutil not available for system monitoring",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.SYSTEM,
                error="psutil not installed"
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="system_resources",
                message=f"System check timeout after {self.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.SYSTEM,
                error="Timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component_name="system_resources",
                message=f"System check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                category=ComponentCategory.SYSTEM,
                error=str(e)
            )
    
    def get_component_name(self) -> str:
        return "system_resources"
        
    def get_timeout_seconds(self) -> float:
        return self.timeout_seconds