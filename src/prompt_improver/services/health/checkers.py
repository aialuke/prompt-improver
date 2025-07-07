"""Individual health checker implementations for APES components.
PHASE 3: Health Check Consolidation - Component Checkers
"""

import time
from typing import Any, Dict

from .base import HealthChecker, HealthResult, HealthStatus

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
                error="Database credentials not configured"
            )
        
        try:
            start_time = time.time()
            async with get_session() as session:
                await session.execute("SELECT 1")
                response_time = (time.time() - start_time) * 1000
                
                # Check for long-running queries
                result = await session.execute("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND query_start < NOW() - INTERVAL '30 seconds'
                """)
                long_queries = result.scalar() or 0
                
                # Get active connections
                conn_result = await session.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                active_connections = conn_result.scalar() or 0
                
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
                        "active_connections": active_connections
                    }
                )
                
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="Database connection failed"
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
                    error="MCP server not configured"
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
                message=f"MCP server responding in {response_time:.1f}ms"
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="MCP server health check failed"
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
                    error="Analytics service not configured"
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
                details={
                    "data_points": len(result.get("trends", []))
                }
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="Analytics service check failed"
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
                    error="ML service not configured"
                )
            
            start_time = time.time()
            ml_service = await get_ml_service()
            response_time = (time.time() - start_time) * 1000
            
            return HealthResult(
                status=HealthStatus.HEALTHY,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML service available in {response_time:.1f}ms"
            )
            
        except Exception as e:
            return HealthResult(
                status=HealthStatus.WARNING,  # ML service is optional
                component=self.name,
                error=str(e),
                message="ML service unavailable (fallback to rule-based)"
            )


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
            
            message = f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_usage_percent:.1f}%, Disk {disk_usage_percent:.1f}%"
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
                    "warnings": warnings
                }
            )
            
        except ImportError:
            return HealthResult(
                status=HealthStatus.WARNING,
                component=self.name,
                message="psutil not available for system monitoring"
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message="System resource check failed"
            )
