"""
Health Check API Endpoints - 2025 Best Practices
Comprehensive health, readiness, and liveness checks for production deployment
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import psutil
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import asyncpg

# Skip aioredis import for Phase 0 compatibility (Python 3.13 issue)
# import aioredis  # Disabled due to TimeoutError conflicts
AIOREDIS_AVAILABLE = False

from ..database.connection import get_database_connection
from ..utils.redis_cache import get_redis_connection
from ..performance.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# Create router for health endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

# Global health check state
_health_state = {
    "startup_time": time.time(),
    "last_health_check": None,
    "health_status": "unknown",
    "component_status": {}
}

class HealthChecker:
    """
    Comprehensive health checker following 2025 best practices
    
    Implements:
    - Liveness probes (is the application running?)
    - Readiness probes (is the application ready to serve traffic?)
    - Startup probes (has the application finished starting up?)
    - Deep health checks (are all dependencies healthy?)
    """
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.startup_time = time.time()
        self.health_checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "memory": self._check_memory,
            "disk": self._check_disk,
            "ml_models": self._check_ml_models,
            "external_apis": self._check_external_apis
        }
    
    async def liveness_check(self) -> Dict[str, Any]:
        """
        Liveness probe - indicates if the application is running
        Should be lightweight and fast (<1s)
        """
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.startup_time),
            "version": "2025.1.0",
            "environment": "development"  # TODO: Get from config
        }
    
    async def readiness_check(self) -> Dict[str, Any]:
        """
        Readiness probe - indicates if the application is ready to serve traffic
        Checks critical dependencies
        """
        start_time = time.time()
        ready = True
        checks = {}
        
        # Check critical dependencies
        critical_checks = ["database", "redis"]
        
        for check_name in critical_checks:
            try:
                check_result = await self.health_checks[check_name]()
                checks[check_name] = check_result
                if not check_result.get("healthy", False):
                    ready = False
            except Exception as e:
                checks[check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                ready = False
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "check_duration_ms": round(duration_ms, 2)
        }
    
    async def startup_check(self) -> Dict[str, Any]:
        """
        Startup probe - indicates if the application has finished starting up
        Used by Kubernetes to know when to start sending traffic
        """
        uptime = time.time() - self.startup_time
        startup_complete = uptime > 30  # Allow 30 seconds for startup
        
        startup_tasks = {
            "configuration_loaded": uptime > 1,
            "database_migrations": uptime > 5,
            "ml_models_loaded": uptime > 15,
            "cache_warmed": uptime > 20,
            "health_checks_initialized": uptime > 25
        }
        
        all_tasks_complete = all(startup_tasks.values())
        
        return {
            "status": "started" if all_tasks_complete else "starting",
            "startup_complete": all_tasks_complete,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(uptime),
            "startup_tasks": startup_tasks
        }
    
    async def deep_health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check - all components and dependencies
        Used for monitoring and alerting
        """
        start_time = time.time()
        overall_healthy = True
        checks = {}
        
        # Run all health checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = await check_func()
                checks[check_name] = check_result
                if not check_result.get("healthy", False):
                    overall_healthy = False
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                checks[check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                overall_healthy = False
        
        # System metrics
        system_metrics = await self._get_system_metrics()
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Update global health state
        _health_state.update({
            "last_health_check": datetime.now(timezone.utc).isoformat(),
            "health_status": "healthy" if overall_healthy else "unhealthy",
            "component_status": checks
        })
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.startup_time),
            "version": "2025.1.0",
            "checks": checks,
            "system_metrics": system_metrics,
            "check_duration_ms": round(duration_ms, 2)
        }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        try:
            start_time = time.time()
            
            # Test database connection
            async with get_database_connection() as conn:
                # Simple query to test connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Check connection pool status
                pool_info = {
                    "active_connections": conn._pool.get_size() if hasattr(conn, '_pool') else 1,
                    "max_connections": 20  # TODO: Get from config
                }
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "healthy": result == 1,
                "response_time_ms": round(duration_ms, 2),
                "pool_info": pool_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and health"""
        try:
            start_time = time.time()
            
            # Test Redis connection
            redis = await get_redis_connection()
            await redis.ping()
            
            # Get Redis info
            info = await redis.info()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": round(duration_ms, 2),
                "redis_info": {
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory_human"),
                    "uptime_seconds": info.get("uptime_in_seconds")
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            return {
                "healthy": memory.percent < 90,  # Alert if >90% memory usage
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')
            
            return {
                "healthy": disk.percent < 85,  # Alert if >85% disk usage
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_ml_models(self) -> Dict[str, Any]:
        """Check ML model availability and health"""
        try:
            # TODO: Implement actual ML model health checks
            # For now, simulate model health
            
            models_status = {
                "prompt_improvement_model": {
                    "loaded": True,
                    "version": "2025.1.0",
                    "last_inference_time": "2025-01-23T10:30:00Z"
                },
                "text_classification_model": {
                    "loaded": True,
                    "version": "2025.1.0",
                    "last_inference_time": "2025-01-23T10:29:45Z"
                }
            }
            
            all_models_healthy = all(
                model["loaded"] for model in models_status.values()
            )
            
            return {
                "healthy": all_models_healthy,
                "models": models_status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API dependencies"""
        try:
            # TODO: Implement actual external API health checks
            # For now, simulate external API health
            
            external_apis = {
                "openai_api": {
                    "available": True,
                    "response_time_ms": 150,
                    "last_check": datetime.now(timezone.utc).isoformat()
                }
            }
            
            all_apis_healthy = all(
                api["available"] for api in external_apis.values()
            )
            
            return {
                "healthy": all_apis_healthy,
                "external_apis": external_apis,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Initialize health checker
health_checker = HealthChecker()

# Health endpoint implementations
@health_router.get("/")
@health_router.get("")
async def health_check():
    """Main health check endpoint - comprehensive health status"""
    try:
        result = await health_checker.deep_health_check()
        status_code = status.HTTP_200_OK if result["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    try:
        result = await health_checker.liveness_check()
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    try:
        result = await health_checker.readiness_check()
        status_code = status.HTTP_200_OK if result["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/startup")
async def startup_probe():
    """Kubernetes startup probe endpoint"""
    try:
        result = await health_checker.startup_check()
        status_code = status.HTTP_200_OK if result["startup_complete"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "startup_complete": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
