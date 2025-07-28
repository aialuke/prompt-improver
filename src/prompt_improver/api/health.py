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
from fastapi.responses import JSONResponse, Response
import asyncpg
import aiohttp
from pathlib import Path

# Note: Using coredis instead of aioredis for better Python 3.13 compatibility
AIOREDIS_AVAILABLE = False

from ..core.config import get_config
from ..database.connection import get_session_context
from ..utils.redis_cache import redis_client
from ..performance.monitoring.performance_monitor import PerformanceMonitor

# ML Health Monitoring Imports
try:
    from ..ml.health.ml_health_monitor import get_ml_health_monitor
    from ..ml.health.drift_detector import get_drift_detector
    from ..ml.health.model_performance_tracker import get_performance_tracker
    from ..ml.health.resource_monitor import get_resource_monitor
    from ..ml.core.ml_integration import get_ml_service
    ML_HEALTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML health monitoring not available: {e}")
    ML_HEALTH_AVAILABLE = False
from ..cache.redis_health import get_redis_health_summary, RedisHealthMonitor
from ..database.health import get_database_health_monitor
from ..database.connection_pool_optimizer import get_connection_pool_optimizer

# Import unified health orchestration system
from ..monitoring.health_orchestrator import get_health_orchestrator, SystemHealthStatus

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
        
        # Initialize comprehensive database monitoring
        self.db_health_monitor = None
        self.pool_optimizer = None
    
    async def liveness_check(self) -> Dict[str, Any]:
        """
        Liveness probe - indicates if the application is running
        Should be lightweight and fast (<1s)
        """
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.startup_time),
            "version": get_config().environment.version,
            "environment": get_config().environment.environment
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
            "version": get_config().environment.version,
            "checks": checks,
            "system_metrics": system_metrics,
            "check_duration_ms": round(duration_ms, 2)
        }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic health"""
        try:
            start_time = time.time()
            
            # Test database connection
            async with get_session_context() as session:
                # Simple query to test connectivity
                result = await session.execute("SELECT 1")
                await session.commit()
                
                # Check connection pool status
                config = get_config()
                pool_info = {
                    "active_connections": 1,  # Using session, not raw connection
                    "max_connections": config.database.pool_max_size,
                    "pool_timeout": config.database.pool_timeout,
                    "pool_max_lifetime": config.database.pool_max_lifetime
                }
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "healthy": result is not None,
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
    
    async def comprehensive_database_health_check(self) -> Dict[str, Any]:
        """
        Comprehensive database health check with real PostgreSQL metrics
        
        Includes:
        - Connection pool metrics with age tracking
        - Query performance analysis with execution plans
        - Replication lag monitoring
        - Storage utilization and table bloat detection
        - Lock monitoring and deadlock detection
        - Cache hit rates and buffer analysis
        - Transaction metrics and longest transactions
        """
        try:
            start_time = time.time()
            
            # Initialize monitors if needed
            if self.db_health_monitor is None:
                self.db_health_monitor = get_database_health_monitor()
            
            if self.pool_optimizer is None:
                self.pool_optimizer = get_connection_pool_optimizer()
            
            # Collect comprehensive metrics
            logger.info("Starting comprehensive database health check")
            
            # Run all health monitoring components in parallel for efficiency
            results = await asyncio.gather(
                self.db_health_monitor.collect_comprehensive_metrics(),
                self.pool_optimizer.get_optimization_summary(),
                return_exceptions=True
            )
            
            comprehensive_metrics = results[0] if not isinstance(results[0], Exception) else None
            pool_optimization = results[1] if not isinstance(results[1], Exception) else None
            
            # Basic connectivity check
            basic_health = await self._check_database()
            
            collection_time = (time.time() - start_time) * 1000
            
            # Build comprehensive response
            response = {
                "healthy": basic_health.get("healthy", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_time_ms": round(collection_time, 2),
                "basic_connectivity": basic_health,
            }
            
            # Add comprehensive metrics if available
            if comprehensive_metrics and not isinstance(comprehensive_metrics, Exception):
                response["comprehensive_metrics"] = {
                    "health_score": comprehensive_metrics.health_score,
                    "connection_pool": comprehensive_metrics.connection_pool,
                    "query_performance": {
                        "slow_queries_count": comprehensive_metrics.query_performance.get("slow_queries", {}).get("count", 0),
                        "avg_query_time_ms": comprehensive_metrics.query_performance.get("performance_summary", {}).get("avg_query_time_ms", 0),
                        "cache_hit_ratio": comprehensive_metrics.query_performance.get("cache_performance", {}).get("overall_cache_hit_ratio_percent", 100)
                    },
                    "replication": {
                        "enabled": comprehensive_metrics.replication.get("replication_enabled", False),
                        "lag_seconds": comprehensive_metrics.replication.get("lag_seconds", 0),
                        "replica_count": comprehensive_metrics.replication.get("replica_count", 0)
                    },
                    "storage": {
                        "database_size_mb": comprehensive_metrics.storage.get("database_size_bytes", 0) / (1024 * 1024),
                        "bloated_tables": comprehensive_metrics.storage.get("bloat_metrics", {}).get("bloated_tables_count", 0),
                        "index_to_table_ratio": comprehensive_metrics.storage.get("index_to_table_ratio", 0)
                    },
                    "locks": {
                        "total_locks": comprehensive_metrics.locks.get("total_locks", 0),
                        "blocking_locks": comprehensive_metrics.locks.get("blocking_locks", 0),
                        "long_running_locks": comprehensive_metrics.locks.get("long_running_locks", 0)
                    },
                    "cache": {
                        "overall_hit_ratio_percent": comprehensive_metrics.cache.get("overall_cache_hit_ratio_percent", 100),
                        "buffer_cache_hit_ratio": comprehensive_metrics.cache.get("buffer_cache", {}).get("hit_ratio_percent", 100),
                        "index_cache_hit_ratio": comprehensive_metrics.cache.get("index_cache", {}).get("hit_ratio_percent", 100)
                    },
                    "transactions": {
                        "total_transactions": comprehensive_metrics.transactions.get("total_transactions", 0),
                        "commit_ratio_percent": comprehensive_metrics.transactions.get("commit_ratio_percent", 100),
                        "rollback_ratio_percent": comprehensive_metrics.transactions.get("rollback_ratio_percent", 0),
                        "long_running_count": len(comprehensive_metrics.transactions.get("long_running_transactions", []))
                    },
                    "issues": comprehensive_metrics.issues[:5],  # Top 5 issues
                    "recommendations": comprehensive_metrics.recommendations[:5]  # Top 5 recommendations
                }
                
                # Update overall health based on comprehensive score
                if comprehensive_metrics.health_score < 70:
                    response["healthy"] = False
                    response["health_status"] = "critical"
                elif comprehensive_metrics.health_score < 85:
                    response["health_status"] = "warning"
                else:
                    response["health_status"] = "healthy"
            
            # Add pool optimization metrics if available
            if pool_optimization and not isinstance(pool_optimization, Exception):
                response["connection_pool_optimization"] = {
                    "pool_state": pool_optimization.get("current_state", "unknown"),
                    "utilization_percent": pool_optimization.get("pool_metrics", {}).get("utilization_percent", 0),
                    "efficiency_score": pool_optimization.get("performance", {}).get("pool_efficiency_percent", 100),
                    "database_load_reduction_percent": pool_optimization.get("optimization", {}).get("database_load_reduction_percent", 0),
                    "connections_saved": pool_optimization.get("optimization", {}).get("connections_saved", 0),
                    "recommended_pool_size": pool_optimization.get("optimization", {}).get("recommended_pool_size", 0)
                }
            
            # Handle errors in comprehensive monitoring
            if isinstance(comprehensive_metrics, Exception):
                response["comprehensive_metrics_error"] = str(comprehensive_metrics)
                logger.warning(f"Comprehensive database metrics collection failed: {comprehensive_metrics}")
            
            if isinstance(pool_optimization, Exception):
                response["pool_optimization_error"] = str(pool_optimization)
                logger.warning(f"Pool optimization metrics collection failed: {pool_optimization}")
            
            logger.info(f"Comprehensive database health check completed in {collection_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Comprehensive database health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and comprehensive health"""
        try:
            # Use comprehensive Redis health monitoring
            return await get_redis_health_summary()
            
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
        """Comprehensive ML model health check with real-time monitoring"""
        start_time = time.time()
        
        try:
            config = get_config()
            
            # Check if ML model serving is enabled
            if not config.ml.model_serving_enabled:
                return {
                    "healthy": True,
                    "status": "disabled",
                    "message": "ML model serving is disabled",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "check_duration_ms": (time.time() - start_time) * 1000
                }
            
            # Check if ML health monitoring is available
            if not ML_HEALTH_AVAILABLE:
                return await self._fallback_ml_health_check(config, start_time)
            
            # Initialize comprehensive ML health monitoring
            health_results = {}
            overall_healthy = True
            
            # 1. Model Loading Status and Memory Metrics
            try:
                ml_health_monitor = await get_ml_health_monitor()
                system_health = await ml_health_monitor.get_system_health()
                all_models_health = await ml_health_monitor.get_all_models_health()
                
                health_results["model_loading"] = {
                    "system_healthy": system_health.get("healthy", False),
                    "health_score": system_health.get("health_score", 0.0),
                    "total_models_loaded": system_health.get("models", {}).get("total_loaded", 0),
                    "total_memory_mb": system_health.get("models", {}).get("total_memory_mb", 0.0),
                    "avg_memory_per_model": system_health.get("models", {}).get("memory_per_model_avg", 0.0)
                }
                
                # Individual model health details
                model_details = []
                for model_health in all_models_health:
                    model_details.append({
                        "model_id": model_health.model_id,
                        "model_type": model_health.model_type,
                        "status": model_health.status,
                        "memory_mb": model_health.memory_mb,
                        "total_predictions": model_health.total_predictions,
                        "success_rate": model_health.success_rate,
                        "error_rate": model_health.error_rate,
                        "version": model_health.version,
                        "last_accessed": model_health.last_accessed.isoformat() if model_health.last_accessed else None
                    })
                
                health_results["individual_models"] = model_details
                
                if not system_health.get("healthy", False):
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"ML health monitor check failed: {e}")
                health_results["model_loading"] = {"error": str(e), "healthy": False}
                overall_healthy = False
            
            # 2. Inference Latency Tracking (p50/p95/p99)
            try:
                performance_tracker = await get_performance_tracker()
                performance_summaries = await performance_tracker.get_all_models_performance()
                
                # Aggregate performance metrics
                total_requests = sum(s.get("throughput_summary", {}).get("total_requests", 0) for s in performance_summaries)
                avg_latencies = [s.get("latency_summary", {}).get("p95_ms", 0) for s in performance_summaries if s.get("latency_summary")]
                avg_success_rates = [s.get("quality_summary", {}).get("avg_success_rate", 0) for s in performance_summaries if s.get("quality_summary")]
                
                health_results["inference_performance"] = {
                    "total_requests_processed": total_requests,
                    "avg_p95_latency_ms": sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0,
                    "avg_success_rate": sum(avg_success_rates) / len(avg_success_rates) if avg_success_rates else 0.0,
                    "models_with_performance_data": len(performance_summaries),
                    "performance_details": performance_summaries
                }
                
                # Check performance thresholds
                if avg_latencies and max(avg_latencies) > 5000:  # >5 seconds
                    overall_healthy = False
                    health_results["inference_performance"]["alert"] = "High latency detected"
                    
            except Exception as e:
                logger.error(f"Performance tracker check failed: {e}")
                health_results["inference_performance"] = {"error": str(e), "healthy": False}
                overall_healthy = False
            
            # 3. Model Drift Detection
            try:
                drift_detector = await get_drift_detector()
                drift_statuses = await drift_detector.get_all_models_drift_status()
                
                # Analyze drift across all models
                models_with_drift = [s for s in drift_statuses if s.get("current_drift", {}) and s["current_drift"].get("drift_detected")]
                high_risk_models = [s for s in drift_statuses if s.get("risk_level") == "high"]
                
                health_results["model_drift"] = {
                    "total_models_monitored": len(drift_statuses),
                    "models_with_drift_detected": len(models_with_drift),
                    "high_risk_models": len(high_risk_models),
                    "drift_summaries": drift_statuses
                }
                
                # Alert on high drift
                if high_risk_models:
                    overall_healthy = False
                    health_results["model_drift"]["alert"] = f"{len(high_risk_models)} models have high drift risk"
                    
            except Exception as e:
                logger.error(f"Drift detector check failed: {e}")
                health_results["model_drift"] = {"error": str(e), "healthy": False}
            
            # 4. GPU/CPU Resource Utilization
            try:
                resource_monitor = await get_resource_monitor()
                resource_summary = await resource_monitor.get_resource_summary(hours=1)
                
                health_results["resource_utilization"] = {
                    "cpu_avg_percent": resource_summary.get("cpu_summary", {}).get("avg_percent", 0.0),
                    "memory_avg_percent": resource_summary.get("memory_summary", {}).get("avg_percent", 0.0),
                    "gpu_available": resource_summary.get("gpu_summary", {}).get("available", False),
                    "gpu_avg_utilization": resource_summary.get("gpu_summary", {}).get("avg_utilization_percent", 0.0),
                    "gpu_memory_percent": resource_summary.get("gpu_summary", {}).get("avg_memory_percent", 0.0),
                    "resource_health": resource_summary.get("resource_health", {}),
                    "alerts": resource_summary.get("alerts", [])
                }
                
                # Check resource health
                resource_health_status = resource_summary.get("resource_health", {}).get("status", "unknown")
                if resource_health_status in ["poor", "critical"]:
                    overall_healthy = False
                    health_results["resource_utilization"]["alert"] = f"Resource health: {resource_health_status}"
                    
            except Exception as e:
                logger.error(f"Resource monitor check failed: {e}")
                health_results["resource_utilization"] = {"error": str(e), "healthy": False}
            
            # 5. Model Version Tracking and Registry Integration
            try:
                ml_service = await get_ml_service()
                cache_stats = await ml_service.get_model_cache_stats()
                
                health_results["model_registry"] = {
                    "cache_healthy": cache_stats.get("status") == "success",
                    "total_cached_models": cache_stats.get("cache_stats", {}).get("total_models", 0),
                    "active_models": cache_stats.get("cache_stats", {}).get("active_models", 0),
                    "memory_utilization": cache_stats.get("cache_stats", {}).get("memory_utilization", 0.0),
                    "cache_recommendations": cache_stats.get("recommendations", [])
                }
                
                # Production deployment status (if enabled)
                if hasattr(ml_service, '_production_enabled') and ml_service._production_enabled:
                    production_deployments = await ml_service.list_production_deployments()
                    health_results["model_registry"]["production_deployments"] = len(production_deployments)
                    
            except Exception as e:
                logger.error(f"Model registry check failed: {e}")
                health_results["model_registry"] = {"error": str(e), "healthy": False}
            
            # 6. Storage and File System Health
            model_storage_path = config.ml.model_storage_path
            try:
                # Check model storage accessibility
                storage_accessible = model_storage_path.exists()
                model_files = []
                total_size_mb = 0.0
                
                if storage_accessible:
                    model_files = (
                        list(model_storage_path.glob("**/*.pkl")) +
                        list(model_storage_path.glob("**/*.joblib")) +
                        list(model_storage_path.glob("**/*.pt")) +
                        list(model_storage_path.glob("**/*.h5")) +
                        list(model_storage_path.glob("**/*.onnx"))
                    )
                    total_size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                
                health_results["storage"] = {
                    "path": str(model_storage_path),
                    "accessible": storage_accessible,
                    "model_files_count": len(model_files),
                    "total_size_mb": total_size_mb,
                    "cache_size_limit": config.ml.model_cache_size
                }
                
                if not storage_accessible:
                    overall_healthy = False
                    health_results["storage"]["alert"] = "Model storage path not accessible"
                    
            except Exception as e:
                logger.error(f"Storage check failed: {e}")
                health_results["storage"] = {"error": str(e), "healthy": False}
                overall_healthy = False
            
            # Overall health assessment
            check_duration_ms = (time.time() - start_time) * 1000
            
            # Generate recommendations based on health status
            recommendations = self._generate_ml_health_recommendations(health_results, overall_healthy)
            
            return {
                "healthy": overall_healthy,
                "status": "comprehensive_monitoring_enabled",
                "health_components": health_results,
                "config": {
                    "serving_enabled": config.ml.model_serving_enabled,
                    "warmup_enabled": config.ml.model_warmup_enabled,
                    "inference_timeout": config.ml.inference_timeout,
                    "batch_size": config.ml.batch_size,
                    "gpu_memory_fraction": config.ml.gpu_memory_fraction,
                    "cpu_threads": config.ml.cpu_threads
                },
                "recommendations": recommendations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "check_duration_ms": check_duration_ms
            }
            
        except Exception as e:
            logger.error(f"Comprehensive ML health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "check_duration_ms": (time.time() - start_time) * 1000
            }
    
    async def _fallback_ml_health_check(self, config, start_time: float) -> Dict[str, Any]:
        """Fallback ML health check when comprehensive monitoring is not available"""
        try:
            models_status = {}
            all_models_healthy = True
            
            # Check model storage path accessibility
            model_storage_path = config.ml.model_storage_path
            if not model_storage_path.exists():
                return {
                    "healthy": False,
                    "status": "fallback_mode",
                    "error": f"Model storage path not accessible: {model_storage_path}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "check_duration_ms": (time.time() - start_time) * 1000
                }
            
            # Check for actual model files
            model_files = list(model_storage_path.glob("**/*.pkl")) + \
                         list(model_storage_path.glob("**/*.joblib")) + \
                         list(model_storage_path.glob("**/*.pt")) + \
                         list(model_storage_path.glob("**/*.h5"))
            
            # Basic ML service check
            try:
                from ..ml.core.ml_integration import get_ml_service
                ml_service = await get_ml_service()
                cache_stats = await ml_service.get_model_cache_stats()
                
                models_status["model_service"] = {
                    "available": True,
                    "cache_status": cache_stats.get("status", "unknown")
                }
                
            except Exception as ml_error:
                logger.warning(f"ML service check failed: {ml_error}")
                models_status["model_service"] = {
                    "available": False,
                    "error": str(ml_error)
                }
                all_models_healthy = False
            
            # Storage check
            models_status["storage"] = {
                "path": str(model_storage_path),
                "accessible": model_storage_path.exists(),
                "model_files_count": len(model_files),
                "total_size_mb": sum(f.stat().st_size for f in model_files) / (1024 * 1024) if model_files else 0
            }
            
            return {
                "healthy": all_models_healthy and models_status["storage"]["accessible"],
                "status": "fallback_mode",
                "message": "ML health monitoring not available - using basic checks",
                "models": models_status,
                "config": {
                    "serving_enabled": config.ml.model_serving_enabled,
                    "inference_timeout": config.ml.inference_timeout,
                    "batch_size": config.ml.batch_size
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "check_duration_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error(f"Fallback ML health check failed: {e}")
            return {
                "healthy": False,
                "status": "fallback_error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "check_duration_ms": (time.time() - start_time) * 1000
            }
    
    def _generate_ml_health_recommendations(self, health_results: Dict[str, Any], overall_healthy: bool) -> List[str]:
        """Generate actionable recommendations based on ML health status"""
        recommendations = []
        
        if not overall_healthy:
            recommendations.append("⚠️ ML system health issues detected - review detailed metrics below")
        
        # Model loading recommendations
        model_loading = health_results.get("model_loading", {})
        if model_loading.get("total_models_loaded", 0) == 0:
            recommendations.append("📝 No models currently loaded - verify model deployment")
        elif model_loading.get("total_memory_mb", 0) > 2000:  # >2GB
            recommendations.append("💾 High model memory usage - consider model optimization")
        
        # Performance recommendations
        perf_data = health_results.get("inference_performance", {})
        if perf_data.get("avg_p95_latency_ms", 0) > 1000:  # >1 second
            recommendations.append("⚡ High inference latency detected - optimize model or infrastructure")
        if perf_data.get("avg_success_rate", 1.0) < 0.95:  # <95% success
            recommendations.append("🎯 Low inference success rate - investigate error patterns")
        
        # Drift recommendations
        drift_data = health_results.get("model_drift", {})
        if drift_data.get("models_with_drift_detected", 0) > 0:
            recommendations.append("📊 Model drift detected - consider retraining affected models")
        if drift_data.get("high_risk_models", 0) > 0:
            recommendations.append("🚨 High-risk drift models require immediate attention")
        
        # Resource recommendations
        resource_data = health_results.get("resource_utilization", {})
        if resource_data.get("cpu_avg_percent", 0) > 80:
            recommendations.append("🖥️ High CPU utilization - consider scaling or optimization")
        if resource_data.get("memory_avg_percent", 0) > 85:
            recommendations.append("💾 High memory utilization - monitor for memory leaks")
        if resource_data.get("gpu_available") and resource_data.get("gpu_avg_utilization", 0) > 90:
            recommendations.append("🎮 High GPU utilization - consider load balancing")
        
        # Registry recommendations
        registry_data = health_results.get("model_registry", {})
        cache_util = registry_data.get("memory_utilization", 0.0)
        if cache_util > 0.9:
            recommendations.append("🗃️ Model cache near capacity - consider cleanup or expansion")
        
        # Default recommendation for healthy systems
        if not recommendations:
            recommendations.append("✅ ML system health is optimal - continue monitoring")
        
        return recommendations
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API dependencies using comprehensive monitoring"""
        try:
            # Import the advanced external API monitor
            from ..monitoring.external_api_health import ExternalAPIHealthMonitor
            
            # Initialize monitor if not already cached
            if not hasattr(self, '_external_api_monitor'):
                self._external_api_monitor = ExternalAPIHealthMonitor()
            
            # Perform comprehensive health checks
            health_snapshots = await self._external_api_monitor.check_all_endpoints()
            
            # Convert to legacy format for compatibility while preserving new data
            external_apis = {}
            all_apis_healthy = True
            
            for endpoint_name, snapshot in health_snapshots.items():
                is_healthy = snapshot.status.value in ["healthy", "degraded"]
                if not is_healthy:
                    all_apis_healthy = False
                
                # Legacy format
                api_result = {
                    "available": is_healthy,
                    "status": snapshot.status.value,
                    "sla_compliance": snapshot.sla_compliance.value,
                    "last_check": snapshot.last_check.isoformat(),
                    
                    # Response details
                    "status_code": snapshot.current_status_code,
                    "response_time_ms": snapshot.current_response_time_ms,
                    "error": snapshot.current_error,
                    
                    # Performance metrics
                    "performance": {
                        "p50_ms": snapshot.response_times.get("p50", 0.0),
                        "p95_ms": snapshot.response_times.get("p95", 0.0),
                        "p99_ms": snapshot.response_times.get("p99", 0.0),
                        "availability": snapshot.availability,
                        "total_requests": snapshot.total_requests,
                        "successful_requests": snapshot.successful_requests,
                        "failed_requests": snapshot.failed_requests
                    },
                    
                    # Rate limiting
                    "rate_limit": snapshot.rate_limit_status or {},
                    
                    # Infrastructure health
                    "dns": snapshot.dns_status or {},
                    "ssl": snapshot.ssl_status or {},
                    
                    # Circuit breaker status
                    "circuit_breaker": {
                        "state": snapshot.circuit_breaker_state,
                        "metrics": snapshot.circuit_breaker_metrics
                    }
                }
                
                external_apis[endpoint_name] = api_result
            
            # Check for additional discovered dependencies
            discovered_deps = await self._external_api_monitor.discover_dependencies()
            
            # Internal service dependencies
            internal_services = {
                "mcp_server": {
                    "enabled": True,
                    "status": "healthy"
                },
                "external_api_monitor": {
                    "enabled": True,
                    "status": "healthy",
                    "discovered_dependencies": len(discovered_deps)
                }
            }
            
            # Overall summary
            total_apis = len(health_snapshots)
            healthy_count = sum(1 for s in health_snapshots.values() if s.status.value == "healthy")
            degraded_count = sum(1 for s in health_snapshots.values() if s.status.value == "degraded")
            unhealthy_count = sum(1 for s in health_snapshots.values() if s.status.value == "unhealthy")
            
            config = get_config()
            
            return {
                "healthy": all_apis_healthy,
                "external_apis": external_apis,
                "internal_services": internal_services,
                "summary": {
                    "total_apis": total_apis,
                    "healthy_apis": healthy_count,
                    "degraded_apis": degraded_count,
                    "unhealthy_apis": unhealthy_count,
                    "overall_health_percentage": (healthy_count + degraded_count) / max(1, total_apis) * 100
                },
                "config": {
                    "advanced_monitoring_enabled": True,
                    "circuit_breaker_enabled": True,
                    "sla_tracking_enabled": True,
                    "dns_monitoring_enabled": True,
                    "ssl_monitoring_enabled": True,
                    "rate_limit_monitoring_enabled": True,
                    "historical_tracking_enabled": True,
                    "ml_model_timeout": config.health.ml_model_health_timeout,
                    "mlflow_tracking_enabled": bool(config.ml.mlflow_tracking_uri)
                },
                "total_apis_checked": total_apis,
                "discovered_dependencies": len(discovered_deps),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except ImportError as e:
            logger.warning(f"Advanced external API monitoring not available: {e}")
            # Fallback to simple checks
            return await self._check_external_apis_fallback()
        except Exception as e:
            logger.error(f"External API health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "advanced_monitoring_available": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _check_external_apis_fallback(self) -> Dict[str, Any]:
        """Fallback external API check if advanced monitoring fails"""
        try:
            config = get_config()
            external_apis = {}
            all_apis_healthy = True
            
            # Simple endpoint checks
            api_endpoints = {
                "openai_api": {
                    "url": "https://api.openai.com/v1/models",
                    "timeout": 10,
                    "expected_status": [200, 401]
                },
                "huggingface_api": {
                    "url": "https://huggingface.co/api/models", 
                    "timeout": 10,
                    "expected_status": [200]
                }
            }
            
            if config.ml.mlflow_tracking_uri:
                mlflow_url = config.ml.mlflow_tracking_uri.rstrip('/') + '/api/2.0/mlflow/experiments/list'
                api_endpoints["mlflow_tracking"] = {
                    "url": mlflow_url,
                    "timeout": 5,
                    "expected_status": [200, 401, 403]
                }
            
            timeout = aiohttp.ClientTimeout(total=config.health.ml_model_health_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for api_name, api_config in api_endpoints.items():
                    start_time = time.time()
                    
                    try:
                        async with session.get(
                            api_config["url"],
                            timeout=aiohttp.ClientTimeout(total=api_config["timeout"])
                        ) as response:
                            response_time_ms = (time.time() - start_time) * 1000
                            
                            is_healthy = response.status in api_config["expected_status"]
                            if not is_healthy:
                                all_apis_healthy = False
                            
                            external_apis[api_name] = {
                                "available": is_healthy,
                                "status_code": response.status,
                                "response_time_ms": round(response_time_ms, 2),
                                "url": api_config["url"],
                                "last_check": datetime.now(timezone.utc).isoformat(),
                                "monitoring_mode": "fallback"
                            }
                    
                    except asyncio.TimeoutError:
                        external_apis[api_name] = {
                            "available": False,
                            "error": "timeout",
                            "timeout_seconds": api_config["timeout"],
                            "url": api_config["url"],
                            "last_check": datetime.now(timezone.utc).isoformat(),
                            "monitoring_mode": "fallback"
                        }
                        all_apis_healthy = False
                    
                    except Exception as api_error:
                        external_apis[api_name] = {
                            "available": False,
                            "error": str(api_error),
                            "url": api_config["url"],
                            "last_check": datetime.now(timezone.utc).isoformat(),
                            "monitoring_mode": "fallback"
                        }
                        all_apis_healthy = False
            
            internal_services = {
                "mcp_server": {
                    "enabled": True,
                    "status": "healthy"
                }
            }
            
            return {
                "healthy": all_apis_healthy,
                "external_apis": external_apis,
                "internal_services": internal_services,
                "config": {
                    "advanced_monitoring_enabled": False,
                    "fallback_mode": True,
                    "ml_model_timeout": config.health.ml_model_health_timeout,
                    "mlflow_tracking_enabled": bool(config.ml.mlflow_tracking_uri)
                },
                "total_apis_checked": len(api_endpoints),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback external API health check failed: {e}")
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
    """Main health check endpoint - unified comprehensive health status"""
    try:
        # Use unified health orchestrator for comprehensive monitoring
        orchestrator = await get_health_orchestrator()
        health_snapshot = await orchestrator.execute_comprehensive_health_check()
        
        # Convert to legacy format for backward compatibility
        result = {
            "status": health_snapshot.overall_status.value,
            "healthy": health_snapshot.overall_status in [SystemHealthStatus.HEALTHY, SystemHealthStatus.DEGRADED],
            "health_score": health_snapshot.overall_health_score,
            "timestamp": health_snapshot.timestamp.isoformat(),
            "uptime_seconds": int((datetime.now(timezone.utc) - datetime.fromtimestamp(_health_state["startup_time"], tz=timezone.utc)).total_seconds()),
            "version": get_config().environment.version,
            "execution_time_ms": health_snapshot.execution_time_ms,
            
            # Enhanced unified monitoring data
            "unified_monitoring": {
                "enabled": True,
                "orchestrator_version": "2025.1.0",
                "component_count": len(health_snapshot.component_results),
                "dependency_chain_status": health_snapshot.dependency_chain_status,
                "cascade_failures": health_snapshot.cascade_failures,
                "active_alerts": len(health_snapshot.active_alerts),
                "circuit_breaker_states": health_snapshot.circuit_breaker_states
            },
            
            # Component results
            "checks": {
                name: {
                    "healthy": result.status == SystemHealthStatus.HEALTHY,
                    "status": result.status.value,
                    "health_score": result.health_score,
                    "response_time_ms": result.response_time_ms,
                    "message": result.message,
                    "error": result.error,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for name, result in health_snapshot.component_results.items()
            },
            
            # Performance aggregation
            "performance_metrics": health_snapshot.aggregated_metrics,
            
            # Recommendations
            "recommendations": health_snapshot.recommendations
        }
        
        # Determine HTTP status code
        if health_snapshot.overall_status == SystemHealthStatus.HEALTHY:
            status_code = status.HTTP_200_OK
        elif health_snapshot.overall_status == SystemHealthStatus.DEGRADED:
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(content=result, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Unified health check failed: {e}", exc_info=True)
        
        # Fallback to legacy health checker
        try:
            result = await health_checker.deep_health_check()
            result["unified_monitoring"] = {
                "enabled": False,
                "fallback_mode": True,
                "error": str(e)
            }
            status_code = status.HTTP_200_OK if result["healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
            return JSONResponse(content=result, status_code=status_code)
        except Exception as fallback_error:
            logger.error(f"Fallback health check also failed: {fallback_error}")
            return JSONResponse(
                content={
                    "status": "error",
                    "healthy": False,
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "unified_monitoring": {"enabled": False, "error": str(e)},
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

@health_router.get("/redis")
async def redis_comprehensive_health():
    """Comprehensive Redis health check endpoint with detailed metrics"""
    try:
        monitor = RedisHealthMonitor()
        result = await monitor.collect_all_metrics()
        
        # Determine HTTP status code based on Redis health
        redis_status = result.get("status", "failed")
        if redis_status in ["healthy", "warning"]:
            status_code = status.HTTP_200_OK
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(content=result, status_code=status_code)
    
    except Exception as e:
        logger.error(f"Comprehensive Redis health check failed: {e}")
        return JSONResponse(
            content={
                "status": "failed",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/redis/memory")
async def redis_memory_health():
    """Redis memory-specific health check"""
    try:
        monitor = RedisHealthMonitor()
        await monitor._collect_memory_metrics()
        
        memory_data = monitor._memory_metrics_to_dict()
        memory_status = monitor.memory_metrics.get_status()
        
        status_code = status.HTTP_200_OK if memory_status.value != "failed" else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content={
                "status": memory_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory": memory_data
            },
            status_code=status_code
        )
    
    except Exception as e:
        logger.error(f"Redis memory health check failed: {e}")
        return JSONResponse(
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/redis/performance")
async def redis_performance_health():
    """Redis performance-specific health check"""
    try:
        monitor = RedisHealthMonitor()
        await monitor._collect_performance_metrics()
        
        performance_data = monitor._performance_metrics_to_dict()
        performance_status = monitor.performance_metrics.get_status()
        
        status_code = status.HTTP_200_OK if performance_status.value != "failed" else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content={
                "status": performance_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "performance": performance_data
            },
            status_code=status_code
        )
    
    except Exception as e:
        logger.error(f"Redis performance health check failed: {e}")
        return JSONResponse(
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/redis/slowlog")
async def redis_slowlog_analysis():
    """Redis slow log analysis endpoint"""
    try:
        monitor = RedisHealthMonitor()
        await monitor._collect_slowlog_metrics()
        
        slowlog_data = monitor._slowlog_metrics_to_dict()
        slowlog_status = monitor.slowlog_metrics.get_status()
        
        status_code = status.HTTP_200_OK if slowlog_status.value != "failed" else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content={
                "status": slowlog_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "slowlog": slowlog_data
            },
            status_code=status_code
        )
    
    except Exception as e:
        logger.error(f"Redis slow log analysis failed: {e}")
        return JSONResponse(
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )



@health_router.get("/database/comprehensive")
async def comprehensive_database_health():
    """
    Comprehensive database health check with real PostgreSQL metrics
    
    Provides detailed analysis including:
    - Connection pool metrics with age tracking
    - Query performance analysis with execution plans
    - Replication lag monitoring (if applicable)
    - Storage utilization and table bloat detection
    - Lock monitoring and deadlock detection
    - Cache hit rates and buffer analysis
    - Transaction metrics and longest transactions
    - Pool optimization recommendations
    """
    try:
        result = await health_checker.comprehensive_database_health_check()
        
        # Determine HTTP status code based on health
        if not result.get("healthy", False):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif result.get("health_status") == "critical":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif result.get("health_status") == "warning":
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_200_OK
        
        return JSONResponse(content=result, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Comprehensive database health check failed: {e}")
        return JSONResponse(
            content={
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/database/pool")
async def database_connection_pool_health():
    """
    Detailed connection pool health and optimization metrics
    """
    try:
        pool_optimizer = get_connection_pool_optimizer()
        pool_summary = await pool_optimizer.get_optimization_summary()
        
        # Determine status based on pool health
        pool_utilization = pool_summary.get("pool_metrics", {}).get("utilization_percent", 0)
        efficiency_score = pool_summary.get("performance", {}).get("pool_efficiency_percent", 100)
        
        if pool_utilization > 95 or efficiency_score < 60:
            health_status = "critical"
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif pool_utilization > 80 or efficiency_score < 80:
            health_status = "warning"
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            health_status = "healthy"
            status_code = status.HTTP_200_OK
        
        response = {
            "healthy": health_status != "critical",
            "status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pool_metrics": pool_summary
        }
        
        return JSONResponse(content=response, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Database connection pool health check failed: {e}")
        return JSONResponse(
            content={
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/database/query-performance")
async def database_query_performance():
    """
    Database query performance analysis
    """
    try:
        from ..database.health.query_performance_analyzer import QueryPerformanceAnalyzer
        
        analyzer = QueryPerformanceAnalyzer()
        performance_summary = await analyzer.get_query_performance_summary()
        
        # Determine status based on performance
        perf_status = performance_summary.get("status", "unknown")
        
        if perf_status == "critical":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif perf_status == "warning":
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_200_OK
        
        return JSONResponse(content=performance_summary, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Database query performance check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/database/bloat")
async def database_bloat_analysis():
    """
    Database table and index bloat analysis
    """
    try:
        from ..database.health.table_bloat_detector import TableBloatDetector
        
        detector = TableBloatDetector()
        bloat_summary = await detector.get_bloat_summary()
        
        # Determine status based on bloat health
        bloat_status = bloat_summary.get("status", "unknown")
        
        if bloat_status == "critical":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif bloat_status == "warning":
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_200_OK
        
        return JSONResponse(content=bloat_summary, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Database bloat analysis failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/database/indexes")
async def database_index_health():
    """
    Database index health assessment
    """
    try:
        from ..database.health.index_health_assessor import IndexHealthAssessor
        
        assessor = IndexHealthAssessor()
        index_summary = await assessor.get_index_health_summary()
        
        # Determine status based on index health
        index_status = index_summary.get("status", "unknown")
        
        if index_status == "critical":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif index_status == "warning":
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_200_OK
        
        return JSONResponse(content=index_summary, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Database index health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

# Unified Health Orchestration Endpoints
@health_router.get("/dashboard")
async def unified_health_dashboard():
    """
    Unified Health Dashboard - Complete system health overview
    
    Provides comprehensive system health information including:
    - Overall system health score (0-100)
    - Individual component health details
    - Dependency chain analysis
    - Performance metrics aggregation
    - Circuit breaker states
    - Active alerts and recommendations
    - Historical trends
    """
    try:
        orchestrator = await get_health_orchestrator()
        dashboard_data = await orchestrator.get_health_dashboard()
        
        # Determine HTTP status code
        overall_status = dashboard_data.get("overall_status", "failed")
        if overall_status == "healthy":
            status_code = status.HTTP_200_OK
        elif overall_status == "degraded":
            status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(content=dashboard_data, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Unified health dashboard failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unified_monitoring": {"enabled": False, "error": str(e)}
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/orchestrator/status")
async def orchestrator_status():
    """
    Health Orchestrator Status - System orchestration health
    
    Provides information about the health orchestration system itself:
    - Orchestrator component status
    - Execution performance metrics
    - Dependency management status
    - Circuit breaker coordination status
    """
    try:
        orchestrator = await get_health_orchestrator()
        
        # Get component dependencies information
        dependencies_info = orchestrator.get_component_dependencies()
        
        # Get execution history
        execution_history = list(orchestrator._execution_history)
        
        # Calculate orchestrator health metrics
        if execution_history:
            recent_executions = execution_history[-10:]  # Last 10 executions
            avg_execution_time = sum(e["execution_time_ms"] for e in recent_executions) / len(recent_executions)
            success_rate = len([e for e in recent_executions if e["overall_status"] in ["healthy", "degraded"]]) / len(recent_executions)
        else:
            avg_execution_time = 0.0
            success_rate = 0.0
        
        orchestrator_health = {
            "status": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "critical",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # Performance metrics
            "performance": {
                "avg_execution_time_ms": avg_execution_time,
                "success_rate": success_rate,
                "total_executions": len(execution_history),
                "last_execution_time": orchestrator._last_execution_time
            },
            
            # Component management
            "component_management": {
                "total_components": len(dependencies_info["components"]),
                "dependency_graph_size": len(dependencies_info["dependency_graph"]),
                "execution_groups": len(dependencies_info["execution_order"])
            },
            
            # Sub-system status
            "subsystems": {
                "health_score_calculator": {"status": "active"},
                "dependency_manager": {"status": "active"},
                "performance_aggregator": {"status": "active"},
                "alerting_manager": {"status": "active"},
                "circuit_breaker_coordinator": {"status": "active"}
            },
            
            # Monitoring system availability
            "monitoring_systems": {
                "enhanced_health_service": orchestrator._enhanced_health_service is not None,
                "ml_health_manager": orchestrator._ml_health_manager is not None,
                "external_api_monitor": orchestrator._external_api_monitor is not None,
                "redis_health_monitor": orchestrator._redis_health_monitor is not None,
                "database_health_monitor": orchestrator._database_health_monitor is not None
            },
            
            # Recent execution history
            "execution_history": execution_history[-5:],  # Last 5 executions
            
            # Configuration
            "configuration": dependencies_info
        }
        
        status_code = status.HTTP_200_OK if orchestrator_health["status"] == "healthy" else status.HTTP_206_PARTIAL_CONTENT
        return JSONResponse(content=orchestrator_health, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Orchestrator status check failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus Metrics Endpoint - Monitoring integration
    
    Returns Prometheus-formatted metrics for external monitoring systems.
    Includes health scores, component status, circuit breaker states, etc.
    """
    try:
        orchestrator = await get_health_orchestrator()
        metrics_data = orchestrator.get_prometheus_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Prometheus metrics export failed: {e}")
        error_metrics = f"""# Health orchestrator metrics export failed
# ERROR: {e}
# TIMESTAMP: {datetime.now(timezone.utc).isoformat()}
health_orchestrator_export_error{{error="{e}"}} 1
"""
        return Response(
            content=error_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@health_router.post("/circuit-breakers/reset")
async def reset_circuit_breakers():
    """
    Reset Circuit Breakers - Emergency recovery operation
    
    Resets all circuit breakers across the system. Use this endpoint
    for emergency recovery when services are healthy but circuit breakers
    are preventing normal operation.
    """
    try:
        orchestrator = await get_health_orchestrator()
        result = await orchestrator.reset_circuit_breakers()
        
        status_code = status.HTTP_200_OK if result["status"] == "success" else status.HTTP_500_INTERNAL_SERVER_ERROR
        return JSONResponse(content=result, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Failed to reset circuit breakers: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@health_router.get("/dependencies")
async def component_dependencies():
    """
    Component Dependencies - Dependency graph and execution order
    
    Provides information about component dependencies:
    - Component configurations
    - Dependency graph visualization data
    - Execution order for health checks
    - Dependency levels and weights
    """
    try:
        orchestrator = await get_health_orchestrator()
        dependencies_data = orchestrator.get_component_dependencies()
        
        return JSONResponse(content=dependencies_data, status_code=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Component dependencies endpoint failed: {e}")
        return JSONResponse(
            content={
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@health_router.get("/component/{component_name}")
async def individual_component_health(component_name: str):
    """
    Individual Component Health - Detailed health for specific component
    
    Provides detailed health information for a specific component including:
    - Component health status and score
    - Response time metrics
    - Sub-component checks
    - Circuit breaker status
    - Historical performance data
    """
    try:
        # First try the unified orchestrator
        orchestrator = await get_health_orchestrator()
        dashboard_data = await orchestrator.get_health_dashboard()
        
        component_data = dashboard_data.get("components", {}).get(component_name)
        
        if component_data:
            # Enhance with circuit breaker information
            circuit_breaker_data = dashboard_data.get("circuit_breakers", {}).get(component_name, {})
            component_data["circuit_breaker"] = circuit_breaker_data
            
            # Add dependency information
            dependencies_info = orchestrator.get_component_dependencies()
            component_config = dependencies_info.get("components", {}).get(component_name, {})
            component_data["configuration"] = component_config
            
            # Determine status code
            component_status = component_data.get("status", "failed")
            if component_status == "healthy":
                status_code = status.HTTP_200_OK
            elif component_status == "degraded":
                status_code = status.HTTP_206_PARTIAL_CONTENT
            else:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            
            return JSONResponse(content=component_data, status_code=status_code)
        
        # Fallback to legacy health service if component not found in orchestrator
        legacy_result = await health_checker.run_specific_check(component_name)
        if legacy_result.status.value != "failed":
            legacy_data = {
                "status": legacy_result.status.value,
                "message": legacy_result.message,
                "error": legacy_result.error,
                "timestamp": legacy_result.timestamp.isoformat() if legacy_result.timestamp else None,
                "response_time_ms": getattr(legacy_result, 'response_time_ms', None),
                "metadata": getattr(legacy_result, 'metadata', {}),
                "legacy_mode": True
            }
            
            status_code = status.HTTP_200_OK if legacy_result.status.value == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
            return JSONResponse(content=legacy_data, status_code=status_code)
        
        # Component not found
        return JSONResponse(
            content={
                "error": f"Component '{component_name}' not found",
                "available_components": list(dashboard_data.get("components", {}).keys()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_404_NOT_FOUND
        )
        
    except Exception as e:
        logger.error(f"Individual component health check failed for {component_name}: {e}")
        return JSONResponse(
            content={
                "error": str(e),
                "component": component_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

