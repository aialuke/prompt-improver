"""
ML Model Health Monitor - 2025 Best Practices

Comprehensive ML model health monitoring with real-time metrics collection,
performance tracking, and resource utilization monitoring.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

# import numpy as np  # Converted to lazy loading
from ...core.utils.lazy_ml_loader import get_numpy

from ...utils.datetime_utils import aware_utc_now
from prompt_improver.core.utils.lazy_ml_loader import get_numpy

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthMetrics:
    """Model health metrics snapshot"""
    model_id: str
    model_type: str
    status: str
    memory_mb: float
    last_inference_ms: float | None
    total_predictions: int
    success_rate: float
    error_rate: float
    version: str | None = None
    loaded_at: datetime | None = None
    last_accessed: datetime | None = None
    
    # Performance metrics
    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_p99: float | None = None
    
    # Resource usage
    cpu_usage_percent: float | None = None
    gpu_memory_mb: float | None = None
    gpu_utilization_percent: float | None = None


@dataclass
class InferenceMetrics:
    """Real-time inference performance metrics"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_types: dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count
    
    def get_latency_percentiles(self) -> tuple[float, float, float]:
        """Get p50, p95, p99 latency percentiles"""
        if not self.latency_samples:
            return 0.0, 0.0, 0.0
        
        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)
        
        p50_idx = int(n * 0.5)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        return (
            sorted_samples[p50_idx] if p50_idx < n else 0.0,
            sorted_samples[p95_idx] if p95_idx < n else 0.0,
            sorted_samples[p99_idx] if p99_idx < n else 0.0,
        )


class MLHealthMonitor:
    """
    Comprehensive ML model health monitoring system.
    
    Tracks model loading status, inference performance, resource utilization,
    and provides real-time health assessments.
    """
    
    def __init__(self, max_history_size: int = 10000):
        self._lock = Lock()
        self.max_history_size = max_history_size
        
        # Model registry and status tracking
        self._loaded_models: dict[str, dict[str, Any]] = {}
        self._model_metrics: dict[str, InferenceMetrics] = defaultdict(InferenceMetrics)
        
        # Performance history
        self._performance_history: deque = deque(maxlen=max_history_size)
        
        # Resource monitoring
        self._last_resource_check = 0.0
        self._resource_cache_ttl = 5.0  # 5-second cache
        self._cached_resources: dict[str, Any] | None = None
        
        # GPU availability detection
        self._gpu_available: bool | None = None
        self._gpu_utils = self._initialize_gpu_monitoring()
        
        logger.info("ML Health Monitor initialized")
    
    def _initialize_gpu_monitoring(self) -> Any | None:
        """Initialize GPU monitoring utilities with graceful fallback"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            logger.info("GPU monitoring enabled via pynvml")
            return pynvml
        except (ImportError, Exception) as e:
            logger.info(f"GPU monitoring not available: {e}. Using CPU-only monitoring.")
            self._gpu_available = False
            return None
    
    async def register_model(
        self, 
        model_id: str, 
        model: Any, 
        model_type: str = "unknown",
        version: str | None = None,
        memory_mb: float | None = None
    ) -> bool:
        """Register a model for health monitoring"""
        try:
            with self._lock:
                # Estimate memory usage if not provided
                if memory_mb is None:
                    memory_mb = self._estimate_model_memory(model)
                
                model_info = {
                    "model": model,
                    "model_type": model_type,
                    "version": version,
                    "memory_mb": memory_mb,
                    "loaded_at": aware_utc_now(),
                    "last_accessed": aware_utc_now(),
                    "status": "loaded",
                    "access_count": 0
                }
                
                self._loaded_models[model_id] = model_info
                
                # Initialize metrics for this model
                if model_id not in self._model_metrics:
                    self._model_metrics[model_id] = InferenceMetrics()
                
                logger.info("Registered model {model_id} for health monitoring (%.1fMB)", memory_mb)
                return True
                
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    async def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from health monitoring"""
        try:
            with self._lock:
                if model_id in self._loaded_models:
                    del self._loaded_models[model_id]
                    logger.info(f"Unregistered model {model_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to unregister model {model_id}: {e}")
            return False
    
    async def record_inference(
        self,
        model_id: str,
        latency_ms: float,
        success: bool,
        error_type: str | None = None
    ) -> None:
        """Record inference metrics for a model"""
        try:
            with self._lock:
                # Update model access info
                if model_id in self._loaded_models:
                    self._loaded_models[model_id]["last_accessed"] = aware_utc_now()
                    self._loaded_models[model_id]["access_count"] += 1
                
                # Update inference metrics
                metrics = self._model_metrics[model_id]
                metrics.request_count += 1
                metrics.total_latency_ms += latency_ms
                metrics.latency_samples.append(latency_ms)
                
                if success:
                    metrics.success_count += 1
                else:
                    metrics.error_count += 1
                    if error_type:
                        metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
                
                # Add to performance history
                self._performance_history.append({
                    "timestamp": aware_utc_now(),
                    "model_id": model_id,
                    "latency_ms": latency_ms,
                    "success": success,
                    "error_type": error_type
                })
                
        except Exception as e:
            logger.error(f"Failed to record inference for {model_id}: {e}")
    
    async def get_model_health(self, model_id: str) -> ModelHealthMetrics | None:
        """Get comprehensive health metrics for a specific model"""
        try:
            with self._lock:
                if model_id not in self._loaded_models:
                    return None
                
                model_info = self._loaded_models[model_id]
                metrics = self._model_metrics[model_id]
                
                # Get latency percentiles
                p50, p95, p99 = metrics.get_latency_percentiles()
                
                # Get resource usage
                resources = await self._get_resource_metrics()
                
                return ModelHealthMetrics(
                    model_id=model_id,
                    model_type=model_info["model_type"],
                    status=model_info["status"],
                    memory_mb=model_info["memory_mb"],
                    last_inference_ms=metrics.latency_samples[-1] if metrics.latency_samples else None,
                    total_predictions=metrics.request_count,
                    success_rate=metrics.success_rate,
                    error_rate=metrics.error_rate,
                    version=model_info.get("version"),
                    loaded_at=model_info["loaded_at"],
                    last_accessed=model_info["last_accessed"],
                    latency_p50=p50,
                    latency_p95=p95,
                    latency_p99=p99,
                    cpu_usage_percent=resources.get("cpu_percent"),
                    gpu_memory_mb=resources.get("gpu_memory_mb"),
                    gpu_utilization_percent=resources.get("gpu_utilization")
                )
                
        except Exception as e:
            logger.error(f"Failed to get health for model {model_id}: {e}")
            return None
    
    async def get_all_models_health(self) -> list[ModelHealthMetrics]:
        """Get health metrics for all registered models"""
        health_metrics = []
        
        with self._lock:
            model_ids = list(self._loaded_models.keys())
        
        for model_id in model_ids:
            health = await self.get_model_health(model_id)
            if health:
                health_metrics.append(health)
        
        return health_metrics
    
    async def get_system_health(self) -> dict[str, Any]:
        """Get overall ML system health status"""
        try:
            with self._lock:
                total_models = len(self._loaded_models)
                total_memory_mb = sum(info["memory_mb"] for info in self._loaded_models.values())
                
                # Calculate aggregate metrics
                total_requests = sum(metrics.request_count for metrics in self._model_metrics.values())
                total_successes = sum(metrics.success_count for metrics in self._model_metrics.values())
                total_errors = sum(metrics.error_count for metrics in self._model_metrics.values())
                
            # Get resource metrics
            resources = await self._get_resource_metrics()
            
            # Calculate health score
            health_score = await self._calculate_system_health_score()
            
            return {
                "healthy": health_score > 0.7,
                "health_score": health_score,
                "timestamp": aware_utc_now().isoformat(),
                "models": {
                    "total_loaded": total_models,
                    "total_memory_mb": total_memory_mb,
                    "memory_per_model_avg": total_memory_mb / max(total_models, 1)
                },
                "inference": {
                    "total_requests": total_requests,
                    "total_successes": total_successes,
                    "total_errors": total_errors,
                    "overall_success_rate": total_successes / max(total_requests, 1),
                    "overall_error_rate": total_errors / max(total_requests, 1)
                },
                "resources": resources,
                "gpu_available": self._gpu_available
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }
    
    async def _get_resource_metrics(self) -> dict[str, Any]:
        """Get system resource metrics with caching"""
        current_time = time.time()
        
        # Use cached resources if still valid
        if (self._cached_resources is not None and 
            current_time - self._last_resource_check < self._resource_cache_ttl):
            return self._cached_resources
        
        try:
            resources = {
                "timestamp": aware_utc_now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            }
            
            # Add GPU metrics if available
            if self._gpu_available and self._gpu_utils:
                gpu_metrics = await self._get_gpu_metrics()
                resources.update(gpu_metrics)
            
            # Cache the results
            self._cached_resources = resources
            self._last_resource_check = current_time
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            return {
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }
    
    async def _get_gpu_metrics(self) -> dict[str, Any]:
        """Get GPU utilization metrics"""
        if not self._gpu_available or not self._gpu_utils:
            return {}
        
        try:
            gpu_metrics = {
                "gpu_count": self._gpu_utils.nvmlDeviceGetCount(),
                "gpu_devices": []
            }
            
            total_memory_mb = 0
            total_utilization = 0
            
            for i in range(gpu_metrics["gpu_count"]):
                handle = self._gpu_utils.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = self._gpu_utils.nvmlDeviceGetMemoryInfo(handle)
                memory_total_mb = mem_info.total / (1024**2)
                memory_used_mb = mem_info.used / (1024**2)
                memory_free_mb = mem_info.free / (1024**2)
                
                # Utilization info
                util_info = self._gpu_utils.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util_info.gpu
                
                device_info = {
                    "device_id": i,
                    "memory_total_mb": memory_total_mb,
                    "memory_used_mb": memory_used_mb,
                    "memory_free_mb": memory_free_mb,
                    "memory_utilization_percent": (memory_used_mb / memory_total_mb) * 100,
                    "gpu_utilization_percent": gpu_util
                }
                
                gpu_metrics["gpu_devices"].append(device_info)
                total_memory_mb += memory_used_mb
                total_utilization += gpu_util
            
            # Aggregate metrics
            gpu_metrics.update({
                "gpu_memory_mb": total_memory_mb,
                "gpu_utilization": total_utilization / max(gpu_metrics["gpu_count"], 1)
            })
            
            return gpu_metrics
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return {"gpu_error": str(e)}
    
    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        try:
            health_factors = []
            
            # Model loading health (based on successful model registrations)
            with self._lock:
                loaded_models = len(self._loaded_models)
                if loaded_models > 0:
                    health_factors.append(1.0)  # Models are loaded
                else:
                    health_factors.append(0.0)  # No models loaded
            
            # Inference performance health
            total_requests = sum(metrics.request_count for metrics in self._model_metrics.values())
            if total_requests > 0:
                overall_success_rate = sum(metrics.success_count for metrics in self._model_metrics.values()) / total_requests
                health_factors.append(overall_success_rate)
            else:
                health_factors.append(0.8)  # Neutral score if no requests yet
            
            # Resource utilization health
            resources = await self._get_resource_metrics()
            cpu_health = max(0.0, 1.0 - (resources.get("cpu_percent", 0) / 100.0))
            memory_health = max(0.0, 1.0 - (resources.get("memory_percent", 0) / 100.0))
            health_factors.extend([cpu_health, memory_health])
            
            # Latency health (based on recent performance)
            if self._performance_history:
                recent_latencies = [
                    entry["latency_ms"] for entry in list(self._performance_history)[-100:]
                ]
                avg_latency = sum(recent_latencies) / len(recent_latencies)
                # Good health if average latency < 500ms, poor health if > 5000ms
                latency_health = max(0.0, min(1.0, (5000 - avg_latency) / 4500))
                health_factors.append(latency_health)
            
            # Calculate weighted average
            return sum(health_factors) / len(health_factors) if health_factors else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            # Try different methods to estimate memory
            
            # Method 1: Try pickle serialization
            try:
                import pickle
                serialized = pickle.dumps(model)
                return len(serialized) / (1024 * 1024)
            except Exception:
                pass
            
            # Method 2: Check if it's a scikit-learn model with parameters
            if hasattr(model, 'get_params'):
                # Estimate based on model type and parameters
                model_type = type(model).__name__
                if 'RandomForest' in model_type:
                    n_estimators = getattr(model, 'n_estimators', 100)
                    return max(5.0, n_estimators * 0.1)  # Rough estimate
                elif 'GradientBoosting' in model_type:
                    n_estimators = getattr(model, 'n_estimators', 100)
                    return max(3.0, n_estimators * 0.08)
                else:
                    return 10.0  # Default sklearn model estimate
            
            # Method 3: Check for PyTorch models
            if hasattr(model, 'parameters'):
                try:
                    total_params = sum(p.numel() for p in model.parameters())
                    # Assume float32 (4 bytes per parameter)
                    return (total_params * 4) / (1024 * 1024)
                except Exception:
                    return 50.0  # Default neural network estimate
            
            # Method 4: Fallback estimate
            return 5.0
            
        except Exception as e:
            logger.warning(f"Could not estimate model memory: {e}")
            return 5.0
    
    async def cleanup_stale_models(self, max_age_hours: int = 24) -> int:
        """Remove models that haven't been accessed recently"""
        try:
            cutoff_time = aware_utc_now().timestamp() - (max_age_hours * 3600)
            stale_models = []
            
            with self._lock:
                for model_id, model_info in self._loaded_models.items():
                    last_accessed = model_info.get("last_accessed")
                    if last_accessed and last_accessed.timestamp() < cutoff_time:
                        stale_models.append(model_id)
                
                # Remove stale models
                for model_id in stale_models:
                    del self._loaded_models[model_id]
                    if model_id in self._model_metrics:
                        del self._model_metrics[model_id]
            
            if stale_models:
                logger.info(f"Cleaned up {len(stale_models)} stale models")
            
            return len(stale_models)
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale models: {e}")
            return 0
    
    async def get_performance_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get performance summary for the specified time period"""
        try:
            cutoff_time = aware_utc_now().timestamp() - (hours * 3600)
            recent_entries = [
                entry for entry in self._performance_history
                if entry["timestamp"].timestamp() > cutoff_time
            ]
            
            if not recent_entries:
                return {
                    "period_hours": hours,
                    "total_requests": 0,
                    "message": "No performance data available for this period"
                }
            
            # Calculate metrics
            total_requests = len(recent_entries)
            successful_requests = sum(1 for entry in recent_entries if entry["success"])
            failed_requests = total_requests - successful_requests
            
            latencies = [entry["latency_ms"] for entry in recent_entries]
            
            # Error analysis
            error_types = {}
            for entry in recent_entries:
                if not entry["success"] and entry.get("error_type"):
                    error_types[entry["error_type"]] = error_types.get(entry["error_type"], 0) + 1
            
            # Model breakdown
            model_breakdown = {}
            for entry in recent_entries:
                model_id = entry["model_id"]
                if model_id not in model_breakdown:
                    model_breakdown[model_id] = {"requests": 0, "successes": 0, "latencies": []}
                
                model_breakdown[model_id]["requests"] += 1
                if entry["success"]:
                    model_breakdown[model_id]["successes"] += 1
                model_breakdown[model_id]["latencies"].append(entry["latency_ms"])
            
            return {
                "period_hours": hours,
                "timestamp": aware_utc_now().isoformat(),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / total_requests,
                "error_rate": failed_requests / total_requests,
                "latency_stats": {
                    "min_ms": min(latencies) if latencies else 0,
                    "max_ms": max(latencies) if latencies else 0,
                    "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                    "p50_ms": get_numpy().percentile(latencies, 50) if latencies else 0,
                    "p95_ms": get_numpy().percentile(latencies, 95) if latencies else 0,
                    "p99_ms": get_numpy().percentile(latencies, 99) if latencies else 0
                },
                "error_types": error_types,
                "model_breakdown": {
                    model_id: {
                        "requests": stats["requests"],
                        "success_rate": stats["successes"] / stats["requests"],
                        "avg_latency_ms": sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
                    }
                    for model_id, stats in model_breakdown.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                "error": str(e),
                "period_hours": hours,
                "timestamp": aware_utc_now().isoformat()
            }


# Global health monitor instance
_health_monitor: MLHealthMonitor | None = None

async def get_ml_health_monitor() -> MLHealthMonitor:
    """Get or create global ML health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = MLHealthMonitor()
    return _health_monitor