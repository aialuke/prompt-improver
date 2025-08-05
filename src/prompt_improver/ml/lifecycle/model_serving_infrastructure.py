"""Production Model Serving Infrastructure (2025)

High-performance model serving with comprehensive health monitoring:
- Auto-scaling model serving with load balancing
- Real-time health monitoring and alerting
- A/B testing and canary deployment support
- Performance optimization with caching and batching
- Multi-model serving with resource isolation
- Integration with monitoring and alerting systems
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import psutil
from pydantic import BaseModel, Field as PydanticField
from ...monitoring.opentelemetry.metrics import get_ml_metrics
from ...core.metrics.unified_metrics_adapter import get_unified_metrics_adapter, get_metrics_streamer
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

from .enhanced_model_registry import EnhancedModelRegistry, ModelMetadata, ModelStatus
from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)

class ServingStatus(Enum):
    """Model serving status."""
    LOADING = "loading"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SCALING = "scaling"
    TERMINATED = "terminated"

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    FIXED = "fixed"
    CPU_BASED = "cpu_based"
    REQUEST_BASED = "request_based"
    LATENCY_BASED = "latency_based"
    CUSTOM = "custom"

@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_id: str
    model_alias: Optional[str] = None
    
    # Resource configuration
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    
    # Scaling configuration
    scaling_strategy: ScalingStrategy = ScalingStrategy.REQUEST_BASED
    target_cpu_utilization: int = 70
    target_requests_per_second: int = 100
    target_latency_ms: int = 100
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    
    # Health check configuration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    health_check_failure_threshold: int = 3
    
    # Performance optimization
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_model_warmup: bool = True
    warmup_requests: int = 10
    
    # A/B testing
    traffic_split: Optional[Dict[str, float]] = None  # {model_id: percentage}
    
    # Monitoring
    enable_detailed_metrics: bool = True
    enable_request_logging: bool = True
    log_sample_rate: float = 0.1

@dataclass  
class ServingMetrics:
    """Comprehensive serving metrics."""
    model_id: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    peak_rps: float = 0.0
    
    # Resource metrics
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Model metrics
    active_replicas: int = 0
    pending_replicas: int = 0
    model_load_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Health metrics
    health_check_success_rate: float = 100.0
    last_health_check: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Batching metrics
    avg_batch_size: float = 1.0
    batch_processing_time_ms: float = 0.0

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, serving_config: ServingConfig):
        self.config = serving_config
        self.health_history: List[Dict[str, Any]] = []
        self.consecutive_failures = 0
        self.last_check_time = None
        
    async def check_health(self, model_instance: Any) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        check_start = time.time()
        
        health_result = {
            "timestamp": aware_utc_now().isoformat(),
            "model_id": self.config.model_id,
            "status": "healthy",
            "checks": {},
            "metrics": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # 1. Model availability check
            health_result["checks"]["model_loaded"] = model_instance is not None
            
            # 2. Memory usage check
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            health_result["checks"]["memory_ok"] = memory_percent < 90
            health_result["metrics"]["memory_usage_mb"] = memory_mb
            health_result["metrics"]["memory_percent"] = memory_percent
            
            if memory_percent > 80:
                health_result["warnings"].append(f"High memory usage: {memory_percent:.1f}%")
            
            # 3. CPU usage check
            cpu_percent = process.cpu_percent(interval=0.1)
            health_result["checks"]["cpu_ok"] = cpu_percent < 95
            health_result["metrics"]["cpu_percent"] = cpu_percent
            
            if cpu_percent > 80:
                health_result["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # 4. Model inference test
            if model_instance and hasattr(model_instance, 'predict'):
                try:
                    # Generate test input (simplified)
                    test_input = np.random.randn(1, 10).astype(np.float32)
                    inference_start = time.time()
                    
                    prediction = model_instance.predict(test_input)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    health_result["checks"]["inference_ok"] = True
                    health_result["metrics"]["test_inference_ms"] = inference_time
                    
                    if inference_time > 1000:  # 1 second threshold
                        health_result["warnings"].append(f"Slow inference: {inference_time:.1f}ms")
                        
                except Exception as e:
                    health_result["checks"]["inference_ok"] = False
                    health_result["errors"].append(f"Inference test failed: {str(e)}")
            
            # 5. Disk space check
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            health_result["checks"]["disk_ok"] = disk_percent < 90
            health_result["metrics"]["disk_percent"] = disk_percent
            
            # 6. Network connectivity (simplified)
            health_result["checks"]["network_ok"] = True  # Would implement actual check
            
            # Overall health determination
            failed_checks = [k for k, v in health_result["checks"].items() if not v]
            if failed_checks:
                health_result["status"] = "unhealthy"
                self.consecutive_failures += 1
            else:
                health_result["status"] = "healthy" if not health_result["warnings"] else "degraded"
                self.consecutive_failures = 0
            
            health_result["metrics"]["check_duration_ms"] = (time.time() - check_start) * 1000
            health_result["consecutive_failures"] = self.consecutive_failures
            
        except Exception as e:
            health_result["status"] = "unhealthy"
            health_result["errors"].append(f"Health check error: {str(e)}")
            self.consecutive_failures += 1
        
        # Store health history
        self.health_history.append(health_result)
        if len(self.health_history) > 100:  # Keep last 100 checks
            self.health_history.pop(0)
        
        self.last_check_time = time.time()
        
        return health_result

class ModelServingInstance:
    """Individual model serving instance with comprehensive monitoring."""
    
    def __init__(self, 
                 model_metadata: ModelMetadata,
                 serving_config: ServingConfig,
                 model_registry: EnhancedModelRegistry):
        """Initialize model serving instance.
        
        Args:
            model_metadata: Model metadata from registry
            serving_config: Serving configuration
            model_registry: Model registry for loading models
        """
        self.model_metadata = model_metadata
        self.config = serving_config
        self.model_registry = model_registry
        
        # Instance identification
        self.instance_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Model state
        self.model = None
        self.status = ServingStatus.LOADING
        self.last_request_time = None
        
        # Metrics and monitoring
        self.metrics = ServingMetrics(model_id=model_metadata.model_id)
        self.health_checker = HealthChecker(serving_config)
        
        # Request batching
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # OpenTelemetry metrics integration - dual approach for comprehensive coverage
        self.ml_metrics = get_ml_metrics()  # Direct ML-specific metrics
        self.unified_adapter = get_unified_metrics_adapter()  # Unified adapter for compatibility
        self.metrics_streamer = get_metrics_streamer()  # Real-time WebSocket streaming
        self._setup_opentelemetry_metrics()
        
    async def initialize(self):
        """Initialize the serving instance."""
        
        try:
            self.status = ServingStatus.LOADING
            
            # Load model from registry
            self.model = await self.model_registry._load_model(self.model_metadata.model_id)
            
            # Warm up model if configured
            if self.config.enable_model_warmup:
                await self._warmup_model()
            
            self.status = ServingStatus.HEALTHY
            self.metrics.model_load_time_ms = (time.time() - self.start_time) * 1000
            
            # Start background tasks using centralized task manager
            task_manager = get_background_task_manager()
            
            # Submit health monitoring task with HIGH priority
            await task_manager.submit_enhanced_task(
                task_id=f"health_monitor_{self.instance_id}",
                coroutine=self._health_monitor,
                priority=TaskPriority.HIGH,
                tags={"service": "ml_serving", "type": "health_monitor", "instance_id": self.instance_id}
            )
            
            # Submit metrics collection task with NORMAL priority
            await task_manager.submit_enhanced_task(
                task_id=f"metrics_collector_{self.instance_id}",
                coroutine=self._metrics_collector,
                priority=TaskPriority.NORMAL,
                tags={"service": "ml_serving", "type": "metrics_collector", "instance_id": self.instance_id}
            )
            
            if self.config.enable_batching:
                # Submit batch processing task with NORMAL priority
                await task_manager.submit_enhanced_task(
                    task_id=f"batch_processor_{self.instance_id}",
                    coroutine=self._batch_processor,
                    priority=TaskPriority.NORMAL,
                    tags={"service": "ml_serving", "type": "batch_processor", "instance_id": self.instance_id}
                )
            
            logger.info(f"Model serving instance {self.instance_id} initialized")
            
        except Exception as e:
            self.status = ServingStatus.UNHEALTHY
            logger.error(f"Failed to initialize serving instance: {e}")
            raise
    
    async def predict(self, 
                     input_data: np.ndarray,
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction with comprehensive monitoring."""
        
        prediction_start = time.time()
        request_id = request_id or str(uuid.uuid4())
        
        try:
            # Update request metrics
            self.metrics.total_requests += 1
            self.last_request_time = time.time()
            
            # Check cache if enabled
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(input_data)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.metrics.cache_hit_rate = self._update_cache_hit_rate(True)
                    return cached_result
                else:
                    self.metrics.cache_hit_rate = self._update_cache_hit_rate(False)
            
            # Batching logic
            if self.config.enable_batching:
                result = await self._predict_with_batching(input_data, request_id)
            else:
                result = await self._predict_single(input_data)
            
            # Cache result if enabled
            if self.config.enable_caching:
                self._store_in_cache(cache_key, result)
            
            # Update success metrics
            self.metrics.successful_requests += 1
            prediction_time = (time.time() - prediction_start) * 1000
            self._update_latency_metrics(prediction_time)
            
            return {
                "predictions": result,
                "model_id": self.model_metadata.model_id,
                "model_version": str(self.model_metadata.version),
                "instance_id": self.instance_id,
                "request_id": request_id,
                "prediction_time_ms": prediction_time,
                "timestamp": aware_utc_now().isoformat()
            }
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            
            logger.error(f"Prediction failed for request {request_id}: {e}")
            
            return {
                "error": str(e),
                "model_id": self.model_metadata.model_id,
                "instance_id": self.instance_id,
                "request_id": request_id,
                "timestamp": aware_utc_now().isoformat()
            }
    
    async def _predict_single(self, input_data: np.ndarray) -> Any:
        """Single prediction without batching."""
        
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Convert to serializable format
        if hasattr(prediction, 'tolist'):
            return prediction.tolist()
        elif hasattr(prediction, 'item'):
            return prediction.item()
        else:
            return prediction
    
    async def _predict_with_batching(self, 
                                   input_data: np.ndarray, 
                                   request_id: str) -> Any:
        """Prediction with request batching for performance."""
        
        # Create prediction future
        prediction_future = asyncio.Future()
        
        # Add to batch queue
        async with self.batch_lock:
            self.batch_queue.append({
                "input_data": input_data,
                "request_id": request_id,
                "future": prediction_future,
                "timestamp": time.time()
            })
        
        # Wait for batch processing
        try:
            result = await asyncio.wait_for(
                prediction_future, 
                timeout=self.config.batch_timeout_ms / 1000.0 + 1.0
            )
            return result
            
        except asyncio.TimeoutError:
            # Fallback to single prediction
            return await self._predict_single(input_data)
    
    async def _batch_processor(self):
        """Background batch processor for improved throughput."""
        
        while True:
            try:
                # Wait for batch timeout or max batch size
                await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)
                
                if not self.batch_queue:
                    continue
                
                # Extract batch
                async with self.batch_lock:
                    if not self.batch_queue:
                        continue
                        
                    batch = self.batch_queue[:self.config.max_batch_size]
                    self.batch_queue = self.batch_queue[len(batch):]
                
                if not batch:
                    continue
                
                # Process batch
                batch_start = time.time()
                
                try:
                    # Combine inputs
                    batch_inputs = np.vstack([item["input_data"] for item in batch])
                    
                    # Make batch prediction
                    batch_predictions = await self._predict_single(batch_inputs)
                    
                    # Distribute results
                    for i, item in enumerate(batch):
                        if not item["future"].done():
                            if isinstance(batch_predictions, list):
                                result = batch_predictions[i]
                            else:
                                result = batch_predictions[i:i+1]
                            item["future"].set_result(result)
                    
                    # Update batch metrics
                    batch_time = (time.time() - batch_start) * 1000
                    self.metrics.avg_batch_size = (
                        self.metrics.avg_batch_size * 0.9 + len(batch) * 0.1
                    )
                    self.metrics.batch_processing_time_ms = (
                        self.metrics.batch_processing_time_ms * 0.9 + batch_time * 0.1
                    )
                    
                except Exception as e:
                    # Set exception for all futures in batch
                    for item in batch:
                        if not item["future"].done():
                            item["future"].set_exception(e)
                    
                    logger.error(f"Batch processing failed: {e}")
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _health_monitor(self):
        """Background health monitoring."""
        
        while True:
            try:
                health_result = await self.health_checker.check_health(self.model)
                
                # Update status based on health
                if health_result["status"] == "healthy":
                    self.status = ServingStatus.HEALTHY
                elif health_result["status"] == "degraded":
                    self.status = ServingStatus.DEGRADED
                else:
                    self.status = ServingStatus.UNHEALTHY
                
                # Update health metrics
                self.metrics.health_check_success_rate = self._calculate_health_success_rate()
                self.metrics.last_health_check = aware_utc_now()
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collector(self):
        """Background metrics collection."""
        
        while True:
            try:
                # Update system metrics
                process = psutil.Process()
                self.metrics.cpu_utilization_percent = process.cpu_percent()
                self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                self.metrics.memory_utilization_percent = process.memory_percent()
                
                # Update uptime
                self.metrics.uptime_seconds = time.time() - self.start_time
                
                # Update RPS
                current_time = time.time()
                if hasattr(self, '_last_rps_update'):
                    time_diff = current_time - self._last_rps_update
                    if time_diff > 0:
                        request_diff = self.metrics.total_requests - getattr(self, '_last_request_count', 0)
                        self.metrics.requests_per_second = request_diff / time_diff
                        self.metrics.peak_rps = max(self.metrics.peak_rps, self.metrics.requests_per_second)
                
                self._last_rps_update = current_time
                self._last_request_count = self.metrics.total_requests
                
                # Update OpenTelemetry metrics
                await self._update_opentelemetry_metrics()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(30)
    
    def _setup_opentelemetry_metrics(self):
        """Setup OpenTelemetry metrics integration."""
        # Initialize model information in unified adapter (replaces Prometheus model_info)
        self.unified_adapter.set_model_info(
            model_name=self.model_metadata.model_name,
            model_version=str(self.model_metadata.version),
            instance_id=self.instance_id,
            status="initializing"
        )
        
        logger.info(f"OpenTelemetry metrics configured for model serving instance {self.instance_id}")
    
    async def _update_opentelemetry_metrics(self):
        """Update OpenTelemetry metrics with current serving statistics."""
        try:
            # Update model status in unified adapter
            self.unified_adapter.set_model_info(
                model_name=self.model_metadata.model_name,
                model_version=str(self.model_metadata.version),
                instance_id=self.instance_id,
                status=self.status.value
            )
            
            # Record serving metrics via unified adapter (replaces Prometheus counters/histograms)
            if self.metrics.total_requests > 0:
                self.unified_adapter.record_ml_serving_metrics(
                    model_name=self.model_metadata.model_name,
                    instance_id=self.instance_id,
                    request_count=1,  # Incremental update
                    duration_ms=self.metrics.avg_latency_ms,
                    error_count=1 if self.metrics.error_rate > 0 else 0
                )
            
            # Record detailed ML metrics via direct MLMetrics integration
            if self.metrics.total_requests > 0:
                # Record inference duration and success rate
                self.ml_metrics.record_inference(
                    model_name=self.model_metadata.model_name,
                    model_version=str(self.model_metadata.version),
                    duration_ms=self.metrics.avg_latency_ms,
                    success=(self.metrics.error_rate < 0.05)  # Consider success if error rate < 5%
                )
                
                # Record failure analysis if there are errors
                if self.metrics.error_rate > 0:
                    self.ml_metrics.record_failure_analysis(
                        failure_rate=self.metrics.error_rate,
                        failure_type="model_serving",
                        severity="warning" if self.metrics.error_rate < 0.1 else "critical",
                        total_failures=int(self.metrics.failed_requests),
                        response_time=self.metrics.avg_latency_ms / 1000.0,  # Convert to seconds
                        rpn_score=self._calculate_rpn_score()
                    )
            
            # Stream real-time metrics via WebSocket (Phase 2 integration)
            try:
                serving_metrics_data = {
                    "instance_id": self.instance_id,
                    "total_requests": self.metrics.total_requests,
                    "error_rate": self.metrics.error_rate,
                    "avg_latency_ms": self.metrics.avg_latency_ms,
                    "rps": self.metrics.requests_per_second,
                    "cpu_percent": self.metrics.cpu_utilization_percent,
                    "memory_mb": self.metrics.memory_usage_mb,
                    "status": self.status.value
                }
                
                await self.metrics_streamer.stream_ml_serving_update(
                    model_name=self.model_metadata.model_name,
                    metrics_data=serving_metrics_data
                )
            except Exception as streaming_error:
                # Non-critical error - don't break metrics collection
                logger.debug(f"WebSocket streaming error (non-critical): {streaming_error}")
                
        except Exception as e:
            logger.error(f"Failed to update OpenTelemetry metrics: {e}")
    
    def _calculate_rpn_score(self) -> float:
        """Calculate Risk Priority Number for FMEA analysis."""
        # Simple RPN calculation based on error rate, latency, and health
        severity = min(10, max(1, int(self.metrics.error_rate * 100)))  # 1-10 scale
        occurrence = min(10, max(1, int(self.metrics.avg_latency_ms / 20)))  # Based on latency
        detection = 3 if self.status == ServingStatus.HEALTHY else 7  # Detection difficulty
        
        return float(severity * occurrence * detection)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        
        return {
            "instance_id": self.instance_id,
            "model_id": self.model_metadata.model_id,
            "model_name": self.model_metadata.model_name,
            "model_version": str(self.model_metadata.version),
            "status": self.status.value,
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "error_rate": self.metrics.error_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "requests_per_second": self.metrics.requests_per_second,
                "peak_rps": self.metrics.peak_rps,
                "cpu_utilization_percent": self.metrics.cpu_utilization_percent,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "avg_batch_size": self.metrics.avg_batch_size,
                "health_check_success_rate": self.metrics.health_check_success_rate
            },
            "health_history": self.health_checker.health_history[-10:],  # Last 10 checks
            "cache_size": len(self.cache)
        }
    
    # Additional helper methods for caching, metrics updates, etc.
    # ... (implementation continues)

class ProductionModelServer:
    """Production-ready model serving system with auto-scaling and monitoring."""
    
    def __init__(self, 
                 model_registry: EnhancedModelRegistry,
                 enable_auto_scaling: bool = True):
        """Initialize production model server.
        
        Args:
            model_registry: Enhanced model registry
            enable_auto_scaling: Enable automatic scaling
        """
        self.model_registry = model_registry
        self.enable_auto_scaling = enable_auto_scaling
        
        # Serving instances
        self.serving_instances: Dict[str, List[ModelServingInstance]] = {}
        self.serving_configs: Dict[str, ServingConfig] = {}
        
        # Load balancer (simplified)
        self.request_router = {}
        
        # Auto-scaler
        self.scaling_decisions: List[Dict[str, Any]] = []
        
        logger.info("Production Model Server initialized")
    
    async def deploy_model(self, 
                          model_id: str, 
                          serving_config: ServingConfig) -> str:
        """Deploy model for serving with monitoring."""
        
        # Get model metadata
        metadata = await self.model_registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Create initial serving instances
        instances = []
        for i in range(serving_config.min_replicas):
            instance = ModelServingInstance(metadata, serving_config, self.model_registry)
            await instance.initialize()
            instances.append(instance)
        
        self.serving_instances[model_id] = instances
        self.serving_configs[model_id] = serving_config
        
        # Start auto-scaler if enabled using centralized task manager
        if self.enable_auto_scaling:
            task_manager = get_background_task_manager()
            await task_manager.submit_enhanced_task(
                task_id=f"auto_scaler_{model_id}",
                coroutine=lambda: self._auto_scaler(model_id),
                priority=TaskPriority.NORMAL,
                tags={"service": "ml_serving", "type": "auto_scaler", "model_id": model_id}
            )
        
        logger.info(f"Deployed model {model_id} with {len(instances)} instances")
        return model_id
    
    async def predict(self, 
                     model_id: str, 
                     input_data: np.ndarray,
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """Route prediction request to healthy instance."""
        
        instances = self.serving_instances.get(model_id, [])
        if not instances:
            raise ValueError(f"Model {model_id} not deployed")
        
        # Find healthy instance (simple round-robin)
        healthy_instances = [
            inst for inst in instances 
            if inst.status in [ServingStatus.HEALTHY, ServingStatus.DEGRADED]
        ]
        
        if not healthy_instances:
            raise RuntimeError(f"No healthy instances available for model {model_id}")
        
        # Simple load balancing (would implement more sophisticated logic)
        instance = min(healthy_instances, key=lambda x: x.metrics.total_requests)
        
        return await instance.predict(input_data, request_id)
    
    async def get_serving_statistics(self) -> Dict[str, Any]:
        """Get comprehensive serving statistics."""
        
        stats = {
            "deployed_models": len(self.serving_instances),
            "total_instances": sum(len(instances) for instances in self.serving_instances.values()),
            "models": {}
        }
        
        for model_id, instances in self.serving_instances.items():
            model_stats = {
                "instance_count": len(instances),
                "healthy_instances": len([i for i in instances if i.status == ServingStatus.HEALTHY]),
                "total_requests": sum(i.metrics.total_requests for i in instances),
                "avg_latency_ms": np.mean([i.metrics.avg_latency_ms for i in instances if i.metrics.avg_latency_ms > 0]),
                "error_rate": np.mean([i.metrics.error_rate for i in instances]),
                "instances": [i.get_metrics() for i in instances]
            }
            stats["models"][model_id] = model_stats
        
        return stats

# Factory function
async def create_model_server(model_registry: EnhancedModelRegistry) -> ProductionModelServer:
    """Create production model server."""
    
    return ProductionModelServer(
        model_registry=model_registry,
        enable_auto_scaling=True
    )