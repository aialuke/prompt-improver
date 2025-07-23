# Priority 4B Modernization Implementation Plan

## Overview

This document provides a detailed implementation plan for modernizing the MultiLevelCache and ResourceManager components to meet 2025 best practices.

## Phase 1: Core Modernization (2 weeks)

### 1.1 OpenTelemetry Integration

**MultiLevelCache OpenTelemetry Enhancement**
```python
# File: src/prompt_improver/utils/modern_multi_level_cache.py

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

class ModernMultiLevelCache:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Metrics
        self.cache_operations = self.meter.create_counter(
            "cache_operations_total",
            description="Total cache operations",
            unit="1"
        )
        self.cache_hit_ratio = self.meter.create_histogram(
            "cache_hit_ratio",
            description="Cache hit ratio by level",
            unit="ratio"
        )
    
    async def get(self, key: str) -> Any:
        with self.tracer.start_as_current_span("cache.get") as span:
            span.set_attribute("cache.key", key)
            span.set_attribute("cache.operation", "get")
            
            try:
                result = await self._get_with_fallback(key)
                span.set_attribute("cache.hit", result is not None)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
```

**ResourceManager OpenTelemetry Enhancement**
```python
# File: src/prompt_improver/ml/orchestration/core/modern_resource_manager.py

class ModernResourceManager:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Resource metrics
        self.resource_allocations = self.meter.create_counter(
            "resource_allocations_total",
            description="Total resource allocations",
            unit="1"
        )
        self.resource_utilization = self.meter.create_gauge(
            "resource_utilization_ratio",
            description="Resource utilization ratio",
            unit="ratio"
        )
    
    async def allocate_resource(self, resource_type: ResourceType, amount: float) -> str:
        with self.tracer.start_as_current_span("resource.allocate") as span:
            span.set_attribute("resource.type", resource_type.value)
            span.set_attribute("resource.amount", amount)
            
            try:
                allocation_id = await self._allocate_with_circuit_breaker(resource_type, amount)
                span.set_attribute("resource.allocation_id", allocation_id)
                span.set_status(Status(StatusCode.OK))
                return allocation_id
            except ResourceExhaustionError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
```

### 1.2 Prometheus Metrics Integration

**Enhanced Metrics Collection**
```python
# File: src/prompt_improver/ml/orchestration/monitoring/prometheus_exporter.py

from prometheus_client import Counter, Gauge, Histogram, start_http_server

class PrometheusExporter:
    def __init__(self):
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['cache_type', 'operation', 'level']
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio by level',
            ['cache_type', 'level']
        )
        
        # Resource metrics
        self.resource_utilization_ratio = Gauge(
            'resource_utilization_ratio',
            'Resource utilization ratio',
            ['resource_type', 'component']
        )
        
        self.resource_allocation_duration = Histogram(
            'resource_allocation_duration_seconds',
            'Time spent allocating resources',
            ['resource_type']
        )
    
    def start_server(self, port: int = 8000):
        start_http_server(port)
```

### 1.3 Circuit Breaker Implementation

**Resource Manager Circuit Breaker**
```python
# File: src/prompt_improver/ml/orchestration/resilience/circuit_breaker.py

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
import asyncio
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
```

## Phase 2: Advanced Features (3 weeks)

### 2.1 Cache Consistency Patterns

**Write-Through and Cache-Aside Implementation**
```python
# File: src/prompt_improver/utils/cache_consistency.py

class CacheConsistencyManager:
    def __init__(self, cache: MultiLevelCache, storage: Any):
        self.cache = cache
        self.storage = storage
        self.event_bus = EventBus()
    
    async def write_through(self, key: str, value: Any) -> None:
        """Write to cache and storage synchronously"""
        async with self.tracer.start_as_current_span("cache.write_through"):
            # Write to storage first
            await self.storage.set(key, value)
            # Then update cache
            await self.cache.set(key, value)
            # Publish invalidation event
            await self.event_bus.publish(CacheInvalidationEvent(key))
    
    async def cache_aside_get(self, key: str, loader: Callable) -> Any:
        """Cache-aside pattern with loader function"""
        # Try cache first
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # Load from source
        value = await loader(key)
        if value is not None:
            # Populate cache
            await self.cache.set(key, value)
        
        return value
    
    async def invalidate_pattern(self, pattern: str) -> None:
        """Distributed cache invalidation by pattern"""
        await self.event_bus.publish(CachePatternInvalidationEvent(pattern))
```

### 2.2 Kubernetes Integration

**Resource Manager Kubernetes Integration**
```python
# File: src/prompt_improver/ml/orchestration/k8s/resource_integration.py

from kubernetes import client, config
from kubernetes.client.rest import ApiException

class KubernetesResourceManager:
    def __init__(self):
        config.load_incluster_config()  # or load_kube_config() for local
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v2 = client.AutoscalingV2Api()
    
    async def get_node_resources(self) -> Dict[str, Any]:
        """Get available node resources"""
        try:
            nodes = self.v1.list_node()
            resources = {}
            for node in nodes.items:
                node_name = node.metadata.name
                allocatable = node.status.allocatable
                resources[node_name] = {
                    'cpu': allocatable.get('cpu', '0'),
                    'memory': allocatable.get('memory', '0'),
                    'nvidia.com/gpu': allocatable.get('nvidia.com/gpu', '0')
                }
            return resources
        except ApiException as e:
            logger.error(f"Error getting node resources: {e}")
            return {}
    
    async def create_hpa(self, deployment_name: str, namespace: str, 
                        min_replicas: int, max_replicas: int) -> bool:
        """Create Horizontal Pod Autoscaler"""
        hpa = client.V2HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(name=f"{deployment_name}-hpa"),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=70
                            )
                        )
                    )
                ]
            )
        )
        
        try:
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=namespace, body=hpa
            )
            return True
        except ApiException as e:
            logger.error(f"Error creating HPA: {e}")
            return False
```

## Phase 3: Observability & Security (2 weeks)

### 3.1 Grafana Dashboard Templates

**Cache Performance Dashboard**
```json
{
  "dashboard": {
    "title": "Multi-Level Cache Performance",
    "panels": [
      {
        "title": "Cache Hit Ratio",
        "type": "stat",
        "targets": [
          {
            "expr": "cache_hit_ratio{cache_type=\"prompt\"}",
            "legendFormat": "{{level}}"
          }
        ]
      },
      {
        "title": "Cache Operations Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cache_operations_total[5m])",
            "legendFormat": "{{operation}} - {{level}}"
          }
        ]
      }
    ]
  }
}
```

### 3.2 Security Enhancements

**Access Control and Audit Logging**
```python
# File: src/prompt_improver/security/cache_security.py

class CacheSecurityManager:
    def __init__(self, rbac_provider: RBACProvider):
        self.rbac = rbac_provider
        self.audit_logger = AuditLogger()
    
    async def authorize_cache_access(self, user_id: str, cache_key: str, 
                                   operation: str) -> bool:
        """Authorize cache access with RBAC"""
        permission = f"cache:{operation}"
        resource = f"cache_key:{cache_key}"
        
        authorized = await self.rbac.check_permission(user_id, permission, resource)
        
        # Audit log
        await self.audit_logger.log_access_attempt(
            user_id=user_id,
            resource=resource,
            operation=operation,
            authorized=authorized,
            timestamp=datetime.utcnow()
        )
        
        return authorized
    
    async def encrypt_cache_value(self, value: Any) -> bytes:
        """Encrypt cache values at rest"""
        # Implementation depends on encryption provider
        pass
```

## Implementation Timeline

**Week 1-2: Phase 1**
- [ ] OpenTelemetry integration for both components
- [ ] Prometheus metrics export
- [ ] Circuit breaker implementation
- [ ] Basic testing and validation

**Week 3-5: Phase 2**
- [ ] Cache consistency patterns
- [ ] Kubernetes resource integration
- [ ] Advanced GPU management
- [ ] Comprehensive testing

**Week 6-7: Phase 3**
- [ ] Grafana dashboards
- [ ] Security enhancements
- [ ] Performance optimization
- [ ] Documentation and training

## Success Metrics

1. **Performance**: 50% reduction in cache miss penalties
2. **Reliability**: 99.9% resource allocation success rate
3. **Observability**: 100% operation tracing coverage
4. **Security**: Zero unauthorized cache access incidents

## Risk Mitigation

1. **Backward Compatibility**: Maintain existing APIs during transition
2. **Gradual Rollout**: Feature flags for new functionality
3. **Monitoring**: Enhanced alerting during migration
4. **Rollback Plan**: Quick revert capability for each phase
