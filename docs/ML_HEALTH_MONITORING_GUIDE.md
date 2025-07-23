# ML Pipeline Health Monitoring Guide

## Overview

This guide provides comprehensive documentation for ML pipeline health monitoring in the prompt-improver system. Our health monitoring follows 2025 best practices with OpenTelemetry integration, circuit breakers, and predictive analysis.

## Architecture

### Health Monitoring Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Health Service                   │
├─────────────────────────────────────────────────────────────┤
│  • Circuit Breaker Integration                              │
│  • Predictive Health Monitoring                             │
│  • Health Check Result Caching                              │
│  • Dependency Graph Analysis                                │
│  • Advanced Observability with Metrics                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Infrastructure │   ML-Specific   │  ML Orchestration │   Performance   │
│   Health Checkers│  Health Checkers│  Health Checkers  │  Health Checkers│
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ • Database      │ • ML Models     │ • Orchestrator  │ • Real-time     │
│ • Redis Cache   │ • Data Quality  │ • Component     │ • Performance   │
│ • System        │ • Training      │   Registry      │ • Analytics     │
│   Resources     │ • Performance   │ • Resource      │                 │
│ • MCP Server    │                 │   Manager       │                 │
│ • Analytics     │                 │ • Workflow      │                 │
│                 │                 │   Engine        │                 │
│                 │                 │ • Event Bus     │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## ML-Specific Health Checkers

### 1. MLModelHealthChecker

Monitors the health of ML models throughout their lifecycle.

**Checks:**
- Model registry accessibility and status
- Deployed model health and cache statistics
- Model cache utilization and memory pressure
- Model performance metrics availability

**Usage:**
```python
from prompt_improver.performance.monitoring.health import MLModelHealthChecker

checker = MLModelHealthChecker()
result = await checker.check()

if result.status == HealthStatus.HEALTHY:
    print("All ML models are healthy")
else:
    print(f"ML model issues: {result.message}")
    print(f"Details: {result.details}")
```

### 2. MLDataQualityChecker

Monitors data quality and pipeline integrity.

**Checks:**
- Training data loader health and statistics
- Synthetic data generator availability
- Data preprocessing pipeline status
- Data quality metrics (missing values, duplicates)

**Usage:**
```python
from prompt_improver.performance.monitoring.health import MLDataQualityChecker

checker = MLDataQualityChecker()
result = await checker.check()

# Check for data quality issues
if result.details.get("training_data", {}).get("data_quality_issues"):
    issues = result.details["training_data"]["data_quality_issues"]
    print(f"Data quality issues detected: {issues}")
```

### 3. MLTrainingHealthChecker

Monitors ML training processes and optimization algorithms.

**Checks:**
- Optimization algorithm availability (RuleOptimizer, ClusteringOptimizer, etc.)
- Batch processing health
- Learning algorithm status (ContextLearner, InsightEngine, etc.)

**Usage:**
```python
from prompt_improver.performance.monitoring.health import MLTrainingHealthChecker

checker = MLTrainingHealthChecker()
result = await checker.check()

# Check available optimization components
optimization_health = result.details.get("optimization", {})
available_components = optimization_health.get("available_components", [])
print(f"Available optimization components: {available_components}")
```

### 4. MLPerformanceHealthChecker

Monitors ML performance evaluation and analytics integration.

**Checks:**
- Evaluation component availability (CausalInferenceAnalyzer, etc.)
- Performance monitoring system health
- Analytics service integration

**Usage:**
```python
from prompt_improver.performance.monitoring.health import MLPerformanceHealthChecker

checker = MLPerformanceHealthChecker()
result = await checker.check()

# Check evaluation components
evaluation_health = result.details.get("evaluation", {})
if not evaluation_health.get("healthy"):
    print(f"Evaluation issues: {evaluation_health.get('error')}")
```

## Enhanced Health Service

### Features

1. **Circuit Breaker Integration**: Prevents cascading failures
2. **Predictive Analysis**: Trend analysis and health predictions
3. **Caching**: Optimized health check performance
4. **Dependency Tracking**: Comprehensive dependency health monitoring
5. **OpenTelemetry Integration**: Distributed tracing and metrics

### Usage

```python
from prompt_improver.performance.monitoring.health import get_health_service

# Get the singleton health service
health_service = get_health_service()

# Run comprehensive health check
result = await health_service.run_enhanced_health_check(
    parallel=True,
    use_cache=True,
    include_predictions=True
)

print(f"Overall health: {result.overall_status}")
print(f"Failed checks: {result.failed_checks}")
print(f"Warning checks: {result.warning_checks}")

# Access individual check results
for component_name, check_result in result.checks.items():
    print(f"{component_name}: {check_result.status} ({check_result.response_time_ms}ms)")
```

### Configuration

```python
# Configure health service with custom settings
health_service = EnhancedHealthService(
    enable_circuit_breakers=True,
    enable_predictive_analysis=True,
    enable_caching=True,
    cache_ttl=30,  # seconds
    trend_window_size=10  # number of historical checks
)
```

## Orchestrator Integration

### Health Monitoring in ML Pipeline Orchestrator

The health service integrates seamlessly with the ML Pipeline Orchestrator:

```python
from prompt_improver.ml.orchestration import MLPipelineOrchestrator

orchestrator = MLPipelineOrchestrator()

# Configure health monitoring
await orchestrator.configure_health_monitoring(
    health_service=health_service,
    check_interval=60,  # seconds
    alert_thresholds={
        "failed_checks_threshold": 2,
        "response_time_threshold": 5000  # ms
    }
)

# Get orchestrator health status
health_status = await orchestrator.get_health_status()
```

## Monitoring Best Practices

### 1. Health Check Frequency

- **Critical Components**: Every 30 seconds
- **Standard Components**: Every 60 seconds
- **Background Components**: Every 5 minutes

### 2. Alerting Strategy

```python
# Configure alerting thresholds
alert_config = {
    "critical_failure_threshold": 1,  # Alert immediately on critical failures
    "warning_accumulation_threshold": 3,  # Alert after 3 warnings
    "response_time_threshold": 2000,  # Alert if response time > 2s
    "cache_utilization_threshold": 0.9  # Alert if cache utilization > 90%
}
```

### 3. Circuit Breaker Configuration

```python
# Configure circuit breakers for dependencies
circuit_breaker_config = {
    "failure_threshold": 5,  # Open after 5 failures
    "recovery_timeout": 60,  # Try recovery after 60 seconds
    "expected_exception": Exception
}
```

## Metrics and Observability

### Prometheus Metrics

The health monitoring system exports the following metrics:

- `health_check_total`: Total number of health checks performed
- `health_check_duration_seconds`: Duration of health checks
- `health_status_gauge`: Current health status (0=failed, 1=warning, 2=healthy)
- `circuit_breaker_state`: Circuit breaker state
- `dependency_health_score`: Dependency health scores

### OpenTelemetry Tracing

Health checks are instrumented with OpenTelemetry spans:

```python
# Health checks automatically create spans
with tracer.start_as_current_span("ml_model_health_check") as span:
    result = await ml_model_checker.check()
    span.set_attribute("health_status", result.status.value)
    span.set_attribute("response_time_ms", result.response_time_ms)
```

## Troubleshooting

### Common Issues

1. **High Response Times**
   - Check system resource utilization
   - Review cache hit rates
   - Analyze dependency health

2. **Frequent Health Check Failures**
   - Verify component availability
   - Check network connectivity
   - Review error logs

3. **Memory Pressure in Model Cache**
   - Monitor cache utilization metrics
   - Adjust cache size limits
   - Review model eviction policies

### Debug Commands

```python
# Get detailed health service statistics
stats = await health_service.get_health_statistics()
print(f"Total checks: {stats['total_checks']}")
print(f"Average response time: {stats['avg_response_time']}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']}")

# Get dependency health graph
dependency_graph = await health_service.get_dependency_health_graph()
for dependency, health_info in dependency_graph.items():
    print(f"{dependency}: {health_info['status']} (score: {health_info['score']})")
```

## Integration Examples

### Custom Health Checker

```python
from prompt_improver.performance.monitoring.health.base import HealthChecker, HealthResult, HealthStatus

class CustomMLComponentChecker(HealthChecker):
    def __init__(self):
        super().__init__(name="custom_ml_component")
    
    async def check(self) -> HealthResult:
        try:
            # Perform custom health check logic
            component_healthy = await self._check_component_health()
            
            return HealthResult(
                status=HealthStatus.HEALTHY if component_healthy else HealthStatus.FAILED,
                component=self.name,
                message="Custom component health check",
                details={"component_healthy": component_healthy}
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                error=str(e),
                message=f"Health check failed: {e}"
            )
    
    async def _check_component_health(self) -> bool:
        # Implement your custom health check logic
        return True

# Register custom checker
health_service = get_health_service()
health_service.add_checker(CustomMLComponentChecker())
```

### Health Check Dashboard Integration

```python
# Create health monitoring dashboard
from prompt_improver.performance.monitoring.health import create_health_dashboard

dashboard = await create_health_dashboard(
    health_service=health_service,
    refresh_interval=30,
    include_predictions=True,
    include_dependency_graph=True
)

# Start dashboard
await dashboard.start()
```

## Next Steps

1. **Implement Custom Health Checkers**: Add health checkers for your specific ML components
2. **Configure Alerting**: Set up alerting based on your operational requirements
3. **Monitor Trends**: Use predictive analysis to identify potential issues before they occur
4. **Optimize Performance**: Use caching and circuit breakers to optimize health check performance
5. **Integrate with Monitoring Stack**: Connect with your existing monitoring and alerting infrastructure

For more detailed information, see the API documentation and example implementations in the codebase.
