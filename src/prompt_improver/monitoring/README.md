# Unified Health Monitoring Orchestration System

## Overview

This directory contains the **Unified Health Monitoring Orchestration System** - a comprehensive, production-ready health monitoring solution following 2025 SRE best practices. The system integrates all existing health monitoring components into a unified orchestration layer that provides advanced capabilities like dependency management, circuit breaker coordination, health scoring, and automated alerting.

## Architecture

### Core Components

1. **HealthOrchestrator** (`health_orchestrator.py`)
   - Central orchestration system that coordinates all health monitoring
   - Manages execution order based on component dependencies
   - Provides unified dashboard and API endpoints
   - Integrates with Prometheus/Grafana for monitoring

2. **HealthScoreCalculator**
   - Calculates overall system health score (0-100)
   - Weighs components based on criticality and dependency levels
   - Provides health status determination (healthy/degraded/critical/failed)

3. **DependencyManager**
   - Manages component dependency chains
   - Detects cascade failures
   - Optimizes execution order for health checks
   - Handles circular dependency detection

4. **PerformanceAggregator**
   - Aggregates metrics from all health monitoring systems
   - Provides trend analysis and historical data
   - Calculates SLA compliance metrics

5. **AlertingManager**
   - Integrates with Prometheus/Grafana
   - Manages alert lifecycle (creation, updates, resolution)
   - Provides alert history and metrics

6. **CircuitBreakerCoordinator**
   - Coordinates circuit breakers across all systems
   - Manages cascade circuit breaker opening
   - Provides centralized breaker management

## Integrated Health Systems

The orchestrator integrates with all existing health monitoring systems:

- **Enhanced Health Service** - General application health
- **ML Health Integration Manager** - ML-specific monitoring
- **External API Health Monitor** - API dependency monitoring  
- **Redis Health Monitor** - Cache system monitoring
- **Database Health Monitor** - PostgreSQL deep monitoring

## Key Features

### 1. Unified Health Dashboard
- **Endpoint**: `GET /health/dashboard`
- Comprehensive system health overview
- Component health scores and detailed status
- Dependency chain analysis
- Performance metrics aggregation
- Active alerts and recommendations
- Historical trends and execution data

### 2. Health Score Algorithm (0-100)
- **Critical Components**: Database (30%), ML System (25%)
- **Important Components**: Cache (15%), External APIs (15%)
- **Optional Components**: System Resources (10%), Application (5%)
- Weighted scoring based on component criticality
- Response time penalties for slow components
- Status-based multipliers for degraded/failed components

### 3. Dependency Chain Management
- Topological sorting for optimal execution order
- Cascade failure detection and analysis
- Dependency impact visualization
- Circular dependency handling

### 4. Circuit Breaker Coordination
- Unified circuit breaker management across all systems
- SLA-based thresholds (response time, success rate)
- Cascade circuit breaker opening for dependent components
- Emergency reset capabilities via API

### 5. Prometheus/Grafana Integration
- **Endpoint**: `GET /health/metrics/prometheus`
- Real-time health score metrics
- Component status gauges
- Alert count metrics by severity
- Health check duration histograms
- Cascade failure counters

### 6. Performance Aggregation
- Response time percentiles (p50, p95, p99)
- Availability calculations
- Error rate tracking
- SLA compliance monitoring
- Trend analysis (improving/stable/degrading)

## API Endpoints

### Main Health Endpoints
- `GET /health/` - Main health check (enhanced with unified monitoring)
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/startup` - Kubernetes startup probe

### Unified Orchestration Endpoints
- `GET /health/dashboard` - **Unified Health Dashboard**
- `GET /health/orchestrator/status` - Orchestrator system status
- `GET /health/metrics/prometheus` - Prometheus metrics
- `POST /health/circuit-breakers/reset` - Reset all circuit breakers
- `GET /health/dependencies` - Component dependency information
- `GET /health/component/{name}` - Individual component health

### Legacy Endpoints (maintained for compatibility)
- Component-specific endpoints (Redis, Database, etc.)
- Individual service health checks
- Performance-specific endpoints

## Health Score Calculation

### Component Weights
```python
{
    "database": 0.30,      # Critical - 30%
    "cache": 0.15,         # Important - 15% 
    "ml_system": 0.25,     # Important - 25%
    "external_apis": 0.15, # Important - 15%
    "system_resources": 0.10, # Optional - 10%
    "application": 0.05    # Optional - 5%
}
```

### Health Score Ranges
- **90-100**: Healthy (all systems operating optimally)
- **70-89**: Degraded (some performance issues)
- **30-69**: Critical (significant problems requiring attention)
- **0-29**: Failed (severe system-wide issues)

### Dependency Level Weighting
- **Critical**: 1.5x weight multiplier (cannot function without)
- **Important**: 1.0x weight multiplier (degraded without)
- **Optional**: 0.5x weight multiplier (can function without)

## Component Configuration

### Health Components
Each monitored component has configuration including:
- **Name**: Unique identifier
- **Category**: database, cache, ml_system, external_apis, system_resources, application
- **Dependency Level**: critical, important, optional
- **Timeout**: Maximum execution time
- **Weight**: Component importance factor
- **Dependencies**: List of components this depends on
- **Circuit Breaker**: Enable/disable circuit breaker protection

### Example Component Configuration
```python
"database_primary": HealthComponent(
    name="database_primary",
    category="database",
    dependency_level=DependencyLevel.CRITICAL,
    timeout_seconds=15.0,
    weight=1.0,
    dependencies=[],
    circuit_breaker_enabled=True
)
```

## Execution Flow

### Health Check Execution Order
1. **Dependency Analysis**: Build dependency graph from component configurations
2. **Execution Grouping**: Group components by dependency level for parallel execution
3. **Circuit Breaker Check**: Evaluate circuit breaker states before execution
4. **Parallel Execution**: Run health checks in dependency order with timeout protection
5. **Result Aggregation**: Collect and normalize results from all components
6. **Health Score Calculation**: Calculate weighted overall health score
7. **Cascade Analysis**: Detect cascade failures and dependency impacts
8. **Alert Processing**: Generate/update alerts based on results
9. **Metrics Export**: Update Prometheus metrics for monitoring

### Error Handling
- **Circuit Breaker Protection**: Automatic failure isolation
- **Timeout Management**: Per-component timeout enforcement
- **Fallback Mechanisms**: Legacy health checker fallback
- **Graceful Degradation**: Partial results if some components fail

## Monitoring Integration

### Prometheus Metrics
- `system_health_score` - Overall system health score (0-100)
- `component_health_score` - Individual component health scores
- `active_health_alerts_total` - Number of active alerts by severity
- `health_check_duration_seconds` - Health check execution time
- `cascade_failures_total` - Cascade failure events

### Grafana Dashboard Recommendations
- **System Overview**: Overall health score, component status grid
- **Performance Metrics**: Response times, availability trends
- **Alert Dashboard**: Active alerts, alert history, resolution times
- **Dependency Visualization**: Component dependency graph
- **Circuit Breaker Status**: Breaker states, failure rates

## Usage Examples

### Basic Health Check
```python
from prompt_improver.monitoring.health_orchestrator import get_health_orchestrator

orchestrator = await get_health_orchestrator()
snapshot = await orchestrator.execute_comprehensive_health_check()

print(f"Overall Health: {snapshot.overall_status.value}")
print(f"Health Score: {snapshot.overall_health_score}/100")
```

### Get Health Dashboard
```bash
curl http://${API_HOST:-localhost}:${API_PORT:-8000}/health/dashboard
```

### Reset Circuit Breakers (Emergency)
```bash
curl -X POST http://${API_HOST:-localhost}:${API_PORT:-8000}/health/circuit-breakers/reset
```

### Get Prometheus Metrics
```bash
curl http://${API_HOST:-localhost}:${API_PORT:-8000}/health/metrics/prometheus
```

## Configuration

### Global Configuration
The system uses the centralized configuration system with circuit breaker settings:

```yaml
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_max_calls: 3
  reset_timeout: 120
  response_time_threshold_ms: 1000
  success_rate_threshold: 0.95
```

### Environment Variables
- `HEALTH_ORCHESTRATOR_TIMEOUT`: Default health check timeout (default: 120s)
- `HEALTH_PARALLEL_EXECUTION`: Enable parallel execution (default: true)
- `HEALTH_DEEP_ANALYSIS`: Enable deep analysis features (default: true)

## Troubleshooting

### Common Issues

1. **Slow Health Checks**
   - Check component timeout configurations
   - Review circuit breaker states
   - Analyze dependency chain execution order

2. **Circuit Breakers Stuck Open**
   - Use `/health/circuit-breakers/reset` endpoint
   - Check underlying service health
   - Review circuit breaker threshold configurations

3. **Missing Component Data**
   - Verify component initialization in orchestrator
   - Check health system availability flags
   - Review error logs for import/initialization issues

4. **Prometheus Metrics Not Updating**
   - Verify Prometheus client installation
   - Check metric registry initialization
   - Review endpoint accessibility

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger('prompt_improver.monitoring.health_orchestrator').setLevel(logging.DEBUG)
```

## Migration from Legacy System

### Backward Compatibility
- All existing endpoints maintained for compatibility
- Legacy health checker used as fallback
- Gradual migration path available
- No breaking changes to existing API contracts

### Migration Steps
1. **Deploy**: Deploy unified orchestration system alongside existing
2. **Monitor**: Verify unified system operates correctly
3. **Switch**: Update clients to use new dashboard endpoints
4. **Optimize**: Remove redundant legacy code after confidence period

## Performance Characteristics

### Execution Performance
- **Parallel Execution**: ~2-5 seconds for complete health check
- **Sequential Fallback**: ~10-15 seconds for complete health check
- **Component Timeout**: Configurable per component (5-30s)
- **Memory Usage**: ~50-100MB for orchestration overhead

### Scalability
- **Component Limit**: Supports 50+ health components
- **Dependency Depth**: Handles 10+ levels of dependencies
- **Historical Data**: Maintains 100 health check snapshots
- **Alert History**: Stores 1000 alert events

## Future Enhancements

### Planned Features
- **Machine Learning Predictions**: Predictive failure analysis
- **Auto-Healing**: Automated recovery actions
- **Service Mesh Integration**: Istio/Envoy health integration
- **Multi-Region Support**: Cross-region health aggregation
- **Advanced Visualization**: Real-time dependency graph UI

### Extension Points
- **Custom Health Checkers**: Plugin architecture for new components
- **Alert Handlers**: Custom alert processing and routing
- **Metric Exporters**: Additional monitoring system integrations
- **Health Policies**: Configurable health determination rules

---

## Support

For questions, issues, or contributions related to the unified health monitoring system:

1. **Documentation**: Refer to component-specific documentation
2. **Logging**: Enable debug logging for detailed troubleshooting
3. **Monitoring**: Use Prometheus/Grafana dashboards for system insights
4. **Emergency**: Use circuit breaker reset and fallback mechanisms

The unified health monitoring orchestration system provides enterprise-grade health monitoring capabilities with production-ready reliability, comprehensive observability, and operational excellence.