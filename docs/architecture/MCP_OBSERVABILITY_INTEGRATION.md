# Observability MCP Server Integration

## Overview

The Observability MCP server enhances the `performance-engineer` agent with real-time monitoring, metrics collection, and distributed tracing capabilities. This integration provides direct access to the project's sophisticated OpenTelemetry infrastructure and SLO monitoring systems.

## Current Monitoring Infrastructure

### Existing OpenTelemetry Configuration
The project has a comprehensive monitoring stack:

- **Service Name**: `apes-ml-pipeline`
- **Metrics Endpoints**: Port 8000 (main), Port 8001 (Prometheus)
- **OTLP Support**: Available with GRPC and HTTP exporters
- **Comprehensive Metrics**: 13 different metric types covering HTTP, ML, database, Redis, cache, and MCP operations
- **SLO Targets**: 99.9% availability, P95 <2s latency, P99 <5s, <0.1% error rate

### Existing Metrics Categories
1. **HTTP Performance**: Request counts, duration histograms
2. **ML Model Performance**: Predictions, inference duration, error tracking
3. **Database Performance**: Connection pools, query duration
4. **Cache Performance**: Hit/miss ratios across cache layers
5. **MCP Operations**: Operation counts and status tracking
6. **System Health**: Comprehensive health checks with SLO monitoring

## MCP Server Configuration

### Observability MCP Server Setup
```json
{
  "observability-monitoring": {
    "command": "python",
    "args": ["-m", "mcp_server_observability"],
    "env": {
      "OTEL_EXPORTER_OTLP_ENDPOINT": "${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}",
      "OTEL_SERVICE_NAME": "${OTEL_SERVICE_NAME:-prompt-improver}",
      "OTEL_RESOURCE_ATTRIBUTES": "${OTEL_RESOURCE_ATTRIBUTES}",
      "METRICS_ENDPOINT": "${METRICS_ENDPOINT:-http://localhost:8080/metrics}",
      "JAEGER_ENDPOINT": "${JAEGER_ENDPOINT:-http://localhost:14268/api/traces}",
      "PROMETHEUS_ENDPOINT": "${PROMETHEUS_ENDPOINT:-http://localhost:9090}"
    },
    "capabilities": [
      "metrics_collection",
      "distributed_tracing",
      "performance_analysis",
      "alerting_management",
      "dashboard_operations",
      "slo_monitoring"
    ]
  }
}
```

### Environment Variables Alignment

To align with the existing project configuration:

```bash
# OpenTelemetry Configuration (aligns with existing setup)
export OTEL_SERVICE_NAME="apes-ml-pipeline"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_RESOURCE_ATTRIBUTES="service.version=1.0.0,environment=production"

# Metrics Endpoints (using existing ports)
export METRICS_ENDPOINT="http://localhost:8000/metrics"
export PROMETHEUS_ENDPOINT="http://localhost:8001"

# Distributed Tracing
export JAEGER_ENDPOINT="http://localhost:14268/api/traces"

# Align with existing metrics config
export METRICS_CONFIG_PATH="/Users/lukemckenzie/prompt-improver/config/metrics_config.json"
```

## Installation & Setup

### Step 1: Install Observability MCP Server

Choose the best observability MCP server for your stack:

#### Option A: Datadog MCP (Recommended for comprehensive monitoring)
```bash
pip install mcp-server-datadog
export DATADOG_API_KEY="your_api_key"
export DATADOG_APP_KEY="your_app_key"
```

#### Option B: Prometheus/OpenTelemetry MCP
```bash
pip install mcp-server-prometheus
# Aligns with existing Prometheus setup on port 8001
```

#### Option C: Custom OTEL MCP Server
```bash
pip install mcp-server-opentelemetry
# Uses existing OpenTelemetry infrastructure
```

### Step 2: Verify Existing Monitoring Stack

Ensure the existing monitoring infrastructure is running:

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus endpoint  
curl http://localhost:8001/metrics

# Verify OpenTelemetry setup
python -c "from prompt_improver.monitoring.opentelemetry.setup import init_telemetry; print('OTEL OK')"
```

### Step 3: Activate MCP Server

```bash
# Add observability MCP server
claude mcp add observability-monitoring

# Verify integration
claude mcp status observability-monitoring
```

## Enhanced performance-engineer Agent Capabilities

### Real-time Performance Monitoring
- **Live Metrics Access**: Direct access to all 13 metric categories
- **SLO Monitoring**: Real-time SLO compliance tracking (99.9% availability target)
- **Threshold Alerting**: Automatic alerts when P95 latency exceeds 2s or P99 exceeds 5s
- **Error Rate Tracking**: Monitoring to maintain <0.1% error rate target

### Distributed Tracing Analysis
- **Request Flow Analysis**: End-to-end tracing through ML pipelines
- **Performance Bottleneck Identification**: Automatic detection of slow components
- **Dependency Mapping**: Visual representation of service dependencies
- **Error Root Cause Analysis**: Trace-based error investigation

### Advanced Performance Analytics
- **ML Model Performance**: Inference duration analysis, prediction throughput optimization
- **Database Performance**: Query duration optimization, connection pool analysis  
- **Cache Performance**: Multi-level cache hit rate optimization (L1/L2/L3)
- **HTTP Performance**: API endpoint optimization and scaling recommendations

### Dashboard & Alerting Management
- **Custom Dashboard Creation**: Performance-engineer can create targeted dashboards
- **Alert Configuration**: Intelligent alerting based on SLO thresholds
- **Performance Reports**: Automated performance analysis and recommendations
- **Capacity Planning**: Resource utilization trends and scaling recommendations

## Usage Examples

### Example 1: Performance Analysis
```
User: "Our API response times are increasing, analyze the performance bottlenecks"
performance-engineer agent → Observability MCP → 
- Retrieves P95/P99 latency metrics
- Analyzes distributed traces for slow operations
- Identifies database queries as bottleneck
- Delegates to database-specialist for optimization
```

### Example 2: SLO Monitoring
```
User: "Are we meeting our SLO targets this month?"
performance-engineer agent → Observability MCP →
- Checks 99.9% availability target (current: 99.95%)
- Validates P95 <2s latency (current: 1.8s average)
- Confirms <0.1% error rate (current: 0.05%)
- Provides SLO compliance report
```

### Example 3: ML Performance Optimization
```
User: "Model inference is slower than expected"
performance-engineer agent → Observability MCP →
- Analyzes ml_model_inference_duration_seconds metrics
- Identifies models exceeding performance thresholds
- Delegates to ml-orchestrator for model optimization
- Sets up monitoring for improvement validation
```

### Example 4: Cache Performance Analysis
```
User: "Optimize our cache performance"
performance-engineer agent → Observability MCP →
- Reviews cache_hits_total vs cache_misses_total metrics
- Analyzes cache performance across L1/L2/L3 levels
- Identifies low hit rate patterns
- Recommends cache strategy improvements
```

## Integration with Existing Architecture

### Alignment with Project Patterns
- **Clean Architecture**: MCP integration respects repository boundaries
- **Multi-Level Caching**: Direct metrics for L1/L2/L3 cache performance
- **Real Behavior Testing**: Monitoring includes testcontainer performance
- **SLO-Driven Development**: Built-in SLO monitoring and compliance tracking

### Performance Baseline Integration
The MCP server integrates with existing performance baselines:
- **Response Time Targets**: P95 <100ms (project achieves <2ms on critical paths)
- **Cache Hit Rates**: >80% target (project achieves 96.67%)
- **Memory Usage**: 10-1000MB range monitoring
- **Test Coverage**: 85%+ service boundary monitoring

### Security & Monitoring
- **Unified Security Integration**: Monitoring respects security component architecture
- **Audit Logging**: All MCP operations logged for security compliance
- **Access Controls**: Role-based access to monitoring data
- **Data Privacy**: Sensitive data filtering in metrics collection

## Advanced Monitoring Scenarios

### Proactive Performance Management
- **Predictive Alerting**: ML-based anomaly detection for performance degradation
- **Capacity Planning**: Automatic scaling recommendations based on usage patterns
- **Performance Regression Detection**: Automated detection of performance regressions
- **Resource Optimization**: Right-sizing recommendations for infrastructure

### Cross-System Performance Analysis
- **End-to-End Latency**: Full request lifecycle analysis
- **Dependency Performance**: Impact analysis of dependent service performance
- **Batch Processing Monitoring**: Batch job performance and optimization
- **ML Pipeline Performance**: Complete ML workflow optimization

## Troubleshooting

### Common Issues

#### MCP Server Connection
```bash
# Test observability endpoint connectivity
curl http://localhost:8000/metrics

# Verify MCP server status
claude mcp status observability-monitoring

# Check OpenTelemetry configuration
python -c "from prompt_improver.monitoring.opentelemetry.setup import get_tracer; print('Tracer OK')"
```

#### Metrics Collection Issues
```bash
# Verify metrics configuration
cat /Users/lukemckenzie/prompt-improver/config/metrics_config.json

# Test metrics endpoint
curl http://localhost:8001/metrics | grep mcp_operations

# Check Prometheus scraping
curl http://localhost:9090/api/v1/query?query=up
```

#### Performance Data Access
```bash
# Test MCP observability integration
claude mcp test observability-monitoring

# Verify OTLP exporter
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
python -c "from opentelemetry import trace; print('OTLP OK')"
```

## Next Steps

1. **Install and configure** observability MCP server
2. **Test integration** with performance-engineer agent
3. **Validate SLO monitoring** and alerting
4. **Optimize performance** based on real-time insights
5. **Document best practices** for ongoing monitoring

## Success Metrics

- **Real-time Performance Insights**: <30s to identify performance issues
- **SLO Compliance**: Maintain 99.9% availability target
- **Performance Optimization**: 20% improvement in issue resolution time
- **Proactive Monitoring**: 80% of issues detected before user impact

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 3*