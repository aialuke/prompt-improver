# APES Monitoring Stack Setup Guide

This guide provides comprehensive instructions for setting up production-ready monitoring for the APES (Adaptive Prompt Enhancement System) ML Pipeline Orchestrator.

## Overview

The monitoring stack provides complete observability for:
- **Application Performance**: HTTP requests, response times, error rates
- **ML Pipeline Metrics**: Model predictions, inference latency, error tracking
- **System Resources**: CPU, memory, disk, network utilization
- **Infrastructure Health**: Database, Redis, container metrics
- **Business Metrics**: User activity, prompt improvements, system usage

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  APES           │───▶│  Prometheus      │───▶│  Grafana        │
│  Application    │    │  (Metrics)       │    │  (Dashboards)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Alertmanager    │    │  Health Checks  │
                       │  (Notifications) │    │  (Monitoring)   │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Setup Monitoring Infrastructure

```bash
# Run the monitoring setup script
./scripts/setup_monitoring.sh

# This will create:
# - monitoring/ directory with Docker Compose configuration
# - config/ directory with Prometheus, Grafana, and Alertmanager configs
# - Pre-built dashboards and alerting rules
# - Management scripts for operations
```

### 2. Configure Environment

```bash
cd monitoring
cp .env.template .env

# Edit .env with your configuration
vim .env
```

### 3. Setup Application Metrics

```bash
# Setup application-specific metrics collection
./scripts/setup_app_metrics.py

# This will create:
# - Prometheus metrics middleware
# - Health check endpoints
# - Configuration files
# - Usage examples
```

### 4. Start Monitoring Stack

```bash
cd monitoring
./start-monitoring.sh
```

### 5. Access Monitoring Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Alertmanager**: http://localhost:9093

## Detailed Setup Instructions

### Infrastructure Components

#### Prometheus Configuration

The Prometheus setup includes:
- **Retention**: 30 days of metrics data
- **Storage**: 50GB limit with compression
- **Scrape Intervals**: 5-30s depending on criticality
- **Targets**: Application, system, and infrastructure metrics

Key scrape targets:
```yaml
scrape_configs:
  - job_name: 'apes-application'
    static_configs:
      - targets: ['apes-app:8000']
    scrape_interval: 5s
    
  - job_name: 'apes-mcp-server'
    static_configs:
      - targets: ['apes-app:8001']
    scrape_interval: 5s
```

#### Grafana Dashboards

Pre-configured dashboards include:

1. **APES Application Overview**
   - Service status and uptime
   - Request rate and response times
   - Error rates and success metrics
   - User activity patterns

2. **System Resources**
   - CPU utilization and load average
   - Memory usage and swap activity
   - Disk I/O and space utilization
   - Network traffic patterns

3. **ML Pipeline Monitoring**
   - Model prediction rates
   - Inference latency (P50, P95, P99)
   - Model error rates by type
   - Batch processing queue status

#### Alerting Rules

Critical alerts include:
- **Service Down**: Application unavailable > 1 minute
- **High Error Rate**: >5% error rate for 5 minutes
- **High Response Time**: P95 latency >2s for 5 minutes
- **Resource Exhaustion**: >85% memory/disk usage
- **ML Model Issues**: High inference latency or error rates

### Application Integration

#### Metrics Middleware

The monitoring setup creates FastAPI middleware for automatic metrics collection:

```python
from prompt_improver.monitoring import PrometheusMiddleware

app = FastAPI()
app.add_middleware(PrometheusMiddleware, metrics_enabled=True)
```

#### Health Check Endpoints

Comprehensive health checks are available at:
- `/health` - Full system health status
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe

#### Custom Metrics

Record application-specific metrics:

```python
from prompt_improver.monitoring.metrics_middleware import (
    record_ml_prediction,
    record_database_query,
    record_cache_hit
)

# Record ML operations
await record_ml_prediction("gpt-4", "v1.0", inference_time)

# Record database operations  
await record_database_query("SELECT", "prompts", query_time)

# Record cache operations
record_cache_hit("redis", "user_sessions")
```

## Operational Procedures

### Starting the Stack

```bash
cd monitoring
./start-monitoring.sh
```

### Stopping the Stack

```bash
cd monitoring
./stop-monitoring.sh
```

### Checking Status

```bash
cd monitoring
./status-monitoring.sh
```

### Health Validation

```bash
./scripts/validate_monitoring.sh
```

### Backup and Recovery

```bash
cd monitoring
./backup-monitoring.sh

# Backups are stored in monitoring/backups/
# Retention: 7 days (configurable)
```

## Configuration Management

### Environment Variables

Key configuration in `monitoring/.env`:

```env
# Grafana Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-password
GRAFANA_SECRET_KEY=your-secret-key

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_USER=apes_user
POSTGRES_PASSWORD=your-postgres-password

# Redis Configuration  
REDIS_HOST=redis
REDIS_PASSWORD=

# SMTP Configuration (for alerts)
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USER=alerts@company.com
SMTP_PASSWORD=your-smtp-password
```

### Custom Dashboards

Add custom dashboards by placing JSON files in:
```
config/grafana/dashboards/your-dashboard.json
```

### Custom Alert Rules

Add custom alert rules in:
```
config/prometheus/rules/your-rules.yml
```

### Scrape Target Configuration

Modify scrape targets in:
```
config/prometheus/prometheus.yml
```

## Metrics Reference

### Application Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `http_requests_total` | Counter | Total HTTP requests | method, endpoint, status |
| `http_request_duration_seconds` | Histogram | Request duration | method, endpoint |
| `ml_model_predictions_total` | Counter | ML predictions | model_name, model_version |
| `ml_model_inference_duration_seconds` | Histogram | Inference duration | model_name, model_version |
| `db_connections_active` | Gauge | Active DB connections | database, pool |
| `cache_hits_total` | Counter | Cache hits | cache_type, key_pattern |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `node_cpu_seconds_total` | Counter | CPU time per mode |
| `node_memory_MemTotal_bytes` | Gauge | Total system memory |
| `node_filesystem_size_bytes` | Gauge | Filesystem size |
| `container_cpu_usage_seconds_total` | Counter | Container CPU usage |

### ML Pipeline Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `batch_processing_queue_size` | Gauge | Queue size | queue_name |
| `ml_model_errors_total` | Counter | Model errors | model_name, error_type |
| `mcp_operations_total` | Counter | MCP operations | operation, status |

## SLI/SLO Definitions

### Service Level Indicators (SLIs)

1. **Availability**: Percentage of successful requests
   - Measurement: `(successful_requests / total_requests) * 100`
   - Target: 99.9% (43.2 minutes downtime/month)

2. **Latency**: Request response time
   - Measurement: 95th percentile response time
   - Target: <2 seconds

3. **Error Rate**: Percentage of failed requests
   - Measurement: `(error_requests / total_requests) * 100`
   - Target: <0.1%

### Service Level Objectives (SLOs)

- **Availability SLO**: 99.9% uptime over 30-day window
- **Latency SLO**: 95% of requests completed in <2s
- **Error Rate SLO**: <0.1% error rate over 24-hour window

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Targets

1. Check target configuration in `prometheus.yml`
2. Verify network connectivity between containers
3. Check application metrics endpoint is accessible
4. Review Prometheus logs: `docker-compose logs prometheus`

#### Grafana Dashboard Not Loading

1. Verify Prometheus datasource configuration
2. Check datasource connectivity in Grafana settings
3. Ensure dashboard JSON is valid
4. Review Grafana logs: `docker-compose logs grafana`

#### High Memory Usage

1. Check Prometheus TSDB retention settings
2. Review metric cardinality (too many label combinations)
3. Adjust scrape intervals for non-critical metrics
4. Consider using recording rules for complex queries

#### Alert Fatigue

1. Review alert thresholds and adjust as needed
2. Implement proper alert grouping and inhibition
3. Add runbooks for common alerts
4. Use alert severity levels appropriately

### Debugging Commands

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f [service-name]

# Test metric endpoints
curl http://localhost:8000/metrics
curl http://localhost:9100/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test alerting rules
curl http://localhost:9090/api/v1/rules

# Validate configuration
docker-compose config

# Health check
./monitoring/health-check.sh
```

## Security Considerations

### Authentication and Authorization

1. **Change Default Credentials**: Update Grafana admin password
2. **Enable HTTPS**: Configure TLS certificates for production
3. **Network Segmentation**: Use firewalls to restrict access
4. **RBAC**: Configure role-based access in Grafana

### Data Protection

1. **Encryption**: Enable encryption at rest for sensitive metrics
2. **Retention**: Configure appropriate data retention policies
3. **Access Logs**: Monitor access to monitoring systems
4. **Backup Security**: Secure backup storage and access

### Production Hardening

```yaml
# Example security configuration
grafana:
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=${SECURE_PASSWORD}
    - GF_SECURITY_SECRET_KEY=${RANDOM_SECRET_KEY}
    - GF_SERVER_CERT_FILE=/etc/ssl/certs/grafana.crt
    - GF_SERVER_CERT_KEY=/etc/ssl/private/grafana.key
    - GF_AUTH_ANONYMOUS_ENABLED=false
```

## Performance Optimization

### Metrics Collection

1. **Adjust Scrape Intervals**: Balance freshness vs. resource usage
2. **Metric Cardinality**: Limit label combinations to prevent explosion
3. **Recording Rules**: Pre-compute expensive queries
4. **Retention Tuning**: Balance storage vs. historical data needs

### Dashboard Performance

1. **Query Optimization**: Use efficient PromQL queries
2. **Time Range Limits**: Set reasonable default time ranges
3. **Refresh Rates**: Don't refresh too frequently
4. **Panel Limits**: Limit number of panels per dashboard

### Resource Management

```yaml
# Example resource limits
prometheus:
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: '2'
      reservations:
        memory: 2G
        cpus: '1'
```

## Maintenance and Updates

### Regular Maintenance Tasks

1. **Weekly**: Review alert noise and adjust thresholds
2. **Monthly**: Analyze metrics trends and capacity planning
3. **Quarterly**: Update monitoring stack versions
4. **Annually**: Review SLI/SLO targets and alerting strategy

### Update Procedure

```bash
cd monitoring
./update-monitoring.sh

# Or manually:
docker-compose pull
docker-compose up -d
```

### Backup Strategy

1. **Automated Backups**: Daily backup of monitoring data
2. **Configuration Backup**: Version control monitoring configs
3. **Recovery Testing**: Regularly test backup restoration
4. **Offsite Storage**: Store backups in separate location

## Best Practices

### Monitoring Strategy

1. **Start Simple**: Begin with basic golden signals (latency, traffic, errors, saturation)
2. **Gradual Expansion**: Add more detailed metrics based on operational needs
3. **Alert Thoughtfully**: Only alert on actionable conditions
4. **Document Everything**: Maintain runbooks and monitoring documentation

### Operational Excellence

1. **Regular Reviews**: Weekly monitoring health checks
2. **Capacity Planning**: Monitor trends for resource planning
3. **Incident Learning**: Improve monitoring based on incidents
4. **Team Training**: Ensure team understands monitoring tools

### Development Integration

1. **Monitoring as Code**: Version control all configurations
2. **Testing**: Test monitoring changes in staging
3. **CI/CD Integration**: Automated deployment of monitoring updates
4. **Documentation**: Keep monitoring docs up to date

## Support and Resources

### Internal Documentation

- [Health Check Implementation](./ML_HEALTH_MONITORING_GUIDE.md)
- [Redis Caching Patterns](./research/redis_caching.md)
- [Performance Optimization](./performance/redis_cache_benchmark.md)

### External Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)

### Getting Help

1. Check monitoring health: `./scripts/validate_monitoring.sh`
2. Review logs: `docker-compose logs [service]`
3. Consult troubleshooting section above
4. Check component documentation
5. Contact SRE team for escalation

---

**Last Updated**: 2025-07-25  
**Version**: 1.0  
**Maintainer**: SRE Team