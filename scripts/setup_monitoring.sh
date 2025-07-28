#!/bin/bash
#
# Comprehensive Monitoring Setup Script for APES ML Pipeline Orchestrator
# Sets up Prometheus, Grafana, and creates dashboards with automated alerting
#
# Created: 2025-07-25
# SRE-Ready Production Monitoring Stack
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/monitoring"
CONFIG_DIR="$PROJECT_ROOT/config"

# Monitoring versions (2025 LTS versions)
PROMETHEUS_VERSION="2.48.1"
GRAFANA_VERSION="10.2.3"
NODE_EXPORTER_VERSION="1.7.0"
ALERTMANAGER_VERSION="0.26.0"
POSTGRES_EXPORTER_VERSION="0.15.0"
REDIS_EXPORTER_VERSION="1.55.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is required but not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=10485760  # 10GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        log_error "Insufficient disk space. At least 10GB is required."
        exit 1
    fi
    
    # Check memory (minimum 4GB)
    AVAILABLE_MEMORY=$(free | grep '^Mem:' | awk '{print $2}')
    REQUIRED_MEMORY=4194304  # 4GB in KB
    
    if [[ $AVAILABLE_MEMORY -lt $REQUIRED_MEMORY ]]; then
        log_warning "Low memory detected. Monitoring stack may be resource constrained."
    fi
    
    log_success "System requirements check passed"
}

# Create directory structure
create_directories() {
    log_info "Creating monitoring directory structure..."
    
    mkdir -p "$MONITORING_DIR"/{prometheus,grafana,alertmanager,exporters}
    mkdir -p "$CONFIG_DIR"/{prometheus,grafana,alertmanager}
    mkdir -p "$CONFIG_DIR/prometheus/rules"
    mkdir -p "$CONFIG_DIR/grafana"/{provisioning,dashboards}
    mkdir -p "$CONFIG_DIR/grafana/provisioning"/{datasources,dashboards,notifiers}
    
    log_success "Directory structure created"
}

# Generate Prometheus configuration
create_prometheus_config() {
    log_info "Creating Prometheus configuration..."
    
    cat > "$CONFIG_DIR/prometheus/prometheus.yml" << 'EOF'
# Prometheus Configuration - Production SRE Setup
# APES ML Pipeline Orchestrator Monitoring

global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'apes-production'
    environment: 'prod'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  - "rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    metrics_path: /metrics

  # APES Application
  - job_name: 'apes-application'
    static_configs:
      - targets: ['apes-app:8000']
    scrape_interval: 5s
    metrics_path: /metrics
    scrape_timeout: 10s
    honor_labels: true
    params:
      format: ['prometheus']

  # APES MCP Server
  - job_name: 'apes-mcp-server'
    static_configs:
      - targets: ['apes-app:8001']
    scrape_interval: 5s
    metrics_path: /metrics
    scrape_timeout: 10s

  # Node Exporter - System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s
    metrics_path: /metrics

  # cAdvisor - Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
    scrape_interval: 15s
    metrics_path: /metrics

  # Redis Exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 15s
    metrics_path: /metrics

  # PostgreSQL Exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 15s
    metrics_path: /metrics

  # APES Health checks
  - job_name: 'apes-health'
    static_configs:
      - targets: ['apes-app:8000']
    scrape_interval: 30s
    metrics_path: /health
    scrape_timeout: 5s

  # APES ML Model metrics
  - job_name: 'apes-ml-metrics'
    static_configs:
      - targets: ['apes-app:8000']
    scrape_interval: 10s
    metrics_path: /api/v1/ml/metrics
    scrape_timeout: 15s

# Storage configuration
storage:
  tsdb:
    retention.time: 30d
    retention.size: 50GB
    wal-compression: true
EOF

    log_success "Prometheus configuration created"
}

# Create alerting rules
create_alerting_rules() {
    log_info "Creating alerting rules..."
    
    # Application-specific alerts
    cat > "$CONFIG_DIR/prometheus/rules/apes-alerts.yml" << 'EOF'
groups:
  - name: apes-application
    rules:
      # High-level service availability
      - alert: APESServiceDown
        expr: up{job="apes-application"} == 0
        for: 1m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "APES application is down"
          description: "APES application has been down for more than 1 minute"
          runbook_url: "https://wiki.company.com/runbooks/apes-service-down"

      - alert: APESMCPServerDown
        expr: up{job="apes-mcp-server"} == 0
        for: 1m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "APES MCP server is down"
          description: "APES MCP server has been down for more than 1 minute"

      # Response time and performance
      - alert: APESHighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "APES high response time"
          description: "95th percentile response time is {{ $value }}s for 5 minutes"

      - alert: APESHighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "APES high error rate"
          description: "Error rate is {{ $value | humanizePercentage }} for 5 minutes"

      # ML-specific alerts
      - alert: APESMLModelLatencyHigh
        expr: histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: warning
          team: ml-ops
        annotations:
          summary: "APES ML model high latency"
          description: "ML model inference latency is {{ $value }}s (95th percentile)"

      - alert: APESMLModelErrorRateHigh
        expr: rate(ml_model_errors_total[5m]) / rate(ml_model_predictions_total[5m]) > 0.02
        for: 3m
        labels:
          severity: warning
          team: ml-ops
        annotations:
          summary: "APES ML model high error rate"
          description: "ML model error rate is {{ $value | humanizePercentage }}"

      # Database connectivity
      - alert: APESDatabaseConnectionsHigh
        expr: db_connections_active / db_connections_max > 0.8
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "APES database connections high"
          description: "Database connection usage is {{ $value | humanizePercentage }}"

      # Redis connectivity and performance
      - alert: APESRedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "Redis is down"
          description: "Redis instance has been down for more than 1 minute"

      - alert: APESRedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "Redis memory usage high"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"
EOF

    # System-level alerts
    cat > "$CONFIG_DIR/prometheus/rules/system-alerts.yml" << 'EOF'
groups:
  - name: system-resources
    rules:
      # CPU alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% on instance {{ $labels.instance }}"

      # Memory alerts
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% on instance {{ $labels.instance }}"

      - alert: CriticalMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 95
        for: 2m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "Critical memory usage detected"
          description: "Memory usage is {{ $value }}% on instance {{ $labels.instance }}"

      # Disk space alerts
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"}) * 100 < 20
        for: 5m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "Disk space low"
          description: "Disk space is {{ $value }}% on {{ $labels.instance }}:{{ $labels.mountpoint }}"

      - alert: DiskSpaceCritical
        expr: (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"}) * 100 < 10
        for: 2m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "Disk space critical"
          description: "Disk space is {{ $value }}% on {{ $labels.instance }}:{{ $labels.mountpoint }}"

      # Load average alerts
      - alert: HighLoadAverage
        expr: node_load15 / on(instance) group_left count by(instance) (node_cpu_seconds_total{mode="idle"}) > 1.5
        for: 10m
        labels:
          severity: warning
          team: sre
        annotations:
          summary: "High load average detected"
          description: "Load average is {{ $value }} on instance {{ $labels.instance }}"
EOF

    log_success "Alerting rules created"
}

# Create Grafana datasource configuration
create_grafana_datasources() {
    log_info "Creating Grafana datasource configuration..."
    
    cat > "$CONFIG_DIR/grafana/provisioning/datasources/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    basicAuth: false
    jsonData:
      httpMethod: POST
      prometheusType: Prometheus
      prometheusVersion: 2.48.1
      cacheLevel: 'High'
      disableMetricsLookup: false
      customQueryParameters: ''
      timeInterval: "30s"
    secureJsonData: {}
EOF

    log_success "Grafana datasources configured"
}

# Create Grafana dashboard provisioning
create_grafana_dashboard_provisioning() {
    log_info "Creating Grafana dashboard provisioning..."
    
    cat > "$CONFIG_DIR/grafana/provisioning/dashboards/dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'APES Dashboards'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    log_success "Grafana dashboard provisioning configured"
}

# Create comprehensive dashboards
create_grafana_dashboards() {
    log_info "Creating Grafana dashboards..."
    
    # APES Application Overview Dashboard
    cat > "$CONFIG_DIR/grafana/dashboards/apes-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "APES - Application Overview",
    "tags": ["apes", "application", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"apes-application\"}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {
                "options": {
                  "0": {
                    "text": "DOWN",
                    "color": "red"
                  },
                  "1": {
                    "text": "UP",
                    "color": "green"
                  }
                },
                "type": "value"
              }
            ]
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Response Time (95th percentile)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    # System Resources Dashboard
    cat > "$CONFIG_DIR/grafana/dashboards/system-resources.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "System Resources",
    "tags": ["system", "resources", "infrastructure"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Disk Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "(1 - (node_filesystem_avail_bytes{fstype!=\"tmpfs\"} / node_filesystem_size_bytes{fstype!=\"tmpfs\"})) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Network I/O",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total[5m])",
            "refId": "A",
            "legendFormat": "Receive"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total[5m])",
            "refId": "B",
            "legendFormat": "Transmit"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "Bps"
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    # ML Pipeline Dashboard
    cat > "$CONFIG_DIR/grafana/dashboards/ml-pipeline.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "ML Pipeline Monitoring",
    "tags": ["ml", "pipeline", "models"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "ML Model Predictions/sec",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total[5m])",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "ML Model Latency (95th percentile)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s"
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "ML Model Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(ml_model_errors_total[5m]) / rate(ml_model_predictions_total[5m])",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit"
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Batch Processing Queue Size",
        "type": "timeseries",
        "targets": [
          {
            "expr": "batch_processing_queue_size",
            "refId": "A"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log_success "Grafana dashboards created"
}

# Create Alertmanager configuration
create_alertmanager_config() {
    log_info "Creating Alertmanager configuration..."
    
    cat > "$CONFIG_DIR/alertmanager/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'
  smtp_auth_username: 'alerts@company.com'
  smtp_auth_password: 'password'

# The directory from which notification templates are read.
templates:
- '/etc/alertmanager/templates/*.tmpl'

# The root route on which each incoming alert enters.
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 0s
    repeat_interval: 5m
  - match:
      severity: warning
    receiver: 'warning-alerts'
    repeat_interval: 30m

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true

- name: 'critical-alerts'
  email_configs:
  - to: 'sre-team@company.com'
    subject: 'CRITICAL: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Runbook: {{ .Annotations.runbook_url }}
      {{ end }}
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#critical-alerts'
    title: 'CRITICAL Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  email_configs:
  - to: 'dev-team@company.com'
    subject: 'WARNING: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF

    log_success "Alertmanager configuration created"
}

# Create Docker Compose for monitoring stack
create_monitoring_compose() {
    log_info "Creating monitoring Docker Compose configuration..."
    
    cat > "$MONITORING_DIR/docker-compose.yml" << EOF
# Production Monitoring Stack - Docker Compose
# APES ML Pipeline Orchestrator

version: '3.8'

services:
  # Prometheus - Metrics collection and storage
  prometheus:
    image: prom/prometheus:v${PROMETHEUS_VERSION}
    container_name: apes-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ${CONFIG_DIR}/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ${CONFIG_DIR}/prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=50GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--log.level=info'
      - '--web.route-prefix=/'
      - '--web.external-url=http://localhost:9090'
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Grafana - Visualization and dashboards
  grafana:
    image: grafana/grafana:${GRAFANA_VERSION}
    container_name: apes-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ${CONFIG_DIR}/grafana/provisioning:/etc/grafana/provisioning:ro
      - ${CONFIG_DIR}/grafana/dashboards:/var/lib/grafana/dashboards:ro
    environment:
      # Security settings
      - GF_SECURITY_ADMIN_USER=\${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD:-admin123}
      - GF_SECURITY_SECRET_KEY=\${GRAFANA_SECRET_KEY:-your-secret-key-here}
      
      # Server settings
      - GF_SERVER_DOMAIN=\${GRAFANA_DOMAIN:-localhost}
      - GF_SERVER_ROOT_URL=\${GRAFANA_ROOT_URL:-http://localhost:3000}
      
      # Database settings
      - GF_DATABASE_TYPE=sqlite3
      - GF_DATABASE_PATH=/var/lib/grafana/grafana.db
      
      # Analytics and telemetry
      - GF_ANALYTICS_REPORTING_ENABLED=false
      - GF_ANALYTICS_CHECK_FOR_UPDATES=false
      
      # Logging
      - GF_LOG_LEVEL=info
      - GF_LOG_MODE=console
      
      # Feature toggles
      - GF_FEATURE_TOGGLES_ENABLE=publicDashboards,lokiLive,traceqlEditor
      
      # Authentication
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    networks:
      - monitoring
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Alertmanager - Alert handling
  alertmanager:
    image: prom/alertmanager:v${ALERTMANAGER_VERSION}
    container_name: apes-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ${CONFIG_DIR}/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--web.route-prefix=/'
      - '--cluster.listen-address='
      - '--log.level=info'
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Node Exporter - System metrics
  node-exporter:
    image: prom/node-exporter:v${NODE_EXPORTER_VERSION}
    container_name: apes-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)(\$\$|/)'
      - '--collector.systemd'
      - '--collector.processes'
      - '--web.listen-address=:9100'
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9100/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3

  # cAdvisor - Container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.2
    container_name: apes-cadvisor
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:v${POSTGRES_EXPORTER_VERSION}
    container_name: apes-postgres-exporter
    restart: unless-stopped
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://\${POSTGRES_USER:-apes_user}:\${POSTGRES_PASSWORD:-secure_password_2025}@\${POSTGRES_HOST:-postgres}:5432/\${POSTGRES_DB:-apes}?sslmode=disable
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9187/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:v${REDIS_EXPORTER_VERSION}
    container_name: apes-redis-exporter
    restart: unless-stopped
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=\${REDIS_HOST:-redis}:6379
      - REDIS_PASSWORD=\${REDIS_PASSWORD:-}
    networks:
      - monitoring
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9121/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3

# Networks
networks:
  monitoring:
    driver: bridge
    name: apes-monitoring

# Volumes
volumes:
  prometheus_data:
    driver: local
    name: apes-prometheus-data
    
  grafana_data:
    driver: local
    name: apes-grafana-data
    
  alertmanager_data:
    driver: local
    name: apes-alertmanager-data
EOF

    log_success "Monitoring Docker Compose configuration created"
}

# Create environment file template
create_env_template() {
    log_info "Creating environment template..."
    
    cat > "$MONITORING_DIR/.env.template" << 'EOF'
# APES Monitoring Stack Environment Variables
# Copy this file to .env and update with your values

# Grafana Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-password-here
GRAFANA_SECRET_KEY=your-secret-key-here
GRAFANA_DOMAIN=localhost
GRAFANA_ROOT_URL=http://localhost:3000

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_USER=apes_user
POSTGRES_PASSWORD=your-postgres-password
POSTGRES_DB=apes

# Redis Configuration
REDIS_HOST=redis
REDIS_PASSWORD=

# SMTP Configuration (for alerts)
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USER=alerts@company.com
SMTP_PASSWORD=your-smtp-password

# Slack Configuration (for alerts)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EOF

    if [[ ! -f "$MONITORING_DIR/.env" ]]; then
        cp "$MONITORING_DIR/.env.template" "$MONITORING_DIR/.env"
        log_info "Created .env file from template. Please update with your values."
    fi
    
    log_success "Environment template created"
}

# Create management scripts
create_management_scripts() {
    log_info "Creating management scripts..."
    
    # Start script
    cat > "$MONITORING_DIR/start-monitoring.sh" << 'EOF'
#!/bin/bash
# Start APES monitoring stack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting APES monitoring stack..."

# Check if .env file exists
if [[ ! -f .env ]]; then
    echo "Error: .env file not found. Please copy .env.template to .env and configure."
    exit 1
fi

# Start the monitoring stack
docker-compose up -d

echo "Monitoring stack started successfully!"
echo ""
echo "Access URLs:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000"
echo "- Alertmanager: http://localhost:9093"
echo ""
echo "Default Grafana credentials: admin / admin123 (change in .env file)"
EOF

    # Stop script
    cat > "$MONITORING_DIR/stop-monitoring.sh" << 'EOF'
#!/bin/bash
# Stop APES monitoring stack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping APES monitoring stack..."
docker-compose down

echo "Monitoring stack stopped."
EOF

    # Status script
    cat > "$MONITORING_DIR/status-monitoring.sh" << 'EOF'
#!/bin/bash
# Check status of APES monitoring stack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "APES Monitoring Stack Status:"
echo "============================="
docker-compose ps

echo ""
echo "Container Health:"
echo "=================="
docker-compose exec prometheus wget --quiet --tries=1 --spider http://localhost:9090/-/healthy && echo "✓ Prometheus: Healthy" || echo "✗ Prometheus: Unhealthy"
docker-compose exec grafana curl -f http://localhost:3000/api/health &>/dev/null && echo "✓ Grafana: Healthy" || echo "✗ Grafana: Unhealthy"
docker-compose exec alertmanager wget --quiet --tries=1 --spider http://localhost:9093/-/healthy &>/dev/null && echo "✓ Alertmanager: Healthy" || echo "✗ Alertmanager: Unhealthy"
EOF

    # Update script
    cat > "$MONITORING_DIR/update-monitoring.sh" << 'EOF'
#!/bin/bash
# Update APES monitoring stack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Updating APES monitoring stack..."

# Pull latest images
docker-compose pull

# Restart services
docker-compose up -d

echo "Monitoring stack updated successfully!"
EOF

    # Make scripts executable
    chmod +x "$MONITORING_DIR"/*.sh
    
    log_success "Management scripts created"
}

# Create backup script
create_backup_script() {
    log_info "Creating backup script..."
    
    cat > "$MONITORING_DIR/backup-monitoring.sh" << 'EOF'
#!/bin/bash
# Backup APES monitoring data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="$SCRIPT_DIR/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Creating backup of monitoring data..."

# Backup Prometheus data
echo "Backing up Prometheus data..."
docker run --rm -v apes-prometheus-data:/data -v "$BACKUP_DIR":/backup alpine:latest tar czf "/backup/prometheus_$TIMESTAMP.tar.gz" -C /data .

# Backup Grafana data
echo "Backing up Grafana data..."
docker run --rm -v apes-grafana-data:/data -v "$BACKUP_DIR":/backup alpine:latest tar czf "/backup/grafana_$TIMESTAMP.tar.gz" -C /data .

# Backup Alertmanager data
echo "Backing up Alertmanager data..."
docker run --rm -v apes-alertmanager-data:/data -v "$BACKUP_DIR":/backup alpine:latest tar czf "/backup/alertmanager_$TIMESTAMP.tar.gz" -C /data .

# Clean up old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed successfully!"
echo "Backup files created in: $BACKUP_DIR"
EOF

    chmod +x "$MONITORING_DIR/backup-monitoring.sh"
    
    log_success "Backup script created"
}

# Create health check script
create_health_check_script() {
    log_info "Creating health check script..."
    
    cat > "$MONITORING_DIR/health-check.sh" << 'EOF'
#!/bin/bash
# Comprehensive health check for APES monitoring stack

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓${NC} $service_name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is unhealthy"
        return 1
    fi
}

echo "APES Monitoring Stack Health Check"
echo "=================================="
echo ""

# Check if containers are running
echo "Container Status:"
docker-compose ps --format "table {{.Name}}\t{{.Status}}" | grep -v "Exit 0" || true
echo ""

# Service health checks
echo "Service Health Checks:"
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"
check_service "Alertmanager" "http://localhost:9093/-/healthy"
check_service "Node Exporter" "http://localhost:9100/metrics" "200"
check_service "cAdvisor" "http://localhost:8080/healthz"

echo ""

# Check Prometheus targets
echo "Prometheus Targets Status:"
TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job): \(.health)"' 2>/dev/null || echo "Failed to fetch targets")
echo "$TARGETS"

echo ""

# Check disk usage
echo "Storage Usage:"
docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}"

echo ""
echo "Health check completed."
EOF

    chmod +x "$MONITORING_DIR/health-check.sh"
    
    log_success "Health check script created"
}

# Create README
create_readme() {
    log_info "Creating README documentation..."
    
    cat > "$MONITORING_DIR/README.md" << 'EOF'
# APES Monitoring Stack

This directory contains a comprehensive monitoring setup for the APES (Adaptive Prompt Enhancement System) ML Pipeline Orchestrator.

## Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert handling and notification
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics
- **PostgreSQL Exporter**: Database metrics
- **Redis Exporter**: Cache metrics

## Quick Start

1. **Configure Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

2. **Start Monitoring Stack**:
   ```bash
   ./start-monitoring.sh
   ```

3. **Access Services**:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000
   - Alertmanager: http://localhost:9093

## Management Commands

- `./start-monitoring.sh` - Start the monitoring stack
- `./stop-monitoring.sh` - Stop the monitoring stack
- `./status-monitoring.sh` - Check status of all services
- `./update-monitoring.sh` - Update to latest versions
- `./backup-monitoring.sh` - Backup monitoring data
- `./health-check.sh` - Comprehensive health check

## Dashboards

The following Grafana dashboards are automatically provisioned:

1. **APES Application Overview** - High-level application metrics
2. **System Resources** - CPU, memory, disk, network metrics
3. **ML Pipeline Monitoring** - ML-specific metrics and performance

## Alerting

Alert rules are configured for:

- **Critical Alerts**:
  - Service down
  - High error rates
  - Critical resource usage

- **Warning Alerts**:
  - High response times
  - Resource usage warnings
  - ML model performance issues

## Configuration

### Environment Variables

Key environment variables in `.env`:

- `GRAFANA_ADMIN_USER` - Grafana admin username
- `GRAFANA_ADMIN_PASSWORD` - Grafana admin password
- `POSTGRES_*` - Database connection settings
- `REDIS_*` - Redis connection settings

### Customization

To customize the monitoring setup:

1. **Add New Dashboards**: Place JSON files in `../config/grafana/dashboards/`
2. **Modify Alert Rules**: Edit files in `../config/prometheus/rules/`
3. **Update Scrape Targets**: Modify `../config/prometheus/prometheus.yml`

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the Docker daemon is running and your user is in the docker group
2. **Port Conflicts**: Check if ports 3000, 9090, 9093 are available
3. **Disk Space**: Ensure sufficient disk space for metrics storage

### Logs

View logs for troubleshooting:
```bash
docker-compose logs -f [service-name]
```

### Health Checks

Run comprehensive health check:
```bash
./health-check.sh
```

## Security Considerations

- Change default Grafana credentials
- Configure proper authentication for production
- Set up TLS/SSL for external access
- Restrict network access using firewalls
- Regular backup of monitoring data

## Backup and Recovery

- Automated backups: `./backup-monitoring.sh`
- Backups stored in `./backups/` directory
- Retention: 7 days by default

## Monitoring Best Practices

1. **SLI/SLO Definition**: Define clear Service Level Indicators and Objectives
2. **Alert Fatigue**: Avoid excessive alerting by tuning thresholds
3. **Runbooks**: Maintain runbooks for alert response
4. **Capacity Planning**: Monitor trends for capacity planning
5. **Regular Review**: Regularly review and update monitoring configuration

## Support

For issues with the monitoring setup:

1. Check logs using `docker-compose logs`
2. Run health check: `./health-check.sh`
3. Consult the troubleshooting section above
4. Review Prometheus/Grafana documentation

## Version Information

- Prometheus: v2.48.1
- Grafana: v10.2.3
- Alertmanager: v0.26.0
- Node Exporter: v1.7.0

Last updated: 2025-07-25
EOF

    log_success "README documentation created"
}

# Main installation function
main() {
    log_info "Starting APES monitoring setup..."
    
    check_root
    check_requirements
    create_directories
    create_prometheus_config
    create_alerting_rules
    create_grafana_datasources
    create_grafana_dashboard_provisioning
    create_grafana_dashboards
    create_alertmanager_config
    create_monitoring_compose
    create_env_template
    create_management_scripts
    create_backup_script
    create_health_check_script
    create_readme
    
    log_success "APES monitoring setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Configure environment: cd $MONITORING_DIR && cp .env.template .env"
    echo "2. Edit .env file with your configuration"
    echo "3. Start monitoring: cd $MONITORING_DIR && ./start-monitoring.sh"
    echo ""
    echo "Access URLs:"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin123)"
    echo "- Alertmanager: http://localhost:9093"
    echo ""
    echo "Documentation: $MONITORING_DIR/README.md"
}

# Run main function
main "$@"