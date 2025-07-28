#!/bin/bash
#
# Monitoring Stack Validation Script for APES
# Validates that all monitoring components are working correctly
#
# Created: 2025-07-25
# SRE Operational Validation Tool
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test if service is responding
test_service_health() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    local timeout=${4:-10}
    
    log_info "Testing $service_name health at $url"
    
    if curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" | grep -q "$expected_status"; then
        log_success "$service_name is responding correctly"
        return 0
    else
        log_failure "$service_name is not responding or returned unexpected status"
        return 1
    fi
}

# Test Prometheus query
test_prometheus_query() {
    local query=$1
    local description=$2
    
    log_info "Testing Prometheus query: $description"
    
    local response
    response=$(curl -s "http://localhost:9090/api/v1/query?query=$query" || echo "")
    
    if echo "$response" | jq -e '.status == "success"' >/dev/null 2>&1; then
        local result_count
        result_count=$(echo "$response" | jq '.data.result | length')
        if [[ $result_count -gt 0 ]]; then
            log_success "$description - Query returned $result_count results"
            return 0
        else
            log_failure "$description - Query returned no results"
            return 1
        fi
    else
        log_failure "$description - Query failed or returned error"
        return 1
    fi
}

# Test Docker containers
test_docker_containers() {
    log_info "Testing Docker containers status"
    
    cd "$MONITORING_DIR" || {
        log_failure "Cannot access monitoring directory"
        return 1
    }
    
    # Check if containers are running
    local containers=("apes-prometheus" "apes-grafana" "apes-alertmanager" "apes-node-exporter" "apes-cadvisor")
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log_success "Container $container is running"
        else
            log_failure "Container $container is not running"
        fi
    done
}

# Test Prometheus targets
test_prometheus_targets() {
    log_info "Testing Prometheus targets"
    
    local targets_response
    targets_response=$(curl -s "http://localhost:9090/api/v1/targets" || echo "")
    
    if [[ -n "$targets_response" ]]; then
        local healthy_targets
        local total_targets
        
        healthy_targets=$(echo "$targets_response" | jq '[.data.activeTargets[] | select(.health == "up")] | length' 2>/dev/null || echo "0")
        total_targets=$(echo "$targets_response" | jq '.data.activeTargets | length' 2>/dev/null || echo "0")
        
        if [[ $healthy_targets -eq $total_targets ]] && [[ $total_targets -gt 0 ]]; then
            log_success "All $total_targets Prometheus targets are healthy"
        else
            log_failure "$healthy_targets out of $total_targets Prometheus targets are healthy"
            
            # Show unhealthy targets
            echo "$targets_response" | jq -r '.data.activeTargets[] | select(.health != "up") | "  - \(.labels.job)@\(.scrapeUrl): \(.lastError // "unknown error")"' 2>/dev/null || true
        fi
    else
        log_failure "Cannot retrieve Prometheus targets"
    fi
}

# Test Grafana datasources
test_grafana_datasources() {
    log_info "Testing Grafana datasources"
    
    # Default credentials (should be changed in production)
    local grafana_url="http://admin:admin123@localhost:3000"
    
    local datasources_response
    datasources_response=$(curl -s "$grafana_url/api/datasources" || echo "")
    
    if [[ -n "$datasources_response" ]]; then
        local datasource_count
        datasource_count=$(echo "$datasources_response" | jq 'length' 2>/dev/null || echo "0")
        
        if [[ $datasource_count -gt 0 ]]; then
            log_success "Grafana has $datasource_count datasource(s) configured"
            
            # Test datasource connectivity
            local prometheus_ds_id
            prometheus_ds_id=$(echo "$datasources_response" | jq -r '.[] | select(.type == "prometheus") | .id' 2>/dev/null || echo "")
            
            if [[ -n "$prometheus_ds_id" ]]; then
                local test_response
                test_response=$(curl -s "$grafana_url/api/datasources/$prometheus_ds_id/proxy/api/v1/label/__name__/values" || echo "")
                
                if echo "$test_response" | jq -e '.status == "success"' >/dev/null 2>&1; then
                    log_success "Prometheus datasource connectivity test passed"
                else
                    log_failure "Prometheus datasource connectivity test failed"
                fi
            fi
        else
            log_failure "No Grafana datasources configured"
        fi
    else
        log_failure "Cannot retrieve Grafana datasources (check credentials)"
    fi
}

# Test alert rules
test_alert_rules() {
    log_info "Testing Prometheus alert rules"
    
    local rules_response
    rules_response=$(curl -s "http://localhost:9090/api/v1/rules" || echo "")
    
    if [[ -n "$rules_response" ]]; then
        local rule_groups
        rule_groups=$(echo "$rules_response" | jq '.data.groups | length' 2>/dev/null || echo "0")
        
        if [[ $rule_groups -gt 0 ]]; then
            log_success "Prometheus has $rule_groups alert rule group(s) loaded"
            
            # Check for any rule evaluation errors
            local rule_errors
            rule_errors=$(echo "$rules_response" | jq -r '.data.groups[].rules[] | select(.lastError != null) | .name + ": " + .lastError' 2>/dev/null || echo "")
            
            if [[ -z "$rule_errors" ]]; then
                log_success "All alert rules are evaluating without errors"
            else
                log_failure "Some alert rules have evaluation errors:"
                echo "$rule_errors" | while read -r error; do
                    echo "  - $error"
                done
            fi
        else
            log_failure "No alert rules loaded"
        fi
    else
        log_failure "Cannot retrieve alert rules"
    fi
}

# Test metric ingestion
test_metric_ingestion() {
    log_info "Testing metric ingestion"
    
    # Test for basic system metrics
    local test_metrics=(
        "up"
        "node_cpu_seconds_total"
        "node_memory_MemTotal_bytes"
        "container_cpu_usage_seconds_total"
    )
    
    for metric in "${test_metrics[@]}"; do
        test_prometheus_query "$metric" "Metric availability: $metric"
    done
}

# Test alert manager
test_alertmanager() {
    log_info "Testing Alertmanager"
    
    test_service_health "Alertmanager" "http://localhost:9093/-/healthy"
    
    # Test Alertmanager configuration
    local config_response
    config_response=$(curl -s "http://localhost:9093/api/v1/status" || echo "")
    
    if [[ -n "$config_response" ]]; then
        if echo "$config_response" | jq -e '.status == "success"' >/dev/null 2>&1; then
            log_success "Alertmanager configuration is valid"
        else
            log_failure "Alertmanager configuration has issues"
        fi
    else
        log_failure "Cannot retrieve Alertmanager status"
    fi
}

# Test storage usage
test_storage_usage() {
    log_info "Testing storage usage"
    
    # Check Prometheus TSDB stats
    local tsdb_response
    tsdb_response=$(curl -s "http://localhost:9090/api/v1/status/tsdb" || echo "")
    
    if [[ -n "$tsdb_response" ]]; then
        local series_count
        series_count=$(echo "$tsdb_response" | jq '.data.seriesCountByMetricName | to_entries | map(.value) | add' 2>/dev/null || echo "0")
        
        if [[ $series_count -gt 0 ]]; then
            log_success "Prometheus TSDB contains $series_count time series"
            
            # Check if series count is reasonable (not too high)
            if [[ $series_count -gt 1000000 ]]; then
                log_warning "High series count ($series_count) - monitor for cardinality issues"
            fi
        else
            log_failure "Prometheus TSDB has no time series data"
        fi
    else
        log_failure "Cannot retrieve Prometheus TSDB status"
    fi
}

# Test network connectivity
test_network_connectivity() {
    log_info "Testing network connectivity between services"
    
    # Test if Prometheus can reach its targets
    cd "$MONITORING_DIR" || return 1
    
    # Test container-to-container connectivity
    if docker-compose exec -T prometheus wget --quiet --tries=1 --spider http://node-exporter:9100/metrics 2>/dev/null; then
        log_success "Prometheus can reach Node Exporter"
    else
        log_failure "Prometheus cannot reach Node Exporter"
    fi
    
    if docker-compose exec -T prometheus wget --quiet --tries=1 --spider http://cadvisor:8080/metrics 2>/dev/null; then
        log_success "Prometheus can reach cAdvisor"
    else
        log_failure "Prometheus cannot reach cAdvisor"
    fi
}

# Test performance
test_performance() {
    log_info "Testing monitoring stack performance"
    
    # Test query response time
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s.%N)
    curl -s "http://localhost:9090/api/v1/query?query=up" >/dev/null
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    
    if (( $(echo "$duration < 1.0" | bc -l 2>/dev/null || echo "0") )); then
        log_success "Prometheus query response time: ${duration}s"
    else
        log_warning "Prometheus query response time is slow: ${duration}s"
    fi
    
    # Check Grafana dashboard load time
    start_time=$(date +%s.%N)
    curl -s -o /dev/null "http://localhost:3000/api/health"
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    
    if (( $(echo "$duration < 2.0" | bc -l 2>/dev/null || echo "0") )); then
        log_success "Grafana health check response time: ${duration}s"
    else
        log_warning "Grafana health check response time is slow: ${duration}s"
    fi
}

# Test security configuration
test_security() {
    log_info "Testing security configuration"
    
    # Check if default passwords are still in use
    if curl -s -u admin:admin123 "http://localhost:3000/api/user" | jq -e '.login == "admin"' >/dev/null 2>&1; then
        log_warning "Grafana is using default credentials - change in production!"
    else
        log_success "Grafana default credentials have been changed"
    fi
    
    # Check for exposed services (basic check)
    local exposed_ports=("3000" "9090" "9093")
    for port in "${exposed_ports[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            log_warning "Port $port is exposed - ensure proper firewall configuration in production"
        fi
    done
}

# Main validation function
main() {
    echo "======================================================"
    echo "APES Monitoring Stack Validation"
    echo "======================================================"
    echo ""
    
    # Check if monitoring stack is running
    if ! docker ps --format "table {{.Names}}" | grep -q "apes-prometheus"; then
        log_failure "Monitoring stack is not running. Start it first with: cd $MONITORING_DIR && ./start-monitoring.sh"
        exit 1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 5
    
    # Run all tests
    test_docker_containers
    test_service_health "Prometheus" "http://localhost:9090/-/healthy"
    test_service_health "Grafana" "http://localhost:3000/api/health"
    test_service_health "Node Exporter" "http://localhost:9100/metrics"
    test_service_health "cAdvisor" "http://localhost:8080/healthz"
    
    test_prometheus_targets
    test_grafana_datasources
    test_alert_rules
    test_metric_ingestion
    test_alertmanager
    test_storage_usage
    test_network_connectivity
    test_performance
    test_security
    
    # Summary
    echo ""
    echo "======================================================"
    echo "Validation Summary"
    echo "======================================================"
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    echo "Total Tests:  $TESTS_TOTAL"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed! Monitoring stack is healthy."
        exit 0
    else
        log_failure "$TESTS_FAILED test(s) failed. Please review the output above."
        exit 1
    fi
}

# Check if required tools are available
check_dependencies() {
    local missing_deps=()
    
    for cmd in curl jq bc docker docker-compose netstat; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_failure "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
}

# Run dependency check and main function
check_dependencies
main "$@"