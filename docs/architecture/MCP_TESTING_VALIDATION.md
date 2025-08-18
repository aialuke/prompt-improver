# MCP Server Testing & Validation Guide

## Overview

This document provides comprehensive testing and validation procedures for all MCP server integrations. Tests are designed to validate functionality, security, performance, and real-world usage scenarios.

## Test Environment Setup

### Prerequisites Checklist
```bash
# ✅ Database Prerequisites
□ PostgreSQL server running on localhost:5432
□ apes_production and apes_test databases created
□ mcp_db_agent user created with proper permissions
□ SSL certificates configured (if using SSL)

# ✅ Monitoring Prerequisites  
□ OpenTelemetry collector running on port 4317
□ Metrics endpoint accessible on port 8000
□ Prometheus endpoint accessible on port 8001
□ Jaeger collector running (optional)

# ✅ GitHub Prerequisites
□ GitHub repository access configured
□ GitHub App or Personal Access Token created
□ Webhook endpoint configured (if using webhooks)
□ Repository permissions verified

# ✅ Security Prerequisites
□ All environment variables configured in .env.mcp
□ SSL/TLS certificates installed
□ Authentication credentials validated
□ Network access controls tested
```

### Environment Validation Script
```bash
#!/bin/bash
# test_mcp_environment.sh

echo "🧪 MCP Environment Validation"

# Test PostgreSQL connection
echo "Testing PostgreSQL connection..."
if psql "$DATABASE_URL" -c "SELECT version();" > /dev/null 2>&1; then
    echo "✅ PostgreSQL connection successful"
else
    echo "❌ PostgreSQL connection failed"
    exit 1
fi

# Test monitoring endpoints
echo "Testing monitoring endpoints..."
if curl -s http://localhost:8000/metrics > /dev/null; then
    echo "✅ Metrics endpoint accessible"
else
    echo "❌ Metrics endpoint not accessible"
fi

if curl -s http://localhost:8001/metrics > /dev/null; then
    echo "✅ Prometheus endpoint accessible"
else
    echo "❌ Prometheus endpoint not accessible"
fi

# Test GitHub API access
echo "Testing GitHub API access..."
if curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user > /dev/null; then
    echo "✅ GitHub API access successful"
else
    echo "❌ GitHub API access failed"
fi

echo "🎯 Environment validation complete"
```

## MCP Server Testing Procedures

### 1. PostgreSQL MCP Server Testing

#### Basic Connectivity Test
```bash
# Test MCP server activation
claude mcp add postgresql-database
claude mcp status postgresql-database

# Expected output: Status: Active
```

#### Database Operations Test
```bash
# Test 1: Schema Analysis
echo "Testing schema analysis..."
claude agent database-specialist "Analyze the database schema for the prompt_improvement table"

# Expected: Returns table structure, indexes, constraints

# Test 2: Query Performance Analysis  
echo "Testing query performance..."
claude agent database-specialist "Analyze the performance of SELECT * FROM user_sessions WHERE created_at > NOW() - INTERVAL '1 day'"

# Expected: Returns EXPLAIN plan and optimization recommendations

# Test 3: Connection Pool Monitoring
echo "Testing connection pool monitoring..."
claude agent performance-engineer "Check database connection pool status and performance"

# Expected: Returns pool metrics and optimization suggestions
```

#### Security Validation
```bash
# Test database user permissions
psql "$DATABASE_URL" -c "SELECT current_user, session_user;"
psql "$DATABASE_URL" -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"

# Verify read-only access (should fail)
psql "$DATABASE_URL" -c "CREATE TABLE test_table (id int);" 2>&1 | grep -q "permission denied"
if [ $? -eq 0 ]; then
    echo "✅ Database security: Write operations properly restricted"
else
    echo "❌ Database security: Write operations not restricted"
fi
```

### 2. Observability MCP Server Testing

#### Monitoring Integration Test
```bash
# Test MCP server activation
claude mcp add observability-monitoring
claude mcp status observability-monitoring

# Expected output: Status: Active
```

#### Metrics Collection Test
```bash
# Test 1: Real-time Metrics Access
echo "Testing metrics collection..."
claude agent performance-engineer "Show current application performance metrics including response times and error rates"

# Expected: Returns current metrics from OpenTelemetry

# Test 2: SLO Monitoring
echo "Testing SLO monitoring..."
claude agent performance-engineer "Check if we're meeting our SLO targets: 99.9% availability, P95 <2s latency"

# Expected: Returns SLO compliance status

# Test 3: Alert Management
echo "Testing alert management..."
claude agent performance-engineer "Create an alert for when P95 latency exceeds 2 seconds"

# Expected: Creates alert configuration
```

#### Performance Analysis Test
```bash
# Test distributed tracing
echo "Testing distributed tracing analysis..."
claude agent performance-engineer "Analyze the performance trace for the ML prediction endpoint over the last hour"

# Expected: Returns trace analysis and bottleneck identification

# Test dashboard operations
echo "Testing dashboard operations..."
claude agent performance-engineer "Create a performance dashboard for the ML inference pipeline"

# Expected: Creates dashboard configuration
```

### 3. GitHub MCP Server Testing

#### GitHub Integration Test
```bash
# Test MCP server activation
claude mcp add github-integration
claude mcp status github-integration

# Expected output: Status: Active
```

#### Repository Operations Test
```bash
# Test 1: Issue Management
echo "Testing issue management..."
claude agent infrastructure-specialist "Create an issue for optimizing the CI/CD pipeline performance"

# Expected: Creates GitHub issue with proper labels and description

# Test 2: Pull Request Operations
echo "Testing pull request analysis..."
claude agent infrastructure-specialist "Analyze the performance impact of the latest pull request"

# Expected: Returns PR analysis and performance implications

# Test 3: Workflow Automation
echo "Testing workflow automation..."
claude agent infrastructure-specialist "Optimize the GitHub Actions workflow in .github/workflows/ci.yml"

# Expected: Provides workflow optimization recommendations
```

#### Security Validation
```bash
# Test GitHub token permissions
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/repos/$GITHUB_OWNER/$GITHUB_REPO \
     | jq '.permissions'

# Expected: Shows read/write permissions as configured

# Test webhook security (if configured)
echo '{"test": "payload"}' | \
openssl dgst -sha256 -hmac "$GITHUB_WEBHOOK_SECRET" -binary | \
base64

# Expected: Generates valid webhook signature
```

## Integration Testing Scenarios

### Scenario 1: Database Performance Optimization Workflow

```bash
# Complete workflow test involving multiple agents
echo "🔄 Testing database performance optimization workflow..."

# Step 1: Performance engineer identifies bottleneck
claude agent performance-engineer "Our analytics dashboard is loading slowly. Identify the performance bottleneck."

# Step 2: Delegates to database specialist for optimization
claude agent database-specialist "Optimize the slow queries identified in the user_analytics view"

# Step 3: Infrastructure specialist implements monitoring
claude agent infrastructure-specialist "Set up monitoring for the optimized database queries"

# Step 4: Security architect reviews security implications
claude agent security-architect "Review security implications of the database query optimizations"

# Expected: Coordinated response across all agents with proper delegation
```

### Scenario 2: ML Model Performance Analysis

```bash
echo "🔄 Testing ML model performance analysis workflow..."

# Step 1: Performance engineer detects ML performance issue
claude agent performance-engineer "Analyze the performance of our ML inference pipeline"

# Step 2: ML orchestrator optimizes model
claude agent ml-orchestrator "Optimize the model architecture based on the performance analysis"

# Step 3: Infrastructure specialist updates deployment
claude agent infrastructure-specialist "Update the deployment configuration for the optimized ML model"

# Expected: End-to-end ML performance optimization workflow
```

### Scenario 3: Security Incident Response

```bash
echo "🔄 Testing security incident response workflow..."

# Step 1: Security architect identifies vulnerability
claude agent security-architect "Analyze potential security vulnerabilities in our authentication system"

# Step 2: Infrastructure specialist implements fixes
claude agent infrastructure-specialist "Implement the security fixes recommended by the security architect"

# Step 3: Create GitHub tracking issue
claude agent infrastructure-specialist "Create a GitHub issue to track the security vulnerability remediation"

# Expected: Coordinated security response with tracking
```

## Real Project Scenario Testing

### Test Case 1: Optimize User Analytics Query

```sql
-- Real slow query from the project
SELECT 
    u.id,
    u.username,
    COUNT(s.id) as session_count,
    AVG(s.duration) as avg_duration,
    MAX(s.created_at) as last_session
FROM users u
LEFT JOIN user_sessions s ON u.id = s.user_id
WHERE s.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id, u.username
ORDER BY session_count DESC
LIMIT 100;
```

#### Testing Steps:
```bash
# 1. Performance engineer identifies the issue
claude agent performance-engineer "This analytics query is taking 8 seconds to execute and affecting dashboard performance"

# 2. Database specialist analyzes and optimizes
claude agent database-specialist "Analyze and optimize this specific SQL query: [paste query above]"

# 3. Validate improvements
psql "$DATABASE_URL" -c "EXPLAIN ANALYZE [optimized query]"

# Expected Results:
# - Query execution time reduced by >50%
# - Specific index recommendations
# - Connection pool impact analysis
# - Monitoring setup for ongoing tracking
```

### Test Case 2: ML Model Inference Optimization

```bash
# Real ML performance scenario
echo "🧪 Testing ML model inference optimization..."

# 1. Performance engineer detects slow inference
claude agent performance-engineer "Our ML model inference is taking 3 seconds per prediction, which is above our 1-second target"

# 2. ML orchestrator analyzes and optimizes
claude agent ml-orchestrator "Optimize the model inference pipeline to achieve sub-1-second predictions"

# 3. Monitor improvements
claude agent performance-engineer "Set up monitoring to track ML inference performance improvements"

# Expected Results:
# - Inference time optimization strategies
# - Model architecture recommendations
# - Resource allocation improvements
# - Real-time performance tracking
```

### Test Case 3: GitHub Actions Pipeline Optimization

```bash
# Real CI/CD optimization scenario
echo "🧪 Testing CI/CD pipeline optimization..."

# 1. Infrastructure specialist analyzes current pipeline
claude agent infrastructure-specialist "Our CI/CD pipeline takes 15 minutes to run. Optimize it for faster execution."

# 2. Create optimization PR
claude agent infrastructure-specialist "Create a pull request with optimized GitHub Actions workflow"

# 3. Monitor implementation
claude agent performance-engineer "Track the performance improvement of the optimized CI/CD pipeline"

# Expected Results:
# - Pipeline execution time reduction
# - Specific optimization recommendations (caching, parallelization)
# - GitHub PR with workflow improvements
# - Performance tracking setup
```

## Performance Validation

### Response Time Benchmarks
```bash
# Benchmark MCP server response times
echo "⏱️ Performance Benchmark Testing..."

# PostgreSQL MCP Response Time
time claude agent database-specialist "Get table count for public schema"
# Target: <2 seconds

# Monitoring MCP Response Time  
time claude agent performance-engineer "Get current system metrics"
# Target: <1 second

# GitHub MCP Response Time
time claude agent infrastructure-specialist "List recent issues"
# Target: <3 seconds
```

### Load Testing
```bash
# Test MCP server performance under load
echo "🔄 Load Testing MCP Servers..."

# Concurrent requests test
for i in {1..10}; do
    (claude agent database-specialist "Check database health" &)
done
wait

# Expected: All requests complete successfully within 10 seconds
```

## Error Handling Testing

### Connection Failure Testing
```bash
# Test graceful failure handling
echo "🚨 Testing error handling..."

# Stop PostgreSQL and test MCP behavior
sudo systemctl stop postgresql
claude agent database-specialist "Analyze database schema"
# Expected: Graceful error message, not system crash

# Restart PostgreSQL
sudo systemctl start postgresql

# Test GitHub API rate limiting
for i in {1..100}; do
    claude agent infrastructure-specialist "List repository issues" &
done
# Expected: Graceful handling of rate limits
```

### Security Failure Testing
```bash
# Test security validation
echo "🔒 Testing security failure handling..."

# Test with invalid credentials
export GITHUB_TOKEN="invalid_token"
claude agent infrastructure-specialist "List repository issues"
# Expected: Authentication error, security event logged

# Restore valid token
export GITHUB_TOKEN="$VALID_GITHUB_TOKEN"
```

## Automated Test Suite

### Comprehensive Test Script
```bash
#!/bin/bash
# comprehensive_mcp_test.sh

set -euo pipefail

echo "🧪 Comprehensive MCP Testing Suite"

# Environment validation
./test_mcp_environment.sh

# Individual MCP server tests
echo "Testing PostgreSQL MCP..."
./test_postgresql_mcp.sh

echo "Testing Observability MCP..."
./test_observability_mcp.sh

echo "Testing GitHub MCP..."
./test_github_mcp.sh

# Integration scenarios
echo "Testing integration scenarios..."
./test_integration_scenarios.sh

# Performance validation
echo "Testing performance..."
./test_performance.sh

# Security validation
echo "Testing security..."
./test_security.sh

echo "✅ All MCP tests completed successfully"
```

## Test Results Documentation

### Expected Test Outcomes

#### Functionality Tests
- ✅ All MCP servers activate successfully
- ✅ Database operations return expected results
- ✅ Monitoring data is accessible and accurate
- ✅ GitHub operations complete successfully
- ✅ Agent delegation works as designed

#### Performance Tests
- ✅ Response times meet targets (<2s database, <1s monitoring, <3s GitHub)
- ✅ Concurrent operations handle load appropriately
- ✅ Resource usage remains within acceptable limits

#### Security Tests
- ✅ Authentication mechanisms work correctly
- ✅ Authorization policies are enforced
- ✅ Invalid credentials are rejected gracefully
- ✅ Audit logging captures all security events

#### Integration Tests
- ✅ Multi-agent workflows complete successfully
- ✅ Agent boundaries are respected
- ✅ Delegation patterns work as designed
- ✅ Real project scenarios are handled effectively

## Troubleshooting Guide

### Common Issues and Solutions

#### MCP Server Won't Activate
```bash
# Check MCP server configuration
claude mcp list
claude mcp status [server-name]

# Verify environment variables
env | grep -E "(POSTGRES|GITHUB|MONITORING)"

# Check logs for errors
claude mcp logs [server-name]
```

#### Database Connection Issues
```bash
# Test direct database connection
psql "$DATABASE_URL" -c "SELECT 1;"

# Check SSL configuration
openssl s_client -connect localhost:5432 -servername postgres

# Verify user permissions
psql "$DATABASE_URL" -c "SELECT current_user, session_user;"
```

#### Monitoring Integration Issues
```bash
# Check OpenTelemetry configuration
curl http://localhost:4317/health

# Verify metrics endpoints
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics

# Test OAuth2 token
curl -H "Authorization: Bearer $MONITORING_API_KEY" \
     https://monitoring.local/api/health
```

#### GitHub Integration Issues
```bash
# Test GitHub API access
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/user

# Check repository permissions
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/repos/$GITHUB_OWNER/$GITHUB_REPO

# Verify webhook configuration
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/repos/$GITHUB_OWNER/$GITHUB_REPO/hooks
```

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 3*