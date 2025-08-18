# MCP Server Security Policies & Authentication

## Overview

This document establishes comprehensive security policies and authentication mechanisms for all MCP server integrations in the Claude Code agent system. Security policies are designed by the `security-architect` agent and implemented by the `infrastructure-specialist` agent.

## Security Architecture

### Security Responsibility Matrix

| Component | security-architect Role | infrastructure-specialist Role |
|-----------|-------------------------|--------------------------------|
| **Authentication Design** | OAuth strategies, token policies, access control design | Token storage, authentication infrastructure setup |
| **Authorization Policies** | RBAC design, permission matrices, security boundaries | Permission enforcement, access control implementation |
| **Secret Management** | Credential rotation policies, encryption requirements | Secret storage systems, key management infrastructure |
| **Network Security** | TLS requirements, endpoint security design | SSL/TLS configuration, network access controls |
| **Audit & Monitoring** | Security event requirements, compliance policies | Security logging infrastructure, audit trail systems |

## MCP Server Security Configurations

### 1. PostgreSQL MCP Server Security

#### Authentication Configuration
```json
{
  "postgresql-database": {
    "security": {
      "authentication": {
        "type": "database_credentials",
        "encryption": "TLS_1.3",
        "credential_rotation": "monthly",
        "connection_pooling": {
          "max_connections": 10,
          "connection_timeout": 30,
          "idle_timeout": 300
        }
      },
      "authorization": {
        "rbac_enabled": true,
        "permissions": {
          "read_schema": ["database-specialist", "performance-engineer"],
          "execute_queries": ["database-specialist"],
          "modify_schema": ["database-specialist"],
          "performance_monitoring": ["performance-engineer", "database-specialist"]
        }
      },
      "network_security": {
        "tls_required": true,
        "allowed_hosts": ["localhost", "127.0.0.1"],
        "ssl_mode": "require",
        "ssl_cert": "${POSTGRES_SSL_CERT}",
        "ssl_key": "${POSTGRES_SSL_KEY}"
      }
    }
  }
}
```

#### Secure Environment Variables
```bash
# PostgreSQL Authentication (encrypted storage required)
export POSTGRES_USERNAME="mcp_db_agent"
export POSTGRES_PASSWORD="${POSTGRES_ENCRYPTED_PASSWORD}"
export DATABASE_URL="postgresql://mcp_db_agent:${POSTGRES_ENCRYPTED_PASSWORD}@localhost:5432/apes_production?sslmode=require"

# SSL Configuration
export POSTGRES_SSL_CERT="/path/to/client-cert.pem"
export POSTGRES_SSL_KEY="/path/to/client-key.pem"
export POSTGRES_SSL_ROOT_CERT="/path/to/ca-cert.pem"
```

#### Database User Permissions
```sql
-- Create dedicated MCP user with minimal permissions
CREATE USER mcp_db_agent WITH PASSWORD 'secure_generated_password';

-- Grant schema read access only
GRANT USAGE ON SCHEMA public TO mcp_db_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_db_agent;

-- Grant specific analysis permissions
GRANT EXECUTE ON FUNCTION pg_stat_user_tables TO mcp_db_agent;
GRANT EXECUTE ON FUNCTION pg_stat_activity TO mcp_db_agent;

-- Revoke dangerous permissions
REVOKE CREATE ON SCHEMA public FROM mcp_db_agent;
REVOKE ALL ON DATABASE apes_production FROM mcp_db_agent;

-- Audit logging
ALTER USER mcp_db_agent SET log_statement = 'all';
```

### 2. Observability MCP Server Security

#### Authentication Configuration
```json
{
  "observability-monitoring": {
    "security": {
      "authentication": {
        "type": "oauth2_client_credentials",
        "token_endpoint": "https://auth.monitoring.local/oauth/token",
        "client_id": "${MONITORING_CLIENT_ID}",
        "client_secret": "${MONITORING_CLIENT_SECRET}",
        "scope": "metrics:read traces:read alerts:write"
      },
      "authorization": {
        "rbac_enabled": true,
        "permissions": {
          "read_metrics": ["performance-engineer", "infrastructure-specialist"],
          "write_alerts": ["performance-engineer"],
          "manage_dashboards": ["performance-engineer"],
          "view_traces": ["performance-engineer", "database-specialist", "ml-orchestrator"]
        }
      },
      "network_security": {
        "api_rate_limiting": {
          "requests_per_minute": 1000,
          "burst_limit": 100
        },
        "endpoint_authentication": "bearer_token",
        "allowed_origins": ["https://localhost:8000", "https://monitoring.local"]
      }
    }
  }
}
```

#### Secure Environment Variables
```bash
# OAuth2 Configuration
export MONITORING_CLIENT_ID="mcp_observability_client"
export MONITORING_CLIENT_SECRET="${MONITORING_ENCRYPTED_SECRET}"
export MONITORING_TOKEN_ENDPOINT="https://auth.monitoring.local/oauth/token"

# API Security
export MONITORING_API_KEY="${MONITORING_ENCRYPTED_API_KEY}"
export OTEL_HEADERS="authorization=Bearer ${MONITORING_API_KEY}"

# Rate Limiting
export MONITORING_RATE_LIMIT="1000/minute"
export MONITORING_BURST_LIMIT="100"
```

### 3. GitHub MCP Server Security

#### Authentication Configuration
```json
{
  "github-integration": {
    "security": {
      "authentication": {
        "type": "github_app",
        "app_id": "${GITHUB_APP_ID}",
        "private_key": "${GITHUB_PRIVATE_KEY_PATH}",
        "installation_id": "${GITHUB_INSTALLATION_ID}",
        "token_rotation": "hourly"
      },
      "authorization": {
        "rbac_enabled": true,
        "permissions": {
          "read_repository": ["infrastructure-specialist", "performance-engineer", "security-architect"],
          "manage_issues": ["infrastructure-specialist"],
          "manage_pull_requests": ["infrastructure-specialist"],
          "manage_workflows": ["infrastructure-specialist"],
          "security_advisories": ["security-architect"]
        },
        "repository_access": {
          "type": "selected",
          "repositories": ["prompt-improver"]
        }
      },
      "network_security": {
        "webhook_security": {
          "secret_validation": true,
          "signature_verification": "sha256",
          "ip_whitelist": ["192.30.252.0/22", "185.199.108.0/22"]
        },
        "api_security": {
          "rate_limiting": "5000/hour",
          "secondary_rate_limiting": true
        }
      }
    }
  }
}
```

#### GitHub App Security Configuration
```bash
# GitHub App Authentication (most secure)
export GITHUB_APP_ID="123456"
export GITHUB_PRIVATE_KEY_PATH="/secure/path/to/github-app-private-key.pem"
export GITHUB_INSTALLATION_ID="987654"

# Webhook Security
export GITHUB_WEBHOOK_SECRET="${GITHUB_ENCRYPTED_WEBHOOK_SECRET}"
export GITHUB_WEBHOOK_ENDPOINT="https://secure.local/webhooks/github"

# API Security
export GITHUB_API_VERSION="2022-11-28"
export GITHUB_USER_AGENT="MCP-Agent/1.0"
```

## Comprehensive Security Policies

### 1. Authentication Policies

#### Multi-Factor Authentication
- **GitHub Integration**: GitHub App authentication with private key + installation ID
- **Database Access**: Database credentials + SSL certificate authentication
- **Monitoring Access**: OAuth2 client credentials + API key authentication

#### Token Management
```yaml
Token Rotation Schedule:
  - GitHub App Tokens: Hourly (automatic)
  - Database Passwords: Monthly (manual with notification)
  - OAuth2 Tokens: Daily (automatic)
  - API Keys: Quarterly (manual with audit)

Token Storage Requirements:
  - Encryption: AES-256-GCM at rest
  - Transport: TLS 1.3 minimum
  - Access Logging: All token access logged
  - Audit Trail: 90-day retention minimum
```

#### Credential Encryption
```bash
# Use encrypted storage for all credentials
# Example with SOPS (Secrets Operations)
export POSTGRES_PASSWORD=$(sops -d secrets.yaml | jq -r '.database.password')
export GITHUB_PRIVATE_KEY_PATH=$(sops -d secrets.yaml | jq -r '.github.private_key_path')
export MONITORING_CLIENT_SECRET=$(sops -d secrets.yaml | jq -r '.monitoring.client_secret')
```

### 2. Authorization Policies

#### Role-Based Access Control (RBAC)

```yaml
Agent Permissions Matrix:

database-specialist:
  postgresql-database:
    - read_schema: true
    - execute_queries: true
    - modify_schema: false  # Read-only for safety
    - performance_monitoring: true
  
performance-engineer:
  postgresql-database:
    - read_schema: true
    - execute_queries: false
    - performance_monitoring: true
  observability-monitoring:
    - read_metrics: true
    - write_alerts: true
    - manage_dashboards: true
    - view_traces: true
  github-integration:
    - read_repository: true
    - manage_issues: false
    - view_performance_data: true

security-architect:
  github-integration:
    - security_advisories: true
    - security_policy_management: true
    - audit_access: true
  all_mcp_servers:
    - security_audit: true
    - access_review: true

infrastructure-specialist:
  github-integration:
    - manage_issues: true
    - manage_pull_requests: true
    - manage_workflows: true
  observability-monitoring:
    - read_metrics: true
    - manage_infrastructure: true
```

#### Permission Enforcement
```python
# Example permission enforcement in MCP integration
@require_permission("database-specialist", "execute_queries")
async def execute_database_query(query: str) -> str:
    """Execute database query with security validation."""
    
@require_permission("performance-engineer", "read_metrics") 
async def get_performance_metrics() -> dict:
    """Retrieve performance metrics with authorization."""

@require_permission("security-architect", "security_audit")
async def audit_mcp_access() -> dict:
    """Audit MCP server access patterns."""
```

### 3. Network Security Policies

#### TLS/SSL Requirements
```yaml
TLS Configuration:
  minimum_version: "TLS_1.3"
  cipher_suites:
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
  certificate_validation: strict
  certificate_pinning: enabled
  
Connection Security:
  database:
    ssl_mode: "require"
    ssl_cert_verification: true
  monitoring:
    https_only: true
    certificate_validation: strict
  github:
    tls_verification: true
    webhook_signature_validation: true
```

#### Network Access Controls
```yaml
Firewall Rules:
  postgresql-mcp:
    - source: localhost
    - port: 5432
    - protocol: TCP
    - encryption: required
    
  monitoring-mcp:
    - source: localhost, monitoring.local
    - port: 8000, 8001
    - protocol: HTTPS
    - rate_limiting: 1000/minute
    
  github-mcp:
    - source: api.github.com
    - port: 443
    - protocol: HTTPS
    - webhook_ips: GitHub webhook IP ranges
```

### 4. Audit & Monitoring Policies

#### Security Event Logging
```yaml
Audit Requirements:
  authentication_events:
    - login_attempts: all
    - token_usage: successful and failed
    - permission_checks: failed attempts
    - session_management: creation, expiration
    
  authorization_events:
    - permission_grants: all
    - permission_denials: all
    - role_changes: all
    - access_violations: all
    
  data_access_events:
    - database_queries: all
    - api_calls: high-sensitivity operations
    - file_access: configuration and secrets
    - system_changes: all modifications
```

#### Security Monitoring Integration
```python
# Integration with existing OpenTelemetry monitoring
from prompt_improver.monitoring.opentelemetry.security import SecurityTracer

@SecurityTracer.trace_security_event
async def mcp_authentication(agent_id: str, mcp_server: str) -> bool:
    """Trace MCP authentication events."""
    
@SecurityTracer.audit_data_access
async def mcp_data_access(agent_id: str, operation: str, resource: str) -> None:
    """Audit MCP data access patterns."""
```

## Implementation Guidelines

### 1. Secret Management Setup

#### Using SOPS for Secret Encryption
```bash
# Install SOPS
brew install sops

# Initialize secrets file
sops secrets.yaml

# Example secrets.yaml structure:
database:
  username: mcp_db_agent
  password: secure_generated_password
  ssl_cert_path: /secure/certs/postgres-client.pem

github:
  app_id: 123456
  private_key_path: /secure/keys/github-app-private.pem
  webhook_secret: secure_webhook_secret

monitoring:
  client_id: mcp_observability_client
  client_secret: secure_oauth_secret
  api_key: secure_api_key
```

#### Environment Variable Loading
```bash
# Create secure environment loading script
#!/bin/bash
# load_mcp_secrets.sh

set -euo pipefail

# Load secrets using SOPS
export POSTGRES_USERNAME=$(sops -d secrets.yaml | jq -r '.database.username')
export POSTGRES_PASSWORD=$(sops -d secrets.yaml | jq -r '.database.password')
export GITHUB_APP_ID=$(sops -d secrets.yaml | jq -r '.github.app_id')
export GITHUB_PRIVATE_KEY_PATH=$(sops -d secrets.yaml | jq -r '.github.private_key_path')
export MONITORING_CLIENT_SECRET=$(sops -d secrets.yaml | jq -r '.monitoring.client_secret')

# Validate all secrets are loaded
if [[ -z "$POSTGRES_PASSWORD" || -z "$GITHUB_APP_ID" || -z "$MONITORING_CLIENT_SECRET" ]]; then
    echo "Error: Failed to load required secrets"
    exit 1
fi

echo "✅ MCP secrets loaded successfully"
```

### 2. Security Validation Scripts

#### MCP Security Health Check
```bash
#!/bin/bash
# mcp_security_check.sh

echo "🔒 MCP Security Health Check"

# Check TLS configurations
echo "Checking TLS configurations..."
openssl s_client -connect localhost:5432 -servername postgres </dev/null 2>/dev/null | grep "Verify return code"

# Validate GitHub webhook signatures
echo "Validating GitHub webhook security..."
curl -H "X-Hub-Signature-256: sha256=$(echo -n 'test' | openssl dgst -sha256 -hmac "$GITHUB_WEBHOOK_SECRET" -binary | base64)" \
     -X POST localhost:8000/test/webhook

# Check OAuth2 token validity
echo "Validating monitoring OAuth2 tokens..."
curl -H "Authorization: Bearer $MONITORING_API_KEY" \
     https://monitoring.local/api/v1/health

echo "✅ Security health check completed"
```

### 3. Incident Response Procedures

#### Security Incident Detection
```yaml
Incident Triggers:
  - Failed authentication attempts > 5 in 1 minute
  - Unauthorized data access attempts
  - Credential usage from unexpected locations
  - SSL/TLS certificate validation failures
  - Rate limiting violations
  
Automated Response:
  - Temporary credential suspension
  - Enhanced logging activation
  - Security team notification
  - Incident documentation creation
  
Manual Response Required:
  - Credential rotation
  - Access review and audit
  - Security policy updates
  - Root cause analysis
```

## Compliance & Governance

### Security Review Schedule
- **Weekly**: Automated security scans and vulnerability checks
- **Monthly**: Access review and permission audit
- **Quarterly**: Credential rotation and security policy review
- **Annually**: Comprehensive security architecture review

### Compliance Requirements
- **Data Protection**: Ensure GDPR/CCPA compliance for data access
- **Access Logging**: Maintain comprehensive audit trails
- **Encryption Standards**: AES-256 minimum for data at rest
- **Network Security**: TLS 1.3 minimum for data in transit

### Security Metrics
- **Authentication Success Rate**: >99.9%
- **Authorization Violations**: <0.1%
- **Credential Rotation Compliance**: 100%
- **Security Incident Response Time**: <15 minutes

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 3*