# Environment Configuration Setup Guide

## Overview

This guide explains how to configure the APES Prompt Improver application using environment variables. The application follows 2025 best practices for production-ready configuration management with zero hardcoded values in production deployments.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual configuration:**
   ```bash
   # Edit with your preferred editor
   nano .env
   # or
   vim .env
   ```

3. **Set required environment variables:**
   - Database credentials (PostgreSQL)
   - Redis connection details  
   - Security keys and secrets
   - Service hostnames and ports

## Required Configuration

### Core Database (PostgreSQL) - REQUIRED

The application requires a PostgreSQL database. Configure using either:

**Method 1: Full DATABASE_URL (Recommended)**
```bash
DATABASE_URL=postgresql+asyncpg://username:password@hostname:5432/database_name
```

**Method 2: Individual Components**
```bash
POSTGRES_HOST=your_database_host
POSTGRES_PORT=5432
POSTGRES_DATABASE=prompt_improver
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_secure_password
```

### Core Cache (Redis) - REQUIRED

Redis is required for caching and session management:

**Method 1: Full REDIS_URL (Recommended)**
```bash
REDIS_URL=redis://username:password@hostname:6379/0
```

**Method 2: Individual Components**
```bash
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
REDIS_SSL=false
```

### Security Configuration - REQUIRED

**Critical security settings that MUST be changed in production:**

```bash
SECURITY_SECRET_KEY=your_secure_32_character_or_longer_secret_key_here
SECURITY_ENCRYPTION_KEY=your_encryption_key_for_sensitive_data
```

**Security Requirements:**
- `SECURITY_SECRET_KEY`: Minimum 32 characters, use a cryptographically secure random string
- `SECURITY_ENCRYPTION_KEY`: For encrypting sensitive data at rest
- Never use development defaults in production

## Optional Configuration

### Machine Learning Services

Only required if using ML orchestration features:

```bash
# ML Database (separate from main database)
ML_POSTGRES_HOST=ml_hostname_or_ip
ML_POSTGRES_USERNAME=ml_username  
ML_POSTGRES_PASSWORD=ml_secure_password

# ML Cache (separate Redis instance recommended)
ML_REDIS_HOST=ml_redis_hostname_or_ip
ML_REDIS_PASSWORD=ml_redis_password

# MLFlow Tracking Server
ML_MLFLOW_TRACKING_URI=http://mlflow_host:5000
ML_MLFLOW_HOST=mlflow_hostname_or_ip
```

### External API Integration

Configure external services as needed:

```bash
# OpenAI Integration
EXTERNAL_API_OPENAI_API_KEY=your_openai_api_key

# HuggingFace Integration  
EXTERNAL_API_HUGGINGFACE_API_KEY=your_huggingface_api_key

# Custom APES API
EXTERNAL_API_APES_API_URL=https://api.apes.local
```

### Observability & Monitoring

OpenTelemetry configuration for monitoring:

```bash
# OpenTelemetry Endpoints
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_EXPORTER_OTLP_HTTP_ENDPOINT=http://otel-collector:4318

# Service Information
OTEL_SERVICE_NAME=apes-prompt-improver
OTEL_ENVIRONMENT=production
```

## Deployment Configurations

### Development Environment

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DEVELOPMENT_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

### Production Environment

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DEVELOPMENT_MODE=false

# Production Optimizations
PRODUCTION_OPTIMIZATIONS_ENABLED=true
CONNECTION_POOL_OPTIMIZATION=true
QUERY_CACHE_ENABLED=true
RESPONSE_COMPRESSION_ENABLED=true
```

### Container Deployments

For Docker/Kubernetes deployments:

```bash
# Container Configuration
DEPLOYMENT_MODE=container
REPLICA_COUNT=3
CONTAINER_REGISTRY=your_registry_url
IMAGE_TAG=latest

# Load Balancing
LOAD_BALANCER_ENABLED=true
HEALTH_CHECK_PATH=/health

# Multi-region Setup
PRIMARY_REGION=us-east-1
BACKUP_REGIONS=us-west-2,eu-west-1
```

## Security Best Practices

### Secrets Management

**Development:**
- Use `.env` file (never commit to version control)
- Ensure `.env` is in `.gitignore`

**Production:**
Consider using dedicated secrets management:

```bash
# AWS Secrets Manager
SECRETS_PROVIDER=aws_secrets_manager

# Azure Key Vault
SECRETS_PROVIDER=azure_key_vault

# HashiCorp Vault
SECRETS_PROVIDER=vault

# Kubernetes Secrets
SECRETS_PROVIDER=k8s_secrets
```

### Security Hardening

Enable security features for production:

```bash
SECURITY_HARDENING_ENABLED=true
CORS_ALLOWED_ORIGINS=https://yourdomain.com
CSRF_PROTECTION_ENABLED=true
RATE_LIMITING_ENABLED=true
SECURITY_HEADERS_ENABLED=true
```

## Configuration Validation

The application validates configuration on startup:

1. **Required fields**: Application fails to start if required environment variables are missing
2. **Type validation**: Environment variables are validated for correct types and ranges  
3. **Security validation**: Warns if development secrets are used in production
4. **Connection testing**: Validates database and Redis connections on startup

### Testing Configuration

Test your configuration:

```bash
# Basic validation
python -c "from src.prompt_improver.core.config import get_config; print('✅ Config valid')"

# Full application startup test
python -m src.prompt_improver --validate-config
```

## Troubleshooting

### Common Issues

**1. Missing Required Environment Variables**
```
ValidationError: Field required [type=missing]
```
Solution: Set all required environment variables listed in `.env.example`

**2. Database Connection Failed**
```
asyncpg.exceptions.ConnectionDoesNotExistError
```
Solution: Verify `DATABASE_URL` or PostgreSQL connection parameters

**3. Redis Connection Failed**  
```
redis.exceptions.ConnectionError
```
Solution: Verify `REDIS_URL` or Redis connection parameters

**4. Secret Key Validation Error**
```
ValueError: Development secret key detected in production
```
Solution: Set a production-ready `SECURITY_SECRET_KEY`

### Configuration Debug Mode

Enable configuration debugging:

```bash
CONFIG_DEBUG=true
LOG_LEVEL=DEBUG
```

This will log all configuration values (secrets are masked) during startup.

## Environment Variables Reference

### Core Application
- `ENVIRONMENT`: Application environment (development/staging/production)
- `DEBUG`: Enable debug mode (true/false)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

### Database
- `DATABASE_URL`: Complete PostgreSQL connection URL
- `POSTGRES_HOST`: PostgreSQL hostname
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DATABASE`: Database name
- `POSTGRES_USERNAME`: Database username
- `POSTGRES_PASSWORD`: Database password

### Cache
- `REDIS_URL`: Complete Redis connection URL
- `REDIS_HOST`: Redis hostname
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password
- `REDIS_SSL`: Use SSL connection (default: false)

### Security
- `SECURITY_SECRET_KEY`: Application secret key (min 32 characters)
- `SECURITY_ENCRYPTION_KEY`: Encryption key for sensitive data
- `SECURITY_TOKEN_EXPIRY_SECONDS`: Token expiry time (default: 3600)
- `SECURITY_HASH_ROUNDS`: Password hash rounds (default: 12)
- `SECURITY_MAX_LOGIN_ATTEMPTS`: Maximum login attempts (default: 5)

### MCP Server
- `MCP_HOST`: MCP server bind address (default: 0.0.0.0)
- `MCP_PORT`: MCP server port (default: 8080)
- `MCP_TIMEOUT`: Request timeout in seconds (default: 30)
- `MCP_BATCH_SIZE`: Batch processing size (default: 100)

### Monitoring
- `MONITORING_METRICS_ENABLED`: Enable metrics collection (default: true)
- `MONITORING_HEALTH_CHECK_INTERVAL`: Health check interval in seconds (default: 30.0)
- `MONITORING_LOG_FORMAT`: Log format (json/text, default: json)
- `MONITORING_OPENTELEMETRY_ENDPOINT`: OpenTelemetry collector endpoint

### Feature Flags
- `FEATURE_ML_ORCHESTRATION_ENABLED`: Enable ML orchestration features (default: true)
- `FEATURE_REAL_TIME_ANALYTICS_ENABLED`: Enable real-time analytics (default: true)
- `FEATURE_ADVANCED_CACHING_ENABLED`: Enable advanced caching (default: true)
- `FEATURE_DISTRIBUTED_TRACING_ENABLED`: Enable distributed tracing (default: true)

## Migration from Hardcoded Values

If upgrading from a version with hardcoded values:

1. **Backup your current configuration**
2. **Copy `.env.example` to `.env`**
3. **Set required environment variables**
4. **Test configuration loading**
5. **Deploy with environment-based configuration**

The application will no longer start with hardcoded development values in production environments.