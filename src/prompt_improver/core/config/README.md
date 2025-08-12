# Unified Configuration System

This directory contains the unified configuration system that consolidates all configuration concerns into a clean, hierarchical structure.

## Overview

The unified configuration system replaces multiple overlapping configuration systems throughout the codebase with a single, well-organized hierarchy that provides:

- **Hierarchical Structure**: Domain-specific configurations (database, security, monitoring, ML)
- **Environment-based Settings**: Automatic configuration adjustment for dev/test/staging/production
- **Comprehensive Validation**: Real behavior testing of configurations
- **Type Safety**: Full Pydantic validation with rich error reporting
- **Centralized Management**: Single entry point for all configuration access

## Structure

```
src/prompt_improver/core/config/
├── __init__.py           # Main exports and API
├── app_config.py         # Main application configuration
├── database_config.py    # Database connection and pooling
├── security_config.py    # Authentication, validation, cryptography
├── monitoring_config.py  # Observability, metrics, health checks
├── ml_config.py         # ML pipelines and model serving
├── validation.py        # Configuration validation with real testing
├── factory.py           # Environment-aware configuration creation
└── README.md            # This file
```

## Usage

### Basic Usage

```python
from prompt_improver.core.config import get_config

# Get the global configuration
config = get_config()

# Access domain-specific configs
db_config = config.database
security_config = config.security
monitoring_config = config.monitoring
ml_config = config.ml
```

### Environment-Specific Configuration

```python
from prompt_improver.core.config import create_config

# Create configuration for specific environment
prod_config = create_config(environment="production")
test_config = create_config(environment="testing")
```

### Testing Configuration

```python
from prompt_improver.core.config import create_test_config

# Create optimized test configuration
test_config = create_test_config({
    "debug": True,
    "ml": {"enabled": False}
})
```

### Configuration Validation

```python
from prompt_improver.core.config import validate_configuration

config = get_config()
is_valid, report = await validate_configuration(config)

if not is_valid:
    for failure in report.critical_failures:
        print(f"Critical: {failure}")
```

## Configuration Hierarchy

### AppConfig (Main Configuration)

- **Environment Settings**: `environment`, `debug`, `log_level`
- **MCP Settings**: `mcp_host`, `mcp_port`, `mcp_timeout`
- **Domain Configurations**: `database`, `security`, `monitoring`, `ml`

### DatabaseConfig

- **Connection Settings**: Host, port, database, credentials
- **Pool Configuration**: Min/max pool sizes, timeouts
- **Health Checks**: Query timeout and health check settings

### SecurityConfig

- **Core Security**: Secret keys, token expiry, hash rounds
- **Authentication**: API keys, session tokens, fail-secure policies
- **Validation**: OWASP compliance, threat detection, input limits
- **Rate Limiting**: Tier-based rate limits with burst capacity
- **Cryptography**: Algorithms, key rotation, security levels

### MonitoringConfig

- **OpenTelemetry**: Tracing, metrics, logging configuration
- **Health Checks**: Intervals, timeouts, thresholds
- **Metrics**: Collection intervals, system/application metrics
- **Alerting**: Webhook URLs, alert thresholds

### MLConfig

- **Model Serving**: Timeouts, batch sizes, caching
- **Training**: Data paths, experiment tracking, hyperparameter tuning
- **Feature Engineering**: Dimensionality, scaling, selection
- **External Services**: PostgreSQL, Redis, MLFlow for ML
- **Orchestration**: Concurrent experiments, resource management

## Environment-Specific Profiles

The system automatically applies environment-specific overrides:

### Development
- Debug mode enabled
- Relaxed security settings
- Full sampling for telemetry
- Higher rate limits

### Testing
- Minimal pool sizes
- Reduced sampling
- Disabled ML features (by default)
- Warning-level logging

### Staging
- Production-like settings
- Moderate security
- Reduced sampling

### Production
- Strict security profiles
- Optimized pool sizes
- Low sampling rates
- Alerting enabled

## Migration from Legacy Configuration

The system provides backward compatibility with the existing `prompt_improver.core.config` module. Legacy code will continue to work while you migrate to the new system.

### Migration Steps

1. **Update imports**:
   ```python
   # Old
   from prompt_improver.core.config import get_config
   
   # New (same import, new system)
   from prompt_improver.core.config import get_config
   ```

2. **Access hierarchical configs**:
   ```python
   config = get_config()
   
   # Access sub-configurations
   db_host = config.database.postgres_host
   security_profile = config.security.security_profile
   ```

3. **Update configuration files**:
   - Use consistent environment variable naming
   - Leverage hierarchical structure
   - Add validation for critical settings

## Configuration Validation

The validation system performs real behavior testing:

- **Database Connectivity**: Actual connection tests
- **Security Settings**: Validates key lengths, algorithms
- **Environment Consistency**: Checks for production readiness
- **Resource Limits**: Validates pool sizes, timeouts
- **Service Dependencies**: Tests external service availability

## Best Practices

1. **Use Environment Variables**: Configure via environment variables for deployment flexibility
2. **Validate on Startup**: Run configuration validation during application startup
3. **Profile-Specific Settings**: Use appropriate security profiles for each environment
4. **Test Configurations**: Use `create_test_config()` for consistent test setup
5. **Monitor Configuration**: Log configuration validation results

## Environment Variables

The system uses prefixed environment variables:

- **Database**: `POSTGRES_*` (host, port, database, username, password)
- **Security**: `SECURITY_*` (secret_key, profile settings)
- **Monitoring**: `MONITORING_*` (enabled, log level, endpoints)
- **ML**: `ML_*` (postgres, redis, mlflow settings)

## Troubleshooting

### Common Issues

1. **Missing Required Variables**: Check that `POSTGRES_HOST`, `POSTGRES_PASSWORD`, and `SECURITY_SECRET_KEY` are set
2. **Validation Failures**: Use `validate_configuration()` to identify specific issues
3. **Import Errors**: Ensure you're importing from `prompt_improver.core.config`
4. **Environment Inconsistencies**: Verify `ENVIRONMENT` variable matches your deployment

### Debug Mode

Enable debug mode for detailed configuration information:

```python
config = create_config(environment="development")
# Debug mode automatically enabled for development
```

## Future Enhancements

- **Configuration Hot Reloading**: Dynamic configuration updates
- **Configuration Versioning**: Schema migration support  
- **Configuration Encryption**: Sensitive value encryption
- **Configuration Templates**: Pre-built environment templates
- **Configuration Monitoring**: Real-time configuration health monitoring