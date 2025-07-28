# Configuration Hot-Reload System Guide

## Overview

The Configuration Hot-Reload System is a comprehensive configuration management solution designed for high-availability, zero-downtime operations following Site Reliability Engineering (SRE) best practices. It provides atomic configuration updates, multi-source support, and comprehensive monitoring capabilities.

## Features

### Core Capabilities
- **Hot-reload**: Configuration changes without service restart
- **Atomic Updates**: All-or-nothing configuration changes
- **Thread-Safe**: Concurrent access support with async/await patterns
- **Multi-Source**: Files, environment variables, and remote sources
- **Rollback**: Automatic and manual rollback capabilities
- **Performance**: <100ms reload times guaranteed
- **Zero-Downtime**: Continuous service operation during updates

### Monitoring & Observability
- Real-time metrics collection
- Configuration change audit trails
- Performance monitoring and alerting
- Health checks for configuration sources
- Structured logging integration

### Security Features
- Hardcoded credential detection
- Configuration validation
- Encrypted configuration sections
- Access control and audit logging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ConfigManager                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   ConfigStore   │  │  WatchManager   │  │ MetricsCollector │
│  │  (Thread-Safe)  │  │  (File Monitor) │  │ (Performance) │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ FileConfigSource │  │ EnvConfigSource │  │RemoteConfigSource│
│  │   (YAML/JSON)   │  │  (Environment)  │  │   (HTTP/API) │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
import asyncio
from src.prompt_improver.core.config_manager import (
    ConfigManager, 
    FileConfigSource, 
    initialize_config_manager
)

async def main():
    # Initialize with file configuration
    source = FileConfigSource("config/app.yaml")
    manager = await initialize_config_manager([source])
    
    # Get configuration values
    db_host = await get_config("database.host", "localhost")
    debug_mode = await get_config("app.debug", False)
    
    print(f"Database host: {db_host}")
    print(f"Debug mode: {debug_mode}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Source Configuration

```python
from src.prompt_improver.core.config_manager import (
    FileConfigSource,
    EnvironmentConfigSource,
    RemoteConfigSource
)

async def setup_production_config():
    sources = [
        # Remote configuration (highest priority)
        RemoteConfigSource(
            url="https://config.company.com/app/config.json",
            headers={"Authorization": "Bearer token"},
            priority=20
        ),
        
        # Local overrides (medium priority)
        FileConfigSource("config/local.yaml", priority=10),
        
        # Environment variables (lowest priority)
        EnvironmentConfigSource(prefix="APP_", priority=5)
    ]
    
    manager = await initialize_config_manager(sources)
    return manager
```

## Configuration Sources

### File Sources

Supports YAML and JSON formats with environment variable substitution:

```yaml
# config/database.yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  password: ${DB_PASSWORD}  # Required from environment

connection_pool:
  min_size: 4
  max_size: 20
  timeout: ${CONNECTION_TIMEOUT:30}
```

```python
# Usage
source = FileConfigSource("config/database.yaml", priority=10)
```

### Environment Variables

Automatically converts environment variables to configuration:

```bash
# Environment variables
export APP_DATABASE_HOST=production-db
export APP_DATABASE_PORT=5432
export APP_DEBUG_MODE=false
export APP_FEATURE_FLAGS='{"new_ui": true, "beta_features": false}'
```

```python
# Usage
source = EnvironmentConfigSource(prefix="APP_", priority=15)
```

### Remote Sources

Fetch configuration from HTTP endpoints with circuit breaker protection:

```python
source = RemoteConfigSource(
    url="https://config-service.company.com/api/v1/config",
    headers={
        "Authorization": "Bearer your-token",
        "Accept": "application/json"
    },
    timeout=10,
    priority=20
)
```

## Advanced Features

### Configuration Rollback

```python
# Get version history
versions = manager.get_version_history()
print(f"Available versions: {len(versions)}")

# Rollback to previous version
result = await manager.rollback(steps=1)
if result.status == ReloadStatus.ROLLED_BACK:
    print("Rollback successful")
```

### Change Subscriptions

```python
def config_change_handler(changes):
    for change in changes:
        print(f"Config changed: {change.path} = {change.new_value}")

# Subscribe to changes
subscription = manager.subscribe(config_change_handler)

# Later: unsubscribe
manager.unsubscribe(subscription)
```

### Performance Monitoring

```python
metrics = manager.get_metrics()
print(f"Total reloads: {metrics.total_reloads}")
print(f"Success rate: {metrics.successful_reloads / metrics.total_reloads * 100:.1f}%")
print(f"Average reload time: {metrics.average_reload_time_ms:.1f}ms")
```

## Integration Examples

### Database Configuration Integration

```python
from src.prompt_improver.database.config import DatabaseConfig

class HotReloadDatabaseConfig(DatabaseConfig):
    @classmethod
    async def from_config_manager(cls, manager: ConfigManager):
        db_config = await manager.get_section("database")
        
        return cls(
            postgres_host=db_config.get("host", "localhost"),
            postgres_port=db_config.get("port", 5432),
            postgres_database=db_config.get("database", "apes_production"),
            # ... other fields
        )
    
    async def reload_from_manager(self, manager: ConfigManager):
        new_config = await self.from_config_manager(manager)
        # Update current instance attributes
        for field, value in new_config.__dict__.items():
            setattr(self, field, value)
```

### Feature Flags Replacement

```python
# Replace existing feature flags system
async def is_feature_enabled(feature_name: str, user_id: str = None) -> bool:
    feature_config = await get_config(f"features.{feature_name}")
    if not feature_config or not feature_config.get("enabled", False):
        return False
    
    rollout_percentage = feature_config.get("rollout_percentage", 100)
    if rollout_percentage >= 100:
        return True
    
    # User-based rollout logic
    if user_id and rollout_percentage > 0:
        user_hash = hash(f"{feature_name}:{user_id}") % 100
        return user_hash < rollout_percentage
    
    return False
```

## Production Deployment

### High Availability Setup

```python
async def setup_ha_config():
    sources = [
        # Primary remote source
        RemoteConfigSource(
            url="https://config-primary.company.com/config",
            priority=30,
            name="primary"
        ),
        
        # Backup remote source
        RemoteConfigSource(
            url="https://config-backup.company.com/config",
            priority=25,
            name="backup"
        ),
        
        # Local cache (always available)
        FileConfigSource("/var/cache/app/config.yaml", priority=20),
        
        # Environment fallback
        EnvironmentConfigSource(prefix="APP_", priority=10)
    ]
    
    manager = ConfigManager(watch_files=True)
    for source in sources:
        await manager.add_source(source)
    
    await manager.start()
    return manager
```

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Configuration directory
RUN mkdir -p /etc/app/config
VOLUME ["/etc/app/config"]

# Environment for config manager
ENV CONFIG_DIR=/etc/app/config
ENV APP_CONFIG_WATCH=true
ENV APP_CONFIG_SOURCES="file,env,remote"

COPY . /app
WORKDIR /app

CMD ["python", "-m", "src.prompt_improver.main"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - ./config:/etc/app/config:ro
    environment:
      - CONFIG_DIR=/etc/app/config
      - APP_CONFIG_WATCH=true
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### Kubernetes Integration

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database.yaml: |
    database:
      host: postgres-service
      port: 5432
      pool:
        min_size: 4
        max_size: 20
  
  features.yaml: |
    features:
      advanced_ml:
        enabled: true
        rollout_percentage: 50

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prompt-improver
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: prompt-improver:latest
        env:
        - name: CONFIG_DIR
          value: /etc/config
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: app-config
```

## Monitoring & Alerting

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Configuration metrics
config_reloads_total = Counter('config_reloads_total', 'Total configuration reloads', ['status'])
config_reload_duration = Histogram('config_reload_duration_seconds', 'Configuration reload duration')
config_sources_healthy = Gauge('config_sources_healthy', 'Number of healthy configuration sources')

def setup_metrics_collection(manager: ConfigManager):
    def metrics_handler(changes):
        config_reloads_total.labels(status='success').inc()
        
        # Update metrics from manager
        metrics = manager.get_metrics()
        config_sources_healthy.set(metrics.config_sources_count)
    
    manager.subscribe(metrics_handler)
```

### Health Checks

```python
async def config_health_check() -> Dict[str, Any]:
    """Health check endpoint for configuration system."""
    if not get_config_manager():
        return {"status": "unhealthy", "reason": "Config manager not initialized"}
    
    manager = get_config_manager()
    metrics = manager.get_metrics()
    
    # Check recent reload performance
    recent_reload_ok = (
        metrics.last_reload_time and 
        (datetime.utcnow() - metrics.last_reload_time).total_seconds() < 300 and
        metrics.average_reload_time_ms < 100
    )
    
    # Check source health
    sources = manager.get_sources()
    healthy_sources = 0
    
    for source in sources:
        try:
            await source.is_modified()
            healthy_sources += 1
        except Exception:
            pass
    
    is_healthy = (
        recent_reload_ok and 
        healthy_sources > 0 and 
        metrics.failed_reloads < metrics.successful_reloads
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "metrics": {
            "total_reloads": metrics.total_reloads,
            "success_rate": metrics.successful_reloads / max(metrics.total_reloads, 1),
            "avg_reload_time_ms": metrics.average_reload_time_ms,
            "healthy_sources": f"{healthy_sources}/{len(sources)}"
        },
        "last_reload": metrics.last_reload_time.isoformat() if metrics.last_reload_time else None
    }
```

## Security Considerations

### Configuration Validation

```python
# Security validation is built-in
source = FileConfigSource("config/app.yaml")
config_data = await source.load_config()

# Validation includes:
# - Hardcoded password detection
# - Secret scanning
# - Schema validation
# - Cross-field validation

is_valid, errors = await source.validate_config(config_data)
if not is_valid:
    logger.error(f"Security validation failed: {errors}")
```

### Secrets Management

```python
# Use environment variables for secrets
database_config = {
    "host": "localhost",
    "port": 5432,
    "username": "app_user",
    "password": "${DB_PASSWORD}"  # From environment only
}

# Or integrate with secret management systems
async def get_secret_from_vault(secret_path: str) -> str:
    # Integration with HashiCorp Vault, AWS Secrets Manager, etc.
    pass
```

## Troubleshooting

### Common Issues

1. **Slow Reload Times**
   ```python
   # Check metrics
   metrics = manager.get_metrics()
   if metrics.average_reload_time_ms > 100:
       print("Performance issue detected")
       
   # Enable debug logging
   logging.getLogger('src.prompt_improver.core.config_manager').setLevel(logging.DEBUG)
   ```

2. **Configuration Not Updating**
   ```python
   # Check file watching
   sources = manager.get_sources()
   for source in sources:
       if isinstance(source, FileConfigSource):
           is_modified = await source.is_modified()
           print(f"Source {source.name} modified: {is_modified}")
   ```

3. **Remote Source Failures**
   ```python
   # Check circuit breaker status
   for source in manager.get_sources():
       if isinstance(source, RemoteConfigSource):
           is_open = source._is_circuit_open()
           print(f"Circuit breaker for {source.name}: {'OPEN' if is_open else 'CLOSED'}")
   ```

### Debug Mode

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create manager with debug options
manager = ConfigManager(watch_files=True, max_config_versions=50)

# Monitor all changes
def debug_handler(changes):
    for change in changes:
        print(f"DEBUG: {change.change_type.value} {change.path}: {change.old_value} -> {change.new_value}")

manager.subscribe(debug_handler)
```

## Best Practices

### Development
- Use local file sources with environment variable overrides
- Enable file watching for hot-reload during development
- Validate configurations early in the development cycle

### Staging
- Mirror production configuration sources
- Test rollback procedures
- Validate performance requirements

### Production
- Use multiple configuration sources for redundancy
- Implement comprehensive monitoring and alerting
- Regular backup of configuration states
- Automated security scanning of configurations

### Performance Optimization
- Keep configuration files reasonably sized (<1MB)
- Use efficient serialization formats (JSON over YAML for large configs)
- Implement configuration caching where appropriate
- Monitor and optimize reload times

## API Reference

### ConfigManager

```python
class ConfigManager:
    async def add_source(self, source: ConfigSource) -> None
    async def start(self) -> None
    async def stop(self) -> None
    async def get_config(self, path: str = None, default: Any = None) -> Any
    async def get_section(self, section: str) -> Dict[str, Any]
    async def reload(self, source_name: str = None) -> ReloadResult
    async def rollback(self, steps: int = 1) -> ReloadResult
    def subscribe(self, callback: Callable) -> weakref.ReferenceType
    def unsubscribe(self, subscription: weakref.ReferenceType) -> None
    def get_metrics(self) -> ConfigMetrics
    def get_sources(self) -> List[ConfigSource]
    def get_version_history(self) -> List[ConfigVersion]
```

### ConfigSource

```python
class ConfigSource(ABC):
    @abstractmethod
    async def load_config(self) -> Dict[str, Any]
    @abstractmethod
    async def is_modified(self) -> bool
    async def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]
```

### Convenience Functions

```python
async def initialize_config_manager(sources: List[ConfigSource], watch_files: bool = True) -> ConfigManager
def get_config_manager() -> Optional[ConfigManager]
async def get_config(path: str = None, default: Any = None) -> Any
async def get_config_section(section: str) -> Dict[str, Any]
```

## Support

For questions, issues, or contributions:

1. Check existing documentation and examples
2. Review test files for usage patterns
3. Enable debug logging for troubleshooting
4. Consult monitoring metrics and health checks

The Configuration Hot-Reload System is designed to provide reliable, high-performance configuration management for production systems. Following these guidelines will ensure optimal operation and maintainability.