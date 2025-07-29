# Health Monitoring Migration Guide

## Overview
This guide provides step-by-step instructions for migrating from the fragmented health monitoring systems to the unified `UnifiedHealthSystem` with plugin-based architecture.

## Migration Timeline
- **Phase 1**: Setup unified health system and core plugins (Weeks 1-2)
- **Phase 2**: Migrate existing health checks to plugins (Weeks 3-4)
- **Phase 3**: Update monitoring integrations and remove legacy code (Weeks 5-6)

## Before You Begin

### Prerequisites
- [ ] Python 3.8+ with async/await support
- [ ] Prometheus client library (optional but recommended)
- [ ] Access to monitoring/alerting systems
- [ ] Understanding of existing health check patterns

### Backup Recommendations
```bash
# Backup current health monitoring configuration  
cp config/monitoring_config.yaml config/monitoring_config.backup.yaml

# Tag current state
git tag pre-health-migration-$(date +%Y%m%d)
git push origin pre-health-migration-$(date +%Y%m%d)
```

## Understanding the New Architecture

### Plugin-Based Design
The unified health system uses a plugin architecture where each plugin registers multiple health checkers:

```python
# New plugin structure
class DatabaseHealthPlugin(HealthPlugin):
    def register_checkers(self, system: HealthMonitorProtocol) -> None:
        system.register_checker("db_connection", self._check_connection, critical=True)
        system.register_checker("db_pool_status", self._check_pool_health)
        system.register_checker("db_query_performance", self._check_query_perf)
```

### Core Components
- **UnifiedHealthSystem**: Central health monitoring coordinator
- **HealthPlugin**: Base class for creating health monitoring plugins
- **HealthCheckResult**: Standardized health check response format
- **HealthMonitorProtocol**: Interface for health monitoring operations

## Migration by Component

### 1. Database Health Monitoring Migration

#### Before (Multiple Legacy Monitors)
```python
# database_health_monitor.py
from prompt_improver.database.health.database_health_monitor import DatabaseHealthMonitor

db_monitor = DatabaseHealthMonitor(connection_pool)
db_health = await db_monitor.check_connection_health()

# connection_pool_monitor.py  
from prompt_improver.database.health.connection_pool_monitor import ConnectionPoolMonitor

pool_monitor = ConnectionPoolMonitor(pool)
pool_status = await pool_monitor.get_pool_status()

# query_performance_analyzer.py
from prompt_improver.database.health.query_performance_analyzer import QueryPerformanceAnalyzer

query_analyzer = QueryPerformanceAnalyzer(connection)
query_metrics = await query_analyzer.analyze_performance()
```

#### After (Unified Plugin)
```python
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem
from prompt_improver.performance.monitoring.health.plugin_adapters import DatabaseHealthPlugin

# Setup unified system
health_system = UnifiedHealthSystem(config)

# Register database plugin (consolidates all database monitoring)
db_plugin = DatabaseHealthPlugin(connection_manager)
health_system.register_plugin(db_plugin)

# Get comprehensive database health
health_results = await health_system.check_health(component_name="database")

# Results include all previous monitors:
# - db_connection: Connection health
# - db_pool_status: Pool status  
# - db_query_performance: Query performance
# - db_storage_health: Storage health
# - db_index_health: Index health
```

#### Migration Steps
1. **Replace individual imports**:
   ```python
   # Remove these legacy imports
   from prompt_improver.database.health.database_health_monitor import DatabaseHealthMonitor
   from prompt_improver.database.health.connection_pool_monitor import ConnectionPoolMonitor
   from prompt_improver.database.health.query_performance_analyzer import QueryPerformanceAnalyzer
   from prompt_improver.database.health.table_bloat_detector import TableBloatDetector
   from prompt_improver.database.health.index_health_assessor import IndexHealthAssessor
   
   # Add unified imports
   from prompt_improver.performance.monitoring.health import UnifiedHealthSystem
   from prompt_improver.performance.monitoring.health.plugin_adapters import DatabaseHealthPlugin
   ```

2. **Update initialization code**:
   ```python
   # Old: Multiple monitor initialization
   db_monitor = DatabaseHealthMonitor(connection_pool)
   pool_monitor = ConnectionPoolMonitor(pool)
   query_analyzer = QueryPerformanceAnalyzer(connection)
   
   # New: Single plugin registration
   health_system = UnifiedHealthSystem(config)
   db_plugin = DatabaseHealthPlugin(connection_manager)
   health_system.register_plugin(db_plugin)
   ```

3. **Update health check calls**:
   ```python
   # Old: Multiple separate calls
   db_health = await db_monitor.check_connection_health()
   pool_status = await pool_monitor.get_pool_status()  
   query_metrics = await query_analyzer.analyze_performance()
   
   # New: Single unified call
   health_results = await health_system.check_health()
   
   # Access specific results
   db_connection = health_results.get("db_connection")
   pool_status = health_results.get("db_pool_status")
   query_performance = health_results.get("db_query_performance")
   ```

### 2. ML Health Monitoring Migration

#### Before (Multiple ML Monitors)
```python
# ml_health_monitor.py
from prompt_improver.ml.health.ml_health_monitor import MLHealthMonitor

ml_monitor = MLHealthMonitor(model_registry)
model_health = await ml_monitor.check_model_availability()

# model_performance_tracker.py
from prompt_improver.ml.health.model_performance_tracker import ModelPerformanceTracker

perf_tracker = ModelPerformanceTracker()
performance_metrics = await perf_tracker.get_performance_metrics()

# drift_detector.py
from prompt_improver.ml.health.drift_detector import DriftDetector

drift_detector = DriftDetector(model_config)
drift_status = await drift_detector.detect_drift()

# resource_monitor.py  
from prompt_improver.ml.health.resource_monitor import ResourceMonitor

resource_monitor = ResourceMonitor()
resource_usage = await resource_monitor.get_resource_usage()
```

#### After (Unified Plugin)
```python
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem
from prompt_improver.performance.monitoring.health.plugin_adapters import MLHealthPlugin

# Setup unified system with ML plugin
health_system = UnifiedHealthSystem(config)
ml_plugin = MLHealthPlugin(model_registry, ml_config)
health_system.register_plugin(ml_plugin)

# Get comprehensive ML health
ml_health = await health_system.check_health(component_name="ml")

# Results include all ML monitoring:
# - ml_models: Model availability and health
# - ml_performance: Performance metrics
# - ml_drift: Data/model drift detection  
# - ml_resources: GPU/CPU resource usage
```

#### Migration Steps
1. **Replace ML health imports**:
   ```python
   # Remove legacy ML health imports
   from prompt_improver.ml.health.ml_health_monitor import MLHealthMonitor
   from prompt_improver.ml.health.model_performance_tracker import ModelPerformanceTracker
   from prompt_improver.ml.health.drift_detector import DriftDetector
   from prompt_improver.ml.health.resource_monitor import ResourceMonitor
   
   # Add unified ML plugin
   from prompt_improver.performance.monitoring.health.plugin_adapters import MLHealthPlugin
   ```

2. **Consolidate ML health configuration**:
   ```yaml
   # Old: Separate ML health configs
   ml_health:
     model_check_interval: 60
     performance_check_interval: 300
     drift_check_interval: 3600
     resource_check_interval: 30
   
   # New: Unified ML plugin config
   health:
     plugins:
       ml:
         model_check_interval: 60
         performance_check_interval: 300
         drift_check_interval: 3600
         resource_check_interval: 30
         critical_checks: ["ml_models"]  # Which checks are critical
   ```

### 3. Performance Health Monitoring Migration

#### Before (Separate Performance Monitors)
```python
# performance_monitor.py
from prompt_improver.performance.monitoring.performance_monitor import PerformanceMonitor

perf_monitor = PerformanceMonitor()
perf_metrics = await perf_monitor.get_performance_metrics()

# sla_monitor.py
from prompt_improver.performance.sla_monitor import SLAMonitor

sla_monitor = SLAMonitor(sla_config)
sla_status = await sla_monitor.check_sla_compliance()

# background_manager.py (health aspects)
from prompt_improver.performance.monitoring.health.background_manager import BackgroundManager

bg_manager = BackgroundManager()
task_health = await bg_manager.check_background_tasks()
```

#### After (Unified Plugin)
```python
from prompt_improver.performance.monitoring.health.plugin_adapters import PerformanceHealthPlugin

# Register performance plugin
performance_plugin = PerformanceHealthPlugin(performance_config)
health_system.register_plugin(performance_plugin)

# Get comprehensive performance health
perf_health = await health_system.check_health(component_name="performance")

# Results include:
# - performance_metrics: System performance metrics
# - sla_compliance: SLA compliance status
# - background_tasks: Background task health
# - resource_utilization: System resource usage
```

### 4. External Service Health Migration

#### Before (Individual Service Monitors)
```python
# external_api_health.py
from prompt_improver.monitoring.external_api_health import ExternalAPIHealth

api_health = ExternalAPIHealth(api_configs)
api_status = await api_health.check_all_apis()

# redis_health.py
from prompt_improver.cache.redis_health import RedisHealth

redis_health = RedisHealth(redis_config)
redis_status = await redis_health.check_redis_health()
```

#### After (Unified Plugin)
```python
from prompt_improver.performance.monitoring.health.plugin_adapters import ExternalServicePlugin

# Register external service plugin
external_plugin = ExternalServicePlugin(service_configs)
health_system.register_plugin(external_plugin)

# Get external service health
external_health = await health_system.check_health(component_name="external")

# Results include:
# - redis_cache: Redis health and connectivity
# - external_apis: Third-party API health
# - message_queues: Message queue health (if applicable)
```

## Configuration Migration

### Legacy Configuration Format
```yaml
# Separate health configurations
database_health:
  connection_check_interval: 30
  pool_check_interval: 60
  query_performance_threshold: 1000

ml_health:
  model_check_interval: 60  
  drift_check_interval: 3600
  resource_check_interval: 30

performance_health:
  sla_check_interval: 120
  background_task_check_interval: 60
  
external_health:
  api_check_interval: 180
  redis_check_interval: 45
```

### Unified Configuration Format
```yaml
# Single unified health configuration
health:
  # Global settings
  critical_check_interval: 30      # How often to run critical checks
  standard_check_interval: 300     # How often to run standard checks  
  extended_check_interval: 1800    # How often to run extended checks
  
  # Metrics and alerting
  enable_metrics: true
  enable_prometheus: true
  alert_on_critical: true
  
  # Plugin configurations
  plugins:
    database:
      connection_check_critical: true
      pool_check_critical: false
      query_performance_threshold: 1000
      storage_check_interval: 1800  # Extended check
      
    ml:
      model_check_critical: true
      performance_check_critical: false
      drift_check_interval: 3600     # Extended check
      resource_threshold_cpu: 80
      resource_threshold_memory: 85
      
    performance:
      sla_check_critical: false
      background_task_check_critical: true
      response_time_threshold: 500
      
    external:
      redis_check_critical: true
      api_check_critical: false
      timeout_threshold: 10.0
```

### Configuration Migration Script
```python
# health_config_migration.py
import yaml
from pathlib import Path
from typing import Dict, Any

def migrate_health_config(old_config_path: Path, new_config_path: Path):
    """Migrate legacy health configurations to unified format"""
    
    with open(old_config_path) as f:
        old_config = yaml.safe_load(f)
    
    # Extract legacy settings
    db_health = old_config.get('database_health', {})
    ml_health = old_config.get('ml_health', {})
    perf_health = old_config.get('performance_health', {})
    ext_health = old_config.get('external_health', {})
    
    # Build unified configuration
    new_config = {
        'health': {
            # Global settings
            'critical_check_interval': 30,
            'standard_check_interval': 300,
            'extended_check_interval': 1800,
            
            # Enable features
            'enable_metrics': True,
            'enable_prometheus': True,
            'alert_on_critical': True,
            
            # Plugin configurations
            'plugins': {
                'database': {
                    'connection_check_critical': True,
                    'pool_check_critical': False,
                    'query_performance_threshold': db_health.get(
                        'query_performance_threshold', 1000
                    ),
                    'storage_check_interval': 1800,
                },
                'ml': {
                    'model_check_critical': True,
                    'performance_check_critical': False,
                    'drift_check_interval': ml_health.get('drift_check_interval', 3600),
                    'resource_threshold_cpu': 80,
                    'resource_threshold_memory': 85,
                },
                'performance': {
                    'sla_check_critical': False,
                    'background_task_check_critical': True,
                    'response_time_threshold': 500,
                },
                'external': {
                    'redis_check_critical': True,
                    'api_check_critical': False,
                    'timeout_threshold': 10.0,
                }
            }
        }
    }
    
    # Write unified configuration
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    print(f"Health configuration migrated from {old_config_path} to {new_config_path}")

if __name__ == "__main__":
    migrate_health_config(
        Path("config/monitoring_config.yaml"),
        Path("config/unified_health_config.yaml")
    )
```

## Creating Custom Health Plugins

### Plugin Template
```python
from typing import Dict, Any, Optional
from prompt_improver.performance.monitoring.health.base import HealthPlugin
from prompt_improver.core.protocols.health_protocol import (
    HealthMonitorProtocol, 
    HealthCheckResult, 
    HealthStatus
)

class CustomHealthPlugin(HealthPlugin):
    """Template for creating custom health monitoring plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize any resources needed for health checks
    
    def register_checkers(self, system: HealthMonitorProtocol) -> None:
        """Register health checkers with the system"""
        system.register_checker(
            name="custom_service",
            checker=self._check_custom_service,
            timeout=30.0,
            critical=True  # Mark as critical if service is essential
        )
        
        system.register_checker(
            name="custom_metrics",
            checker=self._check_custom_metrics,
            timeout=10.0,
            critical=False
        )
    
    async def _check_custom_service(self) -> HealthCheckResult:
        """Check health of custom service"""
        try:
            # Implement your health check logic
            is_healthy = await self._perform_service_check()
            
            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Custom service is operational",
                    details={"last_check": "success"}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,  
                    message="Custom service is not responding",
                    details={"error": "Service unavailable"}
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    async def _check_custom_metrics(self) -> HealthCheckResult:
        """Check custom metrics thresholds"""
        try:
            metrics = await self._get_service_metrics()
            
            # Example threshold checking
            if metrics.get('response_time', 0) > 1000:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="High response time detected",
                    details=metrics
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="All metrics within normal ranges",
                details=metrics
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Metrics check failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    async def _perform_service_check(self) -> bool:
        """Implement actual service health check"""
        # Your service-specific health check logic
        return True
        
    async def _get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics for threshold checking"""
        # Your metrics collection logic
        return {"response_time": 150, "error_rate": 0.001}
```

### Registering Custom Plugins
```python
# Register custom plugin with the health system
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem

health_system = UnifiedHealthSystem(config)

# Register built-in plugins
health_system.register_plugin(DatabaseHealthPlugin(connection_manager))
health_system.register_plugin(MLHealthPlugin(model_registry))

# Register custom plugin
custom_plugin = CustomHealthPlugin(custom_config)
health_system.register_plugin(custom_plugin)

# All plugins are now available
health_results = await health_system.check_health()
```

## API Integration Migration

### Legacy Health Endpoints
```python
# Before: Multiple health endpoints
from fastapi import FastAPI

app = FastAPI()

@app.get("/health/database")
async def database_health():
    db_monitor = DatabaseHealthMonitor(connection_pool)
    return await db_monitor.check_connection_health()

@app.get("/health/ml")
async def ml_health():
    ml_monitor = MLHealthMonitor(model_registry)
    return await ml_monitor.check_model_availability()

@app.get("/health/performance")
async def performance_health():
    perf_monitor = PerformanceMonitor()
    return await perf_monitor.get_performance_metrics()
```

### Unified Health Endpoints
```python
# After: Single unified health API
from fastapi import FastAPI
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem

app = FastAPI()
health_system = UnifiedHealthSystem(config)

# Setup all plugins
health_system.register_plugin(DatabaseHealthPlugin(connection_manager))
health_system.register_plugin(MLHealthPlugin(model_registry))
health_system.register_plugin(PerformanceHealthPlugin(perf_config))

@app.get("/health")
async def get_overall_health():
    """Get overall system health status"""
    overall_health = await health_system.get_overall_health()
    return {
        "status": overall_health.status.value,
        "message": overall_health.message,
        "timestamp": overall_health.timestamp.isoformat()
    }

@app.get("/health/detailed")  
async def get_detailed_health():
    """Get detailed health information for all components"""
    detailed_health = await health_system.check_health(include_details=True)
    return {
        component: {
            "status": result.status.value,
            "message": result.message,
            "duration_ms": result.duration_ms,
            "details": result.details
        }
        for component, result in detailed_health.items()
    }

@app.get("/health/{component}")
async def get_component_health(component: str):
    """Get health status for specific component"""
    component_health = await health_system.check_health(
        component_name=component,
        include_details=True
    )
    
    if component not in component_health:
        return {"error": f"Component '{component}' not found"}
    
    result = component_health[component]
    return {
        "status": result.status.value,
        "message": result.message,
        "duration_ms": result.duration_ms,
        "details": result.details
    }
```

## Testing Migration

### Unit Test Migration
```python
# Before: Testing individual monitors
import pytest
from prompt_improver.database.health.database_health_monitor import DatabaseHealthMonitor

@pytest.fixture
def db_monitor():
    return DatabaseHealthMonitor(mock_connection_pool)

@pytest.mark.asyncio
async def test_database_health_check(db_monitor):
    result = await db_monitor.check_connection_health()
    assert result.status == "healthy"

# After: Testing unified system
import pytest
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem
from prompt_improver.performance.monitoring.health.plugin_adapters import DatabaseHealthPlugin

@pytest.fixture
def health_system():
    config = {"critical_check_interval": 30}
    system = UnifiedHealthSystem(config)
    
    # Register test plugins
    db_plugin = DatabaseHealthPlugin(mock_connection_manager)
    system.register_plugin(db_plugin)
    
    return system

@pytest.mark.asyncio
async def test_unified_health_check(health_system):
    """Test unified health system"""
    # Test overall health
    overall_health = await health_system.get_overall_health()
    assert overall_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    # Test component-specific health
    db_health = await health_system.check_health(component_name="database")
    assert "db_connection" in db_health
    assert db_health["db_connection"].status == HealthStatus.HEALTHY

@pytest.mark.asyncio
async def test_plugin_registration(health_system):
    """Test plugin registration and discovery"""
    registered_checkers = health_system.get_registered_checkers()
    
    # Should include database plugin checkers
    assert "db_connection" in registered_checkers
    assert "db_pool_status" in registered_checkers
    assert "db_query_performance" in registered_checkers
```

### Integration Test Updates
```python
@pytest.mark.asyncio
async def test_health_monitoring_integration():
    """Test integration between health system and monitoring"""
    health_system = UnifiedHealthSystem(config)
    
    # Register all plugins
    health_system.register_plugin(DatabaseHealthPlugin(connection_manager))
    health_system.register_plugin(MLHealthPlugin(model_registry))
    health_system.register_plugin(ExternalServicePlugin(service_configs))
    
    # Test comprehensive health check
    start_time = time.time()
    health_results = await health_system.check_health(include_details=True)
    duration = time.time() - start_time
    
    # Verify performance
    assert duration < 5.0  # Should complete within 5 seconds
    
    # Verify all expected components are checked
    expected_components = [
        "db_connection", "db_pool_status", 
        "ml_models", "ml_resources",
        "redis_cache", "external_apis"
    ]
    
    for component in expected_components:
        assert component in health_results
        assert health_results[component].duration_ms > 0
```

## Monitoring Integration Updates

### Prometheus Metrics Migration
```python
# Before: Individual metric collectors
from prometheus_client import Gauge, Counter

db_health_gauge = Gauge('database_health_status', 'Database health status')
ml_health_gauge = Gauge('ml_health_status', 'ML health status')

# After: Unified metric collection (built-in)
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem

# Metrics are automatically collected by the unified system
health_system = UnifiedHealthSystem(config)

# Access built-in metrics collector if needed
metrics_collector = health_system._metrics_collector

# Metrics automatically exported:
# - health_check_status{component="db_connection"} 
# - health_check_duration_seconds{component="db_connection"}
# - health_checks_total{component="db_connection", status="healthy"}
```

### Dashboard Updates
```yaml
# Grafana dashboard query migration

# Before: Multiple separate queries
# Query 1: database_health_status
# Query 2: ml_health_status  
# Query 3: performance_health_status

# After: Single unified query pattern
# Query: health_check_status{component=~".*"}
# This shows health status for all components

# Example dashboard queries:
grafana_queries:
  overall_health:
    expr: 'min(health_check_status)'
    description: 'Overall system health (minimum of all components)'
    
  component_health:
    expr: 'health_check_status'
    description: 'Health status by component'
    
  health_check_performance:
    expr: 'health_check_duration_seconds'
    description: 'Health check execution time by component'
    
  health_check_frequency:
    expr: 'rate(health_checks_total[5m])'
    description: 'Health check execution rate'
```

## Troubleshooting

### Common Migration Issues

#### Issue 1: Plugin Registration Errors
```python
# Problem: Plugin not properly registered
health_system = UnifiedHealthSystem(config)
# Plugin registration forgotten
health_results = await health_system.check_health()  # Empty results

# Solution: Ensure all plugins are registered
health_system = UnifiedHealthSystem(config)
health_system.register_plugin(DatabaseHealthPlugin(connection_manager))
health_system.register_plugin(MLHealthPlugin(model_registry))
# ... register all needed plugins
```

#### Issue 2: Configuration Mismatch
```python
# Problem: Plugin expects config that doesn't exist
db_plugin = DatabaseHealthPlugin(connection_manager)
# Plugin expects specific config keys but they're missing

# Solution: Provide complete plugin configuration
plugin_config = {
    'query_performance_threshold': 1000,
    'pool_check_interval': 60,
    'critical_checks': ['db_connection']
}
db_plugin = DatabaseHealthPlugin(connection_manager, plugin_config)
```

#### Issue 3: Async/Sync Issues in Health Checks
```python
# Problem: Mixing sync and async in health checks
class CustomPlugin(HealthPlugin):
    def register_checkers(self, system):
        system.register_checker("sync_check", self._sync_check)  # ❌ Wrong
    
    def _sync_check(self):  # Sync function
        return HealthCheckResult(status=HealthStatus.HEALTHY)

# Solution: All health checkers must be async
class CustomPlugin(HealthPlugin):
    def register_checkers(self, system):
        system.register_checker("async_check", self._async_check)  # ✅ Correct
    
    async def _async_check(self):  # Async function
        return HealthCheckResult(status=HealthStatus.HEALTHY)
```

### Debug Mode
```python
# Enable health system debug logging
import logging
logging.getLogger("prompt_improver.performance.monitoring.health").setLevel(logging.DEBUG)

# Enable detailed health check logging
config = {
    'critical_check_interval': 30,
    'debug_mode': True,  # Enable debug mode
    'log_health_check_details': True
}

health_system = UnifiedHealthSystem(config)
```

## Rollback Plan

### Preparation
```bash
# Tag current state
git tag health-migration-checkpoint-$(date +%Y%m%d)

# Keep legacy monitoring code temporarily
mkdir legacy_health_backup
cp -r src/prompt_improver/*/health/ legacy_health_backup/
```

### Rollback Steps
1. **Restore configuration**:
   ```bash
   cp config/monitoring_config.backup.yaml config/monitoring_config.yaml
   ```

2. **Restore legacy health monitors**:
   ```bash
   # Restore individual health monitors if needed
   git checkout health-migration-checkpoint-$(date +%Y%m%d) -- src/prompt_improver/database/health/
   git checkout health-migration-checkpoint-$(date +%Y%m%d) -- src/prompt_improver/ml/health/
   ```

3. **Update imports back to legacy**:
   ```python
   # Revert to individual monitor imports
   from prompt_improver.database.health.database_health_monitor import DatabaseHealthMonitor
   from prompt_improver.ml.health.ml_health_monitor import MLHealthMonitor
   ```

## Validation Checklist

### Pre-Migration
- [ ] All existing health checks identified and documented
- [ ] Current health monitoring performance baseline established
- [ ] Legacy health configuration backed up
- [ ] Monitoring dashboards and alerts documented

### During Migration
- [ ] All plugins registered and functional
- [ ] Configuration migrated and validated
- [ ] Health check results match legacy behavior
- [ ] Performance meets or exceeds baseline

### Post-Migration
- [ ] All health functionality working as expected
- [ ] Monitoring dashboards updated with new metrics
- [ ] Alerting rules updated for new health check names
- [ ] Legacy health monitoring code removed
- [ ] Documentation updated

## Getting Help
- **Documentation**: [Health Monitoring API Reference](../user/health-monitoring-api.md)
- **Plugin Development**: [Health Plugin Development Guide](../developer/health-plugin-development.md)
- **Examples**: [Health Monitoring Examples](../../examples/health_monitoring_examples.py)
- **Issues**: Create issues in the project repository with `health-migration` tag
- **Team Support**: Reach out to the monitoring team for migration assistance

---

*Last Updated: 2025-01-28*  
*Migration Guide Version: 1.0*