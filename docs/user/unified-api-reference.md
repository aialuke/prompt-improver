# Unified API Reference

## Overview
This document provides comprehensive API reference for all unified interfaces and protocol boundaries implemented in the APES refactoring project.

## Core Protocols

### ConnectionManagerProtocol

The unified interface for all connection management operations.

```python
from prompt_improver.core.protocols.connection_protocol import (
    ConnectionManagerProtocol, 
    ConnectionMode
)
```

#### Interface Definition
```python
from typing import TypedDict
from typing_extensions import Unpack

class ConnectionOptions(TypedDict, total=False):
    timeout: float
    retries: int
    readonly: bool

class ConnectionManagerProtocol(Protocol):
    """Unified protocol for connection management"""

    async def get_connection(
        self,
        mode: ConnectionMode = ConnectionMode.READ_WRITE,
        **kwargs: Unpack[ConnectionOptions]
    ) -> AsyncContextManager[Any]:
        """
        Get a connection with specified mode.

        Args:
            mode: Connection operation mode (READ_ONLY, READ_WRITE, BATCH, TRANSACTIONAL)
            **kwargs: Additional connection parameters

        Returns:
            Async context manager for the connection

        Raises:
            ConnectionError: If connection cannot be established
            TimeoutError: If connection timeout exceeded
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform connection health check.
        
        Returns:
            Dictionary containing:
            - status: "healthy" | "degraded" | "unhealthy"
            - active_connections: int
            - pool_utilization: float (0.0-1.0)
            - avg_response_time: float (milliseconds)
            - last_check: datetime
        """
        ...

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        ...

    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection pool information.
        
        Returns:
            Dictionary containing:
            - pool_size: int
            - active_connections: int  
            - idle_connections: int
            - overflow_connections: int
            - invalid_connections: int
        """
        ...

    def is_healthy(self) -> bool:
        """
        Quick health status check.
        
        Returns:
            True if connection manager is healthy
        """
        ...
```

#### Connection Modes
```python
class ConnectionMode(Enum):
    """Connection operation modes"""
    READ_ONLY = "read_only"        # Read operations only
    READ_WRITE = "read_write"      # Standard read/write operations
    BATCH = "batch"                # Optimized for batch operations
    TRANSACTIONAL = "transactional" # Full transaction support
```

#### Usage Examples
```python
# Basic connection usage
async with connection_manager.get_connection() as conn:
    result = await conn.execute(select(User).where(User.active == True))
    users = result.scalars().all()

# Read-only connection for queries
async with connection_manager.get_connection(
    mode=ConnectionMode.READ_ONLY
) as conn:
    result = await conn.execute(select(User.name))
    names = result.scalars().all()

# Batch operations
async with connection_manager.get_connection(
    mode=ConnectionMode.BATCH
) as conn:
    # Batch insert/update operations
    await conn.execute(insert(User).values(users_data))
    await conn.commit()

# Transactional operations
async with connection_manager.get_connection(
    mode=ConnectionMode.TRANSACTIONAL
) as conn:
    try:
        await conn.execute(update_query_1)
        await conn.execute(update_query_2)
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
```

### HealthMonitorProtocol

The unified interface for health monitoring operations.

```python
from prompt_improver.core.protocols.health_protocol import (
    HealthMonitorProtocol,
    HealthStatus,
    HealthCheckResult
)
```

#### Interface Definition
```python
class HealthMonitorProtocol(Protocol):
    """Unified protocol for health monitoring operations"""
    
    async def check_health(
        self,
        component_name: Optional[str] = None,
        include_details: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """
        Perform health checks on registered components.
        
        Args:
            component_name: Specific component to check, None for all
            include_details: Whether to include detailed health information
            
        Returns:
            Dictionary mapping component names to health results
        """
        ...

    def register_checker(
        self,
        name: str,
        checker: Callable[[], Any],
        timeout: float = 30.0,
        critical: bool = False
    ) -> None:
        """
        Register a health checker function.
        
        Args:
            name: Unique name for the health checker
            checker: Async callable that performs the health check
            timeout: Timeout for the health check in seconds
            critical: Whether this check is critical for overall health
        """
        ...

    def unregister_checker(self, name: str) -> bool:
        """
        Unregister a health checker.
        
        Args:
            name: Name of the checker to remove
            
        Returns:
            True if checker was removed, False if not found
        """
        ...

    def get_registered_checkers(self) -> List[str]:
        """
        Get list of registered health checker names.
        
        Returns:
            List of registered checker names
        """
        ...

    async def get_overall_health(self) -> HealthCheckResult:
        """
        Get overall system health status.
        
        Returns:
            Combined health result for the entire system
        """
        ...

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health monitoring summary and statistics.
        
        Returns:
            Dictionary containing:
            - total_checkers: int
            - critical_checkers: int
            - last_check_time: datetime
            - check_frequency: Dict[str, int]
            - failure_rates: Dict[str, float]
        """
        ...
```

#### Health Status Enumeration
```python
class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"      # All systems operational
    DEGRADED = "degraded"    # Some issues but functional
    UNHEALTHY = "unhealthy"  # Critical issues detected
    UNKNOWN = "unknown"      # Unable to determine status
```

#### Health Check Result
```python
@dataclass
class HealthCheckResult:
    """Standardized health check result"""
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    check_name: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Usage Examples
```python
# Register health checkers
health_monitor.register_checker(
    name="database_connection",
    checker=check_database_connection,
    timeout=10.0,
    critical=True
)

# Check overall system health
overall_health = await health_monitor.get_overall_health()
print(f"System status: {overall_health.status.value}")
print(f"Message: {overall_health.message}")

# Check specific component health
db_health = await health_monitor.check_health(component_name="database")
for check_name, result in db_health.items():
    print(f"{check_name}: {result.status.value} ({result.duration_ms}ms)")

# Get health summary statistics
summary = health_monitor.get_health_summary()
print(f"Total checkers: {summary['total_checkers']}")
print(f"Critical checkers: {summary['critical_checkers']}")
```

## Unified Implementation Classes

### DatabaseServices

The consolidated connection manager implementation.

```python
from prompt_improver.database import DatabaseServicesV2
from prompt_improver.database.config import UnifiedDatabaseConfig
```

#### Configuration
```python
@dataclass
class UnifiedDatabaseConfig:
    """Consolidated configuration for all connection features"""
    
    # Connection settings
    host: str
    port: int = 5432
    database: str
    username: str
    password: str
    
    # Pool configuration
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # High availability 
    replica_hosts: List[str] = field(default_factory=list)
    failover_timeout: float = 5.0
    health_check_interval: int = 30
    
    # Performance tuning
    statement_cache_size: int = 1000
    prepared_statement_cache_size: int = 1000
    connection_timeout: float = 10.0
    
    # Feature flags
    enable_ha: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    strict_mode: bool = False
```

#### Usage Examples
```python
# Basic setup
config = UnifiedDatabaseConfig(
    host="localhost",
    database="apes_db",
    username=os.getenv("DATABASE_USER"),
    password=os.getenv("DATABASE_PASSWORD")
)

connection_manager = DatabaseServices(config)

# High availability setup
ha_config = UnifiedDatabaseConfig(
    host="db-primary",
    database="apes_db",
    username=os.getenv("DATABASE_USER"),
    password=os.getenv("DATABASE_PASSWORD"),
    replica_hosts=["db-replica1", "db-replica2"],
    enable_ha=True,
    failover_timeout=3.0
)

ha_manager = DatabaseServices(ha_config)

# Performance-optimized setup
perf_config = UnifiedDatabaseConfig(
    host="localhost",
    database="apes_db",
    username=os.getenv("DATABASE_USER"),
    password=os.getenv("DATABASE_PASSWORD"),
    pool_size=50,
    max_overflow=100,
    statement_cache_size=2000,
    connection_timeout=5.0
)

perf_manager = DatabaseServices(perf_config)
```

#### Methods
```python
class DatabaseServices:
    """Consolidated connection manager implementation"""
    
    def __init__(self, config: UnifiedDatabaseConfig):
        """Initialize with unified configuration"""
        ...
    
    async def get_connection(
        self,
        mode: ConnectionMode = ConnectionMode.READ_WRITE,
        **kwargs: Unpack[ConnectionOptions]
    ) -> AsyncContextManager[AsyncConnection]:
        """Get connection with automatic failover and optimization"""
        ...
    
    async def get_session(
        self, 
        mode: ConnectionMode = ConnectionMode.READ_WRITE
    ) -> AsyncContextManager[AsyncSession]:
        """Get SQLAlchemy session with connection mode"""
        ...
    
    def get_sync_session(self) -> Session:
        """Get synchronous session for legacy compatibility"""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with all features"""
        ...
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get detailed connection and performance metrics"""
        ...
    
    async def close(self) -> None:
        """Clean shutdown with connection draining"""
        ...
```

### UnifiedHealthSystem

The consolidated health monitoring implementation.

```python
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem
from prompt_improver.performance.monitoring.health.config import HealthConfig
```

#### Configuration
```python
@dataclass
class HealthConfig:
    """Configuration for unified health monitoring"""
    
    # Check intervals
    critical_check_interval: int = 30      # seconds
    standard_check_interval: int = 300     # seconds  
    extended_check_interval: int = 1800    # seconds
    
    # Feature flags
    enable_metrics: bool = True
    enable_prometheus: bool = True
    enable_alerting: bool = True
    
    # Timeout settings
    default_check_timeout: float = 30.0
    critical_check_timeout: float = 10.0
    
    # Plugin settings
    plugin_discovery: bool = True
    custom_plugin_paths: List[str] = field(default_factory=list)
    
    # Alerting configuration
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)
```

#### Usage Examples
```python
# Basic health system setup
config = HealthConfig(
    critical_check_interval=30,
    standard_check_interval=300,
    enable_metrics=True
)

health_system = UnifiedHealthSystem(config)

# Register built-in plugins
from prompt_improver.performance.monitoring.health.plugin_adapters import (
    DatabaseHealthPlugin,
    MLHealthPlugin,
    ExternalServicePlugin
)

health_system.register_plugin(DatabaseHealthPlugin(connection_manager))
health_system.register_plugin(MLHealthPlugin(model_registry))
health_system.register_plugin(ExternalServicePlugin(service_configs))

# Custom plugin registration
from myapp.health import CustomHealthPlugin

custom_plugin = CustomHealthPlugin(custom_config)
health_system.register_plugin(custom_plugin)

# Start health monitoring
await health_system.start_monitoring()

# Get health status
overall_health = await health_system.get_overall_health()
detailed_health = await health_system.check_health(include_details=True)
```

## Plugin Development API

### HealthPlugin Base Class

Base class for developing health monitoring plugins.

```python
from prompt_improver.performance.monitoring.health.base import HealthPlugin
```

#### Interface
```python
class HealthPlugin(ABC):
    """Base class for health monitoring plugins"""
    
    @abstractmethod
    def register_checkers(self, system: HealthMonitorProtocol) -> None:
        """
        Register health checkers with the monitoring system.
        
        Args:
            system: Health monitoring system to register checkers with
        """
        ...
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get plugin metadata and information.
        
        Returns:
            Dictionary containing plugin information
        """
        return {
            "name": self.__class__.__name__,
            "version": getattr(self, "__version__", "1.0.0"),
            "description": self.__doc__ or "Health monitoring plugin"
        }
    
    async def initialize(self) -> None:
        """Initialize plugin resources (optional)"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources (optional)"""
        pass
```

#### Plugin Development Example
```python
class CustomServiceHealthPlugin(HealthPlugin):
    """Health monitoring plugin for custom service"""
    
    def __init__(self, service_config: Dict[str, Any]):
        self.config = service_config
        self.service_client = None
    
    async def initialize(self) -> None:
        """Initialize service client"""
        self.service_client = CustomServiceClient(self.config)
        await self.service_client.connect()
    
    def register_checkers(self, system: HealthMonitorProtocol) -> None:
        """Register service health checkers"""
        system.register_checker(
            name="custom_service_connectivity",
            checker=self._check_connectivity,
            timeout=10.0,
            critical=True
        )
        
        system.register_checker(
            name="custom_service_performance",
            checker=self._check_performance,
            timeout=30.0,
            critical=False
        )
        
        system.register_checker(
            name="custom_service_resources",
            checker=self._check_resources,
            timeout=15.0,
            critical=False
        )
    
    async def _check_connectivity(self) -> HealthCheckResult:
        """Check service connectivity"""
        try:
            response = await self.service_client.ping()
            if response.success:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Service is reachable",
                    details={"response_time": response.duration}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Service ping failed",
                    details={"error": response.error}
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Connectivity check failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    async def _check_performance(self) -> HealthCheckResult:
        """Check service performance metrics"""
        try:
            metrics = await self.service_client.get_metrics()
            
            # Check response time threshold
            if metrics.avg_response_time > self.config.get("response_time_threshold", 1000):
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="High response time detected",
                    details=metrics.to_dict()
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Performance within normal ranges",
                details=metrics.to_dict()
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Performance check failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    async def _check_resources(self) -> HealthCheckResult:
        """Check service resource utilization"""
        try:
            resources = await self.service_client.get_resource_usage()
            
            # Check resource thresholds
            cpu_threshold = self.config.get("cpu_threshold", 80)
            memory_threshold = self.config.get("memory_threshold", 85)
            
            if resources.cpu_usage > cpu_threshold:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"High CPU usage: {resources.cpu_usage}%",
                    details=resources.to_dict()
                )
            
            if resources.memory_usage > memory_threshold:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {resources.memory_usage}%",
                    details=resources.to_dict()
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Resource usage within normal ranges",
                details=resources.to_dict()
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Resource check failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    async def cleanup(self) -> None:
        """Cleanup service client"""
        if self.service_client:
            await self.service_client.disconnect()
```

## REST API Endpoints

### Health Monitoring Endpoints

```python
# FastAPI endpoints for health monitoring
from fastapi import FastAPI, HTTPException
from prompt_improver.performance.monitoring.health import UnifiedHealthSystem

app = FastAPI()

@app.get("/health")
async def get_overall_health():
    """
    Get overall system health status.
    
    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy" | "unknown",
            "message": "Status description",
            "timestamp": "2025-01-28T10:30:00Z",
            "checks_count": 15,
            "critical_failures": 0
        }
    """
    overall_health = await health_system.get_overall_health()
    summary = health_system.get_health_summary()
    
    return {
        "status": overall_health.status.value,
        "message": overall_health.message,
        "timestamp": overall_health.timestamp.isoformat(),
        "checks_count": summary["total_checkers"],
        "critical_failures": len([
            name for name, result in (await health_system.check_health()).items()
            if result.status == HealthStatus.UNHEALTHY and result.critical
        ])
    }

@app.get("/health/detailed")
async def get_detailed_health():
    """
    Get detailed health information for all components.
    
    Returns:
        {
            "overall": {...},
            "components": {
                "db_connection": {
                    "status": "healthy",
                    "message": "Database connection is healthy",
                    "duration_ms": 15.2,
                    "details": {...},
                    "critical": true
                },
                ...
            }
        }
    """
    overall_health = await health_system.get_overall_health()
    detailed_health = await health_system.check_health(include_details=True)
    
    return {
        "overall": {
            "status": overall_health.status.value,
            "message": overall_health.message,
            "timestamp": overall_health.timestamp.isoformat()
        },
        "components": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "details": result.details,
                "critical": result.critical,
                "timestamp": result.timestamp.isoformat()
            }
            for name, result in detailed_health.items()
        }
    }

@app.get("/health/{component}")
async def get_component_health(component: str):
    """
    Get health status for specific component.
    
    Args:
        component: Component name (e.g., "database", "ml", "external")
        
    Returns:
        {
            "component": "database",
            "checks": {
                "db_connection": {...},
                "db_pool_status": {...}
            }
        }
    """
    # Get health for specific component type
    all_health = await health_system.check_health(include_details=True)
    
    # Filter checks for the requested component
    component_checks = {
        name: result for name, result in all_health.items()
        if name.startswith(f"{component}_")
    }
    
    if not component_checks:
        raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
    
    return {
        "component": component,
        "checks": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "details": result.details,
                "critical": result.critical,
                "timestamp": result.timestamp.isoformat()
            }
            for name, result in component_checks.items()
        }
    }

@app.get("/health/summary")
async def get_health_summary():
    """
    Get health monitoring summary and statistics.
    
    Returns:
        {
            "total_checkers": 15,
            "critical_checkers": 5,
            "last_check_time": "2025-01-28T10:30:00Z",
            "check_frequency": {
                "critical": 30,
                "standard": 300,
                "extended": 1800
            },
            "failure_rates": {
                "db_connection": 0.001,
                "ml_models": 0.005
            }
        }
    """
    return health_system.get_health_summary()
```

### Connection Manager Endpoints

```python
@app.get("/admin/connections")
async def get_connection_info():
    """
    Get connection pool information (admin endpoint).
    
    Returns:
        {
            "pool_size": 20,
            "active_connections": 8,
            "idle_connections": 12,
            "overflow_connections": 0,
            "pool_utilization": 0.4,
            "avg_response_time": 45.2
        }
    """
    info = await connection_manager.get_connection_info()
    return info

@app.get("/admin/connections/health")
async def get_connection_health():
    """
    Get connection health status (admin endpoint).
    
    Returns:
        {
            "status": "healthy",
            "active_connections": 8,
            "pool_utilization": 0.4,
            "avg_response_time": 45.2,
            "last_check": "2025-01-28T10:30:00Z",
            "failover_status": "primary",
            "replica_health": ["healthy", "healthy"]
        }
    """
    health = await connection_manager.health_check()
    return health

@app.post("/admin/connections/failover")
async def trigger_failover():
    """
    Trigger manual failover to replica (admin endpoint).
    
    Returns:
        {
            "success": true,
            "message": "Failover completed successfully",
            "new_primary": "db-replica1",
            "failover_time": 2.3
        }
    """
    # This would only be available if HA is enabled
    if hasattr(connection_manager, 'trigger_failover'):
        result = await connection_manager.trigger_failover()
        return result
    else:
        raise HTTPException(status_code=501, detail="Failover not supported")
```

## Error Handling

### Connection Errors
```python
from prompt_improver.database.exceptions import (
    ConnectionError,
    ConnectionTimeoutError,
    FailoverError,
    PoolExhaustedError
)

try:
    async with connection_manager.get_connection() as conn:
        result = await conn.execute(query)
except ConnectionTimeoutError as e:
    logger.error(f"Connection timeout: {e}")
    # Implement retry logic or fallback
except PoolExhaustedError as e:
    logger.error(f"Connection pool exhausted: {e}")
    # Implement queue/backoff logic
except FailoverError as e:
    logger.error(f"Failover failed: {e}")
    # Implement emergency procedures
except ConnectionError as e:
    logger.error(f"Connection error: {e}")
    # Implement general error handling
```

### Health Check Errors
```python
from prompt_improver.performance.monitoring.health.exceptions import (
    HealthCheckTimeoutError,
    PluginRegistrationError,
    HealthMonitoringError
)

try:
    health_results = await health_system.check_health()
except HealthCheckTimeoutError as e:
    logger.warning(f"Health check timeout: {e}")
    # Some checks may have timed out, partial results available
except PluginRegistrationError as e:
    logger.error(f"Plugin registration failed: {e}")
    # Plugin setup issue, check plugin configuration
except HealthMonitoringError as e:
    logger.error(f"Health monitoring error: {e}")
    # General health monitoring failure
```

## Performance Considerations

### Connection Pool Tuning
```python
# High-throughput configuration
high_throughput_config = UnifiedDatabaseConfig(
    pool_size=50,                    # Large pool for concurrent connections
    max_overflow=100,                # Allow overflow for spikes
    pool_timeout=60,                 # Wait longer for available connections
    connection_timeout=10.0,         # Quick connection establishment
    statement_cache_size=2000,       # Large cache for repeated queries
    prepared_statement_cache_size=2000
)

# Memory-optimized configuration
memory_optimized_config = UnifiedDatabaseConfig(
    pool_size=10,                    # Smaller pool to save memory
    max_overflow=20,                 # Limited overflow
    pool_recycle=1800,               # Recycle connections frequently
    statement_cache_size=500,        # Smaller cache
    prepared_statement_cache_size=500
)

# Low-latency configuration
low_latency_config = UnifiedDatabaseConfig(
    pool_size=20,
    connection_timeout=3.0,          # Fast connection timeout
    pool_pre_ping=True,              # Verify connections before use
    pool_reset_on_return='commit'    # Quick connection reset
)
```

### Health Check Performance
```python
# Performance-optimized health configuration
perf_health_config = HealthConfig(
    critical_check_interval=30,      # Frequent critical checks
    standard_check_interval=300,     # Less frequent standard checks
    extended_check_interval=1800,    # Infrequent extended checks
    
    critical_check_timeout=5.0,      # Quick timeout for critical checks
    default_check_timeout=15.0,      # Moderate timeout for others
    
    enable_metrics=True,             # Enable for monitoring
    enable_prometheus=True           # Export to Prometheus
)
```

## Monitoring and Metrics

### Prometheus Metrics
```python
# Connection manager metrics (automatically exported)
connection_pool_size                    # Current pool size
connection_pool_active                  # Active connections
connection_pool_idle                    # Idle connections  
connection_pool_overflow                # Overflow connections
connection_pool_utilization             # Pool utilization (0.0-1.0)
connection_response_time_seconds        # Connection response time histogram
connection_errors_total                 # Total connection errors by type

# Health monitoring metrics (automatically exported)
health_check_status{component}          # Health status by component (1.0=healthy, 0.5=degraded, 0.0=unhealthy)
health_check_duration_seconds{component} # Health check duration histogram
health_checks_total{component,status}   # Total health checks by component and status
```

### Custom Metrics Collection
```python
# Access built-in metrics collectors
connection_metrics = await connection_manager.get_metrics()
health_metrics = health_system.get_health_summary()

# Custom metrics integration
from prometheus_client import Counter, Histogram

custom_requests = Counter('custom_requests_total', 'Total custom requests')
custom_duration = Histogram('custom_request_duration_seconds', 'Custom request duration')

# Integrate with unified systems
async with connection_manager.get_connection() as conn:
    with custom_duration.time():
        result = await conn.execute(query)
        custom_requests.inc()
```

---

*Last Updated: 2025-01-28*  
*API Reference Version: 1.0*