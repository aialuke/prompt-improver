# Startup Task Orchestration - APES

This document describes the implementation of `init_startup_tasks()` that orchestrates the startup of all core APES system components.

## Overview

The startup task orchestration system provides:

1. **Systematic Component Initialization** - Starts components in proper dependency order
2. **Background Task Management** - Manages long-running async tasks
3. **Session Store Cleanup** - Automatic cleanup of expired sessions  
4. **Periodic Batch Processing** - Continuous training data processing
5. **Health Monitoring** - Continuous system health monitoring
6. **Graceful Shutdown** - Clean shutdown of all components

## Core Components

### 1. BackgroundTaskManager

**Purpose**: Manages background async tasks with monitoring and graceful shutdown

**Features**:
- Concurrent task execution with configurable limits
- Task lifecycle monitoring (pending, running, completed, failed, cancelled)
- Automatic task cleanup and memory management
- Graceful shutdown with timeout handling
- Task restart capabilities on failure

**Integration**: Started first as other components depend on it for background task management.

### 2. SessionStore

**Purpose**: In-memory session management with TTL and automatic cleanup

**Features**:
- TTL-based session expiration
- Automatic periodic cleanup of expired sessions
- Thread-safe async operations with locks
- Session touch functionality to extend TTL
- Comprehensive error handling and logging

**Integration**: Initialized with configurable TTL and cleanup intervals.

### 3. BatchProcessor

**Purpose**: Processes training data in batches with retry logic and rate limiting

**Features**:
- Configurable batch size and processing timeouts
- Priority-based queue processing
- Exponential backoff retry logic with jitter
- Rate limiting and concurrency control
- Metrics collection and dry-run mode

**Integration**: Configured for optimal performance with periodic processing coroutine.

### 4. Health Monitor

**Purpose**: Continuous monitoring of system component health

**Features**:
- Parallel health checks across all components
- Hierarchical status calculation (healthy > warning > failed)
- Detailed error reporting and component diagnostics
- Configurable check intervals
- Integration with all major system components

**Integration**: Runs as background task with 60-second intervals.

## Implementation Details

### Startup Sequence

```python
async def init_startup_tasks(
    max_concurrent_tasks: int = 10,
    session_ttl: int = 3600,
    cleanup_interval: int = 300,
    batch_config: Optional[Dict] = None
) -> Dict[str, any]:
```

**Step 1: BackgroundTaskManager Initialization**
- Creates global background task manager instance
- Configures maximum concurrent task limits
- Starts internal monitoring coroutines

**Step 2: SessionStore Setup**
- Initializes session store with TTL configuration
- Starts automatic cleanup task
- Configures memory limits and cleanup intervals

**Step 3: BatchProcessor Configuration**
- Creates batch processor with custom or default configuration
- Sets up priority queues and retry logic
- Configures rate limiting and concurrency

**Step 4: Periodic Batch Processing**
- Submits periodic batch processing coroutine to BackgroundTaskManager
- Ensures continuous processing of queued training data
- Handles processing failures with restart logic

**Step 5: Health Monitor Activation**
- Initializes health service with all component checkers
- Starts continuous health monitoring background task
- Provides real-time system health status

**Step 6: Initial Health Verification**
- Runs comprehensive health check across all components
- Logs initial system status
- Provides startup validation

### Error Handling

**Partial Startup Cleanup**:
- Automatic cleanup of partially initialized components on startup failure
- Graceful shutdown of background tasks
- Resource cleanup and memory management

**Startup Failure Recovery**:
- Detailed error logging with component-specific failure information
- Graceful degradation for non-critical component failures
- Comprehensive error reporting in return status

### Graceful Shutdown

```python
async def shutdown_startup_tasks(timeout: float = 30.0) -> Dict[str, any]:
```

**Shutdown Sequence**:
1. Signal shutdown to all monitoring coroutines
2. Cancel all active background tasks
3. Wait for task completion with timeout
4. Shutdown BackgroundTaskManager with graceful timeout
5. Clean up global state and references

**Timeout Handling**:
- Configurable shutdown timeout
- Graceful handling of tasks that don't complete within timeout
- Force termination prevention of hanging shutdown

## Usage Patterns

### Basic Usage

```python
from prompt_improver.services.startup import init_startup_tasks, shutdown_startup_tasks

# Initialize all components
result = await init_startup_tasks(
    max_concurrent_tasks=15,
    session_ttl=7200,  # 2 hours
    cleanup_interval=600,  # 10 minutes
    batch_config={
        "batch_size": 20,
        "batch_timeout": 60,
        "max_attempts": 3,
        "concurrency": 5
    }
)

if result["status"] == "success":
    components = result["component_refs"]
    # Use components in application
    
# Graceful shutdown
await shutdown_startup_tasks()
```

### Context Manager Usage

```python
from prompt_improver.services.startup import startup_context

async with startup_context(max_concurrent_tasks=10) as components:
    # Components automatically started
    session_store = components["session_store"]
    batch_processor = components["batch_processor"]
    health_service = components["health_service"]
    
    # Use components in application logic
    
# Components automatically shut down on exit
```

### Application Integration

```python
class APESApplication:
    async def start(self):
        startup_result = await init_startup_tasks(**config)
        if startup_result["status"] != "success":
            raise RuntimeError(f"Startup failed: {startup_result['error']}")
        self.components = startup_result["component_refs"]
        
    async def stop(self):
        await shutdown_startup_tasks(timeout=30.0)
```

## Configuration Options

### BackgroundTaskManager
- `max_concurrent_tasks`: Maximum concurrent background tasks (default: 10)

### SessionStore  
- `session_ttl`: Session time-to-live in seconds (default: 3600)
- `cleanup_interval`: Cleanup task interval in seconds (default: 300)

### BatchProcessor
- `batch_size`: Records per batch (default: 10)
- `batch_timeout`: Seconds between batch processing (default: 30) 
- `max_attempts`: Maximum retry attempts (default: 3)
- `concurrency`: Concurrent processing workers (default: 3)
- `dry_run`: Enable dry-run mode for testing (default: False)

## Monitoring and Observability

### Startup Metrics
- Startup time measurement in milliseconds
- Component initialization success/failure tracking
- Active background task counting
- Error collection and reporting

### Runtime Monitoring
- Continuous health check results
- Component status tracking (healthy/warning/failed)
- Background task lifecycle monitoring
- Session store statistics and cleanup metrics

### Logging Integration
- Structured logging with component-specific loggers
- Startup/shutdown event logging with emojis for clarity
- Error logging with full context and stack traces
- Debug-level logging for detailed troubleshooting

## Best Practices

### Startup Configuration
- Configure timeouts appropriate for your environment
- Use larger batch sizes for high-throughput scenarios
- Adjust session TTL based on user workflow patterns
- Enable dry-run mode for testing and development

### Error Handling
- Always check startup result status before proceeding
- Implement proper error handling for startup failures
- Use graceful shutdown with appropriate timeouts
- Monitor startup time for performance regression detection

### Production Deployment
- Use context manager pattern for automatic cleanup
- Implement signal handlers for graceful shutdown
- Monitor health check results for early problem detection
- Configure logging levels appropriately for production

### Testing
- Use shorter timeouts and intervals for faster test execution
- Enable dry-run mode to prevent side effects during testing
- Mock external dependencies for reliable unit testing
- Test both successful startup and failure scenarios

## Integration with MCP Server

The startup orchestration integrates seamlessly with the MCP server:

```python
# In MCP server startup
async def mcp_server_startup():
    # Initialize all APES components
    startup_result = await init_startup_tasks(
        max_concurrent_tasks=20,
        batch_config={"dry_run": False}
    )
    
    if startup_result["status"] != "success":
        raise RuntimeError("Failed to initialize APES components")
    
    # MCP server ready to handle requests
    return startup_result["component_refs"]

# In MCP server shutdown
async def mcp_server_shutdown():
    await shutdown_startup_tasks(timeout=30.0)
```

## Performance Considerations

### Memory Management
- Background task references are properly managed to prevent memory leaks
- Session store has configurable size limits
- Automatic cleanup prevents unbounded memory growth

### Concurrency
- Configurable concurrent task limits prevent resource exhaustion
- Batch processing uses controlled concurrency for optimal performance
- Health checks run in parallel for faster system validation

### Scalability
- Components scale independently based on configuration
- Background task management handles varying workloads
- Batch processing adapts to queue size and processing capacity

## Troubleshooting

### Common Issues

**Startup Timeout**:
- Increase timeout values for slower environments
- Check for blocking operations in component initialization
- Review log files for specific component failures

**Memory Usage**:
- Reduce session TTL or cleanup interval
- Lower maximum concurrent task limits
- Check for memory leaks in background tasks

**Performance Issues**:
- Adjust batch size and processing intervals
- Review health check frequency
- Monitor background task queue sizes

### Debugging

**Enable Debug Logging**:
```python
import logging
logging.getLogger('prompt_improver.services.startup').setLevel(logging.DEBUG)
```

**Monitor Component Status**:
```python
from prompt_improver.services.startup import is_startup_complete, get_startup_task_count

print(f"Startup complete: {is_startup_complete()}")
print(f"Active tasks: {get_startup_task_count()}")
```

**Health Check Diagnostics**:
```python
health_result = await health_service.run_health_check()
for component, result in health_result.checks.items():
    if result.status.value != "healthy":
        print(f"{component}: {result.error}")
```
