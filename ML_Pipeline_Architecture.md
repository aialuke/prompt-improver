# ML Pipeline Architecture

**Last Updated**: 2025-08-06  
**Status**: Production Ready

## Overview

The ML Pipeline uses a modern Protocol-based dependency injection architecture with factory patterns for component creation and management. The system follows 2025 best practices with zero legacy patterns.

## Core Architecture

### 🏗️ Dependency Injection Pattern

The ML pipeline uses **pure dependency injection** without fallback patterns:

```python
# All components require explicit dependency injection
class MLPipelineOrchestrator:
    def __init__(
        self,
        event_bus: EventBusProtocol,
        workflow_engine: WorkflowEngineProtocol,
        resource_manager: ResourceManagerProtocol,
        config: OrchestratorConfig
    ):
        # No fallback patterns - all dependencies required
        self.event_bus = event_bus
        self.workflow_engine = workflow_engine
        self.resource_manager = resource_manager
```

### 🔌 Protocol Interfaces

Eight core Protocol interfaces define the contract for all services:

1. **MLflowServiceProtocol** - ML experiment tracking and model management
2. **CacheServiceProtocol** - Redis cache operations
3. **DatabaseServiceProtocol** - Database operations and transactions
4. **EventBusProtocol** - Event publishing and subscription
5. **ComponentRegistryProtocol** - Component discovery and registration
6. **ComponentFactoryProtocol** - Component instantiation with DI
7. **ComponentLoaderProtocol** - Component loading and management
8. **ServiceContainerProtocol** - Service lifecycle management

Location: `src/prompt_improver/core/protocols/ml_protocols.py`

### 🏭 Factory Pattern

Components are created through the factory pattern with automatic dependency resolution:

```python
# ComponentFactory handles dependency injection
factory = ComponentFactory(service_container)
component = await factory.create_component(component_spec)
```

Key factories:
- **MLPipelineOrchestratorFactory** - Creates orchestrator with all dependencies
- **ComponentFactory** - Creates components with Protocol-based DI
- **ExternalServiceFactory** - Creates external service adapters

Location: `src/prompt_improver/core/factories/`

### 📦 Component Tier System

Components are organized into a simple 3-tier system:

- **TIER_1**: Critical path components (security, core ML)
- **TIER_2**: Important but not critical (optimization, evaluation)
- **TIER_3**: Optional/experimental components

```python
class ComponentTier(Enum):
    TIER_1 = "tier_1"  # Critical
    TIER_2 = "tier_2"  # Important
    TIER_3 = "tier_3"  # Optional
```

## Service Container

The `MLServiceContainer` manages service lifecycle and dependency resolution:

```python
container = MLServiceContainer()
await container.initialize_all_services()
service = await container.get_service("event_bus")
await container.shutdown_all_services()
```

Features:
- Lazy service instantiation
- Automatic dependency resolution
- Health check integration
- Graceful shutdown

## Component Architecture

### Component Registration

Components are registered with metadata and dependencies:

```python
@dataclass
class ComponentSpec:
    name: str
    module_path: str
    class_name: str
    tier: str  # ComponentTier value
    dependencies: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    enabled: bool = True
```

### Component Discovery

The registry discovers components dynamically:

```python
registry = ComponentRegistry(config)
tier1_components = await registry.discover_components(ComponentTier.TIER_1)
```

## Performance Characteristics

- **Factory Instantiation**: 0.13 μs (extremely fast)
- **Factory + Config Setup**: 0.019ms (well under targets)
- **Event Throughput**: Supports >8,000 events/second
- **Memory Overhead**: Minimal service container overhead

## Testing Infrastructure

### Mock Services

Test fixtures provide Protocol-compliant mock services:

```python
@pytest.fixture
async def ml_service_container():
    """Provides mock service container for testing."""
    # Returns container with mock Protocol implementations
```

Available mocks:
- `mock_mlflow_service`
- `mock_cache_service`
- `mock_database_service`
- `mock_event_bus`
- `mock_service_container`

Location: `tests/conftest.py`

### Test Patterns

Tests use Protocol-based DI patterns:

```python
async def test_orchestrator(ml_service_container, component_factory):
    factory = MLPipelineOrchestratorFactory()
    orchestrator = await factory.create_from_container(
        ml_service_container,
        config.to_dict()
    )
```

## Key Files and Locations

### Core Infrastructure
- `src/prompt_improver/core/protocols/ml_protocols.py` - Protocol interfaces
- `src/prompt_improver/core/factories/` - Factory implementations
- `src/prompt_improver/core/di/ml_container.py` - Service container

### ML Orchestration
- `src/prompt_improver/ml/orchestration/core/ml_pipeline_orchestrator.py` - Main orchestrator
- `src/prompt_improver/ml/orchestration/core/component_registry.py` - Component registry
- `src/prompt_improver/ml/orchestration/connectors/` - Component connectors

### Configuration
- `src/prompt_improver/ml/orchestration/config/orchestrator_config.py` - Orchestrator config
- `src/prompt_improver/ml/orchestration/config/component_definitions.py` - Component definitions

## Design Principles

1. **No Fallback Patterns** - All dependencies must be explicitly injected
2. **Protocol-First** - All services implement Protocol interfaces
3. **Factory Pattern** - Components created through factories with DI
4. **Simple Tier System** - 3-tier component organization
5. **Testability** - Protocol-based mocking for all services
6. **Performance** - Minimal overhead from DI infrastructure

## Usage Examples

### Creating an Orchestrator

```python
from prompt_improver.core.factories.ml_pipeline_factory import create_production_orchestrator

# Create with automatic service initialization
orchestrator = await create_production_orchestrator(config_dict)
await orchestrator.initialize()

# Use orchestrator
workflow_id = await orchestrator.start_workflow("training", params)
status = await orchestrator.get_workflow_status(workflow_id)

await orchestrator.shutdown()
```

### Using Component Factory

```python
from prompt_improver.core.factories.component_factory import ComponentFactory

factory = ComponentFactory(service_container)
spec = await registry.get_component_spec("training_data_loader")
component = await factory.create_component(spec)
```

## Architecture Benefits

- **Clean Separation** - Clear boundaries between components via Protocols
- **Testability** - Easy to mock any service for testing
- **Maintainability** - No hidden dependencies or fallback patterns
- **Performance** - Minimal overhead from clean architecture
- **Type Safety** - Protocol interfaces provide type checking
- **Flexibility** - Easy to swap implementations via DI