# Circular Import Remediation Plan - 2025 Best Practices

## Executive Summary

This document outlines a comprehensive remediation plan for the 4 circular import chains identified in the APES codebase, using modern 2025 Python best practices. The plan prioritizes clean break modernization over legacy compatibility.

## Analysis Results

### Severity Breakdown
- **HIGH Priority**: 3 chains requiring immediate attention
- **LOW Priority**: 1 chain (TYPE_CHECKING only)
- **CRITICAL**: 0 chains (✅ good baseline)

### Identified Circular Chains

#### 1. ML Orchestration (HIGH) 🔴
**Chain**: `component_registry` ↔ `component_definitions`
- **Root Cause**: Bidirectional dependency between registry and definitions
- **Impact**: Blocks ML orchestration module loading
- **Strategy**: Extract shared types to separate module + dependency injection

#### 2. MCP Server (HIGH) 🔴  
**Chain**: `mcp_server` → `mcp_server` (self-referential)
- **Root Cause**: Module importing from itself via relative import
- **Impact**: Prevents MCP server initialization
- **Strategy**: Remove self-referential import, restructure module

#### 3. CLI Core (HIGH) 🔴
**Chain**: `emergency_operations` ↔ `signal_handler`
- **Root Cause**: Emergency operations and signal handling are tightly coupled
- **Impact**: CLI emergency features cannot load
- **Strategy**: Lazy imports + event-driven architecture

#### 4. AutoML (LOW) 🟡
**Chain**: `orchestrator` ↔ `callbacks` 
- **Root Cause**: Type hints creating circular reference
- **Impact**: Minimal (TYPE_CHECKING only)
- **Strategy**: Already properly using TYPE_CHECKING, verify implementation

## Modern Remediation Strategies

### Strategy 1: TYPE_CHECKING Pattern (2025 Standard)

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import SomeType

def function(param: "SomeType") -> None:
    # Runtime imports inside functions
    from .other_module import SomeType
    actual_param = SomeType(param)
```

### Strategy 2: Lazy Import Functions

```python
def _get_dependency():
    """Lazy import to avoid circular dependency."""
    from .other_module import Dependency
    return Dependency

class MyClass:
    def method(self):
        Dependency = _get_dependency()
        return Dependency()
```

### Strategy 3: Dependency Injection

```python
class ComponentRegistry:
    def __init__(self, definitions_provider=None):
        self._definitions_provider = definitions_provider or self._get_default_provider()
    
    def _get_default_provider(self):
        from .component_definitions import ComponentDefinitions
        return ComponentDefinitions()
```

### Strategy 4: Shared Module Extraction

```python
# New file: shared_types.py
@dataclass
class ComponentInfo:
    name: str
    tier: ComponentTier
    capabilities: List[ComponentCapability]

# component_registry.py and component_definitions.py both import from shared_types
```

### Strategy 5: Event-Driven Architecture

```python
# Instead of direct imports, use event bus
class EmergencyOperations:
    def __init__(self, event_bus=None):
        self.event_bus = event_bus or get_event_bus()
    
    def trigger_emergency(self):
        self.event_bus.emit("emergency_triggered", data)
```

## Implementation Priority Matrix

| Chain | Severity | Complexity | Impact | Priority |
|-------|----------|------------|--------|----------|
| MCP Server | HIGH | Low | High | 1 |
| CLI Core | HIGH | Medium | Medium | 2 |
| ML Orchestration | HIGH | High | Low | 3 |
| AutoML | LOW | Low | Low | 4 |

## Detailed Remediation Plans

### 1. MCP Server (Priority 1)

**Problem**: Self-referential import in `mcp_server/__init__.py`
```python
from .mcp_server import APESMCPServer, main  # Line 3
```

**Solution**: Remove self-referential import
```python
# mcp_server/__init__.py - AFTER
from .server import APESMCPServer  # Rename mcp_server.py to server.py
from .main import main             # Extract main to separate file

__all__ = ["APESMCPServer", "main"]
```

**Steps**:
1. Rename `mcp_server.py` to `server.py`
2. Extract `main()` function to `main.py`
3. Update `__init__.py` imports
4. Update external references

### 2. CLI Core (Priority 2)

**Problem**: `emergency_operations` ↔ `signal_handler` circular dependency

**Solution**: Lazy imports + event-driven pattern
```python
# emergency_operations.py - AFTER
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .signal_handler import SignalContext

def _get_signal_handler():
    from .signal_handler import SignalHandler
    return SignalHandler

class EmergencyOperationsManager:
    def __init__(self):
        self._signal_handler = None
    
    @property
    def signal_handler(self):
        if self._signal_handler is None:
            SignalHandler = _get_signal_handler()
            self._signal_handler = SignalHandler()
        return self._signal_handler
```

### 3. ML Orchestration (Priority 3)

**Problem**: `component_registry` ↔ `component_definitions` bidirectional dependency

**Solution**: Extract shared types + dependency injection
```python
# New file: shared/component_types.py
from enum import Enum
from dataclasses import dataclass
from typing import List

class ComponentTier(Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"

@dataclass
class ComponentInfo:
    name: str
    tier: ComponentTier
    capabilities: List[str]

# component_registry.py - AFTER
from typing import TYPE_CHECKING
from ..shared.component_types import ComponentTier, ComponentInfo

if TYPE_CHECKING:
    from ..config.component_definitions import ComponentDefinitions

def _get_component_definitions():
    from ..config.component_definitions import ComponentDefinitions
    return ComponentDefinitions()

class ComponentRegistry:
    def __init__(self, definitions=None):
        self._definitions = definitions or _get_component_definitions()
```

## Implementation Checklist

### Phase 3A: Quick Wins (1-2 hours)
- [ ] Fix MCP Server self-referential import
- [ ] Verify AutoML TYPE_CHECKING implementation
- [ ] Add import linting rules to prevent future issues

### Phase 3B: Medium Complexity (2-4 hours)  
- [ ] Implement CLI Core lazy imports
- [ ] Add event-driven patterns for emergency operations
- [ ] Update signal handler to use dependency injection

### Phase 3C: Complex Refactoring (4-6 hours)
- [ ] Extract ML orchestration shared types
- [ ] Implement dependency injection for component registry
- [ ] Restructure component definitions module
- [ ] Update all dependent modules

### Phase 4: Validation (1-2 hours)
- [ ] Test all imports load without circular dependency errors
- [ ] Run comprehensive test suite
- [ ] Verify metrics system functionality
- [ ] Update architectural documentation

## Success Criteria

1. **Zero circular imports** detected by analysis tool
2. **All modules import successfully** without errors
3. **Existing functionality preserved** (no regressions)
4. **Modern Python patterns** implemented throughout
5. **Import linting rules** prevent future circular imports
6. **Metrics system** fully functional and testable

## Risk Mitigation

1. **Incremental Implementation**: Fix one chain at a time
2. **Comprehensive Testing**: Test after each fix
3. **Rollback Plan**: Git branches for each phase
4. **Documentation**: Update architectural docs as we go
5. **Validation**: Continuous import testing during implementation

## Next Steps

1. Begin with Priority 1 (MCP Server) - lowest risk, highest impact
2. Implement import linting rules immediately
3. Create shared modules for common types
4. Gradually migrate to dependency injection patterns
5. Document new architectural patterns for future development
