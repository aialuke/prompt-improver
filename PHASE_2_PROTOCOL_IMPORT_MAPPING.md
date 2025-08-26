# Phase 2.1: Comprehensive Protocol Import Mapping Document

**EXECUTION STATUS**: Phase 1 Complete - 58 files deleted with zero impact  
**CURRENT PHASE**: P2.1 - Import Path Standardization for remaining 36 files  
**OBJECTIVE**: 100% mapping coverage for all core.protocols imports to consolidated equivalents

## Executive Summary

**Files to Process**: 43 files importing from `prompt_improver.core.protocols`  
**Protocol Categories**: 8 distinct protocol categories requiring standardization  
**Consolidation Target**: `shared/interfaces/protocols/` structure  
**Success Criteria**: All imports updated to consolidated paths with maintained functionality

## Detailed Import Mappings

### 1. ML PROTOCOLS CATEGORY (13 files)
**Source**: `prompt_improver.core.protocols.ml_protocols`  
**Target**: `shared.interfaces.protocols.ml` (lazy loading via get_ml_protocols())

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/cli/services/training_orchestrator.py:26` | `from prompt_improver.core.protocols.ml_protocols import (ComponentFactoryProtocol, ServiceContainerProtocol,)` | ComponentFactoryProtocol, ServiceContainerProtocol | `from prompt_improver.shared.interfaces.protocols.ml import ComponentFactoryProtocol, ServiceContainerProtocol` |
| `/cli/core/cli_orchestrator.py:27` | `from prompt_improver.core.protocols.ml_protocols import ServiceContainerProtocol` | ServiceContainerProtocol | `from prompt_improver.shared.interfaces.protocols.ml import ServiceContainerProtocol` |
| `/performance/monitoring/health/plugin_adapters.py:21` | `from prompt_improver.core.protocols.ml_protocols import EventBusProtocol` | EventBusProtocol | `from prompt_improver.shared.interfaces.protocols.ml import EventBusProtocol` |
| `/database/optimization_integration.py:15` | `from prompt_improver.core.protocols.ml_protocols import (DatabaseServiceProtocol, MLflowServiceProtocol,)` | DatabaseServiceProtocol, MLflowServiceProtocol | `from prompt_improver.shared.interfaces.protocols.ml import DatabaseServiceProtocol, MLflowServiceProtocol` |
| `/core/di/container_orchestrator.py:28` | `from prompt_improver.core.protocols.ml_protocols import (ComponentInvokerProtocol, ComponentLoaderProtocol, ComponentRegistryProtocol,)` | ComponentInvokerProtocol, ComponentLoaderProtocol, ComponentRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.ml import ComponentInvokerProtocol, ComponentLoaderProtocol, ComponentRegistryProtocol` |
| `/core/di/container_orchestrator.py:46` | `from prompt_improver.core.protocols.ml_protocols import (CacheServiceProtocol, DatabaseServiceProtocol, ExternalServicesConfigProtocol,)` | CacheServiceProtocol, DatabaseServiceProtocol, ExternalServicesConfigProtocol | `from prompt_improver.shared.interfaces.protocols.ml import CacheServiceProtocol, DatabaseServiceProtocol, ExternalServicesConfigProtocol` |
| `/core/di/database_container.py:21` | `from prompt_improver.core.protocols.ml_protocols import (CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo,)` | CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo | `from prompt_improver.shared.interfaces.protocols.ml import CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo` |
| `/core/di/database_container.py:35` | `from prompt_improver.core.protocols.ml_protocols import (CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo,)` | CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo | `from prompt_improver.shared.interfaces.protocols.ml import CacheServiceProtocol, DatabaseServiceProtocol, ServiceConnectionInfo` |
| `/core/factories/ml_pipeline_factory.py:12` | `from prompt_improver.core.protocols.ml_protocols import (...,)` | [Multiple protocols] | `from prompt_improver.shared.interfaces.protocols.ml import ...` |
| `/core/factories/component_factory.py:15` | `from prompt_improver.core.protocols.ml_protocols import (...,)` | [Multiple protocols] | `from prompt_improver.shared.interfaces.protocols.ml import ...` |
| `/ml/orchestration/core/component_factory_simple.py:5` | `from prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol` | ComponentFactoryProtocol | `from prompt_improver.shared.interfaces.protocols.ml import ComponentFactoryProtocol` |
| `/ml/orchestration/core/event_bus_simple.py:6` | `from prompt_improver.core.protocols.ml_protocols import EventBusProtocol` | EventBusProtocol | `from prompt_improver.shared.interfaces.protocols.ml import EventBusProtocol` |
| `/ml/orchestration/core/workflow_engine_simple.py:6` | `from prompt_improver.core.protocols.ml_protocols import WorkflowEngineProtocol` | WorkflowEngineProtocol | `from prompt_improver.shared.interfaces.protocols.ml import WorkflowEngineProtocol` |

### 2. HEALTH/MONITORING PROTOCOLS CATEGORY (6 files)  
**Source**: `prompt_improver.core.protocols.health_protocol`  
**Target**: `shared.interfaces.protocols.monitoring` (lazy loading via get_monitoring_protocols())

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/performance/monitoring/health/unified_health_system.py:30` | `from prompt_improver.core.protocols.health_protocol import (HealthCheckResult, HealthStatus,)` | HealthCheckResult, HealthStatus | `from prompt_improver.shared.interfaces.protocols.monitoring import HealthCheckResult, HealthStatus` |
| `/performance/monitoring/health/unified_health_system.py:73` | `from prompt_improver.core.protocols.health_protocol import (HealthCheckResult, HealthMonitorProtocol, HealthStatus,)` | HealthCheckResult, HealthMonitorProtocol, HealthStatus | `from prompt_improver.shared.interfaces.protocols.monitoring import HealthCheckResult, HealthMonitorProtocol, HealthStatus` |
| `/performance/monitoring/health/plugin_adapters.py:17` | `from prompt_improver.core.protocols.health_protocol import (HealthCheckResult, HealthStatus,)` | HealthCheckResult, HealthStatus | `from prompt_improver.shared.interfaces.protocols.monitoring import HealthCheckResult, HealthStatus` |
| `/performance/monitoring/health/plugin_adapters.py:222` | `from prompt_improver.core.protocols.health_protocol import (HealthCheckResult, HealthMonitorProtocol, HealthStatus,)` | HealthCheckResult, HealthMonitorProtocol, HealthStatus | `from prompt_improver.shared.interfaces.protocols.monitoring import HealthCheckResult, HealthMonitorProtocol, HealthStatus` |
| `/monitoring/unified/facade.py:20` | `from prompt_improver.core.protocols.health_protocol import (...,)` | [Multiple health protocols] | `from prompt_improver.shared.interfaces.protocols.monitoring import ...` |
| `/core/protocols/__init__.py:79` | `from prompt_improver.core.protocols.health_protocol import (HealthCheckResult as SimpleHealthCheckResult, HealthMonitorProtocol, HealthStatus as SimpleHealthStatus,)` | Multiple health protocols | Update __init__.py to import from consolidated location |

### 3. RESILIENCE/RETRY PROTOCOLS CATEGORY (7 files)
**Source**: `prompt_improver.core.protocols.retry_protocols`  
**Target**: `shared.interfaces.protocols.core`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/core/services/resilience/retry_orchestrator_service.py:34` | `from prompt_improver.core.protocols.retry_protocols import (MetricsRegistryProtocol, RetryConfigProtocol, RetryObserverProtocol,)` | MetricsRegistryProtocol, RetryConfigProtocol, RetryObserverProtocol | `from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol, RetryConfigProtocol, RetryObserverProtocol` |
| `/core/services/resilience/backoff_strategy_service.py:39` | `from prompt_improver.core.protocols.retry_protocols import RetryStrategy` | RetryStrategy | `from prompt_improver.shared.interfaces.protocols.core import RetryStrategy` |
| `/core/services/resilience/circuit_breaker_service.py:19` | `from prompt_improver.core.protocols.retry_protocols import (CircuitBreakerProtocol, MetricsRegistryProtocol,)` | CircuitBreakerProtocol, MetricsRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.core import CircuitBreakerProtocol, MetricsRegistryProtocol` |
| `/core/services/resilience/retry_service_facade.py:31` | `from prompt_improver.core.protocols.retry_protocols import (RetryConfigProtocol, RetryObserverProtocol,)` | RetryConfigProtocol, RetryObserverProtocol | `from prompt_improver.shared.interfaces.protocols.core import RetryConfigProtocol, RetryObserverProtocol` |
| `/core/services/resilience/retry_configuration_service.py:28` | `from prompt_improver.core.protocols.retry_protocols import RetryConfigProtocol` | RetryConfigProtocol | `from prompt_improver.shared.interfaces.protocols.core import RetryConfigProtocol` |
| `/core/di/core_container.py:17` | `from prompt_improver.core.protocols.retry_protocols import MetricsRegistryProtocol` | MetricsRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol` |
| `/core/di/container_orchestrator.py:54` | `from prompt_improver.core.protocols.retry_protocols import MetricsRegistryProtocol` | MetricsRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol` |

### 4. CLI/FACADE PROTOCOLS CATEGORY (4 files)
**Source**: `prompt_improver.core.protocols.facade_protocols`  
**Target**: `shared.interfaces.protocols.cli`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/cli/core/unified_components.py:21` | `from prompt_improver.core.protocols.facade_protocols import CLIFacadeProtocol` | CLIFacadeProtocol | `from prompt_improver.shared.interfaces.protocols.cli import CLIFacadeProtocol` |
| `/performance/baseline/unified_system.py:20` | `from prompt_improver.core.protocols.facade_protocols import PerformanceFacadeProtocol` | PerformanceFacadeProtocol | `from prompt_improver.shared.interfaces.protocols.cli import PerformanceFacadeProtocol` |
| `/core/config/unified_config.py:18` | `from prompt_improver.core.protocols.facade_protocols import ConfigFacadeProtocol` | ConfigFacadeProtocol | `from prompt_improver.shared.interfaces.protocols.cli import ConfigFacadeProtocol` |
| `/core/di/cli_container.py:19` | `from prompt_improver.core.protocols.facade_protocols import CLIFacadeProtocol` | CLIFacadeProtocol | `from prompt_improver.shared.interfaces.protocols.cli import CLIFacadeProtocol` |

### 5. SECURITY PROTOCOLS CATEGORY (1 file)
**Source**: `prompt_improver.core.protocols.security_service.security_protocols`  
**Target**: `shared.interfaces.protocols.security`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/security/services/security_service_facade.py:12` | `from prompt_improver.core.protocols.security_service.security_protocols import (SecurityServiceFacadeProtocol, AuthenticationServiceProtocol, AuthorizationServiceProtocol,)` | SecurityServiceFacadeProtocol, AuthenticationServiceProtocol, AuthorizationServiceProtocol | `from prompt_improver.shared.interfaces.protocols.security import SecurityServiceFacadeProtocol, AuthenticationServiceProtocol, AuthorizationServiceProtocol` |

### 6. PROMPT SERVICE PROTOCOLS CATEGORY (5 files)
**Source**: `prompt_improver.core.protocols.prompt_service.prompt_protocols`  
**Target**: `shared.interfaces.protocols.application`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/services/prompt/validation_service.py:18` | `from prompt_improver.core.protocols.prompt_service.prompt_protocols import (ValidationServiceProtocol,)` | ValidationServiceProtocol | `from prompt_improver.shared.interfaces.protocols.application import ValidationServiceProtocol` |
| `/services/prompt/prompt_analysis_service.py:25` | `from prompt_improver.core.protocols.prompt_service.prompt_protocols import (PromptAnalysisServiceProtocol,)` | PromptAnalysisServiceProtocol | `from prompt_improver.shared.interfaces.protocols.application import PromptAnalysisServiceProtocol` |
| `/services/prompt/rule_application_service.py:23` | `from prompt_improver.core.protocols.prompt_service.prompt_protocols import (RuleApplicationServiceProtocol,)` | RuleApplicationServiceProtocol | `from prompt_improver.shared.interfaces.protocols.application import RuleApplicationServiceProtocol` |
| `/services/prompt/facade.py:17` | `from prompt_improver.core.protocols.prompt_service.prompt_protocols import (PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol,)` | PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol | `from prompt_improver.shared.interfaces.protocols.application import PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol` |
| `/core/validation/service_registration.py:17` | `from prompt_improver.core.protocols.prompt_service.prompt_protocols import (PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol,)` | PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol | `from prompt_improver.shared.interfaces.protocols.application import PromptServiceFacadeProtocol, PromptAnalysisServiceProtocol, RuleApplicationServiceProtocol` |

### 7. RULE SELECTION PROTOCOLS CATEGORY (1 file)
**Source**: `prompt_improver.core.protocols.rule_selection_protocols`  
**Target**: `shared.interfaces.protocols.application`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/application/services/rule_selection_application_service.py:21` | `from prompt_improver.core.protocols.rule_selection_protocols import (RuleSelectionProtocol, RuleCacheProtocol, RuleLoaderProtocol,)` | RuleSelectionProtocol, RuleCacheProtocol, RuleLoaderProtocol | `from prompt_improver.shared.interfaces.protocols.application import RuleSelectionProtocol, RuleCacheProtocol, RuleLoaderProtocol` |

### 8. SYSTEM/CORE PROTOCOLS CATEGORY (7 files)
**Source**: Various core protocol files  
**Target**: `shared.interfaces.protocols.core`

| File | Current Import | Protocols | New Import |
|------|---------------|-----------|------------|
| `/core/protocols/__init__.py:63` | `from prompt_improver.core.protocols.connection_protocol import (ConnectionManagerProtocol, ConnectionMode,)` | ConnectionManagerProtocol, ConnectionMode | `from prompt_improver.shared.interfaces.protocols.core import ConnectionManagerProtocol, ConnectionMode` |
| `/core/protocols/__init__.py:74` | `from prompt_improver.core.protocols.datetime_protocol import (DateTimeServiceProtocol, DateTimeUtilsProtocol, TimeZoneServiceProtocol,)` | DateTimeServiceProtocol, DateTimeUtilsProtocol, TimeZoneServiceProtocol | `from prompt_improver.shared.interfaces.protocols.core import DateTimeServiceProtocol, DateTimeUtilsProtocol, TimeZoneServiceProtocol` |
| `/core/protocols/__init__.py:84` | `from prompt_improver.core.protocols.monitoring_protocol import (AdvancedHealthCheckProtocol, AlertingProtocol, BasicHealthCheckProtocol,)` | Multiple monitoring protocols | `from prompt_improver.shared.interfaces.protocols.monitoring import ...` |
| `/core/protocols/__init__.py:96` | `from prompt_improver.core.protocols.retry_protocols import (AnyMetricsRegistry, AnyRetryConfig, BackgroundTaskProtocol,)` | Multiple retry protocols | `from prompt_improver.shared.interfaces.protocols.core import ...` |
| `/core/di/monitoring_container.py:15` | `from prompt_improver.core.protocols.retry_protocols import MetricsRegistryProtocol` | MetricsRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.core import MetricsRegistryProtocol` |
| `/ml/orchestration/core/component_registry_simple.py:5` | `from prompt_improver.core.protocols.ml_protocols import ComponentRegistryProtocol` | ComponentRegistryProtocol | `from prompt_improver.shared.interfaces.protocols.ml import ComponentRegistryProtocol` |
| `/ml/orchestration/integration/direct_component_loader.py:10` | `from prompt_improver.core.protocols.ml_protocols import ComponentFactoryProtocol, ComponentRegistryProtocol, ComponentSpec` | ComponentFactoryProtocol, ComponentRegistryProtocol, ComponentSpec | `from prompt_improver.shared.interfaces.protocols.ml import ComponentFactoryProtocol, ComponentRegistryProtocol, ComponentSpec` |

## Implementation Priority

### High Priority (Critical Path)
1. **ML Protocols** (13 files) - Core functionality dependency
2. **Resilience Protocols** (7 files) - System stability critical
3. **Health Protocols** (6 files) - Monitoring functionality

### Medium Priority (Service Layer)
4. **Prompt Service Protocols** (5 files) - Business logic
5. **CLI/Facade Protocols** (4 files) - User interface

### Low Priority (Specialized)
6. **Security Protocols** (1 file) - Isolated functionality
7. **Rule Selection Protocols** (1 file) - Specific feature
8. **System Core Protocols** (7 files) - Foundation updates

## Validation Requirements

### Pre-Execution Validation
- [ ] Confirm all target protocols exist in consolidated files
- [ ] Verify lazy loading mechanisms work for ML/monitoring protocols  
- [ ] Test import resolution for all target paths

### Post-Execution Validation
- [ ] All imports resolve successfully
- [ ] No breaking changes in functionality
- [ ] Performance benchmarks maintained
- [ ] Test suite passes (85%+ coverage maintained)

## Risk Mitigation

### High-Risk Files
- `/core/protocols/__init__.py` - Central import hub (batch update required)
- ML orchestration files - Critical ML pipeline functionality
- DI container files - Dependency injection integrity

### Rollback Strategy
- Git commit after each category completion
- Immediate rollback capability for failed categories
- Automated test validation before proceeding

## Execution Commands for Phase 2

### Step 1: ML Protocols (13 files)
```bash
# Execute ML protocol import updates
find src -name "*.py" -exec grep -l "from prompt_improver.core.protocols.ml_protocols import" {} \; | head -13
```

### Step 2: Health/Monitoring Protocols (6 files)
```bash
# Execute health protocol import updates  
find src -name "*.py" -exec grep -l "from prompt_improver.core.protocols.health_protocol import" {} \; | head -6
```

### Step 3: Continue remaining categories...

## Success Metrics
- **Files Updated**: 43/43 (100%)
- **Import Paths Consolidated**: 8 categories standardized
- **Test Coverage**: Maintained at 85%+
- **Performance**: <2ms response times maintained
- **Zero Regression**: All functionality preserved

---
**NEXT PHASE**: P2.2 - Execute systematic import path updates per category priority