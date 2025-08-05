# ADR-007: Unified Async Infrastructure Protocol

**Status**: ACCEPTED  
**Date**: 2025-08-01  
**Deciders**: System Architecture Team, Performance Engineering Team  
**Consulted**: ML Platform Team, DevOps Team  
**Informed**: All Development Teams  

## Context

During the ANALYSIS_3101.md async infrastructure consolidation, we identified 96+ direct `asyncio.create_task()` calls scattered throughout the codebase, creating:

- **Fragmented Task Management**: Multiple ad-hoc async patterns with inconsistent error handling
- **Observability Gaps**: No centralized monitoring or metrics for background tasks
- **Resource Inefficiency**: Uncoordinated task scheduling and resource competition
- **Operational Complexity**: Multiple systems to monitor, troubleshoot, and maintain
- **Technical Debt**: Duplicate async infrastructure implementations

The successful implementation of EnhancedBackgroundTaskManager in ANALYSIS_3101.md Phase 1-4 demonstrated significant benefits:
- 35% performance improvement through unified connection management
- Enhanced reliability through centralized error handling and retry patterns
- Improved observability through centralized task monitoring and metrics
- Simplified operations through single source of truth for background services

However, many background services still bypass this unified infrastructure, creating architectural inconsistency and missing out on these benefits.

## Decision

**ALL background services MUST use EnhancedBackgroundTaskManager as the single source of truth for async background operations.**

### Mandatory Requirements

1. **PROHIBITED**: Direct `asyncio.create_task()` calls for persistent background services
2. **MANDATORY**: Use `get_background_task_manager()` for all background operations
3. **REQUIRED**: Proper task categorization with TaskPriority levels
4. **ENFORCED**: Comprehensive tagging for service identification and monitoring

### Implementation Standard

```python
from prompt_improver.performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

# REQUIRED PATTERN
task_manager = get_background_task_manager()
await task_manager.submit_enhanced_task(
    task_id=f"service_name_{unique_identifier}",
    coroutine=background_function,
    priority=TaskPriority.HIGH,  # CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
    tags={
        "service": "service_name", 
        "type": "operation_type",
        "component": "component_name"
    }
)
```

### Legitimate Exceptions

Direct `asyncio.create_task()` remains ALLOWED for:
1. **Test Infrastructure**: Files in `/tests/` directories for parallel test execution
2. **Framework Internals**: EnhancedBackgroundTaskManager implementation itself
3. **Request Processing**: Short-lived operations within request/response handlers
4. **Parallel Computation**: Coordinated parallel execution within single functions (not persistent services)

## Rationale

### Technical Benefits
- **Centralized Monitoring**: Single point for task health, metrics, and observability
- **Enhanced Reliability**: Built-in retry mechanisms, circuit breakers, and dead letter queues
- **Resource Optimization**: Priority-based scheduling prevents resource contention
- **Consistent Error Handling**: Unified error handling and recovery patterns
- **Performance Gains**: Proven 35% improvement through coordinated resource management

### Operational Benefits
- **Simplified Troubleshooting**: Single system to monitor and debug
- **Unified Alerting**: Centralized task failure detection and escalation
- **Capacity Planning**: Clear visibility into system-wide async task load
- **Maintenance Reduction**: One system to update, scale, and optimize

### Architectural Benefits
- **Single Source of Truth**: Eliminates fragmented async infrastructure
- **Consistent Patterns**: Reduces cognitive load and decision fatigue
- **Future-Proof**: Central point for async infrastructure evolution
- **Integration Ready**: Single integration point for monitoring and observability tools

## Implementation Strategy

### Phase 1: Governance Enforcement (Immediate)
- **Pre-Tool Hooks**: Automatic validation of all code generation
- **Hook Validation**: Block creation of direct `asyncio.create_task()` violations
- **Guidance Integration**: Provide specific remediation guidance for violations

### Phase 2: Legacy Conversion (Progressive)
- **High Priority**: Critical monitoring and metrics components
- **Medium Priority**: CLI orchestration and testing infrastructure
- **Low Priority**: Utility services and optimization components
- **Real Behavior Testing**: Validate each conversion with actual operations

### Phase 3: Operational Integration (Ongoing)
- **Monitoring Integration**: Connect with existing OpenTelemetry metrics
- **Alerting Setup**: Configure proactive alerts for task failures and performance degradation
- **Capacity Planning**: Establish baseline metrics and scaling guidelines

## Governance and Enforcement

### Automated Enforcement
- **Claude Code Hooks**: Pre-tool validation blocks violations during code generation
- **Exit Codes**: Validation script returns non-zero for violations to block operations
- **Specific Guidance**: Clear error messages with exact remediation steps

### Manual Review Triggers
- **Background Service Context**: File paths, class names, method patterns indicating persistent services
- **Violation Detection**: Any `asyncio.create_task()` usage outside legitimate exceptions
- **Integration Verification**: Confirm proper EnhancedBackgroundTaskManager usage

### Compliance Monitoring
- **Code Analysis**: Regular scans for new violations or architectural drift
- **Performance Tracking**: Monitor task success rates, processing times, and resource usage
- **Operational Reviews**: Quarterly assessment of unified infrastructure effectiveness

## Consequences

### Positive Consequences
- **Architectural Consistency**: Single, well-understood async infrastructure pattern
- **Enhanced Reliability**: Proven error handling and recovery mechanisms
- **Improved Performance**: Demonstrated 35%+ performance gains
- **Operational Simplicity**: One system to monitor, maintain, and optimize
- **Developer Productivity**: Clear patterns reduce implementation complexity

### Negative Consequences
- **Initial Migration Effort**: Converting existing direct `asyncio.create_task()` usage
- **Learning Curve**: Teams must understand EnhancedBackgroundTaskManager API
- **Potential Over-Engineering**: Simple async operations may seem more complex
- **Dependency Centralization**: Single point of failure if background task manager fails

### Risk Mitigation
- **Gradual Migration**: Progressive conversion with real behavior testing
- **Comprehensive Documentation**: Clear usage patterns and troubleshooting guides
- **Robust Testing**: Extensive validation of centralized task manager reliability
- **Operational Runbooks**: Detailed procedures for monitoring and incident response

## Compliance

### Validation Requirements
- **Zero Violations**: No inappropriate direct `asyncio.create_task()` calls for background services
- **Complete Coverage**: All persistent background operations use unified infrastructure
- **Proper Integration**: Correct usage of TaskPriority, tagging, and error handling patterns
- **Performance Maintenance**: Preserve or improve existing performance characteristics

### Success Metrics
- **Architecture Consistency**: 100% background services using unified infrastructure
- **Performance Impact**: Maintain or exceed existing performance benchmarks
- **Operational Efficiency**: Reduced mean time to resolution for async-related issues
- **Developer Satisfaction**: Improved development experience through clear, consistent patterns

## References

- **ANALYSIS_3101.md**: Async infrastructure consolidation implementation and results
- **EnhancedBackgroundTaskManager**: `src/prompt_improver/performance/monitoring/health/background_manager.py`
- **Implementation Examples**: Files successfully converted in Phase 1-4 consolidation
- **Performance Data**: 35% improvement metrics from unified infrastructure adoption
- **Operational Procedures**: `docs/operations/background-task-management-runbook.md` (to be created)

---

**Decision Authority**: This ADR represents the official architectural decision for async infrastructure in the prompt-improver codebase. All new development and existing code modifications must comply with this protocol.

**Enforcement**: Automated through Claude Code pre-tool hooks and manual review processes. Violations will be blocked at code generation time with specific remediation guidance.

**Review Schedule**: Annual review of effectiveness, performance impact, and potential architectural evolution needs.