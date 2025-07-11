# Integration Tests

## Overview

This directory contains integration tests that verify real component interactions with minimal mocking. These tests focus on ensuring that different parts of the system work together correctly, including database operations, service interactions, and end-to-end workflows.

## Test Organization

### Directory Structure
```
tests/integration/
├── cli/                    # CLI command integration tests
├── services/               # Service layer integration tests
├── rule_engine/           # Rule engine integration tests
├── test_async_validation.py   # Async infrastructure validation
├── test_implementation.py     # Phase 2 implementation tests
├── test_mcp_integration.py    # MCP server integration tests
├── test_performance.py        # Performance validation tests
├── test_service_integration.py # Service layer integration tests
└── README.md              # This documentation
```

### Test Characteristics

**Purpose**: Test real component interactions with minimal mocking  
**Scope**: Multi-component interactions, database operations, service workflows  
**Dependencies**: Real database connections, actual service instances  
**Speed**: < 5 seconds per test  
**Markers**: `@pytest.mark.integration`

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration test categories
pytest tests/integration/cli/ -v
pytest tests/integration/services/ -v
pytest tests/integration/rule_engine/ -v

# Run integration tests with coverage
pytest tests/integration/ --cov=src --cov-report=term-missing

# Run async integration tests
pytest tests/integration/ -k async -v

# Run performance validation tests
pytest tests/integration/ -m performance -v
```

## Test Categories

### CLI Integration Tests (`cli/`)
- **Purpose**: Command-line interface functionality with real services
- **Scope**: All CLI commands, argument parsing, output formatting
- **Dependencies**: Real database connections, minimal service mocking
- **Examples**: Train commands, discover-patterns, ml-status

### Service Integration Tests (`services/`)
- **Purpose**: Service layer business logic with real dependencies
- **Scope**: PromptImprovementService, AnalyticsService, MLModelService
- **Dependencies**: Real database sessions, actual ML model interactions
- **Examples**: Prompt improvement workflows, A/B testing, ML model caching

### Rule Engine Integration Tests (`rule_engine/`)
- **Purpose**: Rule application with real text processing and LLM interactions
- **Scope**: Rule combinations, rule effectiveness, LLM transformer integration
- **Dependencies**: Real rule logic, actual text transformations
- **Examples**: Clarity rule application, specificity improvements

## Test Writing Guidelines

### Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestServiceIntegration:
    async def test_database_integration(self, test_db_session, sample_data):
        # Use real database, minimal mocking
        service = PromptImprovementService()
        result = await service.store_improvement(sample_data, test_db_session)
        assert result["status"] == "success"
```

### Async Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_improve_prompt(self, test_db_session):
    # Test actual MCP server functionality
    result = await improve_prompt(
        prompt="Test prompt for validation",
        context={"domain": "testing"},
        session_id="integration_test"
    )
    
    assert "improved_prompt" in result
    assert result["processing_time_ms"] > 0
```

### Performance Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
async def test_response_time_requirement(self):
    start_time = asyncio.get_event_loop().time()
    result = await prompt_improvement_service.improve_prompt("test prompt")
    end_time = asyncio.get_event_loop().time()
    
    response_time = (end_time - start_time) * 1000
    assert response_time < 200  # <200ms requirement
```

## Best Practices

### Integration Test Guidelines
1. **Real Dependencies**: Use actual database sessions, real service instances
2. **Minimal Mocking**: Only mock external services (MLflow, external APIs)
3. **Realistic Data**: Use actual data patterns and workflows
4. **Transaction Isolation**: Use database transactions for test isolation
5. **Async Patterns**: Properly handle async operations with pytest-asyncio

### What We Mock vs. What We Don't Mock

**We DO Mock:**
- External ML tracking services (MLflow)
- External API calls
- File system operations for error simulation
- Time-dependent operations for deterministic testing

**We DON'T Mock:**
- Database operations (use real test database)
- Service layer business logic
- Text processing and rule application
- Internal async operations
- MCP server functionality

### Database Testing Strategy
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_real_database(self, test_db_session, sample_rule_metadata):
    # Populate database with test data
    for rule in sample_rule_metadata:
        test_db_session.add(rule)
    await test_db_session.commit()
    
    # Test real database operations
    service = PromptImprovementService()
    result = await service.get_active_rules(test_db_session)
    assert len(result) > 0
```

## Common Patterns

### Database Test Isolation
All integration tests use database transactions that are rolled back after each test to ensure isolation while still testing real database operations.

### Service Layer Testing
Integration tests focus on testing the complete service workflows rather than individual methods, ensuring that the full business logic works correctly.

### CLI Testing with Real Services
CLI tests use the actual CLI framework with real service instances, mocking only external dependencies like MLflow tracking.

## Navigation

- **Parent Directory**: [tests/](../README.md)
- **CLI Integration Tests**: [tests/integration/cli/](cli/README.md)
- **Service Integration Tests**: [tests/integration/services/](services/README.md)
- **Rule Engine Integration Tests**: [tests/integration/rule_engine/](rule_engine/README.md)
- **Unit Tests**: [tests/unit/](../unit/README.md)
- **Main Documentation**: [docs/](../../docs/README.md)
