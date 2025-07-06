# APES Test Suite Documentation

## Overview

The APES (Adaptive Prompt Enhancement System) test suite provides comprehensive testing coverage across all system components, from unit tests with maximum isolation to integration tests with real component interactions.

## Test Suite Organization

### Directory Structure
```
tests/
├── unit/                 # Pure unit tests with maximum mocking
├── integration/          # Tests with real component interactions  
├── cli/                  # Command-line interface tests
├── services/             # Service layer tests (mix of unit/integration)
├── rule_engine/          # Rule engine specific tests
├── conftest.py           # Centralized fixtures and configuration
├── test_async_validation.py  # Async infrastructure validation
└── README.md            # This documentation
```

### Test Categories

#### Unit Tests (`unit/`)
- **Purpose**: Fast, isolated tests with maximum mocking
- **Scope**: Individual functions, classes, and methods
- **Dependencies**: Minimal external dependencies, heavy mocking
- **Speed**: < 100ms per test
- **Markers**: `@pytest.mark.unit`

#### Integration Tests (`integration/`)
- **Purpose**: Test real component interactions with minimal mocking
- **Scope**: Service layer integration, database operations, MCP workflows
- **Dependencies**: Real database connections, actual service instances
- **Speed**: < 5 seconds per test
- **Markers**: `@pytest.mark.integration`

#### CLI Tests (`cli/`)
- **Purpose**: Command-line interface functionality
- **Scope**: All CLI commands, argument parsing, output formatting
- **Dependencies**: Mocked external services, real CLI framework
- **Speed**: < 1 second per test
- **Markers**: Inherit from component being tested

#### Service Tests (`services/`)
- **Purpose**: Service layer business logic
- **Scope**: PromptImprovementService, AnalyticsService, MLModelService
- **Dependencies**: Mix of mocked and real dependencies based on test type
- **Speed**: < 2 seconds per test
- **Markers**: `@pytest.mark.unit` or `@pytest.mark.integration`

#### Rule Engine Tests (`rule_engine/`)
- **Purpose**: Rule application and effectiveness
- **Scope**: Individual rules, rule combinations, rule metadata
- **Dependencies**: Minimal mocking, focus on rule logic
- **Speed**: < 500ms per test
- **Markers**: `@pytest.mark.unit`

## Running Tests

### Quick Start
```bash
# Run all tests with coverage
pytest

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/cli/ -v
```

### By Category
```bash
# Fast unit tests
pytest tests/unit/ -v

# Integration tests with real components
pytest tests/integration/ -v

# CLI functionality tests
pytest tests/cli/ -v

# Service layer tests
pytest tests/services/ -v

# Rule engine tests
pytest tests/rule_engine/ -v

# Skip slow tests (> 1 second)
pytest -m "not slow" -v

# Performance validation tests only
pytest -m performance -v

# Async tests only
pytest -m asyncio -v
```

### Coverage Analysis
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Terminal coverage report with missing lines
pytest --cov=src --cov-report=term-missing

# Coverage with branch analysis
pytest --cov=src --cov-branch --cov-report=term-missing
```

### Performance Testing
```bash
# Run performance-focused tests
pytest -m performance -v

# Test with timeout monitoring
pytest --timeout=300

# Profile test execution time
pytest --durations=10
```

### Async Test Validation
```bash
# All async tests
pytest tests/ -k async -v

# Strict async mode (explicit marking required)
pytest --asyncio-mode=strict

# Async validation tests specifically
pytest tests/test_async_validation.py -v
```

## Test Configuration

### Markers
- `asyncio`: Async tests using pytest-asyncio
- `integration`: Integration tests requiring real components
- `slow`: Tests taking >1 second (skipped in fast test runs)
- `performance`: Tests validating performance requirements
- `unit`: Pure unit tests with maximum isolation

### Coverage Requirements
- **Minimum Coverage**: 85%
- **Branch Coverage**: Enabled
- **Coverage Reports**: Terminal + HTML
- **Coverage Scope**: `src/` directory only

### Performance Thresholds
- **Unit Tests**: < 100ms each
- **Integration Tests**: < 5 seconds each
- **CLI Tests**: < 1 second each
- **Overall Test Suite**: < 5 minutes total
- **Performance Tests**: Validate <200ms response time requirement

## Best Practices

### Test Writing Guidelines

1. **Use Appropriate Test Types**
   - Unit tests for business logic and isolated functionality
   - Integration tests for component interactions and database operations
   - Performance tests for timing-critical requirements

2. **Fixture Usage**
   - Use centralized fixtures from `conftest.py`
   - Choose appropriate fixture scopes (function for isolation, session for expensive setup)
   - Avoid class-specific fixtures, prefer centralized ones

3. **Mocking Strategy**
   - Heavy mocking in unit tests for isolation
   - Minimal mocking in integration tests for realistic behavior
   - Mock external dependencies (APIs, ML services) but test real internal logic

4. **Async Testing**
   - Mark async tests with `@pytest.mark.asyncio`
   - Use proper event loop isolation (function scope)
   - Test concurrent operations where applicable

5. **Performance Considerations**
   - Mark slow tests with `@pytest.mark.slow`
   - Use `@pytest.mark.performance` for performance validation
   - Include realistic timing assertions

### Common Patterns

#### Unit Test Pattern
```python
@pytest.mark.unit
class TestServiceUnit:
    def test_business_logic(self, mock_db_session):
        # Heavy mocking, focus on logic
        service = ServiceClass()
        result = service.process_data(test_input)
        assert result == expected_output
```

#### Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestServiceIntegration:
    async def test_database_integration(self, test_db_session, sample_data):
        # Real database, minimal mocking
        service = ServiceClass()
        result = await service.store_data(sample_data, test_db_session)
        assert result["status"] == "success"
```

#### Performance Test Pattern
```python
@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformance:
    async def test_response_time(self):
        start_time = asyncio.get_event_loop().time()
        result = await fast_operation()
        end_time = asyncio.get_event_loop().time()
        
        response_time = (end_time - start_time) * 1000
        assert response_time < 200  # <200ms requirement
```

## Continuous Integration

### Pre-commit Validation
```bash
# Run before committing
pytest -m "not slow" --cov=src --cov-fail-under=85
```

### Full Validation Pipeline
```bash
# Complete test suite with coverage
pytest --cov=src --cov-branch --cov-report=html --cov-fail-under=85

# Performance validation
pytest -m performance -v

# Integration test validation
pytest -m integration -v
```

## Troubleshooting

### Common Issues

1. **Async Test Failures**
   - Ensure `@pytest.mark.asyncio` is present
   - Check event loop isolation (function scope)
   - Verify async/await usage is correct

2. **Fixture Not Found Errors**
   - Check if fixture is defined in `conftest.py`
   - Verify fixture scope is appropriate
   - Ensure test file can access the fixture

3. **Coverage Issues**
   - Check if new code is in `src/` directory
   - Verify imports are being executed during tests
   - Review exclusion patterns in configuration

4. **Performance Test Failures**
   - Check if system is under load during testing
   - Verify realistic performance expectations
   - Review mocking strategy for external dependencies

### Debug Commands
```bash
# Verbose output with full tracebacks
pytest -vvv --tb=long

# Run single test with debugging
pytest tests/path/to/test.py::TestClass::test_method -vvv

# Show test collection without running
pytest --collect-only

# List all available fixtures
pytest --fixtures tests/
```

## Maintenance

### Regular Tasks

1. **Coverage Review**
   - Monitor coverage trends over time
   - Identify uncovered critical paths
   - Add tests for new functionality

2. **Performance Monitoring**
   - Track test execution times
   - Identify performance regressions
   - Optimize slow tests when possible

3. **Test Quality Assessment**
   - Review test reliability and flakiness
   - Update mocking strategies as system evolves
   - Ensure integration tests cover critical user workflows

### Updating Test Infrastructure

When adding new test types or modifying the test infrastructure:

1. Update this documentation
2. Add appropriate markers to `pyproject.toml`
3. Create centralized fixtures in `conftest.py`
4. Update CI/CD pipelines accordingly
5. Communicate changes to the development team

## Contact

For questions about the test suite or contributions to test infrastructure, please refer to the main project documentation or create an issue in the project repository.