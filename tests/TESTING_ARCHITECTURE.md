# Testing Architecture

This document describes the comprehensive testing architecture with proper test boundaries and categorization.

## Architecture Overview

The testing architecture is organized into four distinct categories, each with specific boundaries, performance requirements, and execution strategies:

```
tests/
├── unit/              # Pure unit tests with complete mocking (<100ms)
├── integration/       # Service boundary tests with test containers (<1s)
├── contract/          # API and protocol contract validation (<5s)
├── e2e/              # End-to-end workflow tests (<10s)
└── utils/            # Testing utilities and helpers
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in complete isolation with all dependencies mocked.

**Characteristics**:
- **Performance**: Each test must run in <100ms
- **Dependencies**: All external dependencies completely mocked
- **Isolation**: Pure functions and business logic only
- **Coverage**: 90%+ on critical business logic
- **Parallelization**: Highly parallel (8+ workers)

**Structure**:
```
tests/unit/
├── services/         # Service layer unit tests
├── repositories/     # Repository unit tests  
├── utils/           # Utility function unit tests
└── conftest.py      # Unit test configuration
```

**Example**:
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_improve_prompt_success(self, mock_service, sample_prompt):
    # All dependencies mocked, fast execution
    result = await mock_service.improve_prompt(sample_prompt)
    assert result["confidence"] > 0
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test service boundaries with real databases and internal services, mocking only external APIs.

**Characteristics**:
- **Performance**: Each test must run in <1 second
- **Dependencies**: Real databases (test containers), real internal services
- **Mocking**: External APIs only (OpenAI, MLflow, etc.)
- **Coverage**: 85%+ on service boundaries
- **Parallelization**: Moderate (4 workers)

**Structure**:
```
tests/integration/
├── api/             # API endpoint integration tests
├── services/        # Service integration tests
├── repositories/    # Repository with database tests
└── conftest.py      # Integration test configuration
```

**Example**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_improvement_with_database(self, prompt_service, test_db_session):
    # Real database, real service, mocked external APIs
    result = await prompt_service.improve_prompt("Fix bug", session_id)
    assert result["improved_prompt"] != "Fix bug"
```

### 3. Contract Tests (`tests/contract/`)

**Purpose**: Validate API contracts, schemas, and protocol compliance.

**Characteristics**:
- **Performance**: Each test must run in <5 seconds
- **Focus**: Schema validation, backward compatibility, protocol adherence
- **Coverage**: All API endpoints and MCP tools
- **Parallelization**: Limited (2 workers for coordination)

**Structure**:
```
tests/contract/
├── rest/            # REST API contract tests
├── mcp/             # MCP protocol contract tests
└── conftest.py      # Contract test configuration
```

**Example**:
```python
@pytest.mark.contract
@pytest.mark.api
def test_improve_prompt_response_schema(self, api_client, schema_validator):
    response = api_client.post("/api/v1/improve", data=valid_request)
    schema_validator(response.json(), "prompt_improvement_response")
```

### 4. End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows with full system deployment.

**Characteristics**:
- **Performance**: Each test must run in <10 seconds
- **Scope**: Complete user scenarios and workflows
- **Environment**: Full system with Docker Compose
- **Coverage**: Critical user journeys
- **Parallelization**: Sequential (1 worker)

**Structure**:
```
tests/e2e/
├── workflows/       # Complete workflow tests
├── scenarios/       # User scenario tests
└── conftest.py      # E2E test configuration
```

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.workflow
def test_new_user_onboarding_workflow(self, e2e_client, scenario_validator):
    # Complete user journey from registration to first improvement
    session_id = e2e_client.create_session("new_user")
    result = e2e_client.improve_prompt(session_id, "Help me debug")
    scenario_validator.validate_improvement_quality(result)
```

## Test Execution

### Running Tests by Category

```bash
# Unit tests (fast, parallel)
pytest tests/unit/ -m unit -n 8

# Integration tests (moderate speed, test containers)
pytest tests/integration/ -m integration -n 4

# Contract tests (API validation)
pytest tests/contract/ -m contract -n 2

# E2E tests (full workflows)
pytest tests/e2e/ -m e2e

# Performance tests (across all categories)
pytest -m performance

# Fast tests only (exclude slow tests)
pytest -m "not slow"
```

### CI/CD Pipeline

The testing pipeline runs tests in the following order:

1. **Unit Tests**: Fast feedback on business logic
2. **Integration Tests**: Validate service boundaries
3. **Contract Tests**: Ensure API compliance
4. **E2E Tests**: Validate complete workflows
5. **Performance Tests**: Validate performance requirements

Each category runs in parallel where possible, with different resource allocations and timeout settings.

## Performance Requirements

| Category | Individual Test | Test Suite | Parallelization |
|----------|----------------|------------|-----------------|
| Unit | <100ms | <30s | 8 workers |
| Integration | <1s | <2min | 4 workers |
| Contract | <5s | <5min | 2 workers |
| E2E | <10s | <15min | 1 worker |

## Coverage Targets

| Category | Target | Focus |
|----------|--------|-------|
| Unit | 90%+ | Business logic, pure functions |
| Integration | 85%+ | Service boundaries, data flow |
| Contract | 100% | All API endpoints, schemas |
| E2E | 100% | Critical user workflows |

## Best Practices

### Unit Tests
1. **Complete Isolation**: Mock all external dependencies
2. **Fast Execution**: Each test <100ms
3. **Pure Logic**: Focus on business logic and algorithms
4. **Deterministic**: Consistent results across runs
5. **Descriptive**: Clear test names describing behavior

### Integration Tests
1. **Real Services**: Use actual internal services
2. **Test Containers**: Isolated database environments
3. **Transaction Rollback**: Clean state after each test
4. **Mock External APIs**: Only external dependencies
5. **Service Boundaries**: Focus on component interactions

### Contract Tests
1. **Schema Validation**: Strict adherence to contracts
2. **Backward Compatibility**: Version compatibility checks
3. **Error Handling**: Proper error response formats
4. **Performance Contracts**: Response time requirements
5. **Protocol Compliance**: Full protocol specification coverage

### E2E Tests
1. **Complete Workflows**: End-to-end user journeys
2. **Real Environment**: Full system deployment
3. **Realistic Scenarios**: Actual user behavior patterns
4. **System State**: Proper setup and teardown
5. **Performance Monitoring**: Real-world performance validation

## Migration Guide

### From Existing Tests

1. **Identify Current Test Type**:
   - Does it mock all dependencies? → Unit test
   - Does it use real databases? → Integration test
   - Does it validate API responses? → Contract test
   - Does it test complete workflows? → E2E test

2. **Move to Appropriate Directory**:
   - Update import paths
   - Add appropriate test markers
   - Update fixtures and dependencies

3. **Update Test Configuration**:
   - Use category-specific conftest.py
   - Apply performance requirements
   - Update mock strategies

4. **Validate Test Boundaries**:
   - Ensure proper isolation for unit tests
   - Verify appropriate mocking for integration tests
   - Confirm schema validation for contract tests
   - Validate complete workflows for E2E tests

## Troubleshooting

### Common Issues

1. **Unit Test Too Slow**: Check for unmocked dependencies
2. **Integration Test Flaky**: Verify test container setup
3. **Contract Test Failing**: Validate schema definitions
4. **E2E Test Timeout**: Check system resource allocation

### Performance Optimization

1. **Unit Tests**: Reduce computation in test setup
2. **Integration Tests**: Optimize database queries and connections
3. **Contract Tests**: Reuse API client connections
4. **E2E Tests**: Optimize Docker container startup

## Validation Commands

```bash
# Validate test categorization
python -m pytest tests/ --collect-only --tb=no

# Run performance validation
python -m pytest tests/ -m performance --tb=short

# Test coverage validation  
python -m pytest tests/unit/ --cov=src/prompt_improver --cov-fail-under=90
python -m pytest tests/integration/ --cov=src/prompt_improver --cov-fail-under=85

# Test boundary validation
python -m pytest tests/unit/ --tb=short --timeout=5
python -m pytest tests/integration/ --tb=short --timeout=30
python -m pytest tests/contract/ --tb=short --timeout=60
python -m pytest tests/e2e/ --tb=short --timeout=300
```