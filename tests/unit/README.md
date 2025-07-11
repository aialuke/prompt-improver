# Unit Tests

## Overview

This directory contains pure unit tests that focus on testing individual components, functions, and classes with real data and behavior. Unit tests emphasize testing the actual logic and text processing capabilities rather than mocking core functionality.

## Test Organization

### Files in this directory:
- `test_rule_engine_unit.py` - Unit tests for rule engine components with property-based testing

### Test Characteristics

**Purpose**: Fast, isolated tests with real data and behavior  
**Scope**: Individual functions, classes, and methods  
**Dependencies**: Real text processing logic, minimal external dependencies  
**Speed**: < 100ms per test  
**Markers**: `@pytest.mark.unit`

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_rule_engine_unit.py -v

# Run unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=term-missing

# Run only fast unit tests (skip slow tests)
pytest tests/unit/ -m "not slow" -v

# Run property-based tests with extended examples
pytest tests/unit/ --hypothesis-show-statistics
```

## Test Writing Guidelines

### Unit Test Pattern
```python
@pytest.mark.unit
class TestRuleEngineUnit:
    def test_clarity_rule_improvement(self):
        # Test with real rule logic and actual text processing
        rule = ClarityRule()
        result = rule.apply("fix this")
        
        assert result.improved_prompt != "fix this"
        assert result.confidence > 0
        assert result.success is True
```

### Property-Based Testing
```python
@given(st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=100))
def test_rule_always_succeeds(self, prompt):
    rule = ClarityRule()
    result = rule.apply(prompt)
    
    assert result.success is True
    assert isinstance(result.improved_prompt, str)
    assert 0 <= result.confidence <= 1.0
```

### Best Practices
1. **Real Behavior**: Test actual text processing and rule logic
2. **Fast Execution**: Each test should run in < 100ms
3. **Property-Based Testing**: Use Hypothesis for comprehensive input testing
4. **Convergence Testing**: Verify rule behavior doesn't grow unbounded
5. **Descriptive Names**: Test names should describe the behavior being tested

## Testing Approach

Unit tests in this project focus on:
- **Text Processing Logic**: Real rule application and text transformations
- **Confidence Scoring**: Actual confidence calculation algorithms
- **Input Validation**: Handling various text inputs and edge cases
- **Rule Convergence**: Ensuring rules don't expand text indefinitely
- **Property Verification**: Using Hypothesis for comprehensive testing

## What We DON'T Mock

- Text processing algorithms
- Rule application logic
- Confidence scoring mechanisms
- String transformations
- Core business logic

## Navigation

- **Parent Directory**: [tests/](../README.md)
- **Integration Tests**: [tests/integration/](../integration/README.md)
- **Regression Tests**: [tests/regression/](../regression/README.md)
- **Main Documentation**: [docs/](../../docs/README.md)
