# Rule Engine Integration Tests

This directory contains integration tests for rule engine components that interact with services, database, and external systems.

## Structure
- `test_[rule_type].py` - Tests for specific rule types
- Focus on rule engine interactions with other components
- Test rule execution in realistic scenarios
- Test rule persistence and loading

## Guidelines
- Test rule engine with real data flows
- Include tests for rule interactions with services
- Use fixtures for consistent rule configurations
- Test rule engine performance under load
- Test error handling and validation

## Run Tests
```bash
pytest tests/integration/rule_engine/ -v
```
