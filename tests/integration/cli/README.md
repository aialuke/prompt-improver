# CLI Integration Tests

This directory contains integration tests for CLI commands that interact with multiple components (services, database, external systems).

## Structure
- `test_[command].py` - Tests for specific CLI commands
- Focus on end-to-end CLI functionality
- Test CLI commands with real service interactions

## Guidelines
- Test CLI commands in realistic scenarios
- Include tests for error handling and edge cases
- Use fixtures for consistent test data
- Mock external dependencies when needed

## Run Tests
```bash
pytest tests/integration/cli/ -v
```
