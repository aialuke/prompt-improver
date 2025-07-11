# Service Integration Tests

This directory contains integration tests for services that interact with multiple components (database, external APIs, other services).

## Structure
- `test_[service_name].py` - Tests for specific services
- Focus on service-to-service interactions
- Test services with real database connections
- Test external API integrations

## Guidelines
- Test services in realistic scenarios
- Include tests for inter-service communication
- Use fixtures for consistent test data
- Mock external APIs when appropriate
- Test error handling and retry mechanisms

## Run Tests
```bash
pytest tests/integration/services/ -v
```
