# Task Completion Protocol

## Code Quality Checks (Required)
1. **Lint & Format**: `ruff check src/ && ruff format src/`
2. **Type Check**: `pyright src/`
3. **Architecture Validation**: `import-linter --config pyproject.toml`
4. **Test Suite**: `pytest tests/ -v` (focus on integration tests)

## Performance Validation
- Run performance benchmarks if changes affect critical paths
- Validate <2ms cache response times maintained
- Check P95 <100ms endpoint response requirements

## Architecture Compliance
- Verify repository pattern usage (no direct database imports)
- Ensure protocol-based dependency injection
- Validate clean architecture layer separation
- Check service facade patterns (*Facade, *Service, *Manager)

## Testing Standards (2025)
- Use testcontainers for real service testing
- No mocks for external dependencies
- Integration tests preferred over unit tests
- Real behavior validation for all changes